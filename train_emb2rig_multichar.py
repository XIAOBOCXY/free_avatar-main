import cv2
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import torchvision.transforms as transforms
from utils.common import *
from models.DCGAN import Generator
from models.gan_loss import GANLoss

def Train(epoch, loader, model, model_D):
    # 获取当前学习率
    lr = optimizer.param_groups[0]['lr']
    # 打印当前epoch、学习率和时间戳
    print(f"*** Epoch {epoch}, lr:{lr:.5f}, timestamp:{timestamp}")
    loss_sum = 0.0
    loss_sum_D = 0.
    # 将判别器设置为训练模式
    model_D = model_D.train()
    # 设置每个epoch的训练步数
    train_step = min(args.train_step_per_epoch, len(loader))
    # 设置进度条
    b = '{l_bar}{bar:20}{r_bar}{bar:10b}'
    pbar = tqdm(enumerate(loader), bar_format=b, total=train_step)
    logger = []
    time0 = time.time()
    for i, data in pbar:
        if i>train_step:
            break
        # 梯度清零
        optimizer.zero_grad()
        # 将生成器设置为训练模式
        model.train()
        # 将判别器设置为评估模式
        model_D.eval()
        loss = dict()

        # 获取数据
        sources = data['img'].cuda().float()
        targets = data['target'].cuda().float()
        target_rigs = data['rigs'].cuda().float()
        is_render = data['is_render'].cuda().float()
        ldmk = data['ldmk'].cuda().float()
        role_id = data['role_id'].cuda().long()
        do_pixel = data['do_pixel'].cuda().int()
        bs_input = data['bs'].cuda().float() 
        has_rig = data['has_rig'].cuda().float() 
        
        
        # 获取真实和渲染的索引
        real_idx = ((is_render == 0).nonzero(as_tuple=True)[0])
        render_idx = ((is_render == 1).nonzero(as_tuple=True)[0])
        has_rig = ((has_rig == 1).nonzero(as_tuple=True)[0])
        do_pixel_idx = ((do_pixel == 1).nonzero(as_tuple=True)[0])
        
        # 获取角色索引
        role_idxes = [((role_id == CHARACTER_NAMES.index(name_e)).nonzero(as_tuple=True)[0]) for name_e in characters]
        
        # source image to emb
        with torch.no_grad():
            # 将source image resize后输入到model_emb中，得到emb_hidden_in和emb_in
            emb_hidden_in, emb_in = model_emb(resize(sources))
            # 如果args.weight_symm为True，则将source image resize后输入到model_symm中，得到emb_hidden_symm_in和emb_symm_in
            if args.weight_symm:
                emb_hidden_symm_in, emb_symm_in = model_symm(resize_symm(sources))
                # 将emb_hidden_in和emb_hidden_symm_in在dim=1维度上拼接
                emb_hidden_in = torch.cat((emb_hidden_in, emb_hidden_symm_in), dim=1)
            
            
        # 将emb_hidden_in和role_id输入到model中，得到output_rig
        output_rig = model(emb_hidden_in, role_id)
        
        with torch.no_grad():
            # 初始化output_imgs_c列表
            output_imgs_c = []
            # 遍历characters
            for c_i, cname in enumerate(characters):
                # 将output_rig中对应role_idxes[c_i]的元素reshape后输入到configs_characters[cname]["model_rig2img"]中，得到output_imgs_c
                output_imgs_c.append(configs_characters[cname]["model_rig2img"](output_rig[role_idxes[c_i]][:,:configs_characters[cname]['n_rig']].reshape(-1, configs_characters[cname]['n_rig'], 1, 1)))
            
            # 获取output_imgs_c的C, H, W
            C, H, W = output_imgs_c[0].shape[1:]
            # 获取sources的B
            B = sources.shape[0]
            # 初始化output_img
            output_img = torch.empty((B, C, H, W), dtype=torch.float32).cuda() 
            # 遍历characters
            for c_i, cname in enumerate(characters):
                # 将output_imgs_c中对应role_idxes[c_i]的元素赋值给output_img
                output_img[role_idxes[c_i]] = output_imgs_c[c_i]
            
        # 如果args.weight_rig为True，且do_pixel_idx不为空，则计算loss['rig']
        if args.weight_rig:
            if len(do_pixel_idx)>0:
                loss['rig'] = criterion_l2(output_rig[has_rig], target_rigs[has_rig])
        
        # 如果args.weight_emb为True，则计算loss['emb']
        if args.weight_emb:
            emb_hidden_out, emb_out = model_emb(resize(output_img))
            loss['emb']  = criterion_l2(emb_out, emb_in)

        # 如果args.weight_img为True，且do_pixel_idx不为空，则计算loss['image']
        if args.weight_img:
            if len(do_pixel_idx) > 0:
                loss['image'] = criterion_l1(output_img[do_pixel_idx], targets[do_pixel_idx]) * args.weight_img

        # 如果args.weight_D为True，则计算loss['G_D']
        if args.weight_D:
            output_D_G = model_D(output_img)
            loss['G_D'] = criterion_gan(output_D_G, True, False)[0] * args.weight_D

        # 如果args.weight_symm为True，则计算loss['symm']
        if args.weight_symm:
            with torch.no_grad():
                emb_hidden_symm_out, emb_symm_out = model_symm(resize_symm(output_img))
            loss['symm'] =  criterion_l2(emb_symm_out, emb_symm_in) * args.weight_symm
        # 如果loss为空，则跳过本次循环
        if not loss:
            continue

        
        # 计算loss_value
        loss_value = sum([v for k, v in loss.items()])
        
        # 将loss_value累加到loss_sum中
        loss_sum += loss_value.item()
        # 反向传播
        loss_value.backward(retain_graph=True)
        # 更新参数
        optimizer.step()
        # 更新学习率
        scheduler.step()
        
        # discriminator
        # 将optimizer_D梯度置零
        optimizer_D.zero_grad()
        # 将model_D设置为训练模式
        model_D.train()
        # 将model设置为评估模式
        model.eval()
        # 将output_img输入到model_D中，得到outputs_fake
        outputs_fake = model_D(output_img.detach())
        # 将targets[do_pixel_idx]输入到model_D中，得到outputs_real
        outputs_real = model_D(targets[do_pixel_idx])
        # 计算loss_fake
        loss_fake = criterion_gan(outputs_fake, False, True)[0]
        # 计算loss_real
        loss_real = criterion_gan(outputs_real, True, True)[0]
        # 计算loss_D_train
        loss_D_train = (loss_fake + loss_real) / 2.
        # 反向传播
        loss_D_train.backward()
        # 更新参数
        optimizer_D.step()
        # 更新学习率
        scheduler_D.step()

        # 将loss添加到writer中
        writer.add_scalars(f'train/loss_G', loss, epoch * train_step + i)
        # 将loss_value添加到writer中
        writer.add_scalar(f'train/loss_G_total', loss_value.item(), epoch * train_step + i)
        # 将loss_D_train添加到writer中
        writer.add_scalar(f'train/loss_D', loss_D_train.item(), epoch * train_step + i)
        
        # 将loss转换为字符串
        _loss_str = str({k: "{:.4f}".format(v/(i+1)) for k, v in loss.items()})
        # 将日志信息添加到logger中
        _log = f"Epoch {epoch}({timestamp}) (lr:{optimizer.param_groups[0]['lr']:05f}): [{i}/{len(train_dataloader)}] loss_G:{_loss_str}, loss_D: {loss_D_train.item():.04f}"
        logger.append(_log+'\n')
        # 更新进度条
        pbar.set_description(_log)
        
    writer.add_images(f'train/img', torch.cat([sources, targets, output_img], dim=-2)[::4], epoch * train_step + i)
    # 将训练过程中的图像添加到TensorBoard中
    avg_loss = loss_sum / train_step
    # 计算平均损失
    _log = "==> [Train] Epoch {} ({}), training loss={}".format(epoch, timestamp, avg_loss)
    print(_log)
    # 打印训练过程中的损失
    with open(os.path.join(log_save_path, f'emb2render_{timestamp}.log'), "a+") as log_file:
        log_file.writelines(logger)
    # 将训练过程中的日志写入文件
    if epoch % args.save_step == 0:
        torch.save({'state_dict': model.state_dict()}, model_path.replace('.pt', f'_{epoch}.pt'))
        torch.save({'state_dict': model_D.state_dict()}, model_path.replace('.pt', f'_{epoch}_D.pt'))
    # 每隔一定步数保存模型
    return avg_loss

def Eval(epoch, loader, model, model_D, best_score):
    # 初始化损失和模型为评估模式
    loss_sum = 0.0
    model = model.eval()
    model_D = model_D.eval()

    # 设置评估步长
    eval_step = min(args.eval_step_per_epoch, len(loader))
    b = '{l_bar}{bar:20}{r_bar}{bar:10b}'
    pbar = tqdm(enumerate(loader), bar_format=b, total=eval_step)
    logger = []
    time0 = time.time()
    for i, data in pbar:
        if i>eval_step:
            break
        loss = dict()

        # 获取数据
        sources = data['img'].cuda().float()
        targets = data['target'].cuda().float()
        target_rigs = data['rigs'].cuda().float()
        is_render = data['is_render'].cuda().float()
        ldmk = data['ldmk'].cuda().float()
        role_id = data['role_id'].cuda().long()
        do_pixel = data['do_pixel'].cuda().int()
        bs_input = data['bs'].cuda().float() 
        
        
        # 获取索引
        real_idx = ((is_render == 0).nonzero(as_tuple=True)[0])
        render_idx = ((is_render == 1).nonzero(as_tuple=True)[0])
        do_pixel_idx = ((do_pixel == 1).nonzero(as_tuple=True)[0])
        role_idx0 = ((role_id == 0).nonzero(as_tuple=True)[0])
        role_idx1 = ((role_id == 1).nonzero(as_tuple=True)[0])
        role_idxes = [((role_id == CHARACTER_NAMES.index(name_e)).nonzero(as_tuple=True)[0]) for name_e in characters]
        # source image to emb
        with torch.no_grad():
            
            # 获取嵌入
            emb_hidden_in, emb_in = model_emb(resize(sources))
            if args.weight_symm:
                emb_hidden_symm_in, emb_symm_in = model_symm(resize_symm(sources))
                emb_hidden_in = torch.cat((emb_hidden_in, emb_hidden_symm_in), dim=1)
            # 获取角色
            output_rig = model(emb_hidden_in, role_id)
            output_imgs_c = []
            for c_i, cname in enumerate(characters):
                output_imgs_c.append(configs_characters[cname]["model_rig2img"](output_rig[role_idxes[c_i]][:,:configs_characters[cname]['n_rig']].reshape(-1, configs_characters[cname]['n_rig'], 1, 1)))
            
            # 获取输出图像
            C, H, W = output_imgs_c[0].shape[1:]
            B = sources.shape[0]
            output_img = torch.empty((B, C, H, W), dtype=torch.float32).cuda()  # 假设输出的类型为float32
            for c_i, cname in enumerate(characters):
                output_img[role_idxes[c_i]] = output_imgs_c[c_i]
            
        # 计算损失
        if args.weight_rig:
            if len(do_pixel_idx)>0:
                loss['rig'] = criterion_l2(output_rig[do_pixel_idx], target_rigs[do_pixel_idx])
        
        if args.weight_emb:
            emb_hidden_out, emb_out = model_emb(resize(output_img))
            loss['emb']  = criterion_l2(emb_out, emb_in)

        if args.weight_img:
            if len(do_pixel_idx) > 0:
                loss['image'] = criterion_l1(output_img[do_pixel_idx], targets[do_pixel_idx]) * args.weight_img
        
        if args.weight_D:
            output_D_G = model_D(output_img)
            loss['G_D'] = criterion_gan(output_D_G, True, False)[0] * args.weight_D
            
        if args.weight_symm:
            with torch.no_grad():
                emb_hidden_symm_out, emb_symm_out = model_symm(resize_symm(output_img))
            loss['symm'] =  criterion_l2(emb_symm_out, emb_symm_in) * args.weight_symm
            
        # 如果没有损失，跳过
        if not loss:
            continue
        
        # 计算总损失
        loss_value = sum([v for k, v in loss.items()])
        
        # 累加损失
        loss_sum += loss_value.item()

        # 添加损失到tensorboard
        writer.add_scalars(f'eval/loss_G', loss, epoch * eval_step + i)
        writer.add_scalar(f'eval/loss_G_total', loss_value.item(), epoch * eval_step + i)
        
        # 记录日志
        _loss_str = str({k: "{:.4f}".format(v/(i+1)) for k, v in loss.items()})
        _log = f"Epoch {epoch}({timestamp}) (lr:{optimizer.param_groups[0]['lr']:05f}): [{i}/{len(train_dataloader)}] loss_G:{_loss_str}"
        logger.append(_log+'\n')
        pbar.set_description(_log)
        
    # 添加图像到tensorboard
    writer.add_images(f'eval/img', torch.cat([sources, targets, output_img], dim=-2)[::4], epoch * eval_step + i)
    # 计算平均损失
    avg_loss = loss_sum / eval_step
    _log = "==> [Eval] Epoch {} ({}), evaluation loss={}".format(epoch, timestamp, avg_loss)

    
    # 如果平均损失小于最佳分数，保存模型
    if avg_loss < best_score:
        patience_cur = args.patience
        best_score = avg_loss        
        torch.save({'state_dict': model.state_dict()}, model_path)
        torch.save({'state_dict': model_D.state_dict()}, model_path.replace('.pt', f'_D.pt'))
        _log += '\n Found new best model!\n'
    else:
        patience_cur -= 1
    print(_log)
    logger.append(_log)
    # 保存日志
    with open(os.path.join(log_save_path, f'emb2render_{timestamp}.log'), "a+") as log_file:
        log_file.writelines(logger)  
    return avg_loss

def Test(signature, model, model_emb, model_rig2img, resize):
    # 将模型设置为评估模式
    model = model.eval()
    # 定义保存路径
    save_root ='/project/qiuf/expr-capture/test'
    # 定义数据路径
    root = '/data/Workspace/Rig2Face/data'
    # 定义文件夹列表
    folders = ['ziva']
    
    # 遍历文件夹列表
    for fold in folders:
        # 定义保存文件夹路径
        save_fold = os.path.join(save_root, f'{signature}_{fold}')
        # 创建保存文件夹
        os.makedirs(save_fold, exist_ok=True)
        # 获取文件夹中的图片列表
        imgnames = os.listdir(os.path.join(root, fold))
        # 对图片列表进行排序
        imgnames.sort()
        # 保留图片列表
        imgnames = imgnames[:]

        # 定义rigs字典
        rigs = {}
        # 遍历characters列表
        for charname in characters:
            # 将characters列表中的元素作为键，初始化为空列表
            rigs[charname] = []
        # 遍历图片列表
        for i, img in tqdm(enumerate(imgnames), total=len(imgnames)):
            # 读取图片
            img = cv2.resize(cv2.imread(os.path.join(root, fold, img)), (256,256))
            # 将图片转换为RGB格式
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 将图片转换为张量
            img_tensor = torch.FloatTensor(img).permute(2,0,1).unsqueeze(0).cuda()/255.
            
            # 不计算梯度
            with torch.no_grad():
                # 获取图片的embedding
                emb_hidden, emb = model_emb(resize(img_tensor))
                # 如果需要计算对称性权重
                if args.weight_symm:
                    # 获取图片的对称性embedding
                    emb_hidden_symm_in, emb_symm_in = model_symm(resize_symm(img_tensor))
                    # 将对称性embedding与原始embedding拼接
                    emb_hidden = torch.cat((emb_hidden, emb_hidden_symm_in), dim=1)
                
                # 定义img_outs列表
                img_outs=  []
                # 定义emb_dists列表
                emb_dists = []
                # 遍历characters列表
                for c_i, cname in enumerate(characters):
                    # 获取rig
                    rig = model(emb_hidden, torch.LongTensor([CHARACTER_NAMES.index(cname),]).cuda())
                    # 将rig添加到rigs字典中
                    rigs[cname].append(rig.cpu().numpy())
                    # 将rig转换为图片
                    img_outs.append(configs_characters[cname]["model_rig2img"](rig[:, :configs_characters[cname]['n_rig']].reshape(1,-1,1,1)))
                    # 获取图片的embedding
                    emb_hidden_out0, emb_out0 = model_emb(resize(img_outs[-1]))
                    # 计算embedding的距离
                    emb_dists.append(torch.dist(emb, emb_out0))
                

                # 将图片和生成的图片拼接
                img_vis = torch.cat((img_tensor, *img_outs), dim=-1).squeeze()*255. 
                # 将图片转换为numpy数组
                img_vis = img_vis.cpu().numpy().transpose(1,2,0).astype(np.uint8)
                # 将图片转换为BGR格式
                img_vis = np.ascontiguousarray(img_vis[...,::-1], dtype=np.uint8)
                # 遍历characters列表
                for c_i, cname in enumerate(characters):
                    # 在图片上添加文本
                    cv2.putText(img_vis, str(np.round(emb_dists[c_i].item(), 6)), (256*(c_i+1), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                # 保存图片
                cv2.imwrite(os.path.join(save_fold, f'{i:05d}.jpg'), img_vis)
                
        # 遍历characters列表
        for c_i, cname in enumerate(characters):
            # 保存rigs字典中的数据
            np.savetxt(os.path.join(save_root, f'ziva_{cname}.txt'), np.array(rigs[cname]).squeeze())

        # 将图片转换为视频
        imgs2video(save_fold)


if __name__ == '__main__':
    import time
    from choose_character import character_choice
    from models.load_emb_model import load_emb_model
    from models.CascadeNet import get_model
    from models.discriminator import MultiscaleDiscriminator, get_parser
    from dataset.ABAWData import ABAWDataset2_multichar
    from torch.utils.tensorboard import SummaryWriter
    args = parse_args_from_yaml('configs_emb2rig_multi.yaml')
    setup_seed(args.seed)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    os.system("git add .")
    os.system("git commit -m" + timestamp)
    os.system("git push")
    
    # emb_model
    characters = args.character.replace(' ','').split(',')
    CHARACTER_NAMES = args.CHARACTER_NAME
    configs_characters = {e:character_choice(e) for e in characters}
    n_rig = max([e['n_rig'] for e in configs_characters.values()])
    for character in configs_characters:
        configs_characters[character]['mouth_crop'] = torch.tensor(configs_characters[character]['mouth_crop']).cuda().float()
        params = {'nz': configs_characters[character]['n_rig'], 'ngf': 64*2, 'nc': 3}
        model_rig2img = Generator(params)
        model_rig2img = model_rig2img.eval().cuda()
        ckpt_generator = torch.load(configs_characters[character]['ckpt_rig2img'])
        model_rig2img.load_state_dict(ckpt_generator['state_dict'])
        configs_characters[character]['model_rig2img'] = model_rig2img
        print('load generator model from {}'.format(configs_characters[character]['ckpt_rig2img']))
        
    model_emb, emb_dim, resize = load_emb_model(args.emb_backbone)
    model_emb = model_emb.eval().cuda()
    model_emb_params = count_parameters(model_emb)
    print('emb model:', model_emb_params)
    
    # dissymm model 
    if args.weight_symm:
        model_symm, emb_dim2, resize_symm = load_emb_model('dissymm_repvit')
        model_symm.cuda().eval()
        emb_dim += emb_dim2

    # img2rig model
    pass

    # emb2rig model
    model_path = os.path.join(args.save_root, 'ckpt', "emb2rig_multi_{}.pt".format(timestamp))
    model = get_model(1, refine_3d=False,
                                 norm_twoD=False,
                                 num_blocks=2, #5,
                                 input_size=emb_dim,
                                 output_size=n_rig,
                                 linear_size=512, #1024,
                                 dropout=0.1,
                                 leaky=False,
                                 use_multichar=args.use_multichar,
                                 id_embedding_dim=args.id_embedding_dim
                                 )
    model = model.cuda()
    model_params = count_parameters(model)
    print('emb2rig model:', model_params)
    
    # D_model
    opt = get_parser()
    model_D = MultiscaleDiscriminator(opt).cuda()

    # 如果有预训练模型，则加载预训练模型
    if args.pretrained:
        # 获取预训练模型的路径
        ckpt_pretrained = os.path.join(args.save_root, 'ckpt', "emb2rig_multi_{}.pt".format(args.pretrained))
        ckpt_pretrained_D = os.path.join(args.save_root, 'ckpt', "emb2rig_multi_{}_D.pt".format(args.pretrained))
        # 加载预训练模型
        checkpoint = torch.load(ckpt_pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        checkpoint_D = torch.load(ckpt_pretrained_D)
        model_D.load_state_dict(checkpoint_D['state_dict'])

        # 打印加载预训练模型的路径
        print("load pretrained model {}".format(ckpt_pretrained))
    # 如果没有预训练模型，则初始化模型
    else:
        # 初始化模型
        model.apply(init_weights)
        model_D.apply(init_weights)
        # 打印模型初始化
        print("Model initialized")         
    
    # transforms
    # 定义数据增强变换
    transform1 = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 随机调整亮度、对比度、饱和度和色相
        transforms.Resize([256,256]),  # 调整图像大小为256x256
        transforms.ToTensor(),  # 将图像转换为张量
    ])
    
    transform2 = transforms.Compose([
        transforms.Resize([256, 256]),  # 调整图像大小为256x256
        transforms.ToTensor(),  # 将图像转换为张量
    ])
    
    # optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.0, 0.99))
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, 2, 1e-6)
    optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, model_D.parameters()), lr=args.lr_D, betas=(0.0, 0.99), weight_decay=1e-6)
    scheduler_D = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_D, 500, 2, 1e-7)
    
    # loss function 
    criterion_l1 = nn.L1Loss()
    criterion_l2 = nn.MSELoss()
    criterion_BCE = nn.BCELoss()
    criterion_gan = GANLoss('hinge')
    
    # Test
    if args.mode == 'test':
        Test(args.pretrained, model, model_emb, model_rig2img, resize)
        exit()
    
    # datasets
    train_dataset = ABAWDataset2_multichar(configs_characters, data_split='train',CHARACTER_NAME=CHARACTER_NAMES, transform=transform1, return_rigs=True)
    test_dataset = ABAWDataset2_multichar(configs_characters, data_split='test',CHARACTER_NAME=CHARACTER_NAMES, transform=transform2, return_rigs=True)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,num_workers=8)
    val_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True, num_workers=8)
    
    # save files
    ck_save_path = f'{args.save_root}/ckpt'
    pred_save_path = f'{args.save_root}/test'
    log_save_path = f'{args.save_root}/logs'
    tensorboard_path = f'{args.save_root}/tensorboard/{timestamp}'
    
    os.makedirs(ck_save_path,exist_ok=True)
    os.makedirs(pred_save_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)

    writer = SummaryWriter(log_dir=tensorboard_path)
    
    patience_cur = args.patience
    best_score = float('inf')


    for epoch in range(500000000):
        avg_loss = Train(epoch, train_dataloader, model, model_D)
        avg_loss_eval = Eval(epoch, val_dataloader, model, model_D, best_score)
        if epoch % args.save_step == 0:
            Test(timestamp+'_'+str(epoch), model, model_emb, model_rig2img, resize)

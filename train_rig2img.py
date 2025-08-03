from tqdm import tqdm

def Train(epoch, loader, model):
    # 获取当前学习率
    lr = optimizer.param_groups[0]['lr']
    # 打印当前epoch、学习率和时间戳
    print(f"*** Epoch {epoch}, lr:{lr:.5f}, timestamp:{timestamp}")
    # 初始化损失和模型为训练模式
    loss_sum = 0.0
    model.train()    
    # 设置每个epoch的训练步数
    train_step = min(args.train_step_per_epoch, len(loader))
    # 设置进度条格式
    b = '{l_bar}{bar:20}{r_bar}{bar:10b}'
    # 创建进度条
    pbar = tqdm(enumerate(loader), bar_format=b, total=train_step)
    # 初始化日志
    logger = []
    # 记录开始时间
    time0 = time.time()
    # 遍历训练数据
    for i, data in pbar:
        # 如果超过训练步数，则跳出循环
        if i>train_step:
            break
        # 梯度清零
        optimizer.zero_grad()
        # 初始化损失字典
        loss = dict()

        # 将数据转换为cuda张量
        targets = data['img'].cuda().float()
        rigs = data['rigs'].cuda().float()
        # 确保数据中有rig
        assert (data['has_rig'] == 1).all()
        # 将rigs reshape为(-1, configs_character['n_rig'], 1, 1)
        outputs = model(rigs.reshape(-1, configs_character['n_rig'], 1, 1))
        # 计算图像损失
        loss['image'] = criterion_l1(outputs, targets) * args.weight_img
        # 计算嘴巴损失
        loss['mouth'] = criterion_l1(outputs*mouth_crop, targets*mouth_crop) * args.weight_mouth

        # 计算总损失
        loss_value = sum([v for k, v in loss.items()])
        
        # 累加损失
        loss_sum += loss_value.item()
        # 反向传播
        loss_value.backward(retain_graph=True)
        # 更新参数
        optimizer.step()
        # 更新学习率
        scheduler.step()
        

        # 将损失写入tensorboard
        writer.add_scalars(f'train/loss', loss, epoch * train_step + i)
        # 将总损失写入tensorboard
        writer.add_scalar(f'train/loss_total', loss_value.item(), epoch * train_step + i)
        
        # 计算平均损失
        _loss_str = str({k: "{:.4f}".format(v/(i+1)) for k, v in loss.items()})
        # 记录日志
        _log = f"Epoch {epoch}({timestamp}) (lr:{optimizer.param_groups[0]['lr']:05f}): [{i}/{len(train_dataloader)}] loss_G:{_loss_str}"
        logger.append(_log+'\n')
        # 更新进度条描述
        pbar.set_description(_log)
        
    writer.add_images(f'train/img', torch.cat([outputs, targets], dim=-2)[::4], epoch * train_step + i)
    # 将outputs和targets按dim=-2维度拼接，然后每隔4个取一个，将结果添加到writer中
    avg_loss = loss_sum / train_step
    # 计算平均损失
    _log = "==> [Train] Epoch {} ({}), training loss={}".format(epoch, timestamp, avg_loss)
    # 格式化输出日志
    print(_log)
    # 打印日志
    with open(os.path.join(log_save_path, f'{task}_{timestamp}.log'), "a+") as log_file:
        log_file.writelines(logger)
    # 将日志写入文件
    if epoch % args.save_step == 0:
        torch.save({'state_dict': model.state_dict()}, model_path.replace('.pt', f'_{epoch}.pt'))
    # 每隔args.save_step个epoch保存一次模型
    return avg_loss

def Eval(epoch, loader, model, best_score):
    # 初始化损失和模型为评估模式
    loss_sum = 0.0
    model.eval()    
    # 设置评估步长
    eval_step = min(args.eval_step_per_epoch, len(loader))
    # 设置进度条格式
    b = '{l_bar}{bar:20}{r_bar}{bar:10b}'
    # 创建进度条
    pbar = tqdm(enumerate(loader), bar_format=b, total=eval_step)
    logger = []
    time0 = time.time()
    # 遍历数据集
    for i, data in pbar:
        if i>eval_step:
            break
        loss = dict()

        # 获取数据和标签
        targets = data['img'].cuda().float()
        rigs = data['rigs'].cuda().float()
        assert (data['has_rig'] == 1).all()
        # 不计算梯度
        with torch.no_grad():
            # 获取模型输出
            outputs = model(rigs.reshape(-1, configs_character['n_rig'], 1, 1))
            # 计算损失
            loss['image'] = criterion_l1(outputs, targets) * args.weight_img
            loss['mouth'] = criterion_l1(outputs*mouth_crop, targets*mouth_crop) * args.weight_mouth

        loss_value = sum([v for k, v in loss.items()])
        
        loss_sum += loss_value.item()

        # 将损失写入tensorboard
        writer.add_scalars(f'train/loss', loss, epoch * eval_step + i)
        writer.add_scalar(f'train/loss_total', loss_value.item(), epoch * eval_step + i)
        
        # 计算平均损失
        _loss_str = str({k: "{:.4f}".format(v/(i+1)) for k, v in loss.items()})
        # 记录日志
        _log = f"Epoch {epoch}({timestamp}) (lr:{optimizer.param_groups[0]['lr']:05f}): [{i}/{len(train_dataloader)}] loss_G:{_loss_str}"
        logger.append(_log+'\n')
        pbar.set_description(_log)
        
    # 将图像写入tensorboard
    writer.add_images(f'train/img', torch.cat([outputs, targets], dim=-2)[::4], epoch * eval_step + i)
    avg_loss = loss_sum / eval_step
    _log = "==> [Eval] Epoch {} ({}), training loss={}".format(epoch, timestamp, avg_loss)
    
    # 如果平均损失小于最佳得分，则更新最佳得分和模型
    if avg_loss < best_score:
        patience_cur = args.patience
        best_score = avg_loss        
        torch.save({'state_dict': model.state_dict()}, model_path)
        _log += '\n Found new best model!\n'
    else:
        patience_cur -= 1
        
    print(_log)
    # 将日志写入文件
    with open(os.path.join(log_save_path, f'{task}_{timestamp}.log'), "a+") as log_file:
        log_file.writelines(logger)
    return avg_loss

if __name__ == '__main__':
    import time
    import os
    import torch
    from choose_character import character_choice
    from utils.common import parse_args_from_yaml, setup_seed, init_weights
    from models.DCGAN import Generator
    import torchvision.transforms as transforms
    import torch.nn as nn
    from dataset.ABAWData import ABAWDataset2
    from torch.utils.data import DataLoader
    from torch.optim import lr_scheduler
    from torch.utils.tensorboard import SummaryWriter
    task = 'rig2img'
    # 从yaml文件中解析参数
    args = parse_args_from_yaml(f'configs_{task}.yaml')
    # 设置随机种子
    setup_seed(args.seed)
    # 获取当前时间戳
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    # 提交git
    os.system("git add .")
    os.system("git commit -m" + timestamp)
    os.system("git push")
    
    # 选择角色
    configs_character = character_choice(args.character)
    # 获取角色嘴巴的裁剪区域
    mouth_crop = torch.tensor(configs_character['mouth_crop']).cuda().float()

    # 获取模型保存路径
    model_path = os.path.join(args.save_root,'ckpt', f"{task}_{timestamp}.pt")
    # 定义模型参数
    params = {'nz': configs_character['n_rig'], 'ngf': 64*2, 'nc': 3}
    # 初始化模型
    model = Generator(params)
    model = model.cuda()
    
    # 如果有预训练模型，则加载预训练模型
    if args.pretrained:
        ckpt_pretrained = os.path.join(args.save_root, 'ckpt', f"{task}_{args.pretrained}.pt")
        checkpoint = torch.load(ckpt_pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        print("load pretrained model {}".format(ckpt_pretrained))
    else:
        # 初始化模型权重
        model.apply(init_weights)
        print("Model initialized")      
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor()])
    
    # 定义损失函数
    criterion_l1 = nn.L1Loss()
    # 定义优化器
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.0, 0.99))
    # 定义学习率调度器
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, 2, 1e-6)
 
    # 加载训练数据集
    train_dataset = ABAWDataset2(root_path=configs_character['data_path'],character=args.character, only_render=True,
                                 data_split='train', transform=transform, return_rigs=True, n_rigs=configs_character['n_rig'])
    # 加载测试数据集
    test_dataset = ABAWDataset2(root_path=configs_character['data_path'],character=args.character,only_render=True,
                                data_split='test', transform=transform, return_rigs=True, n_rigs=configs_character['n_rig'])
    # 加载训练数据集的dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,num_workers=12)
    # 加载测试数据集的dataloader
    val_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True, num_workers=12)

    # 定义保存路径
    ck_save_path = f'{args.save_root}/ckpt'
    pred_save_path = f'{args.save_root}/test'
    log_save_path = f'{args.save_root}/logs'
    tensorboard_path = f'{args.save_root}/tensorboard/{timestamp}'
    
    # 创建保存路径
    os.makedirs(ck_save_path,exist_ok=True)
    os.makedirs(pred_save_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)

    # 初始化tensorboard
    writer = SummaryWriter(log_dir=tensorboard_path)
    
    # 初始化耐心和最佳得分
    patience_cur = args.patience
    best_score = float('inf')


    # 训练模型
    for epoch in range(500000000):
        avg_loss = Train(epoch, train_dataloader, model)
        avg_loss_eval = Eval(epoch, val_dataloader, model, best_score)

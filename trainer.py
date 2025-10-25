import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import OrderedDict  # 导入 OrderedDict

from utils import DiceLoss


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val",
                             transform=transforms.Compose(
                                 [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')

    # --- 初始化 iter_num, start_epoch 和 best_loss ---
    iter_num = 0
    start_epoch = 0
    best_loss = 10e10

    # --- 添加继续训练 (resume) 逻辑 ---
    if args.resume and os.path.exists(args.resume):
        try:
            logging.info(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume,
                                    map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # 处理 DataParallel 包装
            state_dict = checkpoint['model_state_dict']
            if args.n_gpu > 1:
                # 如果模型是 DataParallel，确保 checkpoint 键有 'module.' 前缀
                is_parallel = any(k.startswith('module.') for k in state_dict.keys())
                if not is_parallel:
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        new_state_dict['module.' + k] = v
                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(state_dict)
            else:
                # 如果模型不是 DataParallel，确保 checkpoint 键没有 'module.' 前缀
                is_parallel = any(k.startswith('module.') for k in state_dict.keys())
                if is_parallel:
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        new_state_dict[k.replace('module.', '')] = v
                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(state_dict)

            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'iter_num' in checkpoint:
                iter_num = checkpoint['iter_num']
            if 'best_loss' in checkpoint:
                best_loss = checkpoint['best_loss']

            logging.info(f"Resuming training from epoch {start_epoch}, iter_num {iter_num}, best_loss {best_loss}")

        except Exception as e:
            logging.warning(f"Could not load checkpoint from {args.resume}: {e}")
            logging.info("Starting training from scratch.")
    elif args.resume:
        logging.warning(f"Resume path {args.resume} not found. Starting from scratch.")
    else:
        logging.info("No resume path provided. Starting from scratch.")
    # --- 结束 resume 逻辑 ---

    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))

    # --- 修改 epoch 范围以支持 resume ---
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    # best_loss = 10e10 # 已在前面定义

    for epoch_num in iterator:
        model.train()
        batch_dice_loss = 0
        batch_ce_loss = 0
        for i_batch, sampled_batch in tqdm(enumerate(train_loader), desc=f"Train: {epoch_num}", total=len(train_loader),
                                           leave=False):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            batch_dice_loss += loss_dice.item()
            batch_ce_loss += loss_ce.item()
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        batch_ce_loss /= len(train_loader)
        batch_dice_loss /= len(train_loader)
        batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
        logging.info('Train epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
            epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))

        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            batch_dice_loss = 0
            batch_ce_loss = 0
            with torch.no_grad():
                for i_batch, sampled_batch in tqdm(enumerate(val_loader), desc=f"Val: {epoch_num}",
                                                   total=len(val_loader), leave=False):
                    image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                    outputs = model(image_batch)
                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    batch_dice_loss += loss_dice.item()
                    batch_ce_loss += loss_ce.item()

                batch_ce_loss /= len(val_loader)
                batch_dice_loss /= len(val_loader)
                batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
                logging.info('Val epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
                    epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))

                # --- 修改保存逻辑 ---

                # 准备要保存的状态
                model_state_to_save = model.module.state_dict() if args.n_gpu > 1 else model.state_dict()
                checkpoint_data = {
                    'epoch': epoch_num,
                    'iter_num': iter_num,
                    'best_loss': best_loss,  # 保存当前的 best_loss
                    'model_state_dict': model_state_to_save,
                    'optimizer_state_dict': optimizer.state_dict(),
                }

                # 保存 best_model
                if batch_loss < best_loss:
                    best_loss = batch_loss
                    checkpoint_data['best_loss'] = best_loss  # 更新 dict 中的 best_loss
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    try:
                        torch.save(checkpoint_data, save_mode_path)
                        logging.info("save best model to {}".format(save_mode_path))
                    except Exception as e:
                        logging.warning(f"Failed to save best checkpoint: {e}")

                # 总是保存 last_model
                save_mode_path_last = os.path.join(snapshot_path, 'last_model.pth')
                try:
                    # 保存最后的状态（可能也是最好的状态）
                    torch.save(checkpoint_data, save_mode_path_last)
                    logging.info("save last model to {}".format(save_mode_path_last))
                except Exception as e:
                    logging.warning(f"Failed to save last checkpoint: {e}")

                # --- 原始逻辑 (已替换) ---
                # if batch_loss < best_loss:
                #     save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                #     torch.save(model.state_dict(), save_mode_path)
                #     best_loss = batch_loss
                # else:
                #     save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
                #     torch.save(model.state_dict(), save_mode_path)
                # logging.info("save model to {}".format(save_mode_path))
                # --- 结束修改 ---

    writer.close()
    return "Training Finished!"

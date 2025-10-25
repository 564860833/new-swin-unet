from os.path import split
import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict  # 导入 OrderedDict

from config import get_config
from datasets.dataset_synapse import Synapse_dataset
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import test_single_volume

# ... (parser 定义部分保持不变) ...
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/test_vol_h5',
                    help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='datasets', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

parser.add_argument("--n_class", default=4, type=int)
parser.add_argument("--split_name", default="test", help="Directory of the input list")

args = parser.parse_args()

if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
config = get_config(args)


def inference(args, model, test_save_path=None):
    db_test = Synapse_dataset(base_dir=args.volume_path, split=args.split_name, list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        if args.dataset == "datasets":
            case_name = split(case_name.split(",")[0])[-1]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                      patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
            i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        args.dataset: {
            'root_path': args.root_path,
            'list_dir': f'./lists/{args.dataset}',
            'num_classes': args.n_class,
            "z_spacing": 1
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['root_path']
    # args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    # --- 修改加载逻辑 ---
    snapshot = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(snapshot):
        snapshot = snapshot.replace('best_model', 'epoch_' + str(args.max_epochs - 1))
    if not os.path.exists(snapshot):
        snapshot = os.path.join(args.output_dir, 'last_model.pth')

    if not os.path.exists(snapshot):
        logging.error(f"Snapshot not found at {snapshot} or fallbacks.")
        sys.exit(1)

    logging.info(f"Loading model snapshot from {snapshot}")
    try:
        # 加载 checkpoint 字典
        checkpoint = torch.load(snapshot, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # 提取模型状态字典
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint  # 兼容旧格式

        # 处理 DataParallel 包装的键
        new_state_dict = OrderedDict()
        is_parallel = any(k.startswith('module.') for k in state_dict.keys())
        is_model_parallel = isinstance(net, torch.nn.DataParallel)  # 检查当前模型是否为 parallel

        if is_parallel and not is_model_parallel:
            # Checkpoint 是 parallel, 模型不是: 移除 'module.'
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            msg = net.load_state_dict(new_state_dict)
        elif not is_parallel and is_model_parallel:
            # Checkpoint 不是 parallel, 模型是: 添加 'module.'
            for k, v in state_dict.items():
                new_state_dict['module.' + k] = v
            msg = net.load_state_dict(new_state_dict)
        else:
            # 状态一致
            msg = net.load_state_dict(state_dict)

        print("Loaded self trained swin unet.", msg)
        logging.info(f"Loaded model state. Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}")

    except Exception as e:
        logging.error(f"Failed to load checkpoint from {snapshot}: {e}")
        # 尝试后备加载（假设是旧格式）
        try:
            logging.info("Attempting fallback load (assuming old format)...")
            state_dict = torch.load(snapshot, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            msg = net.load_state_dict(state_dict)
            print("Loaded self trained swin unet (fallback).", msg)
            logging.info(
                f"Loaded model state (fallback). Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}")
        except Exception as e2:
            logging.error(f"Fallback load also failed: {e2}")
            sys.exit(1)

    # msg = net.load_state_dict(torch.load(snapshot)) # <-- 原始代码
    # print("self trained swin unet", msg) # <-- 原始代码
    snapshot_name = snapshot.split('/')[-1]
    # --- 结束修改 ---

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)

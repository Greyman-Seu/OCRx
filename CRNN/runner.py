import argparse
from argparse import Namespace
import logging
import os

import torch
from torch.utils.data import DataLoader

import configure as cfg
import dataset
import utils
# from model import crnn
from model import crnn_sim as crnn


def transfer_channel(args: Namespace):
    transfer = utils.ImageChannelTransformer(args.src, args.dst)
    transfer()
    return


def count(args: Namespace):
    # srcDir = '/home/chuan/dataset/captcha/raw_data/fake_pic_train'
    chars, colors = utils.labelDistribution(args.src)
    return chars, colors


def generate_lmdb(args: Namespace):
    maker = dataset.LMDBMaker(args.src, args.dst)
    maker()
    return


def infer(args: Namespace):
    if not args.src.endswith('/test/'):
        logging.warning("LMDB dataset here is expected to be a test set.")

    test_dataset = dataset.lmdbDataset(
        root=args.src, transform=dataset.ResizeNormalize((cfg.imgW, cfg.imgH), crop=cfg.crop))
    test_loader = DataLoader(test_dataset, shuffle=False,
                             batch_size=2*cfg.batchSize, num_workers=int(cfg.workers))

    device = torch.device("cuda")
    codec = dataset.Codec(cfg.alphabet)
    nClass = len(cfg.alphabet) + 1
    model = crnn.CRNN(cfg.imgH, cfg.nc, nClass, cfg.nh).to(device)
    print(model)
    para = torch.load(os.path.join(
        "/home/chuan/captcha/crnn/weights", args.para))
    model.load_state_dict(para)

    utils.infer(model, test_loader, device, codec)
    return


def resolve(args: Namespace):
    filename = os.path.join('/home/chuan/captcha/crnn/log', args.log)
    # log = utils.readInferLog(filename, flag1='=>', flag2="gt:")
    log = utils.readInferLog(filename)
    info = utils.StatisticalInfo(log, cfg.alphabet)
    print(info.acc_total)
    print(info.acc_unit)
    print(info.acc_table[:10])
    return log, info


if __name__ == '__main__':
    path_opts = {"lmdb_10w": "/home/chuan/dataset/captcha/raw_data/lmdb",
                 "lmdb_31w": "/home/chuan/dataset/captcha/red_data/lmdb/",
                 "path_10w_train": '/home/chuan/dataset/captcha/raw_data/fake_pic_train',
                 "path_10w_test": '/home/chuan/dataset/captcha/raw_data/fake_pic_test',
                 "path_31w_train": '/home/chuan/dataset/captcha/red_data/train',
                 "path_31w_test": '/home/chuan/dataset/captcha/red_data/test',
                 }
    funcs = {"transfer_channel": transfer_channel,
             "count": count,
             "generate_lmdb": generate_lmdb,
             "infer": infer,
             "resolve": resolve}

    # print("Here are some path options:")
    # for k, v in path_opts.items():
    #     print(k, v, sep=": ")

    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, required=True,
                        help='Function you want to run.')
    parser.add_argument('--src', type=str, default="/home/chuan/dataset/captcha/red_data/train",
                        help='Path to folder which contains the images.')
    parser.add_argument('--dst', type=str, default="/home/chuan/dataset/captcha/red_data/lmdb/train",
                        help='LMDB data output path.')
    parser.add_argument('--log', type=str, default='inference.log',
                        help='Log file you want to process, should be in path /home/chuan/captcha/crnn/log')
    parser.add_argument('--para', type=str, default='0.85.pt')
    args = parser.parse_args()
    if args.func not in funcs:
        logging.warning(f"Function {args.func} doesn't exist!")
    else:
        funcs[args.func](args)

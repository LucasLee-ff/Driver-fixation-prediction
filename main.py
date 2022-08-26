import argparse
import os
import time
import pickle
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from data_load import ImageList
import random
import warnings
import logging
import numpy as np
import json
from TransCDNN_model import config
from TransCDNN_model.vit_cdnn_modeling import TransCDNN
from tqdm import tqdm

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=36, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('-g', '--gpu', default='0,1', type=str,
                    metavar='N', help='GPU NO. (default: 0)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--split', default=0, type=int)
args = parser.parse_args()

name = 'TransCDNN'
num_layers = 8
path = './ckpts-0412/reshape/transcdnn_'
ckpts = path + str(num_layers) + '/'

if not os.path.exists(ckpts): os.makedirs(ckpts)

log_file = os.path.join(ckpts + "/train_log_%s.txt" % (name, ))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename=log_file)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logging.getLogger('').addHandler(console)


def main():
    # global args, best_score
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    config_vit = config.get_cdnn_b16_config()
    config_vit.transformer['num_layers'] = num_layers

    model = TransCDNN(config_vit, num_classes=config_vit.n_classes)
    #model = nn.DataParallel(model)
    #model = model.cuda()
    # model.load_from(weights=np.load(config_vit.pretrained_path))

    params = model.parameters()

    cudnn.benchmark = True

    optimizer = torch.optim.Adam(params, args.lr,
                                weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    loss_history_path = ckpts + 'loss_history.pkl'

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            best_loss = checkpoint['best_loss']  # epoch 19, 59
            best_epoch = checkpoint['best_epoch']
            with open(loss_history_path, 'rb') as f:
                loss_history = pickle.load(f)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
    else:
        loss_history = {'train_loss_history': [], 'valid_loss_history': []}
        best_loss = float('inf')
        best_epoch = 0
        print("=> no checkpoint found at '{}'".format(args.resume))

    root = './traffic_dataset/traffic_frames/'

    train_imgs = [json.loads(line) for line in open(root + 'train.json')]

    valid_imgs = [json.loads(line) for line in open(root + 'valid.json')]

    test_imgs = [json.loads(line) for line in open(root + 'test.json')]

    train_loader = DataLoader(
            ImageList(root, train_imgs, for_train=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers,
            pin_memory=True)
    valid_loader = DataLoader(
            ImageList(root, valid_imgs),
            batch_size=16, shuffle=False,
            num_workers=args.workers,
            pin_memory=True)
    test_loader = DataLoader(
            ImageList(root, test_imgs),
            batch_size=64, shuffle=False,
            num_workers=args.workers,
            pin_memory=True)

    criterion = nn.BCELoss().cuda()

    logging.info('-------------- New training session, LR = %f ----------------' % (args.lr, ))
    logging.info('-- length of training images = %d--length of valid images = %d--' % (len(train_imgs),len(valid_imgs)))
    logging.info('-- length of test images = %d--' % (len(test_imgs)))

    best_file_name = os.path.join(ckpts, 'model_best.tar')

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(
                train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        valid_loss = validate(
                valid_loader, model, criterion)

        scheduler.step()

        loss_history['train_loss_history'].append(train_loss)
        loss_history['valid_loss_history'].append(valid_loss)

        file_name_last = os.path.join(ckpts, 'model_epoch_%d.tar' % (epoch + 1,))
        file_name_former = os.path.join(ckpts, 'model_epoch_%d.tar' % epoch)

        best_loss = min(valid_loss, best_loss)

        # remember best lost and save checkpoint as 'model_best.tar'
        if valid_loss == best_loss:
            best_epoch = epoch + 1
            if os.path.isfile(best_file_name):
                os.remove(best_file_name)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
                'valid_loss': valid_loss,
                'best_loss': best_loss
            }, best_file_name)

        # save latest model
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(),
            'valid_loss': valid_loss,
            'best_loss': best_loss,
            'best_epoch': best_epoch
        }, file_name_last)

        # delete models from former epochs
        if epoch != 0:
            os.remove(file_name_former)

        msg = 'Epoch: {:02d} Train loss {:.4f} | Valid loss {:.4f} | Best loss {:.4f} from epoch {:02d} '.format(
                epoch+1, train_loss, valid_loss, best_loss, best_epoch)
        logging.info(msg)

        # save latest loss history
        if os.path.isfile(loss_history_path):
            os.remove(loss_history_path)
        with open(loss_history_path, 'wb') as f:
            pickle.dump(loss_history, f)

    checkpoint = torch.load(best_file_name)
    model.load_state_dict(checkpoint['state_dict'])
    outputs, targets = predict(test_loader, model)

    np.save(ckpts + 'p_' + str(num_layers) + '.npy', outputs)
    np.save(ckpts + 't_' + str(num_layers) + '.npy', targets)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    bar = tqdm(enumerate(train_loader), total=len(train_loader))
    # switch to train mode
    model.train()
    start = time.time()
    for i, (input, target) in bar:
        if (i+1) % 100 == 0:
            print('Epoch', epoch+1, 'Iter', i+1)

        input = input#.cuda()
        target = target#.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.item(), target.size(0))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if (i+1) % 750 == 0:
            msg = 'Training Epoch {:03d}  Iter {:03d} Loss_avg {:.6f} in {:.3f}s'.format(epoch+1, i+1, losses.avg, time.time() - start)
            start = time.time()
            logging.info(msg)
            print(msg)

    return losses.avg


def validate(valid_loader, model, criterion):
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    start = time.time()
    for i, (input, target) in enumerate(valid_loader):
        input = input.cuda()
        target = target.cuda()

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)
        # measure accuracy and record loss
        losses.update(loss.item(), target.size(0))

        if (i+1) % 100 == 0:
            msg = 'Validating Iter {:03d} Loss {:.6f} in {:.3f}s'.format(i+1, losses.avg, time.time() - start)
            start = time.time()
            # logging.info(msg)
            print(msg)

    return losses.avg


def predict(valid_loader, model):

    # switch to evaluate mode
    model.eval()

    targets = []
    outputs = []

    for i, (input, target) in enumerate(valid_loader):
        print(i+1, '/', len(valid_loader))
        targets.append(target.numpy().squeeze(1))

        input = input.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            # compute output
            output = model(input_var)
        outputs.append(output.data.cpu().numpy().squeeze(1))

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    return outputs, targets


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // (args.epochs//5)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

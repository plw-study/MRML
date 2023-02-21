# encoding:utf-8
# ------------------------------------------------------------
# ICASSP 2023
# Code for "MRML: Multimodal Rumor Detection by Deep Metric Learning"
# ------------------------------------------------------------

import os
import torch
import argparse
import yaml
import sys
from tensorboardX import SummaryWriter
import torch.nn as nn

from pytorchtools import EarlyStopping
import utils
import data_prase
import model
import engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--options_file', default='./options.yaml',
                        help='yaml file for some experiment settings.')
    parser.add_argument('--log_dir', default='./logs', help='dir for logs.')
    parser.add_argument('--dataset_dir', default='./pre_data/weibo_dataset',
                        help='dataset dir for weibo or Twitter.')
    # parser.add_argument('--text_pretrained_model', default='Bert',
    #                     help='text pretrain model name (w2v, Bert or XLNET).')
    parser.add_argument('--save_dir', default='./save_model', help='model save dir.')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate of the model.')
    parser.add_argument('--batch_size', default=128, type=int, help='size of a training batch.')
    parser.add_argument('--num_epochs', default=300, type=int, help='number of training epochs.')
    args = parser.parse_args()

    with open(args.options_file, 'r') as f_opt:
        options = yaml.load(f_opt)

    print('1.load data from {}...............'.format(args.dataset_dir))
    train_loader, val_loader, test_loader = data_prase.get_loaders(
        args.dataset_dir, options['dataset']['val_rate'], args.batch_size, options['dataset']['val_from'])

    print('2.build the model..........')
    model1 = model.factory(options['model'], cuda=True)
    print('  model has {} parameters.'.format(utils.params_count(model1)))

    # tensorboard
    new_log_dir = args.log_dir + './lr{}_ep{}_bs{}/'.format(args.learning_rate, args.num_epochs, args.batch_size)
    if not os.path.exists(new_log_dir):
        os.makedirs(new_log_dir)
    else:
        for files in os.listdir(new_log_dir):
            os.remove(new_log_dir + '/' + files)
    sum_writter = SummaryWriter(log_dir=new_log_dir)
    print('  the tensorboard log dir is:', new_log_dir)

    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer_model1 = torch.optim.Adam(model1.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer_model1, step_size=20, gamma=0.5)

    print('3.start training................')
    model_file = '{}/lr{}_es_bs{}.pth.tar'.format(args.save_dir, args.learning_rate, args.batch_size)
    early_stopping = EarlyStopping(patience=10, verbose=False, path=model_file)
    ifEarlyStop = False
    for epoch in range(0, args.num_epochs):

        engine.train_part(train_loader, model1, criterion, optimizer_model1,
                          epoch, options['optim'], sum_writter)

        val_loss, val_acc = engine.val_part(val_loader, model1, criterion,
                                            epoch, options['optim'], sum_writter, 'val')

        early_stopping(val_loss, model1)
        if early_stopping.early_stop:
            ifEarlyStop = True
            print('4.main model early stop in epoch {}........'.format(epoch-10))
            break
    if not ifEarlyStop:
        torch.save(model1.state_dict(), model_file)
        print('4.main model did not early stop, finish running at {} epoch.....'.format(args.num_epochs))

    test_model1 = model.factory(options['model'], cuda=True)
    test_model1.load_state_dict(torch.load(model_file))
    whole_acc, fake_pre, fake_rec, fake_f1, fake_su, real_pre, real_rec, real_f1, real_su = \
        engine.test_part(test_loader, test_model1)
    with open('./final_result.txt', 'a+') as f:
        f.write('{}_{}_{}_{}_{}_{}_{}_{}_{}\n'.format(
            whole_acc, fake_pre, fake_rec, fake_f1, fake_su,
            real_pre, real_rec, real_f1, real_su))

    sum_writter.close()


if __name__ == '__main__':
    main()

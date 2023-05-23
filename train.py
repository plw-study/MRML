# encoding:utf-8
# ------------------------------------------------------------
# Liwen Peng
# Writen by Liwen Peng, 2023
# the code for ICASSP 2023 paper: MRML: MULTIMODAL RUMOR DETECTION BY DEEP METRIC LEARNING
# ------------------------------------------------------------

import torch
import argparse
import yaml
import torch.nn as nn
import numpy as np

from pytorchtools import EarlyStopping
import model
import data_prase
import engine
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--options_file', default='./options.yaml', type=str, help='file for options.')
    parser.add_argument('--log_dir', default='./logs', type=str,
                        help='dir for logs.')
    parser.add_argument('--dataset', default='Weibo', help='the dataset to use.')
    parser.add_argument('--pretrained_model', default='Bert', help='the pretrained_model to use.')
    parser.add_argument('--learning_rate', default=1e-4, help='learning rate of the model.')
    parser.add_argument('--batch_size', default=128, type=int, help='size of a training batch.')
    parser.add_argument('--num_epochs', default=300, type=int, help='number of training epochs.')
    parser.add_argument('--l2_reg', default=1e-4, help='weight decay for optimizer.')
    parser.add_argument('--fusion_type', default='fusion', type=str, help='type of multimodal fusion.')
    parser.add_argument('--seed', default=42, type=int, help='the random seed for all.')
    args = parser.parse_args()

    with open(args.options_file, 'r') as f_opt:
        options = yaml.load(f_opt, Loader=yaml.FullLoader)
    options['model']['fusion']['type'] = args.fusion_type

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.dataset == 'Weibo':
        args.learning_rate = 1e-5
        dataset_dir = './datafiles/MM17-WeiboRumorSet/Bert_50_768'
        options['model']['fusion']['dim_t'] = 50*768

    elif args.dataset == 'Twitter':
        args.learning_rate = 5e-4
        dataset_dir = './datafiles/image-verification-corpus/Bert_50_768'
        options['model']['fusion']['dim_t'] = 50*768

    else:
        raise ValueError('dataset must be Weibo or Twitter !!!!!!!!!1')

    print('1.load {} data using {} model ...............'.format(args.dataset, args.pretrained_model))
    train_loader, test_loader = data_prase.get_loaders(dataset_dir, args.batch_size)

    print('2.build the model..........')
    model1 = model.factory(options['model'], cuda=True)

    criterion = nn.BCEWithLogitsLoss().cuda()

    params_base = list(model1.linear_calculsim.parameters())
    params_base += list(model1.linear_v.parameters())
    params_base += list(model1.linear_t.parameters())
    params_base += list(model1.bn_layer_v.parameters())
    params_base += list(model1.bn_layer_t.parameters())
    params_base += list(model1.list_linear_hv.parameters())
    params_base += list(model1.list_linear_ht.parameters())
    params_base += list(model1.fake_ln1.parameters())
    params_base += list(model1.fake_last.parameters())
    params_base += list(model1.bn_layer1.parameters())
    params_base += list(model1.bn_layer4.parameters())

    params_img = params_base + list(model1.dist_learning_v.parameters())
    params_txt = params_base + list(model1.dist_learning_t.parameters())
    optimizer_txt = torch.optim.Adam(params_txt, lr=args.learning_rate, weight_decay=float(args.l2_reg))
    optimizer_img = torch.optim.Adam(params_img, lr=args.learning_rate, weight_decay=float(args.l2_reg))

    start_epoch = 0
    print('3.start training from checkpoint {} ................'.format(start_epoch))

    model_file = '{}/{}_whole.pth.tar'.format(
        options['optim']['save_model_dir'], args.dataset)
    early_stopping = EarlyStopping(patience=20, verbose=False, path=model_file)
    ifEarlyStop = False
    for epoch in range(start_epoch, args.num_epochs):

        engine.train_part(train_loader, model1, criterion, optimizer_txt, optimizer_img,
                          epoch, options['optim'])
        val_loss, val_acc = engine.val_part(test_loader, model1, criterion, options['optim'], 'val')

        early_stopping(-val_acc, model1)
        if early_stopping.early_stop:
            ifEarlyStop = True
            print('4.main model early stop in epoch {}........'.format(epoch-30))
            break

    if not ifEarlyStop:
        torch.save(model1.state_dict(), model_file)
        print('4.main model did not early stop, finish running at {} epoch.....'.format(args.num_epochs))

    test_model1 = model.factory(options['model'], cuda=True)
    test_model1.load_state_dict(torch.load(model_file))
    whole_acc, fake_pre, fake_rec, fake_f1, fake_su, real_pre, real_rec, real_f1, real_su = \
        engine.test_part(test_loader, test_model1)
    print('5. test acc results: ', whole_acc)
    print('fake news results:', fake_pre, fake_rec, fake_f1, fake_su)
    print('real news results:', real_pre, real_rec, real_f1, real_su)


if __name__ == '__main__':
    main()

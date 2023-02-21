import torch
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import utils


def train_part(train_load, model, criterion, optimizer_t, epoch, optim, writter):
    model.train()
    train_num = 0
    train_loss = 0.0
    train_triplet_loss = 0.0
    train_pairwise_loss = 0.0
    train_detection_loss = 0.0
    skl_acc = 0.0

    for step, train_data in enumerate(train_load):

        input_image, input_text, input_labels = train_data
        if torch.cuda.is_available():
            input_image = input_image.cuda()
            input_text = input_text.cuda()
        scores, image_emb, text_emb, image_dis, text_dis, mm_emb, fake_det = model(input_image, input_text)
        torch.cuda.synchronize()

        pre_labels = fake_det.clone().detach()
        pre_labels[pre_labels > 0.5] = 1
        pre_labels[pre_labels <= 0.5] = 0
        acc = accuracy_score(input_labels.detach().cpu(), pre_labels.cpu())
        skl_acc += acc * input_image.size(0)

        rumor_detection_loss = criterion(fake_det, input_labels.float())
        pairwise_loss = utils.calcul_pairwise_loss(scores, optim, image_emb, text_emb) / input_image.size(0)
        if optim['direct_modality'] == 'text':
            triplet_loss = utils.calcul_triplet_loss(text_emb, image_emb, optim, image_dis) / input_image.size(0)
        else:
            triplet_loss = utils.calcul_triplet_loss(image_emb, text_emb, optim, text_dis) / input_image.size(0)
        loss_all = triplet_loss + pairwise_loss + rumor_detection_loss

        optimizer_t.zero_grad()
        loss_all.backward()
        torch.cuda.synchronize()
        optimizer_t.step()
        torch.cuda.synchronize()

        train_num += input_image.size(0)
        train_loss += loss_all
        train_detection_loss += rumor_detection_loss
        train_pairwise_loss += pairwise_loss
        train_triplet_loss += triplet_loss

        niter = epoch * len(train_load) + step + 1
        if niter % optim['print_freq'] == 0:
            writter.add_scalar('train_loss/pairwise_loss', pairwise_loss, global_step=niter)
            writter.add_scalar('train_loss/triplet_loss', triplet_loss, global_step=niter)
            writter.add_scalar('train_loss/detection_loss', rumor_detection_loss, global_step=niter)
            writter.add_scalar('train_loss/all_loss', loss_all, niter)

    print('  epoch {:<3d}:train loss {:<6f} triplet {:<6f} pairwise {:<6f} fake {:<6f} acc {:<6f}'.format(
        epoch, train_loss/len(train_load), train_triplet_loss/len(train_load),
        train_pairwise_loss/len(train_load), train_detection_loss/len(train_load),
        skl_acc/train_num), end=' ')


def val_part(val_load, model, criterion, epoch, optim, writter, data_type='val'):
    model.eval()
    val_loss_all = 0.0
    val_pairwise_loss = 0.0
    val_triplet_loss = 0.0
    val_detection_loss = 0.0
    skl_acc = 0.0
    val_num = 0
    for step, val_data in enumerate(val_load):
        input_image, input_text, input_label = val_data
        scores, image_emb, text_emb, image_dis, text_dis, mm_emb, fake_dec = model(input_image, input_text)

        pre_labels = fake_dec.clone().detach()
        pre_labels[pre_labels > 0.5] = 1
        pre_labels[pre_labels <= 0.5] = 0
        acc = accuracy_score(input_label.detach().cpu(), pre_labels.cpu())
        skl_acc += acc * input_image.size(0)

        rumor_detection_loss = criterion(fake_dec, input_label.float())
        pairwise_loss = utils.calcul_pairwise_loss(scores, optim, image_emb, text_emb) / input_image.size(0)
        if optim['direct_modality'] == 'text':
            triplet_loss = utils.calcul_triplet_loss(text_emb, image_emb, optim, image_dis) / input_image.size(0)
        else:
            triplet_loss = utils.calcul_triplet_loss(image_emb, text_emb, optim, text_dis) / input_image.size(0)
        val_loss = triplet_loss + rumor_detection_loss + pairwise_loss

        val_loss_all += val_loss
        val_pairwise_loss += pairwise_loss
        val_triplet_loss += triplet_loss
        val_detection_loss += rumor_detection_loss
        val_num += input_image.size(0)

        niter = epoch * len(val_load) + step + 1
        if niter % optim['print_freq'] == 0:
            writter.add_scalar('{}_loss/all_loss'.format(data_type), val_loss, niter)
            writter.add_scalar('{}_loss/pairwise_loss'.format(data_type), pairwise_loss, niter)
            writter.add_scalar('{}_loss/triplet_loss'.format(data_type), triplet_loss, niter)
            writter.add_scalar('{}_loss/detection_loss'.format(data_type), rumor_detection_loss, niter)

    print('| {} {:<6f} pairwise {:<6f} triplet {:<6f} fake {:<6f} acc {:<6f}'.format(
        data_type, val_loss_all/len(val_load), val_pairwise_loss/len(val_load), val_triplet_loss/len(val_load),
        val_detection_loss/len(val_load), skl_acc/val_num))

    return val_loss_all/len(val_load), skl_acc/val_num


def test_part(test_load, model):
    print('5.test the best model............')
    model.eval()
    test_labels = []
    predict_labels = []
    for step, test_data in enumerate(test_load):
        input_image, input_text, input_label = test_data
        _, _, _, _, _, mm_embs, fake_dec = model(input_image, input_text)
        pre_labels = fake_dec.clone().detach()
        pre_labels[pre_labels > 0.5] = 1
        pre_labels[pre_labels <= 0.5] = 0
        test_labels.extend(input_label.detach().cpu().numpy().tolist())
        predict_labels.extend(pre_labels.cpu().numpy().tolist())

    test_labels = np.array(test_labels)
    predict_labels = np.array(predict_labels)
    test_acc = accuracy_score(test_labels, predict_labels)
    pres, recalls, f1s, supports = precision_recall_fscore_support(
        test_labels[:, 0], predict_labels[:, 0], labels=[0, 1], average=None)

    print('  test acc is:', test_acc)
    print('  fake news pre, recall, f1, support results:', pres[0], recalls[0], f1s[0], supports[0])
    print('  real news pre, recall, f1, support results:', pres[1], recalls[1], f1s[1], supports[1])
    return test_acc, pres[0], recalls[0], f1s[0], supports[0], pres[1], recalls[1], f1s[1], supports[1]





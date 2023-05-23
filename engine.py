import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score

import utils


def train_part(train_load, model, criterion, optimizer_t, optimizer_i, epoch, optim):
    model.train()
    train_indicator = (epoch // optim['train_change_num']) % 2
    train_num = 0
    train_loss = 0.0
    train_cross_loss = 0.0
    train_inner_loss = 0.0
    train_fake_loss = 0.0
    ind_str = 'text' if train_indicator == 0 else 'img'
    skl_acc = 0.0

    for step, train_data in enumerate(train_load):

        input_image, input_text, input_labels, _, _ = train_data
        if torch.cuda.is_available():
            input_image = input_image.cuda()
            input_text = input_text.cuda()
        scores, image_emb, text_emb, image_dis, text_dis, _, fake_res = model(input_image, input_text)
        torch.cuda.synchronize()

        pre_labels = fake_res.clone().detach()
        pre_labels[pre_labels > 0.5] = 1
        pre_labels[pre_labels <= 0.5] = 0
        acc = accuracy_score(input_labels.detach().cpu(), pre_labels.cpu())
        skl_acc += acc * input_image.size(0)

        fake_detection_loss = criterion(fake_res, input_labels.float())
        cross_loss = utils.pairwise_loss_with_label(scores, optim, input_labels) / input_image.size(0)

        if ind_str == 'img':
            inner_loss = utils.triplet_loss_with_label(
                image_emb, text_emb, optim, text_dis, input_labels) / input_image.size(0)
            loss_all = inner_loss + cross_loss + fake_detection_loss

            optimizer_t.zero_grad()
            loss_all.backward()
            torch.cuda.synchronize()
            optimizer_t.step()
            torch.cuda.synchronize()
        else:
            inner_loss = utils.triplet_loss_with_label(
                text_emb, image_emb, optim, image_dis, input_labels) / input_image.size(0)
            loss_all = inner_loss + cross_loss + fake_detection_loss

            optimizer_i.zero_grad()
            loss_all.backward()
            torch.cuda.synchronize()
            optimizer_i.step()
            torch.cuda.synchronize()

        train_num += input_image.size(0)
        train_loss += loss_all
        train_fake_loss += fake_detection_loss
        train_cross_loss += cross_loss
        train_inner_loss += inner_loss

    print('epoch {:<3d}: {:<4s} train loss {:<6f} cross {:<6f} inner {:<6f} fake {:<6f} acc {:<6f}'.format(
        epoch, ind_str, train_loss/len(train_load), train_cross_loss/len(train_load),
        train_inner_loss/len(train_load), train_fake_loss/len(train_load),
        skl_acc/train_num), end=' ')


def val_part(val_load, model, criterion, optim, data_type='val'):
    model.eval()
    val_loss_all = 0.0
    val_cross_loss = 0.0
    val_inner_loss = 0.0
    val_fake_loss = 0.0
    skl_acc = 0.0
    val_num = 0
    for step, val_data in enumerate(val_load):
        input_image, input_text, input_label, _, _ = val_data
        scores, image_emb, text_emb, image_dis, text_dis, mm_embs, fake_res = model(input_image, input_text)

        pre_labels = fake_res.clone().detach()
        pre_labels[pre_labels > 0.5] = 1
        pre_labels[pre_labels <= 0.5] = 0
        acc = accuracy_score(input_label.detach().cpu(), pre_labels.cpu())
        skl_acc += acc * input_image.size(0)

        fake_loss = criterion(fake_res, input_label.float())
        cross_loss = utils.pairwise_loss_with_label(scores, optim, input_label) / input_image.size(0)
        inner_loss1 = utils.triplet_loss_with_label(
            text_emb, image_emb, optim, image_dis, input_label) / input_image.size(0)
        inner_loss2 = utils.triplet_loss_with_label(
            image_emb, text_emb, optim, text_dis, input_label) / input_image.size(0)
        val_loss = inner_loss1/2.0 + inner_loss2/2.0 + fake_loss + cross_loss

        val_loss_all += val_loss
        val_cross_loss += cross_loss
        val_inner_loss += inner_loss1/2.0
        val_inner_loss += inner_loss2/2.0
        val_fake_loss += fake_loss
        val_num += input_image.size(0)

    print('| {} {:<6f} cross {:<6f} inner {:<6f} fake {:<6f} acc {:<6f}'.format(
        data_type, val_loss_all/len(val_load), val_cross_loss/len(val_load), val_inner_loss/len(val_load),
         val_fake_loss/len(val_load), skl_acc/val_num))

    return val_loss_all/len(val_load), skl_acc/val_num


def test_part(test_load, model):
    print('7.test the best model.............')
    model.eval()
    test_labels = []
    predict_labels = []
    test_ids = []
    for step, test_data in enumerate(test_load):
        input_image, input_text, input_label, _, input_id = test_data
        _, _, _, _, _, mm_embs, fake_dec = model(input_image, input_text)
        pre_labels = fake_dec.clone().detach()
        pre_labels[pre_labels > 0.5] = 1
        pre_labels[pre_labels <= 0.5] = 0
        test_labels.extend(input_label.detach().cpu().numpy().tolist())
        predict_labels.extend(pre_labels.cpu().numpy().tolist())
        test_ids.extend(input_id)

    test_labels = np.array(test_labels)
    predict_labels = np.array(predict_labels)
    test_acc = accuracy_score(test_labels, predict_labels)
    whole_pre = precision_score(test_labels[:, 0], predict_labels[:, 0])
    whole_rec = recall_score(test_labels[:, 0], predict_labels[:, 0])
    whole_f1 = f1_score(test_labels[:, 0], predict_labels[:, 0])
    pres, recalls, f1s, supports = precision_recall_fscore_support(
        test_labels[:, 0], predict_labels[:, 0], labels=[0, 1], average=None)

    print('test acc, pre, rec, f1 results:', test_acc, whole_pre, whole_rec, whole_f1)
    return test_acc, pres[0], recalls[0], f1s[0], supports[0], pres[1], recalls[1], f1s[1], supports[1]





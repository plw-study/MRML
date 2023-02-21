import torch.utils.data as data
import numpy as np
import torch
import torch.utils
import math
from sklearn import preprocessing


class PreDataset(data.Dataset):
    """
    Load Prepare datasets, including images and texts
    Possible datasets: Weibo, Twitter
    """
    def __init__(self, data_path, data_split, val_rate=0.0, random_seed=42):
        data_split1 = 'train' if data_split == 'train' else 'test'
        self.images = np.load('{}/{}_image_embed.npy'.format(data_path, data_split1))
        self.images = self.images.astype(np.float)

        self.labels = np.load('{}/{}_label.npy'.format(data_path, data_split1))

        self.texts_i = np.load('{}/{}_text_embed.npy'.format(data_path, data_split1))
        self.length = len(self.labels)
        texts_dim = self.texts_i.shape[1] * self.texts_i.shape[2]
        self.texts = np.reshape(self.texts_i, (self.length, texts_dim))

        if data_split == 'train':
            pass
        elif data_split == 'val' or data_split == 'test':
            fake_label_indexs_in_test = np.where(self.labels[:, 0] < 1)[0]
            real_label_indexs_in_test = np.where(self.labels[:, 0] > 0)[0]
            fake_label_num_in_test = len(fake_label_indexs_in_test)
            real_label_num_in_test = len(real_label_indexs_in_test)
            fake_val_num = math.floor(fake_label_num_in_test * val_rate)
            real_val_num = math.floor(real_label_num_in_test * val_rate)

            test_indexs = np.array([i for i in range(len(self.labels))])
            np.random.seed(random_seed)
            fake_val_index = np.random.choice(fake_label_indexs_in_test, fake_val_num, replace=False)
            real_val_index = np.random.choice(real_label_indexs_in_test, real_val_num, replace=False)
            test_index = np.delete(test_indexs, fake_val_index)
            test_index = np.delete(test_index, real_val_index)

            if data_split == 'val':
                self.texts = np.concatenate((self.texts[fake_val_index], self.texts[real_val_index]),
                                             axis=0)
                self.images = np.concatenate((self.images[fake_val_index], self.images[real_val_index]),
                                              axis=0)
                self.labels = np.concatenate((self.labels[fake_val_index], self.labels[real_val_index]),
                                             axis=0)
                self.length = len(self.labels)
            else:
                self.texts = self.texts[test_index]
                self.images = self.images[test_index]
                self.labels = self.labels[test_index]
                self.length = len(self.labels)
        else:
            raise ValueError('data split must be train, val or test!')

        print('  {} text emb shape {}:'.format(data_split, self.texts.shape))
        print('  {} image emb shape {}:'.format(data_split, self.images.shape))
        print('  {} label emb shape {} fake {} real {}:'.format(
            data_split, self.labels.shape,
            len(np.where(self.labels[:, 0]<1)[0]), len(np.where(self.labels[:, 0]>0)[0])))

    def __getitem__(self, index):
        image_embs = torch.tensor(self.images[index]).float()
        text_embs = torch.tensor(self.texts[index]).float()
        labels = torch.tensor(self.labels[index])

        if torch.cuda.is_available():
            image_embs = image_embs.cuda()
            text_embs = text_embs.cuda()
            labels = labels.cuda()
        return image_embs, text_embs, labels

    def __len__(self):
        return self.length


def get_loaders(data_path, val_rate, batch_size, val_from='train'):
    if val_from not in ['train', 'test']:
        raise ValueError('val from must be train or test!!')

    train_data = PreDataset(data_path, 'train', 0, 0)
    val_data = PreDataset(data_path, 'val', val_rate, 5)
    test_data = PreDataset(data_path, 'test', val_rate, 5)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True
                                               )
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=batch_size,
                                             shuffle=True
                                             )
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=batch_size,
                                              shuffle=True
                                              )

    return train_loader, val_loader, test_loader

import torch.utils.data as data
import numpy as np
import torch
import torch.utils


class PreDataset(data.Dataset):
    """
    Load Prepare datasets, including images and texts
    Possible datasets: Weibo, Twitter
    """
    def __init__(self, data_path, data_split):
        self.images = np.load('{}/{}_image_embed.npy'.format(data_path, data_split))
        self.images = self.images.astype(np.float)

        self.labels = np.load('{}/{}_label.npy'.format(data_path, data_split))
        self.ids = np.load('{}/{}_post_ids.npy'.format(data_path, data_split))

        self.texts_i = np.load('{}/{}_text_embed.npy'.format(data_path, data_split))
        self.length = len(self.labels)
        texts_dim = self.texts_i.shape[1] * self.texts_i.shape[2]
        self.texts = np.reshape(self.texts_i, (self.length, texts_dim))

        print('  {} text emb shape {}:'.format(data_split, self.texts.shape))
        print('  {} image emb shape {}:'.format(data_split, self.images.shape))
        print('  {} label emb shape {} with rumor {}, non-rumor {}:'.format(
            data_split, self.labels.shape,
            len(np.where(self.labels[:, 0] < 1)[0]), len(np.where(self.labels[:, 0] > 0)[0])))

    def __getitem__(self, index):
        image_embs = torch.tensor(self.images[index]).float()
        text_embs = torch.tensor(self.texts[index]).float()
        labels = torch.tensor(self.labels[index])
        ids = self.ids[index]

        if torch.cuda.is_available():
            image_embs = image_embs.cuda()
            text_embs = text_embs.cuda()
            labels = labels.cuda()
        return image_embs, text_embs, labels, labels, ids

    def __len__(self):
        return self.length


def get_loaders(data_path, batch_size):

    train_data = PreDataset(data_path, 'train')
    test_data = PreDataset(data_path, 'test')

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_loader, test_loader


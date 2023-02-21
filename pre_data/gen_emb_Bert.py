import os
import codecs
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, load_vocabulary
import numpy as np
import pickle as pkl
import tqdm

seq_len = 50


def get_weibo_matrix(data_type, tokenizer):
    corpus_dir = '/XXXX/MM17-WeiboRumorSet/tweets/'
    all_img_embed = pkl.load(
        open('/XXXX/MM17-WeiboRumorSet/img_emb_vgg19.pkl', 'rb'))
    rumor_content = open('{}/{}_rumor.txt'.format(corpus_dir, data_type)).readlines()
    nonrumor_content = open('{}/{}_nonrumor.txt'.format(corpus_dir, data_type)).readlines()

    text_matrix = []
    image_matrix = []
    labels = []

    n_lines = len(rumor_content)
    for idx in range(2, n_lines, 3):
        one_rumor = rumor_content[idx].strip()
        if one_rumor:
            images = rumor_content[idx-1].split('|')
            for image in images:
                img = image.split('/')[-1].split('.')[0]
                if img in all_img_embed:
                    image_matrix.append(all_img_embed[img])
                    labels.append([0, 1])
                    text_tokens, _ = tokenizer.encode(first=one_rumor, max_len=seq_len)
                    text_matrix.append(text_tokens)
                    break

    n_lines = len(nonrumor_content)
    for idx in range(2, n_lines, 3):
        one_rumor = nonrumor_content[idx].strip()
        if one_rumor:
            images = nonrumor_content[idx-1].split('|')
            for image in images:
                img = image.split('/')[-1].split('.')[0]
                if img in all_img_embed:
                    image_matrix.append(all_img_embed[img])
                    labels.append([1, 0])
                    text_tokens, _ = tokenizer.encode(first=one_rumor, max_len=seq_len)
                    text_matrix.append(text_tokens)
                    break
    return text_matrix, image_matrix, labels, labels


def get_twitter_matrix(phase, tokenizer_input):
    text_tokens_matrix = []
    image_matrix = []
    labels = []
    label_dict = {'fake': [0, 1], 'real': [1, 0]}

    corpus_dir = '/XXXX/image-verification-corpus/mediaeval2016'
    all_img_emb = pkl.load(open('{}/img_emb_vgg19.pkl'.format(corpus_dir), 'rb'))
    # translated_tweets = open('{}/translated_tweet_all.txt'.format(corpus_dir), 'r').readlines()
    # translated_dict = {}
    # for line in translated_tweets:
    #    args = line.strip().split('\t')
    #    translated_dict[args[0]] = args[1]
    if phase == 'train':
        tweets = open('{}/devset/posts.txt'.format(corpus_dir), 'r').readlines()[1:]
        image_index = 3
    elif phase == 'test':
        tweets = open('{}/testset/posts_groundtruth.txt'.format(corpus_dir), 'r').readlines()[1:]
        image_index = 4
    else:
        raise ValueError('data type must be train or test!')

    for lines in tweets:
        args = lines.strip().split('\t')
        for img in args[image_index].split(','):
            if img in all_img_emb:
                image_matrix.append(all_img_emb[img])
                labels.append(label_dict[args[-1]])
                tweet_text = args[1]
                if args[0] in translated_dict:
                    tweet_text = translated_dict[args[0]]
                text_tokens, _ = tokenizer_input.encode(first=tweet_text, max_len=seq_len)
                text_tokens_matrix.append(text_tokens)
                break
    return text_tokens_matrix, image_matrix, labels


def train(dataset='weibo'):
    if dataset == 'weibo':
        pretrained_path = '/XXXX/corpus/chinese_L-12_H-768_A-12'  # for Chinese in weibo
    elif dataset == 'Twitter':
        pretrained_path = '/XXXX/corpus/uncased_L-12_H-768_A-12'  # for English in Twitter
    else:
        raise ValueError('ERROR! dataset must be weibo or Twitter!')
    config_path = '{}/bert_config.json'.format(pretrained_path)
    checkpoint_path = '{}/bert_model.ckpt'.format(pretrained_path)
    vocab_path = '{}/vocab.txt'.format(pretrained_path)
    token_dict = load_vocabulary(vocab_path)
    tokenizer = Tokenizer(token_dict)

    model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    model.summary(line_length=120)

    ############################################
    # get formated data from different dataset
    if dataset == 'weibo':
        matrix_save_dir = './weibo_dataset'
        train_t_m, train_i_m, train_l, train_el = get_weibo_matrix('train', tokenizer)
        test_t_m, test_i_m, test_l, test_el = get_weibo_matrix('test', tokenizer)
    else:
        matrix_save_dir = './Twitter_dataset'
        train_t_m, train_i_m, train_l = get_twitter_matrix('train', tokenizer)
        test_t_m, test_i_m, test_l = get_twitter_matrix('test', tokenizer)

    train_text_matrix = []
    for b in tqdm.tqdm(train_t_m):
        results = model.predict([b, np.array([0 for i in range(seq_len)])])[0]
        train_text_matrix.append(results)
    train_text_matrix = np.array(train_text_matrix)

    test_text_matrix = []
    for b in tqdm.tqdm(test_t_m):
        b = np.expand_dims(np.array(b), axis=0)
        results = model.predict([b, np.array([0 for i in range(seq_len)])])[0]
        test_text_matrix.append(results)
    test_text_matrix = np.array(test_text_matrix)

    train_t_m = np.array(train_t_m)
    train_i_m = np.array(train_i_m)
    train_l = np.array(train_l)
    test_t_m = np.array(test_t_m)
    test_i_m = np.array(test_i_m)
    test_l = np.array(test_l)
    print('4. train text:', train_t_m.shape)
    print('train text emb:', train_text_matrix.shape)
    print('train image emb:', train_i_m.shape)
    print('train label emb:', train_l.shape)
    print('5. test text:', test_t_m.shape)
    print('test text emb:', test_text_matrix.shape)
    print('test image emb:', test_i_m.shape)
    print('test labels emb:', test_l.shape)

    np.save('{}/train_text'.format(matrix_save_dir), train_t_m)
    np.save('{}/train_text_embed'.format(matrix_save_dir), train_text_matrix)
    np.save('{}/train_image_embed'.format(matrix_save_dir), train_i_m)
    np.save('{}/train_label'.format(matrix_save_dir), train_l)

    np.save('{}/test_text'.format(matrix_save_dir), test_t_m)
    np.save('{}/test_text_embed'.format(matrix_save_dir), test_text_matrix)
    np.save('{}/test_image_embed'.format(matrix_save_dir), test_i_m)
    np.save('{}/test_label'.format(matrix_save_dir), test_l)


if __name__ == '__main__':
    train('weibo')
    train('Twitter')

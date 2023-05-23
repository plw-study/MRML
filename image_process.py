import torch
import numpy
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import tqdm
import pickle as pkl


def load_vgg():
    # vgg_model = models.vgg19(pretrained=True, progress=False)
    vgg_model = models.vgg19(pretrained=False)
    pre = torch.load('/media/hibird/data/corpus/vgg19-dcbb9e9d.pth')
    vgg_model.load_state_dict(pre)
    
    new_classifier = torch.nn.Sequential(*list(vgg_model.children())[-1][:4])
    # print('new classifier structure:', new_classifier)
    # vgg_model.classifier = new_classifier
    print('===> 0. new vgg model structure:')
    print(vgg_model)

    return vgg_model


def img_prase(vgg_model, images_dir_list, save_emb_dir, output_format='pkl'):

    img2emb = {}
    for image_dir in images_dir_list:
        print('===> 2. images in: {}'.format(image_dir))
        invalid_format_count = 0
        success_processed_count = 0

        for imgs in tqdm.tqdm(os.listdir(image_dir)):
            img_name = imgs.split('.')[0]
            img_type = imgs.split('.')[1]
            if img_type in ['gif', 'txt']:
                invalid_format_count += 1
            else:
                im = Image.open('{}/{}'.format(image_dir, imgs)).convert('RGB')
                trans = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                im = trans(im)
                im.unsqueeze_(dim=0)
                out = vgg_model(im).data[0]
                out_np = list(out.numpy())
                img2emb[img_name] = out_np
                success_processed_count += 1

        print('===> 3. there are {} invalid images. Successed in processing {} images.'.format(
            invalid_format_count, success_processed_count))

    if output_format == 'pkl':
        pkl.dump(img2emb, open(save_emb_dir, 'wb'))
    elif output_format == 'txt':
        f_img = open(save_emb_dir, 'w')
        for img in img2emb:
            line = '{},{}\n'.format(img, ','.join(str(i) for i in img2emb[img]))
            f_img.write(line)
        f_img.close()
    else:
        raise ValueError('the output_format only support pkl or txt!')


def run(dataset='weibo'):

    dataset_dir = '/xxx/fake_news'
    if dataset == 'Twitter':
        main_dir = '{}/image-verification-corpus/mediaeval2016'.format(dataset_dir)
        image_files_dir_list = ['{}/devset/Mediaeval2016_DevSet_Images'.format(main_dir),
                                '{}/testset/Mediaeval2016_TestSet_Images'.format(main_dir)]
    elif dataset == 'weibo':
        main_dir = '{}/MM17-WeiboRumorSet'.format(dataset_dir)
        image_files_dir_list = ['{}/nonrumor_images'.format(main_dir),
                                '{}/rumor_images'.format(main_dir)]
    else:
        raise ValueError('ERROR! dataset must be weibo or Twitter.')

    emb_output_format = 'pkl'
    output_file = 'img_emb_vgg19_4.pkl'
    save_image_emb_file = '{}/plw_preprocess/{}'.format(main_dir, output_file)

    print('process {} dataset....'.format(dataset))
    img_prase(image_files_dir_list, save_image_emb_file, emb_output_format)


if __name__ == '__main__':
    run('weibo')
    run('Twitter')

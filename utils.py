import argparse
import os
import yaml
import shutil
import torch
import random
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import numpy as np
from dataset import FSGDataset
import itertools

import torchvision.transforms as transforms
from PIL import Image

def make_result_folders(output_directory, remove_first=True):
    if remove_first:
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    log_directory = os.path.join(output_directory, 'logs')
    if not os.path.exists(log_directory):
        print("Creating directory: {}".format(log_directory))
        os.makedirs(log_directory)
    return checkpoint_directory, image_directory, log_directory


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if ((not callable(getattr(trainer, attr))
                    and not attr.startswith("__"))
                   and ('loss' in attr
                        or 'grad' in attr
                        or 'nwd' in attr
                        or 'accuracy' in attr
                        or 'obsv' in attr))]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def write_image(iterations, dir, im_ins, im_outs, format='jpeg'):
    B, K1, C, H, W = im_ins.size()
    B, K2, C, H, W = im_outs.size()
    file_name = os.path.join(dir, '%08d' % (iterations + 1) + '.' + format)
    image_tensor = torch.cat([im_ins, im_outs], dim=1)
    image_tensor = image_tensor.view(B*(K1+K2), C, H, W)
    image_grid = vutils.make_grid(image_tensor.data, nrow=K1+K2, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1, format=format)


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def get_loader(dataset, root, mode, n_sample, num_for_seen, batch_size, num_workers, shuffle, drop_last,
               new_size=None, height=28, width=28, crop=False, center_crop=False):

    assert dataset in ['flower', 'vggface', 'animal', 'AID', 'cancer']

    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if center_crop:
        transform_list = [transforms.CenterCrop((height, width))] + \
                         transform_list if crop else transform_list
    else:
        transform_list = [transforms.RandomCrop((height, width))] + \
                         transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list \
        if new_size is not None else transform_list

    transform = transforms.Compose(transform_list)
    dataset = FSGDataset(root, mode, num_for_seen, n_sample, transform)

    loader = DataLoader(dataset,
                        batch_size,
                        shuffle=shuffle,
                        drop_last=drop_last,
                        num_workers=num_workers)
    return loader


def get_loaders(conf):
    dataset = conf['dataset']
    root = conf['data_root']
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    train_loader = get_loader(
        dataset=dataset,
        root=root,
        mode='train',
        n_sample=conf['n_sample_train'],
        num_for_seen=conf['dis']['num_classes'],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True
    )
    test_loader = get_loader(
        dataset=dataset,
        root=root,
        mode='test',
        n_sample=conf['n_sample_test'],
        num_for_seen=conf['dis']['num_classes'],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False
    )
    return train_loader, test_loader


def unloader(img):
    img = (img + 1) / 2
    tf = transforms.Compose([
        transforms.ToPILImage()
    ])
    return tf(img)


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and
                  key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views)
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def batched_scatter(input, dim, index, src):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views)
    index = index.expand(expanse)
    return torch.scatter(input, dim, index, src)


def cal_para(model):
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params: %.3fM' % (trainable_num / 1e6))


class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')



def save_results(trainer, args, config, image_directory, iterations):

    data = np.load(config['data_root'])
    dataset = args.dataset
    if dataset == 'flower':
        testdata = data[85:]
        num = 10
    elif dataset == 'animal':
        testdata = data[119:]
        num = 10
    elif dataset == 'vggface':
        testdata = data[1802:]
        num = 30

    epoch_path = os.path.join(image_directory, 'epoch_'+str(iterations))
    if not os.path.isdir(epoch_path): os.mkdir(epoch_path)

    transform_list = [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)


    n_classes, n_samples = testdata.shape[0], testdata.shape[1]

    interpolation_ind = torch.linspace(0,1,10)
    

    indxs_ = list(itertools.combinations(np.arange(n_samples), 3))

    for cls in range(n_classes):

        cls_path = os.path.join(image_directory, 'epoch_'+str(iterations), 'class_'+str(cls),)
        if not os.path.isdir(cls_path): os.mkdir(cls_path)

        base_index = np.random.choice(3)

        for i_,idx in enumerate(indxs_):

            imgs = testdata[cls, idx, :, :, :]
            imgs = torch.cat([transform(img).unsqueeze(0) for img in imgs], dim=0).unsqueeze(0).cuda()

            fake_xs_alpha = []

            base = np.array(unloader(imgs[0][base_index].cpu()))
            rf_index = [i for i in range(3) if i!=base_index]
            r1_img = np.array(unloader(imgs[0][rf_index[0]].cpu()))
            r2_img = np.array(unloader(imgs[0][rf_index[1]].cpu()))

            for i in interpolation_ind:
                

                noise = torch.FloatTensor(np.random.normal(0, 1, (1, 1024))).cuda()
                alphas = i.repeat(1).unsqueeze(1).cuda()

                fake_x = trainer.generate(imgs, alphas=alphas, base_index=base_index, noise = noise)
                output = unloader(fake_x[0].cpu())

                fake_xs_alpha.append(output)


            inps_imgs = np.concatenate([base, r1_img, r2_img], 1)


            fake_xs_alpha = np.concatenate([np.array(im) for im in fake_xs_alpha], 1)
            fake_xs_alpha = np.concatenate([inps_imgs, fake_xs_alpha], 1)

            fake_xs_alpha = Image.fromarray(fake_xs_alpha)

            saved_path = os.path.join(image_directory, 'epoch_'+str(iterations), 'class_'+str(cls), 'img_'+str(i_)+'.png')

            fake_xs_alpha.save(saved_path)

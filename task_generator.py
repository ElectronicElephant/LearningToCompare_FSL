# code is based on https://github.com/katerakelly/pytorch-maml
from os import environ
from os.path import split
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler
import csv


def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x


class SkinTask(object):
    def __init__(self, paths, num_classes, train_num, test_num):
        # cats = dict(MEL=0, NV=1, BCC=2, AK=3, BKL=4, DF=5, VASC=6,
        #                  SCC=7, UNK=8)
        # train_set = (1, 0, 2, 4, 3)
        # test_set = (5, 6, 7)

        self.num_classes = num_classes
        assert num_classes <= 3  # NO MORE THAN 3
        self.train_num = train_num
        self.test_num = test_num

        csv_path, img_root, split = paths
        assert split in ['train', 'test']

        prior_classes = (1, 0, 2, 4, 3) if split == 'train' else (5, 6, 7)
        classes = random.sample(prior_classes, self.num_classes)
        labels = dict(zip(classes, np.array(range(len(classes)))))
        samples = {i:[] for i in range(8)}

        with open(csv_path, newline='') as csvfile:
            annotations = list(csv.reader(csvfile))[1:]
            random.shuffle(annotations)
            for anno in annotations:
                assert len(anno) == 10 and anno[-1] == '0.0'
                label = np.array(anno[1:], dtype=np.float).astype(np.bool)
                assert label.sum() == 1
                cat_id = label.argmax()
                if cat_id in classes:
                    samples[cat_id].append(os.path.join(img_root, anno[0]+'.png'))
                
                lengths = (len(samples[cat]) for cat in classes)
                if min(lengths) >= train_num + test_num:
                    break

        self.train_roots, self.test_roots, self.train_labels, self.test_labels = [], [], [], []
        for cat_id in classes:
            new_cat_id = labels[cat_id]
            self.train_roots += samples[cat_id][:train_num]
            self.test_roots += samples[cat_id][train_num:train_num + test_num]

            self.train_labels += [new_cat_id] * train_num
            self.test_labels += [new_cat_id] * test_num


# class SkinTaskOld(object):
#     def __init__(self, character_folders, num_classes, train_num, test_num):
#         # Only three catagories
#         self.character_folders = [str(os.path.join(character_folders, label))
#                                   for label in ('melanoma', 'nevus', 'seborrheic_keratosis')]
#         self.num_classes = num_classes
#         assert num_classes <= 3
#         self.train_num = train_num
#         self.test_num = test_num

#         class_folders = random.sample(self.character_folders, self.num_classes)
#         labels = np.array(range(len(class_folders)))
#         labels = dict(zip(class_folders, labels))
#         samples = dict()

#         self.train_roots = []
#         self.test_roots = []
#         for c in class_folders:

#             temp = [os.path.join(c, x) for x in os.listdir(c)]
#             samples[c] = random.sample(temp, len(temp))
#             random.shuffle(samples[c])

#             self.train_roots += samples[c][:train_num]
#             self.test_roots += samples[c][train_num:train_num + test_num]

#         self.train_labels = [
#             labels[self.get_class(x)] for x in self.train_roots
#         ]
#         self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

#     def get_class(self, sample):
#         return os.path.join(*sample.split('/')[:-1])


class FewShotDataset(Dataset):
    def __init__(self,
                 task,
                 split='train',
                 transform=None,
                 target_transform=None):
        self.transform = transform  # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError(
            "This is an abstract class. Subclass this class for your particular dataset."
        )


class MiniImagenet(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(MiniImagenet, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''
    def __init__(self, num_cl, num_inst, shuffle=True):

        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batches = [[
                i + j * self.num_inst for i in torch.randperm(self.num_inst)
            ] for j in range(self.num_cl)]
        else:
            batches = [[i + j * self.num_inst for i in range(self.num_inst)]
                       for j in range(self.num_cl)]
        batches = [[batches[j][i] for j in range(self.num_cl)]
                   for i in range(self.num_inst)]

        if self.shuffle:
            random.shuffle(batches)
            for sublist in batches:
                random.shuffle(sublist)
        batches = [item for sublist in batches for item in sublist]
        return iter(batches)

    def __len__(self):
        return 1


class ClassBalancedSamplerOld(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''
    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[
                i + j * self.num_inst
                for i in torch.randperm(self.num_inst)[:self.num_per_class]
            ] for j in range(self.num_cl)]
        else:
            batch = [[
                i + j * self.num_inst
                for i in range(self.num_inst)[:self.num_per_class]
            ] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_mini_imagenet_data_loader(task,
                                  num_per_class=1,
                                  split='train',
                                  shuffle=False):
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206],
                                     std=[0.08426, 0.08426, 0.08426])

    dataset = MiniImagenet(task,
                           split=split,
                           transform=transforms.Compose(
                               [transforms.ToTensor(), normalize]))
    if split == 'train':
        sampler = ClassBalancedSamplerOld(num_per_class,
                                          task.num_classes,
                                          task.train_num,
                                          shuffle=shuffle)

    else:
        sampler = ClassBalancedSampler(task.num_classes,
                                       task.test_num,
                                       shuffle=shuffle)

    loader = DataLoader(dataset,
                        batch_size=num_per_class * task.num_classes,
                        sampler=sampler)
    return loader

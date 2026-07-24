import os
import random
from typing import List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
import torch
from torchvision import datasets, transforms

import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform


class PruneDataset:
    '''
    1. load and process calibration dataset
    2. load and process evaluation dataset
    '''
    def __init__(self, args, dataset_name, tokenizer, seq_len,
                 local_path=None, eval_mode=False):
        '''
        :param args: -->SimpleNamespace
        :param dataset_name: c4
        :param local_path: load data from local directory
        param tokenizer & seq_len: parameters to get dataloader
        '''
        self.args = args
        self.logger = args.logger
        assert dataset_name is not None
        self.dataset_name = dataset_name.lower()  # c4, wikitext2
        self.local_path = local_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eval_mode = eval_mode
        self.data_cache_dir = self.args.data_cache_dir
        if not os.path.exists(self.data_cache_dir):
            os.makedirs(self.data_cache_dir)
        # tokenize data
        self.dataloader = self.get_dataloader(self.tokenizer, self.seq_len)

    def load_data(self):
        """
        :return:eval_data if eval_mode else train_data
        """
        if self.local_path:
            # print('load data from the local directory')
            self.logger.info('load data from the local directory')
            if self.dataset_name in ["c4"]:
                if self.eval_mode:
                    return load_dataset(self.local_path, split='validation')
                return load_dataset(self.local_path, split='train')
            elif self.dataset_name in ["wikitext2", "wikitext"]:
                if self.eval_mode:
                    return load_dataset(self.local_path, split='test')
                return load_dataset(self.local_path, split='train')
            else:
                raise NotImplementedError
        self.logger.info(
            'load data from huggingface, make sure that you have opened vpn and download the dataset from huggingface')
        if self.dataset_name in ["c4"]:
            if self.eval_mode:
                return load_dataset('allenai/c4',
                                     data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                                     split='validation')
            return load_dataset('allenai/c4',
                                      data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
                                      split='train')
        elif self.dataset_name in ["wikitext2", "wikitext"]:
            if self.eval_mode:
                return load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
            return load_dataset('wikitext',
                                      'wikitext-2-raw-v1',
                                      split='train')
        else:
            raise NotImplementedError

    def get_dataloader(self, tokenizer, seq_len) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        '''
        segment and tokenize data
        '''
        model_name = tokenizer.name_or_path.split("/")[-1]
        if self.eval_mode:
            cache_dataloader = f'{self.data_cache_dir}/eval_{self.dataset_name}_{model_name}_seqlen{seq_len}.cache'
            if os.path.exists(cache_dataloader):
                self.logger.info(f"load eval processed data from {cache_dataloader}")
                return torch.load(cache_dataloader, weights_only=True)
        else:
            cache_dataloader = f'{self.data_cache_dir}/cali_{self.dataset_name}_{model_name}_seqlen{self.seq_len}_nsamples{self.args.nsamples}.cache'
            # cache_dataloader = f'{self.data_cache_dir}/cali_{self.dataset_name}_{model_name}_seqlen{self.seq_len}_nsamples{self.args.nsamples}_seed{self.args.seed}.cache'
            if os.path.exists(cache_dataloader):
                self.logger.info(f"load calibration processed data from {cache_dataloader}")
                return torch.load(cache_dataloader, weights_only=True)  # single-->calibration processed data

        data = self.load_data()
        # process
        if self.eval_mode:
            return self.process_eval_data(data, tokenizer, seq_len, cache_dataloader)
        return self.process_calibration_data(data, tokenizer, seq_len, cache_dataloader)

    def process_calibration_data(self, train_data, tokenizer, seq_len, cache_dataloader) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        random.seed(self.args.seed)
        train_loader = []
        if self.dataset_name == "c4":
            for _ in tqdm(range(self.args.nsamples), desc="Processing calibration data"):
                while True:
                    i = random.randint(0, len(train_data) - 1)  # 第i个句子

                    train_enc = tokenizer(train_data[i]['text'], return_tensors='pt')

                    if train_enc['input_ids'].shape[1] > seq_len:  # trainenc
                        break
                    else:
                        continue

                j = random.randint(0, train_enc['input_ids'].shape[1] - seq_len - 1)
                inp = train_enc.input_ids[:, j:(j + seq_len)]  # 获取词序列

                tar = inp.clone()
                tar[:, :-1] = -100
                train_loader.append((inp, tar))
            self.logger.info(f"processing {self.dataset_name} calibration data finished")
            try:
                torch.save(train_loader, cache_dataloader)
            except:
                pass
            return train_loader

        elif self.dataset_name in ["wikitext2", "wikitext"]:
            train_enc = tokenizer(" ".join(train_data['text']), return_tensors='pt')
            for i in tqdm(range(self.args.nsamples), desc="Processing calibration data"):
                j = random.randint(0, train_enc['input_ids'].shape[1] - seq_len - 1)
                inp = train_enc.input_ids[:, j:(j + seq_len)]
                tar = inp.clone()
                tar[:, :-1] = -100
                train_loader.append((inp, tar))
            self.logger.info(f"processing {self.dataset_name} calibration data finished")
            try:
                torch.save(train_loader, cache_dataloader)
            except:
                pass
            return train_loader
        else:
            raise NotImplementedError

    def process_eval_data(self, test_data, tokenizer, seq_len, cache_dataloader) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        '''
        nsamples: test_data samples // seq_len, may not be self.args.nsamples (128)
        '''
        self.logger.info("start loading test loader")
        if self.dataset_name == "c4":
            test_enc = tokenizer("\n\n".join(test_data[:1100]['text']), return_tensors='pt')
        elif self.dataset_name in ["wikitext2", "wikitext"]:
            test_enc = tokenizer("\n\n".join(test_data['text']), return_tensors='pt')
        else:
            raise NotImplementedError

        # tokenize data
        nsamples = test_enc.input_ids.numel() // seq_len
        test_loader = []
        for i in tqdm(range(nsamples), desc="Processing eval data"):
            j = i + 1
            inp = test_enc.input_ids[:, (i * seq_len):(j * seq_len)]
            tar = inp.clone()
            tar[:, :-1] = -100
            test_loader.append((inp, tar))
        self.logger.info(
            f"{self.dataset_name} testenc numel: {test_enc.input_ids.numel()}, seq_len: {seq_len}, test_loader length/nsamples: {nsamples}")
        self.logger.info(f"processing {self.dataset_name} test loader finished")
        try:
            torch.save(test_loader, cache_dataloader)
        except:
            pass
        return test_loader

    def stack_loaders(self, dataloader: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        '''
        :param dataloader: # [(loader,tar), ..., (loader,tar)]-->list len: nsamples(128)
        :return: torch.Tenser: (128,1,seqlen)
        '''
        return torch.stack([loader[0] for loader in dataloader])



class ImageNet:
    def __init__(self, args,  dataset_name="ImageNet", local_path=None, eval_mode = False):
        self.args = args
        self.dataset_name = dataset_name
        self.model_name = self.args.model_name
        self.nsamples = self.args.nsamples
        self.local_path = local_path
        self.data_cache_dir = self.args.data_cache_dir
        self.eval_mode = eval_mode
        if not os.path.exists(self.data_cache_dir):
            os.makedirs(self.data_cache_dir)
        self.logger = args.logger
        self.logger.level("INFO")

        self.dataloader = self.get_dataloader()



    def get_dataloader(self):
        if self.eval_mode:
            return self.load_eval_data()
        return self.load_calibation_data()


    def build_transform(self):
        resize_im = self.args.input_size > 32

        mean = IMAGENET_DEFAULT_MEAN  # (0.485, 0.456, 0.406)
        std = IMAGENET_DEFAULT_STD  # (0.229, 0.224, 0.225)

        if self.args.is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=self.args.input_size,
                is_training=True,
                color_jitter=self.args.color_jitter,
                auto_augment=self.args.aa,
                interpolation=self.args.train_interpolation,
                re_prob=self.args.reprob,
                re_mode=self.args.remode,
                re_count=self.args.recount,
                mean=mean,
                std=std,
            )
            if not resize_im:
                transform.transforms[0] = transforms.RandomCrop(
                    self.args.input_size, padding=4)
            return transform
        t = []
        if resize_im:
            # warping (no cropping) when evaluated at 384 or larger
            if self.args.input_size >= 384:
                t.append(
                    transforms.Resize((self.args.input_size, self.args.input_size),
                                      interpolation=transforms.InterpolationMode.BICUBIC),
                )
                self.logger.info(f"Warping {self.args.input_size} size input images...")
            else:
                if self.args.crop_pct is None:
                    crop_pct = 224 / 256
                size = int(self.args.input_size / crop_pct)  # 256
                t.append(
                    # to maintain same ratio w.r.t. 224 images
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                )
                t.append(transforms.CenterCrop(self.args.input_size))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)


    def build_dataset(self):
        transform = self.build_transform()

        self.logger.info("Transform = ")
        if isinstance(transform, tuple):
            for trans in transform:
                print(" - - - - - - - - - - ")
                for t in trans.transforms:
                    print(t)
        else:
            for t in transform.transforms:
                print(t)
        self.logger.info("---------------------------")

        if self.dataset_name == 'CIFAR':
            dataset = datasets.CIFAR100(self.local_path, train=self.args.is_train, transform=transform, download=True)
            nb_classes = 100
        elif self.dataset_name == "ImageNet":
            print("reading from datapath", self.local_path)
            root = os.path.join(self.local_path, 'train' if self.args.is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
            nb_classes = 1000  # 1000
        elif self.dataset_name == "image_folder":
            root = self.local_path if self.args.is_train else self.args.eval_data_path
            dataset = datasets.ImageFolder(root, transform=transform)
            nb_classes = self.args.nb_classes
            assert len(dataset.class_to_idx) == nb_classes
        else:
            raise NotImplementedError()
        print("Number of the class = %d" % nb_classes)

        return dataset, nb_classes

    def load_calibation_data(self, seed=0):
        assert self.args.is_train == True
        dataset, nb_classes = self.build_dataset()
        # sampler = torch.utils.data.DistributedSampler(
        #     dataset, num_replicas=1, rank=0, shuffle=True, seed=self.args.seed,
        #     )

        cache_dataloader = f'{self.data_cache_dir}/cali_{self.dataset_name}_{self.model_name}_size{self.args.input_size}_nsamples{self.nsamples}.cache'
        if os.path.exists(cache_dataloader):
            self.logger.info(f"loading calibration data from cache {cache_dataloader}")
            dataloader = torch.load(cache_dataloader)
        else:
            np.random.seed(seed)
            calibration_ids = np.random.choice(len(dataset), self.nsamples)
            dataloader = []
            for i in calibration_ids:
                dataloader.append(dataset[i][0].unsqueeze(dim=0))
            dataloader = torch.cat(dataloader, dim=0)  # [4096, 3, 224,224], cpu
            self.logger.info(f"finish loading {self.dataset_name} calibdation data")
            try:
                torch.save(dataloader, cache_dataloader)
            except:
                pass
        return dataloader

    def load_eval_data(self):
        assert self.args.is_train == False
        cache_dataloader = f'{self.data_cache_dir}/eval_{self.dataset_name}_{self.model_name}_size{self.args.input_size}.cache'
        if os.path.exists(cache_dataloader):
            self.logger.info(f"load eval data from {cache_dataloader}")
            dataloader = torch.load(cache_dataloader)
            return dataloader

        dataset, nb_classes = self.build_dataset()
        sampler= torch.utils.data.DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler,
                                                 batch_size=int(1.5 * self.args.batch_size),
                                                 num_workers=self.args.num_workers,
                                                 pin_memory=True,drop_last=False)

        self.logger.info(f"{self.dataset_name} , test_loader length: {len(dataloader)}")
        self.logger.info(f"finish loading {self.dataset_name} test loader")
        try:
            torch.save(dataloader, cache_dataloader)
        except:
            pass
        return dataloader

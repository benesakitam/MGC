# encoding: utf-8
import torch
from torch import nn
from models import resnet_mgc as resnet
import torch.distributed as dist
from exps.base_exp import BaseExp


class ResNetWithLinear(nn.Module):
    def __init__(self):
        super(ResNetWithLinear, self).__init__()

        self.encoder = resnet.resnet50(width=1, bn="vanilla")
        for p in self.encoder.parameters():
            p.requires_grad = False
        # evaluate AID
        self.classifier = nn.Sequential(nn.Linear(2048, 30), nn.BatchNorm1d(30))
        # evaluate MLRSNet
        # self.classifier = nn.Sequential(nn.Linear(2048, 46), nn.BatchNorm1d(46))
        # evaluate NWPU45
        # self.classifier = nn.Sequential(nn.Linear(2048, 45), nn.BatchNorm1d(45))
        self.criterion = nn.CrossEntropyLoss()
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def train(self, mode: bool = True):
        self.training = mode
        self.encoder.eval()
        self.classifier.train(mode)

    def forward(self, x, target=None):
        with torch.no_grad():
            feat = self.encoder(x, res5=True).detach()
        logits = self.classifier(feat)
        if self.training:
            loss = self.criterion(logits, target)
            return logits, loss
        else:
            return logits


class Exp(BaseExp):
    def __init__(self, args):
        super(Exp, self).__init__(args)

        self.basic_lr_per_img = 0.2 / 256.0
        self.max_epochs = 80
        self.scheduler = "cos"
        self.epoch_of_stage = None
        self.save_folder_prefix = "mgc"

    def get_model(self):
        if "model" not in self.__dict__:
            self.model = ResNetWithLinear()
        return self.model

    def get_data_loader(self, batch_size, is_distributed):
        if "data_loader" not in self.__dict__:

            from data.dataset_lmdb import ImageNet
            from data.transforms import typical_imagenet_transform

            train_set = ImageNet(True, typical_imagenet_transform(True))
            eval_set = ImageNet(False, typical_imagenet_transform(False))

            if is_distributed:
                batch_size = batch_size // dist.get_world_size()

            train_dataloader_kwargs = {
                "num_workers": 6,
                "pin_memory": False,
                "batch_size": batch_size,
                "shuffle": False,
                "drop_last": True,
                "sampler": torch.utils.data.distributed.DistributedSampler(train_set) if is_distributed else None,
            }
            train_loader = torch.utils.data.DataLoader(train_set, **train_dataloader_kwargs)

            eval_loader = torch.utils.data.DataLoader(
                eval_set,
                batch_size=100,
                shuffle=False,
                num_workers=2,
                pin_memory=False,
                drop_last=False,
                sampler=torch.utils.data.distributed.DistributedSampler(eval_set) if is_distributed else None,
            )
            self.data_loader = {"train": train_loader, "eval": eval_loader}
        return self.data_loader

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            lr = self.basic_lr_per_img * batch_size
            self.optimizer = torch.optim.SGD(
                self.model.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0, nesterov=False
            )
        return self.optimizer

    def get_optimizer_new(self, model, batch_size):
        # Noticing hear we only optimize student_encoder
        if "optimizer" not in self.__dict__:
            lr = self.basic_lr_per_img * batch_size
            self.optimizer = torch.optim.SGD(
                model.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0, nesterov=False
            )
        return self.optimizer


# encoding: utf-8
from exps.linear_eval_exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self, args):
        super(Exp, self).__init__(args)

        # ----------------------------- byol setting ------------------------------- #
        self.basic_lr_per_img = 0.2 / 256.0
        self.max_epochs = 80
        self.scheduler = "cos"  # "multistep"
        self.epoch_of_stage = None
        self.save_folder_prefix = "mgc_"

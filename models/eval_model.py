from .base_model import BaseModel
from . import networks
import click
from .models import models
from .models import get_LD_model
from .networks import get_random_module_eval as get_random_module

class EvalModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'EvalModel cannot be used during training time'
        parser.set_defaults(dataset_mode='test')
        
        parser.add_argument('--model_Seg', default='unet2d', type=click.Choice(models.keys()))

        return parser

    def __init__(self, opt):
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        self.loss_names = []
        self.visual_names = ['pred', 'pred_1', 'pred_2', 'pred_3', 'image_paths', 'test_B', 'test_B_mask', 'test_B_mask_ori']
        self.model_names = [opt.model_suffix]
        self.netSeg = get_LD_model(opt.model_Seg, finetune=True)
        
        self.rand_module = get_random_module(net=None, opt=opt, data_mean=((0.5,)), data_std=((0.5,)))
        self.rand_module.to(self.device)

        setattr(self, 'net_' + opt.model_suffix, self.netSeg)

    def set_input(self, input):
        
        self.image_paths = input['B_test_paths']
        self.test_B = input['B_test'].to(self.device)
        self.test_B_mask = input['B_test_mask'].to(self.device)
        self.test_B_mask_ori = input['B_test_mask_ori']

    def forward(self):
        self.pred = self.netSeg.predict(self.test_B)
        self.rand_module.randomize()
        self.pred_1 = self.netSeg.predict(self.rand_module(self.test_B))
        self.rand_module.randomize()
        self.pred_2 = self.netSeg.predict(self.rand_module(self.test_B))
        self.rand_module.randomize()
        self.pred_3 = self.netSeg.predict(self.rand_module(self.test_B))

    def optimize_parameters_Seg(self):
        pass
    


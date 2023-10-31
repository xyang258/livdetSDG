from .base_options import BaseOptions


class EvalOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--aspect_ratio', type=float, default=1.0)
        parser.add_argument('--phase', type=str, default='eval', help='train, val, test, etc')
        parser.add_argument('--eval', action='store_true')
        parser.add_argument('--train_val_test', type=str, default='test')
        parser.add_argument('--datasetname', type=str, default='liver2D')
        parser.add_argument('--datadir', type=str, default='')
        parser.add_argument('--eval_result_folder', type=str, default='experiments')
        parser.add_argument('--filtering', default=False)
        parser.add_argument('--fix_test', default=True)
        parser.add_argument('--test_all', default=True)
        parser.add_argument('--model_suffix', type=str, default='Seg')
        parser.set_defaults(model='eval')
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser

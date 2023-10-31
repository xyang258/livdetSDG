from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--display_freq', type=int, default=400)
        parser.add_argument('--display_ncols', type=int, default=4)
        parser.add_argument('--display_id', type=int, default=1)
        parser.add_argument('--display_server', type=str, default="http://localhost")
        parser.add_argument('--display_env', type=str, default='main')
        parser.add_argument('--display_port', type=int, default=8097)
        parser.add_argument('--update_html_freq', type=int, default=1000)
        parser.add_argument('--print_freq', type=int, default=100)
        parser.add_argument('--no_html', action='store_true')
        parser.add_argument('--save_latest_freq', type=int, default=5000)
        parser.add_argument('--save_epoch_freq', type=int, default=10)
        parser.add_argument('--save_by_iter', action='store_true')
        parser.add_argument('--continue_train', action='store_true')
        parser.add_argument('--epoch_count', type=int, default=1)
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--n_epochs', type=int, default=100)
        parser.add_argument('--n_epochs_decay', type=int, default=100)
        parser.add_argument('--lr_LD', type=float, default=5e-4)
        parser.add_argument('--pool_size', type=int, default=50)
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--wt_c', type=float, default=1., help='weight of consistency loss')
        parser.add_argument('--wt_d', type=float, default=1., help='weight of domain loss')
        

        self.isTrain = True
        return parser

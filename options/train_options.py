from argparse import ArgumentParser
from configs.paths_config import model_paths
from utils.dist import init_dist, get_dist_info
from options.base_options import BaseOptions, str2bool


class TrainOptions(BaseOptions):
    def initialize(self):
        super(TrainOptions, self).initialize()

        # 1. Basic training options
        self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps.')
        self.parser.add_argument('--image_interval', default=500, type=int,
                                 help='Interval for logging train images during training.')
        self.parser.add_argument('--board_interval', default=50, type=int,
                                 help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=5000, type=int, help='Validation interval.')
        self.parser.add_argument('--save_interval', default=10000, type=int, help='Model checkpoint interval.')
        self.parser.add_argument('--start_step', default=0, type=int, help='Initial step.')
        self.parser.add_argument('--seed', default=0, type=int, help="Random seed.")

        # optimizer
        self.parser.add_argument('--optimizer', default='ranger', type=str, help='Which optimizer to use.')
        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate.')
        self.parser.add_argument('--weight_decay', default=0., type=float, help='Weight decay.')
        self.parser.add_argument('--optim_beta1', default=0.95, type=float, help='beta1.')
        self.parser.add_argument('--optim_beta2', default=0.999, type=float, help='beta2.')

        # Wandb
        self.parser.add_argument('--use_wandb', default=False, type=str2bool,
                                 help='Whether to use Weights & Biases to track experiment.')
        self.parser.add_argument('--wandb_project', default='GAN_Inverter', type=str, help='continue to train.')

        # 2. Loss options
        self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor.')
        self.parser.add_argument('--id_lambda', default=0.1, type=float, help='ID loss multiplier factor.')
        self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor.')
        self.parser.add_argument('--w_norm_lambda', default=0, type=float, help='W-norm loss multiplier factor.')
        self.parser.add_argument('--moco_lambda', default=0, type=float,
                                 help='Moco-based feature similarity loss multiplier factor.')
        self.parser.add_argument('--delta_norm', type=int, default=2, help="norm type of the deltas")
        self.parser.add_argument('--delta_norm_lambda', type=float, default=0., help="lambda for delta norm loss")

        # e4e
        self.parser.add_argument('--w_discriminator_lambda', default=0., type=float, help='Dw loss multiplier.')
        self.parser.add_argument("--r1", type=float, default=10, help="Weight of the r1 regularization.")
        self.parser.add_argument("--d_reg_every", type=int, default=16,
                                 help="Interval for applying r1 regularization.")
        self.parser.add_argument('--discriminator_lr', default=2e-5, type=float, help='Dw learning rate')

        self.parser.add_argument('--use_w_pool', action='store_true',
                                 help='Whether to store a latnet codes pool for the discriminator\'s training')
        self.parser.add_argument("--w_pool_size", type=int, default=50,
                                 help="W\'s pool size, depends on --use_w_pool")

        self.parser.add_argument('--progressive_steps', nargs='+', type=int, default=None,
                                 help="The training steps of training new deltas. steps[i] starts the delta_i training")
        self.parser.add_argument('--progressive_start', type=int, default=0,
                                 help="The training step to start training the deltas, overrides progressive_steps")
        self.parser.add_argument('--progressive_step_every', type=int, default=0,
                                 help="Amount of training steps for each progressive step")

        # lsap
        self.parser.add_argument('--sncd_lambda', default=0., type=float, help='SNCD loss multiplier factor.')

        # HFGI
        self.parser.add_argument('--distortion_scale', type=float, default=0.15, help="lambda for delta norm loss")
        self.parser.add_argument('--aug_rate', type=float, default=0.8, help="lambda for delta norm loss")
        self.parser.add_argument('--res_lambda', default=0., type=float, help='L2 loss multiplier factor')

        # Distributed Training
        self.parser.add_argument('--local_rank', default=0, type=int, help='local rank for distributed training.')
        self.parser.add_argument('--gpu_num', default=1, type=int, help='num of gpu.')

    def parse(self):
        opts = super(TrainOptions, self).parse()
        opts.dist = True if opts.gpu_num != 1 else False
        if opts.dist:
            init_dist()
        opts.rank, opts.world_size = get_dist_info()
        return opts

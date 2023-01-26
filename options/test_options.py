from options.base_options import BaseOptions, str2bool


class TestOptions(BaseOptions):
	def initialize(self):
		super(TestOptions, self).initialize()

		# arguments for inference script
		self.parser.add_argument('--embed_mode', default='optim', type=str, help='Which mode to embed image. "encoder" for encoder-based and "optim" for optimization-based.')
		self.parser.add_argument('--refinement_mode', default=None, type=str, help='Refinement mode. Support PTI.')

		self.parser.add_argument('--output_resolution', default=None, nargs="+", help="Output resolution.")
		self.parser.add_argument('--output_dir', default=None, type=str, help="Output path.")
		self.parser.add_argument('--save_code', default=False, type=str2bool, help="Whether to save latent code.")
		self.parser.add_argument('--save_intermediate', default=False, type=str2bool, help="Whether to save latent code.")
		self.parser.add_argument('--code_path', default=None, type=str)
		self.parser.add_argument('--mode', default='inversion', type=str, help='which task to inference')

		# arguments for optimization-based inversion
		self.parser.add_argument('--w_plus', default=True, type=str2bool, help='w or w plus')
		self.parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
		self.parser.add_argument('--optim_step', default=1000, type=int, help='optimize iterations')
		self.parser.add_argument('--noise', default=0.05, type=float, help='strength of the noise level')
		self.parser.add_argument('--noise_ramp', default=0.75, type=float, help='duration of the noise level decay')
		self.parser.add_argument('--noise_regularize', default=1e5, type=float, help='weight of the noise regularization')
		self.parser.add_argument('--optim_l2_lambda', default=0., type=float, help='weight of mse loss')
		self.parser.add_argument('--optim_lpips_lambda', default=1., type=float, help='weight of lpips loss')

		# arguments for PTI inversion
		self.parser.add_argument('--pti_lr', default=3e-4, type=float, help='learning rate')
		self.parser.add_argument('--pti_step', default=800, type=int, help='tune epoches')
		self.parser.add_argument('--pti_l2_lambda', default=1., type=float, help='weight of mse loss in pt')
		self.parser.add_argument('--pti_lpips_lambda', default=1., type=float, help='weight of lpips loss in pt')
		self.parser.add_argument('--pti_regulizer_lambda', default=1., type=float, help='weight of locality regularization')
		self.parser.add_argument('--pti_regulizer_l2_lambda', default=0.1, type=float, help='weight of mse loss in locality regularization')
		self.parser.add_argument('--pti_regulizer_lpips_lambda', default=0.1, type=float, help='weight of lpips loss in locality regularization')
		self.parser.add_argument('--pti_use_regularization', default=False, type=str2bool, help='Whether to use locality regularization.')
		self.parser.add_argument('--pti_locality_regularization_interval', default=100, type=int, help='interval of locality regularization')
		self.parser.add_argument('--pti_regulizer_alpha', default=30, type=float, help='weight of interpolation between embedding and samples')
		self.parser.add_argument('--pti_latent_ball_num_of_samples', default=1, type=int)

		# arguments for edit script
		self.parser.add_argument('--edit_mode', type=str, default='interfacegan', help='which way to edit images')
		self.parser.add_argument('--edit_factor', type=float, default=1.0, help='the weight of interfacegan direction')
		self.parser.add_argument('--edit_path', type=str, default='', help='the path about edit')
		self.parser.add_argument('--ganspace_directions', nargs='+', type=int, default=[54, 7, 8, 20], help='about ganspace: pca_idx, start, end, strength')
		self.parser.add_argument('--indices', default='all', help='about sefa: which layers of stylegan to pca')
		self.parser.add_argument('--sem_id', type=int, default=0, help='about sefa: which semantic to be direction')
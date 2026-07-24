

import argparse
import ast

def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Prune_Args:
    def __init__(self, cfg):
        '''
        :param cfg: -->dict-->example:{model: llama-2-13b, nsamples: 128}
        '''
        self.cfg = cfg
        self.parser = argparse.ArgumentParser()
        self._gen_args()
        self.args = self.parser.parse_args()
        self.args.target_layer_names = ast.literal_eval(self.args.target_layer_names)
        self.args.tasks = ast.literal_eval(self.args.tasks)

    def _gen_args(self):
        self.parser.add_argument('--model', type=str, help='path to pre-trained llm directory, i.e. llama-2-13b', default=self.cfg["model"])
        self.parser.add_argument('--exp_name', type=str, help='experiment name', default=self.cfg["exp_name"])
        self.parser.add_argument("--cali_dataset", default=self.cfg['cali_dataset'],
                                 type=str, help="calibration dataset")
        self.parser.add_argument('--cali_data_path', type=str, help='calibration data path', default=self.cfg["cali_data_path"])
        self.parser.add_argument("--eval_dataset", default=self.cfg['eval_dataset'],
                                 type=str, help="calibration dataset")
        self.parser.add_argument('--eval_data_path', type=str, help='eval data path', default=self.cfg["eval_data_path"])
        self.parser.add_argument("--data_cache_dir", default=self.cfg['data_cache_dir'], type=str, help="processed cali/eval data cache dir")
        # self.parser.add_argument('--log_dir', type=str, help='log dir path', default=self.cfg["log_dir"])
        self.parser.add_argument('--output_dir', type=str, help='output dir path', default=self.cfg["output_dir"])
        self.parser.add_argument('--seed', type=int, default=self.cfg['seed'],
                                 help='Seed for sampling the calibration data.')
        self.parser.add_argument('--nsamples', type=int, default=self.cfg['nsamples'],
                                 help='Number of calibration samples.')  # 128 default
        self.parser.add_argument('--seqlen', type=int, default=None,
                                 help='sequence length.')  # 128 default
        self.parser.add_argument('--prune_m', type=int, default=self.cfg['prune_m'],
                                 help='parameter m of n:m pruning')  #
        self.parser.add_argument('--prune_n', type=int, default=self.cfg['prune_n'],
                                 help='parameter n of n:m pruning')  #
        self.parser.add_argument('--percdamp', type=float, default=.01,
                            help='Percent of the average Hessian diagonal to use for dampening.')

        self.parser.add_argument('--sparsity_ratio', type=float, default=self.cfg['sparsity_ratio'], help='Sparsity level')
        self.parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4", "3:4"],
                                 default=self.cfg['sparsity_type'])
        self.parser.add_argument("--prune_method",
                                 type=str,
                                 choices=["magnitude", "wanda",
                                          "sparsegpt", "ablate_mag_seq",
                                          "ablate_wanda_seq", "ablate_mag_iter",
                                          "ablate_wanda_iter", "search", "pruner-zero", "sparsellm", "admm-grad", "d2prune"],
                                 default=self.cfg['prune_method'])

        self.parser.add_argument("--cache_dir", default=self.cfg['cache_dir'], type=str, help="cache dir")

        self.parser.add_argument('--use_variant', action="store_true",
                                 help="whether to use the wanda variant described in the appendix")

        self.parser.add_argument('--save_model', type=str, default=self.cfg['save_model'],
                                 help='Path to save the pruned model.')

        self.parser.add_argument("--eval_zero_shot", action="store_true")
        self.parser.add_argument("--eval_ppl", action="store_true")
        self.parser.add_argument("--test_offload", action="store_true", help="whether to offload memory to cpu")

        self.parser.add_argument("--device", type=str, default=self.cfg['device'], help="Device to use for calibration")
        self.parser.add_argument('--kmeans', action="store_true")
        self.parser.add_argument('--s', type=float, default=self.cfg['s'], help='activation manitude')
        self.parser.add_argument('--auto_s', action="store_true", help='model seq len for auto s')
        self.parser.add_argument('--r1', type=float, default=self.cfg['r1'], help='First-order activation bias term coefficient 1, i.e., $\lambda_1$ ywx')
        self.parser.add_argument('--r2', type=float, default=self.cfg['r2'], help='Second-order activation bias term coefficient 2, i.e, $\lambda_2$ x^tww^tx')
        self.parser.add_argument('--beta', type=float, default=self.cfg['beta'], help='d2admm beta')
        self.parser.add_argument('--d2_wanda', action="store_true")
        self.parser.add_argument('--d2_sparsegpt', action="store_true")
        self.parser.add_argument('--d2_admm', action="store_true")
        self.parser.add_argument('--EA', action="store_true", help="Exponential adaptation/adjustment for activations ||X||^R or ||Y||^R, R=[0, 1/2, 1, 2]")
        self.parser.add_argument('--free', action="store_true")
        self.parser.add_argument('--distribute', action="store_true")
        # self.parser.add_argument('--blocksize', type=int, default=self.cfg['blocksize'], help='sparsegpt block')

        self.parser.add_argument('--target_layer_names', type=str,
                                 default=self.cfg['target_layer_names'],
                                 help='which layer to prune without weights update')



        self.parser.add_argument('--tasks', type=str, default=self.cfg['tasks'], help='zero-shot tasks')
        self.parser.add_argument('--dsm', type=str, default=None, choices=['owl', 'besa', 'evopress', 'als', 'dsa','mezo', 'mezo_greedy', 'owl-mezo', 'greedy', 'owl-greedy'],
                                 help="dynamic layer sparsity method")
        self.parser.add_argument('--outlier_m', type=str, default='mean', choices=['mean', 'quantile', '3sigma'],
                                 help="outlier detection method for dsm")
        self.parser.add_argument('--granularity', type=str, default='per-block', choices=['per-block', 'per-layer', 'uniform'],
                                 help="dynamic layer sparsity method")
        self.parser.add_argument(
            "--Lambda",
            default=0.08,
            type=float,
            help="Lambda for owl",
        )
        self.parser.add_argument(
            "--Hyper_m",
            type=float,
            default=3,
            help="Hyper_m for owl",
        )
        ## mezo
        self.parser.add_argument('--zo_eps', type=float, default=0.2,
                                 help='eps for mezo')
        self.parser.add_argument('--epochs', type=int, default=10,
                                 help='epochs for mezo grad caculation') # lr=0.08
        self.parser.add_argument('--batch_size', type=int, default=16,
                                 help='batch size for mezo grad caculation') # lr=0.08
        self.parser.add_argument('--lr', type=float, default=0.8,
                                 help='learning rate for mezo')
        # sparsity reg:
        self.parser.add_argument("--tb_enabled", type=bool, default=False,
                            help="Enable or disable TensorBoard logging (default: True)")

        # TensorBoard control flags
        self.parser.add_argument("--sparse_reg", type=bool, default=False,
                            help="sparsity regularization")
        # lambda_global
        self.parser.add_argument('--lambda_global', type=float, default=1.0,
                                 help='sparsity regularization factor')
        # self.parser.add_argument("--tb_logdir", type=str, default="./runs",
        #                     help="Directory to store TensorBoard logs")
        # self.parser.add_argument("--tb_runname", type=str, default="default_run",
        #                     help="Custom name for the TensorBoard run")


        # mezo use wanda forward:
        self.parser.add_argument("--use_wanda_forward", type=bool, default=False,
                            help="whether to use wanda-based forward for mezo")

        #-----------------------------add deit args-----------------------------------

        # pruning deit args
        self.parser.add_argument('--root_data_path', type=str, help='ImageNet root data dir', default=self.cfg["root_data_path"])
        self.parser.add_argument('--is_train', action="store_true", help="train or eval")
        self.parser.add_argument("--prune_granularity", type=str) # row, layer
        self.parser.add_argument('--input_size', default=224, type=int, help='image input size')
        self.parser.add_argument('--dropout', type=float, default=0, metavar='PCT',help='Drop path rate (default: 0.0)')
        self.parser.add_argument('--num_workers', default=10, type=int)
        # Augmentation parameters
        self.parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                            help='Color jitter factor (default: 0.4)')
        self.parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                            help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
        self.parser.add_argument('--smoothing', type=float, default=0.1,
                            help='Label smoothing (default: 0.1)')
        self.parser.add_argument('--train_interpolation', type=str, default='bicubic',
                            help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

        self.parser.add_argument('--crop_pct', type=float, default=None)

        # * Random Erase params
        self.parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                            help='Random erase prob (default: 0.25)')
        self.parser.add_argument('--remode', type=str, default='pixel',
                            help='Random erase mode (default: "pixel")')
        self.parser.add_argument('--recount', type=int, default=1,
                            help='Random erase count (default: 1)')
        self.parser.add_argument('--resplit', type=str2bool, default=False,
                            help='Do not random erase first (clean) augmentation split')
        self.parser.add_argument('--is_dense', action="store_true")
        self.parser.add_argument('--only_prune_moe', action="store_true")

        # moe
        self.parser.add_argument('--tune_router', action='store_true', help='Router Refinement (ROSE-Align)')
        self.parser.add_argument('--router_method', type=str, default='analytical', choices=['analytical'],
                            help='Router refinement method: analytical (ridge regression to fit dense teacher logits)')
        self.parser.add_argument('--prune_moe', action='store_true', help='rose-->moe pruning')

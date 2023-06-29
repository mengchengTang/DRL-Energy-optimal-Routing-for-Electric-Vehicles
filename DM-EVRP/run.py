import os
import argparse
from trainer import train_EVRP
from utils.functions import parse_softmax_temperature

# 主程序
parser = argparse.ArgumentParser(description='Reinforcement learning solving for electric vehicle routing problem')
parser.add_argument('--test', action='store_true', default=True)   # 测试或者训练
parser.add_argument('--CVRP_lib_test', action='store_true', default=True)   # 在标准算例上进行测试
parser.add_argument('--plot_num', default=1, type=int, help="测试画图的张数")
parser.add_argument('--seed', default=12346, type=int)
parser.add_argument("--test_seed", default=12345, type=int)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--task', default="evrp")
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--train-size', default=1280000, type=int)
parser.add_argument('--valid-size', default=10240, type=int)
parser.add_argument('--iterations', default=100, type=int)
parser.add_argument('--problem', default='evrp', help="The problem to solve, default 'evrp'")
parser.add_argument("--test_file",default=None, help="test file")
parser.add_argument("--obj",default="MD-EVRP",help="ME-EVRP,MD-EVRP")

# 神经网络的设置
parser.add_argument('--model', default='attention', help="Model, 'attention' (default) or 'pointer'")
parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
parser.add_argument('--max_grad_norm', default=2, type=float)
parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
parser.add_argument('--n_encode_layers', type=int, default=3,
                    help='Number of layers in the encoder/critic network')
parser.add_argument('--tanh_clipping', type=float, default=10.,
                    help='Clip the parameters to within +- this value using tanh. '
                         'Set to 0 to not perform any clipping.')
parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--layers', dest='num_layers', default=1, type=int)
parser.add_argument('--lr_decay', type=float, default=1, help='Learning rate decay per epoch')
parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
parser.add_argument('--actor_lr', default=1e-4, type=float)
parser.add_argument('--critic_lr', default=1e-4, type=float)


# 车辆及图的设置
parser.add_argument('--nodes', dest='num_nodes', default=100, type=int)
parser.add_argument('--Start_SOC', default=80, type=float, help='SOC, unit: kwh')
parser.add_argument('--velocity', default=50, type=float, help='unit: km/h')
parser.add_argument('--charging_num', default=8, type=int, help='number of charging_station')
parser.add_argument('--t_limit', default=10, type=float, help='tour duration time limitation, 12 hours')


# 强化学习算法设置
parser.add_argument('--baselines', default='rollout', help="Baseline to use: 'rollout', 'critic' or 'exponential'.")
parser.add_argument('--exp_beta', type=float, default=0.8,
                    help='Exponential moving average baseline decay (default 0.8)')
parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                    help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
                         'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
parser.add_argument('--bl_alpha', type=float, default=0.05,
                    help='Significance in the t-test for updating rollout baseline')


# 测试设置
parser.add_argument('--test_size', type=int, default=1,
                    help='Number of instances used for reporting test performance')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help="Batch size to use during (baseline) evaluation")
parser.add_argument('--width', type=int, default=[6400], nargs='+',
                    help='Sizes of beam to use for beam search (or number of samples for sampling), '
                         '0 to disable (default), -1 for infinite')
parser.add_argument('--decode_strategy', default="sample", type=str,
                    help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                    help="Softmax temperature (sampling or bs)")
parser.add_argument('--max_calc_batch_size', type=int, default=10000000, help='Size for subbatches')

args = parser.parse_args()
if args.bl_warmup_epochs is None:
    args.bl_warmup_epochs = 1 if args.baselines == 'rollout' else 0

print('NOTE: SETTTING CHECKPOINT:')
args.checkpoint = os.path.join('evrp', '100', "rollout", "MD-best.pt")
# args.checkpoint = os.path.join('evrp', '20', "rollout","C20_13_01_17.595265","checkpoints", "99", "epoch-99.pt")
print(args.checkpoint)

if args.test:
    if args.CVRP_lib_test:
        args.test_file = None
        args.CVRP_lib_path = "CVRPlib/P-n101-k4.txt"
    else:
        args.test_file = os.path.join('test_data', f'{args.num_nodes}', f'{args.test_size}_seed{args.test_seed}.pkl')
        print(f"  [*] the test date from: {args.test_file}")
        if not os.path.exists(args.test_file):
            args.test_file = None   # 生成数据
if args.task == 'evrp':
    train_EVRP(args)
else:
    raise ValueError('Task <%s> not understood' % args.task)

import argparse
import json
import os

SUPPORTED_SYSTEMS = [
    'carla'
]

def parse_args():

    parser = argparse.ArgumentParser(description='ROCBF training')

    # Margins for optimization constraints
    parser.add_argument('--gamma-safe', type=float, default=0.3,
                            help='Margin for safe constraint.')
    parser.add_argument('--gamma-unsafe', type=float, default=0.3,
                            help='Margin for unsafe constraint.')
    parser.add_argument('--gamma-dyn', type=float, default=0.05,
                            help='Margin for CBF constraint.')
    
    # Lagrange multipliers (fixed)
    parser.add_argument('--lambda-grad', type=float, default=0.01,
                            help='Lagrange multiplier for penalty on gradient of h(x).')
    parser.add_argument('--lambda-param', type=float, default=0.01,
                            help='Lagrange multiplier for penalty on weights of h(x).')

    # Robustness term
    parser.add_argument('--robust', action='store_true',
                            help='Adds the robustness term to the CBF inequality.')
    parser.add_argument('--delta-f', type=float, default=0.4, 
                            help='Robutness constant (Delta_F in the paper).')
    parser.add_argument('--delta-g', type=float, default=0.3, 
                            help='Robutness constant (Delta_G in the paper).')

    # Lipschitz term for output map
    parser.add_argument('--use-lip-output-term', action='store_true',
                            help='Adds the lipschitz term to the CBF inequality.')
    parser.add_argument('--lip-const-a', type=float, default=0.2,
                            help='Robustness term L_1 for output map.')
    parser.add_argument('--lip-const-b', type=float, default=0.2,
                            help='Robustness term L_2 for output map.')

    # Data augmentation (synthetic right turns)
    parser.add_argument('--data-augmentation', action='store_true',
                            help='Data augmentation with synethic right turns.')

    # Normalize states
    parser.add_argument('--normalize-state', action='store_true',
                            help='Normalize state during training.')

    # Paths
    parser.add_argument('--results-path', type=str, default='./results',
                            help='Path at which we will store the outputs.')
    parser.add_argument('--data-path', type=str, required=True, nargs='+',
                            help='Path expert data (saved as a pandas.DataFrame).')

    # System
    parser.add_argument('--system', type=str, choices=SUPPORTED_SYSTEMS, required=True,
                            help='System/dynamics to use.')

    # Training
    parser.add_argument('--net-dims', type=int, nargs='*', required=True,
                            help='Dimensions for MLP h(x).')
    parser.add_argument('--n-epochs', type=int, default=1000, 
                            help='Number of training epochs.')
    parser.add_argument('--learning-rate', type=float, default=0.005,
                            help='Learning rate for training neural network.')
    parser.add_argument('--dual-step-size', type=float, default=0.1,
                            help='Step size for updating dual variables')
    parser.add_argument('--dual-scheme', type=str, default='avg', choices=['avg', 'ae'],
                            help='Scheme for enforcing constraints.')

    # Boundary/Unsafe state sampling
    parser.add_argument('--nbr-thresh', type=float, default=0.08,
                            help='Neighbor threshold for boundary state sampling.')
    parser.add_argument('--min-n-nbrs', type=int, default=200,
                            help='Minimum number of neighbors for boundary state stampling.')

    # Additional state sampling
    parser.add_argument('--n-samp-unsafe', type=int, default=1, 
                            help='Number of unsafe states to sample for each expert unsafe state.')
    parser.add_argument('--n-samp-safe', type=int, default=1,
                            help='Number of safe states to sample for each expert safe state.')
    parser.add_argument('--n-samp-all', type=int, default=1,
                            help='Number of states to sample for each expert state.')
    

    args = parser.parse_args()

    # create path to results directory and save command line arguments
    os.makedirs(args.results_path, exist_ok=True)
    _save_args(args)

    return args

def _save_args(args):
    """Saves command line arguments to JSON file.
    
    Args:
        args: Namespace of command line arguments.
    """

    fname = os.path.join(args.results_path, 'args.json')
    with open(fname, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

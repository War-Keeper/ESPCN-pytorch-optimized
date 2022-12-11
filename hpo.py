import argparse
import warnings
warnings.filterwarnings('ignore', '.*do not.*', )
from nni.experiment import Experiment


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--max_trial_number', type=int, default=2, metavar='N',
                        help='Number of Trials (default: 20)')
    parser.add_argument('--trial_concurrency', type=int, default=2, metavar='N',
                        help='Concurrency (default: 2)')
    parser.add_argument('--port', type=int, default=8080, metavar='N',
                        help='Port Number to see results (default: 8080)')
    parser.add_argument('--visible_after_stop', type=bool, default=False, metavar='N',
                        help=' (default: False)')
    args = parser.parse_args()

    search_space = {
        'lr': {'_type': 'loguniform', '_value': [0.0001, 0.001, 0.001, 0.1, 1, 10, 100, 1000, 10000]}
    }

    experiment = Experiment('local')

    experiment.config.trial_command = 'python train_hpo.py'
    experiment.config.trial_code_directory = '.'

    experiment.config.search_space = search_space

    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

    experiment.config.max_trial_number = args.max_trial_number
    experiment.config.trial_concurrency = args.trial_concurrency

    experiment.run(args.port)

    if args.visible_after_stop:
        input()


if __name__ == '__main__':
    main()
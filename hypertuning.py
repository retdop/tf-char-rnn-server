from train import train
import hyperopt
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/python',
                       help='data directory containing scikit_cleaned.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=128,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    params = parser.parse_args()
    optimize(params)


def objective(args, parameters):
    print(args)
    params = parameters
    params.rnn_size = args['rnn_size']
    params.num_layers = args['num_layers']
    params.learning_rate = args['learning_rate']
    params.save_dir = params.save_dir + '/' + str(args['rnn_size']) + '_' + str(args['num_layers']) + '_' + str(args['learning_rate'])
    loss = train(params)

    return loss

from hyperopt import fmin, tpe, hp
def optimize(params):

    space = {
        'rnn_size': hp.choice('rnn_size', [64, 128, 256, 512]),
        'num_layers': hp.choice('num_layers', [2, 3]),
        'learning_rate': hp.choice('learning_rate', [0.001, 0.002, 0.005])
    }

    best_model = fmin(lambda x : objective(x, params), space, algo=tpe.suggest, max_evals=5)

    print(best_model)
    print(hyperopt.space_eval(space, best_model))


if __name__ == '__main__':
    main()

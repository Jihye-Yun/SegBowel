import sys
import argparse

from model import SegBowel


def get_args():
    parser = argparse.ArgumentParser(description='<<< SegBowel >>>')
    
    parser.add_argument('--gpu_id', type=int, default=0, help='Device # for SegBowel')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # train configuration
    parser.add_argument('--data_train', nargs='+', help='Path where the training data is stored (It can be a list of paths)')
    parser.add_argument('--data_valid', nargs='+', help='Path where the validation data is stored (It can be a list of paths)')
    parser.add_argument('--epochs', type=int, default=100, help='# of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--decay', type=float, default=1e-6, help='Decay of learning process')
    parser.add_argument('--logdir', type=str, default=None, help='Path for logging the training process')    

    # test configuration
    parser.add_argument('--weights', type=str, default=None, help='Path where the trained weight is stored')
    parser.add_argument('--data_test', nargs='+', help='Path where the test data is stored (It can be a list of paths)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.mode == 'train':
        if not args.data_train:
            print ("ERROR: need a path for input data (--data_train)")
            sys.exit()
        if not args.data_valid:
            print ("ERROR: need a path for validation data (--data_valid)")
            sys.exit()
        if not args.logdir:
            print ("ERROR: need a path for logging (--logdir)")
            sys.exit()

    if args.mode == 'test':
        if not args.weights:
            print ("ERROR: need a trained weight (--weights)")
            sys.exit()
        if not args.data_test:
            print ("ERROR: need a path for test data (--data_test)")
            sys.exit()

    #########################################################################################################
    
    model = SegBowel(args)

    if args.mode == 'train':
        model.train()
        
    if args.mode == 'test':
        model.test()
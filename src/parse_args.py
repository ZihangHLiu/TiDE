import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Word2Vec')

    # basic arguments
    parser.add_argument('--name', type=str, default='word2vec', help='name of the experiment')
    parser.add_argument('--load', type=str, default='False', help='if load training data from local file')
    parser.add_argument('--print_tofile', type=str, default='True', help='if print log content to file')
    parser.add_argument('--ckpt_path', type=str, default='./data', help='directory to store and load the checkpoints')
    parser.add_argument('--datadir', type=str, default='./data', help='directory to training/testing data')

    # preprocessing related
    parser.add_argument('--window_size', type=int, default=8, help='window size for skipgram')
    parser.add_argument('--unk', type=str, default='<UNK>', help='token for unknown words')
    parser.add_argument('--max_vocab', type=int, default=50000, help='max vocabulary size')
    parser.add_argument('--filename', type=str, default='wordsim353_agreed.txt', help='corpus file name')

    # training related
    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--n_negs', type=int, default=20, help="number of negative samples")
    parser.add_argument('--epoch', type=int, default=100, help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=4096, help="mini-batch size")
    parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
    parser.add_argument('--cuda', type=str, default='False', help="use CUDA")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="weight decay")
    parser.add_argument('--betas', nargs=2, type=float, default=[0.9, 0.999], help="beta parameters for Adam optimizer")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon parameter for Adam optimizer")

    args = parser.parse_args()
    return args
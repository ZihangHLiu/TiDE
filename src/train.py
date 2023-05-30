from parse_args import *
from models import *
from test import *
import os
import sys
import numpy as np
import pickle
import random
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch import LongTensor as LT
from torch import FloatTensor as FT
from torch.utils.data import DataLoader
from dataloader import *
import json


def save_args_to_file(args, output_file_path):
    with open(output_file_path, "w") as output_file:
        json.dump(vars(args), output_file, indent=4)

if __name__ == '__main__':

    ############################## 1. Parse arguments ################################
    print('parsing arguments...')
    args = parse_args()

    # create the checkpoint directory if it does not exist
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)


    if args.print_tofile == 'True':
        # Open files for stdout and stderr redirection
        stdout_file = open(os.path.join(args.ckpt_path, 'stdout.log'), 'w')
        stderr_file = open(os.path.join(args.ckpt_path, 'stderr.log'), 'w')
        # Redirect stdout and stderr to the files
        sys.stdout = stdout_file
        sys.stderr = stderr_file

    save_args_to_file(args, os.path.join(args.ckpt_path, 'args.json'))

    # print args
    print(args)

    ############################## 2. preprocess and loading the training data ################################
    print('loading the training data...')
    ETdir = os.path.join(args.datadir, 'ETDataset', 'ETT-small')
    train_set = Dataset_ETT_hour(root_path=ETdir, flag='train', data_path=args.dataset + '.csv', features='MS')
    val_set = Dataset_ETT_hour(root_path=ETdir, flag='val', data_path=args.dataset + '.csv', features='MS')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=6, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=6, drop_last=True)
    print('Loading finished. Train data: {}, val data: {}'.format(len(train_set), len(val_set)))

    # get sizes
    sizes = {}
    for _, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(train_loader):
        sizes['lookback'] = (seq_y.shape[1], seq_y.shape[2])
        sizes['attr'] = (seq_x.shape[1], seq_x.shape[2])
        sizes['dynCov'] = (seq_y_mark.shape[1], seq_y_mark.shape[2])
        break

    ############################## 3. build the model ################################
    

    model = TiDEModel(sizes, args)
    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    if args.cuda == 'True':
        model.cuda()
    optim = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss(reduction='mean')

    training_stats = {
        'epoch': [],
        'train_mse': [],
        'train_mae': [],
        'val_mse': [],
        'val_mae': []
    }

    # flush the output
    print('model built.')
    # print model
    print(model)
    sys.stdout.flush()

    ############################## 4. train the model ################################
    for epoch in range(1, args.epoch + 1):
        print('batch num: {}'.format(len(train_loader)))
        train_mse_loss = 0.
        train_mae_loss = 0.
        test_mse_loss = 0.
        test_mae_loss = 0.
        best_loss = 9999
        step = 0
        print('Starting epoch: {}'.format(epoch))
        for _, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(train_loader):
            step += 1
            if epoch == 1 and step <= 1:
                print('seq_x: {}, seq_y: {}, seq_x_mark: {}, seq_y_mark: {}'.format(seq_x.shape, seq_y.shape, seq_x_mark.shape, seq_y_mark.shape))
                sys.stdout.flush()
            optim.zero_grad()
            if args.cuda == 'True':
                seq_x = seq_x.float().cuda()
                seq_y = seq_y.float().cuda()
                # turn marks into float type
                seq_x_mark = seq_x_mark.float().cuda()
                seq_y_mark = seq_y_mark.float().cuda()

            pred, ans = model(seq_x, seq_y, seq_x_mark, seq_y_mark)
            # use MSE loss
            loss = criterion(pred, ans)
            # calculate the MAE loss
            mae_loss = torch.mean(torch.abs(pred - ans))
            train_mse_loss += loss.item()
            train_mae_loss += mae_loss.item()
            loss.backward()
            optim.step()

            # print the training stats
            if step % 100 == 0:
                if train_mse_loss < best_loss:
                    best_loss = train_mse_loss
                    torch.save(model.state_dict(), os.path.join(args.ckpt_path, '{}.pt'.format(args.name)))
                    torch.save(optim.state_dict(), os.path.join(args.ckpt_path, '{}.optim.pt'.format(args.name)))
                    print('Best model saved.')

                # test the model on val set
                val_mse_loss, val_mae_loss = test(args, model, val_loader, criterion)
                test_mse_loss += val_mse_loss / 10
                test_mae_loss += val_mae_loss / 10

                print('Epoch: {}, Step: {}, train_mse: {}, train_mae: {}, val_mse: {}, val_mae: {}'.format(epoch, step, loss.item(), mae_loss.item(), val_mse_loss, val_mae_loss))
                sys.stdout.flush()


        train_mse_loss /= step
        train_mae_loss /= step
        print('Finished Epoch: {}, train_mse_loss: {}, train_mae_loss: {}, test_mse_loss: {}, test_mae_loss: {}'.format(epoch, train_mse_loss, train_mae_loss, test_mse_loss, test_mae_loss))

        # update training stats
        training_stats['epoch'].append(epoch)
        training_stats['train_mse'].append(train_mse_loss)
        training_stats['train_mae'].append(train_mae_loss)
        training_stats['val_mse'].append(test_mse_loss)
        training_stats['val_mae'].append(test_mae_loss)


        # flush the output
        sys.stdout.flush()


    np.save(os.path.join(args.ckpt_path, "training_stats.npy"), training_stats)

    # idx2vec = sgns.get_embeddings('in')
    # pickle.dump(idx2vec, open(os.path.join(args.datadir, 'idx2vec.dat'), 'wb'))
    # torch.save(sgns.state_dict(), os.path.join(args.ckpt_path, '{}.pt'.format(args.name)))
    # torch.save(optim.state_dict(), os.path.join(args.ckpt_path, '{}.optim.pt'.format(args.name)))
    
    if args.print_tofile == 'True':
        # Close the files to flush the output
        stdout_file.close()
        stderr_file.close()
import numpy as np
import re
import operator
from matplotlib import pylab as plt
from collections import OrderedDict

def plot(losses):
    plt.style.use('ggplot')
    fig, (ax_hs) = plt.subplots(len(losses), sharex=True, sharey=False)
    axes = {}
    count = 0
    for key in losses:
        axes[key] = ax_hs[count]
        axes[key].set_ylabel(key)
        count += 1
    ax_hs[-1].set_xlabel('iterations')

    for key in losses:
        iterations = range(len(losses[key]))
        axes[key].plot(iterations, losses[key])
        max_y = max(losses[key])
        #import pdb; pdb.set_trace()
        axes[key].set_ylim([0, max_y])

    plt.show()

def plot_train_dev(train_losses, dev_losses, save_to=''):

    # sort trained dictioary
    train_losses = OrderedDict(sorted(train_losses.items(), key=lambda t: t[0]))
    dev_losses = OrderedDict(sorted(dev_losses.items(), key=lambda t: t[0]))

    plt.style.use('ggplot')
    keys= []
    for key in train_losses:
        if key in keys: continue
        keys.append(key)
    for key in dev_losses:
        if key in keys: continue
        keys.append(key)

    fig, (ax_hs) = plt.subplots(len(keys), sharex=True, sharey=False, figsize=(10, 10))
    axes = {}
    count = 0
    for key in keys:
        axes[key] = ax_hs[count]
        axes[key].set_ylabel(key, fontsize=10)
        # axes[key].set_ylim([0, min(max_error, max(train_losses[key]))])

        count += 1
    ax_hs[-1].set_xlabel('iterations')

    l1, l2 = False, False
    for key in keys:
        # training iterations
        if key in train_losses:
            train_iter = range(len(train_losses[key]))
            if l1 == False:
                axes[key].semilogy(train_iter, train_losses[key], 'b', label='train_loss')
                axes[key].legend(loc='upper right')
                l1 = True
            else :
                axes[key].semilogy(train_iter, train_losses[key], 'b')

            axes[key].set_yscale('log')
            for x in train_iter:
                if (x+1)%5000 == 0:
                    y = round(train_losses[key][x], 2)
                    axes[key].text(x, y, str(y), color='blue', fontsize=8)
            # plot the last one
            y = round(train_losses[key][-1], 2)
            axes[key].text(train_losses[key][-1], y, str(y), color='blue', fontsize=8)

        if key in dev_losses:
            dev_iter = range(len(dev_losses[key]))
            if l2 == False:
                axes[key].semilogy([x*5000 for x in dev_iter] , dev_losses[key], 'g', label='dev_loss')
                axes[key].legend(loc='upper right')
                l2 = True;
            else :
                axes[key].semilogy([x*5000 for x in dev_iter] , dev_losses[key], 'g')

            axes[key].set_yscale('log')
            # write the error number on the graph
            for x in dev_iter:
                y = round(dev_losses[key][x], 2)
                axes[key].text(x*5000, y, str(y),  color='green', fontsize=10)

    if save_to != '':
        fig.savefig(save_to)


def parse_log(log_file, t_dict, TRAIN=True):
    regex_iteration = re.compile('Iteration (\d+)')
    regex_loss = {}
    loss = {}
    for i in t_dict:
        if TRAIN:
            regex_loss[i] = (r"Train net output {:s}: {:s} = ([\.\deE+-]+)".format(t_dict[i], i))
        else:
            regex_loss[i] = (r"Test net output {:s}: {:s} = ([\.\deE+-]+)".format(t_dict[i], i))

        loss[i] = []

    iteration = -1

    with open(log_file, 'r') as f:
        log = f.read()

        for idx in regex_loss:
            losses = re.findall(regex_loss[idx], log)
            losses = [float(x) for x in losses]
            loss[idx] = losses

    return loss

def smoothing_within_epoch(losses, epoch_iter = 2000):
    sum_error = 0
    total_num = len(losses)
    smoothed_loss = []
    for idx in range(total_num):
        idx_in_epoch = idx % epoch_iter
        if idx_in_epoch == 0:
            sum_error = 0

        sum_error += losses[idx]
        smoothed_loss.append(sum_error / (idx_in_epoch+1))

    return smoothed_loss

def read_loss_keys(loss_keys):
    '''
    Plot the training curve of
    '''
    train_dicts = []
    with open(loss_keys) as f:
        lines = f.readlines()
        key_indices = [int(x) for x in lines[0].split(' ')]
        loss_keys = [x.rstrip('\n') for x in lines[1:]]

        key_indices.append(len(loss_keys))

        for i in range(len(key_indices)-1):

            train_dict = {}

            start = key_indices[i]
            end = key_indices[i+1]
            for k in range(start, end):
                train_dict[loss_keys[k]] = '#{:d}'.format(k)

            train_dicts.append(train_dict)

    return train_dicts

if __name__ == '__main__':
    default_log = 'vgg_MC.log'

    train_dicts = read_loss_keys('loss.keys')

    for train_dict in train_dicts:
        train_losses = parse_log(default_log, train_dict, True)
        test_losses = parse_log(default_log, train_dict, False)

        for idx in train_losses:
            train_losses[idx] = smoothing_within_epoch(train_losses[idx])

        for idx in test_losses:
            test_losses[idx] = smoothing_within_epoch(test_losses[idx])

        plot_train_dev(train_losses, test_losses, 'nopose_full_resolution')

    plt.show()

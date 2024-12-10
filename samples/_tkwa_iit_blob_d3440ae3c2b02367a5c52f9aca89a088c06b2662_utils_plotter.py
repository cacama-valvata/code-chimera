import matplotlib.pyplot as plt
import numpy as np
import wandb
from PIL import Image

def plot_probe_stats(correctness_stats_per_layer, 
                     leaky_stats_per_layer, reduction='max', prefix='', use_wandb=False):
    # make arrays
    hookpoints = list(correctness_stats_per_layer.keys())
    get_hl_nodes = lambda stats: list(stats[list(stats.keys())[0]]['probes'].keys())
    hl_nodes = get_hl_nodes(correctness_stats_per_layer)
    # correctness_loss = np.zeros((len(hookpoints), len(hl_nodes)))
    correctness_acc = np.zeros((len(hookpoints), len(hl_nodes)))
    for i, hookpoint in enumerate(hookpoints):
        for j, hl_node in enumerate(hl_nodes):
            correctness_acc[i, j] = correctness_stats_per_layer[hookpoint]['test accuracy'][hl_node]

    # leaky_loss = np.zeros((len(hookpoints), len(hl_nodes)))
    leaky_acc = np.zeros((len(hookpoints), len(hl_nodes)))
    leaky_hl_nodes = get_hl_nodes(leaky_stats_per_layer)
    leaky_accs_all = np.zeros((len(hookpoints), len(leaky_hl_nodes)))
    get_idx = lambda name: hl_nodes.index('hook_{}'.format(name))
    reduction = np.mean if reduction == 'mean' else np.max if reduction == 'max' else np.median if reduction == 'median' else None
    assert reduction is not None, f"reduction must be one of 'mean', 'max', or 'median', got {reduction}"
    for i, hookpoint in enumerate(hookpoints):
        accs = np.zeros((len(hl_nodes), len(hl_nodes)))
        for j, hl_node in enumerate(leaky_hl_nodes):
            acc = leaky_stats_per_layer[hookpoint]['test accuracy'][hl_node]
            leaked_from = hl_node.split('_')[1]
            leaked_to = hl_node.split('_')[-1]
            # print(f"leaked_from: {leaked_from}; leaked_to: {leaked_to}")
            accs[get_idx(leaked_from), get_idx(leaked_to)] = acc
            leaky_accs_all[i, j] = acc
        # print(f"accs: {accs}")
        accs_reduced = reduction(accs, axis=0) # rows = leaked_from; cols = leaked_to
        for j, _ in enumerate(hl_nodes):
            leaky_acc[i, j] = accs_reduced[j]
        # print(f"leaky_acc: {leaky_acc}")

    # plot
    # TODO: maybe add this: https://stackoverflow.com/questions/9707676/defining-a-discrete-colormap-for-imshow
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # print(f"correctness_acc: {correctness_acc}")
    # print(f"leaky_acc: {leaky_acc}")

    np.save('plots/bin/correctness_acc.npy', correctness_acc)
    np.save('plots/bin/leaky_acc.npy', leaky_acc)
    np.save('plots/bin/leaky_accs_all.npy', leaky_accs_all)
    ax[0].imshow(correctness_acc, cmap='viridis', vmin=0, vmax=1)
    ax[0].set_title('Correctness Accuracy')
    im = ax[1].imshow(leaky_acc, cmap='viridis', vmin=0, vmax=1)
    for i in range(2):
        ax[i].set_xlabel('HL Node')
        ax[i].set_xticks(np.arange(len(hl_nodes)))
        ax[i].set_yticks(np.arange(len(hookpoints)))
        ax[i].set_xticklabels(hl_nodes)
        ax[i].set_ylabel('Hookpoint')
        plt.setp(ax[i].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    hook_point_labels = [i.replace('mod.', '').replace('.hook_point', '').replace('.', ' ') for i in hookpoints]
    ax[1].set_title('Leaky Accuracy')
    ax[0].set_yticklabels(hook_point_labels)
    ax[1].set_yticklabels(hook_point_labels)
    fig.colorbar(im, ax=ax.ravel().tolist())
    plt.savefig(f'plots/{prefix}_probe_stats.png')

    fig = plt.figure()
    im = plt.imshow(leaky_accs_all, cmap='viridis')
    plt.colorbar(im)
    plt.xlabel('HL Node')
    plt.ylabel('Hookpoint')
    plt.title('Leaky Accuracy')
    x_tick_string = """{} -> {}"""
    plt.xticks(np.arange(len(leaky_hl_nodes)), 
               [x_tick_string.format(i.split('_')[1], i.split('_')[-1]) for i in leaky_hl_nodes], 
               rotation=90)
    plt.yticks(np.arange(len(hookpoints)), hook_point_labels)
    plt.tight_layout()
    plt.savefig(f'plots/{prefix}_leaky_accs_all.png')

    if use_wandb:
        wandb.log({'probe stats': wandb.Image(f'plots/{prefix}_probe_stats.png')})
        wandb.log({'leaky_accs_all': wandb.Image(f'plots/{prefix}_leaky_accs_all.png')})

    print("Plotted probe stats. Find them in plots folder.")
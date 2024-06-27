import torch
import matplotlib.pyplot as plt
def plot_norms(classifier,config, y_range=None):
    # per-class weight norms vs. class cardinality
    W = classifier.module.fc.weight.cpu()
    tmp = torch.linalg.norm(W, ord=2, dim=1).detach().numpy()
    
    if y_range==None:
        max_val, mid_val, min_val = tmp.max(), tmp.mean(), tmp.min()
        c = min(1/mid_val, mid_val)
        y_range = [min_val-c, max_val+c]
    if config.dataset == 'cifar10':
        class_num = 10
    elif config.dataset == 'cifar100':
        class_num = 100
    elif config.dataset == 'imagenet':
        class_num = 1000
    fig = plt.figure(figsize=(15,3), dpi=64, facecolor='w', edgecolor='k')
    plt.xticks(list(range(class_num)), list(range(class_num)), rotation=90, fontsize=8);  # Set text labels.
    ax1 = fig.add_subplot(111)

    ax1.set_ylabel('norm', fontsize=16)
    ax1.set_ylim(y_range)
    
    plt.plot(tmp, linewidth=2)
    plt.title('norms of per-class weights from the learned classifier vs. class cardinality', fontsize=20)
    plt.savefig("./figure/"+str(config.dataset)+"_"+str(config.imb_factor)+"_norm.png")
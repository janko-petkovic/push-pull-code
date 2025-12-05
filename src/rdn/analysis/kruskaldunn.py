import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors

from scikit_posthocs import posthoc_dunn



def kruskal_dunn(data: list,
                 data_labels: list,
                 plot: bool = False,
                 figsize: tuple = (8,4),
                 dpi: float = 100,
                 xlabel: str = 'groups',
                 ylabel: str = 'value [A.U.]'):
    '''
    Kruskal-Wallis + dunn posthoc if KW is significant.
    Optionally plot a nice chart

    Parameters
    ----------
    data : list or bidimensional array
        A list of data arrays (first dimension is always considered to be
        the group dimension. Second dimension is the data dimension)

    data_labels : list of str
        List containing the group names

    plot : bool
        Do you want to make a nice plot to compare the data groups

    Returns
    -------
    KW or Dunn or I don't know yet

    Note
    ----
    Technically theres no need to do the Dunn if KW is non significant
    but whatever the plots are nice.
    '''

    kw = stats.kruskal(*data)
    dunn = posthoc_dunn(data)

    # If you want only a lower triangular plot
    # mask = np.tri(dunn.to_numpy().shape[0], k=-1)
    # plot_dunn = np.ma.array(dunn.to_numpy(), mask=mask).T
    plot_dunn = dunn

    fig, axs = plt.subplots(1,2, figsize=figsize, dpi=dpi)
    fig.tight_layout(pad=5, rect=[0,0,1,1])

    # Plot the bloxplot (what abuot a violinplot?)
    ax = axs[0]
    # sns.swarmplot(data, size=1, ax=ax, color='blue')
    ax.boxplot(data, showfliers=True, whis=(5,95))
    ax.set_title(f'KW p-value={kw.pvalue:.3}')
    ax.set_xlabel(xlabel)
    # ax.set_xticks([i+1 for i in range(len(data_labels))])
    ax.set_xticklabels(data_labels, rotation=90)
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

    # Show the dunn results
    ax=axs[1]
    pc = ax.imshow(plot_dunn, cmap='coolwarm_r', norm=colors.CenteredNorm(0.05, 0.05))
    fig.colorbar(pc, ax=ax, fraction=0.04, pad=0.1)
    ax.set_title('Pairwise Dunn p-values')
    ax.set_xticks([i for i in range(len(data_labels))])
    ax.set_yticks([i for i in range(len(data_labels))])
    ax.set_xticks([i-0.5 for i in range(len(data_labels))], minor=True)
    ax.set_yticks([i-0.5 for i in range(len(data_labels))], minor=True)
    ax.set_xticklabels(data_labels, rotation=90)
    ax.set_yticklabels(data_labels)
    # ax.grid(which='minor', color='black', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)

    # plt.show()

    return kw, dunn, axs


if __name__ == '__main__':
    import numpy as np
    
    data = [np.random.rand(10)*10+i for i in range(10)]
    data_labels= ([f'{i}' for i in range(10)])
    print(data_labels)
    kruskal_dunn(data, data_labels)
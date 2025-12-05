from ctypes import alignment
from os.path import join
import torch
import matplotlib.pyplot as plt
import numpy as np


from pypesto import visualize


def plot_data_2d(data):
    '''
    2D data heatmap
    '''
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap=plt.cm.coolwarm)
    ax.set_title('Spine luminosity')
    ax.set_xlabel('x')
    ax.set_xticklabels('')
    ax.set_ylabel('t')
    ax.set_yticklabels('')
    plt.colorbar(im, shrink=0.45)
    return ax



def plot_model_predictions(nss, model, result, p_optim=None, fig=None):
    '''
    For now i am using this method to compute the chi squared as 
    well!! It should not be done here but might as well for
    '''

    # Load t, x
    t = torch.from_numpy(np.loadtxt(f'binned-data/{nss}Spine_t.txt')).to(torch.float64)
    x = torch.from_numpy(np.loadtxt(f'binned-data/{nss}Spine_x.txt')).to(torch.float64)

    # Create the plotting t tensor; remember that we don't really
    # know what is happening during the stimulation (from 0 to 1 included)
    t_fine = torch.linspace(t.min(),40,100).to(torch.float64)

    # Create the meshes
    t_mesh, x_mesh = torch.meshgrid(t,x,indexing="ij")
    t_fine_mesh, x_fine_mesh = torch.meshgrid(t_fine,x,indexing="ij")

    # Load the target and errors
    data_name = f'{nss}Spine_data'
    data = torch.from_numpy(np.loadtxt(f'binned-data/{data_name}.txt')).to(torch.float64)
    data_err = torch.from_numpy(np.loadtxt(f'binned-data/{data_name}_errs.txt')).to(torch.float64)

    # Last thing before prediction
    nss_weights = model.generate_nss_weights(nss, t, x)
    nss_weights_fine = model.generate_nss_weights(nss, t_fine, x)

    # Validation
    if p_optim is not None:
        prediction = model.validation_forward(nss_weights, t_mesh, x_mesh,
            torch.tensor(p_optim),   
            torch.tensor(result.optimize_result.as_list()[0].x)
        )

        plot_prediction = model.validation_forward(nss_weights_fine, t_fine_mesh, x_fine_mesh, 
            torch.tensor(p_optim),   
            torch.tensor(result.optimize_result.as_list()[0].x)
        )

    # full optimization
    else:
        prediction = model(nss_weights, t_mesh, x_mesh, torch.tensor(result.optimize_result.as_list()[0].x))
        plot_prediction = model(nss_weights_fine, t_fine_mesh, x_fine_mesh, torch.tensor(result.optimize_result.as_list()[0].x))


    if not fig: fig = plt.figure(figsize=(10,7))
    axs = fig.subplots(4,5)

    for idx, (data_srs, data_srs_err, pred_srs, pos) \
        in enumerate(zip(data.T, data_err.T, plot_prediction.T, x)):

        ax = axs.flatten()[idx]
        ax.errorbar(t, data_srs, yerr = data_srs_err, fmt='.', color='gray')

        # Plot the prediction remembering the gape

        pre_idx = torch.where(t_fine<0)
        ax.plot(t_fine[pre_idx],pred_srs[pre_idx], color='black')

        post_idx = torch.where(t_fine>1)
        ax.plot(t_fine[post_idx], pred_srs[post_idx], color='tab:orange')

        ax.set_title(f'x={pos:.3}')
        ax.set_ylim(5000,20000)

    # plt.subplots_adjust(wspace=3, hspace=3)

    # ndof = len(data.flatten()) - len(p_optim)

    # chi2 = (((data - prediction.numpy())/data_err)**2).sum()
    # print(chi2/ndof)



def plot_optimization_round(nss, model, result):
    '''
    This is good.
    To be used for each optimization cycle, shows
    the cascade plot, the fitted parameters of the best runs
    and the best prediction
    '''

    fig = plt.figure(figsize=(10,10))
    subfigs = fig.subfigures(2,1, wspace=0.07, height_ratios=[0.5,1])

    axs = subfigs[0].subplots(1,2)

    visualize.waterfall(result,n_starts_to_zoom=15, ax=axs[0])
    visualize.parameters(result,start_indices=range(4), ax=axs[1])

    plot_model_predictions(nss, model, result, fig=subfigs[1])

    plt.subplots_adjust(wspace=1, hspace=1)
    plt.show()



def plot_validation_round(nss, model, p_optim, result):
    '''
    This is good.
    To be used for each optimization cycle, shows
    the cascade plot, the fitted parameters of the best runs
    and the best prediction
    '''

    fig = plt.figure(figsize=(10,10))
    subfigs = fig.subfigures(2,1, wspace=0.07, height_ratios=[0.5,1])

    axs = subfigs[0].subplots(1,2)

    visualize.waterfall(result,n_starts_to_zoom=15, ax=axs[0])
    visualize.parameters(result,start_indices=range(4), ax=axs[1])

    plot_model_predictions(nss, model, result, p_optim, fig=subfigs[1])

    plt.subplots_adjust(wspace=1, hspace=1)
    plt.show()




def plot_multi_optimization(nsss, multi_model, result, show_fits=False,
                            start_indices = None):
    fig, axs = plt.subplots(1,2, figsize=(10,10))

    visualize.waterfall(result,n_starts_to_zoom=15, ax=axs[0])
    visualize.parameters(result,start_indices=start_indices, ax=axs[1])


    if show_fits:
        # Import the data
        # Make a list of mesh pairs to use with the submodels
        tx_pairs = []
        tx_pairs_fine = []
        mesh_pairs = []
        mesh_pairs_fine = []
        datas = []
        data_errs = []

        for nss in nsss:
            t = torch.from_numpy(np.loadtxt(f'binned-data/{nss}Spine_t.txt')).to(torch.float64)
            x = torch.from_numpy(np.loadtxt(f'binned-data/{nss}Spine_x.txt')).to(torch.float64)
            t_fine = torch.linspace(t.min(), 40,100).to(torch.float64)

            tx_pairs_fine.append((t_fine, x))
            tx_pairs.append((t,x))

            mesh_pairs.append(torch.meshgrid(t,x,indexing="ij"))
            mesh_pairs_fine.append(torch.meshgrid(t_fine, x, indexing='ij'))

            data_name = f'{nss}Spine_data'
            datas.append(torch.from_numpy(np.loadtxt(f'binned-data/{data_name}.txt')).to(torch.float64))
            data_errs.append(torch.from_numpy(np.loadtxt(f'binned-data/{data_name}_errs.txt')).to(torch.float64))

        nsss_weights_fine = multi_model.generate_nsss_weights(nsss, tx_pairs_fine)

        preds_fine = multi_model(
            False, nsss_weights_fine, mesh_pairs_fine, 
            torch.from_numpy(result.optimize_result.as_list()[1].x))

        for (t, x), (t_fine, _), pred_fine, data, data_err, nss in \
            zip(tx_pairs, tx_pairs_fine, preds_fine, datas, data_errs, nsss):

            fig, axs = plt.subplots(7,4, figsize=(10,7))

            fig.suptitle(f'Number of stimulations = {nss}')

            for idx, (pred_srs, data_srs, data_srs_err, pos) \
                in enumerate(zip(pred_fine.T, data.T, data_err.T, x)):

                ax = axs.flatten()[idx]
                ax.errorbar(t, data_srs, yerr = data_srs_err, fmt='.',
                            color='gray',
                            elinewidth=1)

                # Plot the prediction remembering the gape

                pre_idx = torch.where(t_fine<0)
                ax.plot(t_fine[pre_idx],pred_srs[pre_idx], color='#444444',
                        label='pre')

                post_idx = torch.where(t_fine>=2)
                ax.plot(t_fine[post_idx], pred_srs[post_idx], color='tab:orange',
                        label='post')

                ax.set_title(r'$\mathbf{x_i}$ = ' + f'{pos:.3}', fontsize=10, pad=0.2,
                            loc='right', weight='bold')

                ax.legend(frameon=False, ncol=2, fontsize=8)

                ax.set_xlabel(r'$t$', fontsize=10, labelpad=0.2)
                ax.set_ylabel(r'$P_i$', fontsize=10, labelpad=0.2)
                # ax.set_ylim(0,20000)

            fig.subplots_adjust(wspace=0.5, hspace=1)

    plt.show()



def plot_paper_predictions(nsss, multi_model, result, run_index,
                           save_path: str | None = None):

    # Import the data
    # Make a list of mesh pairs to use with the submodels
    tx_pairs = []
    tx_pairs_fine = []
    mesh_pairs = []
    mesh_pairs_fine = []
    datas = []
    data_errs = []

    for nss in nsss:
        t = torch.from_numpy(np.loadtxt(f'binned-data/{nss}Spine_t.txt')).to(torch.float64)
        x = torch.from_numpy(np.loadtxt(f'binned-data/{nss}Spine_x.txt')).to(torch.float64)
        t_fine = torch.linspace(t.min(), 40,100).to(torch.float64)

        tx_pairs_fine.append((t_fine, x))
        tx_pairs.append((t,x))

        mesh_pairs.append(torch.meshgrid(t,x,indexing="ij"))
        mesh_pairs_fine.append(torch.meshgrid(t_fine, x, indexing='ij'))

        data_name = f'{nss}Spine_data'
        datas.append(torch.from_numpy(np.loadtxt(f'binned-data/{data_name}.txt')).to(torch.float64))
        data_errs.append(torch.from_numpy(np.loadtxt(f'binned-data/{data_name}_errs.txt')).to(torch.float64))

    nsss_weights_fine = multi_model.generate_nsss_weights(nsss, tx_pairs_fine)

    # breakpoint()
    preds_fine = multi_model(
        False, nsss_weights_fine, mesh_pairs_fine, 
        torch.from_numpy(result.optimize_result.as_list()[run_index].x))

    supfig = plt.figure(figsize=(10,12), dpi=60)
    subfigs = supfig.subfigures(len(nsss),1,
                                height_ratios=(3.5,3,2,2),
                                )

    offtops = [0.88,0.85,0.78,0.78]
    offbots = [0.1,0.2,0.2,0.2]

    data_srss = []
    preds = []
    residuals = []
    rel_residuals = []

    for (t, x), (t_fine, _), pred_fine, data, data_err, nss, fig, offtop, offbot in \
        zip(tx_pairs, tx_pairs_fine, preds_fine, datas, data_errs, nsss,
            subfigs,offtops, offbots):

        n_fig_cols = 6
        n_fig_rows = len(x)//n_fig_cols
        if len(x) % n_fig_cols: n_fig_rows += 1


        if nss != 1:
            fig.suptitle(f'Protocol: {nss} stimulations', weight='bold', x=0.15)
        else: 
            fig.suptitle(f'Protocol: 1 stimulation', weight='bold', x=0.15)

        for idx, (pred_srs, data_srs, data_srs_err, pos) \
            in enumerate(zip(pred_fine.T, data.T, data_err.T, x)):

            data_srss.append(data_srs.clone())
            preds.append(pred_srs[[-15,-10,-5,2,10,20,30,50]])
            residuals.append((pred_srs[[-15,-10,-5,2,10,20,30,50]] - data_srs))
            rel_residuals.append((pred_srs[[-15,-10,-5,2,10,20,30,50]] - data_srs)/data_srs)

            pred_srs /= 10000
            data_srs /= 10000
            data_srs_err /= 10000

            ax = fig.add_subplot(n_fig_rows, n_fig_cols, idx+1)
            ax.errorbar(t, data_srs, yerr = data_srs_err, fmt='.',
                        color='gray',
                        elinewidth=1)

            # Plot the prediction remembering the gape
            pre_idx = torch.where(t_fine<0)
            ax.plot(t_fine[pre_idx],pred_srs[pre_idx], color='tab:orange',
                    label='pre')

            post_idx = torch.where(t_fine>=2)
            ax.plot(t_fine[post_idx], pred_srs[post_idx], color='tab:orange',
                    label='post')


            ax.set_xticks(t)
            ax.set_xticklabels([])

            ax.set_title(r'$x^{(i)}$ = ' + f'{pos:.2f} $\mu m$', fontsize=10, pad=0)

        fig.subplots_adjust(wspace=.3, hspace=1, left=0.05, right=0.98,
                            top=offtop, bottom=offbot)


    if save_path is not None:
        plt.savefig(join(save_path, 'optimal_fit.svg'))
        print('Figure saved!')


    preds = torch.concatenate(preds)
    residuals = torch.concatenate(residuals)
    data_srss = torch.concatenate(data_srss)

    rel_residuals = torch.concatenate(rel_residuals)

    fig, axs = plt.subplots(2,2,figsize=(8,3), height_ratios=(1,3))
    fig.subplots_adjust(wspace=0.5, top=0.8, bottom=0.2)

    rmse = torch.sqrt((residuals**2).mean())
    rrmse = torch.sqrt((residuals**2).sum()/(data_srss**2).sum())

    ax = axs[0,0]
    vals, bins, _ = ax.hist(residuals, bins=100, density=True, linewidth=2)
    ax.fill_between((-rmse, rmse), (0,0), (2,2), color='tab:blue',
                    alpha=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(bins[0], bins[-2])
    ax.set_ylim(0, vals.max())
    ax.legend(loc=(0.1,1))
    ax.spines['left'].set_visible(False)


    ax = axs[1,0]
    vals, bins, _ = ax.hist(residuals, bins=100, density=True, cumulative=True,
                         histtype='step', linewidth=2)

    y1 = vals[np.isclose(bins[1:], -rmse, atol=100)][0]
    y2 = vals[np.isclose(bins[1:], rmse, atol=100)][-1]
    ax.plot((bins[0], -rmse),(y1, y1), lw=1, c='tab:blue')
    ax.plot((bins[0], rmse),(y2, y2), lw=1, c='tab:blue')

    offset = (bins[0] - rmse)/3
    ax.plot((offset, offset), (y1, y2), lw=1, linestyle=(0,(3,3)), c='black')
    t = ax.text(offset, (y1+y2)/2, f'{(y2-y1)*100:.0f}%', backgroundcolor='white',
                ha='center', va='center')

    ax.fill_between((-rmse, rmse), (0,0), (2,2), color='tab:blue', alpha=0.1,
                    label=f'|R| < RMSE')
    ax.set_xlim(bins[0], bins[-2])
    ax.set_ylim(0,1.1)
    ax.set_xticks([bins[0], 0, bins[-2]], 
                  (f'{bins[0]:.0f}', 0.0, f'{bins[-2]:.0f}'))
    ax.set_yticks((y1, y2),(f'{y1:.2f}', f'{y2:.2f}'))

    ax.set_xlabel('Residual, R')
    ax.set_ylabel('CDF')
    ax.legend(loc=(0.24,1.5))

    
    ax = axs[0,1]
    vals, bins, _ = ax.hist(rel_residuals, bins=100, density=True, linewidth=2)
    ax.fill_between((-rrmse, rrmse), (0,0), (200,200), color='tab:blue', alpha=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(bins[0], bins[-2])
    ax.set_ylim(0, vals.max())
    ax.spines['left'].set_visible(False)

    ax = axs[1,1]
    vals, bins, _ = ax.hist(rel_residuals, bins=100, density=True, cumulative=True,
                         histtype='step', linewidth=2)

    y1 = vals[np.isclose(bins[1:], -rrmse, atol=0.01)][0]
    y2 = vals[np.isclose(bins[1:], rrmse, atol=0.01)][0]
    ax.plot((bins[0], -rrmse),(y1, y1), lw=1, c='tab:blue')
    ax.plot((bins[0], rrmse),(y2, y2), lw=1, c='tab:blue')

    offset = (bins[0] - rrmse)/3
    ax.plot((offset, offset), (y1, y2), lw=1, linestyle=(0,(3,3)), c='black')
    t = ax.text(offset, (y1+y2)/2, f'{(y2-y1)*100:.0f}%', backgroundcolor='white',
                ha='center', va='center')

    ax.fill_between((-rrmse, rrmse), (0,0), (2,2), color='tab:blue', alpha=0.1,
                    label='|RR| < RRMSE')
    ax.set_xlim(bins[0], bins[-2])
    ax.set_ylim(0,1.1)
    ax.set_xticks([bins[0], 0, bins[-2]], 
                  (f'{bins[0]:.2f}', 0.0, f'{bins[-2]:.2f}'))
    ax.set_yticks((y1, y2),(f'{y1:.2f}', f'{y2:.2f}'))

    ax.set_xlabel('Relative residual, RR')
    ax.set_ylabel('CDF')
    ax.legend(loc=(0.21,1.5))

    axs[0,0].text(-7000, 1e-3, 'a.', weight='bold', fontsize=20)
    axs[0,1].text(-0.5, 10, 'b.', weight='bold', fontsize=20)

    if save_path is not None:
        plt.savefig(join(save_path, 'residuals.png'), dpi=300)
        print('Figure saved!')


    print(f'RMSE = {rmse}')
    print(f'RRMSE = {rrmse}')


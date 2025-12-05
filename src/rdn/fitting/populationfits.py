import numpy as np
from scipy.optimize import least_squares


from rdn.dataloading import df_from_tool
from rdn.defaults import default_goda_transformation, default_goda_filter

def population_fits(axs, path_to_dataset):

    df = df_from_tool(path_to_dataset, dataset='goda', compartment='spine')
    df = df[df['drug']=='Control']
    df = default_goda_transformation(df)
    fdf = default_goda_filter(df)

    nsss = [-1,1,3,5,7,15]

# After stimulus
    t_point = '2'
    close_th = 4

# RID
    tot_RID_per_nss = []
    stim_RID_per_nss = []
    close_RID_per_nss = []
    far_RID_per_nss = []


    for nss in nsss:
        fdff = fdf[fdf['nss']==nss]
        sfdff = fdff[fdff['type']=='Stim']
        cfdff = fdff[fdff['distance'] <= close_th]
        ffdff = fdff[fdff['distance'] > close_th]

        # After stimulus
        tot_RID_per_nss.append(fdff[f'{t_point}'].to_list())
        stim_RID_per_nss.append(sfdff[f'{t_point}'].to_list())

        close_RID_per_nss.append(cfdff[f'{t_point}'].to_list())
        far_RID_per_nss.append(ffdff[f'{t_point}'].to_list())



    def f_tot(x,a,b,c,d):
        return (10**a + b*x)/ (10**c * x + d)

    def f_close(x,a,b,c,d):
        return (a*x)/(10**b + c*x)

    def f_far(x,a,b,c,d):
        return (10**a - 10**b * x)/(10**c + 10**d*x)

    def loss(p, f, x, y, yerr):
        return (f(x,*p) - y)**2/yerr**2


    # Fitplot: the classic method where I give him some RID list of lists,
    # he parses it into numpy, takes means and errs, fits and plots

    def fit_plot(ax, f, X, data, title):
        Y = np.array([np.mean(d) for d in data])
        Y_err = np.array([np.std(d)/np.sqrt(len(d)) for d in data])

        # Plotting limit
        y_range = [
            (Y - Y_err).min() - 200,
            (Y + Y_err).max() + 900,
        ]

        # breakpoint()
        X = np.array(X)
        res = least_squares(loss, [1.,1.,1.,1.], args=(f,X,Y, Y_err))
        X_fit = np.linspace(0.5, 16, 100)
        Y_fit = f(X_fit, *res.x) 
        Y_pred = f(X, *res.x)


        rmse = np.sqrt(((Y - Y_pred)**2).sum())
        rrmse = np.sqrt(((Y-Y_pred)**2).sum()/(Y**2).sum())*100
        
        ax.errorbar(X, Y, Y_err, 
                    fmt='.', c='black', markersize=10, linewidth=1, capsize=2,
                    label='Data')

        ax.plot(X_fit, Y_fit,
                linewidth=3, 
                label=r'Model ($RRMSE = {:.1f}\%$)'.format(rrmse))

        ax.set_ylim(y_range)
        

        ax.legend(frameon=False, loc=[0.3,.8], fontsize=10)
        ax.set_xticks(X)
        ax.set_title(title, pad=25)



    # fig, axs = plt.subplots(1,3, figsize=(12,3), dpi=200)
    # fig.subplots_adjust(wspace=0.5, top=0.8, bottom=0.2, left=0.1, right=0.98)

    ax = axs[0]
    fit_plot(ax,
             f_tot,
             nsss[1:],
             tot_RID_per_nss[1:],
             'All spines')

    ax = axs[1]
    fit_plot(ax,
             f_close,
             nsss[1:],
             close_RID_per_nss[1:],
             'Close spines')

    ax = axs[2]
    fit_plot(ax,
             f_far,
             nsss[1:],
             far_RID_per_nss[1:],
             'Far spines')

    for ax in axs:
        ax.set_xlabel('Number of stimulations')
        ax.set_ylabel('Fluorescence [a.u.]')

    # axs[0].text(-4,1.32e4,'a.', weight='bold', size=20)
    # axs[1].text(-4,1.65e4,'b.', weight='bold', size=20)
    # axs[2].text(-4,1.25e4,'c.', weight='bold', size=20)

    # plt.show()

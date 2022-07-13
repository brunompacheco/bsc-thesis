import pandas as pd


def get_runs_data(runs, keys=None):
    summary_list = [] 
    config_list = [] 
    name_list = [] 
    id_list = [] 
    histories = []
    for run in runs: 
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict) 

        # run.config is the input metrics.
        # We remove special values that start with _.
        config = {k:v for k,v in run.config.items() if not k.startswith('_')}
        config_list.append(config) 

        # run.name is the name of the run.
        name_list.append(run.name)       
        id_list.append(run.id)       

        h = run.history(int(1e5), keys=keys)
        h['name'] = run.name
        histories.append(h)

    summary_df = pd.DataFrame.from_records(summary_list) 
    config_df = pd.DataFrame.from_records(config_list) 
    name_df = pd.DataFrame({'name': name_list}) 
    id_df = pd.DataFrame({'id': id_list}) 
    all_df = pd.concat([name_df, id_df, config_df,summary_df], axis=1)

    histories = pd.concat(histories).reset_index(drop=True)
    histories['epoch'] = histories['_step'] + 1

    return all_df, histories

def plot_learning_curve(ax, histories, label, window=100, shadow=True):
    iae_low = histories.groupby('epoch')['val_loss_iae'].min()
    iae_high = histories.groupby('epoch')['val_loss_iae'].max()
    # iae_low = histories.groupby('epoch')['val_loss_iae'].quantile(.25)
    # iae_high = histories.groupby('epoch')['val_loss_iae'].quantile(.75)
    # iae_std = histories.groupby('epoch')['val_loss_iae'].std()
    iae = histories.groupby('epoch')['val_loss_iae'].mean()

    iae_low = iae_low.rolling(window, min_periods=1).mean()
    iae_high = iae_high.rolling(window, min_periods=1).mean()
    # iae_std = iae_std.rolling(window, min_periods=1).mean()
    iae = iae.rolling(window, min_periods=1).mean()

    if shadow:
        ax.fill_between(iae_low.index, iae_low, iae_high, alpha=.5, linewidth=0)
        # ax.fill_between(iae.index, iae-iae_std, iae+iae_std, alpha=.5, linewidth=0)
    ax.plot(iae, label=label, linewidth=1)

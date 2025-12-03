import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import seaborn as sns
from IPython.display import display
from sklearn.compose import make_column_selector as selector, ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BIC, K2, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination



# Helpers

def plot_label_dist(train_labels, seed, colormap):
    # plot distribution of positive and negative examples in train:
    fig = plt.figure(figsize=(4, 4))
    ax = sns.countplot(pd.DataFrame(train_labels), x='Oxd', hue='Oxd', stat="percent", palette=colormap, alpha=0.9, edgecolor='black', linewidth=0.25, legend=False)
    plt.suptitle(f'Label distribution for seed={seed}')
    for i in ax.containers:
        ax.bar_label(i, fmt='{:,.2f}%')
        ax.set_ylim(0,100)
    plt.tight_layout()
    plt.show()
    #plt.close(fig)
    

    
def define_type(col):
    
    '''
    identify type of a descriptor (column)
    '''
    
    res = map(lambda x: x % 1 == 0, [min(col), np.percentile(col, 25), np.percentile(col, 50), np.percentile(col, 75), max(col), sum(col)])
    if all(list(res)):
        return np.int64
    else:
        return np.float64



def drop_col(data):
    
    '''
    drop uninformative descriptors (low variation)
    '''
    
    dropped = []
    for col in data.columns.to_list():
        rel_freqs = data[col].value_counts(normalize=True)

        if max(rel_freqs)>0.95:
            data = data.drop(col, axis=1)
            dropped.append(col)

    print(f'The following columns have been dropped:') 
    print(*dropped, sep=',')
    return data




def check_freq(data, col, thres=0.01):
    
    '''
    auxiliary function
    returns True if the feature requires binning, that is, if for at least one category the frequency is < threshold value
    '''
    
    if col.empty:
        raise ValueError("Column is empty.")
    if col.nunique() == 1:
        return False
    col = pd.Series(col)
    freq = data.groupby([col]).size().reset_index(name='counts')
    freq['rel_freq'] = freq['counts']/sum(freq['counts'])
    if sum(freq['rel_freq']<thres) > 0: 
        return True
    else:
        return False



def drop_dups(data):
    '''
    cast the substrate names to lower case
    drop duplicates if any and drop the name column  
    '''
    if 'Name' in data.columns:
        data['Name'] = data['Name'].str.lower()
        data = data.drop_duplicates().reset_index(drop=True)
        data.drop(['Name'], axis=1, inplace=True)
    return data



def set_coltypes(data):
    '''
    function that takes in a dataframe
    * replaces missing labels with 0
    * changes some dtypes (for Qindex, Wap and CENT) manually
    returns modified dataframe
    '''
    if 'Oxd' in data.columns:
        data['Oxd'].fillna(0, inplace=True) 
        data['Oxd'] = data['Oxd'].round().astype('int64')
    cols_tp = list(map(lambda x: data.iloc[:, x], np.arange(0, len(data.columns))))
    convert_dict = dict(zip(data.columns, list(map(define_type, cols_tp)))) 
    convert_dict.update(Qindex=np.float64, Wap=np.float64, CENT=np.float64) 
    data = data.astype(convert_dict)
    
    return data


def merge_bins(col, thres = 0.05):
    
    '''
    function that merges adjascent bins of a descriptor in case one of the bins contains too few observations
    therefore provides coarser binning with low information loss
    '''
    
    # initialize bins: each value in a separate bin
    bin_data, bin_edges = pd.cut(col, bins=sorted([-np.inf]+list(col.sort_values().unique())), retbins=True, include_lowest=True)
    # count instances in bins:
    freq = bin_data.value_counts().reset_index()
    freq.rename(columns={ freq.columns[0]: 'bin' }, inplace = True)
    freq['rel_freq'] = freq['count']/sum(freq['count'])
    freq = pd.DataFrame(freq).sort_values(by=['bin'])
    freq['edges'] = freq['bin'].values.categories.left # lower bounds of the bin intervals
    bins_ = list(bin_edges)
    
    if len(bins_)>0:
        # take the corresponding rel_freq and check the condition:
        while any(freq['rel_freq'].to_numpy() < thres):
            rmv =  max(freq[freq['rel_freq'] < thres]['edges']) # the eldest bucket with low rel_freq
            # adjust binning by excluding the above edge:
            try:  
                bins_.remove(rmv)
            except:   
                print('Something went wrong!')
                pass
            bin_data, bin_edges = pd.cut(col, bins=bins_, retbins=True, include_lowest=True)  
            
            # recompute frequencies with new bins: 
            freq = bin_data.value_counts().reset_index()
            freq.rename(columns={ freq.columns[0]: 'bin' }, inplace = True)
            freq['rel_freq'] = freq['count']/sum(freq['count'])
            freq = pd.DataFrame(freq).sort_values(by=['bin'])
            freq['edges'] = freq['bin'].values.categories.left
            
        # as we remove the left edge, if the very first bin is combined the very first edge is lost, after the binning is finished, 
        # the bin_edges need to be adjusted by replacing the very left edge by -inf
        bins_[0] = -np.inf 
        bins_[-1] = +np.inf 
        bin_data, bin_edges = pd.cut(col, bins=bins_, retbins=True, include_lowest=True)
        return bin_data      
    else: 
        raise RuntimeError(f"Failed to remove bin edge: {rmv}")
    
    

# -----------------------------
# 1. Entropy utility functions
# -----------------------------
def entropy(y):
    """Shannon entropy."""
    _, counts = np.unique(y, return_counts=True)
    prob = counts / counts.sum()
    return -np.sum(prob * np.log2(prob))


def information_gain(y, y_left, y_right):
    """Information gain from splitting y into y_left and y_right."""
    H = entropy(y)
    H_left = entropy(y_left)
    H_right = entropy(y_right)

    w_left = len(y_left) / len(y)
    w_right = len(y_right) / len(y)

    return H - (w_left * H_left + w_right * H_right)


def mdl_stop(y, y_left, y_right, ig):
    """Minimum Description Length stopping criterion."""
    k = len(np.unique(y))
    k_left = len(np.unique(y_left))
    k_right = len(np.unique(y_right))

    # Entropy gain threshold ("delta")
    delta = np.log2(3**k - 2) - (
        k * entropy(y)
        - k_left * entropy(y_left)
        - k_right * entropy(y_right)
    )

    return ig <= 0.25 * ((np.log2(len(y) - 1) / len(y)) + (delta / len(y))) # ig <= (np.log2(len(y) - 1) / len(y)) + (delta / len(y))


# -----------------------------
# 2. Find cut points for one variable 
# -----------------------------
def find_mdl_cut_points(x, y):
    """
    x: continuous variable (1D array)
    y: target variable (class labels)
    Returns: sorted list of cut points
    """

    # Sort according to x
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # Candidate split indices where class label changes
    candidates = []
    for i in range(1, len(x_sorted)):
        if y_sorted[i] != y_sorted[i - 1]:
            split = (x_sorted[i] + x_sorted[i - 1]) / 2.0
            candidates.append(split)

    if not candidates:
        return []

    # Evaluate all candidates and pick best IG
    best_ig = -np.inf
    best_cut = None

    for cut in candidates:
        left_idx = x_sorted <= cut
        right_idx = x_sorted > cut

        y_left = y_sorted[left_idx]
        y_right = y_sorted[right_idx]

        ig = information_gain(y_sorted, y_left, y_right)

        if ig > best_ig:
            best_ig = ig
            best_cut = cut

    # Stopping rule
    left_idx = x_sorted <= best_cut
    right_idx = x_sorted > best_cut
    y_left = y_sorted[left_idx]
    y_right = y_sorted[right_idx]

    if mdl_stop(y_sorted, y_left, y_right, best_ig):
        return []

    # Recursive splitting
    left_cuts = find_mdl_cut_points(x_sorted[left_idx], y_left)
    right_cuts = find_mdl_cut_points(x_sorted[right_idx], y_right)

    return left_cuts + [best_cut] + right_cuts


# -----------------------------
# 3. Discretize a column using MDL
# -----------------------------
def mdl_discretize_column(x, y):
    """Return discretized version of x based on MDL cuts."""
    cuts = sorted(find_mdl_cut_points(x.values, y.values))
    if not cuts:
        return pd.Series(np.zeros(len(x)), index=x.index), cuts

    # Use pandas cut
    bins = [-np.inf] + cuts + [np.inf]
    labels = list(range(len(bins) - 1))
    return pd.cut(x, bins=bins, labels=labels).astype(int), cuts


# -----------------------------
# 4. Apply MDL discretization to a dataset
# -----------------------------
def bin_features_mdl(df, target_col='Oxd'):
    """
    df: DataFrame with both continuous & categorical variables
    target_col: column used to evaluate entropy (class label)
                If None, use each variable's own coarse bin as secondary target.
    """

    df_disc = df.copy()

    # If no label provided, create a pseudo-label (weak supervision)
    if target_col is None:
        # fallback: self-supervised bin using quantiles
        tmp_target = pd.qcut(df.iloc[:, 0], q=4, duplicates='drop').cat.codes
        y = tmp_target
    else:
        y = df[target_col]

    for col in df.columns:
        if col == target_col:
            continue

        if df[col].dtype.kind in "biufc":  # numeric types
            x = df[col]

            # MDL discretize
            x_disc, cuts = mdl_discretize_column(x, y)
            df_disc[col] = x_disc

        else:
            # keep categorical as-is
            df_disc[col] = df[col]

    return df_disc



def plot_bn_graph(relevant_edges, seed, reduced, colormap):
    
    '''
    auxiliary function for plotting the whole BN graph or a Markov blanket if reduced=True
    groups of discriptors are depicted in different colors (hardcoded)
    
    args:
    relevant edges: either the whole graph or the reduced graph
    seed: used for naming the plot when it is saved in the directory
    reduced: if None, the whole network is plotted, else the Markov Blanket 
    
    outputs: a networkx plot
    '''
    
    # create a graph object and add edges
    G = nx.DiGraph()
    G.add_edges_from(relevant_edges)

    # color the nodes by groups
    node_colors = []
    target_node = 'Oxd'
    for node in G.nodes:
        if node == target_node:
            node_colors.append(colormap[0])  # tgt node
        elif node in ['nBM', 'nTB', 'nAB', 'nH', 'nC', 'nN', 'nO', 'nP', 'nS', 'nCL', 'nHM', 'nHet', 'nCsp3', 'nCsp2', 'nCsp', 'nCIC', 'nCIR']:
            node_colors.append(colormap[1]) 
        elif node in ['nR05', 'nR06', 'nR09', 'nR10', 'nR11', 'nR12', 'Psi_i_A', 'Psi_i_t', 'Psi_i_0d', 'Psi_i_1s']:
            node_colors.append(colormap[2]) 
        elif node in ['C%', 'N%', 'O%', 'X%', 'PW2', 'PW3', 'PW4', 'PW5']:
            node_colors.append(colormap[3]) 
        elif node in ['P_VSA_m_1', 'P_VSA_m_2', 'P_VSA_m_4', 'P_VSA_v_2', 'P_VSA_v_3', 'P_VSA_e_3', 'P_VSA_i_1', 'P_VSA_i_2', 'P_VSA_i_3', 'P_VSA_s_1', 'P_VSA_s_3', 'P_VSA_s_4', 'P_VSA_s_6']:
            node_colors.append(colormap[4]) 
        elif node in ['nCs', 'nCt', 'nCq', 'nCrs', 'nCrq', 'nCconj', 'nR=Cs', 'nR=Ct', 'nRCOOR', 'nRCO', 'nCONN', 'nRNH2', 'nRNR2', 'nRCN', 'nRNO', 'nC=N-N<', 'nROR', 'nRSR', 'nS(=O)2', 'nSO3', 'nSO2N', 'nCXr=', 'nCconjX', 'nTriazoles', 'nHDon']:
            node_colors.append(colormap[5]) 
        else:
            node_colors.append(colormap[6])  # all other nodes

    # draw the whole or reduced graph and save
    pos = nx.spiral_layout(G)  
    if reduced:
        fig = plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_size=1000, node_color=node_colors, font_size=8, arrowsize=10, alpha=0.85)
        plt.title("Bayesian Network ('Oxd' Markov Blanket)")
        #plt.savefig(f'plots/markov_blanket_{seed}.png', bbox_inches='tight') 
    else:
        fig = plt.figure(figsize=(18, 16))
        nx.draw(G, pos, with_labels=True, node_size=1500, node_color=node_colors, font_size=10, arrowsize=10, alpha=0.85)
        plt.title("Bayesian Network")
        #plt.savefig(f'plots/bayes_net_{seed}.png', bbox_inches='tight') 
    #plt.close(fig) 
    plt.show()



def select_edges(best_bn):
    
    '''
    function for constructing the Markov Blanket from the learned Bayesian Network
    
    args:
    * best_bn: learned Bayesian Network (bnlearn object)
    
    outputs: edges constituting the Markov Blanket 
    '''
    
    # get the Markov blanket of the target node:
    markov_blanket = best_bn.get_markov_blanket('Oxd')

    # Identify relevant edges
    relevant_edges = []
    for edge in best_bn.edges():
        if edge[0] in markov_blanket or edge[1] in markov_blanket:
            relevant_edges.append(edge)
  
    return relevant_edges



    
def plot_bins(ord_features, seed, binned):
    
    '''
    auxiliary function for plotting the ordinals prior to binning and after: 
    allows to check how the binning function works
    args:
    ord_features: list of 
    '''
    
    l = len(ord_features.columns)//3
    fig, axn = plt.subplots(l, 3, figsize=(8, l*3), sharex=False, sharey=True)


    #cmap = colormap*(len(ord_features.columns)//7)
    n_ = len(ord_features.columns)
    cmap_ = mpl.colormaps['RdYlBu']
    # take colors at regular intervals spanning the colormap.
    cmap = cmap_(np.linspace(0, 1, n_))
    
    
    for i, ax in enumerate(axn.flat):
        labs = range(min(ord_features.iloc[:, i]), max(ord_features.iloc[:, i])+1, 1)
        g = sns.histplot(ord_features.iloc[:, i], discrete=True, stat='probability', color=cmap[i], edgecolor='darkslategray', ax=ax)  # Use the existing figure
        g.set_title(f'{ord_features.columns.tolist()[i]}', fontsize=9)
        g.set_xticks(range(len(labs))) 
        g.set_xticklabels(labs, fontsize=6) 
        g.set_xlabel(None)
        g.set_yticklabels(np.round(np.linspace(0, 1, 6),1), fontsize=6)
           
    fig.suptitle(f'Distribution of ordinal descriptors', fontsize=11)
    fig.tight_layout()
    #plt.savefig(f'plots/ord_binned_{binned}_{seed}.png', bbox_inches='tight')
    #plt.close(fig)
    plt.show()
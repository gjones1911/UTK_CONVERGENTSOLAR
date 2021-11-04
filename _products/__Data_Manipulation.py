import numpy as np
import pandas as pd
import time
import sys
pd.options.mode.use_inf_as_na = True

def get_prefix(str, end):
    return str[:-end]

def get_missing(dfa, verbose=False, threshold=.20):
    desc = dfa.describe()
    rl = dict()
    rl['vars'] = list()
    rl['missing'] = list()
    rl['% missing'] = list()
    bad = list()
    for v in dfa.columns.tolist():
        if v in desc.columns.tolist():
            rl['vars'].append(v)
            rl['missing'].append(dfa.shape[0] - desc.loc['count', v])
            rl['% missing'].append( (dfa.shape[0] - desc.loc['count', v])/dfa.shape[0])
            if rl['% missing'][-1] > threshold:
                bad.append(v)
            if verbose:
                print('V: {}, missing: {}'.format(v, rl['missing'][-1]))
    rl = pd.DataFrame(rl).sort_values(by='missing', ascending=False)
    return rl, bad

def pandas_printer(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

def add_GIS_geoid(df, fips_col='fips', gis_col='G_fips'):
    gis_join = list()
    for f in df[fips_col]:
        if len(str(f)) % 2 == 0:
            gis_join.append('0' + str(f))
        else:
            gis_join.append(str(f))
    df[gis_col] = gis_join

def df_colname_conv(df, conversion_dict):
    new_col_list = df.columns.tolist().copy()
    for tocon in conversion_dict:
        df[conversion_dict[tocon]] = df[tocon].values.tolist()
        rmv_list(new_col_list, tocon)
        new_col_list.append(conversion_dict[tocon])
    print('the size of the columns at first was: {}'.format(len(df.columns.tolist())))
    print('The size of the columns at now is: {}'.format(len(new_col_list)))
    return new_col_list


def rmv_list(l, r):
    del l[l.index(r)]
    return l

def rmv_list_list(l, rl):
    for r in rl:
        l = rmv_list(l, r)
    return l



def percentage_generator(df, part, total, newvar=None):
    """    will calculate the percentage
           of the total value part is in a list
    :param df:
    :param part:
    :param total:
    :return:
    """
    if newvar is None:
        return list(df[part]/df[total])
    df[newvar] = list(df[part]/df[total])
    return

def report_var_stats(df, name, saveit=True, sort_type=None, sort_list=[], ascending=True, axis=0,
                     re_nan=(-999,), verbose=False, csv=False):
    """Creates an excel file containing:
                    * missing counts for each variable
                    * the range for each variable
                    * mean for each variable
                    * standard deviation for each variable
                    * the range for each variable
                    * TODO: need to add skew to table
    :param df:   The data frame containing the data to add to report
    :param name: The name of the new file
    :param saveit: if true report will be saved under given name
    :param sort_type: options are:
                      * 'index' for row sorting
                      * 'columns' for column sorting
                      * None (default) for no sorting
    :param sort_list: the list of columns or indices to sort by, empty will just do lex sort
    :return: returns the newly created data frame used
    """
    # add given list of nan representations
    for re in re_nan:
        df.replace(re, np.nan)
    # grab stat statistices
    descr = df.describe()
    # grab total number of entries
    if verbose:
        print('-------         There are {:d} entries in the set         -------')
    N = len(df)
    # set the indices to that of the given data frame dummy
    dfskew = df.skew()
    print('skew index\n',dfskew.index)
    print('given df index\n',df.index)
    rdic = {'Missing':[], 'Range':[], 'Mean':[], 'std':[], 'Skew':[]}
    #rdic = {'Missing':[], 'Range':[], 'Mean':[], 'std':[]}
    for var in descr.columns.values.tolist():
        rdic['Missing'].append(np.around((N-descr.loc['count',var])/len(df), 3))
        rdic['Range'].append([np.around(descr.loc['min', var], 4), np.around(descr.loc['max', var],4)])
        rdic['Mean'].append(descr.loc['mean',var])
        rdic['std'].append(descr.loc['std',var])
        rdic['Skew'].append(dfskew.loc[var])
    # create data from from created dictionary
    rdf = pd.DataFrame(rdic, index=descr.columns.values.tolist())
    if sort_type is not None:
        if sort_type == 'value':
            rdf.sort_values(by=sort_list, axis=axis, inplace=True, ascending=ascending)
        elif sort_type == 'index':
            rdf.sort_index(axis=axis, inplace=True, ascending=ascending)
    if saveit:
        if not csv:
            rdf.to_excel(name)
        else:
            rdf.to_csv(name)
    return rdf

def concat_columns(df, cols, datas, verbose=False):
    rdf = {}
    for col, data in zip(cols, datas):
        if verbose:
            print('col',col)
            print('data',data)
        rdf[col] = df[col].values.tolist()
        rdf[col].append(data)
    rdfdf = pd.DataFrame(rdf)
    if verbose:
        print('return df', rdf)
    return rdfdf

def concat_col(df, col, data, verbose=False):
    ldf = df[col].values.tolist()
    dl = [data]
    if verbose:
        print('data frame \n', ldf,'\ndata\n', dl)
    return ldf + dl

def create_combo_var_sum(df, list_to_sum, newvar=None):
    if newvar is None:
        return df[list_to_sum].sum(axis=1).values.tolist()
    df[newvar] = df[list_to_sum].sum(axis=1).values.tolist()
    return


def smart_df_divide(dfn, dfd, rep_val=0):
    rl = list()
    for n, d in zip(dfn, dfd):
        if d <= 0:
            rl.append(rep_val)
        elif d == np.nan or n == np.nan:
            rl.append(np.nan)
        else:
            rl.append(n/d)
    return rl



def add_renewable_gen(df, val, dictl):
    df['Ren'] = list([0]*len(df))
    for st in dictl:
        df.loc[df[val] == st, 'Ren'] = dictl[st]
    #return df
    return

def add_renewable_gen_df_df(dfd, sourcefile, cols, open_method, fillins='STUSPS', cold=None):
    dfs = open_method(sourcefile, usecols=cols + [fillins])
    to_fill = dfs[fillins].values.tolist()
    for c in cols:
        dfd[c] = list([0] * len(dfd))


    for st in to_fill:
        for c in cols:
            #dfd[c] = list([0] * len(dfd))
            if st.lower() in dfd[cold].values.tolist():
                dfd.loc[dfd[cold] == st.lower(), c] = dfs.loc[dfs[fillins] == st, c].values[0]
    #return df
    return


def store_var_ranges(df, vars):
    """
        takes a data frame and the variables you want to find the ranges of and returns a new data frame
        with one column of the variables and the other the corresponding varialbes range
    :param df:
    :param vars:
    :return:
    """
    var_stats = df.describe()
    var = list()
    ranges = list()
    for v in vars:
        var.append(v)
        ranges.append('[{0}, {1}]'.format(var_stats.loc['min', v], var_stats.loc['max', v]))
    return pd.DataFrame({'Variable':var, 'Original Range':ranges})


def recode_var_sub(sought, check, keyd):
    """
        will create a list of recoded variables based on a list of substrings(sought) that will be
        searched for in the check list, useing the recode map keyd
    :param sought:
    :param check:
    :param keyd:
    :return:
    """
    rl = list()
    for c in check:
        for substr in sought:
            #print(substr)
            #print(c)
            if pd.isna(c):
                #print('bad c!',c)
                rl.append(np.nan)
                break
            elif substr in c:
                #print(c)
                #print(substr)
                rl.append(keyd[substr])
                break
    return rl

def load_model_attribs(filename, colname='Variables'):
    """
        Loads a set of features from a given excel file
    :param filename:
    :param colname:
    :return:
    """
    return pd.read_excel(filename).loc[:,colname].values.tolist()


def thresh_binary_recode(df, var, valthresh=0):
    bin_re = list([0]*df.shape[0])
    #print('new list is of size {}'.format(len(bin_re)))
    df[var + '_bin'] = bin_re
    df.loc[df[var] > valthresh, var + '_bin'] = 1

def generate_mixed(df, vars, mix_name):
    df[mix_name] = df[vars[0]].values.tolist()
    for v in range(1, len(vars)):
        df[mix_name] = (df[mix_name].values * df[vars[v]].values).tolist()

def shuffle_deck(deck):
    np.random.shuffle(deck.values)


def recode_locale_data(df, sought, local_recode, local_recodeA):
    local = list(df['locale'])
    df['locale_dummy'] = recode_var_sub(sought, local, local_recode)
    df['locale_recode'] = recode_var_sub(sought, local, local_recodeA)

    empty_1, empty_2, empty_3, empty_4 = np.zeros(df.shape[0]), np.zeros(df.shape[0]), np.zeros(
        df.shape[0]), np.zeros(df.shape[0])
    empty_1[df['locale_dummy'].values == 1] = 1
    empty_2[df['locale_dummy'].values == 2] = 2
    empty_3[df['locale_dummy'].values == 2] = 3
    empty_4[df['locale_dummy'].values == 2] = 4
    df['locale_recode(rural)'] = empty_1
    df['locale_recode(suburban)'] = empty_3
    df['locale_recode(town)'] = empty_2

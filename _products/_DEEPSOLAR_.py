import pandas as pd
import numpy as np

from _products.__DEEPSOLAR_Resources import *

import scipy.stats as stats

from _products.utility_fnc import blocking_sound_player, error_sounds, Alert_sounds
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as dis_mod
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statistics import mean
from .__Data_Manipulation import  *
#from _products.ML_Tools import cross_val_splitterG, generate_training_testing_data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from _products.utility_fnc import sort_dict, Alert_sounds
from Analysis_Scripts.ignores_list import *
from Analysis_Scripts.State_Drops import *
from Analysis_Scripts.Block_Group_Drops import *
import os
pd.options.mode.use_inf_as_na = True

def stat_table_maker(df, new_table):
    desc = df.describe()
    vars = desc.columns.tolist()
    # go through description getting neccesary values
    rd = {
        'Variables':[],
        'Mean': [],
        'std': [],
        'missing':[],
        'Min': [],
        'Max': [],
        'Range': [],
    }
    for v in vars:
        rd['Variables'].append(v)
        rd['Mean'].append(desc.loc['mean', v])
        rd['std'].append(desc.loc['std', v])
        rd['missing'].append(np.around((len(df) - desc.loc['count', v])/len(df), 2))
        rd['Min'].append(desc.loc['min', v])
        rd['Max'].append(desc.loc['max', v])
        rd['Range'].append([desc.loc['min', v], desc.loc['max', v]])
    pd.DataFrame(rd).to_excel(new_table, index=False)
    return rd

def read_stat_table_min_max_dict(tableName):
    df = pd.read_excel(tableName, index_col='Variables')
    rd = {}
    #print(df.index.tolist())
    for v in df.index.tolist():
        #print('v: {}'.format(v))
        rd[v] = {}
        rd[v]['min'] = df.loc[v, 'Min']
        rd[v]['max'] = df.loc[v, 'Max']
        rd[v]['std'] = df.loc[v, 'std']
        rd[v]['mean'] = df.loc[v, 'Mean']

    return rd


def mm_unscaler(val, minval, maxval):
    return val * (maxval - minval) + minval


def mm_scaler_val(og_val, minval, maxval):
    num = (og_val - minval)
    den = (maxval - minval)

    print('Num/denom: {}/{}'.format(num, den))
    # return (og_val - minval)/(maxval - minval)
    return num / den


def show_var_and_stats(var, minV, maxV):
    print('Variable: {}, Min: {}, Max: {}'.format(var, minV, maxV))
    return


min_max_d = {}


def get_unscaled_val(val, variable, min_max_d=min_max_d, verbose=False):
    if verbose:
        show_var_and_stats(variable, min_max_d[variable]['min'], min_max_d[variable]['max'])
    return mm_unscaler(val, min_max_d[variable]['min'], min_max_d[variable]['max'])


def get_scaled_val(val, variable, min_max_d, ):
    # print('Variable: {}, Min: {}, Max: {}'.format(variable, min_max_dict[variable]['min'], min_max_dict[variable]['max']))
    show_var_and_stats(variable, min_max_d[variable]['min'], min_max_d[variable]['max'])
    return mm_scaler_val(val, min_max_d[variable]['min'], min_max_d[variable]['max'])


def generate_min_max_dict(dfs, varsL, rdict):
    for v in varsL:
        rdict[v]['min'] = dfs[v].min()
        rdict[v]['max'] = dfs[v].max()
        rdict[v]['mean'] = dfs[v].mean()
        rdict[v]['std'] = dfs[v].std()
    return rdict


def countmissing(df, reverse=False, verbose=True, retd=True):
    missingsO = df.isna().sum()
    rd = {}
    rdpct = {}
    N = df.shape[0]
    for cc in missingsO.index.tolist():
        if missingsO[cc] > 0:
            if verbose:
                print('{}: {}'.format(cc, missingsO[cc]))
            rd[cc] = missingsO[cc]
            rdpct[cc] = np.around(missingsO[cc]/N, 4)
    rd =sort_dict(rd, reverse=reverse)
    rdpct = sort_dict(rdpct, reverse=reverse)
    if retd:
        return rd, rdpct
    return

def generate_droplist(missingcountD, pctThresh=.21):
    rl = list()
    for v in missingcountD:
        if missingcountD[v] >= pctThresh:
            rl.append(v)
    return rl



def standardize_select(df, to_scale, verbose=False):
    # Scale the data for the things we want to scale before we impute it
    for v in to_scale:
        if verbose:
            print('V: {}'.format(v))
        df[v] = (df[v].values - df[v].mean()) / (df[v].std())
    return df


def scale_select(df, to_scale, verbose=False):
    # Scale the data for the things we want to scale before we impute it
    for v in to_scale:
        if verbose:
            print('V: {}'.format(v))
        df[v] = (df[v].values - df[v].min()) / (df[v].max() - df[v].min())
    return df

def regional_scale_select(df, regIDCOL, to_mm):
    regID = list(set(df[regIDCOL].tolist()))
    for v in to_mm:
        for r in regID:
            df.loc[df[regIDCOL] == r, v] = (df.loc[df[regIDCOL] == r, v] - df.loc[df[regIDCOL] == r, v].min())/(df.loc[df[regIDCOL] == r, v].max() - df.loc[df[regIDCOL] == r, v].min())


def calculate_perNZ(df, cn, cd, newcol, verbose=False):
    # df[newcol] = np.full(len(df), 0)
    rl = []
    cnt = 0
    weirdones = 0
    baddies = []
    for cnum, cdenom in zip(df[cn].tolist(), df[cd].tolist()):
        if cdenom == 0:
            if verbose:
                print('found a 0 at entry: {}'.format(cnt))
            rl.append(0)
            weirdones += 1
            baddies.append(cnt)
        else:
            rl.append(cnum / cdenom)
        cnt += 1
    df[newcol] = rl
    print("There were {} 0 entries for {}".format(weirdones, cd))
    return


def contradiction_seeker(df, binvar, valvar, verbose=False):
    cnt = 0
    baddies = []
    for a, b in zip(df[binvar].tolist(), df[valvar].tolist()):
        if a >= 1 and b <= 0:
            baddies.append(cnt)
            print("Found an Issue at entry: {}".format(cnt))
            print("{}: {}, {}: {}".format(binvar, a, valvar, b))
        cnt += 1
    print("There are {} contradicting entries".format(len(baddies)))
    return baddies


def mark_majority_thresh(df, col, majthresh=.5):
    df[col + '_Maj'] = np.zeros(len(df))
    df.loc[df[col] > majthresh, col + '_Maj'] = 1
    return


def add_HOTSPOTS(df, var, new_var_name, percentile=.949, verbose=False):
    # get the value
    df[new_var_name] = np.zeros(len(df))
    threshold = df[var].quantile(percentile)
    if verbose:
        print('the {} percentile of {} is {}'.format(percentile, var, threshold))
    # generate the threshold
    df.loc[df[var] >= threshold, 'Hot_Spots_hh'] = 1
    return

def add_allval(df, val=0, name='New_Empty'):
    df[name] = np.full(len(df), val)
    return
def get_reg(v, rd):
    for k in rd:
        if v in rd[k]:
            return k
    print('\t\t\t\tUh Oh !!!!')
    return -1


def display_store_regionals(df, regions, region_listd, model_vars, verbose=True):
    reg_di = {
        "Regional": ['West', 'NorthEast', 'Mid West', 'South'],
        'Hot Spots': ['Hot_Spots_hh', 'Hot_Spots_hown', 'Hot_Spots_AvgAr', ],
        'US': ['US', ],
        'Solar Groups': ['High_Solar_Areas', 'Low_Solar_Areas', ],
        'Locale': ['URBAN', 'locale_recode(rural)', 'locale_recode(suburban)', ],
    }

    vars_to_store = []
    reg_re_d = {}

    for reg in regions:
        if verbose:
            print('\t\t\tThe Group: {}'.format(reg))
        gp = get_reg(reg, reg_di)
        if gp not in reg_re_d:
            reg_re_d[gp] = {}
        reg_re_d[gp][reg] = {}
        for v in model_vars:  # + ['URBAN', 'High_Solar_Areas', 'Low_Solar_Areas',] :
            if reg in ['US', ]:
                mu = df.loc[:, v].mean()
            else:
                mu = df.loc[df[reg] == 1, v].mean()
            if verbose:
                print("The average '{}',: {:.3f}".format(v, mu))
            reg_re_d[gp][reg][v] = [mu]
    return reg_re_d


def plot_regional_grouping_averages_from_dict(regdict, vars_to_plot, verbose=True):
    for group in regdict:
        if verbose:
            print("\t\t\t-----------   {}   ----------".format(group))
        title = group + ':\n'
        plot_df = pd.DataFrame()
        for v in vars_to_plot:
            title2 = title + 'Variable: {}'.format(v.upper())

            for region in regdict[group]:
                if verbose:
                    print("\t\t----   {}   ----".format(region))
                    print("The average value of {}: for region {}: {}".format(v, region, regdict[group][region][v][0]))
                plot_df[region] = regdict[group][region][v]
            fgsz = 8
            ax = plot_df.plot.bar(title=title2, figsize=(fgsz, fgsz))
            mn = plot_df.values.min()
            mx = plot_df.values.max()
            if verbose:
                print('min max {}, {}'.format(mn, mx))
            sc = .01
            sc2 = .1
            mn = max(0, mn - mn * sc2)
            mx = mx + mx * sc
            if verbose:
                print('min max {}, {}\n'.format(mn, mx))

            ax.set_ylim([mn, mx])
    return


def generate_filter_clause(a_l, b_l, ops="=", conct=""):
    return str(a_l) + str(ops) + str(b_l) + str(conct)

def generate_or_filter(al, bl, ops,):
    rstr = ""
    cnt = 1
    for a, b in zip(al, bl):
        conct = ''
        if cnt < len(al):
            conct += ' | '
        rstr += generate_filter_clause( a, b, ops, conct=conct)
        cnt += 1
    return rstr

def generate_region_filter(region_dict, region):
    regions = region_dict[region]
    b1 = [1 for i in range(len(regions))]
    return generate_or_filter(regions, b1, '=')

def print_dict_list(regions=('West', 'N. West', 'S. West',
                             'M. West', 'S. East', 'Mid_atlantic', 'N. East'), diction=USRegions):
    for reg in regions:
        print(generate_region_filter(diction, reg))
        print()
    return


def majority_average_list(df, comparevars, avgcmpr):
    med_inc, avg_inc = dict(), dict()
    cmpr_dict = {}
    df_occu = df.filter(items=comparevars + avgcmpr, axis=1)
    for vc in comparevars:
        cmpr_dict[vc] = {}
        for v in comparevars:
            if vc != v:
                tmp_df = df_occu.loc[df_occu[vc] > df_occu[v], :]
                print('----------------\nComparing {} to {}'.format(vc, v))
                print('Shape: {}'.format(tmp_df.shape))
        print('\n\n-----------------------------------------for the variable: {}'.format(vc))
        print('Shape: {}'.format(tmp_df.shape))
        tmp_df = tmp_df.filter(items=avgcmpr + [vc], axis=1)
        for inc in avgcmpr:
            mu = tmp_df[inc].mean()
            print('the {}: {}'.format(inc, mu))
            cmpr_dict[vc][inc] = [mu]
            #if inc == [0]:
                #med_inc[vc] = [mu]
            #else:
                #avg_inc[vc] = [mu]
            print('-------------------\n')

    #avg_inc = sort_dict(avg_inc)
    #med_inc = sort_dict(med_inc)
    offset = 1000
    figszz = 20
    for varcmpr in avgcmpr:
        new_d = {}
        for v in comparevars:
            new_d[v] = cmpr_dict[v][varcmpr]
            print('adding variable: {}, {}'.format(v, varcmpr))
            print('the value: {}'.format(new_d[v]))
        new_d = sort_dict(new_d)
        nw_df = pd.DataFrame(new_d)
        nwax = nw_df.plot.bar(title='The {}'.format(varcmpr), figsize=(figszz, figszz))
        nwax.set_ylim((nw_df.min(axis=1)[0] - offset, nw_df.max(axis=1)[0] + offset))
    #med_incDF = pd.DataFrame(med_inc)
    #avg_incDF = pd.DataFrame(avg_inc)
    #print('---------------------- the thing med: {}'.format(med_incDF))
    #print(' ------------------- med\n', med_incDF.min(axis=1))

    #medax = med_incDF.plot.bar(title='The {} for the available occupations'.format(avgcmpr[0]), figsize=(figszz, figszz))
    #medax.

    #avgax = avg_incDF.plot.bar(title='The {} Income for the available occupations'.format(avgcmpr[1]), figsize=(figszz, figszz))
    #avgax.set_ylim((avg_incDF.min(axis=1)[0] - offset, avg_incDF.max(axis=1)[0] + offset))

    #mid = int(len(comparevars)/2)
    #high_avg = list(avg_inc.keys())[-mid:]
    #high_avg.reverse()
    #low_avg = list(avg_inc.keys())[0:mid]
    #low_avg.reverse()

    #high_med = list(med_inc.keys())[-mid:]
    #high_med.reverse()
    #low_med = list(med_inc.keys())[0:mid]
    #low_med.reverse()
    #return (high_avg, low_avg), (high_med, low_med)
    return

def collect_locale_states(df,  stat):
    plt_d = {}
    locale_d = {'locale_recode(rural)': 1, "locale_recode(suburban)": 3, 'locale_recode(town)': 2,
                'locale_recode(urban)': 4, }
    min, max = np.inf, -np.inf
    for l in list(locale_d.keys()):
        print('locale {}/{}'.format(l, locale_d[l]))
        ldf = df.loc[df['locale_dummy'] == locale_d[l], :].copy()
        ldf = ldf.filter(items=[stat, ])

        mu = ldf.mean()[0]
        if mu > max:
            max = mu
        if mu < min:
            min = mu
        print("mu: {}".format(mu))
        plt_d[l] = [mu]
        print('mu 2 ', plt_d[l])

    print(plt_d)
    axb = pd.DataFrame(plt_d).plot.bar(figsize=(20, 20))
    axb.set_title('{} average by Locale'.format(stat))
    axb.set_ylim([min - min * .1, max + max * .1])
    axb.legend()
    axb.grid(True)
    return
# income nrel_income stuff

def locale_selector(df, locale='locale_dummy', recoded_title='locale_recode(rural)', verbose=False):
          return df.loc[df[locale] == locale_recode_d[recoded_title], :]

def get_cross_val_set(data_df, target, select_model=[], tr=.75, stratify=True,):
    from sklearn.model_selection import train_test_split
    from _products.ML_Tools import show_missing
    if len(select_model) == 0:
        feats = data_df.columns.tolist()

    else:
        feats = select_model
    if target in feats:
        del feats[feats.index(target)]
    df0 = data_df.filter(items=feats,)
    targets0 = data_df.filter(items=[target])
    print('---------------------------------')
    print('the missing')
    show_missing(data_df)
    print('---------------------------------')

    ts = np.around(1 - tr, 2)

    if stratify:
        print('Cross validating with stratification')
        Xtr, Xts, ytr, yts = train_test_split(df0, targets0, test_size=ts, train_size=tr, stratify=targets0)
    else:
        print('Cross validating with no stratification')
        print('targets: {}'.format(target))
        Xtr, Xts, ytr, yts = train_test_split(df0, targets0, test_size=ts, train_size=tr,)

    # Pull out your predictors from your targets for the training set
    Xtr = pd.DataFrame(Xtr, columns=feats, )
    ytr = pd.DataFrame(np.array(ytr).flatten(), columns=[target])
    ytr = ytr.values.flatten()
    print(ytr)
    # Pull out your predictors from your targets for the testing set
    Xts = pd.DataFrame(Xts, columns=feats)
    yts = pd.DataFrame(np.array(yts).flatten(), columns=[target])
    yts = yts.values.flatten()
    #
    N_ts = Xts.shape
    N_tr = Xtr.shape

    print('There are {} Training samples'.format(N_tr))
    print('There are {} Training samples'.format(N_ts))
    print(ytr)

    print('Cross validation set created')
    return Xts, yts, Xtr, ytr

def print_df_like_dict(df):
    dd = dict(df)
    for k in df['vars']:
        print('{}:  {}'.format(k, df.loc[k, 'missing%']))
    return

def string_check(df):
    to_remove = list()
    for c in df.columns.tolist():
        if isinstance(df[c].values.tolist()[0], type('_')):
            print('"{}",'.format(c))
            to_remove.append(c)
    return to_remove

def strip_nrml(usecols):
    ret_cl = list()
    for s in usecols:
        if s != 'Adoption':
            ret_cl.append(s[:-6])
        else:
            ret_cl.append(s)
    return ret_cl

def rmv_target_from_solar_drops(target):
    solar_drops = list(Drop_Lists.expanded_solar_drops)
    del solar_drops[solar_drops.index(target)]
    return solar_drops


def load_model(target='Adoption', usecols=None, model_file=Model_List.DSMIX_13, ret_full=True,
               verbose=False, impute=True, heatmap=True, pdrops=None):
    # will load data frames for a training set and possibly a heat map
    import numpy as np
    df_base = None
    if pdrops is not None:
        udrops = pdrops
        if usecols is not None:
            # if pdrops is None:
            df_base = pd.read_excel(model_file, usecols=usecols + [target]).drop(
                columns=udrops)  # load data set with target
        else:
            df_base = pd.read_excel(model_file, ).drop(columns=udrops)  # load data set with target
    else:
        # df_base = pd.read_excel(model_dec_30_scld).drop(columns=['Anti_Occup_scld', 'E_DAYPOP_scld', 'E_MINRTY_scld',
        #                                                     'cust_cnt_scld', 'employ_rate_scld', ])
        # df_base = pd.read_excel(model_file).drop(columns=drops)
        if usecols is None:
            df_base = pd.read_excel(model_file)  # load data set with target
        else:
            df_base = pd.read_excel(model_file, usecols=usecols + [target])
            # df_base = pd.read_excel(model_file,).drop(columns=pdrops)
        return df_base

def generate_HML_set(df, level=0, metric='PV_per_100HuOwn', levels=[.33, .66]):
    if level == 0:
        q = df[metric].quantile(levels[0])
        return pd.DataFrame(df.loc[df[metric] <=q, :])
    elif level == 1:
        q = df[metric].quantile(levels[0])
        q2 = df[metric].quantile(levels[1])
        print('q1: {}, q2: {}'.format(q, q2))
        dfa =pd.DataFrame(df.loc[df[metric] > q, :])
        dfa =pd.DataFrame(dfa.loc[df[metric] <=q2, :])
        print('len {}'.format(len(dfa)))
        return dfa
    else:
        q2 = df[metric].quantile(levels[1])
        dfa = pd.DataFrame(df.loc[df[metric] > q2, :])
        print('len {}'.format(len(dfa)))
        return pd.DataFrame(df.loc[df[metric] > q2, :])

def get_DeepSolarNREL(regions=State_Sets.thirteen_state, usecol_list=[None, None], use_premade=False):
    if  use_premade:
        usecol_list = [Main_Usecols.nrel_variables,Main_Usecols.ds_variables,]
    #else:
    #    usecol_list = [None, None]
    NREL = pd.read_csv(Data_set_paths.DS_NREL_SVI_paths[1], usecols=usecol_list[0],
                       low_memory=False)
    DeepSolar = pd.read_csv(Data_set_paths.DS_NREL_SVI_paths[0], usecols=usecol_list[1],
                            low_memory=False)
    return DeepSolar, NREL


class DS_Mixed_Model_Generator:
    anti_jobs = ['occupation_agriculture_rate', 'occupation_construction_rate', 'occupation_transportation_rate',
                 'occupation_manufacturing_rate']
    pro_jobs = ['occupation_administrative_rate', 'occupation_information_rate', 'occupation_finance_rate',
                'occupation_arts_rate', 'occupation_education_rate']
    """ source for info below: https://www.energy.gov/maps/renewable-energy-production-state """
    ren = {"al": .1615, 'az': .1201, 'ca': 0.2436, 'ga': .3636, 'ma': 0.4272, 'ny': 0.4479, 'tx': 0.255, 'ut': 0.151,
           'ky': .235, 'ms': .117, 'nc': .2573, 'tn': .3521, 'va': .1054, }
    local_recode = {'Rural': 1, 'Town': 2, 'City': 4, 'Suburban': 3, 'Urban': 4}
    local_recodeA = {'Rural': 'Rural', 'Town': 'Town', 'City': 'City', 'Suburban': 'Suburban', 'Urban': 'City'}
    sought = ['Rural', 'Town', 'City', 'Suburban', 'Urban']

    table_type_options = ['xlsx', 'csv']

    to_log = vars_to_log
    def __init__(self, regions=None, target=None, destination='test.csv',
                 tva_area=False,
                 olrmvl=False,
                 scl_sub=False,
                 scl_type=None,
                 olrVar='PV_HuOwn',
                 stat_dest=None,
                 iscsv=False,
                 corre_dest=None,
                 scale_data=False,
                 impute_target=False, impTrgt="number_of_solar_system_per_household"):
        self.ds, self.nrel = get_DeepSolarNREL()
        self.excludes = []
        self.ignores = ['fips', 'state', 'county']
        self.target = target
        self.scl_type = scl_type
        self.scl_sub = scl_sub
        self.usecols = list()
        # TODO: get list of solar metrics to ignore for our targets
        self.solar_drops = Drop_Lists.expanded_solar_drops
        # TODO: we were given a region
        if regions is not None:
            self.ds = self.ds.loc[self.ds['state'].isin(regions)]
        else:
            print('The states:\n{}'.format(set(self.ds['state'].tolist())))
        if tva_area:
            tva_fips = FIPS_Lists.TVA_fips
            self.ds = self.ds.loc[self.ds['fips'].isin(tva_fips)]
        dsss = [self.ds, self.nrel]
        joins = ('fips', 'geoid')
        self.merged = self.data_merger(dsss, target='Adoption', joins=joins)
        # self.merged.replace([-999, np.inf, ""], np.nan, inplace=True)
        # self.merged.replace(np.inf, np.nan, inplace=True)
        #self.merged.replace('', np.nan, inplace=True)

        #self.generate_solar_metrics()  # generates some new solar metrics
        # if olrmvl:
        #    print('\n\n\n\t\t\t*****    Removing Outliers    ****\n\n\n')
        #    remove_outliers3(self.merged, threshold=3, vars_to_check=[olrVar], verbose=True)
        """
        if olrmvl:
            print('\n\n\n\t\t\t*****    Removing Outliers    ****\n\n\n')
            remove_outliers3(self.merged, threshold=3, vars_to_check=[olrVar], verbose=True)
        if impute_target:
            print('\n\n\nBefore: {}'.format(self.merged.shape[0]))
            self.merged.dropna(subset=[impTrgt], inplace=True)
            print('After: {}\n\n\n'.format(self.merged.shape[0]))
        """
        self.generate_green_travelers() # combine certain % types of travel
        self.mixed_travel_times( )
        # self.make_pro_anti_occu()       # combine jobs +/- correlated to adoption
        self.recode_locale_data()       # make numeric versions of the local labels
        self.add_binary_policies()      # make certain policies binary instead of year based
        self.add_renewable_gen()        # adds the renewable generation for the selected states
        self.combine_edu()              # combines certain types of education ranges/types
        self.recode_hh_size_counts_into_pct()  # convertes household size counts to percentages
        self.add_home_age_range_combos()       # creates combined home age ranges
        self.home_owner_mergers()              # converts home owners to %
        self.generate_mixed_policy()           # generates mixed policy variables
        self.gender__race_pct()                      # converts gender counts to %
        self.generate_solar_metrics()          # generates some new solar metrics
        if olrmvl:
            print('\n\n\n\t\t\t*****    Removing Outliers    ****\n\n\n')
            self.merged = remove_outliers3(self.merged, threshold=3, vars_to_check=[olrVar], verbose=True)

        if impute_target:
            print('\n\n\n\t\t\t*****  Imputing on: {}    ****'.format(impTrgt))
            print('\n\n\nBefore: {}'.format(self.merged.shape[0]))
            self.merged.dropna(subset=[impTrgt], inplace=True)
            print('After: {}\n\n\n'.format(self.merged.shape[0]))
        self.add_hotspots()                    # add binary indicators for CT's that are hot spots for per hh, huown, and avg size
        self.broaden_travel_times()            # creates mixed ranges of travel times
        self.age_range_mixing()                # creates mixed age ranges
        self.add_summed_seedsii_vars()
        self.make_state_dummies()
        self.add_regionalindicators()           # add binary indicators for west, mid west, south, and north east
        self.random_interactions()
        self.split_solarresources()
        self.ma_tribs = self.merged.columns.values.tolist()
        if scale_data:
            self.scale_data()                  # scale the data
        if corre_dest is not None:
            print('storing cree at {}'.format(corre_dest))
            self.store_target_corre(corre_dest)
        if stat_dest is not None:
            self.store_set_stats(stat_dest, iscsv=iscsv)
        print('there are {} entries.'.format(self.merged.shape))
        print('the merged solar---?')
        print(self.merged[solar_metrics])
        print("---------------------------------")
        print(self.merged['state'])
        #self.log_transform()   # log transform specific variables
        print("Dimensions: {}".format(self.merged.shape))
        for f in sorted(self.merged.columns.tolist()):
            print("{}".format(f))
        # self.merged.to_excel(destination, index=False)
        self.merged.to_csv(destination, index=False)


    def log_transform(self):
        log10_transform(self.merged, self.to_log)
        return

    def data_merger(self, data_sets, joins=('fips', 'FIPS', 'geoid'), target=None, verbose=False, drop_joins=False, ):
        """This method can be used to merge a set of data frames using a shared
           data column. the first argument is a list of the dataframes to merge
           and the second argument is a list of the column labels used to perform the merge
           TODO: some work needs to be done for error checking
           TODO: add more flexibility in how the merge is perfomed
           TODO: make sure the copy rows are removed
        :param data_sets: a list of data frames of the data sets that are to be joined
        :param joins: a list of the column labels used to merge, the labels should be in the s
                      same order as the data frames for the method to work. Right now this works
                      best if the label used is the same for all. This makes sured the duplicate
                       columns are not created.
        :param verbose: at this point does nothing but can be used to inform user of what
                        has occured
        :return: a reference to the new merged dataframe
        """
        cnt = 0
        if len(data_sets) == 1:
            return data_sets[0]
        for df in range(1, len(data_sets)):
            data_sets[0] = data_sets[0].merge(data_sets[df], left_on=joins[0], right_on=joins[df], how='left')
            if verbose:
                print(data_sets[0].columns)

            if (joins[0] + '_x') in data_sets[0].columns.values.tolist() or (
                    (joins[0] + '_y') in data_sets[0].columns.values.tolist()):
                data_sets[0].drop(columns=[(joins[0] + '_x'), (joins[1] + '_y')], inplace=True)
            if target is not None and ((target + '_x') in data_sets[0].columns.values.tolist() or (
                    (target + '_y') in data_sets[0].columns.values.tolist())):
                data_sets[0][target] = data_sets[0].loc[:, target + '_x']
                data_sets[0].drop(columns=[(target + '_x'), (target + '_y')], inplace=True)
        if drop_joins:
            data_sets[0].drop(columns=list(joins), inplace=True)
        return data_sets[0]

    def mixed_travel_times(self, ):
        low_travel = ["travel_time_10_19_rate","travel_time_less_than_10_rate",]
        create_combo_var_sum(self.merged, low_travel, newvar='low_commute_times')
        #self.merged['low_commute_times'] = self.merged[low_travel[0]].values + self.merged[low_travel[1]]

    def random_interactions(self):
        # add interaction between income and consumption
        # self.merged['Inc_x_Consmpt_kwh'] = self.merged['avg_monthly_consumption_kwh'].values * self.merged['median_household_income'].values
        # self.merged["Income_x_Suitable_m2"] = self.merged["total_own_devp"].values * self.merged['median_household_income'].values
        # self.merged['Income_x_EnergyCost'] = self.merged['dlrs_kwh'].values * self.merged['median_household_income'].values
        # self.merged['income_x_consumption_energyCost'] = self.merged['dlrs_kwh'].values * self.merged['median_household_income'].values * self.merged['avg_monthly_consumption_kwh'].values
        self.merged['Savings_potential'] = self.merged['daily_solar_radiation'].values * self.merged['total_own_devp'].values * self.merged['dlrs_kwh'].values
        self.merged['Income_x_College_Edu'] = self.merged["median_household_income"] * self.merged['education_college_rate']
        self.merged['Cost_x_Consumption'] = self.merged['dlrs_kwh'] * self.merged['avg_monthly_consumption_kwh']

        self.merged['urban_diversity'] = self.merged['URBAN'] * self.merged['diversity']
        self.merged['suburban_diversity'] = self.merged['locale_recode(suburban)'] * self.merged['diversity']
        self.merged['rural_diversity'] = self.merged['locale_recode(rural)'] * self.merged['diversity']

        # self.merged['popden_x_TotOK_cnt'] =self.merged['population_density'] * self.merged['total_own_Sbldg']
        # self.merged['popden_x_TotOK_RCnt'] =self.merged['population_density'] * self.merged['total_own_devpC']
        # self.merged['popden_x_TotOK_Rm2'] =self.merged['population_density'] * self.merged['total_own_devp']
        #
        # self.merged['ownership_x_TotOK_cnt'] =self.merged['hu_own_pct'] * self.merged['total_own_Sbldg']
        # self.merged['ownership_x_TotOK_Rcnt'] =self.merged['hu_own_pct'] * self.merged['total_own_devpC']
        # self.merged['ownership_x_TotOK_Rm2'] =self.merged['hu_own_pct'] * self.merged['total_own_devp']

        self.merged['Top_3_States'] = np.zeros(len(self.merged))

        self.merged.loc[self.merged['state'].isin(['ca', 'nv', 'az']), 'Top_3_States'] = 1


    def generate_green_travelers(self, merged=None, green_travelers=None):
        if merged is None:
            merged = self.merged
        if green_travelers is None:
            green_travelers = ['transportation_home_rate', 'transportation_bicycle_rate', 'transportation_walk_rate']
        create_combo_var_sum(merged, green_travelers, newvar='Green_Travelers')
        self.excludes += green_travelers + ['Green_Travelers']

    def make_pro_anti_occu(self, ):
        create_combo_var_sum(self.merged, self.anti_jobs, newvar='Anti_Occup')
        create_combo_var_sum(self.merged, self.pro_jobs, newvar='Pro_Occup')
        self.excludes += ['Anti_Occup', 'Pro_Occup', ]

    def check_table_type(self, table_name):
        if table_name[-3:] == 'csv':
            return 'csv'
        elif table_name[-4:] == 'xlsx':
            return 'xlsx'
        else:
            print('Unknown Table storage type for file: {}'.format(table_name))
            print('Terminating program')
            quit()

    def smart_table_opener(self, table_file, usecols=None, sheet='Sheet1'):
        if self.check_table_type(table_file) == self.table_type_options[0]:
            return pd.read_excel(table_file, usecols=usecols, )
        elif self.check_table_type(table_file) == self.table_type_options[1]:
            return pd.read_csv(table_file, usecols=usecols, low_memory=False,)

    def add_renewable_gen(self):
        #add_renewable_gen(self.merged, 'state', self.ren)
        #add_renewable_gen_df_df(self.merged, 'Ren', cold, cols)
        sourcefile = r'../_Data/Renewable_State_Info/renew_prod_2009.csv'
        add_renewable_gen_df_df(self.merged,
                                sourcefile=sourcefile,
                                cols=['renew_prod','hydro_prod','solar_prod',],
                                open_method=self.smart_table_opener,
                                fillins='STUSPS',
                                cold='state')
        self.excludes += ['Ren', 'renew_prod','hydro_prod','solar_prod',]

    def recode_locale_data(self):
        local = list(self.merged['locale'])
        self.merged['locale_dummy'] = recode_var_sub(self.sought, local, self.local_recode)
        self.merged['locale_recode'] = recode_var_sub(self.sought, local, self.local_recodeA)

        empty_1, empty_2, empty_3, empty_4 = np.zeros(self.merged.shape[0]), np.zeros(self.merged.shape[0]), np.zeros(
            self.merged.shape[0]),np.zeros(self.merged.shape[0])
        empty_1[self.merged['locale_dummy'].values == 1] = 1   # RURAL
        empty_2[self.merged['locale_dummy'].values == 2] = 1   # TOWN
        empty_3[self.merged['locale_dummy'].values == 3] = 1   # SUBURBAN
        empty_4[self.merged['locale_dummy'].values == 4] = 1   # URBAN
        self.merged['locale_recode(rural)'] = empty_1
        self.merged['locale_recode(suburban)'] = empty_3
        self.merged['locale_recode(town)'] = empty_2
        self.merged['locale_recode(urban)'] = empty_4

        # add a designation for town and urban called urban
        self.merged['URBAN'] = np.zeros(len(self.merged))
        self.merged['URBAN'] = self.merged['locale_recode(urban)'].values + self.merged['locale_recode(town)'].values
        #self.merged['URBAN'] = ds_HM['locale_recode(urban)'].values + ds_HM['locale_recode(town)'].values
        self.excludes += ['locale_recode', 'locale']
        self.ignores += ['locale_recode', 'locale']

    def combine_edu(self):
        """
                Used to create combo education variables
        :return:
        """
        high_above = ['education_less_than_high_school_rate', 'education_high_school_graduate_rate', 'education_master_rate', 'education_doctoral_rate'] + ['education_bachelor_rate']
        self.merged['education_high_school_or_above_rate'] = create_combo_var_sum(self.merged, high_above)
        high_below = ['education_less_than_high_school_rate', 'education_high_school_graduate_rate']
        self.merged['education_high_school_or_below_rate'] = create_combo_var_sum(self.merged, high_below)
        master_above = ['education_master_rate', 'education_doctoral_rate']
        self.merged['education_master_or_above_rate'] = create_combo_var_sum(self.merged, master_above)
        bachelor_above = ['education_master_rate', 'education_doctoral_rate'] + ['education_bachelor_rate']
        self.merged['education_bachelor_or_above_rate'] = create_combo_var_sum(self.merged, bachelor_above)
        self.merged['educated_population_rate'] = smart_df_divide(self.merged['pop25_some_college_plus'].values,
                                                                 self.merged['population'].values)

        self.excludes += high_below + master_above + bachelor_above + ['high_school_or_below_rate',
                                                                       'masters_or_above_rate',
                                                                       'bachelor_or_above_rate']


    def home_owner_mergers(self):
        self.merged['hu_own_pct'] = (self.merged['hu_own'] / self.merged['housing_unit_count']).values.tolist()
        self.excludes += ['hu_own_pct']

    def add_binary_policies(self):
        nm, pt, litx = 'net_metering_bin', 'property_tax_bin', 'lowincome_tax_credit_bin'
        thresh_binary_recode(self.merged, 'net_metering', )
        thresh_binary_recode(self.merged, 'property_tax', )
        self.merged['lowincome_tax_credit_bin'] = np.zeros(len(self.merged))
        self.merged.loc[self.merged['lihtc_qualified'] == 'FALSE','lowincome_tax_credit_bin' ] = 0
        self.merged.loc[self.merged['lihtc_qualified'] == 'TRUE','lowincome_tax_credit_bin' ] = 1
        self.merged['Policy_Combo'] = np.zeros(len(self.merged))
        self.merged.loc[self.merged[nm] == 1, 'Policy_Combo'] = 1
        self.merged.loc[self.merged[pt] == 1, 'Policy_Combo'] = 2
        self.merged.loc[self.merged[litx] == 1, 'Policy_Combo'] = 3

        self.merged.loc[[a + b == 2 for a, b in zip(self.merged[nm].tolist(), self.merged[pt].tolist())], 'Policy_Combo'] = 4
        self.merged.loc[[a + b == 2 for a, b in zip(self.merged[nm].tolist(), self.merged[litx].tolist())], 'Policy_Combo'] = 5
        self.merged.loc[[a + b == 2 for a, b in zip(self.merged[pt].tolist(), self.merged[litx].tolist())], 'Policy_Combo'] = 6
        self.merged.loc[[a + b + c == 3 for a, b, c in zip(self.merged[nm].tolist(), self.merged[pt].tolist(), self.merged[litx].tolist())], 'Policy_Combo'] = 7


        self.excludes += ['net_metering', 'property_tax', 'lowincome_tax_credit_bin']

    def add_home_age_range_combos(self):
        # make range from 1959 to earlier variable
        hage1959toearlier = ['hu_vintage_1940to1959', 'hu_vintage_1939toearlier']
        self.merged['hu_1959toearlier'] = create_combo_var_sum(self.merged, hage1959toearlier)
        # make 60 to 79 pct variable
        self.merged['hu_1959toearlier_pct'] = (
                    self.merged['hu_1959toearlier'] / self.merged['housing_unit_count']).values.tolist()

        # make 60 to 79 pct variable
        self.merged['hu_1960to1979_pct'] = (
                    self.merged['hu_vintage_1960to1970'] / self.merged['housing_unit_count']).values.tolist()

        # make 80 to 99 pct variable
        self.merged['hu_1980to1999_pct'] = (
                    self.merged['hu_vintage_1980to1999'] / self.merged['housing_unit_count']).values.tolist()

        # make list of variabels to sum to get range variable from 2000 to beyond
        hage2000tobeyond = ['hu_vintage_2000to2009', 'hu_vintage_2010toafter']
        self.merged['hu_2000toafter'] = create_combo_var_sum(self.merged, hage2000tobeyond)

        # make percentage variable out of new variable
        self.merged['hu_2000toafter_pct'] = (
                    self.merged['hu_2000toafter'] / self.merged['housing_unit_count']).values.tolist()
        self.merged['Mid_Agedwellings'] = self.merged["hu_vintage_1960to1970"].values + self.merged["hu_vintage_1980to1999"]
        self.excludes += ['hu_1980to1999_pct', 'hu_2000toafter', 'hu_1960to1979_pct', 'hu_1959toearlier_pct']

    def recode_hh_size_counts_into_pct(self):
        hh_sizes = ['hh_size_1', 'hh_size_2', 'hh_size_3', 'hh_size_4']
        # merged['hh_total'] = create_combo_var_sum(merged, hh_sizes, newvar=None)
        create_combo_var_sum(self.merged, hh_sizes, newvar='hh_total')

        percentage_generator(self.merged, hh_sizes[0], 'hh_total', newvar='%hh_size_1')
        percentage_generator(self.merged, hh_sizes[1], 'hh_total', newvar='%hh_size_2')
        percentage_generator(self.merged, hh_sizes[2], 'hh_total', newvar='%hh_size_3')
        percentage_generator(self.merged, hh_sizes[3], 'hh_total', newvar='%hh_size_4')
        self.excludes += ['%hh_size_1', '%hh_size_2', '%hh_size_3', '%hh_size_4']

    def gender__race_pct(self):
        female_count = 'pop_female'
        male_count = 'pop_male'
        total = 'pop_total'
        # merged[total] = create_combo_var_sum(merged, [female_count, male_count], newvar=total)
        # merged['%female'] = percentage_generator(merged, female_count, total)
        # merged['%male'] = percentage_generator(merged, male_count, total)
        create_combo_var_sum(self.merged, [female_count, male_count], newvar=total)
        percentage_generator(self.merged, female_count, total, newvar='female_pct')
        percentage_generator(self.merged, male_count, total, newvar='male_pct')
        self.merged['Gender_Ratio'] = smart_df_divide(self.merged['pop_female'].values,
                                                      self.merged['pop_male'].values)

        self.merged['political_ratio'] = (self.merged['voting_2012_dem_percentage'] *self.merged['population'])/(self.merged['voting_2012_gop_percentage'] *self.merged['population'])
        self.merged['white_pct'] = smart_df_divide(self.merged['pop_caucasian'].values,
                                                      self.merged['population'].values)
        self.merged['black_pct'] = smart_df_divide(self.merged['pop_african_american'].values,
                                                   self.merged['population'].values)
        self.merged['asian_pct'] = smart_df_divide(self.merged['pop_asian'].values,
                                                   self.merged['population'].values)
        self.merged['hispanic_pct'] = smart_df_divide(self.merged['pop_hispanic'].values,
                                                   self.merged['population'].values)

        self.excludes += ['female_pct', 'male_pct', ]

    def broaden_travel_times(self):
        trav_recodes = ['travel_time_40_59_rate', 'travel_time_60_89_rate']
        create_combo_var_sum(self.merged, trav_recodes, newvar='travel_time_40_89_rate')

        travM_recodes = ['travel_time_20_29_rate','travel_time_30_39_rate',]
        create_combo_var_sum(self.merged, trav_recodes, newvar='travel_time_20_39_rate')
        self.excludes += ['travel_time_40_89_rate','travel_time_20_39_rate' ] + trav_recodes + travM_recodes

    def age_range_mixing(self):
        age_25_44 = ['age_25_34_rate', 'age_35_44_rate']
        age_25_64 = ['age_25_34_rate', 'age_35_44_rate', 'age_45_54_rate', 'age_55_64_rate']
        age_55_85p = ['age_55_64_rate', 'age_65_74_rate','age_75_84_rate','age_more_than_85_rate',]
        age_minor = ["age_5_9_rate", "age_10_14_rate", "age_15_17_rate",]
        age_zoom = age_minor + ["age_18_24_rate",]

        a_25_44 = 'age_25_44_rate'
        a_25_64 = 'age_25_64_rate'
        a_55_more = 'age_55_or_more_rate'
        a_minor_rt = 'age_minor_rate'
        a_zoomer_rt = 'age_zoomer_rate'
        # merged[a_25_44] = create_combo_var_sum(merged, age_25_44, newvar=a_25_44)
        # merged[a_25_64] = create_combo_var_sum(merged, age_25_64)
        # merged[a_55_more] = create_combo_var_sum(merged, age_25_64)
        create_combo_var_sum(self.merged, age_25_44, newvar=a_25_44)
        create_combo_var_sum(self.merged, age_25_64, newvar=a_25_64)
        create_combo_var_sum(self.merged, age_55_85p, newvar=a_55_more)
        create_combo_var_sum(self.merged, age_minor, newvar=a_minor_rt)
        create_combo_var_sum(self.merged, age_zoom, newvar=a_zoomer_rt)

    def make_state_dummies(self):
        states = set(self.merged['state'].values.tolist())
        # for each state make an empty array of zeros of the needed size
        # and then fill in ones where the state is in states
        for st in states:
            self.merged[st] = np.zeros(len(self.merged))
            self.merged.loc[self.merged['state'] == st, st] = 1

    def add_summed_seedsii_vars(self,):

        caps_to_sum_own = [
            'very_low_mf_own_mw',
            'very_low_sf_own_mw',
            'low_mf_own_mw',
            'low_sf_own_mw',
            'mod_mf_own_mw',
            'mod_sf_own_mw',
            'mid_mf_own_mw',
            'mid_sf_own_mw',
            'high_mf_own_mw',
            'high_sf_own_mw',
        ]

        ann_gen = [
            'very_low_mf_own_mwh',
            'very_low_sf_own_mwh',
            'low_mf_own_mwh',
            'low_sf_own_mwh',
            'mod_mf_own_mwh',
            'mod_sf_own_mwh',
            'mid_mf_own_mwh',
            'mid_sf_own_mwh',
            'high_mf_own_mwh',
            'high_sf_own_mwh',
        ]
        high_income_gen =[
            'high_mf_own_mwh',
            'high_sf_own_mwh',]
        mid_income_gen=[
            'mid_mf_own_mwh',
            'mid_sf_own_mwh',
        ]
        mod_income_gen = [
            'mod_mf_own_mwh',
            'mod_sf_own_mwh',
        ]

        low_income_gen = [
            'low_mf_own_mwh',
            'low_sf_own_mwh',
        ]

        # use these to get percentage of that income levels hh owners
        high_owners_hh = ['high_mf_own_hh', 'high_sf_own_hh',]
        mid_owners_hh = ['mid_mf_own_hh', 'mid_sf_own_hh',]
        mod_owners_hh = ['mod_mf_own_hh', 'mod_sf_own_hh',]
        low_owners_hh = ['low_mf_own_hh', 'low_sf_own_hh',]
        verylow_owners_hh = ['very_low_mf_own_hh', 'very_low_sf_own_hh',]

        # make percentage based income_hh counts
        high_hh_r = 'high_hh_rate'
        mid_hh_r = 'mid_hh_rate'
        mod_hh_r = 'mod_hh_rate'
        low_hh_r = 'low_hh_rate'
        verylow_hh_r = 'very_low_hh_rate'

        verylow_income_gen = [
            'very_low_mf_own_mwh',
            'very_low_sf_own_mwh',
        ]

        own_hh_l = ['high_own_hh', 'mid_own_hh', 'mod_own_hh',  'low_own_hh',
                    'verylow_own_hh',]
        sfown_hh_l = ['high_sf_own_hh', 'mid_sf_own_hh', 'mod_sf_own_hh',
                      'low_sf_own_hh', 'very_low_sf_own_hh',]
        mfown_hh_l = ['high_mf_own_hh', 'mid_mf_own_hh', 'mod_mf_own_hh',
                      'low_mf_own_hh', 'very_low_mf_own_hh', ]
        # add total  high income home owners
        create_combo_var_sum(self.merged, high_owners_hh, newvar='high_own_hh')
        # add total  mid income home owners
        create_combo_var_sum(self.merged, mid_owners_hh, newvar='mid_own_hh')
        # add total  mod income home owners
        create_combo_var_sum(self.merged, mod_owners_hh, newvar='mod_own_hh')
        # add total  low income home owners
        create_combo_var_sum(self.merged, low_owners_hh, newvar='low_own_hh')
        # add total  very low income home owners
        create_combo_var_sum(self.merged, verylow_owners_hh, newvar='verylow_own_hh')
        # add total  income home owners
        create_combo_var_sum(self.merged, own_hh_l, newvar='total_own_hh')
        create_combo_var_sum(self.merged, sfown_hh_l, newvar='total_sf_own_hh')
        create_combo_var_sum(self.merged, mfown_hh_l, newvar='total_mf_own_hh')

        # now make rate based versions of owner counts broken into income levels
        #self.merged[high_hh_r] = self.merged['high_own_hh'].values/self.merged['total_own_hh'].values
        self.merged[high_hh_r] = smart_df_divide(self.merged['high_own_hh'].values,self.merged['total_own_hh'].values)
        #self.merged[high_hh_r].replace(np.nan, 0, inplace=True)

        #self.merged[mid_hh_r] = self.merged['mid_own_hh'].values/self.merged['total_own_hh'].values
        self.merged[mid_hh_r] = smart_df_divide(self.merged['mid_own_hh'].values, self.merged['total_own_hh'].values)
        #self.merged[mid_hh_r].replace(np.nan, 0, inplace=True)

        #self.merged[mod_hh_r] = self.merged['mod_own_hh'].values/self.merged['total_own_hh'].values
        self.merged[mod_hh_r] = smart_df_divide(self.merged['mod_own_hh'].values, self.merged['total_own_hh'].values)
        #self.merged[mod_hh_r].replace(np.nan, 0, inplace=True)

        #self.merged[low_hh_r] = self.merged['low_own_hh'].values/self.merged['total_own_hh'].values
        self.merged[low_hh_r] = smart_df_divide(self.merged['low_own_hh'].values, self.merged['total_own_hh'].values)
        #self.merged[low_hh_r].replace(np.nan, 0, inplace=True)

        #self.merged[verylow_hh_r] = self.merged['verylow_own_hh'].values/self.merged['total_own_hh'].values
        self.merged[verylow_hh_r] = smart_df_divide(self.merged['verylow_own_hh'].values, self.merged['total_own_hh'].values)
        #self.merged[verylow_hh_r].replace(np.nan, 0, inplace=True)

        # income level suitable building counts
        high_own_bldg = ['high_mf_own_bldg_cnt', 'high_sf_own_bldg_cnt', ]
        mid_own_bldg = ['mid_mf_own_bldg_cnt', 'mid_sf_own_bldg_cnt', ]
        mod_own_bldg = ['mod_mf_own_bldg_cnt', 'mod_sf_own_bldg_cnt', ]
        low_own_bldg = ['low_mf_own_bldg_cnt', 'low_sf_own_bldg_cnt', ]
        verylow_own_bldg = ['very_low_mf_own_bldg_cnt', 'very_low_sf_own_bldg_cnt', ]

        own_bldg_cnt = ['high_own_Sbldg', 'mid_own_Sbldg', 'mod_own_Sbldg', 'low_own_Sbldg',
                        'verylow_own_Sbldg',]

        # add total suitable  high income home owners
        create_combo_var_sum(self.merged, high_own_bldg, newvar='high_own_Sbldg')
        # add total  suitable mid income home owners
        create_combo_var_sum(self.merged, mid_own_bldg, newvar='mid_own_Sbldg')
        # add total  suitable mod income home owners
        create_combo_var_sum(self.merged, mod_own_bldg, newvar='mod_own_Sbldg')
        # add total  suitable low income home owners
        create_combo_var_sum(self.merged, low_own_bldg, newvar='low_own_Sbldg')
        # add total  suitable very low income home owners
        create_combo_var_sum(self.merged, verylow_own_bldg, newvar='verylow_own_Sbldg')
        # add total  suitable income home owners
        create_combo_var_sum(self.merged, own_bldg_cnt, newvar='total_own_Sbldg')

        # now make rate based versions
        #self.merged['high_own_Sbldg_rt'] = self.merged['high_own_Sbldg'].values / self.merged['total_own_Sbldg'].values
        self.merged['high_own_Sbldg_rt'] = smart_df_divide(self.merged['high_own_Sbldg'].values,
                                                           self.merged['total_own_Sbldg'].values)
        #self.merged['mid_own_Sbldg_rt'] = self.merged['mid_own_Sbldg'].values / self.merged['total_own_Sbldg'].values
        self.merged['mid_own_Sbldg_rt'] = smart_df_divide(self.merged['mid_own_Sbldg'].values,
                                                          self.merged['total_own_Sbldg'].values)
        #self.merged['mod_own_Sbldg_rt'] = self.merged['mod_own_Sbldg'].values / self.merged['total_own_Sbldg'].values
        self.merged['mod_own_Sbldg_rt'] = smart_df_divide(self.merged['mod_own_Sbldg'].values,
                                                          self.merged['total_own_Sbldg'].values)
        #self.merged['low_own_Sbldg_rt'] = self.merged['low_own_Sbldg'].values / self.merged['total_own_Sbldg'].values
        self.merged['low_own_Sbldg_rt'] = smart_df_divide(self.merged['low_own_Sbldg'].values,
                                                          self.merged['total_own_Sbldg'].values)
        #self.merged['verylow_own_Sbldg_rt'] = self.merged['verylow_own_Sbldg'].values / self.merged['total_own_Sbldg'].values
        self.merged['verylow_own_Sbldg_rt'] = smart_df_divide(self.merged['verylow_own_Sbldg'].values,
                                                              self.merged['total_own_Sbldg'].values)


        # add a total owner capacity variable
        create_combo_var_sum(self.merged, caps_to_sum_own, newvar='Tot_own_mw')
        # add a total owner annual generation variable
        create_combo_var_sum(self.merged, ann_gen, newvar='Yr_own_mwh')
        # add a total owner annual generation variable
        create_combo_var_sum(self.merged, high_income_gen, newvar='high_own_mwh')
        # add a total owner annual generation variable
        create_combo_var_sum(self.merged, mid_income_gen, newvar='mid_own_mwh')
        # add a total owner annual generation variable
        create_combo_var_sum(self.merged, mod_income_gen, newvar='mod_own_mwh')
        # add a total owner annual generation variable
        create_combo_var_sum(self.merged, low_income_gen, newvar='low_own_mwh')
        # add a total owner annual generation variable
        create_combo_var_sum(self.merged, verylow_income_gen, newvar='verylow_own_mwh')

        very_low_own_elep = ['very_low_sf_own_elep_hh', 'very_low_mf_own_elep_hh', ]
        low_own_elep = ['low_sf_own_elep_hh', 'low_mf_own_elep_hh', ]
        mod_own_elep = ['mod_sf_own_elep_hh', 'mod_mf_own_elep_hh', ]
        high_own_elep = ['high_sf_own_elep_hh', 'high_mf_own_elep_hh', ]


        # add a total owner annual generation variable
        create_combo_var_sum(self.merged, high_own_elep, newvar='high_own_elep_hh')
        self.merged['high_own_elep_hh'] = self.merged['high_own_elep_hh'].values/2

        # add a total owner annual generation variable
        create_combo_var_sum(self.merged, mod_own_elep, newvar='mod_own_elep_hh')
        self.merged['mod_own_elep_hh'] = self.merged['mod_own_elep_hh'].values / 2

        # add a total owner annual generation variable
        create_combo_var_sum(self.merged, low_own_elep, newvar='low_own_elep_hh')
        self.merged['low_own_elep_hh'] = self.merged['low_own_elep_hh'].values / 2

        # add a total owner annual generation variable
        create_combo_var_sum(self.merged, very_low_own_elep, newvar='verylow_own_elep_hh')
        self.merged['verylow_own_elep_hh'] = self.merged['verylow_own_elep_hh'].values / 2

        total_elp =['verylow_own_elep_hh', 'low_own_elep_hh','mod_own_elep_hh', 'high_own_elep_hh']
        create_combo_var_sum(self.merged, total_elp, newvar='total_own_elep')
        self.merged['total_own_elep'] = self.merged['total_own_elep'].values/len(total_elp)
        # now get the possible savings total and for each group
        self.merged['Yrl_savings_$'] = (self.merged['dlrs_kwh'] *1000)* self.merged['Yr_own_mwh']
        self.merged['Yrl_%_inc'] = self.merged['Yrl_savings_$']/self.merged["average_household_income"]
        self.merged['Yrl_%_$_kwh'] = self.merged['Yrl_savings_$']/self.merged['total_own_elep']


        # use these to get percentage of that income levels hh owners
        high_owners_devp = ['high_mf_own_devp_m2', 'high_sf_own_devp_m2', ]
        mid_owners_devp = ['mid_mf_own_devp_m2', 'mid_sf_own_devp_m2', ]
        mod_owners_devp = ['mod_mf_own_devp_m2', 'mod_sf_own_devp_m2', ]
        low_owners_devp = ['low_mf_own_devp_m2', 'low_sf_own_devp_m2', ]
        verylow_owners_devp = ['very_low_mf_own_devp_m2', 'very_low_sf_own_devp_m2', ]

        devp_own_tot = high_owners_devp + mid_owners_devp + mod_owners_devp + low_owners_devp + verylow_owners_devp

        # use these to get percentage of that income levels hh owners
        high_owners_devpC = ['high_mf_own_devp_cnt', 'high_sf_own_devp_cnt', ]
        mid_owners_devpC = ['mid_mf_own_devp_cnt', 'mid_sf_own_devp_cnt', ]
        mod_owners_devpC = ['mod_mf_own_devp_cnt', 'mod_sf_own_devp_cnt', ]
        low_owners_devpC = ['low_mf_own_devp_cnt', 'low_sf_own_devp_cnt', ]
        verylow_owners_devpC = ['very_low_mf_own_devp_cnt', 'very_low_sf_own_devp_cnt', ]

        devp_own_totC = high_owners_devpC + mid_owners_devpC + mod_owners_devpC + low_owners_devpC + verylow_owners_devpC

        # add total  high income home owners
        create_combo_var_sum(self.merged, high_owners_devp, newvar='high_own_devp')
        # add total  mid income home owners
        create_combo_var_sum(self.merged, mid_owners_devp, newvar='mid_own_devp')
        # add total  mod income home owners
        create_combo_var_sum(self.merged, mod_owners_devp, newvar='mod_own_devp')
        # add total  low income home owners
        create_combo_var_sum(self.merged, low_owners_devp, newvar='low_own_devp')
        # add total  very low income home owners
        create_combo_var_sum(self.merged, verylow_owners_devp, newvar='verylow_own_devp')
        # add total  income home owners
        create_combo_var_sum(self.merged, devp_own_tot, newvar='total_own_devp')
        create_combo_var_sum(self.merged, devp_own_totC, newvar='total_own_devpC')
    def generate_solar_metrics(self):
        popu_t = 'population'
        res_tot = 'solar_system_count_residential'
        pv_ara_res = 'total_panel_area_residential'
        pv_ara_nres = 'total_panel_area_nonresidential'
        h_unit = 'housing_unit_count'
        tot_araa = 'total_area'
        pv_own = 'PV_HuOwn'
        hu_own = 'hu_own'
        avg_PVres = 'AvgSres'
        pv_res_area = 'total_panel_area_residential'
        # adpt10hh, adopt25hh, adopt5hh = 'Adoption_10hh', 'Adoption_25hh', 'Adoption_5hh'
        # adpt10ho, adopt25ho, adopt5ho = 'Adoption_10ho', 'Adoption_25ho', 'Adoption_5ho'

        """ make solar population total  per captia"""
        self.merged['SRpcap'] = self.merged['solar_system_count_residential'] / (self.merged[popu_t])
        self.merged['SNRpcap'] = self.merged['solar_system_count_nonresidential'] / (self.merged[popu_t])
        self.merged['ST_pcap'] = self.merged['solar_system_count'] / (self.merged[popu_t])

        """ make solar area per area """
        self.merged['SRaPa'] = self.merged[pv_ara_res] / self.merged[tot_araa]
        self.merged['SNRaPa'] = self.merged[pv_ara_nres] / self.merged[tot_araa]

        """ make solar area per capita """
        self.merged['SRaPcap'] = self.merged[pv_ara_res] / (self.merged[popu_t])
        self.merged['SNRaPcap'] = self.merged[pv_ara_nres]/ (self.merged[popu_t])

        """ make solar per home owner"""
        self.merged[pv_own] = self.merged[res_tot] / self.merged[hu_own]
        self.merged.loc[self.merged['PV_HuOwn'].isna(), 'PV_HuOwn'] = 0
        self.merged['PV_per_100_HuOwn'] = self.merged[pv_own] * 100
        self.merged.loc[self.merged['PV_per_100_HuOwn'].isna(), 'PV_per_100_HuOwn'] = 0


        """ make Adoption metric based on mean of per houshold rate"""
        # self.merged['Adoption_50'] = list([0]*len(self.merged))
        # self.merged['Adoption_25'] = list([0]*len(self.merged))
        # Q50 = self.merged['PV_per_100_HuOwn'].quantile(.5)
        # Q25 = self.merged['PV_per_100_HuOwn'].quantile(.25)
        # self.merged.loc[self.merged['PV_per_100_HuOwn'] >= Q50,'Adoption_50' ] = 1

        """ make Adoption metrics at 5, 10, and 25% percentile rates for per homeowner (ho) and per household (hh) """
        # Q27ho = self.merged['PV_HuOwn'].quantile(.27)
        # Q30ho = self.merged['PV_HuOwn'].quantile(.30)
        # Q35ho = self.merged['PV_HuOwn'].quantile(.35)
        # Q45ho = self.merged['PV_HuOwn'].quantile(.45)
        # Q50ho = self.merged['PV_HuOwn'].quantile(.5)
        #
        # Q27hh = self.merged['number_of_solar_system_per_household'].quantile(.27)
        # Q30hh = self.merged['number_of_solar_system_per_household'].quantile(.30)
        # Q35hh = self.merged['number_of_solar_system_per_household'].quantile(.35)
        # Q45hh = self.merged['number_of_solar_system_per_household'].quantile(.45)
        # Q50hh = self.merged['number_of_solar_system_per_household'].quantile(.50)
        #
        # self.merged['Adoption_26hh'] = list([0] * len(self.merged))
        # self.merged['Adoption_30hh'] = list([0] * len(self.merged))
        # self.merged['Adoption_35hh'] = list([0] * len(self.merged))
        # self.merged['Adoption_45hh'] = list([0] * len(self.merged))
        # self.merged['Adoption_50hh'] = list([0] * len(self.merged))
        #
        # self.merged['Adoption_26ho'] = list([0] * len(self.merged))
        # self.merged['Adoption_30ho'] = list([0] * len(self.merged))
        # self.merged['Adoption_35ho'] = list([0] * len(self.merged))
        # self.merged['Adoption_45ho'] = list([0] * len(self.merged))
        # self.merged['Adoption_50ho'] = list([0] * len(self.merged))

        # self.merged.loc[self.merged['PV_HuOwn'] > Q27ho, 'Adoption_27ho'] = 1
        # self.merged.loc[self.merged['PV_HuOwn'] > Q30ho, 'Adoption_30ho'] = 1
        # self.merged.loc[self.merged['PV_HuOwn'] > Q35ho, 'Adoption_35ho'] = 1
        # self.merged.loc[self.merged['PV_HuOwn'] > Q45ho, 'Adoption_45ho'] = 1
        # self.merged.loc[self.merged['PV_HuOwn'] > Q50ho, 'Adoption_50ho'] = 1
        #
        # self.merged.loc[self.merged['number_of_solar_system_per_household'] > Q27hh, 'Adoption_27hh'] = 1
        # self.merged.loc[self.merged['number_of_solar_system_per_household'] > Q30hh, 'Adoption_30hh'] = 1
        # self.merged.loc[self.merged['number_of_solar_system_per_household'] > Q35hh, 'Adoption_35hh'] = 1
        # self.merged.loc[self.merged['number_of_solar_system_per_household'] > Q45hh, 'Adoption_45hh'] = 1
        # self.merged.loc[self.merged['number_of_solar_system_per_household'] > Q50hh, 'Adoption_50hh'] = 1


        #self.merged.loc[self.merged['PV_per_100_HuOwn'] >= Q25,'Adoption_' ] = 1

        #self.merged['PV_per_100_HuOwnB'] = self.merged[res_tot] / (self.merged[hu_own]/100)
        """ make average solar panel installation in m^2"""
        rll = list([])
        for ara, cnt in zip(self.merged[pv_res_area].values.tolist(), self.merged[res_tot].values.tolist()):
            if cnt == 0:
                rll.append(0)
            else:
                rll.append(ara/cnt)
        self.merged[avg_PVres] = rll
        # fill the values that were too small with zeros
        # self.merged[avg_PVres] = self.merged[avg_PVres].fillna(0)

        self.excludes += ['SNRaPa', 'ST_pcap', 'SNRpcap', 'SRpcap', 'SRaPa', 'SRaPcap', 'SNRaPcap',
                          pv_own, avg_PVres]


    def add_hotspots(self, ):
        ds_df = self.merged
        # add the hot spots
        add_HOTSPOTS(df=ds_df, var="number_of_solar_system_per_household", new_var_name='Hot_Spots_hh',
                     percentile=.949, verbose=True)

        add_HOTSPOTS(df=ds_df, var='PV_HuOwn', new_var_name='Hot_Spots_hown',
                     percentile=.949, verbose=True)

        add_HOTSPOTS(df=ds_df, var='AvgSres', new_var_name='Hot_Spots_AvgAr',
                     percentile=.949, verbose=True)
        # this below line is completely uneeded and should be removed
        self.merged = ds_df

        return


    def generate_mixed_policy(self):
        net_own = ['net_metering_bin', 'hu_own_pct']
        new_net = 'net_metering_hu_own'
        generate_mixed(self.merged, net_own, new_net)

        ptax_own = ['property_tax_bin', 'hu_own_pct']
        new_ptax = 'property_tax_hu_own'
        generate_mixed(self.merged, ptax_own, new_ptax)

        incent_res_own = ['incentive_count_residential', 'hu_own_pct']
        new_incent_own = 'incent_cnt_res_own'
        generate_mixed(self.merged, incent_res_own, new_incent_own)

        # incent_med_income = ['incentive_residential_state_level', 'median_household_income' ]
        # incent_state_income = 'incent_st_Mincome'
        # generate_mixed(merged, incent_med_income, incent_state_income)

        # incent_avg_income = ['incentive_residential_state_level', 'average_household_income' ]
        # incent_state_Aincome = 'incent_st_Aincome'
        # generate_mixed(merged, incent_avg_income, incent_state_Aincome)

        med_income_ebill = ['avg_monthly_bill_dlrs', 'median_household_income']
        medincebill = 'med_inc_ebill_dlrs'
        generate_mixed(self.merged, med_income_ebill, medincebill)

        avg_income_ebill = ['avg_monthly_bill_dlrs', 'average_household_income']
        avgincebill = 'avg_inc_ebill_dlrs'
        generate_mixed(self.merged, avg_income_ebill, avgincebill)

        med_income_ebill = ['dlrs_kwh', 'median_household_income']
        medincecost = 'dlrs_kwh x median_household_income'
        generate_mixed(self.merged, med_income_ebill, medincecost)

        own_popden = ['population_density', 'hu_own_pct']
        ownpopden = 'own_popden'
        generate_mixed(self.merged, own_popden, ownpopden)
        self.excludes += [new_net, new_incent_own]

    def scale_data(self):
        if self.scl_type is None:
            # grab the variables to scale
            # 1) get a list of all the variables to ignore or not scale
            rmv_scl = list(set(self.ignores + self.solar_drops + Drop_Lists.stripped_excludes))

            # 2) use that list to remove them from the variable list to scale
            scalables = rmv_list_list(self.ma_tribs, rmv_scl)
            self.usecols = scalables + [self.target]  # this assumes Adoption is target
            self.merged = self.merged.loc[:, self.usecols]
        sclr = None
        # below calls sklearns scalers
        if self.scl_type is None:
            return
        elif self.scl_type in ['_nrml_', '_Z_', ]:
            if self.scl_type == '_nrml_':
                sclr = MinMaxScaler()
            elif self.scl_type == '_Z_':
                #sclr = G_Z_scaler()
                sclr = StandardScaler()
        else:
            print('unknown scale type {}\noptions are: _nrml_ (min max) or _Z_ (z score)'.format(self.scl_type))
            quit(-99)
        #rmv_scl = list(set(self.excludes + self.solar_drops + Drop_Lists.basic_excludes))
        rmv_scl = list(set(self.ignores + self.solar_drops + Drop_Lists.stripped_excludes))
        scalables = rmv_list_list(self.ma_tribs, rmv_scl)
        scldf = self.merged.loc[:, scalables]       # store a data frame of the scalables
        self.merged.drop(columns=scalables, inplace=True) # remove the originals
        #nscalables = [s + self.scl_type for s in scalables]
        nscalables = scalables
        # use selected scaler to scale the entire dataset
        scldf = pd.DataFrame(sclr.fit_transform(scldf), columns=nscalables, index=self.merged.index.values.tolist())
        self.merged = self.merged.join(scldf, lsuffix='', rsuffix=self.scl_type) # merge them back into one
        # merged.index = nscalables
        if self.scl_sub:
            #self.merged.drop(columns=scalables, inplace=True)
            self.usecols = scldf.columns.tolist() + [self.target]
            self.merged = self.merged.loc[:, self.usecols]
            #self.merged = self.merged.loc[:, rmv_scl + scldf.columns.tolist()]
        else:
            self.merged = self.merged.loc[:, rmv_scl + scldf.columns.tolist() + scalables]

    def store_target_corre(self, corre_dest):
        from _products._Data_Analysis import generate_kendal_spearman_corre_table
        usecols =self.merged.columns.tolist()
        print('usecols len', len(usecols))
        print('merged len', self.merged.shape[1])
        generate_kendal_spearman_corre_table(self.merged[usecols].dropna(), target='Adoption', usecols=usecols).to_csv(corre_dest)
        return

    def store_set_stats(self, stat_dest, iscsv=False):
        mdf = report_var_stats(pd.DataFrame(self.merged.copy()),
                               name=stat_dest, csv=iscsv)

    def add_regionalindicators(self, ):
        tot = 0
        ds_df = self.merged
        # adds some classification variables for census tracts based on some regions
        for reg in major4_reg:
            print('\t\t{}:'.format(reg))
            add_allval(ds_df, val=0, name=reg)
            ds_df.loc[ds_df['state'].isin(US_4Major[reg]), reg] = 1
            sz = len(ds_df.loc[ds_df[reg] == 1, :])
            tot += sz
        self.merged = ds_df
        return

    def split_solarresources(self):
        # add designations for the higher and lower than average daily solar areas
        self.merged['High_Solar_Areas'] = np.full(len(self.merged), 0.0)
        self.merged.loc[self.merged['daily_solar_radiation'] > self.merged['daily_solar_radiation'].mean(), 'High_Solar_Areas'] = 1
        self.merged['Low_Solar_Areas'] = np.full(len(self.merged), 0.0)
        self.merged.loc[self.merged['daily_solar_radiation'] < self.merged['daily_solar_radiation'].mean(), 'Low_Solar_Areas'] = 1
        self.merged['DS_HighSolar'] = np.zeros(len(self.merged))
        self.merged.loc[self.merged['daily_solar_radiation'] > 4.5,'DS_HighSolar'] = 1

class FIPS_Lists:
    # TVA_fips = pd.read_csv(r'C:\Users\gjone\DeepSolar_Convergence\_Data\Selectors\TVA_fips.csv')['fips'].values.tolist()
    TVA_fips = [1083020802, 1083021100, 1083021200, 1083020500, 1083020801, 1083020900, 1083021000, 1083020101,
                1083020102,
                1083020401, 1083020402, 1083020201, 1083020202, 1083020300, 1083020600, 1083020700, 1071950600,
                1071950100,
                1071950200, 1071950300, 1071950400, 1071950500, 1071950700, 1071950800, 1071950900, 1071951000,
                1071951100,
                1055010502, 1055010700, 1055010900, 1055011001, 1055011002, 1055010602, 1055010800, 1015002300,
                1015002200,
                1015002400, 1015002502, 1089011013, 1089011014, 1089000501, 1089011012, 1089000901, 1089000902,
                1089002000,
                1089001301, 1089001302, 1089000201, 1089000202, 1089000301, 1089000502, 1089000602, 1089000701,
                1089000702,
                1089001000, 1089001200, 1089001401, 1089001402, 1089001500, 1089001700, 1089001901, 1089001902,
                1089001903,
                1089002600, 1089002722, 1089002911, 1089002921, 1089002922, 1089003000, 1089003100, 1089002200,
                1089000503,
                1089000302, 1089000403, 1089010100, 1089010200, 1089010301, 1089010302, 1089010401, 1089010402,
                1089010501,
                1089010502, 1089000601, 1089010612, 1089010621, 1089010622, 1089010623, 1089010701, 1089010702,
                1089010901,
                1089011011, 1089011021, 1089011022, 1089011100, 1089011200, 1089011300, 1089011400, 1089001801,
                1089002100,
                1089002501, 1089002502, 1089002701, 1089002721, 1089002802, 1089002912, 1089002801, 1089002400,
                1089002300,
                1089010624, 1089010800, 1089010902, 1077011102, 1077011801, 1077011200, 1077010200, 1077010300,
                1077011101,
                1077011700, 1077010100, 1077010900, 1077011000, 1077010600, 1077011300, 1077011400, 1077011501,
                1077011502,
                1077011602, 1077011603, 1077011604, 1077011802, 1077010700, 1077010800, 1077010400, 1127020800,
                1079979200,
                1079979700, 1079979800, 1079979900, 1079979100, 1079979300, 1079979400, 1079979500, 1079979600,
                1049960100,
                1049960500, 1049960600, 1049960800, 1049960700, 1049960200, 1049960900, 1049961100, 1049961400,
                1049960300,
                1049961000, 1049961200, 1049960400, 1049961300, 1059972900, 1059973600, 1059973400, 1059973500,
                1059973700,
                1059973000, 1059973100, 1059973300, 1059973200, 1103000100, 1103000200, 1103000300, 1103000400,
                1103000600,
                1103000700, 1103000800, 1103000900, 1103001000, 1103005101, 1103005103, 1103005105, 1103005106,
                1103005107,
                1103005108, 1103005109, 1103005200, 1103005301, 1103005302, 1103005304, 1103005404, 1103005405,
                1103005500,
                1103005303, 1103005600, 1103005701, 1103005702, 1093964200, 1033020100, 1033020300, 1033020500,
                1033020600,
                1033020703, 1033020801, 1033020802, 1033020901, 1033020902, 1033021000, 1033020200, 1033020400,
                1033020701,
                1033020704, 1043964200, 1043964700, 1043965200, 1043965100, 1043964300, 1043964400, 1043964800,
                1043964900,
                1043965300, 1043965401, 1043965402, 1043965500, 1043965600, 1043965700, 1043964500, 1043964600,
                1043965000,
                1043964100, 1095030100, 1095030201, 1095030202, 1095030401, 1095030402, 1095030500, 1095030600,
                1095030701,
                1095030702, 1095030902, 1095030903, 1095030801, 1095030300, 1095030802, 1095030904, 1095031100,
                1095031200,
                1095031000, 1009050300, 1009050400, 1009050500, 1009050602, 1009050601, 1133965501, 1133965502,
                1133965503,
                1133965600, 1029959500, 1019955701, 1019955702, 1019955800, 1019955900, 1019956100, 1019956000,
                13295020100,
                13295020400, 13295020602, 13295020700, 13295020800, 13295020901, 13295020902, 13295020200, 13295020501,
                13295020502, 13295020601, 13295020301, 13295020302, 13129970400, 13129970800, 13129970100, 13129970200,
                13129970900, 13129970300, 13129970500, 13129970600,
                13129970700, 13083040101, 13083040200, 13083040300, 13281960100, 13281960200, 13281960300, 13115000202,
                13115000300, 13311950100, 13311950203, 13313000101,
                13313000102, 13313000200, 13313000301, 13313000400, 13313000501, 13313000502, 13313000600, 13313000700,
                13313000900, 13313001000, 13313001100, 13313001200,
                13313001300, 13313001400, 13313001500, 13313000302, 13313000800, 13137000202, 13187960102, 13187960202,
                13123080100, 13123080200, 13085970100, 13015960101,
                13015960200, 13241970201, 13241970301, 13111050500, 13111050400, 13111050100, 13111050200, 13111050300,
                13213010201, 13213010202, 13213010400, 13213010600,
                13213010700, 13213010500, 13213010100, 13055010300, 13055010100, 13055010600, 13055010200, 13055010400,
                13055010500, 13291000101, 13291000102, 13291000203,
                13291000204, 13291000205, 13291000201, 13047030402, 13047030201, 13047030301, 13047030500, 13047030600,
                13047030304, 13047030401, 13047030100, 13047030303,
                13047030202, 13047030700, 21171930100, 21171930200, 21171930300, 21171930400, 21177960100, 21177960400,
                21177960600, 21177960700, 21177960800, 21177960900,
                21013960500, 21013961100, 21013960400, 21013960600, 21085950100, 21085950200, 21085950300, 21085950400,
                21085950600, 21085950700, 21085950500, 21107971300,
                21099970300, 21099970200, 21053970201, 21053970202, 21053970100, 21147960400, 21033920300, 21139040200,
                21083020400, 21083020900, 21083020300, 21083020500,
                21083020600, 21083020700, 21083020800, 21083020100, 21083020200, 21095970300, 21095971000, 21095971300,
                21003920100, 21003920200, 21003920300, 21003920600,
                21003920400, 21003920500, 21145030100, 21145030200, 21145030400, 21145031301, 21145030300, 21145030700,
                21145030600, 21145030800, 21145030900, 21145031000,
                21145030500, 21145031100, 21145031200, 21145031302, 21145031400, 21141960100, 21141960200, 21141960600,
                21141960300, 21141960400, 21141960500, 21221980100, 21221970100, 21221970200, 21221970300, 21221980200,
                21061980100, 21061920200, 21061920300, 21061920400, 21039960100, 21039960200, 21039960300, 21047201501,
                21047200200, 21047200100, 21047200300, 21047200400, 21047200500, 21047200600, 21047200700, 21047200800,
                21047200900, 21047201000, 21047201200, 21047201301, 21047201302, 21047201400, 21047201502, 21047201503,
                21047201100, 21047980100, 21009950200, 21009950300, 21009950400, 21009950500, 21009950600, 21009950700,
                21009950800, 21009950900, 21009951000, 21235920600, 21235920700, 21235920800, 21183920100, 21183920300,
                21183920500, 21183920600, 21183920200, 21183920400, 21227010100, 21227010200, 21227010300, 21227010400,
                21227010600, 21227010701, 21227010702, 21227010801, 21227010802, 21227010803, 21227011001, 21227011002,
                21227011200, 21227011300, 21227011402, 21227011500, 21227011600, 21227011700, 21227011800, 21227011900,
                21227010500, 21227010900, 21227011100, 21227011401, 21093001600, 21093001700, 21001970500, 21001970600,
                21075960100, 21075960200, 21157950500, 21157950600, 21157950100, 21157950200, 21157950300, 21157950400,
                21219950100, 21219950200, 21219950300, 21219950400, 21035010200, 21035010301, 21035010302, 21035010400,
                21035010500, 21035010600, 21035010700, 21035010800, 21035010100, 21057950100, 21057950200, 21213970200,
                21213970100, 21213970300, 21213970400, 21231920400, 21231920700, 21027960502, 21143960100, 21143960200,
                21143980100, 21105970100, 21007950300, 21169960200, 21169960300, 21031930100, 21031930200, 21031930300,
                21031930400, 21031930500, 21207960200, 28143950100, 28141950100, 28141950300, 28141950400, 28141950200,
                28095950300, 28095950400, 28095950502, 28095950600, 28095950100, 28095950200, 28033070812, 28033070830,
                28033070900, 28033070101, 28033070520, 28033070620, 28033070630, 28033070710, 28033070721, 28033070722,
                28033070811, 28033070821, 28033070822, 28033071120, 28033071200, 28003950500, 28003950100, 28003950200,
                28003950300, 28003950400, 28003950600, 28003950700, 28117950300, 28117950100, 28117950500, 28117950200,
                28117950400, 28081951100, 28081950101, 28081950102, 28081950201, 28081950202, 28081950402, 28081950601,
                28081950602, 28081950901, 28081950902, 28081951001, 28081951002, 28081980000, 28081950301, 28081950302,
                28081950401, 28081950500, 28081950700, 28081950800, 28057950100, 28057950200, 28057950400, 28057950500,
                28057950300, 28119950100, 28119950300, 28135950100, 28135950300, 28135950400, 28135950200, 28137950100,
                28137950200, 28137950301, 28137950302, 28137950400, 28013950400, 28013950100, 28013950200, 28013950300,
                28071950100, 28071950401, 28071950201, 28071950202, 28071950301, 28071950302, 28071950402, 28071950501,
                28071950502, 28071950503, 28145950100, 28145950200, 28145950300, 28145950400, 28145950500, 28145950600,
                28161950100, 28161950200, 28161950300, 28107950100, 28107950200, 28107950300, 28107950400, 28107950600,
                28107950500, 28115950101, 28115950102, 28115950200, 28115950300, 28115950400, 28115950500, 28139950100,
                28139950200, 28139950300, 28139950400, 28093950100, 28093950200, 28093950402, 28093950300, 28093950401,
                28093950500, 28017950200, 28017950400, 28017950100, 28017950300, 28009950100, 28009950200, 37121950100,
                37189920900, 37189921000, 37189920200, 37189920100, 37115010100, 37115010200, 37023020201, 37111970100,
                37039930601, 37039930602, 37039930200, 37039930300, 37039930400, 37039930500, 37075920100, 37075920300,
                37087920102, 37087980100, 37043950200, 37043950100, 37011930200, 37011930400, 37011930301, 37011930302,
                37011930100, 37009970300, 47179060600, 47179060800, 47179061601, 47179061200, 47179060400, 47179061000,
                47179061300, 47179060501, 47179060502, 47179060700, 47179061100, 47179061401, 47179061402, 47179061500,
                47179061602, 47179061701, 47179061702, 47179061800, 47179061901, 47179061902, 47179062000, 47179060900,
                47179060100, 47049965200, 47049965300, 47049965000, 47049965100, 47019071100, 47019070100, 47019070800,
                47019071200, 47019070200, 47019070300, 47019070900, 47019071000, 47019071600, 47019070400, 47019070500,
                47019070600, 47019070700, 47019071300, 47019071400, 47019071500, 47019071700, 47085130200, 47085130300,
                47085130400, 47085130500, 47085130100, 47043060602, 47043060100, 47043060200, 47043060401, 47043060502,
                47043060700, 47043060300, 47043060501, 47043060601, 47043060402, 47151975100, 47151975200, 47151975300,
                47151975400, 47151975000, 47113000800, 47113000900, 47113001609, 47113000100, 47113000300, 47113000400,
                47113000500, 47113000600, 47113000700, 47113001000, 47113001100, 47113001300, 47113001401, 47113001402,
                47113001501, 47113001502, 47113001603, 47113001605, 47113001606, 47113001607, 47113001610, 47113001700,
                47113001800, 47113001900, 47113001604, 47113001608, 47113000200, 47121960100, 47121960200, 47121960300,
                47005963000, 47005963100, 47005963200, 47005963400, 47005963300, 47067960500, 47067960600, 47145030900,
                47145030500, 47145030100, 47145030800, 47145030400, 47145030300, 47145030600, 47145030700, 47145030201,
                47145030202, 47145980100, 47177930100, 47177930300, 47177930400, 47177930500, 47177930800, 47177930900,
                47177930200, 47177930600, 47177930700, 47065001200, 47065001400, 47065010700, 47065010901, 47065010902,
                47065011326, 47065011002, 47065011321, 47065000600, 47065000700, 47065000800, 47065001300, 47065001600,
                47065001800, 47065002000, 47065002400, 47065002500, 47065002600, 47065002800, 47065002900, 47065010101,
                47065010103, 47065010104, 47065010201, 47065010202, 47065010303, 47065010304, 47065010305, 47065010306,
                47065010411, 47065010412, 47065010413, 47065010431, 47065010432, 47065010433, 47065010434, 47065010435,
                47065010501, 47065010502, 47065010600, 47065010800, 47065010903, 47065011001, 47065011100, 47065011201,
                47065011203, 47065011204, 47065011311, 47065011314, 47065011323, 47065011324, 47065011325, 47065011402,
                47065011411, 47065011413, 47065011442, 47065011443, 47065011444, 47065011446, 47065011447, 47065011600,
                47065011700, 47065011800, 47065011900, 47065012000, 47065012100, 47065003400, 47065012200, 47065012400,
                47065012300, 47065011445, 47065000400, 47065001100, 47065010307, 47065001900, 47065003000, 47065003100,
                47065003200, 47065003300, 47065002300, 47065980100, 47065980200, 47159975100, 47159975200, 47159975300,
                47159975400, 47159975000, 47147080605, 47147080104, 47147080402, 47147080101, 47147080103, 47147080200,
                47147080301, 47147080302, 47147080401, 47147080603, 47147080604, 47147080606, 47147080700, 47147080500,
                47093000901, 47093000902, 47093001800, 47093002000, 47093002100, 47093002600, 47093002700, 47093002800,
                47093002900, 47093003100, 47093003200, 47093003400, 47093003902, 47093004000, 47093004100, 47093004200,
                47093004300, 47093004401, 47093004403, 47093004606, 47093004607, 47093004610, 47093004611, 47093004613,
                47093004614, 47093004615, 47093005000, 47093005302, 47093005501, 47093005502, 47093005603, 47093005707,
                47093005708, 47093005711, 47093005712, 47093005803, 47093005807, 47093005808, 47093005809, 47093005810,
                47093005811, 47093005812, 47093005813, 47093005903, 47093005905, 47093005906, 47093005907, 47093006002,
                47093006003, 47093006102, 47093006103, 47093006202, 47093006203, 47093006207, 47093006208, 47093006301,
                47093006302, 47093006401, 47093006402, 47093006403, 47093006501, 47093006502, 47093004608, 47093005301,
                47093005602, 47093005604, 47093000100, 47093001600, 47093001400, 47093001500, 47093001700, 47093006104,
                47093001900, 47093000800, 47093002200, 47093002400, 47093003300, 47093003802, 47093003901, 47093004404,
                47093006600, 47093004500, 47093004609, 47093002300, 47093004700, 47093004800, 47093004900, 47093005201,
                47093005202, 47093005701, 47093005704, 47093005706, 47093005709, 47093005710, 47093006900, 47093005904,
                47093005908, 47093006001, 47093006205, 47093006206, 47093007000, 47093006700, 47093006800, 47093007100,
                47093005401, 47093003000, 47093004612, 47093005402, 47093005100, 47093003500, 47093003801, 47093003700,
                47157020532, 47157020700, 47157020820, 47157021312, 47157021320, 47157021352, 47157021741, 47157000100,
                47157022111, 47157002500, 47157000400, 47157000600, 47157000700, 47157000800, 47157000900, 47157001100,
                47157001200, 47157001300, 47157001400, 47157001500, 47157001600, 47157001700, 47157001900, 47157002000,
                47157002400, 47157002600, 47157002700, 47157002800, 47157003000, 47157003100, 47157003200, 47157003300,
                47157003400, 47157003500, 47157003600, 47157003700, 47157003800, 47157003900, 47157004200, 47157004300,
                47157004500, 47157004600, 47157005000, 47157005300, 47157005500, 47157005600, 47157005700, 47157005800,
                47157005900, 47157006000, 47157006200, 47157006300, 47157006400, 47157006500, 47157006600, 47157006700,
                47157006800, 47157007100, 47157007300, 47157007810, 47157007821, 47157007822, 47157008110, 47157008500,
                47157980200, 47157009200, 47157009300, 47157009500, 47157009700, 47157010110, 47157010120, 47157010210,
                47157010220, 47157010300, 47157010500, 47157010610, 47157010620, 47157010630, 47157010710, 47157010720,
                47157010810, 47157011010, 47157011100, 47157011200, 47157011600, 47157011700, 47157011800, 47157006900,
                47157007000, 47157007200, 47157007400, 47157007500, 47157002900, 47157020101, 47157020102, 47157020210,
                47157020221, 47157020222, 47157020300, 47157008000, 47157020511, 47157020512, 47157020521, 47157020524,
                47157020531, 47157020541, 47157020542, 47157020610, 47157020621, 47157020622, 47157020632, 47157020633,
                47157020634, 47157020635, 47157020642, 47157020643, 47157020644, 47157020651, 47157020652, 47157020810,
                47157020831, 47157020832, 47157021010, 47157021020, 47157021111, 47157021112, 47157021113, 47157021121,
                47157021122, 47157021124, 47157021125, 47157021126, 47157021135, 47157021136, 47157021137, 47157021138,
                47157021139, 47157021141, 47157021142, 47157021311, 47157021331, 47157021334, 47157021341, 47157021342,
                47157021353, 47157021410, 47157021420, 47157021430, 47157021510, 47157021520, 47157021530, 47157021540,
                47157021611, 47157021613, 47157008200, 47157021620, 47157021710, 47157021721, 47157021724, 47157021725,
                47157021726, 47157021731, 47157021732, 47157021744, 47157021745, 47157021746, 47157021747, 47157021751,
                47157021752, 47157021753, 47157021754, 47157022022, 47157022023, 47157022024, 47157022121, 47157022122,
                47157022130, 47157022210, 47157022220, 47157022310, 47157022321, 47157022322, 47157022330, 47157022410,
                47157022600, 47157022700, 47157020523, 47157008900, 47157009901, 47157009902, 47157010000, 47157020900,
                47157007900, 47157011300, 47157011400, 47157021200, 47157008120, 47157021333, 47157011500, 47157021351,
                47157009600, 47157021140, 47157021612, 47157008600, 47157980100, 47157008700, 47157980400, 47157008800,
                47157021900, 47157000200, 47157011020, 47157009800, 47157022112, 47157009100, 47157020400, 47157010820,
                47157002100, 47157980300, 47157000300, 47157009400, 47157022500, 47105060202, 47105060301, 47105060302,
                47105060400, 47105060501, 47105060502, 47105060700, 47105060201, 47105060600, 47105060100, 47099960600,
                47099960800, 47099960900, 47099960402, 47099960501, 47099960502, 47099960700, 47099960100, 47099960200,
                47099960300, 47099960401, 47089070100, 47089070200, 47089070300, 47089070800, 47089070400, 47089070500,
                47089070600, 47089070700, 47089070900, 47123925000, 47123925400, 47123925502, 47123925100, 47123925300,
                47123925200, 47123925501, 47163042802, 47163042702, 47163041000, 47163041100, 47163041200, 47163042801,
                47163043202, 47163043301, 47163043401, 47163043402, 47163043500, 47163043600, 47163041300, 47163041400,
                47163042300, 47163042600, 47163042701, 47163042900, 47163043000, 47163043201, 47163042400, 47163043302,
                47163042500, 47163043100, 47055920600, 47055920800, 47055920300, 47055920500, 47055920100, 47055920700,
                47055920200, 47055920400, 47025970700, 47025970900, 47025970100, 47025970200, 47025970500, 47025970600,
                47025970800, 47025970300, 47025970400, 47183968000, 47183968101, 47183968102, 47183968202, 47183968203,
                47183968400, 47183968500, 47183968600, 47183968201, 47183968300, 47183968700, 47059090900, 47059091000,
                47059090100, 47059090200, 47059090400, 47059090800, 47059091100, 47059091200, 47059091400, 47059091500,
                47059090300, 47059090500, 47059090600, 47059090700, 47059091300, 47091956000, 47091956100, 47091956200,
                47091956300, 47091956400, 47097050200, 47097050300, 47097050100, 47097050400, 47097050503, 47097050504,
                47097050505, 47097050506, 47097050600, 47149040502, 47149040809, 47149041700, 47149040101, 47149040102,
                47149040103, 47149040104, 47149040302, 47149040303, 47149040304, 47149040306, 47149040307, 47149040308,
                47149040403, 47149040501, 47149040600, 47149040701, 47149040702, 47149040805, 47149040806, 47149040808,
                47149040810, 47149040901, 47149040902, 47149040903, 47149040904, 47149040905, 47149041000, 47149041101,
                47149041102, 47149041202, 47149041301, 47149041302, 47149041402, 47149041403, 47149041500, 47149041600,
                47149041800, 47149042200, 47149041900, 47149040105, 47149040200, 47149040305, 47149042000, 47149040807,
                47149041201, 47149041401, 47149042300, 47149042100, 47131965700, 47131965000, 47131965100, 47131965300,
                47131965400, 47131965500, 47131965600, 47131965800, 47131965900, 47131965200, 47169090200, 47169090100,
                47007953200, 47007953000, 47007953100, 47037019500, 47037010104, 47037017000, 47037017401, 47037980200,
                47037010103, 47037010601, 47037011001, 47037011400, 47037012100, 47037012200, 47037018409, 47037015401,
                47037015625, 47037015629, 47037015900, 47037016300, 47037016800, 47037017901, 47037018401, 47037018601,
                47037010602, 47037018902, 47037018905, 47037019108, 47037019118, 47037010702, 47037019400, 47037016900,
                47037010105, 47037010106, 47037010202, 47037010301, 47037010302, 47037010303, 47037010401, 47037010402,
                47037010501, 47037010502, 47037010701, 47037010801, 47037010802, 47037010901, 47037010903, 47037010904,
                47037011002, 47037011300, 47037015300, 47037011500, 47037011600, 47037011700, 47037011800, 47037011900,
                47037012600, 47037012701, 47037012702, 47037012801, 47037012802, 47037013000, 47037013100, 47037013201,
                47037013202, 47037013300, 47037013400, 47037013500, 47037013601, 47037013602, 47037013700, 47037013800,
                47037013900, 47037014200, 47037014300, 47037014400, 47037010201, 47037014800, 47037015100, 47037015200,
                47037015402, 47037015404, 47037015405, 47037015501, 47037015502, 47037015609, 47037015613, 47037015614,
                47037015615, 47037015617, 47037015618, 47037015619, 47037015622, 47037015623, 47037015624, 47037015626,
                47037015627, 47037015628, 47037015630, 47037015631, 47037015700, 47037015802, 47037015803, 47037015804,
                47037016000, 47037016100, 47037016200, 47037011100, 47037016400, 47037016500, 47037016600, 47037016700,
                47037017100, 47037017200, 47037017300, 47037017402, 47037017500, 47037017701, 47037017702, 47037017800,
                47037017902, 47037018000, 47037018101, 47037018102, 47037018201, 47037018202, 47037018203, 47037018301,
                47037018302, 47037018404, 47037018405, 47037018407, 47037018408, 47037018410, 47037018500, 47037018602,
                47037980100, 47037018700, 47037018801, 47037018803, 47037018804, 47037018901, 47037018904, 47037019003,
                47037019004, 47037019005, 47037019006, 47037019105, 47037019106, 47037019109, 47037019110, 47037019111,
                47037019112, 47037019114, 47037019115, 47037019116, 47037019117, 47037019200, 47037019300, 47037011200,
                47037019600, 47037015610, 47037015612, 47037015620, 47069950300, 47069950400, 47069950200, 47069950500,
                47069950600, 47069950100, 47045964200, 47045964300, 47045964400, 47045964800, 47045964900, 47045964000,
                47045964500, 47045964600, 47061955300, 47061955000, 47061955100, 47061955200, 47057500100, 47057500300,
                47057500200, 47057500401, 47057500402, 47125101400, 47125101500, 47125980100, 47125100601, 47125100700,
                47125101001, 47125101305, 47125102001, 47125100100, 47125100200, 47125100300, 47125100400, 47125100500,
                47125100602, 47125100800, 47125100900, 47125101002, 47125101101, 47125101102, 47125101103, 47125101201,
                47125101202, 47125101303, 47125101304, 47125101306, 47125101307, 47125101600, 47125101700, 47125101802,
                47125101803, 47125101804, 47125101902, 47125101903, 47125101904, 47125102002, 47125102003, 47125102004,
                47125102005, 47125102006, 47109930400, 47109930700, 47109930100, 47109930300, 47109930500, 47109930600,
                47109930200, 47071920400, 47071920600, 47071920500, 47071920100, 47071920200, 47071920300, 47155080102,
                47155080500, 47155980100, 47155080601, 47155080101, 47155080201, 47155080202, 47155080300, 47155080400,
                47155080602, 47155080700, 47155080801, 47155080802, 47155080901, 47155080902, 47155081000, 47155081101,
                47155081102, 47011010200, 47011010300, 47011010500, 47011010900, 47011011000, 47011011401, 47011011601,
                47011010800, 47011011201, 47011011202, 47011011300, 47011010100, 47011010400, 47011010600, 47011010700,
                47011011402, 47011011100, 47011011500, 47011011602, 47053980100, 47053966100, 47053966200, 47053966300,
                47053966400, 47053966500, 47053966600, 47053966800, 47053967100, 47053966700, 47053966900, 47053967000,
                47053967300, 47053967400, 47029980100, 47029920501, 47029920100, 47029920200, 47029920300, 47029920400,
                47029920502, 47029920600, 47029920700, 47139950100, 47139950202, 47139950300, 47139950400, 47139950201,
                47165020101, 47165020300, 47165020903, 47165021008, 47165020102, 47165020203, 47165020204, 47165020205,
                47165020206, 47165020207, 47165020209, 47165020403, 47165020404, 47165020405, 47165020406, 47165020407,
                47165020501, 47165020502, 47165020503, 47165020601, 47165020602, 47165020603, 47165020700, 47165020800,
                47165020901, 47165020902, 47165021002, 47165021004, 47165021005, 47165021006, 47165021007, 47165021009,
                47165021103, 47165021105, 47165021106, 47165021107, 47165021201, 47165021203, 47165021204, 47165021205,
                47165020208, 47165021104, 47181950400, 47181950100, 47181950200, 47181950300, 47009010500, 47009010600,
                47009010700, 47009010800, 47009011002, 47009011101, 47009011102, 47009011200, 47009011301, 47009011302,
                47009011402, 47009011501, 47009011502, 47009011503, 47009011602, 47009011603, 47009011604, 47009011605,
                47009010400, 47009980200, 47009980100, 47009010100, 47009010200, 47009010301, 47009010302, 47009011001,
                47009011401, 47009010900, 47141000100, 47141000200, 47141000301, 47141000302, 47141000303, 47141000400,
                47141000500, 47141000600, 47141000700, 47141000800, 47141000900, 47141001000, 47141001100, 47141001200,
                47141001300, 47107970101, 47107970102, 47107970200, 47107970401, 47107970402, 47107970500, 47107970800,
                47107970300, 47107970600, 47107970700, 47021070102, 47021070103, 47021070202, 47021070203, 47021070402,
                47021070104, 47021070201, 47021070300, 47021070401, 47153060101, 47153060102, 47153060200, 47115050101,
                47115050102, 47115050201, 47115050202, 47115050301, 47115050302, 47015960100, 47015960200, 47015960300,
                47001020500, 47001020901, 47001020902, 47001021201, 47001021202, 47001021301, 47001021302, 47001020100,
                47001980100, 47001020201, 47001020400, 47001020700, 47001021100, 47001020600, 47001020202, 47001020300,
                47001020800, 47001021000, 47033961300, 47033961000, 47033961100, 47033961200, 47033961400, 47161980100,
                47161980200, 47161110200, 47161110700, 47161110600, 47041920102, 47041920200, 47041920101, 47041920300,
                47189030202, 47189030903, 47189030600, 47189030102, 47189030308, 47189030700, 47189030101, 47189030203,
                47189030204, 47189030303, 47189030304, 47189030307, 47189030309, 47189030401, 47189030402, 47189030800,
                47189030901, 47189030904, 47189030305, 47189031000, 47189030500, 47039955001, 47039955002, 47039955101,
                47039955102, 47101970200, 47101970100, 47173040100, 47173040202, 47173040201, 47173040300, 47023970200,
                47023970300, 47023970100, 47013951100, 47013950300, 47013950400, 47013950600, 47013950100, 47013950200,
                47013950500, 47013950700, 47013950800, 47013950900, 47013951000, 47077975500, 47077975100, 47077975200,
                47077975300, 47077975400, 47077975000, 47031970300, 47031970400, 47031970500, 47031970802, 47031970900,
                47031971000, 47031980100, 47031970100, 47031970200, 47031970700, 47031970801, 47031970600, 47117955000,
                47117955100, 47117955200, 47117955300, 47117955500, 47117955400, 47081950500, 47081950200, 47081950100,
                47081950301, 47081950302, 47081950400, 47017962000, 47017962201, 47017962202, 47017962100, 47017962300,
                47017962400, 47017962500, 47017980100, 47047060600, 47047060300, 47047060701, 47047060702, 47047060401,
                47047060402, 47047060403, 47047060501, 47047060502, 47047060800, 47047060404, 47133950100, 47133950200,
                47133950301, 47133950302, 47133950500, 47133950600, 47133950400, 47119010201, 47119010202, 47119010301,
                47119010302, 47119010400, 47119010500, 47119010600, 47119010801, 47119010802, 47119010900, 47119011001,
                47119011002, 47119011101, 47119011102, 47119011200, 47119010100, 47119010700, 47127930100, 47127930200,
                47129110200, 47129110500, 47129110100, 47129110300, 47129110400, 47135930100, 47135930200, 47187050103,
                47187050203, 47187050205, 47187050303, 47187050304, 47187050306, 47187050307, 47187050403, 47187050405,
                47187050406, 47187050502, 47187050503, 47187050504, 47187050602, 47187050800, 47187050905, 47187050906,
                47187050907, 47187050908, 47187050909, 47187051001, 47187051002, 47187051201, 47187051202, 47187050101,
                47187050207, 47187050601, 47187050701, 47187050702, 47187050904, 47187051100, 47187050102, 47187050204,
                47187050206, 47187050208, 47187050305, 47187050404, 47087960300, 47087960100, 47087960200, 47087960400,
                47079969000, 47079969200, 47079969300, 47079969400, 47079969500, 47079969800, 47079969700, 47079969600,
                47079969100, 47003950100, 47003950200, 47003950300, 47003950401, 47003950402, 47003950500, 47003950600,
                47003950800, 47003950700, 47185935000, 47185935100, 47185935200, 47185935300, 47185935400, 47185935500,
                47083120100, 47083120200, 47083120300, 47063101200, 47063100100, 47063100200, 47063100300, 47063100400,
                47063100500, 47063100600, 47063100700, 47063100800, 47063100900, 47063101000, 47063101100, 47175925000,
                47175925200, 47167040302, 47167040400, 47167040100, 47167040200, 47167040303, 47167040304, 47167040500,
                47167040601, 47167040602, 47167040700, 47167040800, 47167040900, 47167041000, 47103975100, 47103975200,
                47103975300, 47103975400, 47103975602, 47103975700, 47103975601, 47103975000, 47103975500, 47171080300,
                47171080100, 47171080200, 47171080400, 47111970100, 47111970200, 47111970300, 47111970400, 47051960201,
                47051960600, 47051960700, 47051960100, 47051960202, 47051960300, 47051960400, 47051960500, 47051960800,
                47027955000, 47027955100, 47137925100, 47075930100, 47075930200, 47075930301, 47075930302, 47075930400,
                47075930500, 47035970601, 47035970101, 47035970102, 47035970200, 47035970301, 47035970302, 47035970501,
                47035970502, 47035970602, 47035970603, 47035970701, 47035970702, 47035970800, 47035970400, 47073050100,
                47073050200, 47073050301, 47073050302, 47073050400, 47073050501, 47073050502, 47073050503, 47073050602,
                47073050700, 47073050800, 47073050900, 47095960100, 47095960200, 47143975000, 47143975100, 47143975200,
                47143975300, 47143975401, 47143975402, 51077060300, 51169030300, 51169030200, 51169030500, 51191010401,
                51191010100, 51191010800, 51191010900, 51191011000, 51191010200, 51191010300, 51105950500, 51105950100,
                51105950200, 51105950400, 51105950300, 51105950600, 51167030300, 51520020100, 51520020200, 51520020300,
                51520020400, 51195931100, 51195931200, 51195931300]

class G_Z_scaler:
    def __init__(self, cols=None, scale_type='_Z_',):
        self.scl_type = scale_type
        self.cols = cols
        if self.scl_type in ['_Z_', '_nrml_']:
            if self.scl_type == '_Z_':
                print('performing standard scaling')
                self.scaler = StandardScaler()
            else:
                print('performing normal scaling')
                self.scaler = MinMaxScaler()

        self.mus = None
        self.std = None

    def fit(self, df, cols=None):
        if cols is None and self.cols is None:
            self.cols = df.columns.tolist()
        elif cols is None:
            cols = self.cols
        self.mus = df[cols].mean(axis=0)
        self.std = df[cols].std(axis=0)
        self.cols = cols
    def fit_transform(self, df, cols):
        self.fit(df, cols)
        cols = self.cols
        print('the mean: {} the std: {}'.format(self.mus, self.std))
        rdf = (df[cols] - self.mus)/self.std
        print('description of scaled data:')
        print(rdf.describe())
        return self.transform(df)

    def transform(self, df):
        for col in self.cols:
            # strip current coll of nan
            tcol = df[col]
            tcol.replace(-999, np.nan, inplace=True)
            tcol.replace(np.inf, np.nan, inplace=True)
            tcol.replace('', np.nan, inplace=True)
            #tcol.dropna(inplace=True)
            mu = tcol.mean()
            stdev = tcol.std()
            df[col] = (df[col] - mu)/stdev
        print('description of scaled data:')
        print(df.describe())
        return df

def generate_filter(df, a, b, ftype='le', ):
    """    This will create a filter for a given data frame. The returned filter can be used
           in the following ways:
                * df[filter] =: this will return an series of the values of  indices where the given filter is true
                * df.loc[df[filter], :] =: will return a slice of the data frame where the filter is true
           df: a given dataframe or numpy array
           a: the index or label of the data to be checked
           b: the value to check the data against
           ftype: the filter type desired options are:
                * le  =: less than
                * ge  =: greater than
                * leq =: less than or equal too
                * geq =: greater than or  equal too
                * eq  =: equal too
                * neq =: not equal too
    """
    if ftype == 'le':
        return df[a] < b
    elif ftype == 'ge':
        return df[a] > b
    elif ftype == 'leq':
        return df[a] <= b
    elif ftype == 'geq':
        return df[a] >= b
    elif ftype == 'eq':
        return df[a] == b
    elif ftype == 'neq':
        return df[a] != b

def calculate_delR2_significance(R1, R2, N1, N2, Nobservations):
    return ((R2 - R1)/N2) / ((1-R2)/(Nobservations - (N1 + N2) - 1))

def smooth_set(df, alpha=.3, to_smooth=None):
    if to_smooth is None:
        to_smooth = df.columns.tolist()
    for v in to_smooth:
        df[v] = df[v].ewm(alpha=alpha).mean()
    return df

def remove_outliers(df, threshold=3, vars_to_check=None, verbose=True):
    no = df.shape[0]

    # get a dictionary of the iqr value for each so we do not
    # keep changing it like an idiot
    iqr_d = {}
    q25_d = {}
    q75_d = {}
    lwr_d = {}
    hgr_d = {}
    for v in vars_to_check:
        #iqr_d[v] = stats.iqr(df[v].dropna())
        q25_d[v] = df[v].dropna().quantile(.25)
        q75_d[v] = df[v].dropna().quantile(.75)
        iqr_d[v] = q75_d[v] - q25_d[v]
        #lwr_d[v] = max(q25_d[v] - iqr_d[v]*1.5, 0)
        lwr_d[v] = q25_d[v] - iqr_d[v]*1.5
        hgr_d[v] = q75_d[v] + iqr_d[v]*1.5


    for v in vars_to_check:
        print('==================================')
        print('checking {}'.format(v))
        #dfv = pd.DataFrame()
        #Q1 = df[v].dropna().quantile(.25)
        #Q3 = df[v].dropna().quantile(.75)
        #IQR = stats.iqr(df[v].dropna())
        #print('Q1: {}, Q3: {}, IQR: {}'.format(Q1, Q3, IQR))
        print('Lower: {}, Higher: {}'.format(lwr_d[v], hgr_d[v]))
        #dfv[v] = np.abs(stats.zscore(df[v]))
        ss = df.shape[0]
        print('size of data: {}'.format(ss))
        df = df.loc[df[v] > lwr_d[v], :]
        print('max: {}, min: {}'.format(df[v].max(), df[v].min()))
        print('After removing lower Now it is {}'.format(df.shape[0]))
        df = df.loc[df[v] <= hgr_d[v], :]
        nn= df.shape[0]
        print('After removing upper Now it is {}'.format(df.shape[0]))
        print('a {} % loss'.format(((ss - nn)/ss)*100))
        print('---------------------------------\n')
    if verbose:
        nc = df.shape[0]
        loss = ((no - nc) / no) * 100
        print('11N: {}, cleaned: {}, {:.3f}% data loss'.format(no, nc, loss))
    return df

def log10_transform(df, vars):
    added_strings = []
    for v in vars:
        added_strings.append(v + '_log10')
        fix = 0
        mini = df.filter(items=[v]).describe()
        mini = mini.loc['min', v]
        if mini == 0:
            fix = 1
        df[v + '_log10'] = np.log10(df[v].values + fix)
        print('"{}": "{}",'.format(v + '_log10', label_translation_dict[v] + '_10'))
    return added_strings

def remove_outliers3(df, threshold=3, vars_to_check=None, verbose=True):
    no = df.shape[0]
    z_scores = pd.DataFrame(np.abs(StandardScaler().fit_transform(df.filter(items=vars_to_check, axis=1))), columns=vars_to_check, index=df.index)
    mark_d = pd.DataFrame(df.copy())
    for v in z_scores.columns.tolist():
        print('V: {}'.format(v))
        tng =z_scores[v] > threshold
        print("bool vals: {}".format(tng.values))
        mark_d[v] = np.zeros(len(z_scores[v]))
        count = 0
        outcount = 0
        for bl in tng:
            if bl:
                mark_d.loc[count, v] = 1
                outcount += 1
            count += 1

    pct_loss_d = {}
    print('Out count: {}'.format(outcount))
    for v in vars_to_check:
        print('\n==================================')
        print('checking {}'.format(v))
        #print('the og head {}'.format(df[v].head()))
        ss = df.shape[0]
        print('before removal shape {}'.format(df.shape[0]))
        #dfv = pd.DataFrame()
        #dfv[v] = np.abs(stats.zscore(df[v]))
        #dfv[v] = np.abs((df[v] - df[v].mean())/df[v].std())
        #dfv.index = df.index
        #print('Zhead:\n', dfv.head())
        #print('number not nan ', dfv[v].head())
        #z = np.abs(stats.zscore(df[v]))
        #df = df.loc[dfv[v] <= threshold, :]
        #df = df.loc[z <= threshold, :]
        #df = df.loc[df[v] <= highr, :]
        df = df.loc[mark_d[v] != 1, :]
        mark_d = mark_d.loc[mark_d[v] != 1, :]
        nn= df.shape[0]
        print('Now it is {}'.format(df.shape[0]))
        print('a {} % loss'.format(((ss - nn)/ss)*100))
        pct_loss_d[v] =((ss - nn)/ss)*100
        print('---------------------------------\n')
    if verbose:
        nc = df.shape[0]
        loss = ((no - nc) / no) * 100
        print('11N: {}, cleaned: {}, {:.3f}% data loss'.format(no, nc, loss))

    pct_loss_d = sort_dict(pct_loss_d, reverse=True)
    for v in pct_loss_d:
        print('{} lost {} %'.format(v, pct_loss_d[v]))

    return df

def scale_select(df, to_scale, scaler):
    for v in to_scale:
        print('\nthe v: {}\n'.format(v))
        #print(df.filter(items=[v], axis=1).head())
        df[v] = scaler.fit_transform(df.filter(items=[v]).values)
    return df

def scale_selectNRM(df, to_scale, verbose=False):
    # Scale the data for the things we want to scale before we impute it
    for v in to_scale:
        if verbose:
            print('V: {}'.format(v))
        df[v] = (df[v].values - df[v].min()) / (df[v].max() - df[v].min())
    return df


def remove_outliers2(data, method='z', threshold=3, verbose=False):
    z = np.abs(stats.zscore(data))
    print(z)
    print((z < threshold).all(axis=1))
    clean_data = data.loc[(z < threshold).all(axis=1), :]
    if verbose:
        no = data.shape[0]
        nc = clean_data.shape[0]
        loss = ((no - nc)/no)*100
        print('N: {}, cleaned: {}, {:.3f}% data loss'.format(no, nc, loss))
    return clean_data

def metric_filter(df, filter, metric):
    """ This method given a data frame, filter (see generate_filer() method above),
        and a label or list of labels (metric) will return a slice of the data
        frame where the with the filter is passed with the given column('s) (metric)
    :param df: data frame (pandas)
    :param filter: boolean filter used to select rows in data frame
    :param metric: column('s) of dataframe desired
    :return: a slice of a data frame
    """
    return df.loc[filter, metric]

def generate_median_dfs(df, income_dic=income_dic, metric='median_household_income'):
    ret_d = {}
    for k in income_dic:
        ret_d[k] =select_by_med_income_range(df, income_dic[k], metric=metric)
    return ret_d


def select_by_med_income_range(df, Irange, metric="median_household_income", ):
    llim =df[metric].median() * Irange[0]

    #print('checking values {} '.format(llim))
    df_r = df.loc[ df[metric] >= llim, :]
    #print('there are {} after first range({}) selection'.format(df_r.shape[0], Irange[0]))
    if len(Irange) > 1:
        ulim = df[metric].median() * Irange[1]
        df_r = df.loc[ df[metric] < ulim, :]
        print('checking values {} '.format(ulim))
        print('there are {} after second range({}) selection'.format(df_r.shape[0], Irange[1]))
    return df_r

def generate_fips_dic_cnt(fipsa, fipsb, metricb):
    """
                Will count the shared census tracts
    :param fipsa:
    :param fipsb:
    :param metricb:
    :return:
    """
    cnt = 0
    for fa in fipsa:
        if fa in fipsb:
            cnt += 1
    return cnt

def grab_threshold_selected_regions_quantile(df, threshold_met='daily_solar_radiation', qtthrsh=.5, cmp='lte'):
    print(df[threshold_met].quantile(qtthrsh))

    if cmp == 'lte' or cmp not in ['lte', 'lt', 'gte', 'gt', 'eq', 'neq']:
        return df.loc[df[threshold_met] <= df[threshold_met].quantile(qtthrsh), :]
    elif cmp == 'lt':
        return df.loc[df[threshold_met] < df[threshold_met].quantile(qtthrsh), :]
    if cmp == 'gte':
        return df.loc[df[threshold_met] >= df[threshold_met].quantile(qtthrsh), :]
    if cmp == 'gt':
        return df.loc[df[threshold_met] > df[threshold_met].quantile(qtthrsh), :]
    if cmp == 'eq':
        return df.loc[df[threshold_met] == df[threshold_met].quantile(qtthrsh), :]
    if cmp == 'neq':
        return df.loc[df[threshold_met] != df[threshold_met].quantile(qtthrsh), :]

def grab_threshold_selected_regions_value(df, threshold_met='daily_solar_radiation', val=0, cmp='lte'):
    if cmp == 'lte' or cmp not in ['lte', 'lt', 'gte', 'gt', 'eq', 'neq']:
        return df.loc[df[threshold_met] <= val, :]
    elif cmp == 'lt':
        return df.loc[df[threshold_met] < val, :]
    if cmp == 'gte':
        return df.loc[df[threshold_met] >= val, :]
    if cmp == 'gt':
        return df.loc[df[threshold_met] > val, :]
    if cmp == 'eq':
        return df.loc[df[threshold_met] == val, :]
    if cmp == 'neq':
        return df.loc[df[threshold_met] != val, :]


def process_hotspots_fips(fips_dic, solar_metrics):
    for sm in solar_metrics:
        for smb in solar_metrics:
            if smb not in fips_dic[sm]:
                fips_dic[sm][smb] = {
                    'above two std mean': 0,
                    '90th': 0,
                    '95th': 0,
                }
            if smb != sm:
                print('a {}, b{}'.format(sm, smb))
                fips_dic[sm][smb]['above two std mean'] = generate_fips_dic_cnt(fips_dic[sm]['above two std mean'],
                                                                                fips_dic[smb]['above two std mean'],
                                                                                smb)
                fips_dic[sm][smb]['90th'] = generate_fips_dic_cnt(fips_dic[sm]['90th'],
                                                                  fips_dic[smb]['90th'], smb)
                fips_dic[sm][smb]['95th'] = generate_fips_dic_cnt(fips_dic[sm]['95th'],
                                                                  fips_dic[smb]['95th'], smb)
                fips_dic[sm][smb] = sort_dict(fips_dic[sm][smb])
                print('==================================================================================')
                print('Share fips between {} and {}'.format(sm, smb))
                print(fips_dic[sm][smb])
                print('==================================================================================')
                print()

    return 0


def generate_hotspot_dict(ds_df, tdf, ignores,):
    region_dic = {
        'Mean': list(),
        'Variable': list(),
    }
    for var in ds_df.columns.values.tolist():
        # if var in pv_hotspots_stats.columns.values.tolist():
        if type(ds_df[var].values.tolist()[0]) != type(str('')) and var not in ignores:
            region_dic['Variable'].append(var)
            region_dic['Mean'].append(tdf.loc['mean', var])
    return region_dic

def build_RFC(params=None, learner_type='clf'):
    # TODO: sklearn page: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    if params is None:
        if learner_type == 'clf':
            params = default_RF_params
        else:
            params = default_RFR_params

    if params is not None:
        if learner_type == 'clf':
            for p in params:
                if p in default_RF_params:
                    default_RF_params[p] = params[p]
            params = default_RF_params
        else:
            for p in params:
                if p in params:
                    default_RFR_params[p] = params[p]
            params = default_RFR_params

    if learner_type in ['clf', 'classifier']:
        rfc = RandomForestClassifier(
                             n_estimators = params['n_estimators'],
                             criterion = params['criterion'],
                             max_depth = params['max_depth'],
                             min_samples_split=params['min_samples_split'],
                             min_samples_leaf = params['min_samples_leaf'],
                             min_weight_fraction_leaf = params['min_weight_fraction_leaf'],
                             max_features=params['max_features'],
                             max_leaf_nodes=params['max_leaf_nodes'],
                             min_impurity_decrease=params['min_impurity_decrease'],
                             min_impurity_split=params['min_impurity_split'],
                             bootstrap=params['bootstrap'],
                             oob_score=params['oob_score'],
                             n_jobs=params['n_jobs'],
                             random_state=params['random_state'],
                             verbose=params['verbose'],
                             warm_start=params['warm_start'],
                             class_weight=params['class_weight'],
                             )
    else:
        params = default_RFR_params
        rfc = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            criterion=params['criterion'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
            max_features=params['max_features'],
            max_leaf_nodes=params['max_leaf_nodes'],
            min_impurity_decrease=params['min_impurity_decrease'],
            min_impurity_split=params['min_impurity_split'],
            bootstrap=params['bootstrap'],
            oob_score=params['oob_score'],
            n_jobs=params['n_jobs'],
            random_state=params['random_state'],
            verbose=params['verbose'],
            warm_start=params['warm_start'],
        )
    return rfc

def filter_missing_filter(feats, list_to_check):
    fuse = list()
    for f in feats:
        if f not in list_to_check:
            print('{} was not in the data set and will not be in the model'.format(f))
        else:
            fuse.append(f)
    feats = fuse
    return feats

def all_in_rf_trainer(learner, df, feats, target):
    print('-----------------------------------')
    print('feats:  ', feats)
    print('target: ', target)
    print('-----------------------------------')
    print()

    feats = filter_missing_filter(feats, df.columns.tolist())
    if len(feats) == 0:
        return [], 0, 0, 0, 0
    sampleX, sampleY = df.filter(items=feats), df.filter(items=[target])
    #print('x ', sampleX)
    #print('y ', sampleX)
    learner.fit(sampleX, sampleY)
    if len(sampleX) == 0 or len(sampleY) == 0:
        return [], 0, 0, 0, 0
    ypred, acc, sen, prec, R2 = learner.score(sampleX, sampleY, )
    return ypred, acc, sen, prec, R2

def train_test_trainer(learner, xtr, ytr, xts, yts, feats, target):
    feats = filter_missing_filter(feats, xtr.columns.tolist())
    print('the feats: {}'.format(feats))
    if len(feats) == 0:
        return [], 0, 0, 0, 0
    print()
    #print(xtr.head())
    #print(ytr.head())
    xtr = xtr.filter(items=feats)
    xts = xts.filter(items=feats)
    #print(xtr.head())
    #print(ytr.head())
    learner.fit(xtr, ytr)
    print('{}, lllll  , {}'.format(xtr.shape[1], xts.shape[1]))
    ypred, acc, sen, prec, R2 = learner.score(xts, yts, )
    return ypred, acc, sen, prec, R2

def hierarchical_rf_testing(params_rf, df, group_dic, target='Adoption', verbose=True, m_type='rfclf', hparams_tr=(.75, ),
                            mode='all_in'):
    from _products.ML_Tools import RandomForest_Analzer
    from sklearn.model_selection import train_test_split
    # load stored default params
    param_keeper = Keeper_of_Params()
    params = param_keeper.get_params(m_type=m_type)
    #adjust if needed
    if params_rf is not None:
        for adj_p in params_rf:
            params[adj_p] = params_rf[adj_p]
     # create the necessary learner


    factor_dic_acc, factor_dic_R2 = {}, {}
    factor_dic_sen, factor_dic_pre = {}, {}
    ts, trsz = np.around(1-hparams_tr[0], 2), hparams_tr[0]

    print('training: {}, testing: {}'.format(trsz, ts))

    col_feats = df.columns.tolist()
    del col_feats[col_feats.index(target)]

    if mode != 'all_in':
        print('the len: {} '.format(len(df.loc[df[target].isna(), :])))
        Xtr, Xts, ytr, yts = train_test_split(df[col_feats], df[target], test_size=ts, train_size=trsz,
                                              stratify=df[target])

        xtr = pd.DataFrame(Xtr, columns=col_feats, )
        xts = pd.DataFrame(Xts, columns=col_feats, )

        ytr = pd.DataFrame(ytr, columns=[target], )
        yts = pd.DataFrame(yts, columns=[target], )
        print('tr shape: {}, ts shape: {}'.format(xtr.shape[1], xts.shape[1]))
    for factors in group_dic:
        if m_type in ['rfclf',]:
            learner_Analzer = RandomForest_Analzer(clf_type='randomforest', scaler_type=None,
                                                   params=params, tr_split=hparams_tr, model_type='classifier')
            learner = learner_Analzer
            if mode == 'all_in':
                ypred, acc, sen, prec, R2 = all_in_rf_trainer(learner, df, group_dic[factors], target)
            else:

                ypred, acc, sen, prec, R2 = train_test_trainer(learner, xtr, ytr, xts, yts, group_dic[factors], target)
            #learner.learner.set_params(n_estimators=ne_orig + learner.learner.n_estimators)
            print('for group {}'.format(factors))
            print('accuracy: {}'.format(acc))
            print('sen: {}'.format(sen))
            print('prec: {}'.format(prec))
            print('R2: {}'.format(R2))
            factor_dic_sen[factors] = [sen]
            factor_dic_pre[factors] = [prec]
            factor_dic_acc[factors] = [acc]
            factor_dic_R2[factors] = [R2]

    print('there are {} trees in the end: ')
    dic_return_dic = {'acc': factor_dic_acc, "R2":factor_dic_R2, 'sen':factor_dic_sen, 'pre':factor_dic_pre}
    return dic_return_dic


class Block_group_analysis_tool:
    def __init__(self):
        self.variables=None

    def perform_block_analysis(self, clf, df, feats, target, trsz, log_file, group_labels, reg, group,
                               verbose=False, learner_type='clf', figsize=(10, 10),
                               plot_name='{}_plot.svg', show_it=False, dpi=600, save_it=False):

        # use the given dictionary to easily retrieve a list of features by group keyword
        # store the labels of the groups from the dictionary
        self.feats = feats

        # generate_training and testing sets
        Training, Testing = self.learning_data_generator(df, self.feats, target, trsz)

        # select the desired states from the group_region_dic
        feature_impz_, training_performance, testing_performance = self.train_and_test_clf_RF(clf, Training, Testing,
                                                                                              verbose=verbose,
                                                                                              learner_type=learner_type)

        self.block_group_bar(log_file, group_labels, reg, group, figsize=figsize, plot_name=plot_name,
                             accTS=testing_performance[0], null_acc=testing_performance[-1], show_it=show_it, dpi=dpi,
                             figsize2=figsize, save_it=save_it, )

    def perform_block_analysis4444(self, clf, df, feats, target, trsz, log_file, group_labels_dict, region_dic, reg, group,
                           reg_col='state', verbose=False, learner_type='clf', figsize=(10, 10),
                           plot_name='{}_plot.svg', show_it=False, dpi=600, save_it=False):

        # use the given dictionary to easily retrieve a list of features by group keyword
        # store the labels of the groups from the dictionary
        self.feats = list(set(group_labels_dict[group] + reg_col + target))
        group_labels = list(group_labels_dict.keys())
        feats = self.feats

        # use the regional dictionary to pull out the desired test regions based on the reg_col (column used to pull)
        # regions), and the region_dic, and the selected region
        df = df.loc[df[reg_col].isin(reg_dict[reg]), self.feats]
        # drop the regional column
        df = df.drop(reg_col, axis=1)

        Training, Testing = self.learning_data_generator(df, self.feats, target, trsz)

        # select the desired states from the group_region_dic

        feature_impz_, training_performance, testing_performance = self.train_and_test_clf_RF(clf, Training, Testing,
                                                                                              verbose=verbose,
                                                                                              learner_type=learner_type)
        self.block_group_bar(log_file, group_labels, reg, group, figsize=figsize, plot_name=plot_name,
                             accTS=testing_performance[0], null_acc=testing_performance[-1], show_it=show_it, dpi=dpi,
                             figsize2=figsize, save_it=save_it, )

    def learning_data_generator(self, df, feats, target, trsz=.5):
        from sklearn.model_selection import train_test_split
        ts = np.around(1 - trsz, 2)
        # do a Reg Analysis and show results
        Xtr, Xts, ytr, yts = train_test_split(df[feats], df[target], test_size=ts, train_size=trsz,
                                              stratify=df[target])
        Training = [Xtr, ytr]
        Testing = [Xts, yts]
        return Training, Testing

    def train_and_test_clf_RF(self, clf, Training, Testing, verbose=False, learner_type='clf', num_runs=50):
        from _products.ML_Tools import train_and_test_clf_RF
        from sklearn.metrics import accuracy_score, recall_score, precision_score
        from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
        yptr, ypts, feature_impz_ = train_and_test_clf_RF(clf, Training, Testing, verbose=verbose)

        if learner_type == 'clf':
            accTR = accuracy_score(Training[1], yptr)
            senTR = recall_score(Training[1], yptr)
            precTR = precision_score(Training[1], yptr)
            R2TR = explained_variance_score(Training[1], yptr)
            training_performance = [accTR, senTR, precTR, R2TR]

            accTS = accuracy_score(Testing[1], ypts)
            senTS = recall_score(Testing[1], ypts)
            precTS = precision_score(Testing[1], ypts)
            R2TS = explained_variance_score(Testing[1], ypts)


            null_score = list()
            for i in range(num_runs):
                y_null = np.array(Training[1])
                np.random.shuffle(y_null)
                null_score.append(accuracy_score(Training[1], y_null))
            null_score = np.mean(null_score)
            if verbose:
                print('Testing Accuracy: {:.3f}'.format(accTS))
                print('Testing Sensitivity: {:.3f}'.format(senTS))
                print('Testing Precision: {:.3f}'.format(precTS))
                print('Testing R^2: {:.3f}'.format(R2TS))
                print('Null Score: {:.3f}'.format(null_score))
            testing_performance = [accTS, senTS, precTS, R2TS, null_score]
            return feature_impz_, training_performance, testing_performance
        else:
            maeTr = mean_absolute_error(Training[1], yptr)
            mseTr = mean_squared_error(Training[1], yptr)
            R2TR = explained_variance_score(Training[1], yptr)

            maeTs = mean_absolute_error(Testing[1], ypts)
            mseTs = mean_squared_error(Testing[1], ypts)
            R2TS = explained_variance_score(Testing[1], ypts)

            training_performance = [maeTr, mseTr, R2TR]
            null_score = list()
            for i in range(num_runs):
                y_null = np.array(Training[1])
                np.random.shuffle(y_null)
                null_score.append(mean_absolute_error(Training[1], y_null))
            null_score = np.mean(null_score)
            if verbose:
                print('Testing MAE: {:.3f}'.format(maeTs))
                print('Testing MSE: {:.3f}'.format(mseTs))
                print('Testing R^2: {:.3f}'.format(R2TS))
                print('Null Score: {:.3f}'.format(null_score))
            testing_performance = [maeTs, mseTs, R2TS, null_score]
            return feature_impz_, training_performance, testing_performance


    def block_group_bar(self, filename, group_labels, reg, src, figsize=(10, 10), plot_name='{}_plot.svg',
                        accTS=None, null_acc=None, show_it=False, dpi=600, figsize2=(10,6), save_it=False,
                        grid_on=True):
        """

        :param filename: file to store retrieve the block groups performance values
        :param group_labels: the labels for the block groups
        :param reg: label for what region/type of areas
        :param src: name of this particular runs group
        :param figsize: figure size for plot
        :param plot_name: name to save plot ass
        :param accTS: testing accuracy score
        :param null_acc: null score
        :param show_it: if you want to show plot
        :param dpi: resolution of plot
        :param figsize2:
        :param save_it: do you want to save the plot
        :return:
        """
        # labels
        fontdict2 = {
            'family': 'serif',
            # 'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'medium',
            'size': '10',
        }
        if os.path.isfile(filename):
            df = pd.read_csv(filename, index_col='Block_Groups')
            print('stored data loaded from {}'.format(filename))
        else:
            dictt = {
                'Block_Groups': group_labels,
                'Acc': list([0] * len(group_labels)),
                'Imp': list([0] * len(group_labels)),
                'Nll': list([0] * len(group_labels)),
            }
            df = pd.DataFrame(dictt)
            df.to_csv(filename, index=False)
            print('New storage file created {}'.format(filename))
            df = pd.read_csv(filename, index_col='Block_Groups')

        if src is None:
            src = 'ALL'
        if src == 'income employment':
            src = 'inc/empl'
        if src == 'income_housing':
            src = 'inc/homes'
        if df.loc[src, 'Acc'] != 0:
            df.loc[src,'Acc'] = (df.loc[src,'Acc'] + accTS)/2
            df.loc[src,'Nll'] = (df.loc[src,'Nll'] + null_acc)/2
            #df.loc[src, 'Acc'] = ((accTS - null_acc) + df.loc[src, 'Acc']) / 2
            df.loc[src, 'Imp'] = (df.loc[src,'Imp'] + (accTS-null_acc))  / 2
        else:
            df.loc[src, 'Imp'] = (accTS - null_acc)
            df.loc[src, 'Acc'] = accTS
            df.loc[src, 'Nll'] = null_acc
        print('------------------------------------------------------------------------')
        print('{:s} avg. Accuracy: {:.2f}, Null: {:.2f}'.format(src, df.loc[src, 'Acc']*100, df.loc[src,'Nll']*100))
        print('------------------------------------------------------------------------')
        xv = list()
        for v in range(len(df)):
            if v == 0:
                xv.append(v)
            else:
                xv.append(v+.05)
        impv = df['Imp'].values.tolist()   # grab current values for improvement over null
        htsl = df['Acc'].values.tolist()   # grab current values for Accuracy
        # sub title
        fontdict3 = {
            # 'family': 'serif',
            'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'heavy',
            'size': '12',
        }
        plt.figure(figsize=figsize, dpi=dpi)
        fig3, ax3 = plt.subplots(1, 1, figsize=figsize2)        # create_subplot to get at fancy stuff
        #ax3.set_xticks(range(len(group_labels)))

        print('------------------')
        print('------------------')
        print('xv')
        print(xv)
        print('------------------')
        print('------------------')
        def get_accss(x):
            print('xacc',x)
            print('---------')
            l = list()
            #for xi in xv:
            #    l.append(htsl[xv.index(xi)])
            for xi in x:
                p = .05
                if xi[0] == 0:
                    p = 0
                l.append(htsl[xv.index(xi[0]+p)])
            return l[0], l[1]

        def get_impv(x):
            print('ximpv',x)
            print('---------')
            l = list()
            #for xi in xv:
            #    l.append(impv[xv.index(xi)])
            for xi in x:
                p = .05
                if xi[0] == 0:
                    p = 0
                l.append(htsl[impv.index(xi[0])])
            return l[0], l[1]
        #secax = ax3.secondary_yaxis('right', functions=(get_impv,
        #                                                get_accss))
        #secax.set_ylabel('Accuracy')
        ax3.set_xticks(xv)                                      # use list generated from for loop above to set xpostions of bars
        ax3.set_yticks([ .6, .65, .7, .75, .8, .85, .9,])
        plt.ylim(.6, .9)
        ax3.set_xticklabels(group_labels, rotation=25, fontdict={'size': 10})   # let up the names for the bars
        #ax3.bar(xv, impv, align='center', color='green',)                            # create the bar graph
        ax3.bar(xv, htsl, align='center', color='green',)                            # create the bar graph
        plt.title('Block group Predictive Accuracy: {}'.format(reg.upper()),
                  fontdict=fontdict3)
        plt.grid(grid_on)
        ax3.set_facecolor('xkcd:light grey')
        # plt.xticks(list(range(len(bgs))), bgs,rotation='vertical')
        # ax3.xlabel('Block Group', fontdict={'size':15}, rotation='horizontal')
        #plt.ylabel('Predictive Accuracy', fontdict=fontdict2)
        plt.ylabel('Accuracy of Variable Block Group', fontdict=fontdict2)

        df.to_csv(filename, index_label='Block_Groups', )

        if False:
            for l in range(len(group_labels)):
                #if src == '':
                plt.text(xv[l]-.2, impv[l]+.0002, '{:.1f}'.format(df.loc[group_labels[l], 'Acc']*100), fontdict=fontdict2)
                #else:
                #    plt.text(xv[l] - .2, impv[l] + .0002, '{:.1f}'.format(df.loc[src, 'Acc'] * 100), fontdict=fontdict2)

        if save_it:
            fig3.savefig(plot_name)
        if show_it:
            plt.show()

    def plot_blocks_(self, filename, group_labels, reg, src, figsize=(10, 10), plot_name='{}_plot.svg',
                       show_it=False, dpi=600, figsize2=(10,6), save_it=False,
                        ):

        # labels
        fontdict2 = {
            'family': 'serif',
            # 'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'medium',
            'size': '10',
        }

        df = pd.read_csv(filename, index_col='Block_Groups')
        # df['Block_Groups'] = bgs
        if src is None:
            src = 'ALL'
        if src == 'income employment':
            src = 'inc/empl'


        xv = list()
        for v in range(len(group_labels)):
            if v == 0:
                xv.append(v)
            else:
                xv.append(v + .05)
        impv = df.loc[group_labels, 'Imp'].values.tolist()  # grab current values for improvement over null
        htsl = df.loc[group_labels, 'Acc'].values.tolist()  # grab current values for Accuracy
        # sub title
        fontdict3 = {
            # 'family': 'serif',
            'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'heavy',
            'size': '12',
        }
        plt.figure(figsize=figsize, dpi=dpi)
        fig3, ax3 = plt.subplots(1, 1, figsize=figsize2)  # create_subplot to get at fancy stuff
        # ax3.set_xticks(range(len(group_labels)))

        print('------------------')
        print('------------------')
        print('xv')
        print(xv)
        print('------------------')
        print('------------------')


        ax3.set_xticks(xv)  # use list generated from for loop above to set xpostions of bars
        ax3.set_xticklabels(group_labels, rotation='horizontal', fontdict={'size': 10})  # let up the names for the bars
        ax3.bar(xv, impv, align='center', color='green', )  # create the bar graph
        plt.title('Block group Predictive Accuracy: {}'.format(reg.upper()),
                  fontdict=fontdict3)
        ax3.set_facecolor('xkcd:light grey')
        # plt.xticks(list(range(len(bgs))), bgs,rotation='vertical')
        # ax3.xlabel('Block Group', fontdict={'size':15}, rotation='horizontal')
        # plt.ylabel('Predictive Accuracy', fontdict=fontdict2)
        plt.ylabel('Accuracy Improvement over Null', fontdict=fontdict2)

        df.to_csv(filename, index_label='Block_Groups', )

        for l in range(len(group_labels)):
            # if src == '':
            plt.text(xv[l] - .2, impv[l] + .0002, '{:.1f}'.format(df.loc[group_labels[l], 'Acc'] * 100),
                     fontdict=fontdict2)
            # else:
            #    plt.text(xv[l] - .2, impv[l] + .0002, '{:.1f}'.format(df.loc[src, 'Acc'] * 100), fontdict=fontdict2)
        if save_it:
            fig3.savefig(plot_name)
        if show_it:
            plt.show()


def get_hotspot_fips(df, metric, tile):
    # get the value that meets the limit
    threshold = df[metric].quantile(tile)
    # grab all fips that match the threshold
    print('metric: {}, threshold: {}'.format(metric, threshold))
    hotspots_df = df.loc[df[metric] >= threshold]
    num_fips = len(hotspots_df)
    return num_fips, hotspots_df['fips'].values.tolist()


def interactive_process_conversion_dictionary(conversions, conversion_dic=label_translation_dict, verbose=False,
                                              dict_name='new_dic'):
        for c in conversions:
            if c not in conversion_dic:
                new_c = input('What do you want to name the variable: {}'.format(c))
                conversion_dic[c] = new_c

        print('{} = \[\n'.format(dict_name))
        for c in conversion_dic:
                print('\t{}: {},'.format(c, conversion_dic[c]))
        print('\]\n')
        return conversion_dic

def process_conversion_dictionary(conversions, conversion_dic=label_translation_dict, verbose=False,
                                  interactive=False, ):
    if interactive:
        interactive_process_conversion_dictionary(conversions, conversion_dic=label_translation_dict, verbose=False,)

def get_hot_spot_data(df, metric, tile, region, vars=None, conversion_dict=None,
                      filenameHS = filenameHS,
                      filenamefips=filenamefips):

    stats_dict = {'Variable':list(),
                  'Difference from Pop. Mean':list(),
                  'Difference Magnitude':list(),
                  '% different':list(),
                  'Range':list(),
                  }
    num_fips, hot_fips = get_hotspot_fips(df, metric, tile)
    hot_df = df.loc[df['fips'].isin(hot_fips)]
    # now get its data
    hotspt_stats = hot_df.describe()
    pop_stats = df.describe()
    if vars is None:
        vars = hotspt_stats.columns.tolist()
    if conversion_dict is None:
        conversion_dict = {v:v for v in vars}
    print('The vars area of length ', len(vars))
    # now go through storing the means, std, and dif between pop
    for var in vars:
        stats_dict['Variable'].append(conversion_dict[var])
        stats_dict['Difference from Pop. Mean'].append(hotspt_stats.loc['mean', var]-pop_stats.loc['mean', var])
        stats_dict['Difference Magnitude'].append(abs(hotspt_stats.loc['mean', var]-pop_stats.loc['mean', var]))
        stats_dict['% different'].append((hotspt_stats.loc['mean', var]-pop_stats.loc['mean', var])/(pop_stats.loc['mean', var]))
        stats_dict['Range'].append('[{}-{}]'.format(np.around(hotspt_stats.loc['min', var], 4), np.around(hotspt_stats.loc['max', var], 4)))
    pd.DataFrame(stats_dict).sort_values(by=['Difference Magnitude'], ascending=False, inplace=False).to_excel(filenameHS.format(region, metric), index=False)
    pd.DataFrame({"fips": hot_fips}).to_excel(filenamefips.format(region, metric), index=False)
    return

def get_set_of_hot_spot_data(df, vars, metrics, tile, region, conversion_dict,
                             filenameHS=filenameHS, filenamefips=filenamefips,):
    #print(vars)
    #print(metrics)
    #quit()
    for metric in metrics:
        get_hot_spot_data(df, metric, tile, region, vars, conversion_dict, filenameHS, filenamefips,)


# https://www.epa.gov/greenpower/green-power-equivalency-calculator-calculations-and-references
def solar_MWH_by_state_calculator(df, states, to_check, r=.15, pr=.86):
    result_dict = {}
    totals_dict = {}
    for metric in to_check:
        print('For the metric: {}'.format(metric))
        result_dict[metric] = {}
        totals_dict[metric] = {}
        # go through each state summing the area, mul by daily solar and others
        for st in states:
            # grab the states data
            st_df = df.loc[df['state'] == st]
            print('the state of {}'.format(list(set(st_df['state']))))
            total_pv = st_df[metric].sum()
            totals_dict[metric][st] = total_pv
            print('{} total m^2'.format(total_pv))
            print('{} average daily solar radiation'.format(st_df['daily_solar_radiation'].mean()))
            result_dict[metric][st] = (( total_pv * r * st_df['daily_solar_radiation'].mean() * pr)/24)/1000

    for metric in result_dict:
        print('\t\t\t{}'.format(metric))
        for st in result_dict[metric]:
            #print('------------------')
            print('{:s}: {:.3f} MW daily possible average, {:.3f} MW yearly based on daily average'.format(st, result_dict[metric][st], result_dict[metric][st]*365))
            #print('------------------')
        print()
        print()

    return result_dict

# stores basic hyperparameters for a randomforest regressor and
# classifier
class Keeper_of_Params:
    def __init__(self):
        self.RFC_params = {
            'n_estimators': 200,
            'criterion': 'entropy',  #criterion{“gini”, “entropy”}, default=”gini”
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_features': 'auto',
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'min_impurity_split': None,
            'bootstrap': True,
            'oob_score': False,
            'n_jobs': None,
            'random_state': None,
            'verbose': 0,
            'warm_start': False,
            'class_weight': None,
            'ccp_alpha': 0.0,
            'max_samples': None
        }

        self.RFR_params = {
            'n_estimators':100,
            'criterion':'mse',
            'max_depth':None,
            'min_samples_split':2,
            'min_samples_leaf':1,
            'min_weight_fraction_leaf':0.0,
            'max_features':'auto',
            'max_leaf_nodes':None,
            'min_impurity_decrease':0.0,
            'min_impurity_split':None,
            'bootstrap':True,
            'oob_score':False,
            'n_jobs':None,
            'random_state':None,
            'verbose':0,
            'warm_start':False,
            'ccp_alpha':0.0,
            'max_samples':None
        }

        self.skLR_params = {}

    def get_params(self, m_type='rfclf'):
        if m_type in ['rfclf', ]:
            return self.RFC_params
        elif m_type in ['rfreg', ]:
            return self.RFR_params
        elif m_type in ['sklinReg', ]:
            return self.skLR_params
        else:
            return self.RFC_params

# selects based on the list of items that the sel key corresponds
# with in the regional dictionary passed
def US_regional_selector(df, region_col, region_dict=None, sel='S. East'):
    if region_dict is None:
        region_dict = {'West': ['ca', 'nv'],
                       'N. West': ['wa', 'or', 'id', 'mt', 'wy'],
                       'S. West': ['ut', 'az', 'nm', 'co', 'tx', 'ok'],
                       'M. West': ['nd', 'sd', 'ne', 'ks', 'mo', 'mn', 'ia', 'il', 'in', 'ky', 'oh', 'mi'],
                       'S. East': ['ar', 'la', 'ms', 'al', 'ga', 'fl', 'sc', 'nc', 'tn'],
                       'Mid_atlantic': ['pa', 'dc', 'de', 'nj', 'ny', 'md', 'wv', 'va'],
                       'N. East': ['ma', 'vt', 'me', 'nh', 'ri', 'ct']}

    return df.loc[df[region_col].isin(region_dict[sel]), :]

# removes based on the list of items that the sel key corresponds
# with in the regional dictionary passed
def US_regional_excluder(df, region_col, region_dict, sel='S. East'):
    if region_dict is None:
        region_dict = {'West': ['ca', 'nv'],
                       'N. West': ['wa', 'or', 'id', 'mt', 'wy'],
                       'S. West': ['ut', 'az', 'nm', 'co', 'tx', 'ok'],
                       'M. West': ['nd', 'sd', 'ne', 'ks', 'mo', 'mn', 'ia', 'il', 'in', 'ky', 'oh', 'mi'],
                       'S. East': ['ar', 'la', 'ms', 'al', 'ga', 'fl', 'sc', 'nc', 'tn'],
                       'Mid_atlantic': ['pa', 'dc', 'de', 'nj', 'ny', 'md', 'wv', 'va'],
                       'N. East': ['ma', 'vt', 'me', 'nh', 'ri', 'ct']}

    return df.loc[[not x for x in df[region_col].isin(region_dict[sel])], :]

def regional_plotter(df, cvar, figsize=(10,10), dpi=200, fontdict=None, verbose=True, width = .4):
    fontdict2 = {
        'family': 'serif',
        # 'family': 'sans-serif',
        # 'family': 'monospace',
        'style': 'normal',
        'variant': 'normal',
        'weight': 'medium',
        'color': 'black',
        'size': '15',
    }
    if fontdict is None:
        fontdict = fontdict2
    # create dictionary that will be keyed on region and store a ditionary of the
    # chosen comparison variable
    reg_stat_dict = dict()
    print(USRegions)
    for regs in list(USRegions.keys()):
        reg_stat_dict[regs] = {}
        print(regs)
        print('the states in region {}\n{}'.format(regs, USRegions[regs]))
        reg_stat_dict[regs]['avg'] = df.loc[df['state'].isin(USRegions[regs]), cvar].mean()
        reg_stat_dict[regs]['sum'] = df.loc[df['state'].isin(USRegions[regs]), cvar].sum()
    # now make lists for the bar graph

    xlbles = list(reg_stat_dict.keys())
    xv = list()
    for i in range(len(reg_stat_dict.keys())):
        if i == 0:
            #xv.append(width)
            xv.append(0)
        else:
            xv.append(np.around(xv[-1] + width+ width/2, 2))
    avgs, tots = list(), list()
    for regs in reg_stat_dict:
        avgs.append(reg_stat_dict[regs]['avg'])
        tots.append(reg_stat_dict[regs]['sum'])
    plt.close('all')
    plt.figure(figsize=figsize, dpi=dpi)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    #ax.bar(x_vals, avgs, tick_label=list(reg_stat_dict.keys()))
    ax.bar(xv, avgs, width=width, align='center')
    ax.set_xticklabels(xlbles, rotation=45, fontdict=fontdict)
    ax.set_xticks(xv)
    ax.set_facecolor('xkcd:light grey')
    #ax.ylabel(cvar + ' average')
    try:
        cvar = label_translation_dict[cvar]
    except KeyError:
        new_label = input('Need to add label for {}.\nWhat will its label be?\n:> '.format(cvar))
        label_translation_dict[cvar] = new_label
        cvar = new_label
        print('Add this to the label_translation_dict:\n"{}": "{}",'.format(cvar, new_label))

    plt.title(cvar + ' average', fontdict=fontdict)

    plt.figure(figsize=figsize, dpi=dpi)
    fig,  ax2 = plt.subplots(1, 1, figsize=figsize)
    #plt.bar(x_vals, tots, )
    #plt.bar(x_vals, tots, tick_label=list(reg_stat_dict.keys()), rotation=45)
    ax2.bar(xv, tots, width=width, align='center')
    ax2.set_xticklabels(xlbles, rotation=45, fontdict=fontdict)
    ax2.set_xticks(xv)
    ax2.set_facecolor('xkcd:light grey')
    #ax2.ylabel(cvar + ' total')
    #plt.xlabels(reg_stat_dict.keys(), rotation=45)
    plt.title(cvar + ' total', fontdict=fontdict)
    if verbose:
        plt.show()
    print(xv)
    print(xlbles)

def generate_income_split(df, params={'Iranges':None} ):

    if params['Iranges'] is not None:
        Iranges = params['Iranges']
    else:
        Iranges = income_dic
    ret_l = list()
    for Irange in Iranges:
        ret_l.append(select_by_med_income_range(df, Iranges[Irange], metric="median_household_income", ))
    return ret_l

def find_states_with(df, to_find):
    ret_df = df.filter(items=[to_find, 'state']).dropna()
    return set(ret_df['state'].values.tolist())
def state_selected_set(df, region_col, sel):
    return US_regional_selector(df, region_col, region_dict={"States": sel}, sel="States")

def generate_high_suitable_income_set(data_OG, params=None):
    df_vlow_hsuit = data_OG.loc[data_OG["verylow_own_Sbldg_rt"] > data_OG["verylow_own_Sbldg_rt"].mean(), :]
    df_low_hsuit = data_OG.loc[data_OG["low_own_Sbldg_rt"] > data_OG["low_own_Sbldg_rt"].mean(), :]
    df_mod_hsuit = data_OG.loc[data_OG["mod_own_Sbldg_rt"] > data_OG["mod_own_Sbldg_rt"].mean(), :]
    df_mid_hsuit = data_OG.loc[data_OG["mid_own_Sbldg_rt"] > data_OG["mid_own_Sbldg_rt"].mean(), :]
    df_high_hsuit = data_OG.loc[data_OG["high_own_Sbldg"] > data_OG["high_own_Sbldg"].mean(), :]
    return df_vlow_hsuit, df_low_hsuit, df_mod_hsuit, df_mid_hsuit, df_high_hsuit

def generate_state_list_set(data_OG, state_ls):
    return data_OG.loc[data_OG['state'].isin(state_ls), :]


def generate_locale_split(data_OG, params=None):
    rural_df = pd.DataFrame(
        locale_selector(data_OG, locale='locale_dummy', recoded_title='locale_recode(rural)', verbose=False))
    town_df = pd.DataFrame(
        locale_selector(data_OG, locale='locale_dummy', recoded_title='locale_recode(town)', verbose=False))
    suburban_df = pd.DataFrame(
        locale_selector(data_OG, locale='locale_dummy', recoded_title='locale_recode(suburban)', verbose=False))
    urban_df = pd.DataFrame(
        locale_selector(data_OG, locale='locale_dummy', recoded_title='locale_recode(urban)', verbose=False))
    return rural_df, town_df, suburban_df, urban_df

def generate_solar_split(df, params=None):
    Hgh = grab_threshold_selected_regions_quantile(df, threshold_met='daily_solar_radiation', qtthrsh=.5, cmp='gt')
    low = grab_threshold_selected_regions_quantile(df, threshold_met='daily_solar_radiation', qtthrsh=.5, cmp='lte')
    return [low, Hgh]

def GANOVA(data_df, analysis_feats, group_method, method_params, group_labels):
    fvalue_d, pvalue_d, mu_d = {}, {}, {}
    groups = group_method(data_df, method_params)

    for feature in analysis_feats:
        if len(groups) == 2:
            fvalue, pvalue = stats.f_oneway(groups[0][feature].dropna(), groups[1][feature].dropna(),)
        elif len(groups) == 3:
            fvalue, pvalue = stats.f_oneway(groups[0][feature], groups[1][feature], groups[2][feature],)
        elif len(groups) == 4:
            fvalue, pvalue = stats.f_oneway(groups[0][feature].dropna(), groups[1][feature].dropna(),
                                            groups[2][feature].dropna(),groups[3][feature].dropna(),)
        elif len(groups) == 5:
            fvalue, pvalue = stats.f_oneway(groups[0][feature].dropna(), groups[1][feature].dropna(),
                                            groups[2][feature].dropna(), groups[3][feature].dropna(),
                                            groups[4][feature].dropna(),)
        print(feature)
        print(fvalue, pvalue)
        print('=============================\n\n')
        fvalue_d[feature] = fvalue
        pvalue_d[feature] = pvalue

    big_plot_d = {}
    for sut in list(fvalue_d.keys()):
        plot_dic = {}
        print()
        print('Metric: {}, fvalue: {}, pvalue: {}'.format(sut, fvalue_d[sut], pvalue_d[sut]))
        for gl in group_labels:
            plot_dic[gl] = [groups[group_labels.index(gl)][sut].mean() ]
        big_plot_d[sut] = plot_dic
        print()

    return fvalue_d, pvalue_d, big_plot_d

def GchiSquare(df, targets, group_method, params=None, group_labels=[]):
    fvalue_d, pvalue_d, mu_d = {}, {}, {}
    groups = group_method(df, params)
    for i in range(len(groups)-1):
        for j in range(i+1, len(groups)):
            fvalue, pvalue = stats.chisquare(groups[i][targets], groups[j][targets])
            mu_d[group_labels[i] + '+' + group_labels[j]] = [fvalue, pvalue]
    for com in mu_d:
        print('\n---------------------------------------------------------------')
        print(com)
        print('fstat: {}, pvalue: {}'.format(mu_d[com][0], mu_d[com[1]]))
        print('---------------------------------------------------------------\n')
    return mu_d

# select parts of deep solar based on various groupings or selections
class Selector_Tool:
    def __init__(self):
        self.var = None

    def select_DS_data(self, df, region='US', sel=[], state_l=None, regional_col='state', region_dict=None,
                       excl=[], verbose=True, met_thresh=.5, threshold_metric='daily_solar_radiation', ):
        if isinstance(sel, type(str(''))):
            sel = [sel]
        print(region)

        if region in list(USRegions.keys()):
            print('selecting by region: {}'.format(region))
            if region_dict is None:
                region_dict = USRegions
            # do a regional select
            return self.US_regional_selector(df, region_col=regional_col, region_dict=region_dict, sel=region)
        elif region in ['US', 'state_by_state']:

            if len(state_l) > 0:
                print('----------------\n')
                print('Pulling the States: {}'.format(state_l))
                print('----------------\n')
                return pd.DataFrame(df.loc[df[regional_col].isin(state_l), :])
            print('US selected'.format(region))
            # just grab entire US
            if len(excl) != 0:
                return self.US_regional_excluder(df, regional_col, region_dict, sel=excl)
            return df
        elif region in ['US_town', 'US_urban', 'US_suburban','US_rural',]:
            suff = region.strip().split('_')[1]
            print('\n\n=================================================')
            print('Looking at {} locales'.format(suff))
            print('=================================================\n\n')

            return locale_selector(df, 'locale_dummy', recoded_title=suffix_to_locale[suff])
        elif region in ['low', 'high', 'US_low', 'US_high', 'US_low_low', 'US_low_high',
                        'US_high_low', 'US_high_high', ]:
            print(' ================================================')
            print(' ====   The Binary Solar RADs analysis   ========')
            print(' ====        The Region: {}      ========'.format(region))
            print(' ================================================')
            print()
            cmp = 'lt'
            if region in ['low', 'US_low','US_low_low', 'US_low_high'] or region not in ['low', 'high', 'US_low',
                                                                                         'US_high', 'US_low_low', 'US_low_high',
                                                                                         'US_high_low', 'US_high_high', ]:
                cmp = 'lt'
            elif region in ['high', 'US_high','US_high_low', 'US_high_high']:
                cmp = 'gte'
            return grab_threshold_selected_regions_quantile(df,threshold_met=threshold_metric,
                                                     qtthrsh=met_thresh, cmp=cmp)
        elif region in list(solr_groupings_percentiles.keys()):
            print('selecting by solar radiation: {}'.format(region))
            if region_dict is None:
                region_dict = solr_groupings_percentiles
            # grab areas based on solar radiation
            return self.Solar_Radiation_group_selector(df, region_col=regional_col, solar_dict=region_dict,
                                                       solar_cat=region)
        elif region in ['state_l', ]:
            # grab areas based on passed state_l argument
            if state_l is not None:

                ret_df = df.loc[df[regional_col].isin(state_l), :]
                print('the states left in the set are: {}'.format(set(ret_df['state'].values.tolist())))
                return ret_df
            return df


    def US_regional_selector(self, df, region_col, region_dict, sel='S. East'):
        """
                This can be used to select one of the 7 regions of the US
        :param df: data frame with a column of states names/abbreviations or similar to select states
        :param region_col: the column name that has the state identifiers
        :param region_dict: a dictionary that is keyed on region names, with values of lists of states
        :param sel: the selection of region
        :return: only those parts of the DF that pertain the the "sel" region
        """
        if region_dict is None:
            region_dict = {'West': ['ca', 'nv'],
                           'N. West': ['wa', 'or', 'id', 'mt', 'wy'],
                           'S. West': ['ut', 'az', 'nm', 'co', 'tx', 'ok'],
                           'M. West': ['nd', 'sd', 'ne', 'ks', 'mo', 'mn', 'ia', 'il', 'in', 'ky', 'oh', 'mi'],
                           'S. East': ['ar', 'la', 'ms', 'al', 'ga', 'fl', 'sc', 'nc', 'tn'],
                           'Mid_atlantic': ['pa', 'dc', 'de', 'nj', 'ny', 'md', 'wv', 'va'],
                           'N. East': ['ma', 'vt', 'me', 'nh', 'ri', 'ct']}
        return df.loc[df[region_col].isin(region_dict[sel]), :]

    def state_excluder(self, df, state_col, excl=[], verbose=False):
        if len(excl) == 0:
            return df
        if verbose:
            print('----------------------')
            print('removing the states')
            print(excl)
            print('----------------------')
            print()
        return df.loc[[not x for x in df[state_col].isin(excl)], :]

    def US_regional_excluder(self, df, region_col, region_dict, sel='S. East'):
        """
            performs the inverse action of the regional selector, selecting all regions but the one
            passed in the sel argument. Takes a little longer due to the inversion operation
        :param df:
        :param region_col:
        :param region_dict:
        :param sel:
        :return:
        """
        if region_dict is None:
            region_dict = {'West': ['ca', 'nv'],
                           'N. West': ['wa', 'or', 'id', 'mt', 'wy'],
                           'S. West': ['ut', 'az', 'nm', 'co', 'tx', 'ok'],
                           'M. West': ['nd', 'sd', 'ne', 'ks', 'mo', 'mn', 'ia', 'il', 'in', 'ky', 'oh', 'mi'],
                           'S. East': ['ar', 'la', 'ms', 'al', 'ga', 'fl', 'sc', 'nc', 'tn'],
                           'Mid_atlantic': ['pa', 'dc', 'de', 'nj', 'ny', 'md', 'wv', 'va'],
                           'N. East': ['ma', 'vt', 'me', 'nh', 'ri', 'ct']}
        # invert the boolean indices to get those that ar not in the selected region
        return df.loc[[not x for x in df[region_col].isin(region_dict[sel])], :]

    def Solar_Radiation_group_selector(self, df, region_col='state', solar_dict=None, solar_cat='very low'):
        if solar_dict is None:
            solar_dict = solr_groupings_percentiles
        return df.loc[df[region_col].isin(solar_dict[solar_cat]), :]

    def split_based_on_percentile(self, df, analysis_col, percentile_threshold=.95):
        thresh =df[analysis_col].quantile(percentile_threshold)
        hot = df.loc[df[analysis_col] >= thresh, :]
        n_hot = df.loc[df[analysis_col] < thresh, :]
        return hot, n_hot

# analyze deep solar in various ways
class Analysis_tools:
    def __init__(self):
        from _products.visualization_tools import Visualizer
        self.multi_lcn=False
        self.viz = Visualizer()
        self.var = None

    def htlpc(self, df, target, mtc=0, mxcc=1, corre=('kendall', 'pearson'),verbose=0):
        from _products.ML_Tools import LCN_transform
        X, Y = LCN_transform(df, target=target, mtc=mtc, mxcc=mxcc,
                             corrs=corre, verbose=verbose)
        lcn_feats = X.columns.tolist() + [target]
        print('correlation minimized list:\n', lcn_feats)
        return df[lcn_feats]

    # use for hot spot anlysis
    def mean_comparison(self, dfa, dfb, ):
        # get the dfa description
        dfa_desc = dfa.describe()
        dfb_desc = dfb.describe()
        # get the dfb description

    def smooth_outliers(self, df, feats, target,threshold=3, rmv_out=False, outlier_rmv='std',
                        smooth_data=False, smooth_alpha=.3, verbose=False):
        if smooth_data:
            return smooth_set(df, alpha=smooth_alpha, to_smooth=feats)
        else:
            if outlier_rmv == 'std':
                df = remove_outliers3(df, threshold=3, vars_to_check=feats, verbose=verbose)
                return df
            else:
                return remove_outliers(df, threshold=3, vars_to_check=feats, verbose=verbose)

    def prepare_data_for_regression(self, df, feats, target, scale_typ = 'minmax', verbose = True, outlier_rmv='std',
                                    scale_ignore = [], ts = .75,label_conversions=None, remove_outliers=False,
                                    threshold=3, smooth_data=False, smooth_alpha=.3, model_type='lin'):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        from sklearn.model_selection import train_test_split
        # if we want to smooth the data or scale the data use the data_tool for it
        if remove_outliers or smooth_data:
            df = self.smooth_outliers(df, feats, target,threshold=threshold, outlier_rmv=outlier_rmv,
                                      smooth_data=smooth_data, smooth_alpha=smooth_alpha, verbose=verbose)

        # get the scaler we need
        if scale_typ == 'minmax':
            std_sclr = MinMaxScaler()
        else:
            std_sclr = StandardScaler()

        # make dict with new labels as keys and old labels as vals to deconvert for the data frame
        if label_conversions is not None:
            #feats2 = {label_conversions[oldv]: oldv for oldv in label_conversions}
            feats2 = [label_conversions[oldv] for oldv in feats]
            target2 = label_conversions[target]
        else:
            feats2 = list(feats)
            target2 = str(target)

        # split data into randomized training and testing sets
        trsz = np.around(1 - ts, 2)
        if model_type == 'lin':
            Xtr, Xts, ytr, yts = train_test_split(df[feats], df[[target]], test_size=ts, train_size=trsz,)
        elif model_type == 'basic':
            shuffle_deck(df)
            trend = int(np.around(len(df)*trsz, 0))
            Xtr = df.filter(items=feats).loc[0:trend, :]
            ytr = df.filter(items=[target]).loc[0:trend, :]
            Xts = df.filter(items=feats).loc[trend:, :]
            yts = df.filter(items=[target]).loc[trend:, :]
            print('Basic:  ')
            return Xtr, ytr, Xts, yts
        else:
            Xtr, Xts, ytr, yts = train_test_split(df[feats], df[[target]], test_size=ts, train_size=trsz,
                                              stratify=df[[target]])
        Xtr = pd.DataFrame(Xtr, columns=feats, index=list(range(len(Xtr))))
        Xts = pd.DataFrame(Xts, columns=feats, index=list(range(len(Xts))))
        ytr_std = pd.DataFrame(ytr, columns=[target2], )
        yts_std = pd.DataFrame(yts, columns=[target2], )
        if scale_typ is None:
            print('No scaleing')
            return Xtr, ytr, Xts, yts
        # scale data based on scale_type
        if len(scale_ignore) > 0:
            store_theseTr = Xtr.filter(items=scale_ignore)
            store_theseTs = Xts.filter(items=scale_ignore)
        Xtr_scld = std_sclr.fit_transform(Xtr)
        Xts_scld = std_sclr.transform(Xts)

        if len(scale_ignore) > 0:
            for va in scale_ignore:
                if va in feats:
                    Xtr_scld[va] = store_theseTr[va]
                    Xts_scld[va] = store_theseTs[va]

        Xtr_std = pd.DataFrame(Xtr, columns=feats2, index=list(range(len(Xtr))))
        Xts_std = pd.DataFrame(Xts, columns=feats2, index=list(range(len(Xts))))


        return Xtr_std, ytr_std, Xts_std, yts_std


    def G_logistic_analysis(self, df, feats, target, scale_typ='minmax', verbose=True, scale_ignore=[],
                            ts=.75, remove_outliers=False, threshold=3, smooth_data=False, smooth_alpha=.3,
                            label_conversions=None, sound_file=None,  region_name='US'):
        from _products.utility_fnc import display_dic, blocking_sound_player
        Xtr_std, ytr_std, Xts_std, yts_std = self.prepare_data_for_regression(df, feats, target, scale_typ = scale_typ,
                                                                              verbose = verbose,
                                                                              scale_ignore = scale_ignore,
                                                                              ts = ts,
                                                                              remove_outliers=remove_outliers, threshold=threshold,
                                                                              smooth_data=smooth_data,
                                                                              smooth_alpha=smooth_alpha,
                                                                              label_conversions=label_conversions,
                                                                              model_type='log')
        # -------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------
        # Now lets Try the stats model version with standard scaling
        ysets = [ytr_std, ]
        ytest = [yts_std, ]
        xsets = [Xtr_std, ]
        xtest = [Xts_std, ]
        res = self.analyze_data(ysets, xsets, ytest, xtest, type='LogR', )

        if res > 0:
            print('-----------------------------------------------------')
            print('{} has some mixed vars issues moving on.......'.format(region_name.upper()))
            print('-----------------------------------------------------')
            print()
        else:
            # bsp(r'C:\Users\gjone\DeepLearningDeepSolar\__Media\sounds\great_scott.wav')
            print('-----------------------------------------------------------------------------')
            print('-----------------------------------------------------------------------------')
            print('-----------------------------------------------------------------------------')
            print()
            print()
            blocking_sound_player(sound_file)
        return


    def G_linear_analysis(self, df, feats, target, scale_typ='minmax', verbose=True, scale_ignore=[],
                            ts=.75, remove_outliers=False, threshold=3, smooth_data=False, smooth_alpha=.3,
                          label_conversions=None,sound_file='',  region_name='US', model_type='lin'):
        from _products.utility_fnc import display_dic, blocking_sound_player
        Xtr_std, ytr_std, Xts_std, yts_std = self.prepare_data_for_regression(df, feats, target, scale_typ=scale_typ,
                                                                              verbose=verbose,
                                                                              scale_ignore=scale_ignore,
                                                                              ts=ts,
                                                                              remove_outliers=remove_outliers,
                                                                              threshold=threshold,
                                                                              smooth_data=smooth_data,
                                                                              smooth_alpha=smooth_alpha,
                                                                              label_conversions=label_conversions,
                                                                              model_type=model_type)
        # -------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------
        # Now lets Try the stats model version with standard scaling
        ysets = [ytr_std, ]
        ytest = [yts_std, ]
        xsets = [Xtr_std, ]
        xtest = [Xts_std, ]
        print('the train\n', Xtr_std.head())
        res = self.analyze_data(ysets, xsets, ytest, xtest, type='LinR', )

        if res > 0:
            print('-----------------------------------------------------')
            print('{} has some mixed vars issues moving on.......'.format(region_name.upper()))
            print('-----------------------------------------------------')
            print()
        else:
            # bsp(r'C:\Users\gjone\DeepLearningDeepSolar\__Media\sounds\great_scott.wav')
            print('-----------------------------------------------------------------------------')
            print('-----------------------------------------------------------------------------')
            print('-----------------------------------------------------------------------------')
            print()
            print()
            blocking_sound_player(sound_file)
        return

    def G_Regression_Tool(self, df, feats, target, model_type = 'lin', scale_typ='minmax', verbose=True, scale_ignore=[],
                          ts=.75, remove_outliers=False, threshold=3, smooth_data=False, smooth_alpha=.3,
                          label_conversions=None,sound_file='',  region_name='US'):
        if model_type == 'lin':
            self.G_linear_analysis( df, feats, target, scale_typ=scale_typ, verbose=verbose, scale_ignore=scale_ignore,
                                    ts=ts, remove_outliers=remove_outliers, threshold=threshold, smooth_data=smooth_data,
                                    smooth_alpha=smooth_alpha, label_conversions=label_conversions, sound_file=sound_file,
                                    region_name=region_name)
        else:
            self.G_linear_analysis(df, feats, target, scale_typ=scale_typ, verbose=verbose,
                                   scale_ignore=scale_ignore,
                                   ts=ts, remove_outliers=remove_outliers, threshold=threshold,
                                   smooth_data=smooth_data,
                                   smooth_alpha=smooth_alpha, label_conversions=label_conversions,
                                   sound_file=sound_file,
                                   region_name=region_name)

    def analyze_region(self, df, params, target, run_count, label_conversions, trsz, top=-20, ret_tr_tst=False,
                       scl_method='minmax', model_typ='classification', region_name='US', met_thrsh=.5, just_analyze=False,
                       sound_files=Alert_sounds, state_col='state', state_drop_dic={}, thrsh_met='daily_solar_radiation',
                       solar_groups=['low', ], group_drop_dic=group_drop_dic, state_l=[], use_lcn=False, title='',
                       mtc=0, mxcc=1, corre=('kendall', 'pearson'), verbose=False, multi_lcn=False, all_in=True,
                       figsize=(15,15), fontdict=None, rmv_outliers=False, drop_states=True, perform_OLS=True):
        self.multi_lcn = multi_lcn
        self.ret_tr_tst= ret_tr_tst
        self.fontdict = fontdict
        self.figsize = figsize
        self.rmv_outliers = rmv_outliers
        self.drop_states_feats = drop_states
        if fontdict is None:
            fontdict = {
                # 'family': 'serif',
                'family': 'sans-serif',
                # 'family': 'monospace',
                'style': 'normal',
                'variant': 'normal',
                'weight': 'heavy',
                'size': '15',
            }

        if region_name in  ['US', 'US_low', 'US_high', 'US_low_low', 'US_low_high', 'US_high_low', 'US_high_high', 'US_rural', 'US_town', "US_suburban", "US_urban"] or region_name in list(USRegions.keys()):

            from _products.ML_Tools import LCN_transform
            print(' ================================================')
            print(' ======    The United states analysis    ========')
            print(' ======      region: {}    ==='.format(region_name))
            print(' ================================================')
            print()
            cols, top_cols = {}, {}
            if use_lcn:
                X, Y = LCN_transform(df, target=target, mtc=mtc, mxcc=mxcc,
                                     corrs=corre, verbose=verbose)
                lcn_feats = X.columns.tolist() + Y.columns.tolist()
                # print('correlation minimized list:\n', lcn_feats)
                df_state = df[lcn_feats]
            else:
                if region_name in ['US', 'US_low', 'US_high', 'US_low_low', 'US_low_high', 'US_high_low', 'US_high_high','US_rural', 'US_town', "US_suburban", "US_urban"]:
                    drop_list = select_drops.copy()
                else:
                    drop_list = US_drops.copy()
                drop_list_helper = list(drop_list)
                # check for the drops in the df and make sure they are there to drop
                for f in drop_list_helper:
                    if f not in df.columns.tolist():
                        del drop_list[drop_list.index(f)]
                        if verbose:
                            pass
                            #print('dropping ', f)
                            #print(len(drop_list))
                if verbose:
                    print('now it is ', drop_list)
                df = df.drop(drop_list, axis=1,)
            if not self.ret_tr_tst:
                cols[region_name], top_cols[region_name] = self.Multi_State_analysis(df, params, target, run_count,
                                                                                 label_conversions, trsz,
                                                                                 top=top, scl_method=scl_method,
                                                                                 model_typ=model_typ,
                                                                                 region_name=region_name,
                                                                                 sound_file=sound_files[0],
                                                                                 mtc=mtc, mxcc=mxcc, figsize=figsize,
                                                                                 corre=corre, verbose=verbose,
                                                                                 use_lcn=multi_lcn, title=title,
                                                                                 fontdict=fontdict, all_in=all_in,
                                                                                     perform_OLS=perform_OLS,)
                return cols, top_cols
            else:
                cols[region_name], top_cols[region_name], TR_D = self.Multi_State_analysis(df, params, target, run_count,
                                                                                     label_conversions, trsz,
                                                                                     top=top, scl_method=scl_method,
                                                                                     model_typ=model_typ,
                                                                                     region_name=region_name,
                                                                                     sound_file=sound_files[0],
                                                                                     mtc=mtc, mxcc=mxcc, figsize=figsize,
                                                                                     corre=corre, verbose=verbose,
                                                                                     use_lcn=multi_lcn, title=title,
                                                                                           fontdict=fontdict,
                                                                                           all_in=all_in,
                                                                                           perform_OLS=perform_OLS,)

                return cols, top_cols, TR_D

        elif region_name in list(solr_groupings_percentiles.keys()):
            print(' ================================================')
            print(' ========   The Solar RADs analysis   ===========')
            print(' ================================================')
            print()
            return self.iterative_state_analysis_Solar_Rad_G(df, params, target, run_count,
                                             label_conversions, trsz, state_col=state_col, model_typ=model_typ,
                                             solar_groups=solar_groups, top=top, method=scl_method,
                                             group_drop_dic=group_drop_dic, Alert_sound=None,
                                                             use_lcn=use_lcn, mtc=mtc, mxcc=mxcc, corre=corre)

        elif region_name == 'state_by_state' and 0 < len(state_l):
            print(' ================================================')
            print(' ===================    State    ================')
            print(' ================================================')
            print()
            return self.iterative_state_analysis(df, params, target, run_count, label_conversions, trsz, sound_files,
                                          states_l=state_l, state_col=state_col, top=top, method=scl_method,
                                          state_drop_dic=state_drop_dic, clf_reg=model_typ, use_lcn=use_lcn, mtc=mtc,
                                                 mxcc=mxcc, corre=corre)

    def find_significant(self, x, pvals):
        cnt = -1
        for e in pvals:
            if cnt > -1:
                print(x[cnt], ":", np.around(e, 4))
            cnt += 1

    def calculate_vif(self, x):
        return pd.Series([VIF(x.values, i)
                          for i in range(x.shape[1])],
                         index=x.columns)

    def calculate_log_like(self, attribs, params):
        # attribs.append('const')
        l = []
        for attrib in attribs:
            l.append(params[attrib])
        return np.exp(l).tolist()

    def G_Cox_Snell_R2(self, llnull, llmodel, n):
        v = 2 / n
        print('v', v)
        va = np.exp(llnull)
        vb = np.exp(llmodel)
        print('va, vb', va, vb)
        return 1 - (va / vb) ** v

    def stats_models_linearRegression_analysis(self, X, Xt, Y, Yt, columns, verbose=False):
        n = len(X)
        cols = columns
        #Y = np.around(Y.values.reshape(len(Y), 1), 4)
        #Yt = np.around(Yt.values.reshape(len(Yt), 1), 4)
        Y = Y.values
        Yt = Yt.values
        vif = self.calculate_vif(X)
        df_vif = pd.DataFrame(vif)
        for v in df_vif:
            print('- ', v)
        dir(vif)
        dir(df_vif)
        if True:
            print('---------------------------------------------------------------')
            print('---------------------------------------------------------------')
            print('---------------------------------------------------------------')
            print('---------------------------------------------------------------')
            print('VIF:\n', vif)
            print('---------------------------------------------------------------')
            print('---------------------------------------------------------------')
            print()

        # create the estimator
        est = sm.OLS(Y, X)
        #est = sm.GLM(Y.values, X)
        #est = sm.GLS(Y.values, X)

        # fit the estimator and grab the fitted estimator to analyze results
        try:
            est2 = est.fit()
        except:
            print('We got some bad variables!!!!\nCheck the VIF scores')
            blocking_sound_player(error_sounds[0])
            return 1
        # grab the Macfadden rsquare
        rsqr = est2.rsquared
        pvals = est2.pvalues
        fval = est2.fvalue
        ftest = est2.f_test

        if verbose:
            print('R-squared:', rsqr)
            print('P-values:\n', pvals)
            self.find_significant(cols, pvals)
        print('---------------------------------------------------------------')
        print('---------------------   Regression Analysis   -----------------')
        print('---------------------------------------------------------------')
        print(est2.summary())
        print('---------------------------------------------------------------')
        print('---------------------------------------------------------------')
        print(dir(est2.summary()))
        yp = est2.predict(Xt)
        mse = mean_squared_error(Yt, yp)
        mae = mean_absolute_error(Yt, yp)
        print('Mean Absolute Error: {:.4f}'.format(mae))
        print('Mean Squared Error: {:.4f}'.format(mse))
        return 0

    def stats_models_logisticRegression_analysis(self, X, Xt, Y, Yt, columns, verbose=False):
        n = len(X)
        cols = columns
        vif = self.calculate_vif(X)

        if True:
            print('---------------------------------------------------------------')
            print('---------------------------------------------------------------')
            print('---------------------------------------------------------------')
            print('---------------------------------------------------------------')
            print('VIF:\n', vif)
            print('---------------------------------------------------------------')
            print('---------------------------------------------------------------')
            print()

        # create the estimator
        est = dis_mod.Logit(Y.values, X)

        # try to fit the estimator and grab the fitted estimator to analyze results
        try:
            est2 = est.fit()
        except:
            print('We got some bad variables!!!!\nCheck the VIF scores')
            blocking_sound_player(error_sounds[0])
            return 1
        # grab the Macfadden rsquare
        loglikly = self.calculate_log_like(X, est2.params)


        rsqr = est2.prsquared
        #pvals = est2.pvalues
        #fval = est2.fvalue
        #ftest = est2.f_test

        if verbose:
            print('R-squared:', rsqr)
            #print('P-values:\n', pvals)
            #self.find_significant(cols, pvals)
        print('---------------------------------------------------------------')
        print('---------------------   Regression Analysis   -----------------')
        print('---------------------------------------------------------------')
        print(est2.df_model)
        print(est2.summary())
        print('---------------------------------------------------------------')
        print('---------------------------------------------------------------')
        print()
        llfv = est2.llf
        llnullv = est2.llnull
        # now show log likelyhoods
        print('McFadden’s pseudo-R-squared: ', 1 - (llfv / llnullv))  # https://statisticalhorizons.com/r2logistic
        cxsn = self.G_Cox_Snell_R2(llnullv, llfv, n)
        print('Cox\'s Snell: {}'.format(cxsn))
        # print('model 2',dir(model2))
        print('R squared:', est2.prsquared)  # McFadden’s pseudo-R-squared.
        # print(dir(model2.summary().tables))

        # show_labeled_list(loglikly, x)
        print()
        print('\t\t\tThe log likelyhoods are:')
        for l, lbl in zip(loglikly, X):
            # get the length of label
            lbl_len = len(lbl)
            # calculate how many spaces needed to get 50
            # then create a string of that many
            spcs = 50 - lbl_len
            spc_str = ''
            for i in range(spcs):
                spc_str += ' '
            print('{:s}:{}{:.4f}'.format(lbl, spc_str, l))
        # print('pvalue for {:s}: {:f}'.format(X2.columns.values.tolist()[0], model2.pvalues.loc[x.columns.values.tolist()[0]]))
        y_pred = est2.predict(Xt, linear=True)
        # print(y_pred)
        yp = list()
        for e in y_pred:
            if e > 0:
                yp.append(1)
            else:
                yp.append(0)
        # print(model.loglikeobs(x))
        # df_confusion = pd.crosstab(Y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
        self.viz.plot_confusion_matrix(Yt, yp, classes=['NA', 'A'],
                                  title='Confusion matrix')

        plt.show()

        return 0

    def analyze_data(self, ysets, xsets, ytest, xtest, type='LinR', verbose=False, add_const=True):
        roptions = ['LinR', 'LogR']
        columns = xsets[0].columns.tolist()
        if type not in roptions:
            print('Unknown regression type: {}'.format(type))
            print('options are: {}'.format(roptions))

        old_rsqr = 0

        for y, yt in zip(ysets, ytest):
            cnt = 0
            for x, xt in zip(xsets, xtest):
                Y = y
                Yt = yt
                X = x
                Xt = xt
                if verbose:
                    print('################################################################################')
                    print('\t\tX or dependent variables:\n', x.columns.values.tolist())
                    print('################################################################################')

                # create a version of the X part of the data with a 1 added
                if add_const:
                    X2 = sm.add_constant(X)
                    Xt2 = sm.add_constant(Xt)
                    columns = ['const'] + columns
                else:
                    X2 = X
                    Xt2 = Xt
                if type in ['LinR']:
                    return self.stats_models_linearRegression_analysis(X2, Xt2, Y, Yt, columns, verbose=verbose)
                elif type in ['LogR']:
                    return self.stats_models_logisticRegression_analysis(X2, Xt2, Y, Yt, columns, verbose=verbose)


    def Multi_State_analysis(self, df, params, target, run_count, label_conversions, trsz, ret_tr_tst=False,
                             top=-20, scl_method='minmax', model_typ='classification', region_name='US', title='',
                             sound_file=r'C:\Users\gjone\DeepLearningDeepSolar\__Media\_sounds\ahh_ha.wav',
                             use_lcn=False, mtc=0, mxcc=1, corre=('kendall', 'pearson'), verbose=True,
                             figsize=(15,15), fontdict=None, perform_OLS=True, all_in=True,):
        from _products.utility_fnc import display_dic, blocking_sound_player
        from _products.ML_Tools import feature_importance_logger_df_mixer, make_me_a_box
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        from sklearn.ensemble import RandomForestClassifier as RFC
        from sklearn.ensemble import RandomForestRegressor as RFR
        print('=========================================================')
        print('the top in Multi state is: {}'.format(top))
        print('=========================================================')
        print('the model type--> ', model_typ)
        print('features\n', df.columns.tolist())
        #print('the length {}'.format(len(df)))
        print('=========================================================\n')
        if self.rmv_outliers:
            df = remove_outliers(df, threshold=3, verbose=True)
        else:
            print('Not removing outliers')

        # set up a scaler for the logistic poriton
        if scl_method == 'minmax':
            std_sclr = MinMaxScaler()
        elif scl_method == 'std':
            std_sclr = StandardScaler()
        else:
            std_sclr = None

        if model_typ.lower() == 'regression':
            reg = 'LinR'
            RF = RFR
            justify=False
        else:
            reg = 'LogR'
            RF = RFC
            justify=True

        feats = df.columns.tolist()
        del feats[feats.index(target)]
        # perform given number of random forest FI runs to get a list of
        # top variables to try in a regression based on model_type
        box_data, cols, cm, rf, avg_scr, std_df, mu_df = feature_importance_logger_df_mixer(params, df, feats, target=target,
                                                                                 runs=run_count, justify=justify,
                                                                                 label_conversion=label_conversions,
                                                                                 special=True, tr=(trsz,), plot_fi_stat=True,
                                                                                 all_in=all_in, model=model_typ, )

        verbose=True
        if verbose:
            print('\n\n--------------------------')
            display_dic(avg_scr, prec=.3)
            print('the sorted columns:\n', cols[top:])
            print('---------------------------\n\n')
        title = 'Feature Importance for {}'.format(region_name.upper())

        # make dict with new labels as keys and old labels as vals to deconvert for the data frame
        if label_conversions is not None:
            recon_label = {label_conversions[oldv]: oldv for oldv in label_conversions}

        # use sorted list of featurs cols to get the top cols for display
        # and regression
        if label_conversions is not None:
            df_cols = [recon_label[new_cl] for new_cl in cols]
            df_top_cols = list(df_cols[top:])

        else:
            df_cols = cols
            df_top_cols = list(cols[top:])
        if verbose:
            print('the top columns as seen in the data file:')
            print('{}\n'.format(df_top_cols))


        # creates a box plot form feature importance logger
        make_me_a_box(box_data[top:], cols[top:], title, figsize=self.figsize, fontdict=self.fontdict)

        # grab the top 20 features for the logReg Analysis
        top_cols = cols[top:]
        pfi_cols = list(cols[top:].copy())
        bottom_cols = cols[0: abs(top)]
        if self.multi_lcn:
            df_top_cols = self.htlpc(df[df_top_cols + [target]], target, mtc=mtc, mxcc=mxcc,
                                  corre=corre, verbose=verbose).columns.tolist()
            del df_top_cols[df_top_cols.index(target)]
        ts = np.around(1 - trsz, 2)
        if model_typ == 'classification':
            if verbose:
                As = (df.loc[df[target] == 1, :].shape[0] / df.shape[0]) * 100
                NA = (df.loc[df[target] == 0, :].shape[0] / df.shape[0]) * 100
                print('\n -------------   Class Break down ----------------------')
                print('there area {:.3f}% A\'s and {:.3f}% NA\'s for the state of {:s}'.format(As, NA, region_name))
                print('\n --------------------------------------------------------\n')
            # do a Reg Analysis and show results
            if target in df_top_cols:
                del df_top_cols[df_top_cols.index(target)]
            print('the top cols and what they are: {}\n{}'.format(len(df_top_cols), df_top_cols))
            df0 = df[df_top_cols]
            print('shape form the {} columns {}'.format(df0.shape, df_top_cols))
            Xtr, Xts, ytr, yts = train_test_split(df[df_top_cols], df[[target]], test_size=ts, train_size=trsz,
                                                  stratify=df[[target]])
            print('the shape of tr and ts x {}, {}'.format(Xtr.shape, Xts.shape))
        else:
            avg_rate = df[target].mean()
            print('The average rate of {:s} adoption is {:.2f}'.format(target, avg_rate))
            # do a Reg Analysis and show results
            Xtr, Xts, ytr, yts = train_test_split(df[df_top_cols], df[target], test_size=ts, train_size=trsz,
                                                  )
        x_vals = []
        width = .2

        #std_sorter = pd.DataFrame({'values': std_df.values[0].tolist(),
        #                           'cols': std_df.columns.tolist()}).sort_values(by=['values'], axis=0)
        #sorted_std_cols = std_sorter['cols'].values.tolist()
        #print('the sorted columns for the std')
        #print(sorted_std_cols)
        #print('-----------------------------\n\n')
        for i in range(len(df_top_cols)):
            x_vals.append(width + width*i)

        # grab permutated feature importance statistics
        std_df = std_df.filter(items=pfi_cols)
        #std_df = std_df.sort_values(by=std_df.index.tolist(), axis=1, )
        mu_df = mu_df.filter(items=std_df.columns.tolist())

        x_cntrs = np.arange(len(pfi_cols))
        num_bars = 2
        nrows, ncols = 1, 1
        fig, axstd = plt.subplots(nrows, ncols, figsize=figsize, )
        #print(std_df.values)
        axstd.barh(x_cntrs - width/num_bars, std_df.values[0].tolist(), width, label='std')
        axstd.barh(x_cntrs + width/num_bars, mu_df.values[0].tolist(), width,  label='mu')
        axstd.set_yticklabels(std_df.columns.tolist(), rotation=0, )
        axstd.set_yticks(x_cntrs)
        axstd.set_ylim(0-(width/2)*4, len(x_cntrs))
        axstd.set_title('Std and Mu of PFI Rankings')
        axstd.set_xlabel('std & mu for PFI')
        axstd.legend()

        x_cntrs = np.arange(len(top_cols))
        std_df = std_df.filter(items=top_cols)
        std_df = std_df.sort_values(by=std_df.index.tolist(), axis=1, )
        mu_df = mu_df.filter(items=std_df.columns.tolist())

        fig, axstd = plt.subplots(nrows, ncols, figsize=(12, 12))
        #print(std_df.values)
        axstd.barh(x_cntrs - width / num_bars, std_df.values[0].tolist(), width, label='std')
        axstd.barh(x_cntrs + width / num_bars, mu_df.values[0].tolist(), width, label='mu')
        axstd.set_yticklabels(std_df.columns.tolist(), rotation=0, )
        axstd.set_yticks(x_cntrs)
        axstd.set_ylim(0-(width/2)*4, len(x_cntrs))
        axstd.set_title('Std and Mu of PFI Rankings: Std sorted')
        axstd.set_xlabel('std & mu for PFI')
        axstd.legend()

        x_cntrs = np.arange(len(top_cols))
        mu_df = mu_df.filter(items=top_cols)
        mu_df = mu_df.sort_values(by=mu_df.index.tolist(), axis=1, )
        std_df = std_df.filter(items=mu_df.columns.tolist())

        fig, axstd = plt.subplots(nrows, ncols, figsize=(12, 12))
        #print(std_df.values)
        axstd.barh(x_cntrs - width / num_bars, std_df.values[0].tolist(), width, label='std', color='r')
        axstd.barh(x_cntrs + width / num_bars, mu_df.values[0].tolist(), width, label='mu', color='g')
        axstd.set_yticklabels(std_df.columns.tolist(), rotation=0, )
        axstd.set_yticks(x_cntrs)
        axstd.set_ylim(0-(width/2)*4, len(x_cntrs))
        axstd.set_title('Std and Mu of PFI Rankings: Mu sorted')
        axstd.set_xlabel('std & mu for PFI')
        axstd.legend()


        '''
        fig2, axmu = plt.subplots()
        axmu.bar(x=x_vals, y=mu_df.values.tolist(), figsize=(20, 20))
        axmu.title(title='Mu of PFI')
        axmu.set_xticklabels(top_cols, rotation=45, )
        '''

        if std_sclr is not None:
            Xtr_std = pd.DataFrame(std_sclr.fit_transform(Xtr), columns=df_top_cols, index=list(range(len(Xtr))))
            Xts_std = pd.DataFrame(std_sclr.transform(Xts), columns=df_top_cols, index=list(range(len(Xts))))
        else:
            Xtr_std = pd.DataFrame(Xtr, columns=df_top_cols, index=list(range(len(Xtr))))
            Xts_std = pd.DataFrame(Xts, columns=df_top_cols, index=list(range(len(Xts))))

        ytr_std = pd.DataFrame(ytr, columns=[target], )
        yts_std = pd.DataFrame(yts, columns=[target], )

        if model_typ == 'classification':
            AsTR = (ytr_std.loc[ytr_std[target] == 1, :].shape[0] / ytr_std.shape[0]) * 100
            NATR = (ytr_std.loc[ytr_std[target] == 0, :].shape[0] / ytr_std.shape[0]) * 100

            AsTS = (yts_std.loc[yts_std[target] == 1, :].shape[0] / yts_std.shape[0]) * 100
            NATS = (yts_std.loc[yts_std[target] == 0, :].shape[0] / yts_std.shape[0]) * 100

            print('there area {:.3f}% A\'s and {:.3f}% NA\'s for the state of {:s} Training'.format(AsTR, NATR, region_name))
            print('there area {:.3f}% A\'s and {:.3f}% NA\'s for the state of {:s} Testing'.format(AsTS, NATS, region_name))
            print()
        if perform_OLS:
            # Now lets Try the stats model version with standard scaling
            ysets = [ytr_std, ]
            ytest = [yts_std, ]
            xsets = [Xtr_std, ]
            xtest = [Xts_std, ]
            res = self.analyze_data(ysets, xsets, ytest, xtest, type=reg, )

            if res > 0:
                           print('-----------------------------------------------------')
                           print('{} has some mixed vars issues moving on.......'.format(region_name.upper()))
                           print('-----------------------------------------------------')
                           print()
            else:
                # bsp(r'C:\Users\gjone\DeepLearningDeepSolar\__Media\sounds\great_scott.wav')
                print('-----------------------------------------------------------------------------')
                print('-----------------------------------------------------------------------------')
                print('-----------------------------------------------------------------------------')
                print()
                print()
                blocking_sound_player(sound_file)

        if self.ret_tr_tst:
            #return df_cols, df_top_cols, [Xtr_std, ytr_std, Xts_std, yts_std]
            return df_cols, df_top_cols, [Xtr, ytr, Xts, yts]

        return df_cols, df_top_cols

    def iterative_state_analysis(self, df, params, target, run_count, label_conversions, trsz, sound_files,
                                 states_l, state_col='state', top=-20, method='minmax', state_drop_dic={},
                                 clf_reg='classification', use_lcn=False, mtc=0, mxcc=1, corre=('kendall', 'pearson',),
                                 verbose=True):
        """
            this takes a list of states and analyzes each by first using random forest to detect the top
            "top" features and then uses those for a regression analysis
        :param df:
        :param params:
        :param featsl:
        :param target:
        :param run_count:
        :param label_conversions:
        :param trsz:
        :param sound_files:
        :param states_l:
        :param state_col:
        :param top:
        :param method:
        :param state_drop_dic:
        :param clf_reg:
        :return:
        """
        from _products.utility_fnc import display_dic, blocking_sound_player
        from _products.performance_metrics import analyze_data
        from _products.ML_Tools import feature_importance_logger_df_mixer, make_me_a_box
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        from _products.ML_Tools import LCN_transform
        print('the top in state by state is: {}'.format(top))
        # set up a scaler for the logistic poriton
        if method == 'minmax':
            std_sclr = MinMaxScaler()
        elif method == 'std':
            std_sclr = StandardScaler()

        # go through the state list one by one generating
        # FI, PI, LOGREG
        feats = df.columns.tolist()
        del feats[feats.index(target)]
        st_dic_col, st_dic_top, st_tr_dic = {}, {}, {}
        if len(sound_files) != len(states_l):
            tmpll = []
            while len(sound_files) + len(tmpll) < len(states_l):
                tmpll.append(sound_files[np.random.choice(list(range(len(sound_files))), 1, )[0]])
            sound_files += tmpll
        for state, sound_file in zip(states_l, sound_files):
            # get the data for the current state
            print('-------------------------------------------------------------------')
            print('The State of: {}'.format(state))
            print('-------------------------------------------------------------------')
            df_state = df.loc[df[state_col] == state, :][feats + [target]]

            df_state.drop(state_col, inplace=True, axis=1)

            drop_list = []
            if state_drop_dic is not None:
                drop_list = state_drop_dic[state].copy()
            # check for the drops in the df and make sure they are there to drop
                for f in state_drop_dic[state]:
                    if f not in df_state.columns.tolist():
                        del drop_list[drop_list.index(f)]

            if verbose:
                pass
                #print('now drop list is: ', drop_list)
            if use_lcn:
                print('Using lcn in state by state before multi state analysis')
                df_state = self.htlpc(df, target, mtc=mtc, mxcc=mxcc, corre=corre, verbose=verbose)
            else:
                if len(drop_list) > 0:
                    df_state.drop(drop_list, axis=1, inplace=True)
                if self.ret_tr_tst:
                    st_dic_col[state], st_dic_top[state], st_tr_dic[state] = self.Multi_State_analysis(df_state, params, target, run_count,
                                                                             label_conversions, trsz,
                                                                            top=top, scl_method=method,
                                                                            model_typ=clf_reg, region_name=state,
                                                                            sound_file=sound_file,
                                                                             mtc=mtc, mxcc=mxcc, corre=corre,
                                                                             verbose=verbose)
                else:
                    st_dic_col[state], st_dic_top[state] = self.Multi_State_analysis(df_state, params, target, run_count,
                                                                             label_conversions, trsz,
                                                                            top=top, scl_method=method,
                                                                            model_typ=clf_reg, region_name=state,
                                                                            sound_file=sound_file,
                                                                             mtc=mtc, mxcc=mxcc, corre=corre,
                                                                             verbose=verbose)
        if self.ret_tr_tst:
            return st_dic_col, st_dic_top, st_tr_dic
        else:
            return st_dic_col, st_dic_top

    def iterative_state_analysis_Solar_Rad_G(self, df, params, target, run_count, label_conversions,
                                             trsz, state_col='state', model_typ='classification',met_thresh=.5,
                                             solar_groups=['low', ], top=-20, method='minmax',thresh_mode=False,
                                             group_drop_dic=group_drop_dic, Alert_sound=Alert_sounds,
                                             use_lcn=False, mtc=0, mxcc=1, corre=('kendall', 'pearson'),
                                             verbose=False):
        #from _products.utility_fnc import display_dic, blocking_sound_player
        #from _products.performance_metrics import analyze_data
        #from _products.ML_Tools import feature_importance_logger_df_mixer, make_me_a_box
        #from _products.ML_Tools import LCN_transform
        #from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        print('=========================================================')
        print('the top in solar Rads is: {}'.format(top))
        print('=========================================================')
        # set up a scaler for the logistic poriton
        if method == 'minmax' or method not in ['minmax', 'std']:
            std_sclr = MinMaxScaler()
        elif method == 'std':
            std_sclr = StandardScaler()

        # go through the state list one by one generating
        # FI, PI, LOGREG
        slr_gd = {
            "very low": ['wi', 'mi', 'nh', 'oh', 'wa', 'nd', 'ma', 'mn', 'me', 'vt', 'pa'],
            "low": ['in', 'de', 'ny', 'ia', 'ct', 'ky', 'il', 'mt', 'ri', 'nj', 'sd', 'or', 'md', 'wv'],
            "medium": ['la', 'id', 'ks', 'va', 'dc', 'wy', 'ga', 'mo', 'sc', 'tn', 'al', 'nc',
                       'ms', 'ne', 'ar'],
            "high": ['ut', 'ok', 'az', 'ca', 'tx', 'co', 'nv', 'nm', 'fl'],
        }
        cnt = 0
        feats = df.columns.tolist()
        solar_rad_cols_d = {}
        solar_rad_topcols_d = {}
        del feats[feats.index(target)]

        if Alert_sound is not None:
            len_asounds = len(Alert_sound)
        else:
            Alert_sound = Alert_sounds
            len_asounds = len(Alert_sounds)
        for sgroup in solar_groups:
            # get the data for the current state
            print('-------------------------------------------------------------------')
            print('The Solar Group: {}'.format(sgroup))
            print('-------------------------------------------------------------------')
            if not thresh_mode:
                df_state = df.loc[df[state_col].isin(slr_gd[sgroup]), :][feats + [target]]
                df_state.drop(state_col, inplace=True, axis=1)
            else:
                cmp = 'lt'
                if sgroup == 'low':
                    cmp = 'lt'
                elif sgroup == 'high':
                    cmp = 'gte'
                df_state = grab_threshold_selected_regions_quantile(df,
                                                                    threshold_met=state_col,
                                                                    qtthrsh=met_thresh, cmp=cmp)

            drop_list = group_drop_dic[sgroup].copy()
            # check for the drops in the df and make sure they are there to drop
            for f in group_drop_dic[sgroup]:
                if f not in df_state.columns.tolist():
                    del drop_list[drop_list.index(f)]
                    if verbose:
                        pass
                        #print('dropping ', f)
                        #print(len(drop_list))
            if verbose:
                print('now it is ', drop_list)

            if use_lcn:
                '''
                X, Y = LCN_transform(df_state, target=target, mtc=mtc, mxcc=mxcc,
                                     corrs=corre, verbose=verbose)
                lcn_feats = X.columns.tolist() + Y.columns.tolist()
                # print('correlation minimized list:\n', lcn_feats)

                df_state = df_state[lcn_feats]
                '''
                df_state = self.htlpc(df, target, mtc=mtc, mxcc=mxcc, corre=corre, verbose=verbose)
            else:
                df_state.drop(drop_list, axis=1, inplace=True)
            #df_state.drop(state_col, inplace=True, axis=1)

            #df_state.drop(group_drop_dic[sgroup], axis=1, inplace=True)

            solar_rad_cols_d[sgroup], solar_rad_topcols_d[sgroup] = self.Multi_State_analysis(df_state, params, target,
                                                                                              run_count,
                                                                                              label_conversions, trsz,
                                                                                              top=top,
                                                                                              scl_method=method,
                                                                                              model_typ=model_typ,
                                                                                              region_name=sgroup,
                                                                                              sound_file=Alert_sound[cnt%len_asounds],
                                                                                              mtc=mtc, mxcc=mxcc,
                                                                                              corre=corre,
                                                                                              verbose=verbose)
        return solar_rad_cols_d, solar_rad_topcols_d

# visualize various aspects of solar
class Relationship_Visualizer:
    def __init__(self):
        from _products.visualization_tools import Visualizer
        self.viz = Visualizer()
        self.var = None

    fontdict = {
        # 'family': 'serif',
        'family': 'sans-serif',
        # 'family': 'monospace',
        'style': 'normal',
        'variant': 'normal',
        'weight': 'heavy',
        'size': '15',
    }
    colors = ['xkcd:warm brown', 'xkcd:pale gold', 'xkcd:mid green', 'xkcd:medium grey', 'red', 'blue', 'black']
    def multi_df_bar(self, dfl, a_var, var_l=None, df_titles=None, fontdict=None, xticks=None, title='',
                     yticks=None, xlim=None, ylim=None, width=.4, ypd=[.05, .001], xpd=[.05, .001]):
        from _products.utility_fnc import get_exception_type

        if fontdict is None:
            fontdict = self.fontdict


        if var_l is None:
            var_l = ['df_{}'.format(i) + ' {}' for i in range(len(dfl))]
            for i in range(len(var_l)):
                var_l[i] = var_l[i].format(a_var)
        else:

            for i in range(len(var_l)):
                var_l[i] = var_l[i].format(a_var)

        locale_stat_df = pd.DataFrame()
        for i in range(len(dfl)):
            locale_stat_df[var_l[i]] = [dfl[i][a_var].mean()]

        xmin = -(width/2) - (width/2)*xpd[0]
        xmax = len(dfl) * width + (width/2)*xpd[1]

        ymin =locale_stat_df.min(axis=1)[0]

        ymin = max(ymin - ymin * ypd[0], 0)
        ymax = locale_stat_df.max(axis=1)[0]
        ymax = ymax + ymax * ypd[1]

        if xticks is None:
            xticks = np.arange(0, len(dfl)*width, width)

        if yticks is None:
            #yticks = np.around(np.linspace(locale_stat_df.min(axis=1)[0], locale_stat_df.max(axis=1)[0], len(dfl)), 2)
            yticks = sorted(np.around(locale_stat_df.values[0], 3).tolist())
            #yticks = ['' for i in range(len(locale_stat_df.values[0].tolist()))]

        if ylim is None:
            ylim = (ymin, ymax)

        if xlim is None:
            xlim = (xmin, xmax)


        print('\nxticks: {}\nxlims: {}\nyticks: {}\nylims: {}\n'.format(xticks, xlim, yticks, ylim))

        #try:
        self.plot_df_bar(locale_stat_df, var_l=var_l, figsize=(20, 20), ylim=ylim, title=title, xlim=xlim,
                             xticks=xticks, yticks=yticks, width=width, fontdict=fontdict)
        #except:
        #    print('we got a {} exception'.format(get_exception_type()))

    def plot_df_bar(self, df, var_l, fontdict=None, title=None, ylim=None, width=.4,
                    xlim=None, fontsize=20, figsize=(20, 20), xticks=None, yticks=None):
        if fontdict is None:
            fontdict = self.fontdict
        if var_l is not None:
            print('[0] ',df[var_l].values[0])
            print(xticks)
            fig, axb = plt.subplots(1, 1, figsize=figsize)
            cnt = 0
            for x, mu, lbl in zip(xticks, df[var_l].values[0], var_l):
                axb.bar(x, mu, width=width, color=self.colors[cnt], label=lbl)
                cnt = (cnt + 1)%len(self.colors)
            #axb.bar(xticks, df[var_l].values[0], width=width, color=self.colors[:len(var_l)])
            #axb = df.plot.bar(x=xticks, y=var_l, figsize=figsize, width=width)
        else:
            fig, axb = plt.subplots(figsize=figsize)
            axb.bar(xticks, df[var_l].values[0], width=width, color=self.colors[:len(var_l)])
            axb = df.plot.bar()

        if title is not None:
            axb.set_title(title, fontdict=fontdict)
        if ylim is not None:
            axb.set_ylim(ylim, )
        if xlim is not None:
            axb.set_xlim(xlim, )

        if xticks is not None:
            print('xticks')
            axb.set_xticks(xticks)
            xlabl = ['' for i in range(len(var_l))]
            axb.set_xticklabels(xlabl, rotation=45, fontdict=fontdict)
        if yticks is not None:
            axb.set_yticks(yticks)
            axb.set_yticklabels(yticks, rotation=0, fontdict=fontdict)

        if var_l is not None:
            axb.legend(var_l, fontsize=fontsize)
        else:
            axb.legend()
        axb.grid(True)
        return
    # kind options: bar, barh, hist, box, dke = density, area, pie, scatter, hexbin
    def mass_x_y_plot(self, dfaa, st, deps, indeps, ed=None, verbose='', kind='line', clr=['tab:purple', 'r', 'b'],
                      bins=100, close_all=True):

        """
            Plots a list of independent variables (indeps) against a list of dependent variables (deps) in an
            iterative loop producing x vs y plots and generating an histogram for the values of x (deps) for
            the entire data set, adopters and non-adopters
        :param dfaa: data frame
        :param t: title base, modified as plots are generated so needs to have places for x, y vars see example below
                    Example title: '{} vs {}', will make a title x vs y
        :param deps: list of dependent variables
        :param indeps: list of independent variables
        :param st: ?
        :param ed: ?
        :param verbose: can be used but probably needs to be removed
        :param kind: type of pandas plot to produce, see:
                     https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html
        :param clr: colors to use for plotting if
        :param bins: used for histograms the number of bins to use
        :return:  None
        """
        if close_all:
            plt.close('all')
        fontdict2 = {
            'family': 'serif',
            # 'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'medium',
            'size': '10',
        }
        if ed is None:
            ed = st+1
        # grab an x value
        for sb in indeps[0:]:  # X
            # ignore string variables
            if isinstance(dfaa[sb].values[0], type('')) or sb in deps:
                print('Ignoring {}'.format(sb))
                continue
            else:
                for y in deps[st:ed]:  # Y
                    sort_bys = [sb]
                    dfaa.sort_values(by=sort_bys, inplace=True, ascending=True)
                    dfaa.plot(kind=kind, x=sort_bys[0], y=y,
                              title='{} vs. {}\n'.format(y, sort_bys[0]))
                    # df.hist(column=sb, bins=len(list(df[sb].values.tolist())))
                    #bins = len(set(dfaa[sb].values.tolist()))
                    #dfaa.hist(column=sb, bins=bins, color=clr[0])       # plot a histogram of x for all data
                    if y in solar_metricsB + ['Adoption']:   # if given a solar metric plot a histogram of A vs NA of dependent
                        # plt.figure()
                        #plt.title('{} histogram:'.format(sb))
                        if bins is None:
                            bins = len(set(dfaa[sb].values.tolist()))
                            if verbose:
                                print('The variable {} has {} unique values'.format(sb, bins))
                        #Nadp = pd.DataFrame(dfaa.loc[dfaa[y] == 0, sb]).hist(column=sb, color=clr[1], bins=bins)
                        #plt.title('{} histogram: Nonadopters'.format(sb))
                        #plt.legend(['Nonadopters'])
                        # grab the means for the bar plots below
                        Average_overall = dfaa.loc[:, sb].mean()
                        Nonadopter_average = dfaa.loc[dfaa[y] == 0, sb].mean()
                        Adopter_average = dfaa.loc[dfaa[y] > 0, sb].mean()

                        #pd.DataFrame(dfaa.loc[dfaa[y] > 0, sb]).hist(column=sb, color=clr[2], bins=bins)
                        if y == 'Adoption':
                            Adopter_vals = pd.DataFrame(dfaa.loc[dfaa[y] == 1, sb])[sb].tolist()
                            NAdopter_vals = pd.DataFrame(dfaa.loc[dfaa[y] == 0, sb])[sb].tolist()
                        else:
                            Adopter_vals = pd.DataFrame(dfaa.loc[dfaa[y] > dfaa.loc[y].mean(), sb])[sb].tolist()
                            NAdopter_vals = pd.DataFrame(dfaa.loc[dfaa[y] <= dfaa.loc[y].mean(), sb])[sb].tolist()

                        fig, ax = plt.subplots(1, 1, figsize=(15,15), )
                        ax.hist(Adopter_vals, histtype='bar', color=clr[2], label='Adopters')
                        ax.hist(NAdopter_vals, histtype='bar', color=clr[1], label='Non-adopters')
                        ax.set_title('{} histogram: Adopters'.format(sb))
                        ax.legend()
                        ax.grid(True)
                        

                        # plot a bar of the average values for the independent var for
                        # all data, adopters, and nonadopters
                        pd.DataFrame({'Overall': [Average_overall], 'Nonadopters': [Nonadopter_average],
                                      'Adopters': [Adopter_average]}).plot(kind='bar')
                        plt.title('Overall, Nonadopters, and Adopter {} Average'.format(sb))
        return

    def mulit_var_regional(self, df, deps, fontdict2):
        for varc in deps:
            if isinstance(df[varc][0], type('')):
                print('ignoring {}'.format(varc))
            else:
                self.regional_plotter(df, varc, width=.2, fontdict=fontdict2)
        return

    def regional_plotter(self, df, cvar, figsize=(10, 10), dpi=200, fontdict=None, verbose=True, width=.4):
        """
            Plots given cvar average and total for the 7 defined regions of US
        :param df:
        :param cvar:
        :param figsize:
        :param dpi:
        :param fontdict:
        :param verbose:
        :param width:
        :return:
        """
        fontdict2 = {
            'family': 'serif',
            # 'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'medium',
            'color': 'black',
            'size': '15',
        }
        if fontdict is None:
            fontdict = fontdict2
        # create dictionary that will be keyed on region and store a ditionary of the
        # chosen comparison variable
        reg_stat_dict = dict()
        if verbose:
            print(USRegions)
        for regs in list(USRegions.keys()):
            reg_stat_dict[regs] = {}
            print(regs)
            print('the states in region {}\n{}'.format(regs, USRegions[regs]))
            reg_stat_dict[regs]['avg'] = df.loc[df['state'].isin(USRegions[regs]), cvar].mean()
            reg_stat_dict[regs]['sum'] = df.loc[df['state'].isin(USRegions[regs]), cvar].sum()
        # now make lists for the bar graph

        xlbles = list(reg_stat_dict.keys())
        xv = list()
        for i in range(len(reg_stat_dict.keys())):
            if i == 0:
                # xv.append(width)
                xv.append(0)
            else:
                xv.append(np.around(xv[-1] + width + width / 2, 2))
        avgs, tots = list(), list()
        for regs in reg_stat_dict:
            avgs.append(reg_stat_dict[regs]['avg'])
            tots.append(reg_stat_dict[regs]['sum'])
        plt.close('all')
        plt.figure(figsize=figsize, dpi=dpi)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        # ax.bar(x_vals, avgs, tick_label=list(reg_stat_dict.keys()))
        ax.bar(xv, avgs, width=width, align='center')
        ax.set_xticklabels(xlbles, rotation=45, fontdict=fontdict)
        ax.set_xticks(xv)
        ax.set_facecolor('xkcd:light grey')
        # ax.ylabel(cvar + ' average')
        try:
            cvar = label_translation_dict[cvar]
        except KeyError:
            new_label = input('Need to add label for {}.\nWhat will its label be?\n:> '.format(cvar))
            label_translation_dict[cvar] = new_label
            cvar = new_label
            print('Add this to the label_translation_dict:\n"{}": "{}",'.format(cvar, new_label))

        plt.title(cvar + ' average', fontdict=fontdict)

        plt.figure(figsize=figsize, dpi=dpi)
        fig, ax2 = plt.subplots(1, 1, figsize=figsize)

        ax2.bar(xv, tots, width=width, align='center')
        ax2.set_xticklabels(xlbles, rotation=45, fontdict=fontdict)
        ax2.set_xticks(xv)
        ax2.set_facecolor('xkcd:light grey')

        plt.title(cvar + ' total', fontdict=fontdict)
        if verbose:
            plt.show()
        print(xv)
        print(xlbles)

        return

    def generate_average_values(self, df, var, state_l):
        values = {}
        for st in state_l:
            #print('processing {}'.format(st.upper()))
            values[st] = df.loc[df['state'] == st, var].mean()
        return values

    def state_average_visualizer(self, df, visual_var, width=.4, axstd=None, label='', title='',
                                 xlabel='', ylabel='', color='xkcd:yellowish orange', fontdict=None):
        std_df = df
        fontdict3 = {
            'family': 'serif',
            # 'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'medium',
            'color': 'black',
            'size': '15',
        }
        if fontdict is not None:
            fontdict3 = fontdict

        # get the names of the states and the number of y values we need
        state_labels = sorted(list(set(df['state'].values.tolist())))
        y_pos = np.arange(len(state_labels))
        print('calculating {} average for each state'.format(visual_var))
        rdict = self.generate_average_values(df, visual_var, state_labels)
        rdict = sort_dict(rdict)
        state_labels = list(rdict.keys())
        widths = list(rdict.values())

        num_bars = 2
        nrows, ncols = 1, 1

        print('Plotting {} average for each state'.format(visual_var))
        fig, axstd = plt.subplots(nrows, ncols, figsize=(15, 15))

        #           y,     width of bar
        axstd.barh(y_pos, widths, width, label=visual_var, color=color)
        #axstd.barh(x_cntrs + width / num_bars, mu_df.values[0].tolist(), width, label='mu')
        axstd.set_yticklabels(state_labels, rotation=0, fontdict=fontdict3)
        axstd.set_yticks(y_pos)
        axstd.set_ylim(0 - .75, len(state_labels) + 1)
        axstd.set_title(title, fontdict=fontdict3)
        axstd.set_xlabel(xlabel, fontdict=fontdict3)
        axstd.set_ylabel(ylabel, fontdict=fontdict3)
        axstd.legend()
        axstd.grid(True)
            
    def state_average_visualizer_scaled(self, df, visual_var, width=.4, axstd=None, label='', title='',
                                 xlabel='', ylabel='', color='xkcd:yellowish orange', fontdict=None):

        df = pd.DataFrame(df.filter(items=(['state', visual_var]))).dropna(axis=0)

        print(df[visual_var].head())
        print(df[visual_var].values.min(axis=0))
        print(df[visual_var].values.max(axis=0))
        print()
        df[visual_var] = (df[visual_var].values - df[visual_var].values.min(axis=0))/(df[visual_var].values.max(axis=0) - df[visual_var].values.min(axis=0))
        fontdict3 = {
            'family': 'serif',
            # 'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'medium',
            'color': 'black',
            'size': '15',
        }
        if fontdict is not None:
            fontdict3 = fontdict

        # get the names of the states and the number of y values we need
        state_labels = sorted(list(set(df['state'].values.tolist())))
        y_pos = np.arange(len(state_labels))
        print('calculating {} average for each state'.format(visual_var))
        rdict = self.generate_average_values(df, visual_var, state_labels)
        rdict = sort_dict(rdict)
        state_labels = list(rdict.keys())
        widths = list(rdict.values())
        print(widths[0:10])
        num_bars = 2
        nrows, ncols = 1, 1

        print('Plotting {} average for each state'.format(visual_var))
        fig, axstd = plt.subplots(nrows, ncols, figsize=(15, 15))

        #           y,     width of bar
        axstd.barh(y_pos, widths, width, label=visual_var, color=color)
        #axstd.barh(x_cntrs + width / num_bars, mu_df.values[0].tolist(), width, label='mu')
        axstd.set_yticklabels(state_labels, rotation=0, fontdict=fontdict3)
        axstd.set_yticks(y_pos)
        axstd.set_ylim(0 - .75, len(state_labels) + 1)
        axstd.set_title(title, fontdict=fontdict3)
        axstd.set_xlabel(xlabel, fontdict=fontdict3)
        axstd.set_ylabel(ylabel, fontdict=fontdict3)
        axstd.legend()
        axstd.grid(True)

    def state_average_visualizer_scaled_multi(self, df, visual_var1, visual_var2, width=.4,
                                              axstd=None, label='', title='', scale_type='none',
                                 xlabel='', ylabel='', color='xkcd:yellowish orange', fontdict=None):

        df = pd.DataFrame(df.filter(items=(['state', visual_var1, visual_var2]))).fillna(0)

        # scale the variables so we can look at them and compare
        if scale_type == 'none':
            pass
        elif scale_type == 'both':
            df[visual_var1] = (df[visual_var1].values - df[visual_var1].values.min(axis=0)) / (
                        df[visual_var1].values.max(axis=0) - df[visual_var1].values.min(axis=0))
            df[visual_var2] = (df[visual_var2].values - df[visual_var2].values.min(axis=0)) / (
                    df[visual_var2].values.max(axis=0) - df[visual_var2].values.min(axis=0))
        elif scale_type == 'only 1':
            df[visual_var1] = (df[visual_var1].values - df[visual_var1].values.min(axis=0)) / (
                    df[visual_var1].values.max(axis=0) - df[visual_var1].values.min(axis=0))
        elif scale_type == 'only 2':
            df[visual_var2] = (df[visual_var2].values - df[visual_var2].values.min(axis=0)) / (
                    df[visual_var2].values.max(axis=0) - df[visual_var2].values.min(axis=0))
        fontdict3 = {
            'family': 'serif',
            # 'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'medium',
            'color': 'black',
            'size': '15',
        }
        if fontdict is not None:
            fontdict3 = fontdict

        # get the names of the states and the number of y values we need
        state_labels = sorted(list(set(df['state'].values.tolist())))
        y_pos = np.arange(len(state_labels))
        print('calculating {} average for each state'.format(visual_var1))
        rdict = self.generate_average_values(df, visual_var1, state_labels)
        rdict = sort_dict(rdict)
        state_labels = list(rdict.keys())
        widths = list(rdict.values())


        print('calculating {} average for each state'.format(visual_var2))
        rdictA = self.generate_average_values(df, visual_var2, state_labels)
        rdict2 = {}
        for st in state_labels:
            rdict2[st] = rdictA[st]

        #rdict2 = sort_dict(rdict2)
        #state_labels2 = list(rdict2.keys())
        widths2 = list(rdict2.values())

        num_bars = 2
        nrows, ncols = 1, 1

        print('Plotting {} and {} average for each state'.format(visual_var1, visual_var2))
        fig, axstd = plt.subplots(nrows, ncols, figsize=(15, 15))

        #           y,     width of bar
        axstd.barh(y_pos+width/num_bars, widths, width, label=visual_var1, color=color)
        axstd.barh(y_pos - width/num_bars, widths2, width, label=visual_var2, color='xkcd:green')
        axstd.set_yticklabels(state_labels, rotation=0, fontdict=fontdict3)
        axstd.set_yticks(y_pos)
        axstd.set_ylim(0 - .75, len(state_labels) + 1)
        axstd.set_title(title, fontdict=fontdict3)
        axstd.set_xlabel(xlabel, fontdict=fontdict3)
        axstd.set_ylabel(ylabel, fontdict=fontdict3)
        axstd.legend()
        axstd.grid(True)

# clean the data for analysis
class CleanDataSetGenerator:
    """ This class holds several methods that will produce a cleaned up data set.
        It checks for and helps remove:
            * variables with a high amount of missing values
            * string based variables
                > these can be encodeing if desired using onehot or numerical encoding
            * can use a given list of variables to create the data set
    """
    def __init__(self):
        self.var = None
        self.bad_list = []
        self.rmv_strings = []
        self.high_missing = []
        self.removed = []
        self.target_related=[]

    def string_check(self, df, verbose=False):
        # finds and returns a list of those columns that contain
        # string based info
        to_remove = list()
        for c in df.columns.tolist():
            if isinstance(df[c].values.tolist()[0], type('_')):
                if verbose:
                    print('"{}",'.format(c))
                to_remove.append(c)
        self.rmv_strings += to_remove
        return to_remove

    # method definitions
    def var_missing_counter(self, df):
        """
            Calculates the number of missing entries for each variable in
            given data frame and returns a sorted dataframe of them where one cols are:
                * vars: the variables
                * missing%: the percent of missing variables for that variable in the data set
        :param df: data frame to analyze
        :return: data frame sorted on the most to the least missing variables entries
        """
        des = df.describe()         # grab a discription for the data frame to use to find missing entries
        cols = des.columns.tolist()
        miss_dic = {'vars': [], 'missing%': []}
        N = df.shape[0]
        for c in cols:
            miss_dic['vars'].append(c)
            miss_dic['missing%'].append(np.around(((N - des.loc['count', c]) / N) * 100, 2))
        rdf = pd.DataFrame(miss_dic, index=list(miss_dic['vars'])).sort_values(by='missing%', ascending=False, )
        print_df_like_dict(rdf)
        return rdf

    def get_missing(self, dfa, verbose=False, threshold=.10):
        """
            Will count the number and percentage of missing entries for all
            variables in the dataframe (dfa) and will return those that have
            a missing percentage above the given threshold value in a list as well as
            a data frame sorted on the percentage of missing values for the variables
        :param dfa: dataframe (pandas)
        :param verbose: verbosity of method
        :param threshold: missing % threshold, those above this value are added to the bad list
        :return: sorted_data_frame_var_missing (rl), to_high_var_list(bad)
        """
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
                rl['% missing'].append((dfa.shape[0] - desc.loc['count', v]) / dfa.shape[0])
                if rl['% missing'][-1] > threshold:
                    bad.append(v)
                if verbose:
                    print('V: {}, missing: {}'.format(v, rl['missing'][-1]))
        rl = pd.DataFrame(rl).sort_values(by='missing', ascending=False)
        self.high_missing += bad
        return rl, bad

    def encoding_interface(self, varstr, to_encode=None, interactive=False):
        """
            will allow for the user to encode string variables
        :param varstr:
        :param to_encode:
        :param interactive:
        :return:
        """
        got = ''
        gl = list()
        print('the variables to encode are:\n{}'.format(varstr))
        if to_encode is None and interactive:
            print('Are there any of the string variables you would like to encode?')
            print('Enter q to quit or enter the names displayed below one by one')
            while got.lower() != 'q':
                got = input(':> ')
                if got not in varstr:
                    print('Entered: {}'.format(got))
                    print('Optons:\n{}'.format(varstr))
                    print('you must enter one of the above options')
                else:
                    gl.append(got)
        elif to_encode is not None:
            gl = to_encode
        else:
            gl = []

        return gl

    def numerical_encoder(self, df, col):
        nl = list()
        uniq = set(df[col].values.tolist())
        coding = {u: c for u, c in zip(uniq, range(1, len(uniq) + 1))}
        for v in df[col]:
            nl.append(coding[v])
        df[col + '_nc'] = nl
        return col + '_nc'


    def numerically_encode_feats(self, df, feats):
        chn = list()
        for f in feats:
            chn.append(self.numerical_encoder(df, f))
        return chn

    def remove_target_related(self, target='Adoption', target_related=None):
        if target_related is None:
            target_related = solar_metricsB

        if target in target_related:
            del target_related[target_related.index(target)]
        self.target_related = target_related
        return target_related

    def impute_unwanted(self, df, inplace=True):
        return df.drop(self.removed, axis=1, inplace=inplace)

    def impute_missing(self, df, method='drop', inplace=True):
        if method == 'drop' or method not in['drop', 'avg', 'interpolate']:
            return df.dropna(inplace=inplace)

    def remove_with_bad_list(self, df, select_removes=[], rmv_string=None, rmv_Hmissing=None,
                             rmv_target_related=None, method='drop', inplace=True):
        self.removed = select_removes
        if rmv_string is not None:
            self.removed += self.rmv_strings
        if rmv_Hmissing is not None:
            self.removed += self.high_missing
        if rmv_target_related is None:
            self.removed += self.target_related
        return self.impute_unwanted(df, inplace)


    def X_Y_splitter(self, df, target='Adoption'):
        feats = df.columns.tolist()
        del feats[feats.index(target)]

        X = df.filter(items=feats)
        Y = df.filter(items=target)

        return X, Y, feats


class ModelLab:
    scale_options=['minmax', 'std',]
    def __init__(self, source_file, regional_selection=None, target='PV_HuOwn', sel=[],
               state_l=None, model_type='classification', missing_thresh=.1, keep_this='',
               regional_dict=None, regional_col='state', rmv_enc_str='rmv', clf_target=None,
               verbose=True, rmv_msng=True, rmv_trel=True, trel=None, select_drops=[]):
        from _products.ML_Tools import smart_table_opener
        #                     create the class objects needed to perform the analysis task
        self.source_file = source_file
        self.region= regional_selection
        self.target=target
        self.clf_target=clf_target
        self.m_typ = model_type
        self.miss_thrsh = missing_thresh
        self.Xtr_base=None
        self.ytr_base=None
        self.Xts_base = None
        self.yts_base = None
        self.X, self.Y = None, None
        self.N_ts = 0
        self.N_tr = 0
        self.yts_clf = None
        self.ytr_clf = None
        # tool for selecting certain regions of data
        self.selector_tool = Selector_Tool()

        # tool for cleaning and imputing data
        self.data_cleaner = CleanDataSetGenerator()

        # tool for analyzing data i.e. predicting, regression on some target and analyzing
        # the features
        self.analysis_tools = Analysis_tools()

        # tool for data visualization of various information
        # contains a visualizer object and unique methods
        self.visualizer = Relationship_Visualizer()
        data_df = smart_table_opener(source_file)
        self.data_df = self.selector_tool.select_DS_data(data_df, region=regional_selection, sel=sel, state_l=state_l,
                                    regional_col=regional_col, region_dict=regional_dict, )
        self.data_OG = self.data_df.copy()
        self.str_rmv_l = []
        if rmv_enc_str in ['rmv', 'strip', 'remove', ]:
            str_rmv_l = self.data_cleaner.string_check(self.data_df)
            self.str_rmv_l = str_rmv_l

        self.missing_to_rmv = []
        if rmv_msng:
            #                   get a list of the string variables for removal
            #                   get list of variables with a high percentage of missing variables based on a global threshold
            high_missing_df, missing_to_rmv = self.data_cleaner.get_missing(self.data_df, verbose=False,
                                                                            threshold=missing_thresh)
            self.missing_to_rmv = missing_to_rmv

        self.related_to_target = []
        if rmv_trel:
            related_to_target = self.data_cleaner.remove_target_related(target=target,
                                                                 target_related=trel)
            if clf_target is not None and clf_target in related_to_target:
                del related_to_target[related_to_target.index(clf_target)]
            self.related_to_target = related_to_target

        self.select_drops = select_drops
        self.big_drops = self.str_rmv_l + self.missing_to_rmv + self.related_to_target + select_drops
        if verbose:
            print('the string variables are:\n{}'.format(self.str_rmv_l))
            print('---------------------------------------------------\n')
            print('Those variables with more than {:.2f} % missing entries are:'.format(missing_thresh * 100))
            print(self.missing_to_rmv)
            print('---------------------------------------------------\n')
            print('the variables related to the target to be removed')
            print(self.related_to_target)
            print('---------------------------------------------------\n')
            print('The following {} variables will be removed from the data set as per request'.format(len(select_drops)))
            print(select_drops)
            print('---------------------------------------------------\n')

        if keep_this != "":
            if keep_this in self.big_drops:
                del self.big_drops[self.big_drops.index(keep_this)]

        if len(self.big_drops) > 0:
            #                  remove the undesired variables
            if verbose:
                print('there are originally {} columns'.format(self.data_df.shape[1]))
            self.data_cleaner.remove_with_bad_list(self.data_df, select_removes=self.big_drops, method='drop',
                                                   inplace=True)
            if verbose:
                print('there are now {} columns'.format(self.data_df.shape[1]))
                print('---------------------------------------------------\n')

        self.feats = self.data_df.columns.tolist()
        self.data_df.dropna(inplace=True)
        self.description = self.data_df.describe()
        if keep_this != "":
            cols_to = self.description.columns.tolist() + [keep_this]
        else:
            cols_to = self.description.columns.tolist()
        self.data_df = self.data_df.filter(items=cols_to)
        return

    def strip_targets(self, feats, target,):
        if feats is None:
            return feats
        if target in feats:
            del feats[feats.index(target)]

        return feats

    def generate_X_Y(self, df, target, feats=None):
        #print('feats', feats)
        print(target)
        if feats is not None:
            self.feats = feats
        else:
            feats = self.feats
        if target in feats:
            print('it is there')
        else:
            print('it is not there')

        feats = self.strip_targets(feats, target)
        return df.filter(items=feats), df.filter(items=[target])

    def train_test_split(self, X=None, y=None, df=None, feats=None, target=None, clf_target=None,
                         tr_ts_sp=(.75, ), scale_type=None, stratify=True,verbose=False):
        from sklearn.model_selection import train_test_split
        if self.clf_target is None and clf_target is not None:
            self.clf_target = clf_target
        if X is None and y is None:
            if df is None:
                print('for train test split you must enter X and Y or a df')
                print('for a df you must enter the features you want in the model as feats')
                print('ending program....')
                quit(-1)
            if self.clf_target is not None:
                # since it was passed use the classificaiton target as the target
                self.X, self.Y = self.generate_X_Y(df, self.clf_target, feats)
            else:
                self.X, self.Y = self.generate_X_Y(df, self.target, feats)
        else:
            self.X, self.Y = X, y
        feats = self.X.columns.tolist()
        self.feats = feats
        tr = np.around(tr_ts_sp[0], 2)
        ts = np.around(1-tr, 2)
        #print(self.X.head())
        #print(self.Y.head())
        if stratify and self.clf_target is not None:
            self.Xtr_base, self.Xts_base, self.ytr_clf, self.yts_clf = train_test_split(self.X, self.Y, test_size=ts, train_size=tr, stratify=self.Y)
        else:
            self.Xtr_base, self.Xts_base, self.ytr_base, self.yts_base = train_test_split(self.X, self.Y, test_size=ts, train_size=tr,)

        if self.clf_target is not None:
            self.feats = self.strip_targets(self.feats, target)
            print('Stripped the regression from the feats')
        if scale_type is not None:
            from _products.ML_Tools import standardize_data
            if scale_type not in self.scale_options:
                print('your scale option are: {}'.format(self.scale_options))
                print('data left unscaled if you want to scale it just call')
                print('the ModelLabs.scale_data() method')
            if self.clf_target is None:
                self.Xtr_base, self.Xts_base = standardize_data(self.Xtr_base, self.Xts_base, scaler_ty=scale_type)
            else:
                # store the regression columns, since
                # they are still in there due to a classification target being passed
                c_tr = self.Xtr_base.filter(items=[self.target])
                c_ts = self.Xts_base.filter(items=[self.target])
                self.Xtr_base, self.Xts_base = standardize_data(self.Xtr_base, self.Xts_base, scaler_ty=scale_type)
                self.Xtr_base[self.target] = c_tr[self.target].values
                self.Xts_base[self.target] = c_ts[self.target].values

        # Pull out your predictors from your targets for the training set
        if self.clf_target is None:
            self.ytr_base = pd.DataFrame([i for i in self.ytr_base], columns=[target], index=self.ytr_base.index.tolist())
            self.yts_base = pd.DataFrame([i for i in self.yts_base], columns=[target], index=self.yts_base.index.tolist())
        else:
            self.ytr_base = pd.DataFrame([i for i in self.Xtr_base[self.target].values.tolist()],
                                         columns=[self.target], index=self.Xtr_base.filter(items=self.target).index.tolist())
            self.yts_base = pd.DataFrame([i for i in self.Xts_base[self.target].values.tolist()],
                                        columns=[self.target], index=self.Xts_base.filter(items=self.target).index.tolist())
        self.Xtr_base = pd.DataFrame(self.Xtr_base, columns=self.feats)

        # Pull out your predictors from your targets for the testing set
        self.Xts_base = pd.DataFrame(self.Xts_base, columns=self.feats)
        if self.clf_target in self.Xtr_base.columns.tolist():
            print('classification target not removed from X_tr')
            print('ending program')
            quit()
        if self.clf_target in self.Xts_base.columns.tolist():
            print('classification target not removed from X_ts')
            print('ending program')
            quit()
        if self.target in self.Xtr_base.columns.tolist():
            print('classification target not removed from X_tr')
            print('ending program')
            quit()
        if self.target in self.Xts_base.columns.tolist():
            print('classification target not removed from X_ts')
            print('ending program')
            quit()

        #
        self.N_ts = self.Xts_base.shape
        self.N_tr = self.Xtr_base.shape
        if verbose:
            print('There are {} Training samples/cols'.format(self.N_tr))
            print('There are {} Testing samples/cols'.format(self.N_ts))
        if clf_target is not None:
            return (self.Xtr_base, self.ytr_base, self.ytr_clf), (self.Xts_base, self.yts_base, self.yts_clf)
        return (self.Xtr_base, self.ytr_base), (self.Xts_base, self.yts_base)

class DataExplorer:
    def __init__(self, df=None, file=None,  usecols=None, target='Adoption'):
        from _products.ML_Tools import smart_table_opener
        if df is not None:
            self.df = df
        elif file is not None:
            self.df = smart_table_opener(file)
        else:
            print('Error: The DataExplorer needs a data frame or a file to a csv or excel (.xlsx) file name')
            print('Terminating process.....')
            quit(-11)
        # if we got what we needed load the data set
        self.targets = self.df.filter(items=[target])
        self.features = df.columns.tolist()
        # remove target from model feature if it exists
        if target in self.features:
            del self.features[self.features.index(target)]
        else:
            print('Target not found in data set!!!!')
            print('Target sought but not found: {}'.format(target))
            print('Ending process...')
            quit(-11)
        self.predictors = self.features
    def show_me(self, show='missing'):
        if show in ['missing', 'miss', 'na']:
            self.show_me_missing()
    def show_me_missing(self, min=0):
        # will show which variables have missing values
        # ranked from most to least
        # only displays those that have some missing
        missing_dict = {
            'vars': [],
            'miss': [],
        }

        # get the number of missing values for each feature
        missing = self.df.isna().sum()
        #print('sanity check: ')
        #print(missing)
        #print('-----------------------------------')
        #print('-----------------------------------\n')
        # create a dictionary of the variables and the missing counts
        # convert to a dataframe and sort from most to least
        # and sort by missing
        for cc in missing.index.tolist():
            missing_dict['vars'].append(cc)
            missing_dict['miss'].append(missing[cc])
        missing_df = pd.DataFrame(missing_dict,)

        missing_df.sort_values(by=['miss'], ascending=False, inplace=True)

        for cc in missing_df['vars'].tolist():
            if missing_df.loc[missing_df['vars'] == cc, :]['miss'].values[0] > 0:
                msng = missing_df.loc[missing_df['vars'] == cc, 'miss'].values[0]
                print('{}: {}\n'.format(cc, msng))


class ConvergentAnalysisTool:
    def __init__(self, source_file, target, ):
        pass


class Group_Analzyer:
    def __init__(self):
        self.features = []
        self.df = []
        self.dictn = []
        self.title = []
    def set_features(self, features):
        self.features = features
    def set_df(self, df):
        self.df = df
    def set_dictionary(self, dictionary):
        self.dictionary = dictionary
    def set_title(self, title):
        self.title = title

    def grab_group(self, df, selection, colname, regional_dict=USRegions):
        if selection in list(regional_dict.keys()):
            return df.loc[df[colname].isin(USRegions[selection]), :].copy()
        elif selection in ['ColdSpot', 'HotSpot']:
            return df.loc[df[selection] == 1, :]

class Feature_Swarm:
    def __init__(self, features=(), size=1, r2_threshold=.48, group_=1, self_=0, vif_th=10, dfree=22):
        self.features=features
        self.size = size
        self.Current_list = list([""]*22)
        self.r2_threshold = r2_threshold
        self.group_ = group_
        self.self_ = self_
        self.vif_th = vif_th
        self.BestR2 = 0.0
        self.swarm = pd.DataFrame()
        self.added = 0


    def calculate_vif(self, x):
        return pd.Series([VIF(x.values, i)
                          for i in range(x.shape[1])],
                         index=x.columns)

    def generate_swarm(self,):
        self.swarm = pd.DataFrame(np.full((self.size, len(self.features)), 0.0,  ), columns=self.features)
        print("The swarm")
        print(self.swarm)

    def test_random_feature(self, df, agentnum=0, target=''):

        # get a new feat from the list
        if self.added > 0:
            fnd = False
            while(not fnd):
                rnum = np.random.choice(list(range(len(self.features))), 1, replace=False, )[0]
                if self.features[rnum] not in self.Current_list:
                    fnd = True
                else:
                    print("current: {}".format(self.Current_list))
                    print("Tried to add: {}".format(self.features[rnum]))
            fts = self.Current_list + self.features[rnum:rnum + 1]
            print('Features: {}'.format(fts))
            smpdf = df.filter(items=fts)
            #print('The top: {}'.format(smpdf.head(2)))
            vif_res = self.calculate_vif(df.filter(items=smpdf))
            print('\n\n\t\t\t\t-------------------------------------VIF')
            print(dir(vif_res))
            print(vif_res)
            print(vif_res[0])
            print('\t\t\t\t-------------------------------------VIF\n\n')
        else:
            rnum = np.random.choice(list(range(len(self.features))), 1, replace=False,)[0]
            print(rnum)
            fts = self.Current_list + self.features[rnum:rnum+1]
            smpdf = df.filter(items=fts)
        X = smpdf
        X = sm.add_constant(X)
        #print("X: {}".format(X))
        print("checking: {}".format(self.features[rnum:rnum+1]))
        Y = df.filter(items=[target])
        #print('Y: {}'.format(Y))
        est = sm.OLS(Y.values, X)
        # est = sm.GLM(Y.values, X)
        # est = sm.GLS(Y.values, X)

        # fit the estimator and grab the fitted estimator to analyze results
        try:
            est2 = est.fit()
        except:
            print('We got some bad variables!!!!\nCheck the VIF scores')
            #blocking_sound_player(error_sounds[0])
            return 1
        # grab the Macfadden rsquare
        rsqr = est2.rsquared


        #if rsqr > self.BestR2:
        #    self.BestR2 = rsqr

        print(dir(est2))
        print('---------------------------------------------------------------')
        print('---------------------   Regression Analysis   -----------------')
        print('---------------------------------------------------------------')
        print(est2.summary())
        print('---------------------------------------------------------------')
        print('---------------------------------------------------------------')
        #print("The current list is: {}".format(self.Current_list))


        return rsqr, rnum

    def Swarm_Step(self, df, target):
        for agent in range(len(self.swarm)):
            r2, ridx = self.test_random_feature( df, agentnum=agent, target=target)
            self.swarm.loc[agent, self.features[ridx]] = r2
            if r2 > self.BestR2:
                self.Current_list[self.added] = self.features[ridx]
            print('\n\n\t\t\t\tThe thing below')
            print(self.swarm.loc[agent, self.features[ridx]])
            print('max: {}'.format(self.swarm.loc[agent, :].max(axis=0)))
            print('min: {}'.format(self.swarm.loc[agent, :].min(axis=0)))
            print('------------------------------------------------')
            self.swarm.loc[agent, :] = (self.swarm.loc[agent, :] - self.swarm.loc[agent, :].min(axis=0) )/\
                                       (self.swarm.loc[agent, :].max(axis=0) - self.swarm.loc[agent, :].min(axis=0))

            print(self.swarm.loc[agent, self.features[ridx]])

            print('\t\t\t\tThe thing scaled above\n\n\n')
        self.added = min(self.added + 1, len(self.features)-1)

    def start_swarming(self, df, target, runs=22):
        r = 0
        while(r < runs):
            self.Swarm_Step(df, target)
            r += 1
            print("Step: {}".format(r))
            print('The current List')
            print(self.Current_list)
            print('----------------------\n\n')


class Feature_Forward:
    def __init__(self, features=(), size=1, r2_threshold=.48,  vif_th=10, dfree=22):
        self.features = features
        self.feature_scores = {}
        self.size = size
        self.Current_list = list()
        self.r2_threshold = r2_threshold
        self.vif_th = vif_th
        self.BestR2 = 0.0
        self.added = 0
        self.fitted_model = None

    def calculate_vif(self, x):
        return pd.Series([VIF(x.values, i)
                          for i in range(x.shape[1])],
                         index=x.columns)


    def forward_substitutions(self, df, target, verbose=False):
        best_feat = ''
        # go through all remaining features
        # trying to add them to the model if they are not already there
        # and they do not cause the VIF to go beyond the threshold
        # if so add them and store the R2, if it is better than the last add it to model and continue
        for feature in self.features:
            if feature not in self.Current_list:
                r2 = self.test_feature(df, feature, target)
                if r2 > self.BestR2:
                    self.BestR2 = r2
                    best_feat = feature
                    if verbose:
                        print('New best Feature: {}, R2: {:.3f}'.format(feature, r2))
                if self.added <= 1:
                    self.feature_scores[feature] = r2
        if verbose:
            print('-------------The best Scoring feature for feature {} is: {} , for feature {} '.format(len(self.Current_list)+1,
              self.BestR2, best_feat))
        if best_feat in self.features:
            del self.features[self.features.index(best_feat)]
        if verbose:
            print('Added: {}\nfeatures are now: {}'.format(best_feat, self.features))
        if best_feat != '':
            self.Current_list.append(best_feat)
            return best_feat
        else:
            return ""

    def test_model(self, df, target):
        X = df.filter(items=self.Current_list)
        print('X:\n{}'.format(X))
        X = sm.add_constant(X)
        Y = df.filter(items=[target])

        vif_res = self.calculate_vif(X)
        print('-----------------------------------   VIF   ----------------------------------')
        print(vif_res)
        print('------------------------------------------------------------------------------')
        # now check the model with the new feature added
        est = sm.OLS(Y.values, X)
        # est = sm.GLM(Y.values, X)
        # est = sm.GLS(Y.values, X)

        # fit the estimator and grab the fitted estimator to analyze results
        try:
            est2 = est.fit()
        except:
            print('We got some bad variables!!!!\nCheck the VIF scores')
            # blocking_sound_player(error_sounds[0])
            return 1, 1
        # grab the Macfadden rsquare
        rsqr = est2.rsquared
        self.fitted_model = est2
        # if rsqr > self.BestR2:
        #    self.BestR2 = rsqr

        # print(dir(est2))
        print('---------------------------------------------------------------')
        print('----------   Regression Analysis: R2={:.3f}   -----------------'.format(rsqr))
        print('---------------------------------------------------------------')
        print(est2.summary())
        print('---------------------------------------------------------------')
        print('---------------------------------------------------------------\n\n')
        # print("The current list is: {}".format(self.Current_list))


    def VIF_Check(self, df, feature):
        vif_res = self.calculate_vif(df)
        print('\n\n\t\t\t\t-------------------------------------VIF')
        # print(dir(vif_res))
        print(vif_res)
        print(vif_res[feature])
        too_high = False
        for f in self.Current_list + [feature]:
            if f in vif_res and vif_res[f] > self.vif_th:
                too_high = True
                print('Variable {} has a high VIF: {:.3f}'.format(f, vif_res[f]))
        if too_high:
            return -1
        return 0

    def test_feature(self, df,  feature, target=''):
        # grab subset of data to test for the vif if needed
        print("\t\t\t\t--------checking---------: {}".format(feature))
        smpdf = df.filter(items=self.Current_list + [feature])
        # only test VIF if there is more than 1
        if len(self.Current_list) > 0:
            '''
            # print('The top: {}'.format(smpdf.head(2)))
            vif_res = self.calculate_vif(smpdf)
            print('\n\n\t\t\t\t-------------------------------------VIF')
            #print(dir(vif_res))
            print(vif_res)
            print(vif_res[feature])
            too_high = False
            for f in self.Current_list + [feature]:
                if vif_res[f] > self.vif_th:
                    too_high=True
                    print('Variable {} has a high VIF: {:.3f}'.format(f, vif_res[f]))
            if too_high:
                return -1
            print('\t\t\t\t-------------------------------------VIF\n\n')
            '''
            if self.VIF_Check(smpdf, feature) < 0:
                return -1

        X = smpdf
        X = sm.add_constant(X)
        Y = df.filter(items=[target])
        # now check the model with the new feature added
        est = sm.OLS(Y.values, X)
        # est = sm.GLM(Y.values, X)
        # est = sm.GLS(Y.values, X)

        # fit the estimator and grab the fitted estimator to analyze results
        try:
            est2 = est.fit()
        except:
            print('We got some bad variables!!!!\nCheck the VIF scores')
            # blocking_sound_player(error_sounds[0])
            return 1
        # grab the Macfadden rsquare
        rsqr = est2.rsquared
        self.fitted_model = est2
        # if rsqr > self.BestR2:
        #    self.BestR2 = rsqr

        #print(dir(est2))
        print('---------------------------------------------------------------')
        print('---------------------   Regression Analysis   -----------------')
        print('---------------------------------------------------------------')
        print(est2.summary())
        print('---------------------------------------------------------------')
        print('---------------------------------------------------------------\n\n')
        # print("The current list is: {}".format(self.Current_list))

        return rsqr

    def start_swarming(self, df, target, runs=22):
        r = 0
        while (r < runs):
            resp = self.forward_substitutions(df, target)
            if resp == "":
                print('No New Feature was added ending run..............................................')
                break
            r += 1
            self.added += 1
            print("Step: {}".format(r))
            print('The current List')
            print(self.Current_list)
            print('current_Best R2: {}'.format(self.BestR2))
            print('----------------------\n\n')
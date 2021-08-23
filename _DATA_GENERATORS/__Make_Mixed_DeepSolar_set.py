"""
    Makes a mixed data set of deep solar, nrel and svi, also creates several new columns
"""
"""
import pandas as pd
import numpy as np
from _products.ML_Tools import *
from _products.utility_fnc import *
#from _products.visualization_tools import Visualizer
#viz = Visualizer()
# from Data_Sticher import data_merger
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#from D_Space import *
"""
from _products.utility_fnc import blocking_sound_player as bsp, Alert_sounds
from _products._DEEPSOLAR_ import *
pd.options.mode.use_inf_as_na = True

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# -----------------------   TODO: Control Knobs
# ---------------------------------------------------------------------------------------------

#print(list(set(pd.read_excel(r'C:\Users\gjone\DeepSolar_Convergence\_Data\Selectors\__Nominal_values_exclude_list.xlsx')['variables'].values.tolist())))
#quit(-90)

# Booleans
verbose = False                         # how much of the process gets printed to screen
scale_d = False                          # do you want to scale the data?
scale_sub = False                       # do you want to substitued the scaled version of variables
corr_rep = False                        # do you want to save the correlation report
save_var_stat=False                     # do you want to save the variable statistics
gen_scld_only = False                   # generate only the scaled variables?
scale_type = None                       # 0 == minmax, 1 == standard
scl_sub = True                          # substitute the scaled versions or replace them?

# TODO: the below are the regional selection options Note: if all are set to false you get the US set
thirteen_st = False                     # the thirteen state set?
tva_area = False                        # the TVA area set?
seven_states = False                    # the Seven state set?
regions=None                            # the thirteen state set?
target = 'Adoption'                     # TODO: check exactly what this is doing ( i think I can remove since the sets keep all but the select excluded variables)
if thirteen_st:
    regions = State_Sets.thirteen_state
elif seven_states or tva_area:
    print('should be set to seven {} tva {}'.format(seven_states, tva_area))
    regions = State_Sets.tva_seven_state
if scale_type is None:
    scale_smbl = ''
else:
    scale_smbl = scale_type



#dest_pth = r'C:\Users\gjone\DeepSolar_Convergence\_Data\Mixed\13_sets\DS_NREL_13set.csv'
#corre_pth = r'C:\Users\gjone\DeepSolar_Convergence\_Data\Mixed\13_sets\meta_stats\thirteen_{}_4_11_corre.csv'.format(target)
#stats_pth = r'C:\Users\gjone\DeepSolar_Convergence\_Data\Mixed\13_sets\meta_stats\thirteen_{}_4_11_stats.csv'.format(target)

if tva_area:
    print('making tva')
    dest_pth =  r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\TVA_Region\TVA_DS_NREL_DDset_ALL2{}.csv'.format(scale_smbl)
    #corre_pth = None
elif seven_states:
    print('making the 7 state set')
    dest_pth = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\7_sets\SevenSt_DS_NREL_set{}_ALL2A.csv'.format(scale_smbl)
elif thirteen_st:
    print('making the 13 state set')
    dest_pth = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\13_sets\ThirteenSt_DS_NREL_set{}_ALL2A.csv'.format(scale_smbl)
else:
    print('making the usa')
    dest_pth = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\US_set_all{}_OMEGA_1_24_21_Base.csv'.format(scale_smbl)
    dest_pthStats = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\US_set_all{}_OMEGA_Stat_1_24_21_Base.csv'.format(scale_smbl)
    #corre_pth = None
ts = time.time()
model_generator = DS_Mixed_Model_Generator(regions=regions, tva_area=tva_area, destination=dest_pth, iscsv=True,
                                           corre_dest=None, stat_dest=dest_pthStats, scl_type=scale_type, olrmvl=False,
                                           olrVar='PV_HuOwn', scl_sub=scl_sub, target=target, scale_data=False,
                                           impute_target=False, impTrgt="number_of_solar_system_per_household")


te = time.time()
bsp(Alert_sounds[0])
print('took {} seconds'.format(float(te - ts)))




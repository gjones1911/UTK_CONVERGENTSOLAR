"""
    Makes a mixed data set of deep solar, nrel and svi, also creates several new columns
"""

from _products.utility_fnc import blocking_sound_player as bsp, Alert_sounds
from _products._DEEPSOLAR_ import *
pd.options.mode.use_inf_as_na = True
import datetime
import time
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# -----------------------   TODO: Control Knobs
# ---------------------------------------------------------------------------------------------

today = datetime.date.today()
# print(today)
# quit()
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
scl_sub = False                          # substitute the scaled versions or replace them?

# TODO: the below are the regional selection options Note: if all are set to false you get the US set
thirteen_st = False                     # the thirteen state set?
tva_area = False                        # the TVA area set?
seven_states = False                    # the Seven state set?
southern_states = False                    # the Seven state set?
regions=None                            # the thirteen state set?
target = 'Adoption'                     # TODO: check exactly what this is doing ( i think I can remove since the sets keep all but the select excluded variables)
if thirteen_st:
    regions = State_Sets.thirteen_state
elif seven_states or tva_area:
    print('should be set to seven {} tva {}'.format(seven_states, tva_area))
    regions = State_Sets.tva_seven_state
elif southern_states:
    print('should be set to seven {} tva {}'.format(seven_states, tva_area))
    regions = State_Sets.southern_states
if scale_type is None:
    scale_smbl = ''
else:
    scale_smbl = scale_type


if tva_area:
    print('making tva')
    dest_pth =  r'../_Data/_Mixed/TVA_Region/ConvergentTVA_{}.xlsx'.format(today)
    dest_pthStats =  r'../_Data/_Mixed/TVA_Region/DescriptivesConvergentTVA_{}.xlsx'.format(today)
elif seven_states:
    print('making the 7 state set')
    dest_pth = r'../_Data/_Mixed/7_sets/ConvergentSevenSt_{}.xlsx'.format(today)
    dest_pthStats = r'../_Data/_Mixed/7_sets/DescriptivesConvergentSevenSt_{}.xlsx'.format(today)
elif thirteen_st:
    print('making the 13 state set')
    dest_pth = r'../_Data/_Mixed/13_sets/ConvergentThirteenSt_{}.xlsx'.format(today)
    dest_pthStats = r'../_Data/_Mixed/13_sets/DescriptivesConvergentThirteenSt_{}.xlsx'.format(today)
elif southern_states:
    print('making the Southern set')
    dest_pth = r'../_Data/_Mixed/South_set/ConvergentSouthern_{}.xlsx'.format(today)
    dest_pthStats = r'../_Data/_Mixed/South_set/DescriptivesConvergentSouthern_{}.xlsx'.format(today)
else:
    print('making the usa')
    dest_pth = r'../_Data/_Mixed/US/ConvergentUS_{}.csv'.format(today)
    dest_pthStats = r'../_Data/_Mixed/US/DescriptivesConvergentUS_{}.csv'.format(today)
ts = time.time()
model_generator = DS_Mixed_Model_Generator(regions=regions, tva_area=tva_area, destination=dest_pth, iscsv=True,
                                           corre_dest=None, stat_dest=dest_pthStats, scl_type=scale_type, olrmvl=False,
                                           olrVar='PV_HuOwn', scl_sub=scl_sub, target=target, scale_data=False,
                                           impute_target=False, impTrgt="number_of_solar_system_per_household")


te = time.time()
bsp(Alert_sounds[0])
print('took {} seconds'.format(float(te - ts)))




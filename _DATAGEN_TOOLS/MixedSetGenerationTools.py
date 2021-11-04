import pandas as pd
import numpy as np
import scipy.stats as stats

from _products.utility_fnc import blocking_sound_player, error_sounds, Alert_sounds
import matplotlib.pyplot as plt
from  _products.__Data_Manipulation import *

state_abrev = [
    "al",
    "ar",
    "az",
    "ca",
    "co",
    "ct",
    "dc",
    "de",
    "fl",
    "ga",
    "ia",
    "id",
    "il",
    "in",
    "ks",
    "ky",
    "la",
    "ma",
    "md",
    "me",
    "mi",
    "mn",
    "mo",
    "ms",
    "mt",
    "nc",
    "nd",
    "ne",
    "nh",
    "nj",
    "nm",
    "nv",
    "ny",
    "oh",
    "ok",
    "or",
    "pa",
    "ri",
    "sc",
    "sd",
    "tn",
    "tx",
    "ut",
    "va",
    "vt",
    "wa",
    "wi",
    "wv",
    "wy",
]

Nt3state_abrev = state_abrev.copy()
US_4Major = {
    'NT3': Nt3state_abrev,
    'T3':  [ 'ca', 'nv', 'az',],
    'West':  [ 'ca', 'nv', 'wa', 'or', 'id', 'mt', 'wy', 'ut', 'az', 'nm', 'co',],
    'WestNT3':  ['wa', 'or', 'id', 'mt', 'wy', 'ut', 'nm', 'co',],
    'Mid West': ['nd', 'sd', 'ne', 'ks', 'mo', 'mn', 'ia', 'il', 'in','oh', 'mi','wi',],
    'South': ['tx', 'ok','ar', 'la', 'ms', 'al', 'ga', 'fl', 'sc', 'nc', 'tn','wv', 'va','ky',],
    'NorthEast': ['ma','vt', 'me', 'nh', 'ri', 'ct', 'pa', 'dc', 'de', 'nj', 'ny', 'md', ]
}

for st in ['ca', 'nv', 'az']:
    del Nt3state_abrev[Nt3state_abrev.index(st)]

major4_reg = list(US_4Major.keys())

class MixedSetGenerator:
    """
        used to merge deep solar and nrel replica data sets
    """
    local_recode = {'Rural': 1, 'Town': 2, 'City': 4, 'Suburban': 3, 'Urban': 4}
    local_recodeA = {'Rural': 'Rural', 'Town': 'Town', 'City': 'City', 'Suburban': 'Suburban', 'Urban': 'City'}
    sought = ['Rural', 'Town', 'City', 'Suburban', 'Urban']
    US_4Major = {
        'NT3': Nt3state_abrev,
        'T3': ['ca', 'nv', 'az', ],
        'West': ['ca', 'nv', 'wa', 'or', 'id', 'mt', 'wy', 'ut', 'az', 'nm', 'co', ],
        'WestNT3': ['wa', 'or', 'id', 'mt', 'wy', 'ut', 'nm', 'co', ],
        'Mid West': ['nd', 'sd', 'ne', 'ks', 'mo', 'mn', 'ia', 'il', 'in', 'oh', 'mi', 'wi', ],
        'South': ['tx', 'ok', 'ar', 'la', 'ms', 'al', 'ga', 'fl', 'sc', 'nc', 'tn', 'wv', 'va', 'ky', ],
        'NorthEast': ['ma', 'vt', 'me', 'nh', 'ri', 'ct', 'pa', 'dc', 'de', 'nj', 'ny', 'md', ]
    }
    def __init__(self, newfilename, deepsolar_path, nrel_path, states=None, how="left",
                 usecols=(None, None), low_memory=False, add_additional_variables=True):
        # Load the deep solar and nrel sets for merging
        self.ds, self.nrel = self.get_DeepSolar_NREL(deepsolar_path, nrel_path, usecols=usecols, low_memory=low_memory)

        # check for the merge column in each set and if absence add them
        # use deep solar as base data set so only those state found within
        # the set used for merging will be included
        if states is not None:
            self.ds = self.ds.loc[self.ds['state'].isin(states)]

        self.merged = self.data_merger([self.ds, self.nrel], joins=('fips', 'geoid'), how=how)

        # can only get merged dat if desired
        if not add_additional_variables:
            return
        self.DataDict = dict()                       # used to store the descriptio of additional variables
        self.generate_green_travelers()              # combine certain % types of travel
        self.mixed_travel_times()                    # mixe smaller ranges into broader ones
        self.recode_locale_data()                    # make numeric versions of the local labels
        self.add_binary_policies()                   # make certain policies binary instead of year based
        self.add_renewable_gen()                     # adds the renewable generation for the selected states
        self.combine_edu()                           # combines certain types of education ranges/types
        self.recode_hh_size_counts_into_pct()        # converts household size counts to percentages
        self.add_home_age_range_combos()             # creates combined home age ranges
        self.home_owner_mergers()                    # converts home owners to %
        self.generate_mixed_policy_homeowner()       # generates mixed policy variables with homeownership
        self.generate_energy_income_interactions()   # generates energy cost, bill income interactions
        self.generate_population_density_howmowner_Interaction()           # adds interaction between population density and homeownership
        self.gender__race_pct()                      # converts gender counts to %
        self.generate_solar_metrics()                # generates some new solar metrics, see DataDict or method for list

        self.add_hotspots()                      # add binary indicators for CT's that are hot spots for per hh, huown, and avg size
        self.broaden_travel_times()              # creates mixed ranges of travel times
        self.age_range_mixing()                  # creates mixed age ranges
        self.add_summed_seedsii_vars()           # add combinations of the various income broken down SEEDS II variables
        self.make_state_dummies()                # add state identifier variables for state fixed effects
        self.add_regionalindicators()            # add binary indicators for west, mid west, south, and north east
        self.random_interactions()               # adds some main variables used in paper; Savings_potential, Rural_diversity, Suburban_diversity, Cost_x_Consumption, Income_x_Education
        self.split_solarresources()              # adds indicator variables for above and below average solar radiation, as well as the solar radiation threshold found by DeepSolar to related to adoption spikes

        # self.merged.to_excel(destination, index=False)
        self.merged.to_csv(destination, index=False)

    def add_additional_variables(self, variable_options = None):
        pass

    def smart_table_opener(self, filename, usecols=None, low_memory=False):
        """
        will open both csvs and .xlsx files into a pandas data frame, breaks if the file is either
        missing these identifiers are is not one of them
        """
        if filename.endswith(".csv"):
            return pd.read_csv(filename, usecols=usecols, low_memory=low_memory)
        elif filename.endswith(".xlsx"):
            return pd.read_excel(filename, usecols=usecols)
        else:
            print("unknown file format for filename: {}\nexpecting .csv, or .xlsx")
            quit(-20)

    def get_DeepSolar_NREL(self, deepsolar_path, nrel_path, usecols=(None, None), low_memory=False):
        """

        :param deepsolar_path: path to the DeepSolar data file.
                               Assumes Stanford Deepsolar file is an .xlsx.
                               If it is not download the file and save it as one first.
        :param nrel_path: Path to NREL SEEDS II REPLICA file
        :param usecols:       list or tuple of lists containing the usecolumns desired. index 0 is for Deepsolar,
                              index 1 is for SEEDS II
        :param low_memory:    used for force a read_csv call to not bother data types(see pandas documentation read_csv)
        :return:
        """
        return self.smart_table_opener(deepsolar_path, usecols=usecols[0], low_memory=low_memory),\
               self.smart_table_opener(nrel_path, usecols=usecols[1], low_memory=low_memory)

    def data_merger(self, data_sets, joins=('fips', 'FIPS', 'geoid'), target=None, how='left'):
        """This method can be used to merge a set of data frames using a shared
           data column. the first argument is a list of the dataframes to merge
           and the second argument is a list of the column labels used to perform the merge
           TODO: some work needs to be done for error checking
           TODO: add more flexibility in how the merge is perfomed
           TODO: make sure the copy rows are removed
        :param data_sets: a list of data frames of the data sets that are to be joined
        :param joins: a list of the column labels used to merge, the labels should be in the
                      same order as the data frames for the method to work. Right now this works
                      best if the label used is the same for all. This makes sure that duplicate
                       columns are not created.
        :param verbose: at this point does nothing but can be used to inform user of what
                        has occured
        :return: a reference to the new merged dataframe
        """
        cnt = 0
        if len(data_sets) == 1:
            return data_sets[0]
        for df in range(1, len(data_sets)):
            data_sets[0] = data_sets[0].merge(data_sets[df], left_on=joins[0], right_on=joins[df], how=how)

            # remove the duplicate join columns
            if (joins[0] + '_x') in data_sets[0].columns.values.tolist() or (
                    (joins[0] + '_y') in data_sets[0].columns.values.tolist()):
                data_sets[0].drop(columns=[(joins[0] + '_x'), (joins[1] + '_y')], inplace=True)
            if target is not None and ((target + '_x') in data_sets[0].columns.values.tolist() or (
                    (target + '_y') in data_sets[0].columns.values.tolist())):
                data_sets[0][target] = data_sets[0].loc[:, target + '_x']
                data_sets[0].drop(columns=[(target + '_x'), (target + '_y')], inplace=True)

        return data_sets[0]

    def generate_green_travelers(self, merged=None, green_travelers=None):
        if merged is None:
            merged = self.merged
        if green_travelers is None:
            green_travelers = ['transportation_home_rate', 'transportation_bicycle_rate', 'transportation_walk_rate']
        self.DataDict['Green_Travelers'] = "combination of rates of persons using a bicycle, walkiing, or working from home"
        create_combo_var_sum(merged, green_travelers, newvar='Green_Travelers')
        # self.excludes += green_travelers + ['Green_Travelers']

    def mixed_travel_times(self, ):
        self.DataDict['low_commute_times'] = " % persons commuting for 19 minuts or less (DEEPSOLAR)"
        low_travel = ["travel_time_10_19_rate","travel_time_less_than_10_rate",]
        create_combo_var_sum(self.merged, low_travel, newvar='low_commute_times')

        trav_recodes = ['travel_time_40_59_rate', 'travel_time_60_89_rate']
        self.DataDict['travel_time_40_89_rate'] = ""
        create_combo_var_sum(self.merged, trav_recodes, newvar='travel_time_40_89_rate')
        self.DataDict['high_commute_times'] = ""
        create_combo_var_sum(self.merged, trav_recodes, newvar='high_commute_times')

        travM_recodes = ['travel_time_20_29_rate', 'travel_time_30_39_rate', ]
        self.DataDict['travel_time_20_39_rate'] = ""
        create_combo_var_sum(self.merged, travM_recodes, newvar='travel_time_20_39_rate')

    def recode_locale_data(self):
        local = list(self.merged['locale'])
        self.merged['locale_dummy'] = recode_var_sub(self.sought, local, self.local_recode)
        self.merged['locale_recode'] = recode_var_sub(self.sought, local, self.local_recodeA)

        self.DataDict['locale_dummy'] = "numeric encodiing of locale: 1=rural, 2=town, 3=suburban, 4=urban"
        empty_1, empty_2, empty_3, empty_4 = np.zeros(self.merged.shape[0]), np.zeros(self.merged.shape[0]), np.zeros(
            self.merged.shape[0]),np.zeros(self.merged.shape[0])
        empty_1[self.merged['locale_dummy'].values == 1] = 1   # RURAL
        empty_2[self.merged['locale_dummy'].values == 2] = 1   # TOWN
        empty_3[self.merged['locale_dummy'].values == 3] = 1   # SUBURBAN
        empty_4[self.merged['locale_dummy'].values == 4] = 1   # URBAN

        self.DataDict['locale_(rural)'] = "indicates a rural locale"
        self.merged['locale_(rural)'] = empty_1
        self.DataDict['locale_(suburban)'] = "indicates suburban a rural locale"
        self.merged['locale_(suburban)'] = empty_3
        self.DataDict['locale_(town)'] = "indicates town a rural locale"
        self.merged['locale_(town)'] = empty_2
        self.DataDict['locale_(urban)'] = "indicates urban a rural locale"
        self.merged['locale_(urban)'] = empty_4

        # add a designation for town and urban called urban
        self.DataDict['URBAN'] = "aggregation of urban and town locales into the 'URBAN' locale"
        self.merged['URBAN'] = np.zeros(len(self.merged))
        self.merged['URBAN'] = self.merged['locale_recode(urban)'].values + self.merged['locale_recode(town)'].values

        self.DataDict['Rural'] = "indicates urban a rural locale"
        self.merged['Rural'] = empty_1
        self.DataDict['Suburban'] = "indicates urban a suburban locale"
        self.merged['Suburban'] = empty_3

    def add_binary_policies(self):
        """
            Original net_metering, and property_tax, variables are in years of existence and are used
            to generate binary exists(1) and does not exist (0).
            Also uses the low income tax credit binary variable in the same vain.
            Using these variables a "combination" policy variable is created that represents:

            # Create a policy combo varialbe to represent if:
            # * 1 if there is only net metering
            # * 2 if there is only property tax
            # * 3 if there is only the low income tax credit
            # * 4 there is both property tax and net metering
            # * 5 there is both net metering and low income tax credit
            # * 6 there is both property tax and low income tax credit
            # * 7 if there is all three programs present
        :return:
        """
        nm, pt, litx = 'net_metering_bin', 'property_tax_bin', 'lowincome_tax_credit_bin'
        self.DataDict['net_metering_bin'] = "indication of presence of program (DeepSolar)"
        self.DataDict['property_tax_bin'] = "indication of presence of program  (DeepSolar) "
        self.DataDict['lowincome_tax_credit_bin'] = "indication of presence of program (SEEDSII)"
        thresh_binary_recode(self.merged, 'net_metering', )
        thresh_binary_recode(self.merged, 'property_tax', )
        self.merged['lowincome_tax_credit_bin'] = np.zeros(len(self.merged))
        self.merged.loc[self.merged['lihtc_qualified'] == 'FALSE','lowincome_tax_credit_bin' ] = 0
        self.merged.loc[self.merged['lihtc_qualified'] == 'TRUE','lowincome_tax_credit_bin' ] = 1

        # Create a policy combo varialbe to represent if:
        # * 1 if there is only net metering
        # * 2 if there is only property tax
        # * 3 if there is only the low income tax credit
        # * 4 there is both property tax and net metering
        # * 5 there is both net metering and low income tax credit
        # * 6 there is both property tax and low income tax credit
        # * 7 if there is all three programs present
        self.merged['Policy_Combo'] = np.zeros(len(self.merged))
        self.DataDict['Policy_Combo'] = "indication of :\n* 1 if there is only net metering\n* 2 if there is only property tax\n* 3 if there is only the low income tax credit\n* 4 there is both property tax and net metering\n* 5 there is both net metering and low income tax credit\n* 6 there is both property tax and low income tax credit\n* 7 if there is all three programs present"
        self.merged.loc[self.merged[nm] == 1, 'Policy_Combo'] = 1
        self.merged.loc[self.merged[pt] == 1, 'Policy_Combo'] = 2
        self.merged.loc[self.merged[litx] == 1, 'Policy_Combo'] = 3

        # if the values are 1 and 1 then both of the programs are pregnant
        self.merged.loc[[a + b == 2 for a, b in zip(self.merged[nm].tolist(), self.merged[pt].tolist())], 'Policy_Combo'] = 4
        self.merged.loc[[a + b == 2 for a, b in zip(self.merged[nm].tolist(), self.merged[litx].tolist())], 'Policy_Combo'] = 5
        self.merged.loc[[a + b == 2 for a, b in zip(self.merged[pt].tolist(), self.merged[litx].tolist())], 'Policy_Combo'] = 6
        self.merged.loc[[a + b + c == 3 for a, b, c in zip(self.merged[nm].tolist(), self.merged[pt].tolist(), self.merged[litx].tolist())], 'Policy_Combo'] = 7

    def add_renewable_gen(self, year='2015'):
        # source: https://www.eia.gov/electricity/data/state/
        #add_renewable_gen(self.merged, 'state', self.ren)
        #add_renewable_gen_df_df(self.merged, 'Ren', cold, cols)
        sourcefile = r'../_Data/Renewable_State_Info/renew_prod_2009.csv'
        add_renewable_gen_df_df(self.merged,
                                sourcefile=sourcefile,
                                cols=['renew_prod','hydro_prod','solar_prod',],
                                open_method=self.smart_table_opener,
                                fillins='STUSPS',
                                cold='state')
        self.DataDict['renew_prod'] = "Renewable (overall) energy producation based on data from:ttps://www.eia.gov/electricity/data/state/ "
        self.DataDict['hydro_prod'] = "hydro renewable energy producation based on data from:ttps://www.eia.gov/electricity/data/state/ "
        self.DataDict['solar_prod'] = "solar renewable  energy producation based on data from:ttps://www.eia.gov/electricity/data/state/ "

    def combine_edu(self):
        """
                Used to create combo education variables makiing combinations of percentages.

        :return:
        """

        high_above = ['education_high_school_graduate_rate', 'education_master_rate', 'education_doctoral_rate',
                      'education_bachelor_rate']
        self.DataDict['education_high_school_or_above_rate'] = " percentage of persons high graduate education level or above (DeepSolar)"
        self.merged['education_high_school_or_above_rate'] = create_combo_var_sum(self.merged, high_above)

        high_below = ['education_less_than_high_school_rate', 'education_high_school_graduate_rate']
        self.DataDict['education_high_school_or_below_rate'] = "percentage persons with high school education or less (DeepSolar)"
        self.merged['education_high_school_or_below_rate'] = create_combo_var_sum(self.merged, high_below)

        master_above = ['education_master_rate', 'education_doctoral_rate']
        self.DataDict['education_master_or_above_rate'] = "percentage persons with masters degrees or above (DeepSolar)"
        self.merged['education_master_or_above_rate'] = create_combo_var_sum(self.merged, master_above)

        bachelor_above = ['education_master_rate', 'education_doctoral_rate'] + ['education_bachelor_rate']
        self.DataDict['education_bachelor_or_above_rate'] = "percentage persons with bachelor degress or above (DeepSolar)"
        self.merged['education_bachelor_or_above_rate'] = create_combo_var_sum(self.merged, bachelor_above)

        self.DataDict['educated_population_rate'] = "population 25 or above with some college education (SEEDSII) / population (DeepSolar)"
        self.merged['educated_population_rate'] = smart_df_divide(self.merged['pop25_some_college_plus'].values,
                                                                 self.merged['population'].values)

    def recode_hh_size_counts_into_pct(self):
        """
            Creates a household total from the different household size counts from SEEDS
        :return:
        """
        hh_sizes = ['hh_size_1', 'hh_size_2', 'hh_size_3', 'hh_size_4']
        # merged['hh_total'] = create_combo_var_sum(merged, hh_sizes, newvar=None)
        create_combo_var_sum(self.merged, hh_sizes, newvar='hh_total')

        self.DataDict['hh_total'] = " sum of 'hh_size_1', 'hh_size_2', 'hh_size_3', 'hh_size_4' counts from SEEDSII"
        self.DataDict['%hh_size_1'] = " hh_size_1(SEEDSII) / hh_total(Derived)  "
        percentage_generator(self.merged, hh_sizes[0], 'hh_total', newvar='%hh_size_1')

        self.DataDict['%hh_size_2'] = " hh_size_2(SEEDSII) / hh_total(Derived)  "
        percentage_generator(self.merged, hh_sizes[1], 'hh_total', newvar='%hh_size_2')

        self.DataDict['%hh_size_3'] = " hh_size_3(SEEDSII) / hh_total(Derived)  "
        percentage_generator(self.merged, hh_sizes[2], 'hh_total', newvar='%hh_size_3')

        self.DataDict['%hh_size_4'] = " hh_size_4(SEEDSII) / hh_total(Derived)  "
        percentage_generator(self.merged, hh_sizes[3], 'hh_total', newvar='%hh_size_4')

    def add_home_age_range_combos(self):
        """
            Adds aggregated age range combos from more fine grained age ranges. Age tranges come from
            SEEDS II, housing unit sourced from Deep Solar
        :return:
        """
        # make range from 1959 to earlier variable
        hage1959toearlier = ['hu_vintage_1940to1959', 'hu_vintage_1940to1959']
        self.DataDict['hu_1959toearlier'] = "'hu_vintage_1940to1959' (SEEDSII) * 'hu_vintage_1940to1959'(SEEDSII)"
        self.merged['hu_1959toearlier'] = create_combo_var_sum(self.merged, hage1959toearlier)
        # make 60 to 79 pct variable
        self.DataDict['hu_1959toearlier_pct'] = "'hu_1959toearlier' (derived) * 'housing_unit_count'(DeepSolar)"
        self.merged['hu_1959toearlier_pct'] = (
                    self.merged['hu_1959toearlier'] / self.merged['housing_unit_count']).values.tolist()

        # make 60 to 79 pct variable
        self.DataDict['hu_1960to1979_pct'] = "'hu_vintage_1960to1970' (SEEDSII) * 'housing_unit_count'(DeepSolar)"
        self.merged['hu_1960to1979_pct'] = (
                    self.merged['hu_vintage_1960to1970'] / self.merged['housing_unit_count']).values.tolist()

        # make 80 to 99 pct variable
        self.DataDict['hu_1980to1999_pct'] = "'hu_vintage_1980to1999' (SEEDSII) * 'housing_unit_count'(DeepSolar)"
        self.merged['hu_1980to1999_pct'] = (
                    self.merged['hu_vintage_1980to1999'] / self.merged['housing_unit_count']).values.tolist()

        # make list of variabels to sum to get range variable from 2000 to beyond
        hage2000tobeyond = ['hu_vintage_2000to2009', 'hu_vintage_2010toafter']
        self.DataDict['hu_2000toafter'] = "'hu_vintage_2000to2009 (SEEDSII) * 'hu_vintage_2010toafter'(SEEDSII)"
        self.merged['hu_2000toafter'] = create_combo_var_sum(self.merged, hage2000tobeyond)

        # make percentage variable out of new variable
        self.DataDict['hu_2000toafter_pct'] = "'hu_2000toafter' (SEEDSII) * 'housing_unit_count'(DeepSolar)"
        self.merged['hu_2000toafter_pct'] = (
                    self.merged['hu_2000toafter'] / self.merged['housing_unit_count']).values.tolist()

        self.DataDict['Mid_Agedwellings'] = "'hu_vintage_1960to1970' (SEEDSII) * 'hu_vintage_1980to1999'(SEEDSII)"
        self.merged['Mid_Agedwellings'] = self.merged["hu_vintage_1960to1970"].values + self.merged["hu_vintage_1980to1999"]

    def home_owner_mergers(self):
        """
            Uses home owner count and housing unit count from SEEDS II to calculate %home owners
        :return: None
        """
        self.DataDict['hu_own_pct'] = "number of home owners (SEEDSII) * number of housing units % (DeepSolar)"
        self.merged['hu_own_pct'] = (self.merged['hu_own'] / self.merged['housing_unit_count']).values.tolist()


    def generate_mixed_policy_homeowner(self):
        """
            Multiplies policy variables with different variables createing interacted policy variables.

        :return:
        """

        net_own = ['net_metering_bin', 'hu_own_pct']
        new_net = 'net_metering_hu_own'

        self.DataDict[new_net] = "new metering years (DeepSolar) * homeowner % (SEEDSII)"
        generate_mixed(self.merged, net_own, new_net)

        ptax_own = ['property_tax_bin', 'hu_own_pct']
        new_ptax = 'property_tax_hu_own'
        self.DataDict[new_ptax] = "property tax years (DeepSolar) * homeowner % (SEEDSII)"
        generate_mixed(self.merged, ptax_own, new_ptax)

        incent_res_own = ['incentive_count_residential', 'hu_own_pct']
        new_incent_own = 'incent_cnt_res_own'
        self.DataDict[new_incent_own] = "incentive_count_residential(DeepSolar) * homeowner % (SEEDSII)"
        generate_mixed(self.merged, incent_res_own, new_incent_own)


    def generate_energy_income_interactions(self,):

        med_income_ebill = ['avg_monthly_bill_dlrs', 'hh_med_income']
        medincebill = 'med_inc_ebill_dlrs'
        self.DataDict[medincebill] = "average monthly energy bill (SEEDSII) * median household income(SEEDSII)"
        generate_mixed(self.merged, med_income_ebill, medincebill)

        avg_income_ebill = ['avg_monthly_bill_dlrs', 'average_household_income']
        avgincebill = 'avg_inc_ebill_dlrs'
        self.DataDict[avgincebill] = "average monthly energy bill (SEEDSII) * averages household income(DeepSolar)"
        generate_mixed(self.merged, avg_income_ebill, avgincebill)

        med_income_ebill = ['dlrs_kwh', 'median_household_income']
        medincecost = 'dlrs_kwh x median_household_income'
        self.DataDict[medincecost] = "dollars / kWh (DeepSolar) * median household income(DeepSolar)"
        generate_mixed(self.merged, med_income_ebill, medincecost)

    def generate_population_density_howmowner_Interaction(self):

        own_popden = ['population_density', 'hu_own_pct']
        ownpopden = 'own_popden'
        self.DataDict[ownpopden] = "population_density (DeepSolar) * hu_own_pct (SEEDSII)"
        generate_mixed(self.merged, own_popden, ownpopden)


    def gender__race_pct(self):
        """
            Generates percentage varialbes for gender and race. Uses the
        :return:
        """
        female_count = 'pop_female'
        male_count = 'pop_male'
        total = 'pop_total'
        # merged[total] = create_combo_var_sum(merged, [female_count, male_count], newvar=total)
        # merged['%female'] = percentage_generator(merged, female_count, total)
        # merged['%male'] = percentage_generator(merged, male_count, total)
        create_combo_var_sum(self.merged, [female_count, male_count], newvar=total)

        self.DataDict['female_pct'] = "population of female / (population females + population males) (SEEDSII) "
        percentage_generator(self.merged, female_count, total, newvar='female_pct')

        self.DataDict['male_pct'] = "population of male / (population females + population males) (SEEDSII) "
        percentage_generator(self.merged, male_count, total, newvar='male_pct')


        self.DataDict['Gender_Ratio'] = "'pop_female' / 'pop_male' (SEEDSII) "
        self.merged['Gender_Ratio'] = smart_df_divide(self.merged['pop_female'].values,
                                                      self.merged['pop_male'].values)

        self.DataDict['white_pct'] = "'pop_caucasian'/ 'population' (SEEDSII) "
        self.merged['white_pct'] = smart_df_divide(self.merged['pop_caucasian'].values,
                                                      self.merged['population'].values)
        self.DataDict['black_pct'] = "'pop_african_american'/ 'population' (SEEDSII) "
        self.merged['black_pct'] = smart_df_divide(self.merged['pop_african_american'].values,
                                                   self.merged['population'].values)
        self.DataDict['asian_pct'] = "'pop_asian'/ 'population' (SEEDSII) "
        self.merged['asian_pct'] = smart_df_divide(self.merged['pop_asian'].values,
                                                   self.merged['population'].values)

        self.DataDict['hispanic_pct'] = "'pop_hispanic'/ 'population' (SEEDSII) "
        self.merged['hispanic_pct'] = smart_df_divide(self.merged['pop_hispanic'].values,
                                                   self.merged['population'].values)


    def generate_political_ratio(self):
        """
            Generates the ratio of %Democrate_voting_2012/%Republican_voting_2012
        :return:
        """
        self.DataDict['political_ratio'] = "%Democrat_voting_2012/%Republican_voting_2012 (DeepSolar)"
        self.merged['political_ratio'] = (self.merged['voting_2012_dem_percentage']
                                          * self.merged['population']) / (self.merged['voting_2012_gop_percentage']
                                                                          * self.merged['population'])


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


        self.DataDict['SRpcap'] = "total residential installations / population (DeepSolar)"
        self.DataDict['SNRpcap'] = "total non-residential installations / population (DeepSolar)"
        self.DataDict['ST_pcap'] = "total installations / population (DeepSolar)"
        """ make solar population total  per captia"""
        self.merged['SRpcap'] = self.merged['solar_system_count_residential'] / (self.merged[popu_t])
        self.merged['SNRpcap'] = self.merged['solar_system_count_nonresidential'] / (self.merged[popu_t])
        self.merged['ST_pcap'] = self.merged['solar_system_count'] / (self.merged[popu_t])

        """ make solar area per area """
        self.DataDict['SRaPa'] = "residential area by total panel area (DeepSolar)"
        self.DataDict['SNRaPa'] = "nonresidential area by total panel area (DeepSolar)"
        self.merged['SRaPa'] = self.merged[pv_ara_res] / self.merged[tot_araa]
        self.merged['SNRaPa'] = self.merged[pv_ara_nres] / self.merged[tot_araa]

        """ make solar area per capita """
        self.DataDict['SRaPcap'] = "residential installation area /total population"
        self.DataDict['SNRaPcap'] = "non-residential installation area /total population"
        self.merged['SRaPcap'] = self.merged[pv_ara_res] / (self.merged[popu_t])
        self.merged['SNRaPcap'] = self.merged[pv_ara_nres]/ (self.merged[popu_t])

        """ make solar per home owner"""
        self.DataDict[pv_own] = "residential installations per home owner (DeepSolar/SEEDSII)"
        self.merged[pv_own] = self.merged[res_tot] / self.merged[hu_own]
        self.merged.loc[self.merged['PV_HuOwn'].isna(), 'PV_HuOwn'] = 0
        self.merged['PV_per_100_HuOwn'] = self.merged[pv_own] * 100
        self.merged.loc[self.merged['PV_per_100_HuOwn'].isna(), 'PV_per_100_HuOwn'] = 0



        #self.merged.loc[self.merged['PV_per_100_HuOwn'] >= Q25,'Adoption_' ] = 1

        #self.merged['PV_per_100_HuOwnB'] = self.merged[res_tot] / (self.merged[hu_own]/100)
        """ make average solar panel installation in m^2"""
        rll = list([])
        for ara, cnt in zip(self.merged[pv_res_area].values.tolist(), self.merged[res_tot].values.tolist()):
            if cnt == 0:
                rll.append(0)
            else:
                rll.append(ara/cnt)
        self.DataDict[avg_PVres] = "residential installation panel area / total residential installations"
        self.merged[avg_PVres] = rll
        # fill the values that were too small with zeros
        # self.merged[avg_PVres] = self.merged[avg_PVres].fillna(0)



    def add_hotspots(self, ):
        ds_df = self.merged
        # add the hot spots
        self.DataDict['Hot_Spots_hh'] = "indicates that the census tract is in the 95th percentile for per household adoption"
        add_HOTSPOTS(df=ds_df, var="number_of_solar_system_per_household", new_var_name='Hot_Spots_hh',
                     percentile=.949, verbose=True)

        self.DataDict['Hot_Spots_hown'] = "indicates that the census tract is in the 95th percentile for per homeowner adoption"
        add_HOTSPOTS(df=ds_df, var='PV_HuOwn', new_var_name='Hot_Spots_hown',
                     percentile=.949, verbose=True)

        self.DataDict['Hot_Spots_AvgAr'] = "indicates that the census tract is in the 95th percentile average panel area"
        add_HOTSPOTS(df=ds_df, var='AvgSres', new_var_name='Hot_Spots_AvgAr',
                     percentile=.949, verbose=True)
        # this below line is completely uneeded and should be removed
        self.merged = ds_df

        return


    def broaden_travel_times(self):
        """
            Generates rates for commuters traveleing between 40 and 89 and 20 and 39 minutes
        :return:
        """
        trav_recodes = ['travel_time_40_59_rate', 'travel_time_60_89_rate']
        self.DataDict['travel_time_40_89_rate'] = "% commuting between 40 and 89 minutes to work (DeepSolar)"
        create_combo_var_sum(self.merged, trav_recodes, newvar='travel_time_40_89_rate')

        travM_recodes = ['travel_time_20_29_rate','travel_time_30_39_rate',]
        self.DataDict['travel_time_20_39_rate'] = "% commuting between 20 and 39 minutes to work (DeepSolar)"
        create_combo_var_sum(self.merged, travM_recodes, newvar='travel_time_20_39_rate')


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
        self.DataDict[a_25_44] = "ages between 25 to 44 (DeepSolar)"
        self.DataDict[a_25_64] = "ages between 25 to 64 (DeepSolar)"
        self.DataDict[a_55_more] = "ages between 55 or above (DeepSolar)"
        self.DataDict[a_minor_rt] = "ages between 5 and 17 (DeepSolar)"
        self.DataDict[a_zoomer_rt] = "ages between 5 to 24 (DeepSolar)"
        # merged[a_25_44] = create_combo_var_sum(merged, age_25_44, newvar=a_25_44)
        # merged[a_25_64] = create_combo_var_sum(merged, age_25_64)
        # merged[a_55_more] = create_combo_var_sum(merged, age_25_64)
        create_combo_var_sum(self.merged, age_25_44, newvar=a_25_44)
        create_combo_var_sum(self.merged, age_25_64, newvar=a_25_64)
        create_combo_var_sum(self.merged, age_55_85p, newvar=a_55_more)
        create_combo_var_sum(self.merged, age_minor, newvar=a_minor_rt)
        create_combo_var_sum(self.merged, age_zoom, newvar=a_zoomer_rt)

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
        self.DataDict['high_own_hh'] = "'high_own_hh' high income owned homes (SEEDSII)"
        create_combo_var_sum(self.merged, high_owners_hh, newvar='high_own_hh')
        # add total  mid income home owners
        self.DataDict['mid_own_hh'] = "mid income owned homes (SEEDSII)"
        create_combo_var_sum(self.merged, mid_owners_hh, newvar='mid_own_hh')
        # add total  mod income home owners
        self.DataDict['mod_own_hh'] = "moderate income owned homes (SEEDSII)"
        create_combo_var_sum(self.merged, mod_owners_hh, newvar='mod_own_hh')
        # add total  low income home owners
        self.DataDict['low_own_hh'] = "low income owned homes (SEEDSII)"
        create_combo_var_sum(self.merged, low_owners_hh, newvar='low_own_hh')
        # add total  very low income home owners
        self.DataDict['verylow_own_hh'] = "very low income owned hoomes (SEEDSII)"
        create_combo_var_sum(self.merged, verylow_owners_hh, newvar='verylow_own_hh')

        # add total  income home owners
        self.DataDict['total_own_hh'] = "total owned homes (SEEDSII)"
        self.DataDict['total_sf_own_hh'] = "total single family owned homes (SEEDSII)"
        self.DataDict['total_mf_own_hh'] = "total multifamily owned homes (SEEDSII)"
        create_combo_var_sum(self.merged, own_hh_l, newvar='total_own_hh')
        create_combo_var_sum(self.merged, sfown_hh_l, newvar='total_sf_own_hh')
        create_combo_var_sum(self.merged, mfown_hh_l, newvar='total_mf_own_hh')

        # now make rate based versions of owner counts broken into income levels
        #self.merged[high_hh_r] = self.merged['high_own_hh'].values/self.merged['total_own_hh'].values
        self.DataDict[high_hh_r] = "rate of high income owned homes (SEEDSII)"
        self.merged[high_hh_r] = smart_df_divide(self.merged['high_own_hh'].values,self.merged['total_own_hh'].values)
        #self.merged[high_hh_r].replace(np.nan, 0, inplace=True)

        #self.merged[mid_hh_r] = self.merged['mid_own_hh'].values/self.merged['total_own_hh'].values
        self.DataDict[mid_hh_r] = "rate of mid income owned homes (SEEDSII)"
        self.merged[mid_hh_r] = smart_df_divide(self.merged['mid_own_hh'].values, self.merged['total_own_hh'].values)
        #self.merged[mid_hh_r].replace(np.nan, 0, inplace=True)

        #self.merged[mod_hh_r] = self.merged['mod_own_hh'].values/self.merged['total_own_hh'].values
        self.DataDict[mod_hh_r] = "rate of moderate income owned homes (SEEDSII)"
        self.merged[mod_hh_r] = smart_df_divide(self.merged['mod_own_hh'].values, self.merged['total_own_hh'].values)
        #self.merged[mod_hh_r].replace(np.nan, 0, inplace=True)

        #self.merged[low_hh_r] = self.merged['low_own_hh'].values/self.merged['total_own_hh'].values
        self.DataDict[low_hh_r] = "rate of low income owned homes (SEEDSII)"
        self.merged[low_hh_r] = smart_df_divide(self.merged['low_own_hh'].values, self.merged['total_own_hh'].values)
        #self.merged[low_hh_r].replace(np.nan, 0, inplace=True)

        #self.merged[verylow_hh_r] = self.merged['verylow_own_hh'].values/self.merged['total_own_hh'].values
        self.DataDict[verylow_hh_r] = "rate of very low income owned homes (SEEDSII)"
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
        self.DataDict['high_own_Sbldg'] = "high income owned homes suitable for solar installations (SEEDSII)"
        create_combo_var_sum(self.merged, high_own_bldg, newvar='high_own_Sbldg')
        # add total  suitable mid income home owners
        self.DataDict['mid_own_Sbldg'] = "mid income owned homes suitable for solar installations (SEEDSII)"
        create_combo_var_sum(self.merged, mid_own_bldg, newvar='mid_own_Sbldg')
        # add total  suitable mod income home owners
        self.DataDict['mod_own_Sbldg'] = "moderate income owned homes suitable for solar installations (SEEDSII)"
        create_combo_var_sum(self.merged, mod_own_bldg, newvar='mod_own_Sbldg')
        # add total  suitable low income home owners
        self.DataDict['low_own_Sbldg'] = "low income owned homes suitable for solar installations (SEEDSII)"
        create_combo_var_sum(self.merged, low_own_bldg, newvar='low_own_Sbldg')
        # add total  suitable very low income home owners
        self.DataDict['verylow_own_Sbldg'] = "verylow owned homes suitable for solar installations (SEEDSII)"
        create_combo_var_sum(self.merged, verylow_own_bldg, newvar='verylow_own_Sbldg')
        # add total  suitable income home owners
        self.DataDict['total_own_Sbldg'] = "total owned homes suitable for solar installations (SEEDSII)"
        create_combo_var_sum(self.merged, own_bldg_cnt, newvar='total_own_Sbldg')

        # now make rate based versions
        #self.merged['high_own_Sbldg_rt'] = self.merged['high_own_Sbldg'].values / self.merged['total_own_Sbldg'].values
        self.DataDict['high_own_Sbldg_rt'] = "% high income owned homes suitable for solar installations (SEEDSII)"
        self.merged['high_own_Sbldg_rt'] = smart_df_divide(self.merged['high_own_Sbldg'].values,
                                                           self.merged['total_own_Sbldg'].values)
        #self.merged['mid_own_Sbldg_rt'] = self.merged['mid_own_Sbldg'].values / self.merged['total_own_Sbldg'].values
        self.DataDict['mid_own_Sbldg_rt'] = "% mid income owned homes suitable for solar installations (SEEDSII)"
        self.merged['mid_own_Sbldg_rt'] = smart_df_divide(self.merged['mid_own_Sbldg'].values,
                                                          self.merged['total_own_Sbldg'].values)
        #self.merged['mod_own_Sbldg_rt'] = self.merged['mod_own_Sbldg'].values / self.merged['total_own_Sbldg'].values
        self.DataDict['mod_own_Sbldg_rt'] = "% moderate income owned homes suitable for solar installations (SEEDSII)"
        self.merged['mod_own_Sbldg_rt'] = smart_df_divide(self.merged['mod_own_Sbldg'].values,
                                                          self.merged['total_own_Sbldg'].values)
        #self.merged['low_own_Sbldg_rt'] = self.merged['low_own_Sbldg'].values / self.merged['total_own_Sbldg'].values
        self.DataDict['low_own_Sbldg_rt'] = "% low ncome owned homes suitable for solar installations (SEEDSII)"
        self.merged['low_own_Sbldg_rt'] = smart_df_divide(self.merged['low_own_Sbldg'].values,
                                                          self.merged['total_own_Sbldg'].values)
        #self.merged['verylow_own_Sbldg_rt'] = self.merged['verylow_own_Sbldg'].values / self.merged['total_own_Sbldg'].values
        self.DataDict['verylow_own_Sbldg_rt'] = "% very low income owned homes suitable for solar installations (SEEDSII)"
        self.merged['verylow_own_Sbldg_rt'] = smart_df_divide(self.merged['verylow_own_Sbldg'].values,
                                                              self.merged['total_own_Sbldg'].values)


        # add a total owner capacity variable
        self.DataDict['Tot_own_mw'] = "Total owned generation capacity (SEEDSII)"
        create_combo_var_sum(self.merged, caps_to_sum_own, newvar='Tot_own_mw')
        # add a total owner annual generation variable
        self.DataDict['Yr_own_mwh'] = "Total annual generatio potential (SEEDSII)"
        create_combo_var_sum(self.merged, ann_gen, newvar='Yr_own_mwh')
        # add a total owner annual generation variable
        self.DataDict['high_own_mwh'] = "high income owned generation potential (SEEDSII)"
        create_combo_var_sum(self.merged, high_income_gen, newvar='high_own_mwh')
        # add a total owner annual generation variable
        self.DataDict['mid_own_mwh'] = "mid income owned generation potential (SEEDSII)"
        create_combo_var_sum(self.merged, mid_income_gen, newvar='mid_own_mwh')
        # add a total owner annual generation variable
        self.DataDict['mod_own_mwh'] = "moderate income owned generation potential (SEEDSII)"
        create_combo_var_sum(self.merged, mod_income_gen, newvar='mod_own_mwh')
        # add a total owner annual generation variable
        self.DataDict['low_own_mwh'] = "low income owned generation potential (SEEDSII)"
        create_combo_var_sum(self.merged, low_income_gen, newvar='low_own_mwh')
        # add a total owner annual generation variable
        self.DataDict['verylow_own_mwh'] = "verylow income owned generation potential (SEEDSII)"
        create_combo_var_sum(self.merged, verylow_income_gen, newvar='verylow_own_mwh')

        very_low_own_elep = ['very_low_sf_own_elep_hh', 'very_low_mf_own_elep_hh', ]
        low_own_elep = ['low_sf_own_elep_hh', 'low_mf_own_elep_hh', ]
        mod_own_elep = ['mod_sf_own_elep_hh', 'mod_mf_own_elep_hh', ]
        high_own_elep = ['high_sf_own_elep_hh', 'high_mf_own_elep_hh', ]


        # add a total owner annual generation variable
        self.DataDict['high_own_elep_hh'] = "high income owned electrical price by household (SEEDSII)"
        create_combo_var_sum(self.merged, high_own_elep, newvar='high_own_elep_hh')
        self.merged['high_own_elep_hh'] = self.merged['high_own_elep_hh'].values/2

        # add a total owner annual generation variable
        self.DataDict['mod_own_elep_hh'] = "moderate income owned electrical price by household (SEEDSII)"
        create_combo_var_sum(self.merged, mod_own_elep, newvar='mod_own_elep_hh')
        self.merged['mod_own_elep_hh'] = self.merged['mod_own_elep_hh'].values / 2

        # add a total owner annual generation variable
        self.DataDict['low_own_elep_hh'] = "low income electrical prices (SEEDSII)"
        create_combo_var_sum(self.merged, low_own_elep, newvar='low_own_elep_hh')
        self.merged['low_own_elep_hh'] = self.merged['low_own_elep_hh'].values / 2

        # add a total owner annual generation variable
        self.DataDict['verylow_own_elep_hh'] = "very low income owned electrical price by household (SEEDSII)"
        create_combo_var_sum(self.merged, very_low_own_elep, newvar='verylow_own_elep_hh')
        self.merged['verylow_own_elep_hh'] = self.merged['verylow_own_elep_hh'].values / 2

        total_elp =['verylow_own_elep_hh', 'low_own_elep_hh','mod_own_elep_hh', 'high_own_elep_hh']
        self.DataDict['total_own_elep'] = "total owned electrical price (SEEDSII)"
        create_combo_var_sum(self.merged, total_elp, newvar='total_own_elep')
        self.merged['total_own_elep'] = self.merged['total_own_elep'].values/len(total_elp)
        # now get the possible savings total and for each group
        self.DataDict['Yrl_savings_$'] = "dollars_per_kwH (DeepSolar) / yearly possible genartion potential (SEEDSII)"
        self.merged['Yrl_savings_$'] = (self.merged['dlrs_kwh'] *1000)* self.merged['Yr_own_mwh']
        self.DataDict['Yrl_%_inc'] = "yearly savings / average income"
        self.merged['Yrl_%_inc'] = self.merged['Yrl_savings_$']/self.merged["average_household_income"]
        self.DataDict['Yrl_%_$_kwh'] = "yearly savings ($/kwh/year) / total owned developable planes"
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
        self.DataDict['high_own_devp'] = "high income developable plane area"
        create_combo_var_sum(self.merged, high_owners_devp, newvar='high_own_devp')
        # add total  mid income home owners
        self.DataDict['mid_own_devp'] = "mid income owned developable plane area (SEEDSII)"
        create_combo_var_sum(self.merged, mid_owners_devp, newvar='mid_own_devp')
        # add total  mod income home owners
        self.DataDict['mod_own_devp'] = "moderate income owned developable plane area (SEEDSII)"
        create_combo_var_sum(self.merged, mod_owners_devp, newvar='mod_own_devp')
        # add total  low income home owners
        self.DataDict['low_own_devp'] = "low income owned developable plane area (SEEDSII)"
        create_combo_var_sum(self.merged, low_owners_devp, newvar='low_own_devp')
        # add total  very low income home owners
        self.DataDict['verylow_own_devp'] = "very low income owned developable plane area (SEEDSII)"
        create_combo_var_sum(self.merged, verylow_owners_devp, newvar='verylow_own_devp')
        # add total  income home owners
        self.DataDict['total_own_devp'] = "total owned developable plane area (SEEDSII)"
        create_combo_var_sum(self.merged, devp_own_tot, newvar='total_own_devp')
        self.DataDict['total_own_devpC'] = "total owned developable planes"
        create_combo_var_sum(self.merged, devp_own_totC, newvar='total_own_devpC')

    def make_state_dummies(self):
        states = set(self.merged['state'].values.tolist())
        # for each state make an empty array of zeros of the needed size
        # and then fill in ones where the state is in states
        for st in states:
            self.merged[st] = np.zeros(len(self.merged))
            self.merged.loc[self.merged['state'] == st, st] = 1

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

        self.DataDict['Top_3_States'] = "binary indicator of a census tract being in one of the top 3 adoptions per household census tracts which are CA, NV, AZ"
        self.merged['Top_3_States'] = np.zeros(len(self.merged))

        self.merged.loc[self.merged['state'].isin(['ca', 'nv', 'az']), 'Top_3_States'] = 1
        return

    def random_interactions(self):

        self.DataDict['Savings_potential'] = "product of daily solar radiation (DeepSolar), total owned developable plane area (SEEDSII derived), dollars per kilo watt hour(DeepSolar)"
        self.merged['Savings_potential'] = self.merged['daily_solar_radiation'].values * self.merged['total_own_devp'].values * self.merged['dlrs_kwh'].values
        self.DataDict['Income_x_College_Edu'] = "prduct of median household income(DeepSolar) and education_college percent (DeepSolar)"
        self.merged['Income_x_College_Edu'] = self.merged["median_household_income"] * self.merged['education_college_rate']
        self.DataDict['Cost_x_Consumption'] = "dollars per kilo watt hour (DeepSolar) * average montlhy energy consumption (kwh)(DeepSolar)"
        self.merged['Cost_x_Consumption'] = self.merged['dlrs_kwh'] * self.merged['avg_monthly_consumption_kwh']

        self.DataDict['urban_diversity'] = "binary Urban locale * racial diversity(rate of race mixture) (DeepSolar)"
        self.merged['urban_diversity'] = self.merged['URBAN'] * self.merged['diversity']
        self.DataDict['suburban_diversity'] = "binary suburban locale * racial diversity(rate of race mixture) (DeepSolar) "
        self.merged['suburban_diversity'] = self.merged['locale_recode(suburban)'] * self.merged['diversity']
        self.DataDict['rural_diversity'] = "binary rural locale * racial diversity(rate of race mixture) (DeepSolar) "
        self.merged['rural_diversity'] = self.merged['locale_recode(rural)'] * self.merged['diversity']


    def split_solarresources(self):
        # add designations for the higher and lower than average daily solar areas
        self.DataDict['High_Solar_Areas'] = "binary indicator for census tracts above the average value for daily solar radiation"
        self.merged['High_Solar_Areas'] = np.full(len(self.merged), 0.0)
        self.merged.loc[self.merged['daily_solar_radiation'] > self.merged['daily_solar_radiation'].mean(), 'High_Solar_Areas'] = 1
        self.DataDict['Low_Solar_Areas'] = "binary indicator for census tracts below the average value for daily solar radiation"
        self.merged['Low_Solar_Areas'] = np.full(len(self.merged), 0.0)
        self.merged.loc[self.merged['daily_solar_radiation'] < self.merged['daily_solar_radiation'].mean(), 'Low_Solar_Areas'] = 1
        self.DataDict['DS_HighSolar'] = "binary indicator for census tracts above the threshold found by DeepSolar in daily solar radiation where adoption seemed to spike"
        self.merged['DS_HighSolar'] = np.zeros(len(self.merged))
        self.merged.loc[self.merged['daily_solar_radiation'] > 4.5,'DS_HighSolar'] = 1




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
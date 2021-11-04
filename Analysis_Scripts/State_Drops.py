import pandas

US_drops = ['land_area', 'education_master', 'housing_unit_median_value', 'high_sf_own_mwh', 'high_mf_own_mw', 'education_bachelor',
            'high_mf_own_devp_m2', 'hu_monthly_owner_costs_greaterthan_1000dlrs', 'high_mf_own_mwh', 'pop_caucasian',
            'incentive_count_residential', 'high_sf_own_bldg_cnt', 'high_mf_own_bldg_cnt', 'high_sf_own_devp_m2', 'Yr_own_mwh',
            'high_sf_own_mw', 'high_sf_own_hh','cdd',  'incentive_count_nonresidential', 'education_bachelor_rate',
            'education_high_school_or_below_rate', 'education_high_school_graduate_rate', 'high_sf_own_devp_cnt',
            'education_bachelor_or_above_rate', 'p16_employed', 'fam_med_income', 'average_household_income',
            'number_of_years_of_education', 'high_own_hh', ]

general_dopys = ['land_area',  'travel_time_30_39_rate',        
                 'travel_time_10_19_rate',
                 'hu_med_val','mortgage_with_rate', 'high_mf_own_mwh',  ]
slr_drp = ['pop25_some_college_plus',    ]
slr_drp = []
group_drop_dic = {
        "very low": ['housing_unit_median_value', 'high_mf_own_hh', 'high_mf_own_devp_m2', 'high_mf_own_mw',
                     'land_area', 'education_bachelor_or_above_rate', 'education_bachelor_rate',
                     'high_mf_own_devp_cnt', 'high_mf_own_mwh', 'pop_hispanic', 'pop_caucasian', 'fam_med_income',
                     'p16_unemployed', 'hu_vintage_1940to1959', 'mid_mf_own_devp_m2','population', 'household_count',
                     'education_master_or_above_rate', 'average_household_income', ] + slr_drp,
    
        "lowA": ['housing_unit_median_value','high_mf_own_devp_m2', 'high_mf_own_mw','land_area',
                'mid_mf_own_mw', 'mid_mf_own_devp_m2', 'high_sf_own_bldg_cnt', 'high_sf_own_devp_cnt',
                'mid_mf_own_mwh', 'high_sf_own_hh',  'high_mf_own_mwh', 'hu_own',
                'high_mf_own_devp_cnt', 'education_bachelor', 'pop_female', 'total_area','education_master', 'high_sf_own_devp_cnt',
                'high_sf_own_bldg_cnt', 'high_mf_own_mw', 'high_sf_own_hh', 'hu_own', 'high_sf_own_bldg_cnt', 'high_mf_own_devp_m2',
                'high_mf_own_devp_cnt', 'total_area', 'high_mf_own_devp_m2', 'average_household_income', 'fam_med_income',
                'high_sf_own_mwh', 'Tot_own_mw', 'Yr_own_mwh', 'mid_mf_own_bldg_cnt', 'high_sf_own_mw', 'high_sf_own_devp_m2',
                'median_household_income', 'mid_sf_own_bldg_cnt', 'med_inc_ebill_dlrs', 'mid_sf_own_mw', 'avg_inc_ebill_dlrs',
                'education_bachelor_or_above_rate', 'number_of_years_of_education', 'median_household_income', 'education_bachelor_rate',
                'mid_sf_own_mwh', 'mid_sf_own_devp_cnt',  'mid_sf_own_hh', 'transportation_public_rate', 'pop25_some_college_plus',
                'mid_sf_own_devp_m2', 'mid_own_Sbldg', 'total_own_hh', 'high_own_hh', 'mid_own_Sbldg', 'total_own_Sbldg',
                'high_own_hh', 'high_own_mwh', ] + slr_drp,
        "low": ['housing_unit_median_value', 'land_area',
                 ] + slr_drp,
        'US_low': [ 'heating_fuel_gas_rate', 'heating_fuel_fuel_oil_kerosene',],
        "medium": ['housing_unit_median_value','high_mf_own_devp_m2', 'high_mf_own_mw','land_area', 
                   'education_bachelor',
                  'education_high_school_or_below_rate', 'travel_time_less_than_10_rate', 
                  'education_bachelor_or_above_rate','number_of_years_of_education',
                  'high_sf_own_mwh', 'Yr_own_mwh', 'high_sf_own_mw', 'education_master_or_above_rate',
                  'high_sf_own_devp_m2', 'Tot_own_mw', 'high_mf_own_devp_cnt', 'high_mf_own_mwh',]+ slr_drp,
        "highA": ['housing_unit_median_value','high_mf_own_devp_m2', 'high_mf_own_mw','land_area',
                'hdd', 'high_mf_own_devp_cnt',  'high_mf_own_mwh',
                'mid_mf_own_mwh', 'dlrs_kwh x median_household_income',
                'avg_electricity_retail_rate', 'incentive_residential_state_level', ] + slr_drp,
        "high": ['housing_unit_median_value', 'land_area',
                 ] + slr_drp,
}


state_drop_dic = { 
    "al": ['high_mf_own_mwh', 'high_sf_own_mw', 'Yr_own_mwh', 'education_bachelor', 'high_sf_own_mwh',
           'high_sf_own_hh','education_bachelor_or_above_rate','education_professional_school',
           'high_mf_own_devp_m2', 'high_sf_own_devp_m2',  'mid_sf_own_mw',
          'heating_fuel_gas', 'mid_sf_own_devp_m2', 'high_mf_own_mw', 'Tot_own_mw', 'pop25_some_college_plus',
          'education_bachelor_rate', 'education_master_or_above_rate', 
          'fam_med_income', 'high_sf_own_bldg_cnt', 'high_mf_own_bldg_cnt',
          'hu_monthly_owner_costs_greaterthan_1000dlrs', 'number_of_years_of_education', 'household_count',
          'hu_vintage_2000to2009', 'average_household_income', 'population',
          'pop_asian', 'p16_unemployed', 'pop_female', 'housing_unit_median_value',
          'mid_sf_own_mwh',  'mid_sf_own_hh', 'pop_us_citizen',
          'pop_male', 'hu_own',  'p16_employed', 'education_high_school_or_below_rate', 
          'high_sf_own_devp_cnt',  'mid_sf_own_bldg_cnt', 'poverty_family_count', 'mod_sf_own_mw',
          'mod_sf_own_mwh', 'mod_sf_own_devp_m2', 
           'education_high_school_graduate_rate', 'mod_sf_own_mw', 'poverty_family_count',
          'hu_2000toafter_pct','hh_size_4','median_household_income', 'hh_size_2',
          'median_household_income', 'Pro_Occup', 'med_inc_ebill_dlrs', 'pop_caucasian',
          'mid_mf_own_bldg_cnt', 'mid_mf_own_mwh', 'dlrs_kwh x median_household_income',
          'mid_mf_own_mw', 'mid_mf_own_devp_m2', 'education_college', 'mid_sf_own_devp_cnt', 
          'housing_unit_count','travel_time_40_89_rate',
          'occupation_transportation_rate', 'total_area', 'fam_children_6to17', 'hu_1960to1979_pct',
          'hh_size_3', 'education_less_than_high_school_rate',
          'heating_fuel_electricity', 'mod_sf_own_hh', 'hu_2000toafter', 'education_master',
          'mod_sf_own_bldg_cnt', 'mod_sf_own_devp_cnt', 'hu_monthly_owner_costs_lessthan_1000dlrs',
          'education_professional_school_rate', 'occupation_construction_rate','education_high_school_graduate',
          'low_sf_own_devp_m2','mid_mf_own_devp_cnt', 'low_sf_own_mw', 'education_bachelor_or_above_rate',
          'hu_own_pct', 'mod_own_mwh', 'mid_own_Sbldg', ] + general_dopys,
    
    "ar": ['Anti_Occup','occupation_manufacturing_rate','age_15_17_rate','heating_fuel_gas',
           'education_high_school_or_below_rate','education_master_rate', 'education_master',
           'education_bachelor_or_above_rate', 'travel_time_less_than_10_rate',
          'education_bachelor', 'number_of_years_of_education', 'high_mf_own_devp_m2','high_mf_own_mw',
          'education_high_school_graduate_rate', 'education_professional_school_rate', 'p16_unemployed',
          'pop25_some_college_plus', 'education_master_or_above_rate',  'education_master',
          'high_mf_own_devp_cnt', '%hh_size_1', 'hh_size_1',  'hu_monthly_owner_costs_greaterthan_1000dlrs',
           'p16_employed', 'education_bachelor_rate', 'pop_african_american',
          'hh_gini_index', 'diversity', 'travel_time_40_89_rate', 'hu_1959toearlier_pct',
          'travel_time_20_29_rate',  'hu_vintage_1940to1959', 'fam_med_income',
          'age_45_54_rate', 'high_mf_own_bldg_cnt', 'household_count', 'education_professional_school',
          'high_sf_own_hh', 'age_25_44_rate', 'population', 'pop_male',] + general_dopys,
    
    "az": ['high_mf_own_devp_cnt', 'occupation_administrative_rate', 'high_sf_own_bldg_cnt',
           'occupation_agriculture_rate','net_metering_hu_own', 'high_sf_own_mw', 'high_mf_own_mw',
          'high_mf_own_bldg_cnt','mid_sf_own_devp_m2', 'hu_own_pct', 'hdd', 'high_mf_own_hh', 
           'high_mf_own_devp_m2', 'locale_recode(rural)', 'high_sf_own_mwh', 'fam_med_income',
          'heating_fuel_electricity', 'high_sf_own_devp_m2', 'high_sf_own_hh', 'hu_own', 'high_mf_own_mw',
          'education_bachelor','med_inc_ebill_dlrs', 'education_bachelor_rate', 'high_sf_own_bldg_cnt',
          'high_mf_own_mw', 'high_sf_own_mw', 'high_sf_own_devp_cnt', 'average_household_income',
          'median_household_income', 'hu_monthly_owner_costs_greaterthan_1000dlrs',
          'occupancy_owner_rate',  'cooling_design_temperature', #'high_own_bldg_cnt',
          'education_less_than_high_school_rate',
           'education_high_school_or_below_rate', 'pop_african_american', 'dlrs_kwh x median_household_income',
          'education_bachelor_or_above_rate', 'pop_asian',  'pop25_some_college_plus',
          'low_sf_own_elep_hh', 'cdd', 'education_high_school_graduate_rate', 'mod_sf_own_elep_hh',
           'high_sf_own_elep_hh', 'mid_mf_own_mwh', 
          'very_low_sf_own_elep_hh', 'transportation_walk_rate', 'mid_mf_own_mw', 
           'mid_mf_own_devp_m2', 'Yr_own_mwh', 'mid_mf_own_devp_cnt', 'p16_employed',] + general_dopys,
    
    "ca": ['heating_fuel_gas_rate', 'pop_hispanic',  # 'heating_fuel_other_rate',
           'heating_fuel_fuel_oil_kerosene_rate',  'population', 'age_55_64_rate',
           'high_mf_own_hh', 'pop_female', 'heating_fuel_gas', 'pop_male',  # 'pop_under_18',
           'age_65_74_rate', 'age_median', 'diversity', #heating_fuel_other',
           'p16_unemployed', '%hh_size_2', 'p16_unemployed', '%hh_size_2', 'heating_design_temperature',
           'hh_size_4', 'housing_unit_median_value', 'high_own_Sbldg_rt',
           'poverty_family_count', 'hu_own_pct',  'incent_cnt_res_own',] + general_dopys,
    
    "co": ['%hh_size_2', 'locale_recode(rural)', 'heating_fuel_gas', 'age_55_or_more_rate', 
           'population', 'age_median', 'age_25_34_rate', 'pop_male',
          'p16_unemployed', 'hh_size_4', 'pop_us_citizen','pop_under_18', 'company_ty_nc',
           'Anti_Occup', 'avg_monthly_consumption_kwh', 'dlrs_kwh', 'pop_caucasian', 
           'p16_employed',  'household_count',  'pop_hispanic', 
           'pop_male', 'heating_fuel_gas_rate', 'low_mf_own_bldg_cnt', 'hu_own',
          'locale_dummy', ] + general_dopys,
    
    "ct": ['pop_us_citizen',  'population', 'hh_size_3', 'high_mf_own_bldg_cnt', 'pop_asian',
          'cdd', 'housing_unit_count', 'pop_male', 'pop25_some_college_plus', 'hu_own', 'hh_size_1',
          'household_count', 'p16_employed', 'poverty_family_count',] + general_dopys,
    
    "dc": ['hu_vintage_1939toearlier', 'mid_mf_own_mwh', 'mid_mf_own_devp_m2', 'hu_1959toearlier',
          'mid_sf_own_mwh', 'mid_sf_own_devp_m2', 'Anti_Occup','Tot_own_mw',
          'mid_sf_own_mw', 'mid_mf_own_devp_cnt', 'mod_mf_own_mwh', 'mod_mf_own_mw', 'hu_2000toafter_pct',
           'mod_mf_own_mwh', 'mod_mf_own_devp_cnt', 'Yr_own_mwh',] + general_dopys,
    
    "de": ['mod_sf_own_mwh', 'mod_sf_own_devp_m2', 'mod_sf_own_mw', 'mod_sf_own_bldg_cnt','male_pct', 
          'education_doctoral', 'mod_sf_own_devp_cnt', 'hu_no_mortgage', 'hh_size_2',
          'female_pct', 'hu_monthly_owner_costs_lessthan_1000dlrs', 'education_bachelor_or_above_rate',
          ] + general_dopys,
    
    "fl": ['hu_own', 'cdd', 'Tot_own_mw', 'Yr_own_mwh', 'high_sf_own_hh', 'high_mf_own_bldg_cnt', 
           'high_sf_own_devp_cnt', 'high_sf_own_mw', 'high_sf_own_devp_m2', 'high_sf_own_mwh',
           'high_sf_own_bldg_cnt', 'high_own_bldg_cnt', 'high_mf_own_devp_cnt', 'high_mf_own_mw',
          'net_metering_hu_own', 'mid_sf_own_devp_m2', 'pop25_some_college_plus',
          'mid_sf_own_bldg_cnt', 'number_of_years_of_education', 'average_household_income', 
           'education_high_school_or_below_rate', 'education_less_than_high_school_rate', 
           'mid_sf_own_devp_cnt', 'Yr_own_mwh', 'mid_sf_own_mw', 'high_mf_own_devp_m2',
          'median_household_income', 'fam_med_income', 'hu_own_pct',  'occupancy_owner_rate',
          'mid_mf_own_mw', 'dlrs_kwh x median_household_income', 'mod_sf_own_devp_cnt',
          'mid_mf_own_mwh', 'education_master', 'education_bachelor_rate', 'mid_mf_own_bldg_cnt',
          'med_inc_ebill_dlrs', 'high_mf_own_hh', 'mid_mf_own_devp_m2', 'hu_monthly_owner_costs_greaterthan_1000dlrs',
          'education_master_or_above_rate', 'education_master_rate', 'education_college', 'pop_us_citizen',
          'mod_sf_own_bldg_cnt', 'hh_size_2', 'mod_mf_own_devp_cnt', 'education_bachelor_or_above_rate',] + general_dopys,
    
    "ga": ['high_sf_own_mw', 'high_sf_own_devp_m2', 'high_sf_own_mwh',  'Yr_own_mwh', 
          'high_sf_own_devp_cnt', 'high_sf_own_bldg_cnt', 'high_sf_own_hh', 'population',
           'pop25_some_college_plus', 'Tot_own_mw', 'p16_employed', 'household_count', 'hu_own',
          'poverty_family_count', 'pop_hispanic', 'pop_us_citizen', 'pop_female', 'education_bachelor_rate',
          'hu_monthly_owner_costs_greaterthan_1000dlrs', 'education_bachelor_or_above_rate',] + general_dopys,
    
    "ia": ['housing_unit_count', 'pop_us_citizen', 'age_18_24_rate','population', 'pop25_some_college_plus',
           'pop_us_citizen', 'p16_unemployed', 'hu_own', 'pop_male',  'p16_employed', 'household_count',
          'poverty_family_count', 'pop_female', 'heating_fuel_gas', 
          'pop_caucasian',  'hh_size_2', 'high_sf_own_hh', 'Yr_own_mwh', 'mid_sf_own_hh',] + general_dopys,
    
    "id": ['pop_caucasian', 'population','p16_unemployed', 'pop25_some_college_plus','pop_male',
           'pop_us_citizen', 'p16_employed', 'heating_fuel_gas_rate', 'education_bachelor_rate',
          'hu_own', 'high_mf_own_bldg_cnt', 'household_count', 'Yr_own_mwh',] + general_dopys,
    
    "il": ['high_sf_own_mwh', 'high_sf_own_mw', 'pop_over_65', 'heating_fuel_gas',
          'high_sf_own_devp_m2', 'hu_own', 'Tot_own_mw', 'high_sf_own_devp_cnt', 'Yr_own_mwh',
          'housing_unit_count', 'high_sf_own_bldg_cnt', 'household_count',  'pop25_some_college_plus',
          'high_sf_own_hh', 'population', 'pop_female', 'pop_male', 'pop_us_citizen','fam_children_6to17',
          'hu_monthly_owner_costs_greaterthan_1000dlrs',] + general_dopys,
    
    "in": ['pop25_some_college_plus', 'education_bachelor_rate', 'p16_employed',  'pop_caucasian', 'education_bachelor_or_above_rate', 
          ] + general_dopys,
    
    "ks": ['pop25_some_college_plus', 'hu_own', 'pop_us_citizen', 'pop_caucasian', 'population', 'household_count',
          'high_sf_own_hh', ] + general_dopys,
    
    "ky": ['population', 'pop_african_american', 'education_master', 'education_bachelor',
           'education_master_rate', 'pop25_some_college_plus', 'pop_female', 'number_of_years_of_education', 'household_count',
          'education_bachelor_or_above_rate',] + general_dopys,
    
    "la": ['cooling_design_temperature', 'heating_design_temperature', 'pop25_some_college_plus', 'cdd', 'high_mf_own_mw',
          'hu_monthly_owner_costs_greaterthan_1000dlrs',  'education_high_school_or_below_rate', 'high_mf_own_devp_cnt',
          ] + general_dopys,
    
    "ma": ['poverty_family_count', 'heating_fuel_gas', 'population', 'p16_unemployed', 'pop_us_citizen',  '%hh_size_2', '%hh_size_1', 
          'hh_size_3', '%hh_size_4', ] + general_dopys,
    
    "md": ['med_inc_ebill_dlrs', 'median_household_income', 'average_household_income', 'average_household_income',
           'education_bachelor', 'med_inc_ebill_dlrs', 'median_household_income', 'avg_inc_ebill_dlrs',
           'hu_monthly_owner_costs_greaterthan_1000dlrs', 'pop25_some_college_plus',  'education_high_school_graduate_rate',
          ] + general_dopys,
    
    "me": ['hu_vintage_1939toearlier', 'hu_1959toearlier', 'heating_fuel_electricity', 'p16_employed', 'travel_time_average',
          'pop_asian',] + general_dopys,
    "mi": ['high_mf_own_devp_cnt', 'education_bachelor_or_above_rate', 'pop25_some_college_plus', 'heating_fuel_gas_rate',
          'education_bachelor_rate', 'high_mf_own_devp_m2', 'high_mf_own_bldg_cnt',] + general_dopys,
    
    "mn": ['education_master_or_above_rate', 'education_doctoral', 'p16_unemployed', 'heating_fuel_gas', 
           'education_bachelor_or_above_rate', 'education_high_school_graduate_rate',] + general_dopys,
    
    "mo": ['education_high_school_graduate_rate', 'number_of_years_of_education', 'pop25_some_college_plus',
          'age_25_44_rate', 'education_bachelor_or_above_rate', 'education_high_school_or_below_rate', 'education_bachelor_rate',
          'education_master_or_above_rate', 'high_mf_own_mw',] + general_dopys,
    
    "ms": ['education_bachelor_or_above_rate', 'pop25_some_college_plus', 'education_master', 'education_less_than_high_school_rate',
          'education_bachelor_rate', 'number_of_years_of_education', 'high_sf_own_mw', 'high_sf_own_bldg_cnt',
          ] + general_dopys,
    
    "mt": ['population', 'hh_size_3', 'pop_female', 'hh_size_1', 'pop25_some_college_plus', 'pop_us_citizen', 'pop_male',
          ] + general_dopys,
    
    "nc": ['high_mf_own_bldg_cnt', 'high_mf_own_devp_m2', 'education_master_or_above_rate', 'education_high_school_graduate_rate',
           'education_high_school_or_below_rate', 'education_bachelor_or_above_rate',] + general_dopys,
    "nd": ['heating_fuel_electricity', 'population', 'hu_vintage_2000to2009', 'pop_female',] + general_dopys,
    
    "ne": ['hu_own', 'pop25_some_college_plus', 'household_count', 'education_bachelor_or_above_rate', 'pop_caucasian',
           'education_bachelor_rate',] + general_dopys,
    
    "nh": [] + general_dopys,
    
    "nj": [] + general_dopys,
    
    "nm": [] + general_dopys,
    
    "nv": [] + general_dopys,
    "ny": [] + general_dopys,
    
    "oh": [] + general_dopys,
    
    "ok": [] + general_dopys,
    "or": [] + general_dopys,
    "pa": [] + general_dopys,
    "ri": [] + general_dopys,
    "sc": [] + general_dopys,
    "sd": [] + general_dopys,
    "tn": [] + general_dopys,
    "tx": [] + general_dopys,
    "ut": [] + general_dopys,
    "va": [] + general_dopys,
    "vt": [] + general_dopys,
    "wa": [] + general_dopys,
    "wi": [] + general_dopys,
    "wv": [] + general_dopys,
    "wy": [] + general_dopys,
    
}
import numpy as np
import pandas as pd

edu_25_list = ['pop25_some_college_plus',
               'education_population',
               'education_college_rate',
               'education_bachelor_rate',
               'education_college',
               'education_doctoral',
               'education_master',
               ]


PaperSetVars = [
'Adoption','Solar_installations_per_home_owner','Solar_installations_per_household',
'AverageHouseholdSize','Commutingtime40minsdaily','Cooling_degree_days','Daily_solar_radiation','Electricity_price',
'Electricityconsumption','Employment','Gender_Ratio',
'Has_PropertyTax','Has_NetMetering',
'Homeowner','avg_monthly_consumption_kwh_log10_orig',
'MedianAge','MedianHHIncome','Medianhomevalue',
'Newdwellings','Rural','SomeCollege','Suburban',
'@ne','@or','al','ar','az','ca','co','ct','dc','de','fl','ga','ia','id','il','in','ks','ky','la','ma','md',
'me','mi','mn','mo','ms','mt','nc','nd','nh','nj','nm','nv','ny','oh','ok','pa','ri','sc','sd',
'tn','tx','ut','va','vt','wa','wi','wv','wy',
'age_65_74_rate_log10_orig','age_median_log10_orig',
'age_minor_rate_log10_orig','AgricultureOccu','AnnualGeneration_MWh_10','AnnualGenerationMWh','ArtsOccu',
'Asian_pct','average_household_size_log10_orig','Average_HouseholdSize_10','Average_Income',
'average_monthly_bill_$_10','averagemonthlybill$','AveragePanelArea','avg_monthly_bill_dlrs_log10_orig',
'Bachelors','Bachelors#','BachelorsorAbove','Black_pct', 'cdd_log10_orig',
'cdd_std_log10_orig','Commutingbybicycle','Commutingbybicycle_10','Commutingbywalking','Commutingbywalking_10','Commutingtime1019minsdaily',
'Commutingtime1019minsdaily_10','Commutingtime10minsdaily','Commutingtime10minsdaily_10','ConstructionOccu',
'@_Commuting_time_40minsdaily_10','@_Elderly_10','@_Minors_10','@_solar_generation_10','@#_Asian_10',
'@#_Black_10','@#_Hispanic_10','@#_White_10','@#25yearsofage','@#age252HSEdu','@#age25noHSEdu','@#Asian',
'@#Black','@#Hispanic','@#Homeowner','@#HSGrads','@#LessHS','@#MSs','@#PhDs',
'@#ResidentialIncentives','@#ResidentialIncentives_10','@#ResidentialStateInc',
'@#ResidentialStateInc_10','@#Some_college_or_More_10','@#SomecollegeorMore',
'@#White','@16unemployed','@25yearswithsomecollege', 'Cooling_degree_days_std',
'Coolingdegreeday_10',
'Coolingdegreedaystd_10','CoolingDesignTemp','Daily_solar_radiation_10','daily_solar_radiation_log10_orig',
'Diversityminority','Doctoral','edu_college','EducationOccu',
'Elderly','Electricity_consumption_10','Employed','Employed_10','Estimated_savings_$year',
'Estimated_savings_$year_10','familytypehomes','FinanceOccupation','Gender_female','Green_Travel_habits_10',
'Green_Travelers_log10_orig','GreenTravelhabits','hdd_log10_orig','hdd_std_log10_orig',
'Heating_degree_day_10','Heating_degree_day_std','Heating_degree_days',
'Heating_degree_days_10','heating_fuel_coal_coke_rate_log10_orig','heating_fuel_electricity_rate_log10_orig',
'Heating_Source_Coal__10','Heating_source_electricity__10','HeatingDesignTemp',
'HeatingSourceCoal','Heatingsourceelectricity','HIgenerationMWh','HIgenerationMWh_10','high_avg_occu',
'high_med_occu','high_own_devp_log10_orig','high_own_mwh_log10_orig','HighSchool','Highschoolorbelow',
'HIRPVrooftops','HIRPVrooftops_10','Hispanic_pct',
'Hot_or_Not','Hot_or_Not_75','Hot_or_Not_85',
'hu_med_val_log10_orig','incent_cnt_res_own_log10_orig','incentive_count_residential_log10_orig',
'incentive_residential_state_level_log10_orig','InformationOccu','LessThanHighSchool','LIgenerationMWh',
'LIgenerationMWh_10','LIRPVrooftops','LIRPVrooftops_10','low_avg_occu',
'Low_Commute_Times_10','low_commute_times_log10_orig','low_med_occu','low_own_devp_log10_orig','low_own_mwh_log10_orig',
'LowCommuteTimes','ManufacturingOccu','Masters','Mastersorabove','Median_Age_10','Median_HH_Income_10',
'median_household_income_log10_orig','Medianhomevalue_10','Mid_Agedwellings','Minors','mod_own_devp_log10_orig',
'mod_own_mwh_log10_orig','ModIgenerationMWh','ModIgenerationMWh_10','ModIRPVrooftops','ModIRPVrooftops_10',
'net_metering_hu_own_log10_orig','Net_MeteringxHomeOwners_10','NetMeteringxHomeOwners',
'Olderdwellings','p16_employed_log10_orig','pop_african_american_log10_orig','pop_asian_log10_orig',
'pop_caucasian_log10_orig','pop_hispanic_log10_orig','pop25_some_college_plus_log10_orig',
'PopertyTaxxHomeowners','PopertyTaxxHomeowners_10','Population_density_10','population_density_log10_orig',
'PopulationDensity','Prob_MD','ProfessionalSchool','ProfessionalSchool#','property_tax_hu_own_log10_orig',
'PublicOccu','PVper100Homeowners','Renters',
'ResidentalIncentivesxHomeOwners','ResidentalIncentivesxHomeOwners_10','ResidentialPanelareapercapita',
'Residentialsolarinstallationareabylandarea','Residentialsolarinstallationspercapita',
'RetailOccu','solar_prod_log10_orig',
'solargeneration','State',
'State_fips',
'Total_Area_10',
'total_area_log10_orig',
'total_own_devp_log10_orig',
'total_own_Sbldg_log10_orig',
'Total_RPV_bldgs_10',
'Total_RPV_roof_tops_10',
'TotalArea',
'Totalresidentialsolarinstallations',
'Totalresidentialsolarpanelarea',
'TotalRPVbldgs',
'TotalRPVrooftops',
'Town',
'transportation_bicycle_rate_log10_orig',
'transportation_home_rate_log10_orig',
'transportation_walk_rate_log10_orig',
'TransportationOccu',
'travel_time_10_19_rate_log10_orig',
'travel_time_40_89_rate_log10_orig',
'travel_time_less_than_10_rate_log10_orig',
'un_employ_rt',
'ColdSpot',
'HotSpot',
'High_Solar_Resources',
]

paper_vars_orig = ['']

PaperStateFixers = ['@ne','@or','al','ar','az','ca','co','ct','dc','de','fl','ga','ia','id',
                    'il','in','ks','ky','la','ma','md', 'me','mi','mn','mo','ms','mt','nc',
                    'nd','nh','nj','nm','nv','ny','oh','ok','pa','ri','sc','sd',
                    'tn','tx','ut','va','vt','wa','wi','wv','wy',]

Regional_Solar_Groups = {
    'Very Low':  ['wa', 'vt', 'me', 'nh', 'mi', 'pa', 'mn', 'nd', 'ma', 'wi'] ,
'Low':  ['oh', 'or', 'ct', 'wv', 'ri', 'ny', 'nj', 'in', 'il', 'md'] ,
'Moderate':  ['ia', 'mt', 'de', 'ky', 'sd', 'dc', 'va', 'tn', 'mo', 'ne'] ,
'High':  ['nc', 'ar', 'ks', 'id', 'ga', 'wy', 'al', 'sc', 'ms', 'la'] ,
'Very High':  ['ok', 'co', 'tx', 'ut', 'fl', 'ca', 'nv', 'nm', 'az'] ,
}
PaperSetFeatures = [
'AverageHouseholdSize','Commutingtime40minsdaily','Cooling_degree_days','Daily_solar_radiation','Electricity_price',
'Electricityconsumption','Employment','Gender_Ratio',
'Has_PropertyTax','Has_NetMetering',
'Homeowner','avg_monthly_consumption_kwh_log10_orig',
'MedianAge','MedianHHIncome','Medianhomevalue',
'Newdwellings','Rural','SomeCollege','Suburban',
'age_65_74_rate_log10_orig','age_median_log10_orig',
'age_minor_rate_log10_orig','AgricultureOccu','AnnualGeneration_MWh_10','AnnualGenerationMWh','ArtsOccu',
'Asian_pct','average_household_size_log10_orig','Average_HouseholdSize_10','Average_Income',
'average_monthly_bill_$_10','averagemonthlybill$','avg_monthly_bill_dlrs_log10_orig',
'Bachelors','Bachelors#','BachelorsorAbove','Black_pct', 'cdd_log10_orig',
'cdd_std_log10_orig','Commutingbybicycle','Commutingbybicycle_10','Commutingbywalking','Commutingbywalking_10','Commutingtime1019minsdaily',
'Commutingtime1019minsdaily_10','Commutingtime10minsdaily','Commutingtime10minsdaily_10','ConstructionOccu',
'@_Commuting_time_40minsdaily_10','@_Elderly_10','@_Minors_10','@_solar_generation_10','@#_Asian_10',
'@#_Black_10','@#_Hispanic_10','@#_White_10','@#25yearsofage','@#age252HSEdu','@#age25noHSEdu','@#Asian',
'@#Black','@#Hispanic','@#Homeowner','@#HSGrads','@#LessHS','@#MSs','@#PhDs',
'@#ResidentialIncentives','@#ResidentialIncentives_10','@#ResidentialStateInc',
'@#ResidentialStateInc_10','@#Some_college_or_More_10','@#SomecollegeorMore',
'@#White','@16unemployed','@25yearswithsomecollege', 'Cooling_degree_days_std',
'Coolingdegreeday_10',
'Coolingdegreedaystd_10','CoolingDesignTemp','Daily_solar_radiation_10','daily_solar_radiation_log10_orig',
'Diversityminority','Doctoral','edu_college','EducationOccu',
'Elderly','Electricity_consumption_10','Employed','Employed_10','Estimated_savings_$year',
'Estimated_savings_$year_10','familytypehomes','FinanceOccupation','Gender_female','Green_Travel_habits_10',
'Green_Travelers_log10_orig','GreenTravelhabits','hdd_log10_orig','hdd_std_log10_orig',
'Heating_degree_day_10','Heating_degree_day_std','Heating_degree_days',
'Heating_degree_days_10','heating_fuel_coal_coke_rate_log10_orig','heating_fuel_electricity_rate_log10_orig',
'Heating_Source_Coal__10','Heating_source_electricity__10','HeatingDesignTemp',
'HeatingSourceCoal','Heatingsourceelectricity','HIgenerationMWh','HIgenerationMWh_10','high_avg_occu',
'high_med_occu','high_own_devp_log10_orig','high_own_mwh_log10_orig','HighSchool','Highschoolorbelow',
'HIRPVrooftops','HIRPVrooftops_10','Hispanic_pct',

'hu_med_val_log10_orig','incent_cnt_res_own_log10_orig','incentive_count_residential_log10_orig',
'incentive_residential_state_level_log10_orig','InformationOccu','LessThanHighSchool','LIgenerationMWh',
'LIgenerationMWh_10','LIRPVrooftops','LIRPVrooftops_10','low_avg_occu',
'Low_Commute_Times_10','low_commute_times_log10_orig','low_med_occu','low_own_devp_log10_orig','low_own_mwh_log10_orig',
'LowCommuteTimes','ManufacturingOccu','Masters','Mastersorabove','Median_Age_10','Median_HH_Income_10',
'median_household_income_log10_orig','Medianhomevalue_10','Mid_Agedwellings','Minors','mod_own_devp_log10_orig',
'mod_own_mwh_log10_orig','ModIgenerationMWh','ModIgenerationMWh_10','ModIRPVrooftops','ModIRPVrooftops_10',
'net_metering_hu_own_log10_orig','Net_MeteringxHomeOwners_10','NetMeteringxHomeOwners',
'Olderdwellings','p16_employed_log10_orig','pop_african_american_log10_orig','pop_asian_log10_orig',
'pop_caucasian_log10_orig','pop_hispanic_log10_orig','pop25_some_college_plus_log10_orig',
'PopertyTaxxHomeowners','PopertyTaxxHomeowners_10','Population_density_10','population_density_log10_orig',
'PopulationDensity','Prob_MD','ProfessionalSchool','ProfessionalSchool#','property_tax_hu_own_log10_orig',
'PublicOccu','Renters',
'ResidentalIncentivesxHomeOwners','ResidentalIncentivesxHomeOwners_10',
'RetailOccu','solar_prod_log10_orig',
'solargeneration',
'Total_Area_10',
'total_area_log10_orig',
'total_own_devp_log10_orig',
'total_own_Sbldg_log10_orig',
'Total_RPV_bldgs_10',
'Total_RPV_roof_tops_10',
'TotalArea',
'TotalRPVbldgs',
'TotalRPVrooftops',
'Town',
'transportation_bicycle_rate_log10_orig',
'transportation_home_rate_log10_orig',
'transportation_walk_rate_log10_orig',
'TransportationOccu',
'travel_time_10_19_rate_log10_orig',
'travel_time_40_89_rate_log10_orig',
'travel_time_less_than_10_rate_log10_orig',
'un_employ_rt',
'High_Solar_Resources',
]
Paper_DS_filePath = r'C:\Users\gjone\__Doc_Dump\CSV_EXCEl\BigDeepSet_Min.csv'


income_dic = {
    'verylow': [0, .3],
    'low': [.3, .5],
    'mod': [.5, .8],
    'mid': [.8, 1.2],
    'high': [1.2, ],
}

locale_recode_d = {'locale_recode(rural)': 1, "locale_recode(suburban)": 3, 'locale_recode(town)': 2,
            'locale_recode(urban)': 4, }

suffix_to_locale = {
    'rural':'locale_recode(rural)',
    'town':'locale_recode(town)',
    'suburban':'locale_recode(suburban)',
    'urban':'locale_recode(urban)',
}

label_translation_dict = {
    "Green_Travelers_log10": "Green_Travel_habits_10",
    "Yr_own_mwh_log10": "Annual Generation_(MWh)_10",
    "Yrl_savings_$_log10": "Estimated_savings_$/year_10",
    "age_65_74_rate_log10": "%_Elderly_10",
    "age_median_log10": "Median_Age_10",
    "age_minor_rate_log10": "%_Minors_10",
    "average_household_size_log10": "Average_Household Size_10",
    "avg_monthly_bill_dlrs_log10": "average_monthly_bill_$_10",
    "avg_monthly_consumption_kwh_log10": "Electricity_consumption_10",
    "cdd_log10": "Cooling degree day_10",
    "cdd_std_log10": "Cooling degree day std_10",
    "daily_solar_radiation_log10": "Daily_solar_radiation_10",
    "hdd_log10": "Heating_degree_days_10",
    "hdd_std_log10": "Heating_degree_day_10",
    "heating_fuel_coal_coke_rate_log10": "Heating_Source_Coal_(%)_10",
    "heating_fuel_electricity_rate_log10": "Heating_source_electricity_(%)_10",
    "high_own_devp_log10": "HI RPV roof tops_10",
    "high_own_mwh_log10": "HI generation (MWh)_10",
    "hu_med_val_log10": "Median home value_10",
    "incent_cnt_res_own_log10": "Residental Incentives x Home Owners_10",
    "incentive_count_residential_log10": "# Residential Incentives_10",
    "incentive_residential_state_level_log10": "# Residential State Inc_10",
    "low_commute_times_log10": "Low_Commute_Times_10",
    "low_own_devp_log10": "LI RPV roof tops_10",
    "low_own_mwh_log10": "LI generation (MWh)_10",
    "median_household_income_log10": "Median_HH_Income_10",
    "mod_own_devp_log10": "ModI RPV roof tops_10",
    "mod_own_mwh_log10": "ModI generation (MWh)_10",
    "net_metering_hu_own_log10": "Net_Metering x Home Owners_10",
    "p16_employed_log10": "Employed_10",
    "pop25_some_college_plus_log10": "# Some_college_or_More_10",
    "pop_african_american_log10": "#_Black_10",
    "pop_asian_log10": "#_Asian_10",
    "pop_caucasian_log10": "#_White_10",
    "pop_hispanic_log10": "#_Hispanic_10",
    "population_density_log10": "Population_density_10",
    "property_tax_hu_own_log10": "Poperty Tax x Homeowners_10",
    "solar_prod_log10": "%_solar_generation_10",
    "total_area_log10": "Total_Area_10",
    "total_own_Sbldg_log10": "Total_RPV_bldgs_10",
    "total_own_devp_log10": "Total_RPV_roof_tops_10",
    "transportation_bicycle_rate_log10": "% Commuting by bicycle_10",
    "transportation_home_rate_log10": "% Working from home_10",
    "transportation_walk_rate_log10": "% Commuting by walking_10",
    "travel_time_10_19_rate_log10": "% Commuting time 10-19 mins daily_10",
    "travel_time_40_89_rate_log10": "% _Commuting_time_(% > 40 mins daily)_10",
    "travel_time_less_than_10_rate_log10": "% Commuting time < 10 mins daily_10",
    'hu_rent':'Renters',
    'Adoption_50': 'Above 50 per homeowners',
    'Adoption_25': 'Above 25 per homeowners',
    'PV_per_100_HuOwn': 'PV per 100 Homeowners',
    "avg_ibi_pct": 'Investment Based Incentives',
    "avg_cbi_usd_p_w": 'Capacity Based Incentives',
    "avg_pbi_usd_p_kwh": 'Production Based Incentives',
    'low_commute_times': 'Low Commute Times',
    'property_tax_hu_own': 'Poperty Tax x Homeowners',
    'white_pct': 'White_pct',
    'black_pct': 'Black_pct',
    'asian_pct': 'Asian_pct',
    'hispanic_pct': 'Hispanic_pct',
    'political_ratio': 'Political ratio',
    'total_sf_own_hh': 'Total single family homeowners',
    'total_mf_own_hh': 'Total multifamily homeowners',
    'Gender_Ratio': 'Gender_Ratio',
    'educated_population_rate': "% >= 25 years with some college +",
    'avg_months_tenancy': "Average Months Tenancy",
    'Yrl_%_$_kwh':'Estimated_%_monthly_bill',
    'Yrl_%_inc':'Estimated_%_income_Saved',
    'Yrl_savings_$': 'Estimated_savings_$/year',
    'total_own_elep':'Avg owner montlhy energy bill',
    "number_of_solar_system_per_household": "Solar_installations_per_household",
    'PV_HuOwn': "Solar_installations_per_home_owner",
    'SRpcap': "Residential solar installations per capita",
    'SRaPa': "Residential solar installation area by land area",
    'solar_system_count_residential': "Total residential solar installations ",
    'solar_system_count_nonresidential': "Total nonresidential solar installations",
    'total_panel_area_residential': "Total residential solar panel area",
    'ST_pcap': "Total solar installations per capita",
    'solar_panel_area_per_capita': "Total solar panel area per capita",
    'cdd': 'Cooling_degree_days',
    'cdd_std': 'Cooling_degree_days_std',
    'hdd_std': 'Heating_degree_day_std',
    'hdd': 'Heating_degree_days',
    'daily_solar_radiation': 'Daily_solar_radiation',
    'diversity': 'Diversity(% minority)',
    'population_density': 'PopulationDdensity',
    'heating_fuel_solar':'Heat Fuel Solar',
    'heating_fuel_none':'Heat Fuel None',
    'occupation_finance_rate':'Finance Occupation (%)',
    'med_income': 'Income',
    'fam_med_income': 'Family Income',
    'median_household_income': 'Median HH Income',
    'poverty_family_below_poverty_level_rate':'Families in Poverty (%)',
    'average_household_income':'Average_Income',
    'housing_unit_count': 'Housing unit count',
    'dlrs_kwh':'Electricity_price',
    'employ_rate':'Employment_(%)',
    'female_pct':'Gender_(% female)',
    'pop_female':'Gender_(female)',
    'voting_2012_dem_percentage':'Democrats (%)',
    'voting_2012_gop_percentage':'Republican (%)',
    'hu_own_pct':'Homeowner (%)',
    'hu_own':'# Homeowner',
    'education_college': 'edu_college',
    'age_55_or_more_rate': 'Age (%>55)',
    'hh_med_income':'household median income',
    'locale_recode': 'Area',
    'heating_fuel_electricity_rate': 'Heating source electricity (%)',
    'heating_fuel_coal_coke_rate': 'Heating Source Coal (%)',
    'hu_1959toearlier_pct': 'Older dwellings (%)',
    'hu_2000toafter': 'New dwellings',
    'hu_vintage_2010toafter': 'Homes built after 2010',
    'hu_vintage_1939toearlier': 'Homes built (<=1939)',
    'hu_vintage_2000to2009': 'Homes built (< 2000, <=2009)',
    'hu_vintage_1940to1959': 'Homes built (< 1940, <=1959)',
    'hu_vintage_1980to1999': 'Homes built (< 1980, <=1999)',
    'hu_vintage_1960to1970': 'Homes built (< 1960, <=1970)',
    'hu_mortgage': 'Homes with a mortgage',
    'hu_med_val':'Median home value',
    'Green_Travelers': 'Green Travel habits',
    'travel_time_40_89_rate': '% Commuting time (% > 40 mins daily)',
     'travel_time_40_59_rate':'% Commuting time 40-59 mins daily',
     'travel_time_10_19_rate':'% Commuting time 10-19 mins daily',
     'travel_time_20_29_rate':'% Commuting time 20-29 mins daily',
     'travel_time_60_89_rate':'% Commuting time 60-89 mins daily',
     'transportation_home_rate':'% Working from home',
     'travel_time_30_39_rate':'% Commuting time 30-39 mins daily',
     'travel_time_average':'Average Commute time (mins daily)',
     'travel_time_less_than_10_rate':'% Commuting time < 10 mins daily',
     'transportation_bicycle_rate':'% Commuting by bicycle',
     'transportation_car_alone_rate':'% Commuting by car',
     'transportation_carpool_rate':'% Commuting by carpool',
     'transportation_motorcycle_rate':'% Commuting by motorcycle',
     'transportation_public_rate':'% Commuting by public transportation',
     'transportation_walk_rate':'% Commuting by walking',
    'avg_monthly_consumption_kwh': 'Electricity consumption',
    'dlrs_kwh x median_household_income': 'Electricity price x income',
    'education_high_school_or_below_rate': 'High school or below (%)',
    'education_master_or_above_rate': 'Masters or above (%)',
    'education_professional_school_rate': 'Professional School (%)',
    'number_of_years_of_education':'Years of education',
    'education_professional_school':'Professional School (#)',
    'education_master_rate':'Masters (%)',
    'education_bachelor_rate':'Bachelors (%)',
    'education_high_school_graduate_rate':'High School (%)',
    'education_less_than_high_school_rate':'Less Than High School (%)',
    'education_population':'# >= 25 years of age ',
    'education_bachelor_or_above_rate':'Bachelors or Above (%)',
    'education_bachelor':'Bachelors #',
    'education_doctoral_rate': 'Doctoral (%)',
    'pop25_some_college_plus':'# Some college or More',
    '%hh_size_4': 'Household Size (% > 5 people)',
    'hh_total': 'Total Persons in Household',
    'household_count': 'Total number of households',
    'hu_2000toafter_pct': 'New dwellings (%)',
    'pop_under_18': 'Number of population below 18 yrs',
    'pop_total': 'Total Population',
    'Adoption': 'Adoption',
    'incent_cnt_res_own':'Residental Incentives x Home Owners',
    'net_metering_hu_own':'Net Metering x Home Owners',
    'incentive_count_nonresidential':'# Nonresidential Incentives',
    'incentive_count_residential':'# Residential Incentives',
    'incentive_nonresidential_state_level':'# Nonresidential State Inc',
    'incentive_residential_state_level':'# Residential State Inc',
    'net_metering':'Years of Net Metering',
    'property_tax_bin':'Years of Property Tax (Binary)',
    'Ren':'State Renewable Generation (%)',
    'education_doctoral':"# PhD's",
    'education_high_school_graduate':"# HS Grads",
    'education_less_than_high_school': "# Less HS",
    'education_master': "# MS's",
    'heating_fuel_coal_coke': "# Coal Heat",
    'heating_fuel_electricity': "# Electric Heat",
    'heating_fuel_fuel_oil_kerosene': "# Kerosene Heat",
    'heating_fuel_housing_unit_count': "# Housing Units",
    'heating_fuel_other': "# Other Heat Fuel",
    'population': "population",
    'poverty_family_below_poverty_level':"# families under poverty",
    'poverty_family_count':"# families in poverty",
    'education_college_rate': "% Some College",
    'heating_fuel_gas_rate': "% Gas Heating",
    'heating_fuel_fuel_oil_kerosene_rate': "% Kerosene Heat",
    'heating_fuel_solar_rate': "% Solar Heat",
    'heating_fuel_other_rate': "% Other Heating Fuel",
    'avg_electricity_retail_rate':'Retail Energy Cost',
    'age_median':'Median Age',
    'age_55_64_rate':'Age Range (55-64) %',
    'age_65_74_rate':'% Elderly',
    'age_75_84_rate':'Age Range (75-84) %',
    'age_25_34_rate':'Age Range (25-34) %',
    'age_more_than_85_rate':'Age Range (> 85) %',
    'locale_recode(town)':'Town',
    'locale_recode(suburban)':'Suburban',
    'locale_recode(rural)':'Rural',
    'locale_dummy':'Location Type',
    'total_area':'Total Area',
    'land_area':'Land Area',
    'male_pct':'Gender (Male%)',
    'pop_male':'Gender (Male)',
    'high_mf_own_bldg_cnt': 'HI$ multi-family Suitable',
    'low_mf_own_bldg_cnt': 'LI$ multi-family Suitable',
    'verylow_mf_own_bldg_cnt': 'VL$ multi-family Suitable',
    'mod_mf_own_bldg_cnt': 'Mo$ multi-family Suitable',
    'mid_mf_own_bldg_cnt': 'Mi$ multi-family Suitable',
    'high_own_devp': "HI RPV roof tops",
    'mid_own_devp':"MidI RPV roof tops",
    'mod_own_devp': "ModI RPV roof tops",
    'low_own_devp': 'LI RPV roof tops',
    'verylow_own_devp': 'VLI RPV roof tops',
    'total_own_devp': 'Total RPV roof tops',
    'heating_fuel_none_rate': "% No Heating Fuel",
    'average_household_size': "Average Household Size",
    'housing_unit_occupied_count': "# Occupied Homes",
    'heating_design_temperature': "Heating Design Temp",
    'cooling_design_temperature': "Cooling Design Temp",
    'age_18_24_rate': "% age 18-24",
    'age_35_44_rate': "% age 35-44",
    'age_45_54_rate': "% age 45-54",
    'age_10_14_rate': "% age 10-14",
    'age_15_17_rate': "% age 15-17",
    'age_5_9_rate': "% age 5-9",
    'household_type_family_rate': "% family type homes",
    'occupation_construction_rate': "% Construction Occu",
    'occupation_public_rate': "% Public Occu",
    'occupation_information_rate': "% Information Occu",
    'occupation_education_rate': "% Education Occu",
    'occupation_administrative_rate': "% Admin Occu",
    'occupation_manufacturing_rate': "% Manufacturing Occu",
    'occupation_wholesale_rate': "% Wholesale Occu",
    'occupation_retail_rate': "% Retail Occu",
    'occupation_transportation_rate': "% Transportation Occu",
    'occupation_arts_rate': "% Arts Occu",
    'occupation_agriculture_rate': "% Agriculture Occu",
    'occupancy_owner_rate': "% Owner Occupied",
    'mortgage_with_rate': "% Mortgage",
    'property_tax': "Years of Property Tax",
    'very_low_mf_own_hh': "VL Income multi-family households",
    'very_low_sf_own_hh': "VL Income single-family households",
    'low_mf_own_hh': "L Income multi-family households",
    'low_sf_own_hh': "L Income single-family households",
    'mod_mf_own_hh': "Mod Income multi-family owned households",
    'mod_sf_own_hh': "Mod Income single-family owned households",
    'mod_sf_rent_hh': "Mod Income single-family rented households",
    'mid_mf_own_hh': "Min Income multi-family households",
    'mid_sf_own_hh': "Mid Income single-family households",
    'high_mf_own_hh': "H Income multi-family households",
    'high_sf_own_hh': "H Income single-family households",
    'very_low_mf_own_bldg_cnt': "VL Income multi-family RPV bldgs",
    'very_low_sf_own_bldg_cnt': "VL Income single-family RPV bldgs",
    'low_sf_own_bldg_cnt': "L Income single-family RPV bldgs",
    'mod_sf_own_bldg_cnt': "Mod Income single-family RPV bldgs",
    'mod_sf_rent_bldg_cnt': "mod_sf_rent_bldg_cnt",
    'mid_sf_own_bldg_cnt': "mid_sf_own_bldg_cnt",
    'high_sf_own_bldg_cnt': "H Income single-family RPV bldgs",
    'very_low_mf_own_devp_cnt': "VL Income RPV roof tops",
    'very_low_sf_own_devp_cnt': "VL Income single-family RPV roof tops",
    'low_mf_own_devp_cnt': "L Income multi-family RPV roof tops",
    'low_sf_own_devp_cnt': 'L Income single-family RPV roof tops',
    'mod_mf_own_devp_cnt': "Mod Income multi-family RPV roof tops",
    'mod_sf_own_devp_cnt': "Mod Income single-family RPV roof tops",
    'mod_sf_rent_devp_cnt': 'mod_sf_rent_devp_cnt',
    'mid_mf_own_devp_cnt': 'mid_mf_own_devp_cnt',
    'mid_sf_own_devp_cnt': 'mid_sf_own_devp_cnt',
    'high_mf_own_devp_cnt': 'high_mf_own_devp_cnt',
    'high_sf_own_devp_cnt': "H Income single-family RPV roof tops",
    'very_low_mf_own_devp_m2': "VLI multi-family roof area (m^2)",
    'very_low_sf_own_devp_m2': 'VLI single-family roof area (m^2)',
    'low_mf_own_devp_m2': 'LI multi-family roof area (m^2)',
    'low_sf_own_devp_m2': 'VLI single-family roof area (m^2)',
    'mod_mf_own_devp_m2': 'ModI multi-family roof area (m^2)',
    'mod_sf_own_devp_m2': 'ModI single-family roof area (m^2)',
    'mod_sf_rent_devp_m2': 'mod_sf_rent_devp_m2',
    'mid_mf_own_devp_m2': 'MidI multi-family roof area (m^2)',
    'mid_sf_own_devp_m2': 'MidI multi-family roof area (m^2)',
    'high_mf_own_devp_m2': 'HI multi-family roof area (m^2)',
    'high_sf_own_devp_m2': 'HI single-family roof area (m^2)',
    'very_low_mf_own_mw': 'VLI multi-family capacity (mw)',
    'very_low_sf_own_mw': 'VLI single-family capacity (mw)',
    'low_mf_own_mw': 'LI multi-family capacity (mw)',
    'low_sf_own_mw': 'LI single-family capacity (mw)',
    'mod_mf_own_mw': 'ModI multi-family capacity (mw)',
    'mod_sf_own_mw': 'ModI single-family capacity (mw)',
    'mod_sf_rent_mw': 'mod_sf_rent_mw',
    'mid_mf_own_mw': 'mid_mf_own_mw',
    'mid_sf_own_mw': 'mid_sf_own_mw',
    'high_mf_own_mw': 'HI multi-family capacity (mw)',
    'high_sf_own_mw': 'HI single-family capacity (mw)',
    'very_low_mf_own_mwh': 'VLI multi-family generation (mwh)',
    'very_low_sf_own_mwh': 'VLI single-family generation (mwh)',
    'low_mf_own_mwh': 'LI multi-family generation (mwh)',
    'low_sf_own_mwh': 'LI single-family generation (mwh)',
    'mod_mf_own_mwh': 'ModI multi-family generation (mwh)',
    'mod_sf_own_mwh': 'ModI single-family generation (mwh)',
    'mod_sf_rent_mwh': 'mod_sf_rent_mwh',
    'mid_mf_own_mwh': 'MidI multi-family generation (mwh)',
    'mid_sf_own_mwh': 'MidI single-family generation (mwh)',
    'high_mf_own_mwh': 'HI multi-family generation (mwh)',
    'high_sf_own_mwh': 'HI single-family generation (mwh)',
    'very_low_mf_own_elep_hh': 'VL multifamily monthly elect $',
    'very_low_sf_own_elep_hh': 'VL single-family monthly elect $',
    'low_mf_own_elep_hh': 'LI multi-family monthly elect $',
    'low_sf_own_elep_hh': 'LI single-family elect $',
    'mod_mf_own_elep_hh': 'ModI multi-family monthly elect $',
    'mod_sf_own_elep_hh': 'ModI single-family monthly elect $',
    'mod_sf_rent_elep_hh': 'mod_sf_rent_elep_hh',
    'high_mf_own_elep_hh': 'HI multi-family monthly elect $',
    'high_sf_own_elep_hh': 'HI single-family monthly elect $',
    'avg_monthly_bill_dlrs': 'average monthly bill $',
    'hh_size_1': '# 1 person households',
    'hh_size_2': '# 2 personhouseholds',
    'hh_size_3': '# 3 person households',
    'hh_size_4': '# 4 person households',
    'hh_gini_index': 'Income Diversity',
    'pop_us_citizen': '# US citizens',
    'pop_nat_us_citizen': '# Nat US Citizens',
    'pop_hispanic': '# Hispanic',
    'pop_african_american': '# Black',
    'pop_asian':'# Asian',
    'pop_native_american': '# Native American',
    'pop_caucasian':'# White',
    'pop25_high_school': '# age 25 + 2/ HS Edu',
    'pop25_no_high_school': '# age 25 + no HS Edu',
    'pop_med_age': 'Median age',
    'p16_employed': 'Employed',
    'p16_unemployed': '16+ unemployed',
    'fam_children_under_6': '# Households kids < 6 yrs',
    'fam_children_6to17':'# Households kids 6-17 yrs',
    'pop_over_65': '# > 65 yrs',
    'hu_monthly_owner_costs_lessthan_1000dlrs': '# Homeowners costs > $1k',
    'hu_monthly_owner_costs_greaterthan_1000dlrs': '# Homeowners costs > $1k',
    'hu_no_mortgage': 'Homes w/ no mortgage',
    'hdd_ci': 'hdd_ci',
    'cdd_ci': 'cdd_ci',
    'climate_zone': 'climate zone',
    'Anti_Occup': 'Negative Adoption Occu',
    'Pro_Occup': 'Pro Adoption Occu',
    'locale_recode(urban)': 'Urban',
    'net_metering_bin': 'Has Net Metering',
    'renew_prod': '% renewable generation',
    'hydro_prod': '# hydro generation',
    'solar_prod': '% solar generation',
    '%hh_size_1': '% 1 person households',
    '%hh_size_2': '% 2 person households',
    '%hh_size_3': '% 3 person households',
    'hu_1959toearlier': 'Homes built 1959 or before',
    'hu_1960to1979_pct': 'Homes built from 1960-79',
    'hu_1980to1999_pct': 'Homes built from 1980-99',
    'med_inc_ebill_dlrs': 'Median Income * Monthly Energy Bill',
    'avg_inc_ebill_dlrs': 'Average Income * Monthly Energy Bill',
    'own_popden': 'homeowners * population density',
    'age_25_44_rate': '% 25-44 yrs old',
    'age_25_64_rate': '% 25-64 yrs old',
    'high_own_hh': 'HI owned homes',
    'mid_own_hh': 'MidI owned homes',
    'mod_own_hh': 'ModI owned homes',
    'low_own_hh': 'LI owned homes',
    'verylow_own_hh': 'VLI owned homes',
    'total_own_hh': 'total owned homes',
    'high_hh_rate': '% HI owned homes',
    'mid_hh_rate': '% MidI owned homes',
    'mod_hh_rate': '% ModI owned homes',
    'low_hh_rate': '% LI owned homes',
    'very_low_hh_rate': '% VL owned homes',
    'high_own_Sbldg': 'HI RPV bldgs',
    'mid_own_Sbldg': 'MidI RPV bldgs',
    'mod_own_Sbldg': 'ModI RPV bldgs',
    'low_own_Sbldg': 'LI RPV bldgs',
    'verylow_own_Sbldg': 'VLI RPB bldgs',
    'total_own_Sbldg': 'Total RPV bldgs',
    'high_own_Sbldg_rt': '% HI RPV buildings',
    'mid_own_Sbldg_rt': 'MI RPV bldgs',
    'mod_own_Sbldg_rt': 'ModI RPV bldgs',
    'low_own_Sbldg_rt': "LI RPV bldgs",
    'verylow_own_Sbldg_rt': 'VLI RPV bldgs',
    'Tot_own_mw': 'Total capacity (mw)',
    'Yr_own_mwh': 'Annual Generation (MWh)',
    'high_own_mwh': 'HI generation (MWh)',
    'mid_own_mwh': "MidI generation (MWh)",
    'mod_own_mwh': "ModI generation (MWh)",
    'low_own_mwh': "LI generation (MWh)",
    'verylow_own_mwh': "VLI generation (MWh)",
    'high_own_elep_hh': "HI monthly electrical $",
    'mod_own_elep_hh': "ModI monthly electrical $",
    'low_own_elep_hh': "LI monthly electrical $",
    'verylow_own_elep_hh': "VLI monthly electrical $",
    'age_minor_rate': '% Minors',
    'age_zoomer_rate':'% Zoomers',
    'state_fips': 'State_fips',
    'state': 'State',
    'county_fips': 'County',
    'total_panel_area_nonresidential': 'Nonresidential Panel Area',
    'SNRaPa':'Nonresidential Panel area by total area',
    'SRaPcap': 'Residential Panel area per capita',
    'AvgSres': 'Average Panel Area',
}

label_retranslation_dict = {label_translation_dict[ky]: ky for ky in label_translation_dict}

vars_to_log = [
        "Green_Travelers",
        "Yr_own_mwh",
        "Yrl_savings_$",
        "age_65_74_rate",
        "age_median",
        "age_minor_rate",
        "average_household_size",
        "avg_monthly_bill_dlrs",
        "avg_monthly_consumption_kwh",
        "cdd",
        "cdd_std",
        "daily_solar_radiation",
        "hdd",
        "hdd_std",
        "heating_fuel_coal_coke_rate",
        "heating_fuel_electricity_rate",
        "high_own_devp",
        "high_own_mwh",
        "hu_med_val",
        "incent_cnt_res_own",
        "incentive_count_residential",
        "incentive_residential_state_level",
        "low_commute_times",
        "low_own_devp",
        "low_own_mwh",
        "median_household_income",
        "mod_own_devp",
        "mod_own_mwh",
        "net_metering_hu_own",
        "p16_employed",
        "pop25_some_college_plus",
        "pop_african_american",
        "pop_asian",
        "pop_caucasian",
        "pop_hispanic",
        "population_density",
        "property_tax_hu_own",
        "solar_prod",
        "total_area",
        "total_own_Sbldg",
        "total_own_devp",
        "transportation_bicycle_rate",
        "transportation_home_rate",
        "transportation_walk_rate",
        "travel_time_10_19_rate",
        "travel_time_40_89_rate",
        "travel_time_less_than_10_rate",
    ]

vars_to_log = list(set(vars_to_log))

loged_vars = []
for f in vars_to_log:
    loged_vars.append(f + '_log10')

state_fixed = [
    "al",
    "ar",
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

# TODO: logistic Models
logistic_model_pop = [
    "pop_african_american_log10",
    "pop_asian_log10",
    "pop_caucasian_log10",
    "pop_hispanic_log10",
    #"diversity",

    "education_bachelor_or_above_rate",
    # 'pop25_some_college_plus',

    #"pop_female",
    'Gender_Ratio',

    'age_median',
    "age_65_74_rate", #"age_minor_rate",

    "average_household_size",
    #"hu_own_pct",
    'hu_own',
    'hu_med_val_log10',
    #'hu_rent',

    "median_household_income_log10",
    "p16_employed_log10",
    #"employ_rate",
    'locale_recode(rural)',
    #'locale_recode(town)',
    'locale_recode(suburban)',
    #'locale_recode(urban)',
    "population_density_log10",

    # policy
    'incentive_count_residential_log10',
    # physical suitability
    'daily_solar_radiation',"hu_1959toearlier_pct", "hu_2000toafter_pct",
    # economic factors
    'Yrl_savings_$_log10',
    #"avg_monthly_bill_dlrs",
    'heating_fuel_electricity_rate',

    'low_commute_times',
    'avg_monthly_consumption_kwh_log10',
]

logistic_model_pct = [
    "black_pct",
    "asian_pct",
    "white_pct",
    "hispanic_pct",
    "education_bachelor_or_above_rate",
    # 'pop25_some_college_plus',

    #"pop_female",
    'Gender_Ratio',

    'age_median',
    "age_65_74_rate", #"age_minor_rate",

    "average_household_size",
    #"hu_own_pct",
    'hu_own',
    'hu_med_val_log10',
    #'hu_rent',

    "median_household_income_log10",
    "p16_employed_log10",
    #"employ_rate",
    'locale_recode(rural)',
    #'locale_recode(town)',
    'locale_recode(suburban)',
    #'locale_recode(urban)',
    "population_density_log10",

    # policy
    'incentive_count_residential_log10',
    # physical suitability
    'daily_solar_radiation',"hu_1959toearlier_pct", "hu_2000toafter_pct",
    # economic factors
    'Yrl_savings_$_log10',
    #"avg_monthly_bill_dlrs",
    'heating_fuel_electricity_rate',
    'low_commute_times',
    'avg_monthly_consumption_kwh_log10',
]

logistic_model_cnt = [
    "pop_african_american",
    "pop_asian",
    "pop_caucasian",
    "pop_hispanic",
    #"diversity",

    "education_bachelor_or_above_rate",
    # 'pop25_some_college_plus',

    #"pop_female",
    'Gender_Ratio',

    'age_median',
    "age_65_74_rate", #"age_minor_rate",

    "average_household_size",
    #"hu_own_pct",
    'hu_own',
    'hu_med_val_log10',
    #'hu_rent',

    "median_household_income_log10",
    "p16_employed_log10",
    #"employ_rate",
    'locale_recode(rural)',
    #'locale_recode(town)',
    'locale_recode(suburban)',
    #'locale_recode(urban)',
    "population_density_log10",

    # policy
    'incentive_count_residential_log10',
    # physical suitability
    'daily_solar_radiation',"hu_1959toearlier_pct", "hu_2000toafter_pct",
    # economic factors
    'Yrl_savings_$_log10',
    #"avg_monthly_bill_dlrs",
    'heating_fuel_electricity_rate',

    'low_commute_times',
    'avg_monthly_consumption_kwh_log10',
]

logistic_model_div = [
    "diversity",

    "education_bachelor_or_above_rate",
    # 'pop25_some_college_plus',

    #"pop_female",
    'Gender_Ratio',

    'age_median',
    "age_65_74_rate", #"age_minor_rate",

    "average_household_size",
    #"hu_own_pct",
    'hu_own',
    'hu_med_val_log10',
    #'hu_rent',

    "median_household_income_log10",
    "p16_employed_log10",
    #"employ_rate",
    'locale_recode(rural)',
    #'locale_recode(town)',
    'locale_recode(suburban)',
    #'locale_recode(urban)',
    "population_density_log10",

    # policy
    'incentive_count_residential_log10',
    # physical suitability
    'daily_solar_radiation',"hu_1959toearlier_pct", "hu_2000toafter_pct",
    # economic factors
    'Yrl_savings_$_log10',
    #"avg_monthly_bill_dlrs",
    'heating_fuel_electricity_rate',

    'low_commute_times',
    'avg_monthly_consumption_kwh_log10',
]

# state abbreviations to state name dictionary
abrev_to_name_d= {
'mn': 'Minnesota',
'ny': 'New York',
'ca': 'California',
'wa': 'Washington',
'fl': 'Florida',
'in': 'Indiana',
'nd': 'North Dakota',
'la': 'Louisiana',
'mo': 'Missouri',
'wi': 'Wisconsin',
'ky': 'Kentucky',
'ia': 'Iowa',
'mi': 'Michigan',
'ar': 'Arkansas',
'co': 'Colorado',
'de': 'Delaware',
'dc': 'District of Columbia',
'tx': 'Texas',
'ma': 'Massachusetts',
'nm': 'New Mexico',
'il': 'Illinois',
'ga': 'Georgia',
'pa': 'Pennsylvania',
'nc': 'North Carolina',
'sd': 'South Dakota',
'al': 'Alabama',
'sc': 'South Carolina',
'nj': 'New Jersey',
'va': 'Virginia',
'mt': 'Montana',
'tn': 'Tennessee',
'oh': 'Ohio',
'ks': 'Kansas',
'or': 'Oregon',
'nh': 'New Hampshire',
'wv': 'West Virginia',
'id': 'Idaho',
'ok': 'Oklahoma',
'ct': 'Connecticut',
'md': 'Maryland',
'ms': 'Mississippi',
'az': 'Arizona',
'vt': 'Vermont',
'ne': 'Nebraska',
'wy': 'Wyoming',
'me': 'Maine',
'ut': 'Utah',
'nv': 'Nevada',
'ri': 'Rhode Island',
'ak': 'Alaska',
'hi':'Hawaii',
}

name_to_abrev_d= {
'Hawaii':'hi',
'Alaska':'ak',
'Minnesota': 'mn',
'New York': 'ny',
'California': 'ca',
'Washington': 'wa',
'Florida': 'fl',
'Indiana': 'in',
'North Dakota': 'nd',
'Louisiana': 'la',
'Missouri': 'mo',
'Wisconsin': 'wi',
'Kentucky': 'ky',
'Iowa': 'ia',
'Michigan': 'mi',
'Arkansas': 'ar',
'Colorado': 'co',
'Delaware': 'de',
'District of Columbia': 'dc',
'Texas': 'tx',
'Massachusetts': 'ma',
'New Mexico': 'nm',
'Illinois': 'il',
'Georgia': 'ga',
'Pennsylvania': 'pa',
'North Carolina': 'nc',
'South Dakota': 'sd',
'Alabama': 'al',
'South Carolina': 'sc',
'New Jersey': 'nj',
'Virginia': 'va',
'Montana': 'mt',
'Tennessee': 'tn',
'Ohio': 'oh',
'Kansas': 'ks',
'Oregon': 'or',
'New Hampshire': 'nh',
'West Virginia': 'wv',
'Idaho': 'id',
'Oklahoma': 'ok',
'Connecticut': 'ct',
'Maryland': 'md',
'Mississippi': 'ms',
'Arizona': 'az',
'Vermont': 'vt',
'Nebraska': 'ne',
'Wyoming': 'wy',
'Maine': 'me',
'Utah': 'ut',
'Nevada': 'nv',
'Rhode Island': 'ri',
}

state_id_d = {
'ak': ['Alaska', '02'],
'al': ['Alabama', '01'],
'ar': ['Arkansas', '05'],
'az': ['Arizona', '04'],
'ca': ['California', '06'],
'co': ['Colorado', '08'],
'ct': ['Connecticut', '09'],
'dc': ['District of Columbia', '11'],
'de': ['Delaware', '10'],
'fl': ['Florida', '12'],
'ga': ['Georgia', '13'],
'hi': ['Hawaii', '15'],
'ia': ['Iowa', '19'],
'id': ['Idaho', '16'],
'il': ['Illinois', '17'],
'in': ['Indiana', '18'],
'ks': ['Kansas', '20'],
'ky': ['Kentucky', '21'],
'la': ['Louisiana', '22'],
'ma': ['Massachusetts', '25'],
'md': ['Maryland', '24'],
'me': ['Maine', '23'],
'mi': ['Michigan', '26'],
'mn': ['Minnesota', '27'],
'mo': ['Missouri', '29'],
'ms': ['Mississippi', '28'],
'mt': ['Montana', '30'],
'nc': ['North Carolina', '37'],
'nd': ['North Dakota', '38'],
'ne': ['Nebraska', '31'],
'nh': ['New Hampshire', '33'],
'nj': ['New Jersey', '34'],
'nm': ['New Mexico', '35'],
'nv': ['Nevada', '32'],
'ny': ['New York', '36'],
'oh': ['Ohio', '39'],
'ok': ['Oklahoma', '40'],
'or': ['Oregon', '41'],
'pa': ['Pennsylvania', '42'],
'ri': ['Rhode Island', '44'],
'sc': ['South Carolina', '45'],
'sd': ['South Dakota', '46'],
'tn': ['Tennessee', '47'],
'tx': ['Texas', '48'],
'ut': ['Utah', '49'],
'va': ['Virginia', '51'],
'vt': ['Vermont', '50'],
'wa': ['Washington', '53'],
'wi': ['Wisconsin', '55'],
'wv': ['West Virginia', '54'],
'wy': ['Wyoming', '56'],
}

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

for st in ['ca', 'nv', 'az']:
    del Nt3state_abrev[Nt3state_abrev.index(st)]


solar_capacity_factors = [
"high_mf_own_bldg_cnt",
"high_mf_own_devp_cnt",
"high_mf_own_devp_m2",
"high_own_Sbldg",
"high_own_Sbldg_rt",
"high_sf_own_bldg_cnt",
"high_sf_own_devp_cnt",
"high_sf_own_devp_m2",
"high_sf_own_mw",
"high_sf_own_mwh",
"high_mf_own_mw",
"high_mf_own_mwh",
"high_own_mwh",
"low_mf_own_bldg_cnt",
"low_mf_own_devp_cnt",
"low_mf_own_devp_m2",
"low_sf_own_bldg_cnt",
"low_sf_own_devp_cnt",
"low_sf_own_devp_m2",
"low_mf_own_mw",
"low_mf_own_mwh",
"low_sf_own_mw",
"low_sf_own_mwh",
"low_own_Sbldg",
"low_own_Sbldg_rt",
"low_own_mwh",
"mid_mf_own_bldg_cnt",
"mid_mf_own_devp_cnt",
"mid_mf_own_devp_m2",
"mid_mf_own_mw",
"mid_mf_own_mwh",
"mid_own_Sbldg",
"mid_own_Sbldg_rt",
"mid_own_mwh",
"mid_sf_own_bldg_cnt",
"mid_sf_own_devp_cnt",
"mid_sf_own_devp_m2",
"mid_sf_own_mw",
"mid_sf_own_mwh",
"mod_mf_own_bldg_cnt",
"mod_mf_own_devp_cnt",
"mod_mf_own_devp_m2",
"mod_mf_own_mw",
"mod_mf_own_mwh",
"mod_own_Sbldg",
"mod_own_Sbldg_rt",
"mod_own_mwh",
"mod_sf_own_bldg_cnt",
"mod_sf_own_devp_cnt",
"mod_sf_own_devp_m2",
"mod_sf_own_mw",
"mod_sf_own_mwh",
"total_own_Sbldg",
"very_low_mf_own_bldg_cnt",
"very_low_mf_own_devp_cnt",
"very_low_mf_own_devp_m2",
"very_low_mf_own_mw",
"very_low_mf_own_mwh",
"very_low_sf_own_bldg_cnt",
"very_low_sf_own_devp_cnt",
"very_low_sf_own_devp_m2",
"very_low_sf_own_mw",
"very_low_sf_own_mwh",
"verylow_own_Sbldg",
"verylow_own_Sbldg_rt",
"verylow_own_mwh",
]

#list of states that can be ignored right now for analysis
good_rn_states = ['ar', 'ky', 'ny', 'mt', 'mn', 'me', ]


# ###########################################################
# #####################   TODO: directories
# ###########################################################

# used to current working main data source file
#__Current_Main = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\US_set_all_OMEGA_5.csv'
__Current_Main = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\US_set_all_OMEGA_6.csv'
__Current_Main2 = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\US_set_all_OMEGA_61.csv'
__Current_Main3 = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\US_set_all_OMEGA_61.csv'

scld_set = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\US_set_all_OMEGA_6_scld_MM.csv'

logged_set = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\US_set_all_OMEGA_6_scld_TransFN_MM_alllogadjb.csv'


# lazy version of the current working main data source file
cmainDS = __Current_Main
cmainDS2 = __Current_Main2

__DATA_dir = '_Data/'
__DS_dir = '_Data/DeepSolar/'
__MIXED_dir = '_Data/Mixed/'
__NREL_dir = '_Data/NREL/'
__DS_ORIG = '_Data/DeepSolar/deepsolar_tract_orig_Adoption.csv'
__NREL_ORIG = '_Data/NREL/NREL_seeds.csv'
TVA_dir = r'C:\Users\gjone\DeepSolar_Convergence\_Data\Mixed\TVA_Region\\'
set_13 = r'C:\Users\gjone\DeepSolar_Convergence\_Data\Mixed\13_sets\\'
set_7 = r'C:\Users\gjone\DeepSolar_Convergence\_Data\Mixed\7_sets\\'

# path to store permutation and feature importance plots
RF_vs_PI_path = r'C:\Users\gjone\DeepLearningDeepSolar\__Media\images\RF_vs_PI_BG_plots'

# path to store block group power plots
Block_Group_path = r'C:\Users\gjone\DeepLearningDeepSolar\__Media\_BlockGroups'

# path to feature and permutation importance ranking csv's and excel records
RF_vs_PI_csv_path = r'C:\Users\gjone\DeepLearningDeepSolar\__Media\documents\RF_vs_PI_BG_csv'
HeatMap_Tables_path = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\___HeatMapTables\\'
hot_spot_analysis_path = r'C:\Users\gjone\DeepLearningDeepSolar\Analysis_Scripts\HotSpot_Analysis_Results\\'
filenameHS = hot_spot_analysis_path + 'Hot_Spot_%_diff_Stats_{}_{}.xlsx'
filenamefips = hot_spot_analysis_path + '{}_{}_hot_fips.xlsx'

block_group_file_path = r'C:\Users\gjone\DeepLearningDeepSolar\Analysis_Scripts\HotSpot_Analysis_Results\Block_group_labels.xlsx'


class Data_set_paths:
    US_Model = cmainDS
    US_Convergent_Base = cmainDS2
    US_Convergent_Set_OmegaA = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\US_set_all_OMEGA2.csv'
    tva_set_nrml = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\TVA_Region\TVA_DS_NREL_13set_nrml_.csv'
    seven_set_nrml = r'C:\Users\gjone\DeepSolar_Convergence\_Data\Mixed\7_sets\SevenSt_DS_NREL_13set_nrml_.csv'
    thirteen_set_nrml = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\13_sets\ThirteenSt_DS_NREL_set_nrml_.csv'

    tva_set = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\TVA_Region\TVA_DS_NREL_set.csv'
    seven_set = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\7_sets\SevenSt_DS_NREL_set.csv'
    thirteen_set = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\13_sets\ThirteenSt_DS_NREL_set.csv'
    US_set = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\US_set_.csv'
    paperset = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\US_set_all_OMEGA_2021_Base.csv'
    paperset2 = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\US_set_all_OMEGA_1_20_21_Base.csv'
    paperset3 = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\US_set_all_OMEGA_1_20_21_Top3.csv'
    paperset4 = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\US_set_all_OMEGA_1_20_21_NTop3.csv'
    paperset5 = r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\__CNRGNT_1_24_21_RFimpt.csv'
    DS_NREL_SVI_paths = [
        r'../_Data/DeepSolar/deepsolar_tract_orig_Adoption.csv',
        r'../_Data/NREL/NREL_seeds_all.csv',
    ]

# ##################################################################
# ############   TODO: Models, variables and variable groups  ###
# ##################################################################

above_avg_adopting_states = [
"az",
"ca",
"co",
"ct",
"dc",
"de",
"fl",
"id",
"la",
"ma",
"md",
"nj",
"nm",
"nv",
"or",
"ri",
"tx",
"ut",
"wa",
"wy",
]
avg_or_below_states = [
"al",
"ar",
"ga",
"ia",
"il",
"in",
"ks",
"ky",
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
"ny",
"oh",
"ok",
"pa",
"sc",
"sd",
"tn",
"va",
"vt",
"wi",
"wv",
]

us_high_top = ['high_mf_own_elep_hh', 'hh_size_1', '%hh_size_1', 'very_low_mf_rent_elep_hh', 'hu_rent',
          'hu_vintage_2010toafter', 'mod_mf_rent_elep_hh', 'heating_fuel_gas_rate', 'age_25_44_rate',
          'hu_vintage_1939toearlier', 'hu_1959toearlier_pct', 'education_high_school_graduate_rate',
          'hu_1980to1999_pct', 'age_25_64_rate', 'transportation_walk_rate', 'pop_african_american',
          'low_mf_rent_elep_hh', 'hu_vintage_1940to1959', 'low_mf_own_elep_hh', 'fam_children_under_6',
          'age_25_34_rate', 'avg_monthly_bill_dlrs', 'pop25_some_college_plus', 'hydro_prod', 'age_75_84_rate',
          'hh_gini_index', 'age_55_64_rate', 'transportation_car_alone_rate', 'Anti_Occup', 'Green_Travelers',
          'area_km2', 'education_college_rate', 'pop_non_us_citizen', 'hu_2000toafter_pct',
          'occupation_information_rate', 'mortgage_with_rate', 'avg_monthly_consumption_kwh', 'age_65_74_rate',
          'pop_asian', 'age_18_24_rate', 'transportation_home_rate', 'hdd', 'occupation_education_rate',
          'hu_monthly_owner_costs_greaterthan_1000dlrs', 'age_5_9_rate', 'age_35_44_rate', 'age_more_than_85_rate',
          'hu_1960to1979_pct', '%hh_size_2', 'age_10_14_rate', 'occupation_construction_rate', 'age_45_54_rate',
          'Pro_Occup', 'transportation_carpool_rate', 'education_master', 'age_15_17_rate', 'diversity',
          'occupation_finance_rate', 'high_mf_own_mwh', 'hu_med_val', 'occupation_manufacturing_rate', 'employ_rate',
          'housing_unit_median_value', 'occupation_retail_rate', 'occupation_transportation_rate',
          'occupation_administrative_rate', '%hh_size_3', 'occupation_wholesale_rate', 'occupation_arts_rate',
          'occupation_public_rate', 'travel_time_40_59_rate', 'occupation_agriculture_rate', 'travel_time_30_39_rate',
          'travel_time_average', 'cdd_ci', 'travel_time_60_89_rate', 'travel_time_20_29_rate', 'cdd_std', 'pop_hispanic',
          'housing_unit_median_gross_rent', 'travel_time_40_89_rate', 'cdd', 'total_area', 'heating_fuel_coal_coke_rate',
          'travel_time_10_19_rate', 'land_area', 'high_own_mwh', 'incent_cnt_res_own', 'education_bachelor',
          'travel_time_less_than_10_rate', 'pop_caucasian', 'heating_fuel_gas', 'very_low_mf_own_elep_hh',
          'high_mf_rent_bldg_cnt', 'own_popden', 'high_mf_own_bldg_cnt', 'daily_solar_radiation',
         'population_density', 'pop_nat_us_citizen']

us_high_1 =  ['pop_nat_us_citizen', 'high_mf_own_bldg_cnt', 'housing_unit_median_gross_rent', 'own_popden',
              'incentive_nonresidential_state_level', 'education_high_school_graduate_rate', 'mortgage_with_rate',
              'pop_hispanic', 'heating_fuel_coal_coke_rate', 'Anti_Occup', 'land_area', 'occupation_administrative_rate',
              'avg_monthly_bill_dlrs', 'occupation_agriculture_rate', 'cdd_std', 'heating_fuel_gas', 'hu_rent',
              'pop_caucasian', 'occupation_finance_rate', 'very_low_mf_own_elep_hh', 'occupation_information_rate',
              'fam_children_under_6', 'hu_vintage_1939toearlier', 'hydro_prod', 'age_25_44_rate',
              'transportation_home_rate', 'diversity', 'hdd', 'occupation_arts_rate', 'occupation_construction_rate',
              'occupation_transportation_rate', '%hh_size_1', 'hh_gini_index', 'age_75_84_rate', 'age_55_64_rate',
              '%hh_size_2', 'employ_rate', 'occupation_wholesale_rate',]

us_high_2 =  ['high_mf_own_bldg_cnt', 'housing_unit_median_value',  'high_own_hh',
              'incentive_residential_state_level', 'education_high_school_graduate_rate', 'mortgage_with_rate',
              'pop_hispanic', 'heating_fuel_coal_coke_rate', 'Anti_Occup', 'total_area',
              'avg_monthly_bill_dlrs', 'occupation_agriculture_rate', 'cdd_std', 'heating_fuel_gas', 'hu_rent',
              'pop_caucasian', 'occupation_finance_rate', 'very_low_mf_own_elep_hh',
              'hu_vintage_1939toearlier', 'hydro_prod', "avg_monthly_consumption_kwh","high_own_elep_hh",
              'transportation_home_rate', 'diversity', 'hdd', 'occupation_arts_rate', 'occupation_construction_rate',
              '%hh_size_1', 'hh_gini_index', 'age_75_84_rate', 'age_55_64_rate',
              '%hh_size_2', 'employ_rate', 'population_density', 'daily_solar_radiation',]

us_high_3 =  ['high_mf_own_bldg_cnt', 'housing_unit_median_value',  'high_own_hh',
              'incentive_residential_state_level', 'education_high_school_graduate_rate', 'mortgage_with_rate',
              'heating_fuel_coal_coke_rate', "high_own_mwh",  "very_low_own_mwh", "mod_own_mwh",
              'avg_monthly_bill_dlrs', 'occupation_agriculture_rate', 'cdd_std', 'heating_fuel_gas', 'hu_rent',
              'pop_caucasian', 'occupation_finance_rate', 'very_low_mf_own_elep_hh',
              'hu_vintage_1939toearlier', 'hydro_prod', "avg_monthly_consumption_kwh","high_own_elep_hh",
              'transportation_home_rate', 'diversity', 'hdd', 'occupation_arts_rate', 'occupation_construction_rate',
              '%hh_size_3', 'age_75_84_rate', 'age_55_64_rate',
              '%hh_size_2', 'employ_rate', 'population_density', 'daily_solar_radiation',]

us_high_4 =  ['high_mf_own_bldg_cnt', 'housing_unit_median_value',"travel_time_average",
              'incentive_residential_state_level', 'education_high_school_graduate_rate', 'mortgage_with_rate',
              'heating_fuel_coal_coke_rate', "high_own_mwh",    "low_own_mwh", "travel_time_60_89_rate",
              'avg_monthly_bill_dlrs', 'occupation_agriculture_rate', 'cdd_std', 'heating_fuel_gas', 'hu_rent',
              'pop_caucasian', 'occupation_finance_rate', 'verylow_own_elep_hh','low_own_elep_hh', "mid_own_mwh",
              'hu_vintage_1939toearlier', 'hydro_prod',  "mod_own_elep_hh", "mid_own_elep_hh",
              'diversity', 'hdd', 'occupation_arts_rate',
              '%hh_size_3', 'age_75_84_rate', 'age_55_64_rate', 'education_bachelor',
              '%hh_size_2', 'employ_rate', 'population_density', 'daily_solar_radiation',]
us_high_5 =  ['high_mf_own_bldg_cnt', 'housing_unit_median_value',"travel_time_average",
              'incentive_residential_state_level',  'mortgage_with_rate',
              'heating_fuel_coal_coke_rate', "high_own_mwh",    "low_own_mwh", "travel_time_60_89_rate",
              'avg_monthly_bill_dlrs', 'occupation_agriculture_rate', 'cdd_std', 'heating_fuel_gas', 'hu_rent',
              'pop_caucasian', 'occupation_finance_rate', 'verylow_own_elep_hh', "mid_own_mwh",
              'hu_vintage_1939toearlier', 'hydro_prod',   "mid_own_elep_hh",
              'diversity', 'occupation_arts_rate', "dlrs_kwh", "avg_monthly_consumption_kwh",
              'age_75_84_rate', 'age_55_64_rate', 'education_bachelor',
              '%hh_size_2', 'employ_rate', 'population_density', 'daily_solar_radiation',]

us_low_HighV =  ['high_mf_own_bldg_cnt', 'housing_unit_median_value',"travel_time_average",
              'incentive_residential_state_level',  'mortgage_with_rate',
              'heating_fuel_coal_coke_rate', "high_own_mwh",    "low_own_mwh", "travel_time_60_89_rate",
              'avg_monthly_bill_dlrs', 'occupation_agriculture_rate', 'cdd_std', 'heating_fuel_gas', 'hu_rent',
              'pop_caucasian', 'occupation_finance_rate', 'verylow_own_elep_hh', "mid_own_mwh",
              'hu_vintage_1939toearlier', 'hydro_prod',   "mid_own_elep_hh",
              'diversity', 'occupation_arts_rate', "dlrs_kwh", "avg_monthly_consumption_kwh"
              'age_75_84_rate', 'age_55_64_rate', 'education_bachelor',
              '%hh_size_2', 'employ_rate', 'population_density', 'daily_solar_radiation',]

us_low =  ['high_mf_own_bldg_cnt', 'housing_unit_median_value',"travel_time_average",
              'incentive_residential_state_level',  'mortgage_with_rate',
              'heating_fuel_coal_coke_rate', "high_own_mwh",    "low_own_mwh", "travel_time_60_89_rate",
              'avg_monthly_bill_dlrs', 'occupation_agriculture_rate', 'cdd_std', 'heating_fuel_gas', 'hu_rent',
              'pop_caucasian', 'occupation_finance_rate', 'verylow_own_elep_hh', "mid_own_mwh",
              'hu_vintage_1939toearlier', 'hydro_prod',   "mid_own_elep_hh",
              'diversity', 'occupation_arts_rate', "dlrs_kwh", "avg_monthly_consumption_kwh"
              'age_75_84_rate', 'age_55_64_rate', 'education_bachelor',
              '%hh_size_2', 'employ_rate', 'population_density', 'daily_solar_radiation',]

us_low_1 = ['high_mf_own_devp_cnt', 'pop25_some_college_plus',
                   'mid_own_Sbldg', 'pop_asian', 'mid_mf_rent_mw', 'high_mf_rent_hh',
                   'education_professional_school', 'incent_cnt_res_own', 'education_master_or_above_rate',
                   'p16_unemployed', 'Tot_own_mw', 'heating_fuel_gas', 'median_household_income', 'hu_med_val',
                   'pop_non_us_citizen', 'Anti_Occup', 'occupation_agriculture_rate', 'heating_fuel_coal_coke_rate',
                   'hh_size_1', 'own_popden', 'hu_vintage_1940to1959', 'locale_dummy', 'diversity',
                   'travel_time_10_19_rate', 'heating_fuel_electricity', 'occupation_construction_rate',
                   'incentive_residential_state_level', 'hu_rent', 'hu_1960to1979_pct', 'employ_rate', 'solar_prod',
                   'travel_time_30_39_rate', 'area_km2', 'travel_time_40_59_rate', 'hdd_std', '%hh_size_2',
                   'age_more_than_85_rate', 'travel_time_less_than_10_rate', 'transportation_car_alone_rate',
                   'heating_fuel_gas_rate', 'travel_time_60_89_rate', 'heating_fuel_fuel_oil_kerosene',]
US_low_A = ['pop_nat_us_citizen', 'pop25_some_college_plus', 'locale_recode(rural)', 'incent_cnt_res_own',
            'high_mf_own_devp_cnt', 'high_mf_own_hh', 'hu_med_val', 'pop_hispanic',
            'own_popden',  'heating_fuel_coal_coke_rate',
            'education_master_or_above_rate', 'solar_prod',
            'incentive_residential_state_level', 'daily_solar_radiation', 'occupation_agriculture_rate',
            'mid_sf_own_mwh', 'total_area',  'hh_size_1',  'employ_rate',
            'cdd_std', 'hydro_prod', 'travel_time_60_89_rate', 'verylow_own_elep_hh', 'hu_vintage_1940to1959',
            'pop_african_american',  'travel_time_20_29_rate', "heating_fuel_gas_rate",
            'travel_time_30_39_rate',  ]

regional_models = {
    'US':['pop25_some_college_plus',  'incent_cnt_res_own', 'high_mf_own_devp_cnt',
          'high_mf_own_hh', 'hu_med_val', 'pop_hispanic', 'mortgage_with_rate', 'Anti_Occup', 'locale_dummy',
          'heating_fuel_coal_coke_rate', 'solar_prod', 'population_density', "travel_time_average", 'diversity',
          'daily_solar_radiation', 'total_area', 'hdd',   'avg_electricity_retail_rate',
          'travel_time_less_than_10_rate',   'number_of_years_of_education'],

    'US1': ['pop25_some_college_plus', 'pop_asian', 'incent_cnt_res_own', 'high_mf_own_devp_cnt',
          'high_mf_own_hh', 'hu_med_val', 'pop_hispanic', 'mortgage_with_rate', 'Anti_Occup',
          'heating_fuel_coal_coke_rate', 'solar_prod', 'locale_dummy', 'population_density', "travel_time_average",
          'daily_solar_radiation', 'total_area', 'hdd',  "travel_time_40_89_rate", 'avg_electricity_retail_rate',
          'travel_time_less_than_10_rate', 'renew_prod', 'high_own_mwh', 'number_of_years_of_education'],

    'US2': ['pop25_some_college_plus', 'pop_asian', 'incent_cnt_res_own', 'high_mf_own_devp_cnt', 'cdd_std',
          'high_mf_own_hh', 'hu_med_val', 'pop_hispanic', 'mortgage_with_rate', "occupation_agriculture_rate",
          'heating_fuel_coal_coke_rate', 'solar_prod',  'population_density', "travel_time_average",
          'daily_solar_radiation', 'climate_zone',  'diversity', 'avg_electricity_retail_rate', 'locale_recode(rural)',
          "avg_monthly_consumption_kwh", "dlrs_kwh", 'low_own_mwh',   'high_own_hh', 'net_metering',
          'travel_time_less_than_10_rate',  'high_own_mwh', 'number_of_years_of_education',
          'high_own_elep_hh' ],

    'US3': ['pop25_some_college_plus', 'pop_asian', 'incent_cnt_res_own', 'high_mf_own_devp_cnt', 'cdd_std',
          'high_mf_own_hh', 'hu_med_val', 'pop_hispanic', 'mortgage_with_rate', "occupation_agriculture_rate",
          'heating_fuel_coal_coke_rate',   'population_density', "travel_time_average",
          'daily_solar_radiation', 'climate_zone',    'locale_recode(rural)',
          "dlrs_kwh",  'low_own_mwh',    'net_metering',  'heating_fuel_gas',
          'travel_time_less_than_10_rate',  'high_own_mwh', 'number_of_years_of_education',
          ],
    'US my':["high_own_devp", "cdd_std", 'population_density', "Tot_own_mw", "median_household_income","avg_monthly_consumption_kwh"],
    'US NO WEST':[],
    'very low':[],
    'low':[],
    'US_low_top': ['high_mf_own_devp_cnt', 'pop25_some_college_plus', 'high_mf_rent_bldg_cnt',
                   'mid_mf_own_bldg_cnt', 'pop_asian', 'mid_mf_rent_mw', 'high_mf_rent_hh',
                   'education_professional_school', 'incent_cnt_res_own', 'education_master_or_above_rate',
                   'p16_unemployed', 'Tot_own_mw', 'heating_fuel_gas', 'median_household_income', 'hu_med_val',
                   'pop_non_us_citizen', 'Anti_Occup', 'occupation_agriculture_rate', 'heating_fuel_coal_coke_rate',
                   'hh_size_1', 'own_popden', 'hu_vintage_1940to1959', 'locale_dummy', 'diversity',
                   'travel_time_10_19_rate', 'heating_fuel_electricity', 'occupation_construction_rate',
                   'incentive_residential_state_level', 'hu_rent', 'hu_1960to1979_pct', 'employ_rate', 'solar_prod',
                   'travel_time_30_39_rate', 'area_km2', 'travel_time_40_59_rate', 'hdd_std', '%hh_size_2',
                   'age_more_than_85_rate', 'travel_time_less_than_10_rate', 'transportation_car_alone_rate',
                   'heating_fuel_gas_rate', 'travel_time_60_89_rate', 'cdd_ci', 'heating_fuel_fuel_oil_kerosene',],
'US_low': ['high_mf_own_devp_cnt', 'pop25_some_college_plus',
                   'mid_own_Sbldg', 'pop_asian', 'mid_mf_rent_mw', 'high_mf_rent_hh',
                   'education_professional_school', 'incent_cnt_res_own', 'education_master_or_above_rate',
                   'p16_unemployed', 'Tot_own_mw', 'heating_fuel_gas', 'median_household_income', 'hu_med_val',
                   'pop_non_us_citizen', 'Anti_Occup', 'occupation_agriculture_rate', 'heating_fuel_coal_coke_rate',
                   'hh_size_1', 'own_popden', 'hu_vintage_1940to1959', 'locale_dummy', 'diversity',
                   'travel_time_10_19_rate', 'heating_fuel_electricity', 'occupation_construction_rate',
                   'incentive_residential_state_level', 'hu_rent', 'hu_1960to1979_pct', 'employ_rate', 'solar_prod',
                   'travel_time_30_39_rate', 'area_km2', 'travel_time_40_59_rate', 'hdd_std', '%hh_size_2',
                   'age_more_than_85_rate', 'travel_time_less_than_10_rate', 'transportation_car_alone_rate',
                   'heating_fuel_gas_rate', 'travel_time_60_89_rate', 'cdd_ci', 'heating_fuel_fuel_oil_kerosene',],
    'medium': ['pop25_some_college_plus', 'high_mf_own_bldg_cnt', 'education_bachelor_rate', 'pop_asian',
               'heating_fuel_coal_coke_rate', 'education_professional_school', 'population_density',
               'mortgage_with_rate', 'hh_size_1', 'total_area', 'pop_hispanic', 'travel_time_average',
               'very_low_mf_own_elep_hh', 'heating_fuel_gas_rate', 'employ_rate', 'age_65_74_rate',
               'transportation_carpool_rate', 'occupation_construction_rate', 'occupation_transportation_rate',
               'age_25_34_rate', 'occupation_arts_rate', 'high_mf_own_elep_hh', 'travel_time_30_39_rate',
               '%hh_size_2', 'heating_design_temperature', 'age_18_24_rate', "mid_mf_own_mwh",
               'travel_time_20_29_rate', ],
    'high':[],
    'N. West':[],
    'West':[],
    'S. West':[],
    'M. West':[],
    'S. East':[],
    'Mid_atlantic':[],
    'N. East':[],
}

solar_metrics = [   'Adoption',
                    'solar_system_count',
                    'solar_panel_area_divided_by_area',
                    'solar_panel_area_per_capita',
                    'solar_system_count_residential',
                    'solar_system_count_nonresidential',
                    'total_panel_area_residential',
                    'total_panel_area_nonresidential',
                    'total_panel_area',
                    'number_of_solar_system_per_household',
                    'SRpcap',
                    'SNRpcap',
                    'ST_pcap',
                    'SRaPa',
                    'SNRaPa',
                    'SRaPcap',
                    'SNRaPcap',
                ]
solar_metricsB = solar_metrics + ['PV_HuOwn'] + ['AvgSres'] + ['PV_per_100_HuOwn'] + ['Adoption_50'] + ['Adoption_25']
solar_metricsC = solar_metrics + ['PV_HuOwn']

residential_solar_metrics = [
    'Adoption',
    'Adoption_27hh',
    'Adoption_30hh',
    'Adoption_35hh',
    'Adoption_45hh',
    'Adoption_50hh',
    'Adoption_27ho',
    'Adoption_30ho',
    'Adoption_35ho',
    'Adoption_45ho',
    'Adoption_50ho',
'solar_system_count_residential',
'total_panel_area_residential',
'number_of_solar_system_per_household',
'SRpcap',
'SRaPa',
'AvgSres',
'SRaPcap',
'PV_HuOwn',
]  + ['PV_per_100_HuOwn']

def header_printer(msg, pad=2, lpad_len=65, h_f_pad_orig='='):
    h_f_pad = str(h_f_pad_orig)
    for l in range(lpad_len - 1):
        h_f_pad += h_f_pad_orig

    print()
    for p in range(pad):
        print('{}'.format(h_f_pad))

    lpad_str = ""

    # figure out how long msg is
    msg_l = len(msg)
    # subtract this from the lpad_len - 4 more for the trailing and leading spaces around message
    pad_cnt = lpad_len - msg_l - 4
    # now create half of it add the msg and spaces then add the other half
    pad_half = int(np.around(pad_cnt / 2))

    for ph in range(pad_half):
        lpad_str += h_f_pad_orig
    lpad_str = "{}  {}  ".format(lpad_str, msg)
    for ph in range(pad_cnt - pad_half):
        lpad_str += h_f_pad_orig
    print('{}'.format(lpad_str))

    for p in range(pad):
        print('{}'.format(h_f_pad))
    print()


old_med =  ['pop25_some_college_plus', 'high_mf_own_bldg_cnt', 'education_bachelor_rate', 'pop_asian',
               'heating_fuel_coal_coke_rate', 'education_professional_school', 'population_density',
               'mortgage_with_rate', 'hh_size_1', 'total_area', 'pop_hispanic', 'travel_time_average',
               'very_low_mf_own_elep_hh', 'heating_fuel_gas_rate', 'employ_rate', 'age_65_74_rate',
               'transportation_carpool_rate', 'occupation_construction_rate', 'occupation_transportation_rate',
               'age_25_34_rate', 'occupation_arts_rate', 'age_55_64_rate', 'travel_time_30_39_rate',
               '%hh_size_2', 'heating_design_temperature', 'age_18_24_rate', 'hu_1959toearlier',
               'travel_time_20_29_rate', ]
n_wst ='N. West'
wst = 'West'
s_wst ='S. West'
m_wst = 'M. West'
s_est = 'S. East'
m_atl = 'Mid_atlantic'
n_est = 'N. East'

variable_groups = [
    "suitability",
    "renewables",
    "policy_ownership",
    "population_age",
    "energy_income",
    "habit",
    "climate",
    "solar",
    "policy",
    "population",
    "income_housing",
    "occu",
    "occupation",
    "housing",
    "policy_mix",
    "education",
    "ownership_pop",
    "policy_income",
    "income",
    "income_habit",
    "demo",
    "gender",
    "geography",
    "drop",
    "age",
    "politics",
]


l_b_rt = ['high_own_Sbldg_rt', 'mod_own_Sbldg_rt', 'mid_own_Sbldg_rt', 'low_own_Sbldg_rt',
     'verylow_own_Sbldg_rt']
l2_b_c = ['high_own_Sbldg', 'mod_own_Sbldg', 'mid_own_Sbldg', 'low_own_Sbldg',
      'verylow_own_Sbldg']
l3_mwh = ['high_own_mwh', 'mid_own_mwh', 'mod_own_mwh', 'low_own_mwh', 'verylow_own_mwh', ]
l4_elep = ['high_own_elep_hh', 'mod_own_elep_hh', 'low_own_elep_hh', 'verylow_own_elep_hh', ]
own_hh_l = ['high_own_hh', 'mid_own_hh', 'mod_own_hh',  'low_own_hh',
                    'verylow_own_hh',]
own_rt_l = [ 'high_hh_rate',  'mod_hh_rate',  'mid_hh_rate', 'low_hh_rate', 'very_low_hh_rate',]

low_model_RF = ['hu_monthly_owner_costs_greaterthan_1000dlrs', 'mid_mf_own_devp_cnt',
                'education_high_school_or_below_rate', 'education_doctoral', 'solar_prod', 'pop_asian',
                'dlrs_kwh x median_household_income', 'low_own_mwh', 'pop_over_65', 'mortgage_with_rate',
                'p16_unemployed', 'high_mf_own_hh', 'heating_fuel_gas', 'Anti_Occup', 'occupancy_owner_rate',
                'hu_vintage_1940to1959', 'occupation_agriculture_rate', 'occupation_administrative_rate',
                'pop_hispanic', 'travel_time_10_19_rate', 'hh_size_1', 'low_sf_own_elep_hh', 'hu_1960to1979_pct',
                'occupation_transportation_rate', 'heating_fuel_coal_coke_rate', 'travel_time_30_39_rate', 'employ_rate',
                ]

low_model_G_hown_inc_t = ['hu_monthly_owner_costs_greaterthan_1000dlrs', 'high_hh_rate',
                'education_high_school_or_below_rate', 'low_hh_rate', 'solar_prod', 'mid_hh_rate',
                'dlrs_kwh x median_household_income', 'low_own_mwh', 'pop_over_65', 'mortgage_with_rate',
                'avg_monthly_consumption_kwh', 'high_mf_own_hh', 'heating_fuel_gas', 'Anti_Occup', 'occupancy_owner_rate',
                'hu_vintage_1940to1959', 'incent_cnt_res_own', 'net_metering_hu_own',
                'pop_hispanic', 'travel_time_10_19_rate', 'hh_size_1', 'low_sf_own_elep_hh', 'hu_1960to1979_pct',
                'occupation_transportation_rate', 'heating_fuel_coal_coke_rate', 'travel_time_30_39_rate', 'employ_rate',
                ]

low_model_G_hown_int_t_lcn = ['hu_monthly_owner_costs_greaterthan_1000dlrs', 'education_high_school_or_below_rate',
                          'solar_prod', 'dlrs_kwh x median_household_income', 'low_own_mwh', 'pop_over_65',
                          'mortgage_with_rate', 'high_mf_own_hh', 'heating_fuel_gas', 'Anti_Occup',
                          'occupancy_owner_rate', 'hu_vintage_1940to1959', 'pop_hispanic', 'travel_time_10_19_rate',
                          'hh_size_1', 'low_sf_own_elep_hh', 'hu_1960to1979_pct',
                          'occupation_transportation_rate', 'heating_fuel_coal_coke_rate', 'travel_time_30_39_rate',
                          'employ_rate', 'Adoption']

low_model_G_hown_inc_t2 = ['incent_cnt_res_own', 'high_hh_rate', 'population_density',
                'education_high_school_or_below_rate', 'low_hh_rate', 'solar_prod', 'mid_hh_rate',
                'dlrs_kwh x median_household_income', 'low_own_mwh', 'pop_over_65', 'mortgage_with_rate',
                'avg_monthly_consumption_kwh', 'high_mf_own_hh', 'heating_fuel_gas', 'Anti_Occup', 'occupancy_owner_rate',
                'hu_vintage_1940to1959', 'incent_cnt_res_own', 'net_metering_hu_own',
                'pop_hispanic', 'travel_time_10_19_rate', 'hh_size_1', 'low_sf_own_elep_hh', 'hu_1960to1979_pct',
                'occupation_transportation_rate', 'heating_fuel_coal_coke_rate', 'travel_time_30_39_rate', 'employ_rate',
                ]

low_model_rf_select = ['population_density', 'incent_cnt_res_own', 'heating_fuel_coal_coke_rate', 'solar_prod',
                       'travel_time_less_than_10_rate', 'hu_med_val', 'high_own_Sbldg', 'travel_time_40_89_rate',
                       'dlrs_kwh x median_household_income','pop_over_65', 'avg_monthly_consumption_kwh',
                       'hh_gini_index',  'diversity', 'locale_recode(rural)',
                       'locale_recode(suburban)', 'locale_recode(urban)', 'hu_2000toafter_pct', 'net_metering_hu_own',
                       'climate_zone', 'total_area', 'high_hh_rate','mod_hh_rate',  'mid_hh_rate', 'low_hh_rate',
                       'very_low_hh_rate',
                       ]

low_model_rf_selectO = ['population_density', 'high_own_hh',
                        'incent_cnt_res_own', 'high_own_elep_hh',
                        'heating_fuel_coal_coke_rate',
                        'solar_prod',
                       'travel_time_less_than_10_rate',
                        'hu_med_val', 'high_own_Sbldg',
                        'travel_time_40_89_rate',
                       'dlrs_kwh x median_household_income',
                        'pop_over_65',
                        'avg_monthly_consumption_kwh', 'mod_own_elep_hh',
                       'hh_gini_index',  'diversity', 'locale_recode(rural)',
                        'locale_recode(urban)', 'hu_2000toafter_pct', 'net_metering_hu_own',
                       'climate_zone', 'total_area', 'high_hh_rate','mod_hh_rate',  'mid_hh_rate', 'low_hh_rate',
                       'very_low_hh_rate',
                       ]

high_model_RF = [
    'average_household_income', 'mod_mf_own_mwh', 'pop_female', 'occupation_public_rate',
   'age_55_or_more_rate', 'occupation_finance_rate', 'age_65_74_rate', 'travel_time_average',
   'heating_fuel_electricity_rate', 'mod_mf_own_devp_cnt', 'number_of_years_of_education',
   '%hh_size_4', 'high_sf_own_hh', 'p16_unemployed', 'very_low_mf_own_elep_hh', 'age_55_64_rate',
   'high_sf_own_mwh', 'high_mf_own_elep_hh', 'hh_size_4', 'renew_prod', 'high_own_hh',
   'mid_mf_own_hh', 'high_own_Sbldg', 'mod_mf_own_elep_hh',  'Yr_own_mwh',
   'high_sf_own_bldg_cnt', 'high_sf_own_devp_cnt', 'travel_time_less_than_10_rate',
   'diversity', 'high_sf_own_elep_hh', 'incentive_count_residential',
   'education_high_school_graduate_rate', 'heating_fuel_gas_rate', 'low_sf_own_elep_hh',
   'low_mf_own_elep_hh', 'mod_sf_own_elep_hh', 'travel_time_40_89_rate',
   'travel_time_10_19_rate', 'property_tax', 'locale_recode(rural)',
   'heating_design_temperature', 'cooling_design_temperature', 'pop_hispanic',
   'Pro_Occup', 'high_own_mwh', 'education_master', 'Anti_Occup', 'very_low_sf_own_elep_hh',
   'mid_mf_own_devp_m2', 'mid_mf_own_mw', 'mod_mf_own_bldg_cnt', 'incent_cnt_res_own',
   'heating_fuel_gas', 'pop25_some_college_plus', 'education_bachelor', 'net_metering',
   'avg_monthly_bill_dlrs', 'mid_mf_own_devp_cnt', 'locale_dummy', 'occupation_agriculture_rate',
   'net_metering_hu_own', 'heating_fuel_coal_coke_rate', 'cdd', 'solar_prod', 'hydro_prod',
   'high_mf_own_hh', 'avg_monthly_consumption_kwh', 'dlrs_kwh',
   'hu_monthly_owner_costs_greaterthan_1000dlrs', 'mid_mf_own_bldg_cnt', 'pop_asian',
   'hu_med_val', 'daily_solar_radiation', 'mortgage_with_rate', 'total_area', 'population_density',
   'high_mf_own_bldg_cnt'
]

high_model_RF2 = [
    'average_household_income', 'mod_mf_own_mwh', 'pop_female', 'occupation_public_rate',
   'age_55_or_more_rate', 'occupation_finance_rate', 'age_65_74_rate', 'travel_time_average',
   'heating_fuel_electricity_rate', 'mod_mf_own_devp_cnt', 'number_of_years_of_education',
   '%hh_size_4', 'high_sf_own_hh', 'p16_unemployed', 'very_low_mf_own_elep_hh', 'age_55_64_rate',
   'high_sf_own_mwh', 'high_mf_own_elep_hh', 'hh_size_4', 'renew_prod', 'high_own_hh',
   'mid_mf_own_hh', 'high_own_Sbldg', 'mod_mf_own_elep_hh',  'Yr_own_mwh',
   'high_sf_own_bldg_cnt', 'high_sf_own_devp_cnt', 'travel_time_less_than_10_rate',
   'diversity', 'high_sf_own_elep_hh', 'incentive_count_residential',
   'education_high_school_graduate_rate', 'heating_fuel_gas_rate', 'low_sf_own_elep_hh',
   'low_mf_own_elep_hh', 'mod_sf_own_elep_hh', 'travel_time_40_89_rate',
   'travel_time_10_19_rate', 'property_tax', 'locale_recode(rural)',
   'heating_design_temperature', 'cooling_design_temperature', 'pop_hispanic',
   'Pro_Occup', 'high_own_mwh', 'education_master', 'Anti_Occup', 'very_low_sf_own_elep_hh',
   'mid_mf_own_devp_m2', 'mid_mf_own_mw', 'mod_mf_own_bldg_cnt', 'incent_cnt_res_own',
   'heating_fuel_gas', 'pop25_some_college_plus', 'education_bachelor', 'net_metering',
   'avg_monthly_bill_dlrs', 'mid_mf_own_devp_cnt', 'locale_dummy', 'occupation_agriculture_rate',
   'net_metering_hu_own', 'heating_fuel_coal_coke_rate', 'cdd', 'solar_prod', 'hydro_prod',
   'high_mf_own_hh', 'avg_monthly_consumption_kwh', 'dlrs_kwh',
   'hu_monthly_owner_costs_greaterthan_1000dlrs', 'mid_mf_own_bldg_cnt', 'pop_asian',
   'hu_med_val', 'daily_solar_radiation', 'mortgage_with_rate', 'total_area', 'population_density',
   'high_mf_own_bldg_cnt'
]

high_model_RF_rd = list(set([
    'average_household_income', 'mod_mf_own_mwh', 'pop_female', 'occupation_public_rate',
   'age_55_or_more_rate', 'occupation_finance_rate', 'age_65_74_rate', 'travel_time_average',
   'heating_fuel_electricity_rate', 'mod_mf_own_devp_cnt', 'number_of_years_of_education',
   '%hh_size_4', 'high_sf_own_hh', 'p16_unemployed', 'very_low_mf_own_elep_hh', 'age_55_64_rate',
   'high_sf_own_mwh', 'high_mf_own_elep_hh', 'hh_size_4', 'renew_prod', 'high_own_hh',
   'mid_mf_own_hh', 'high_own_Sbldg', 'mod_mf_own_elep_hh', 'hu_own_pct', 'Yr_own_mwh',
   'high_sf_own_bldg_cnt', 'high_sf_own_devp_cnt', 'travel_time_less_than_10_rate',
   'diversity', 'high_sf_own_elep_hh',
   'education_high_school_graduate_rate', 'heating_fuel_gas_rate', 'low_sf_own_elep_hh',
   'low_mf_own_elep_hh', 'mod_sf_own_elep_hh', 'travel_time_40_89_rate',
   'travel_time_10_19_rate',  'locale_recode(rural)',
   'heating_design_temperature', 'cooling_design_temperature', 'pop_hispanic',
   'Pro_Occup', 'high_own_mwh', 'education_master', 'Anti_Occup', 'very_low_sf_own_elep_hh',
   'mid_mf_own_devp_m2', 'mid_mf_own_mw', 'mod_mf_own_bldg_cnt',  'incent_cnt_res_own',
   'heating_fuel_gas', 'pop25_some_college_plus', 'education_bachelor', 'net_metering',
   'avg_monthly_bill_dlrs', 'mid_mf_own_devp_cnt', 'locale_dummy', 'occupation_agriculture_rate',
   'net_metering_hu_own', 'heating_fuel_coal_coke_rate', 'cdd', 'solar_prod', 'hydro_prod',
   'high_mf_own_hh', 'avg_monthly_consumption_kwh', 'dlrs_kwh',
   'hu_monthly_owner_costs_greaterthan_1000dlrs', 'mid_mf_own_bldg_cnt', 'pop_asian',
   'hu_med_val', 'daily_solar_radiation', 'mortgage_with_rate', 'total_area', 'population_density',
   'high_mf_own_bldg_cnt', 'high_hh_rate',
] +l4_elep))

high_model_RF_rd2 = list(set([
    'average_household_income', 'mod_mf_own_mwh', 'pop_female', 'occupation_public_rate',
   'age_55_or_more_rate', 'occupation_finance_rate', 'age_65_74_rate', 'travel_time_average',
   'heating_fuel_electricity_rate', 'mod_mf_own_devp_cnt', 'number_of_years_of_education',
   '%hh_size_4', 'high_sf_own_hh', 'p16_unemployed', 'very_low_mf_own_elep_hh', 'age_55_64_rate',
   'high_sf_own_mwh', 'high_mf_own_elep_hh', 'hh_size_4', 'renew_prod', 'high_own_hh',
   'mid_mf_own_hh', 'high_own_Sbldg', 'mod_mf_own_elep_hh', 'hu_own_pct', 'Yr_own_mwh',
   'high_sf_own_bldg_cnt', 'high_sf_own_devp_cnt', 'travel_time_less_than_10_rate',
   'diversity', 'high_sf_own_elep_hh',
   'education_high_school_graduate_rate', 'heating_fuel_gas_rate', 'low_sf_own_elep_hh',
   'low_mf_own_elep_hh', 'mod_sf_own_elep_hh', 'travel_time_40_89_rate',
   'travel_time_10_19_rate', 'property_tax', 'locale_recode(rural)',
   'heating_design_temperature', 'cooling_design_temperature', 'pop_hispanic',
   'Pro_Occup', 'high_own_mwh', 'education_master', 'Anti_Occup', 'very_low_sf_own_elep_hh',
   'mid_mf_own_devp_m2', 'mid_mf_own_mw', 'mod_mf_own_bldg_cnt',  'incent_cnt_res_own',
   'heating_fuel_gas', 'pop25_some_college_plus', 'education_bachelor', 'net_metering',
   'avg_monthly_bill_dlrs', 'mid_mf_own_devp_cnt', 'locale_dummy', 'occupation_agriculture_rate',
   'net_metering_hu_own', 'heating_fuel_coal_coke_rate', 'cdd', 'solar_prod', 'hydro_prod',
   'high_mf_own_hh', 'avg_monthly_consumption_kwh', 'dlrs_kwh',
   'hu_monthly_owner_costs_greaterthan_1000dlrs', 'mid_mf_own_bldg_cnt', 'pop_asian',
   'hu_med_val', 'daily_solar_radiation', 'mortgage_with_rate', 'total_area', 'population_density',
   'high_mf_own_bldg_cnt', 'high_hh_rate',
] +l4_elep))

block_group_ = [
"education",            # 0
"age",                  # 1
"income",               # 2
"income_housing",       # 3
"occupation",           # 4
"habit",                # 5
"geography",            # 6
"demo",                 # 7
"demographics",         # 8
"policy",               # 9
"gender",               # 10
"population",           # 11
"suitability",          # 12
"housing",              # 13
"politics",             # 14
"climate",              # 15
"renewables",           # 16
]

hierarchical_resutls_dict = {
    'Demographics':[],
    'Economics': [],
    'Suitability': [],
    'Behaviors': [],
    'Policy': [],
}

hierarchical_groupings = {
    "Demographics":['median_household_income', "education_bachelor_or_above_rate", 'age_median', '%hh_size_1',
                    'pop25_some_college_plus', 'age_minor_rate', 'age_zoomer_rate', 'locale_recode(rural)',
                    '%hh_size_3', '%hh_size_4','high_own_hh', 'mod_own_hh', 'low_own_hh', 'total_own_hh',
                    'diversity', "educated_population_rate", "age_18_24_rate", 'Gender_Ratio'],
    "Demographics1":["education_bachelor_or_above_rate", 'age_median', '%hh_size_1',
                    'pop25_some_college_plus', 'age_minor_rate', 'age_zoomer_rate', 'locale_recode(rural)',
                    '%hh_size_3', '%hh_size_4','high_own_hh', 'mod_own_hh', 'low_own_hh', 'Gender_Ratio',
                    'diversity',  ],
    "Demographics2":["education_bachelor_or_above_rate", 'age_median', '%hh_size_1',
                    'pop25_some_college_plus',  'locale_recode(rural)',
                    '%hh_size_3', '%hh_size_4','high_own_hh', 'mod_own_hh', 'low_own_hh',
                    'diversity', "age_18_24_rate","age_65_74_rate", 'Gender_Ratio'],
    # age, household size, educated pop, local/r/sub, pop citizen, income hh break downs, diversity, gender ratio
    "Demographics3":['age_median', '%hh_size_1', 'pop25_some_college_plus',  'locale_recode(rural)',
                     'locale_recode(suburban)',"pop_nat_us_citizen",
                     '%hh_size_3', '%hh_size_4','high_sf_own_hh','high_mf_own_hh', 'mod_sf_own_hh',
                     'mod_mf_own_hh','low_sf_own_hh','low_mf_own_hh', 'mid_sf_own_hh','mid_mf_own_hh',
                     'very_low_sf_own_hh','very_low_mf_own_hh',
                     'diversity', "age_18_24_rate","age_65_74_rate", 'Gender_Ratio'],
    # age, household size, local/r/sub, pop citizen, income hh break downs, diversity, gender ratio
    "Demographics4":['age_median', '%hh_size_1', 'locale_recode(rural)',
                     'locale_recode(suburban)',"pop_nat_us_citizen",
                     '%hh_size_3', '%hh_size_4','high_sf_own_hh','high_mf_own_hh', 'mod_sf_own_hh',
                     'mod_mf_own_hh','low_sf_own_hh','low_mf_own_hh', 'mid_sf_own_hh','mid_mf_own_hh',
                     'verylow_sf_own_hh','verylow_mf_own_hh',
                     'diversity', "age_18_24_rate","age_65_74_rate", 'Gender_Ratio'],
    # age, household size, local/r/sub, pop citizen, income hh break downs, diversity, gender ratio
    "Demographics4a":['age_median', '%hh_size_1', "age_18_24_rate","age_65_74_rate",
                      'locale_recode(rural)', 'locale_recode(suburban)',
                      "pop_nat_us_citizen","poverty_family_count",
                      "hu_own_pct","mortgage_with_rate",
                      '%hh_size_3', '%hh_size_4','high_sf_own_hh','high_mf_own_hh', 'mod_sf_own_hh',
                      'mod_mf_own_hh','low_sf_own_hh','low_mf_own_hh', 'mid_sf_own_hh','mid_mf_own_hh',
                      'very_low_sf_own_hh','very_low_mf_own_hh',
                      'diversity', 'Gender_Ratio'],
    # age, household size, local/r/sub, pop citizen, income hh break downs, diversity, gender ratio
    "Demographics4b":['age_median', '%hh_size_1', "age_18_24_rate","age_65_74_rate",
                      'locale_recode(rural)',
                      "pop_nat_us_citizen",
                      "hu_own_pct","mortgage_with_rate",
                      '%hh_size_3', '%hh_size_4','high_sf_own_hh','high_mf_own_hh', 'mod_sf_own_hh',
                      'low_sf_own_hh','low_mf_own_hh', 'mid_sf_own_hh','mid_mf_own_hh',
                      'very_low_sf_own_hh','very_low_mf_own_hh',
                      'diversity', 'Gender_Ratio'],
    "Demographics4c":['age_median', '%hh_size_1', "age_18_24_rate","age_65_74_rate",
                      'locale_recode(rural)',
                      "pop_nat_us_citizen","age_25_34_rate",
                      "hu_own_pct","mortgage_with_rate",
                      '%hh_size_3', '%hh_size_4', 'total_sf_own_hh', 'total_mf_own_hh',
                      'diversity', 'Gender_Ratio'],
    "Demographics4d":[  "age_18_24_rate","age_65_74_rate", #'%hh_size_1',
                      'locale_recode(rural)',"age_45_54_rate","age_55_64_rate",
                       "age_minor_rate", '%hh_size_3', "household_type_family_rate",
                      "pop_nat_us_citizen", "age_35_44_rate",  "age_25_34_rate", # "p16_employed",
                      "hu_own_pct","mortgage_with_rate","p16_employed",
                      '%hh_size_4', "high_own_hh",  "low_own_hh", "mid_own_hh",
                      'diversity', 'Gender_Ratio', ],
    "Demographics4d2":['age_median', #"age_65_74_rate",
                      'locale_recode(rural)',
                       #"age_minor_rate", #'%hh_size_3',
                      "pop_nat_us_citizen", "household_type_family_rate",  #"age_25_34_rate",
                      "hu_own_pct",
                      "employ_rate",
                      # "p16_employed",
                      #"average_household_size", #'%hh_size_4',
                       "pop_african_american",
                       "pop_asian",
                       "pop_caucasian",
                       'population_density',
                       'hu_med_val',"median_household_income",
                       "pop_hispanic",
                      'Gender_Ratio'],
    "DemographicsSP":[
        "pop_african_american",
        "pop_asian",
        "pop_caucasian",
        "pop_hispanic",
        'age_median',
        "age_65_74_rate",
        'Gender_Ratio',
        "average_household_size",  # '%hh_size_4',
        "household_type_family_rate",  # "age_25_34_rate",
        "hu_own_pct",
        'hu_med_val',
        "median_household_income",
        "employ_rate",
        'population_density',
        'locale_recode(rural)',
        'voting_2012_dem_percentage',
        'voting_2012_gop_percentage',
        #'political_ratio',
        #'state_fips',
    ],
    "DemographicsO":[
        'age_median',
        #"pop_female",
        'hu_own',
        'hu_rent',
        'pop25_some_college_plus',
        "p16_employed",
        "female_pct",
        "age_65_74_rate", "age_minor_rate",
        'Gender_Ratio',
        'locale_recode(rural)',
        'locale_recode(town)',
        'locale_recode(suburban)',
        'locale_recode(urban)',
          'population_density',
          #"pop_nat_us_citizen",
          "pop_african_american",
          "pop_asian",
          "pop_caucasian",
          "pop_hispanic",
          #"black_pct",
          #"asian_pct",
          #"white_pct",
          #"hispanic_pct",
          #"pop25_some_college_plus",
          #"number_of_years_of_education",
          "education_bachelor_or_above_rate",
          "education_college_rate",
          #"dlrs_kwh x median_household_income",
          "diversity",
          "household_type_family_rate",  #"age_25_34_rate",
          "hu_own_pct",
          "employ_rate",
          #'political_ratio',
          #'voting_2012_dem_percentage',
          #'voting_2012_gop_percentage',
          "average_household_size", #'%hh_size_4',
          'hu_med_val',
          "median_household_income",
                      ],
    "DemographicsO_racecnt":[
        'age_median',
        #"pop_female",
        #'hu_own',
        'hu_rent',
        'pop25_some_college_plus',
        #"p16_employed",
        #"female_pct",
        #"age_65_74_rate",
        #"age_minor_rate",
        'Gender_Ratio',
        'locale_recode(rural)',
        'locale_recode(town)',
        'locale_recode(suburban)',
        'locale_recode(urban)',
        'population_density',
        #"pop_nat_us_citizen",
        "pop_african_american",
        "pop_asian",
        #"pop_caucasian",
        "pop_hispanic",
        #"black_pct",
        #"asian_pct",
        #"white_pct",
        #"hispanic_pct",
       #"pop25_some_college_plus",
       #"number_of_years_of_education",
       #"education_bachelor_or_above_rate",
       #"education_college_rate",
       #"dlrs_kwh x median_household_income",
       "diversity",
       "household_type_family_rate",  #"age_25_34_rate",
       #"hu_own_pct",
       #"employ_rate",
          #'political_ratio',
          #'voting_2012_dem_percentage',
          #'voting_2012_gop_percentage',
          "average_household_size", #'%hh_size_4',
          'hu_med_val',
          "median_household_income",
                      ],
    "DemographicsO_racepct":[
        'age_median',
        #"pop_female",
        #'hu_own',
        'hu_rent',
        'pop25_some_college_plus',
        #"p16_employed",
        #"female_pct",
        #"age_65_74_rate",
        #"age_minor_rate",
        'Gender_Ratio',
        'locale_recode(rural)',
        'locale_recode(town)',
        'locale_recode(suburban)',
        'locale_recode(urban)',
        'population_density',
        #"pop_nat_us_citizen",
        #"pop_african_american",
        #"pop_asian",
        #"pop_caucasian",
        #"pop_hispanic",
        "black_pct",
        "asian_pct",
        #"white_pct",
        "hispanic_pct",
       #"pop25_some_college_plus",
       #"number_of_years_of_education",
       #"education_bachelor_or_above_rate",
       #"education_college_rate",
       #"dlrs_kwh x median_household_income",
       "diversity",
       "household_type_family_rate",  #"age_25_34_rate",
       #"hu_own_pct",
       #"employ_rate",
          #'political_ratio',
          #'voting_2012_dem_percentage',
          #'voting_2012_gop_percentage',
          "average_household_size", #'%hh_size_4',
          'hu_med_val',
          "median_household_income",
                      ],
    "DemographicsO_lograce":[
        'age_median',
        #"pop_female",
        #'hu_own',
        'hu_rent',
        'pop25_some_college_plus',
        #"p16_employed",
        #"female_pct",
        #"age_65_74_rate",
        #"age_minor_rate",
        'Gender_Ratio',
        'locale_recode(rural)',
        'locale_recode(town)',
        'locale_recode(suburban)',
        'locale_recode(urban)',
        'population_density',
        #"pop_nat_us_citizen",
        "pop_african_american_log10",
        "pop_asian_log10",
        #"pop_caucasian_log10",
        "pop_hispanic_log10",
        #"black_pct",
        #"asian_pct",
        #"white_pct",
        #"hispanic_pct",
       #"pop25_some_college_plus",
       #"number_of_years_of_education",
       #"education_bachelor_or_above_rate",
       #"education_college_rate",
       #"dlrs_kwh x median_household_income",
       "diversity",
       "household_type_family_rate",  #"age_25_34_rate",
       #"hu_own_pct",
       #"employ_rate",
          #'political_ratio',
          #'voting_2012_dem_percentage',
          #'voting_2012_gop_percentage',
          "average_household_size", #'%hh_size_4',
          'hu_med_val',
          "median_household_income",
                      ],
    "DemographicsO_logall":[
        'age_median_log',
        #"pop_female",
        #'hu_own',
        'hu_rent_log',
        'pop25_some_college_plus',
        #"p16_employed",
        #"female_pct",
        #"age_65_74_rate",
        #"age_minor_rate",
        'Gender_Ratio_log10',
        'locale_recode(rural)',
        'locale_recode(town)',
        'locale_recode(suburban)',
        'locale_recode(urban)',
        'population_density_log10',
        #"pop_nat_us_citizen",
        "pop_african_american_log10",
        "pop_asian_log10",
        #"pop_caucasian_log10",
        "pop_hispanic_log10",
        #"black_pct",
        #"asian_pct",
        #"white_pct",
        #"hispanic_pct",
       #"pop25_some_college_plus",
       #"number_of_years_of_education",
       #"education_bachelor_or_above_rate",
       #"education_college_rate",
       #"dlrs_kwh x median_household_income",
       "diversity_log10",
       "household_type_family_rate_log10",  #"age_25_34_rate",
       #"hu_own_pct",
       #"employ_rate",
          "average_household_size_log10", #'%hh_size_4',
          'hu_med_val_log10',
          "median_household_income_log10",
                      ],
    "Demographics4d3":['age_median', "age_65_74_rate", #'%hh_size_1',
                      'locale_recode(rural)',  "age_minor_rate", '%hh_size_3',
                      "pop_nat_us_citizen", "household_type_family_rate",  #"age_25_34_rate",
                      "hu_own_pct","mortgage_with_rate","p16_employed",
                      '%hh_size_4',
                       #'',
                       "high_own_hh",  #"low_own_hh", "mid_own_hh",
                      'diversity', 'Gender_Ratio'],
    # age, household size, educated pop, local/r/sub, pop citizen, total owned homes, diversity, gender ratio
    "Demographics5":['age_median', "age_18_24_rate","age_65_74_rate", "age_25_34_rate","age_minor_rate",
                     'pop25_some_college_plus',"pop_nat_us_citizen", 'diversity',
                     'locale_recode(rural)','locale_recode(suburban)',
                     "hu_own_pct","mortgage_with_rate",
                     '%hh_size_1', '%hh_size_2','%hh_size_3', '%hh_size_4',
                     'Gender_Ratio', ],
# age, household size, educated pop, local/r/sub, pop citizen, total owned homes, diversity, gender ratio
    "Demographics5a":['age_median', "age_18_24_rate","age_65_74_rate","age_minor_rate",
                     'pop25_some_college_plus',"pop_nat_us_citizen", 'diversity',
                     'locale_recode(rural)','locale_recode(suburban)',
                     "hu_own_pct","mortgage_with_rate",
                     '%hh_size_1', '%hh_size_4',
                     'Gender_Ratio', ],

    "Economics": ['dlrs_kwh', 'heating_fuel_coal_coke_rate', "avg_monthly_bill_dlrs",
                  "Yrl_savings_$",'heating_fuel_electricity_rate', 'Yrl_%_inc',
                   'hu_med_val',"median_household_income",],
    "Economics1": ["avg_monthly_bill_dlrs",
                   "Yrl_savings_$",'heating_fuel_electricity_rate',
                   'heating_fuel_coal_coke_rate',
                   ],
    "Economics1spss": [
        "Yrl_savings_$",
        "avg_monthly_bill_dlrs",
        'heating_fuel_electricity_rate',
        'heating_fuel_coal_coke_rate', 'dlrs_kwh',
   ],
    "Economics_O": [
        "Yrl_savings_$",
        "avg_monthly_bill_dlrs",
        'heating_fuel_electricity_rate',
   ],
    "Economics1a": ["dlrs_kwh x median_household_income",  "avg_monthly_bill_dlrs",
                  "Yrl_savings_$",'heating_fuel_electricity_rate',
                   'hu_med_val', 'heating_fuel_coal_coke_rate', ],
    "Economics2": ['dlrs_kwh',  "avg_monthly_bill_dlrs",
                  "Yrl_savings_$",'heating_fuel_electricity_rate','Yrl_%_inc',
                  'heating_fuel_coal_coke_rate', "median_household_income",],
    "Economics3": ["dlrs_kwh x median_household_income",  "avg_monthly_bill_dlrs",
                  "Yrl_savings_$",'heating_fuel_electricity_rate','Yrl_%_inc',
                  'heating_fuel_coal_coke_rate', ],
    "Behavior": ['travel_time_less_than_10_rate', "travel_time_average",
                 "travel_time_40_89_rate","voting_2012_gop_percentage",
                  "avg_monthly_consumption_kwh", ],
    "Behaviora": ['travel_time_less_than_10_rate', "travel_time_average",
                 "travel_time_40_89_rate","voting_2012_gop_percentage",
                  "avg_monthly_consumption_kwh", 'Green_Travelers',
                  "travel_time_20_29_rate", 'political_ratio'],
    "Behaviorb": ['travel_time_less_than_10_rate', "travel_time_10_19_rate",
                 "travel_time_40_89_rate","voting_2012_gop_percentage",
                  "avg_monthly_consumption_kwh", 'Green_Travelers',
                  "travel_time_30_39_rate",
                  "travel_time_20_29_rate", 'political_ratio'],
# this
    "Behaviorb2": [
                   'travel_time_less_than_10_rate',
                   "travel_time_10_19_rate",
                   "travel_time_40_89_rate",
                   'transportation_home_rate',
                   'transportation_bicycle_rate',
                   'transportation_walk_rate',
                   'low_commute_times',
                   #"voting_2012_gop_percentage",
                   #"travel_time_average",
                   "avg_monthly_consumption_kwh",
                   #"travel_time_30_39_rate",
                   #"travel_time_20_29_rate",
                   #'political_ratio',
                   'Green_Travelers',
                    ],
    "Behaviorb_O1": [
                   'travel_time_less_than_10_rate',
                   "travel_time_10_19_rate",
                   "travel_time_40_89_rate",
                   #'low_commute_times',
                   #"voting_2012_gop_percentage",
                   #"travel_time_average",
                   #"avg_monthly_consumption_kwh",
                   #"travel_time_30_39_rate",
                   #"travel_time_20_29_rate",
                   #'political_ratio',
                   'Green_Travelers',
                    ],
    "Behaviorb_O2": [
                   'travel_time_less_than_10_rate_log10',
                   "travel_time_10_19_rate_log10",
                   "travel_time_40_89_rate_log10",
                   #'low_commute_times',
                   #"voting_2012_gop_percentage",
                   #"travel_time_average",
                   #"avg_monthly_consumption_kwh",
                   #"travel_time_30_39_rate",
                   #"travel_time_20_29_rate",
                   #'political_ratio',
                   'Green_Travelers_log10',
                    ],
    "Behaviorc": ['travel_time_less_than_10_rate', "travel_time_average",
                 "travel_time_40_89_rate","voting_2012_dem_percentage",
                  "avg_monthly_consumption_kwh", 'Green_Travelers',
                  "travel_time_20_29_rate", "voting_2012_gop_percentage",],
    "Behaviord": ['travel_time_less_than_10_rate', "travel_time_average",
                 "travel_time_40_89_rate",
                  "avg_monthly_consumption_kwh", 'Green_Travelers',
                  "travel_time_20_29_rate", 'political_ratio'],
    "Suitability": ['population_density', 'daily_solar_radiation',
                    "Yr_own_mwh", 'total_area', 'cdd_std', 'high_own_mwh','mod_own_mwh','low_own_mwh',
                    'total_own_devp', ],
    "Suitability2": ['population_density', 'daily_solar_radiation',
                    'total_area', 'cdd_std', 'high_own_mwh','mod_own_mwh','low_own_mwh',
                    'total_own_devp', ],
    "Suitability1": ['population_density', 'daily_solar_radiation',
                    "Yr_own_mwh", 'total_area', 'cdd_std', 'high_own_mwh','low_own_mwh',
                    ],
    #
    "Suitability3": ['population_density', 'total_area',
                     'daily_solar_radiation', 'cdd_std',
                     'high_own_mwh','mod_own_mwh','low_own_mwh',
                    "hu_1959toearlier_pct", "hu_2000toafter_pct",
                     ],
    # .134
    "Suitability4": ['population_density', 'daily_solar_radiation',
                    'total_area', 'cdd_std', "Yr_own_mwh",
                     ],
    # .132
    "Suitability5": ['population_density', 'daily_solar_radiation',
                     'total_area', 'cdd_std', 'total_own_devp',
                     ],
    # .152
    "Suitability6": ['population_density', 'daily_solar_radiation',
                     'total_area', 'cdd_std',
                    #'total_own_devp',
                     #'low_own_devp',
                      #'mod_own_devp', 'high_own_devp',
                     "hu_1959toearlier_pct", "hu_2000toafter_pct",
                     ],
    "Suitability7": ['daily_solar_radiation',
                     'total_area', 'cdd_std', 'hdd_std', 'cdd', 'hdd',
                    'total_own_devp', "Yr_own_mwh",
                     'low_own_devp',
                      'mod_own_devp', 'high_own_devp',
                     "total_own_Sbldg",
                     'high_own_mwh','mod_own_mwh','low_own_mwh',
                     "hu_1959toearlier_pct", "hu_2000toafter_pct",
                     ],
    "Suitability_O1": [
                    'daily_solar_radiation',
                     #'total_area',
                    'cdd_std',
                    #'hdd_std',
                    #'cdd',
                    #'hdd',
                    #'total_own_devp', "Yr_own_mwh",
                    # 'low_own_devp',
                    #  'mod_own_devp', 'high_own_devp',
                    # "total_own_Sbldg",
                    # 'high_own_mwh','mod_own_mwh','low_own_mwh',
                     "hu_1959toearlier_pct", "hu_2000toafter_pct",
                     ],
    "Suitability_O2": [
                    'daily_solar_radiation',
                     #'total_area',
                    #'cdd_std',
                    'hdd_std',
                    #'cdd',
                    #'hdd',
                    #'total_own_devp', "Yr_own_mwh",
                    # 'low_own_devp',
                    #  'mod_own_devp', 'high_own_devp',
                    # "total_own_Sbldg",
                    # 'high_own_mwh','mod_own_mwh','low_own_mwh',
                     "hu_1959toearlier_pct", "hu_2000toafter_pct",
                     ],
    "Suitability_O3": [
                    'daily_solar_radiation',
                     #'total_area',
                    #'cdd_std',
                    #'hdd_std',
                    'cdd',
                    #'hdd',
                    #'total_own_devp', "Yr_own_mwh",
                    # 'low_own_devp',
                    #  'mod_own_devp', 'high_own_devp',
                    # "total_own_Sbldg",
                    # 'high_own_mwh','mod_own_mwh','low_own_mwh',
                     "hu_1959toearlier_pct", "hu_2000toafter_pct",
                     ],
    "Suitability_O3log": [
                    'daily_solar_radiation_log10',
                     #'total_area',
                    #'cdd_std',
                    #'hdd_std',
                    'cdd_log10',
                    #'hdd',
                    #'total_own_devp', "Yr_own_mwh",
                    # 'low_own_devp',
                    #  'mod_own_devp', 'high_own_devp',
                    # "total_own_Sbldg",
                    # 'high_own_mwh','mod_own_mwh','low_own_mwh',
                    "hu_1959toearlier_pct_log10",
                    "hu_2000toafter_pct_log10",
                     ],
    "Suitabilityspss": [
        'daily_solar_radiation',
        'total_area',
        'cdd_std', 'cdd',
        "hu_1959toearlier_pct", "hu_2000toafter_pct",
    ],
    "Policy": ['incent_cnt_res_own',
               'incentive_count_residential',
               'net_metering_hu_own',
               'net_metering_bin',
               "property_tax_hu_own",
                'property_tax_bin',
               #'renew_prod',
               'solar_prod',
    ],
    "PolicyA": ['incent_cnt_res_own',
                'incentive_residential_state_level',
               'incentive_count_residential',
               'net_metering_hu_own',
               'net_metering_bin',
               'net_metering',
                "property_tax_hu_own",
                'property_tax_bin',
                'property_tax',
                #"avg_ibi_pct",
                #"avg_cbi_usd_p_w",
                #"avg_pbi_usd_p_kwh",
                #'renew_prod',
               'solar_prod', ],
    "Policy_O": [
                'incent_cnt_res_own',
                #'incentive_residential_state_level',
                'incentive_count_residential',
                'net_metering_hu_own',
                'net_metering_bin',
                #'net_metering',
                "property_tax_hu_own",
                'property_tax_bin',
                #'property_tax',
                # "avg_ibi_pct",
                # "avg_cbi_usd_p_w",
                # "avg_pbi_usd_p_kwh",
                # 'renew_prod',
                'solar_prod', ],
    "Policy_O1log": [
                'incent_cnt_res_own_log10',
                #'incentive_residential_state_level',
                'incentive_count_residential_log10',
                'net_metering_hu_own_log10',
                'net_metering_bin_log10',
                #'net_metering',
                "property_tax_hu_own_log10",
                'property_tax_bin_log10',
                #'property_tax',
                # "avg_ibi_pct",
                # "avg_cbi_usd_p_w",
                # "avg_pbi_usd_p_kwh",
                # 'renew_prod',
                'solar_prod_log10', ],
    "Policy1": ['incent_cnt_res_own', 'net_metering_hu_own', 'net_metering',
                'property_tax_bin', 'renew_prod', 'solar_prod',],
}



class Block_Groups:
    """
        Class containing several other classes for grabing specific types of variables
    """
    """ will in the end hold variables separated into different block groups"""
    # names of the groups
    block_groups = ['demo',       # 0
                    'policy',     # 1
                    'physical',   # 2
                    'habit',       # 3,
                    'climate',    # 4,
                    'geography',  # 5,
                    'population', # 6,
                    'base',       # 7,
                    'edu',        # 8
                    'age',        # 9,
                    'hh',         # 10,
                    'ct_demo',    # 11,
                    'income',     # 12,
                    'Xu',         # 13
                    'income employment', #14
                    'energy',            #15
                    'dwelling',          #16
                    'hh_size',           #17
                    'political',         #18
                    'gender',            #19
                    None,       #20
                    'All',
                    ]

    basic_sets = ['default','scld_default','nrml_default','top',]
    class _BG_:  # base class for the different block group
        acceptabletypes = [type(int(0)), type(float(.0)), type(str('')), type(np.float), type(()), type(()), ]
        dflt = list()

        def __init__(self, my_sets, set=None, dflt=(dflt,)):
            self.dflt = dflt[0]
            self.model = self.process(my_sets, set)

        def process(self, my_sets, set=None):
            if set is None:
                return my_sets['default']
            elif type(set) in self.acceptabletypes:
                return my_sets[set]

    class _policy_(_BG_):
        """
            Will hold policy style variables that are different policies
            that can be employed
        """
        dflt = [
                'Adoption',
                'incent_cnt_res_own',
                'net_metering_hu_own',
                'incentive_count_nonresidential',
                'incentive_count_residential',
                'incentive_nonresidential_state_level',
                'incentive_residential_state_level',
                'net_metering',
                'property_tax_bin',
                'dlrs_kwh',
                'hydro_prod',
                'renew_prod',
                'solar_prod',
                'avg_electricity_retail_rate',
                ]



        dflt_nrml = dflt

        topp = ['Adoption', 'Ren', 'dlrs_kwh', 'net_metering', ]

        my_sets = {
            'default': dflt,
            'scld_default': dflt,
            'nrml_default': dflt_nrml,
            'top': topp}

        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set

        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _RF_Select_(_BG_):
        """
            Will hold policy style variables that are different policies
            that can be employed
        """
        dflt = [
                'Adoption',
                'population_density',
                'total_area',
                'pop25_some_college_plus',
                'number_of_years_of_education',
                'hu_med_val',
                'dlrs_kwh',
                'avg_monthly_consumption_kwh',
                'education_bachelor',
                'education_population',
                'hdd',
                'hu_mortgage',
                'hu_own',
                'cdd',
                'transportation_car_alone_rate',
                'fam_med_income',
                'travel_time_average',
                'housing_unit_count',
                'transportation_public_rate',
                #'heating_fuel_none',
                'poverty_family_below_poverty_level_rate',
                'occupation_finance_rate',
                'travel_time_10_19_rate',
                #'heating_fuel_solar',
                'travel_time_10_19_rate',
                'hu_vintage_1960to1970',
                'hu_1959toearlier_pct',
                'voting_2012_dem_percentage',
                'voting_2012_gop_percentage',
                'age_55_64_rate',
                'heating_fuel_electricity_rate',
                'heating_fuel_coal_coke_rate',
                'age_median',
                'age_55_or_more_rate',
                'Green_Travelers',
                'travel_time_60_89_rate',
                'transportation_home_rate',
                'transportation_bicycle_rate',
                'transportation_carpool_rate',
                'travel_time_40_89_rate',
                ]

        dfltA = [
                'Adoption',
                'population_density',
                'number_of_years_of_education',

                'total_area',
                'poverty_family_below_poverty_level_rate',
                'occupation_finance_rate',
                'travel_time_10_19_rate',

                'travel_time_10_19_rate',
                'hu_mortgage',
                'hu_med_val',
                'hu_vintage_1960to1970',
                'hu_1959toearlier_pct',
                'voting_2012_dem_percentage',
                'voting_2012_gop_percentage',
                'hu_own',
                'fam_med_income',
                'pop25_some_college_plus',
                'education_population',
                'education_bachelor',
                'age_55_64_rate',
                'heating_fuel_electricity_rate',
                'heating_fuel_coal_coke_rate',
                'age_median',
                'age_55_or_more_rate',
                'dlrs_kwh',
                'housing_unit_count',
                'cdd',
                'hdd',
                'Green_Travelers',
                'travel_time_average',
                'travel_time_60_89_rate',
                'transportation_home_rate',

                'transportation_car_alone_rate',
                'transportation_carpool_rate',
                'transportation_public_rate',
                'avg_monthly_consumption_kwh',
                'travel_time_40_89_rate',
                ]

        dfltB = [
            'Adoption',
            'population_density',
            'total_area',
            'pop25_some_college_plus',
            'number_of_years_of_education',
            'hu_med_val',
            'dlrs_kwh',
            'avg_monthly_consumption_kwh',
            'education_bachelor',
            'education_population',
            'hdd',
            'hu_mortgage',
            'hu_own',
            'cdd',
            'transportation_car_alone_rate',
            'fam_med_income',
            'travel_time_average',
            'housing_unit_count',
            'transportation_public_rate',
            # 'heating_fuel_none',
            'poverty_family_below_poverty_level_rate',
            'occupation_finance_rate',
            'travel_time_10_19_rate',
            # 'heating_fuel_solar',
            'hu_vintage_1960to1970',
            'hu_1959toearlier_pct',
            'voting_2012_dem_percentage',
            'voting_2012_gop_percentage',
            'age_55_64_rate',
            ]

        dfltC = [
            'Adoption',
            'population_density',
            'total_area',
            'land_area',
            'pop25_some_college_plus',
            'number_of_years_of_education',
            'hu_med_val',
            #'dlrs_kwh',
            #'avg_monthly_consumption_kwh',
            'education_bachelor',
            'education_population',
            'hdd',
            #'hu_mortgage',
            'hu_own',
            'cdd',
            'transportation_car_alone_rate',
            'fam_med_income',
            'travel_time_average',
            'housing_unit_count',
            'poverty_family_below_poverty_level_rate',
            'occupation_finance_rate',
            #'travel_time_10_19_rate',
            'hu_vintage_1960to1970',
            'hu_1959toearlier_pct',
            #'voting_2012_dem_percentage',
            #'voting_2012_gop_percentage',
            'age_55_64_rate',
        ]

        dflt_nrml = dflt

        topp = ['Adoption', 'Ren', 'dlrs_kwh', 'net_metering', ]

        my_sets = {
            'default': dflt,
            'defaultA': dfltA,
            'defaultB': dfltB,
            'defaultC': dfltC,
            'scld_default': dflt,
            'nrml_default': dflt_nrml,
            'top': topp}

        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set

        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model


    class _demographics(_BG_):
        dflt = ['Adoption', 'number_of_years_of_education', 'education_less_than_high_school_rate',
                'education_bachelor_scld', 'education_less_than_high_school_rate',
                'education_high_school_graduate_rate',
                'education_bachelor', 'education_bachelor_rate', 'education_master_scld',
                'education_master_rate',
                'education_doctoral_rate', 'masters_or_above_rate', 'bachelor_or_above_rate',
                'high_school_or_below_rate',
                'education_population', 'age_55_64_rate', 'age_65_74_rate', 'age_75_84_rate',
                'age_more_than_85_rate',
                'age_25_34_rate', 'age_median', 'fam_med_income', 'median_household_income',
                'average_household_income_scld',
                'average_household_income', 'diversity', 'pop_female', '%female', '%male', 'Anti_Occup',
                'Pro_Occup',
                'employ_rate', 'voting_2012_dem_percentage', 'voting_2012_gop_percentage', 'hu_own',
                'hu_own_pct',
                'hh_size_1', 'hh_size_2', 'hh_size_3', 'hh_size_4', '%hh_size_1', '%hh_size_2', '%hh_size_4',
                'education_population_scld', 'hh_total', 'employ_rate', '%hh_size_2',
                'high_school_or_below_rate', 'average_household_size', 'average_household_size_scld']
        nrml_dflt =['Adoption', 'number_of_years_of_education_nrml_', 'education_less_than_high_school_rate_nrml_',
                   'education_bachelor_nrml_','education_high_school_graduate_rate_nrml_', 'education_bachelor_nrml_',
                   'education_bachelor_rate_nrml_', 'education_master_nrml_', 'education_master_rate_nrml_',
                   'education_doctoral_rate_nrml_', 'masters_or_above_rate_nrml_', 'bachelor_or_above_rate_nrml_',
                #   'high_school_or_below_rate_nrml_', 'education_population_nrml_', 'age_55_64_rate_nrml_',
                #   'age_65_74_rate_nrml_', 'age_75_84_rate_nrml_', 'age_more_than_85_rate_nrml_',
                #   'age_25_34_rate_nrml_', 'age_median_nrml_', 'fam_med_income_nrml_', 'median_household_income_nrml_',
                #   'average_household_income_nrml_', 'average_household_income_nrml_', 'diversity_nrml_',
                #   'pop_female_nrml_', '%female_nrml_', '%male_nrml_', 'Anti_Occup_nrml_', 'Pro_Occup_nrml_',
                #   'employ_rate_nrml_', 'voting_2012_dem_percentage_nrml_', 'voting_2012_gop_percentage_nrml_',
                #   'hu_own_nrml_', 'hu_own_pct_nrml_', 'hh_size_1_nrml_', 'hh_size_2_nrml_', 'hh_size_3_nrml_',
                #   'hh_size_4_nrml_', '%hh_size_1_nrml_', '%hh_size_2_nrml_', '%hh_size_4_nrml_', 'education_population_nrml_',
                   'hh_total_nrml_', 'employ_rate_nrml_', '%hh_size_2_nrml_', 'high_school_or_below_rate_nrml_', 'average_household_size_nrml_', ]
        scaled_dflt = ['Adoption', 'number_of_years_of_education', 'education_less_than_high_school_rate',
                       'education_bachelor_scld', 'education_less_than_high_school_rate',
                       'education_high_school_graduate_rate',
                       'education_bachelor', 'education_bachelor_rate', 'education_master_scld',
                       'education_master_rate',
                       'education_doctoral_rate', 'masters_or_above_rate', 'bachelor_or_above_rate',
                       'high_school_or_below_rate',
                       'education_population', 'age_55_64_rate', 'age_65_74_rate', 'age_75_84_rate',
                       'age_more_than_85_rate',
                       'age_25_34_rate',
                       'average_household_income_scld',
                       'diversity', '%female', '%male', 'Anti_Occup',
                       'Pro_Occup',
                       'employ_rate', 'voting_2012_dem_percentage', 'voting_2012_gop_percentage', 'hu_own',
                       'hu_own_pct',
                       '%hh_size_1', '%hh_size_2', '%hh_size_4',
                       'education_population_scld', 'hh_total', 'employ_rate', '%hh_size_2',
                       'high_school_or_below_rate', 'average_household_size_scld']
        toppA = ['Anti_Occup', 'diversity', 'age_65_74_rate', 'education_bachelor_scld', 'education_master_scld',
                 'education_high_school_graduate_rate', 'hh_total', 'employ_rate']
        toppB = ['Anti_Occup', 'diversity', 'age_65_74_rate', 'education_bachelor_scld', 'education_master_scld',
                 'education_high_school_graduate_rate', 'hh_total', 'average_household_size_scld']
        toppC = ['Anti_Occup', 'diversity', 'age_65_74_rate', 'education_bachelor_scld', 'education_master_scld',
                 'education_high_school_graduate_rate', 'hh_total', 'average_household_size_scld',
                 'bachelor_or_above_rate', 'hu_own_pct', ]
        toppD = ['Anti_Occup', 'diversity', 'age_65_74_rate', 'education_bachelor_scld', 'education_master_scld',
                 'hh_total', 'average_household_size_scld',
                 'bachelor_or_above_rate', ]
        toppE = ['Anti_Occup', 'diversity', 'age_65_74_rate', 'education_bachelor_scld', 'education_master_scld',
                 'hh_total', 'average_household_income_scld',
                 'bachelor_or_above_rate', ]
        toppF = ['Anti_Occup', 'diversity', '%female',
                 'hh_total', 'average_household_income_scld',
                 'bachelor_or_above_rate', ]

        my_sets = {
            'default': dflt,
            'scld_default': scaled_dflt,
            'nrml_default': nrml_dflt,
            'topA': toppA,
            'topB': toppB,
            'topC': toppC,
            'topD': toppD,
            'topE': toppE,
            'topF': toppF,
        }

        def __init__(self, set=None, ):
            Block_Groups._BG_.__init__(self, my_sets=self.my_sets, set=set, dflt=(self.dflt,))
            self.set = set

        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    # TODO: Add the below to big set list
    class _population_features_(_BG_):
        dflt = [
            'Adoption',
            'pop_total',
            'household_count',
            'hh_total',
            ]
        my_sets = {'default': dflt,}
        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set
        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _household_energy_use_(_BG_):
        dflt = [
                'Adoption',
                'heating_fuel_coal_coke_rate',
                'heating_fuel_electricity_rate',
                'cooling_design_temperature',
                'heating_design_temperature',
                'avg_monthly_bill_dlrs',
                'avg_monthly_consumption_kwh',
            ]
        my_sets = {'default': dflt,}
        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set
        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _dwelling_characteristics_(_BG_):
        dflt = ['Adoption',
                'hu_vintage_1939toearlier',
                'hu_1959toearlier_pct',
                'hu_vintage_1940to1959',
                'hu_vintage_1960to1970',
                'hu_vintage_1980to1999',
                'hu_vintage_2000to2009',
                'hu_vintage_2010toafter',
                'hu_2000toafter',
                'hu_med_val',
                'hu_mortgage',
                ]
        my_sets = {'default': dflt, }
        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set
        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _income_employment_homeownership_(_BG_):
        dflt = ['Adoption',
                'hh_gini_index',
                'hu_monthly_owner_costs_greaterthan_1000dlrs',
                'hu_monthly_owner_costs_lessthan_1000dlrs',
                'fam_med_income',
                'median_household_income',
                'average_household_income',
                'hu_own',
                'mortgage_with_rate',
                'hu_own_pct',
                'low_mf_own_hh',
                'low_sf_own_hh',
                'high_sf_own_hh',
                'high_mf_own_hh',
                'mid_mf_own_hh',
                'mid_sf_own_hh',
                'very_low_mf_own_hh',
                'very_low_sf_own_hh',
                'very_low_sf_own_bldg_cnt',
                'mod_sf_own_bldg_cnt',
                'mod_mf_own_bldg_cnt',
                'mid_mf_own_bldg_cnt',
                'low_sf_own_bldg_cnt',
                'low_mf_own_bldg_cnt',
                'high_sf_own_bldg_cnt',
                'high_mf_own_bldg_cnt',
                ]
        my_sets = {'default': dflt, }
        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set
        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _Generation_Potential_(_BG_):
        dflt = ['Adoption',
                'high_mf_own_mw',
                'high_mf_own_mwh',
                'high_sf_own_mw',
                'high_sf_own_mwh',
                'low_mf_own_mw',
                'low_mf_own_mwh',
                'low_sf_own_mw',
                'low_sf_own_mwh',
                'mid_mf_own_mw',
                'mid_mf_own_mwh',
                'very_low_mf_own_mw',
                'very_low_mf_own_mwh',
                'very_low_sf_own_mw',
                'very_low_sf_own_mwh',
                'mod_sf_own_devp_cnt',
                'mod_sf_own_devp_m2',
                'mod_sf_own_elep_hh',
                'very_low_sf_own_devp_cnt',
                'very_low_sf_own_devp_m2',
                'very_low_sf_own_elep_hh',
                'mod_mf_own_devp_cnt',
                'mod_mf_own_devp_m2',
                'mod_mf_own_elep_hh',
                'mid_sf_own_devp_cnt',
                'mid_sf_own_devp_m2',
                'mid_mf_own_devp_cnt',
                'mid_mf_own_devp_m2',
                'low_sf_own_devp_cnt',
                'low_sf_own_devp_m2',
                'low_sf_own_elep_hh',
                'low_mf_own_devp_cnt',
                'low_mf_own_devp_m2',
                'low_mf_own_elep_hh',
                'high_sf_own_devp_cnt',
                'high_sf_own_devp_m2',
                'high_sf_own_elep_hh',
                'high_mf_own_devp_cnt',
                'high_mf_own_devp_m2',
                'high_mf_own_elep_hh',
                ]
        my_sets = {'default': dflt, }
        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set
        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _gender_(_BG_):
        dflt = ['Adoption',
                'pop_female',
                'pop_male',
                'female_pct',
                'male_pct',
                ]
        my_sets = {'default': dflt, }

        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set

        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _political_affiliation_(_BG_):
        dflt = ['Adoption',
                'voting_2012_dem_percentage',
                'voting_2012_gop_percentage',
                ]
        my_sets = {'default': dflt, }
        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set
        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _household_size_(_BG_):
        dflt = ['Adoption',
                'hh_size_1',
                'hh_size_2',
                'hh_size_3',
                'hh_size_4',
                '%hh_size_1',
                '%hh_size_2',
                '%hh_size_3',
                '%hh_size_4',
                ]
        my_sets = {'default': dflt, }

        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set

        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class __(_BG_):
        dflt = ['Adoption',

                ]
        my_sets = {'default': dflt, }
        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set
        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _habit_(_BG_):
        dflt =  ['Adoption',
                 'Green_Travelers',
                 #'avg_monthly_bill_dlrs',
                 #'avg_monthly_consumption_kwh',
                 'travel_time_40_59_rate',
                 'travel_time_10_19_rate',
                 'travel_time_20_29_rate',
                 'travel_time_60_89_rate',
                 'travel_time_40_89_rate',
                 'transportation_home_rate',
                 'travel_time_30_39_rate',
                 'travel_time_average',
                 'travel_time_less_than_10_rate',
                 'transportation_bicycle_rate',
                 'transportation_car_alone_rate',
                 'transportation_carpool_rate',
                 'transportation_motorcycle_rate',
                 'transportation_public_rate',
                 'transportation_walk_rate',
                 ]

        nrml_dflt = dflt

        topp = ['Green_Travelers', 'avg_monthly_bill_dlrs', 'avg_monthly_consumption_kwh',
                'travel_time_40_59_rate', 'travel_time_10_19_rate', 'travel_time_20_29_rate',
                'travel_time_60_89_rate', 'travel_time_49_89_rate', 'transportation_home_rate',
                'travel_time_30_39_rate', 'travel_time_average', 'travel_time_less_than_10_rate',
                'transportation_bicycle_rate', 'transportation_car_alone_rate', 'transportation_carpool_rate',
                'transportation_motorcycle_rate', 'transportation_public_rate', 'transportation_walk_rate',
                ]
        toppA = ['Green_Travelers', 'avg_monthly_bill_dlrs', 'avg_monthly_consumption_kwh',
                 'travel_time_40_59_rate', 'travel_time_10_19_rate', 'travel_time_20_29_rate',
                 'travel_time_60_89_rate', 'travel_time_49_89_rate', 'transportation_home_rate',
                 'travel_time_30_39_rate', 'travel_time_average', 'travel_time_less_than_10_rate',
                 'transportation_bicycle_rate', 'transportation_car_alone_rate', 'transportation_carpool_rate',
                 'transportation_motorcycle_rate', 'transportation_public_rate', 'transportation_walk_rate',
                 ]

        toppB = [
            'Green_Travelers', 'avg_monthly_bill_dlrs', 'avg_monthly_consumption_kwh',
            'travel_time_49_89_rate',
            'travel_time_60_89_rate',
            'travel_time_average',
            'transportation_carpool_rate',
        ]
        toppC = [
            'Green_Travelers', 'avg_monthly_bill_dlrs', 'avg_monthly_consumption_kwh',
            'travel_time_49_89_rate', 'travel_time_average', 'transportation_carpool_rate',
        ]
        toppD = [
            'Green_Travelers', 'avg_monthly_bill_dlrs', 'avg_monthly_consumption_kwh',
            'travel_time_average', 'transportation_carpool_rate',
        ]
        toppE = [
            'Green_Travelers', 'avg_monthly_bill_dlrs', 'travel_time_49_89_rate',
            'travel_time_average', 'transportation_carpool_rate',
        ]

        my_sets = {'default': dflt,
                   'nrml_default':nrml_dflt,
                   'top': topp,
                   'topA': toppA,
                   'topB': toppB,
                   'topC': toppC,
                   'topD': toppD,
                   'topE': toppE,
                   }

        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set

        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _physical_(_BG_):
        dflt = ['Adoption',
                'heating_fuel_coal_coke_rate',
                'heating_fuel_electricity_rate',
                'hu_vintage_1939toearlier',
                'hu_vintage_1940to1959',
                'hu_vintage_1960to1970',
                'hu_vintage_1980to1999',
                'hu_vintage_2000to2009',
                'hu_vintage_2010toafter',
                'hu_1959toearlier',
                'hu_2000toafter',
                'hu_1959toearlier_pct',
                'hu_2000toafter_pct',
                'household_count',
                'hu_monthly_owner_costs_lessthan_1000dlrs',
                'hu_monthly_owner_costs_greaterthan_1000dlrs',
                'hu_med_val',
                'heating_fuel_fuel_oil_kerosene_rate',
                'housing_unit_count',
                ]

        nrml_dflt = dflt
        topp = ['heating_fuel_coal_coke_rate', 'hu_vintage_1939toearlier',
                'hu_med_val_scld', 'hu_mortgage', 'heating_fuel_fuel_oil_kerosene_rate',
                ]
        toppA = ['heating_fuel_coal_coke_rate', 'hu_1959toearlier_scld',
                 'hu_monthly_owner_costs_greaterthan_1000dlrs_scld',
                 'hu_med_val_scld', 'hu_mortgage', 'hu_vintage_1939toearlier',
                 ]
        toppB = ['heating_fuel_coal_coke_rate', 'hu_1959toearlier_scld',
                 'hu_med_val_scld', 'hu_mortgage',
                 ]
        my_sets = {
                   'default': dflt,
                   'nrml_default': nrml_dflt,
                   'top': topp,
                   'topA': toppA,
                   'topB': toppB,
                   }

        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set

        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _climate_(_BG_):
        dflt = ['Adoption',
                'hdd',
                'cdd_std',
                'cdd',
                'daily_solar_radiation',
                'hdd_std',
                ]

        nrml_dflt = ['Adoption', 'cooling_design_temperature_nrml_',  'heating_design_temperature_nrml_',
                     'hdd_nrml_', 'cdd_std_nrml_', 'cdd_nrml_',
                    'daily_solar_radiation_nrml_', 'hdd_std_nrml_',
                   ]
        topp = ['daily_solar_radiation', 'climate_zone', 'hdd_std_scld', 'cooling_design_temperature_scld', ]
        toppA = ['daily_solar_radiation', 'hdd_scld', 'cdd_scld', ]
        toppB = ['daily_solar_radiation', 'hdd_std_scld', 'cooling_design_temperature_scld', ]
        toppC = ['daily_solar_radiation', 'hdd_std_scld', 'cooling_design_temperature_scld', ]

        my_sets = {'default': dflt,
                   'nrml_default':nrml_dflt,
                   'top': topp,
                   'topA': toppA,
                   'topB': toppB
                   }

        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set

        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _geography_(_BG_):
        dflt = ['Adoption',
                'land_area',
                'locale_dummy',
                'total_area',
                'locale_recode(rural)',
                'locale_recode(suburban)',
                'locale_recode(town)',
                ]

        nrml_dflt = [
            'Adoption',
            'land_area_nrml_',
            'locale_dummy_nrml_',
            #'locale_dummy_2_nrml_',
            #'locale_dummy_3_nrml_',
            #'locale_dummy_4_nrml_',
            'total_area_nrml_',
        ]
        topp = ['land_area_scld', 'locale_dummy', 'total_area_scld']
        toppA = ['locale_dummy', 'total_area_scld']

        my_sets = {
                    'default': dflt,
                    'nrml_default': nrml_dflt,
                    'top': topp,
                    'topA': toppA,
                   }

        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set

        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _Xu_Models_(_BG_):


        stripped =  ['Adoption',
                     'housing_unit_count',
                     'population_density',
                     'land_area',
                     'diversity',
                     'Green_Travelers',
                     'heating_fuel_electricity_rate',
                     'heating_fuel_coal_coke_rate',
                     'locale_recode(rural)',
                     'travel_time_40_89_rate',
                     'hdd',
                     'voting_2012_dem_percentage',
                     'hu_own_pct',
                     'locale_recode(suburban)',
                     'hu_1959toearlier_pct',      #***
                     'age_55_or_more_rate',
                     'female_pct',
                   ]

        strippedB = ['Adoption',
                    'housing_unit_count',
                    'population_density',
                    'land_area',
                    'diversity',
                    'Green_Travelers',
                    'heating_fuel_electricity_rate',
                    'heating_fuel_coal_coke_rate',
                    'locale_recode(rural)',
                    'travel_time_40_89_rate',
                    'hdd',
                    'voting_2012_dem_percentage',
                    'hu_own_pct',
                    'locale_recode(suburban)',
                    'hu_1959toearlier_pct',  # ***
                    'age_55_or_more_rate',
                    'female_pct',
                     'travel_time_less_than_10_rate',
                     'transportation_bicycle_rate',
                     'transportation_carpool_rate',
                     'transportation_public_rate',
                     'transportation_walk_rate',
                     'transportation_home_rate',
                     ]

        strippedC = ['Adoption',
                     'housing_unit_count',
                     'population_density',
                     'land_area',
                     'diversity',
                     'Green_Travelers',
                     'heating_fuel_electricity_rate',
                     'heating_fuel_coal_coke_rate',
                     'locale_recode(rural)',
                     'locale_recode(city)',
                     'travel_time_40_89_rate',
                     'hdd',
                     'voting_2012_dem_percentage',
                     'hu_own_pct',
                     'locale_recode(suburban)',
                     'hu_1959toearlier_pct',  # ***
                     'age_55_or_more_rate',
                     'female_pct',
                     'travel_time_less_than_10_rate',
                     'transportation_bicycle_rate',
                     'transportation_carpool_rate',
                     'transportation_public_rate',
                     'transportation_walk_rate',
                     'transportation_home_rate',
                     ]

        TVA_rep = ['Adoption',
                   'dlrs_kwh',
                   'number_of_years_of_education',
                   'education_high_school_or_below_rate',
                   'education_master_or_above_rate',
                   'median_household_income',
                   'employ_rate',
                   'female_pct',
                   'voting_2012_dem_percentage',
                   'hu_own_pct',
                   'diversity',
                   'age_55_or_more_rate',
                   'population_density',
                   'housing_unit_count',
                   '%hh_size_4',
                   'land_area',
                   'locale_recode(rural)',
                   'locale_recode(suburban)',
                   'locale_recode(town)',
                   'locale_dummy',
                   'hdd',
                   'heating_fuel_electricity_rate',
                   'heating_fuel_coal_coke_rate',
                   'hu_1959toearlier_pct',      #***
                   'hu_2000toafter_pct', #***
                   'Green_Travelers',
                   'travel_time_40_89_rate',
                   'avg_monthly_consumption_kwh',  #***
                   'dlrs_kwh x median_household_income', #***
                   ]
        nrml_dflt = TVA_rep
        dflt = TVA_rep
        TVA_rep_5_20 =['Adoption',
                   'dlrs_kwh',                               #
                   'number_of_years_of_education',                               #
                   'education_high_school_or_below_rate',                               #
                   'education_master_or_above_rate',                               #
                   'median_household_income',                               #
                   'employ_rate',                               #
                   'female_pct',                               #
                   'voting_2012_dem_percentage',                               #
                   'hu_own_pct',                               #
                   'diversity',                               #
                   'age_55_or_more_rate',                               #
                   'population_density',                               #
                   'housing_unit_count',                               #
                   '%hh_size_4',                               #
                   'land_area',                               #
                   'locale_dummy',                               #
                   'hdd',                               #
                   'heating_fuel_electricity_rate',                               #
                   'heating_fuel_coal_coke_rate',                               #
                   'hu_1959toearlier_pct',      #***                               #
                   'hu_2000toafter_pct', #***                               #
                   'Green_Travelers',                               #
                   'travel_time_40_89_rate',                               #
                   'avg_monthly_consumption_kwh',  #***                               #
                   'dlrs_kwh x median_household_income', #***                               #
                   ]
        topp = []

        my_sets = {'default': TVA_rep_5_20,
                   'top': topp,
                   'TVA_rep':TVA_rep,
                   'report_20':TVA_rep_5_20,
                   'stripped':stripped,
                   'strippedB': strippedB,
                   'strippedC': strippedC,
        }

        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set

        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _education_(_BG_):
                #'age_55_64_rate',                      'age_65_74_rate', 'age_75_84_rate',
                #'age_more_than_85_rate',
                #'age_25_34_rate', 'age_median',        'fam_med_income', 'median_household_income',
                #'average_household_income_scld',
                #'average_household_income', 'diversity', 'pop_female', '%female', '%male', 'Anti_Occup',
                #'Pro_Occup',
                #'employ_rate', 'voting_2012_dem_percentage', 'voting_2012_gop_percentage', 'hu_own',
                #'hu_own_pct',
                #'hh_size_1', 'hh_size_2', 'hh_size_3', 'hh_size_4', '%hh_size_1', '%hh_size_2', '%hh_size_4',
                #'education_population_scld', 'hh_total', 'employ_rate', '%hh_size_2',
                #'high_school_or_below_rate', 'average_household_size', 'average_household_size_scld']
                #"""
        nrml_dflt = ['Adoption',
                'education_less_than_high_school_rate',
                'education_high_school_graduate_rate',
                'number_of_years_of_education',
                'education_bachelor',
                'education_bachelor_rate',
                'education_master_rate',
                'education_doctoral_rate',
                'education_master_or_above_rate',
                'education_bachelor_or_above_rate',
                'education_population',
                'education_professional_school',
                'education_professional_school_rate',
                'pop25_some_college_plus',
                ]

        dflt = nrml_dflt

        scaled_dflt = ['Adoption', 'number_of_years_of_education', 'education_less_than_high_school_rate',
                       'education_bachelor_scld', 'education_less_than_high_school_rate',
                       'education_high_school_graduate_rate',
                       'education_bachelor', 'education_bachelor_rate', 'education_master_scld',
                       'education_master_rate',
                       'education_doctoral_rate', 'masters_or_above_rate', 'bachelor_or_above_rate',
                       'high_school_or_below_rate',
                       'education_population', 'age_55_64_rate', 'age_65_74_rate', 'age_75_84_rate',
                       'age_more_than_85_rate',
                       'age_25_34_rate',
                       'average_household_income_scld',
                       'diversity', '%female', '%male', 'Anti_Occup',
                       'Pro_Occup',
                       'employ_rate', 'voting_2012_dem_percentage', 'voting_2012_gop_percentage', 'hu_own',
                       'hu_own_pct',
                       '%hh_size_1', '%hh_size_2', '%hh_size_4',
                       'education_population_scld', 'hh_total', 'employ_rate', '%hh_size_2',
                       'high_school_or_below_rate', 'average_household_size_scld']
        toppA = ['Anti_Occup', 'diversity', 'age_65_74_rate', 'education_bachelor_scld', 'education_master_scld',
                 'education_high_school_graduate_rate', 'hh_total', 'employ_rate']
        toppB = ['Anti_Occup', 'diversity', 'age_65_74_rate', 'education_bachelor_scld', 'education_master_scld',
                 'education_high_school_graduate_rate', 'hh_total', 'average_household_size_scld']
        'bachelor_or_above_rate'
        toppC = ['Anti_Occup', 'diversity', 'age_65_74_rate', 'education_bachelor_scld', 'education_master_scld',
                 'education_high_school_graduate_rate', 'hh_total', 'average_household_size_scld',
                 'bachelor_or_above_rate', 'hu_own_pct', ]
        toppD = ['Anti_Occup', 'diversity', 'age_65_74_rate', 'education_bachelor_scld', 'education_master_scld',
                 'hh_total', 'average_household_size_scld',
                 'bachelor_or_above_rate', ]
        toppE = ['Anti_Occup', 'diversity', 'age_65_74_rate', 'education_bachelor_scld', 'education_master_scld',
                 'hh_total', 'average_household_income_scld',
                 'bachelor_or_above_rate', ]
        toppF = ['Anti_Occup', 'diversity', '%female',
                 'hh_total', 'average_household_income_scld',
                 'bachelor_or_above_rate', ]

        my_sets = {
            'default': dflt,
            'nrml_default': nrml_dflt,
        }
        def __init__(self, set=None, ):
            Block_Groups._BG_.__init__(self, my_sets=self.my_sets, set=set, dflt=(self.dflt,))
            self.set = set

        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _age_(_BG_):
        nrml_dflt = [
                     'Adoption',
                     'age_55_64_rate',
                     'age_65_74_rate',
                     'age_75_84_rate',
                     'age_more_than_85_rate',
                     'age_25_34_rate',
                     'age_median',
                     'age_55_or_more_rate',
                     ]
        dflt = nrml_dflt
        my_sets = {
            'default': dflt,
            'nrml_default': nrml_dflt,
        }

        def __init__(self, set=None, ):
            Block_Groups._BG_.__init__(self, my_sets=self.my_sets, set=set, dflt=(self.dflt,))
            self.set = set

        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _hh_(_BG_):
        nrml_dflt = [
                     'Adoption',
                     'hu_own',
                     'hu_mortgage',
                     'hu_own_pct',
                     'hh_size_1',
                     'hh_size_2',
                     'hh_size_3',
                     'hh_size_4',
                     '%hh_size_1',
                     '%hh_size_2',
                     '%hh_size_3',
                     '%hh_size_4',
                     'hh_total',
                     ]
        dflt = nrml_dflt
        my_sets = {
            'default': dflt,
            'nrml_default': nrml_dflt,
        }

        def __init__(self, set=None, ):
            Block_Groups._BG_.__init__(self, my_sets=self.my_sets, set=set, dflt=(self.dflt,))
            self.set = set

        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _CT_demos_(_BG_):
        nrml_dflt = [
                     'Adoption',
                     'fam_med_income',
                     'median_household_income',
                     'average_household_income',
                     'diversity',
                     'pop_female','pop_male',
                     'female_pct', 'male_pct',
                     'Anti_Occup',
                     'Pro_Occup',
                     'employ_rate',
                     'voting_2012_dem_percentage', 'voting_2012_gop_percentage',
                     #'voting_2016_dem_percentage_nrml_', 'voting_2016_gop_percentage_nrml_',
                     #'hu_own_nrml_',
                     #'hu_own_pct_nrml_',
                     #'hh_size_1_nrml_', 'hh_size_2_nrml_', 'hh_size_3_nrml_', 'hh_size_4_nrml_',
                     #'%hh_size_1_nrml_', '%hh_size_2_nrml_', '%hh_size_3_nrml_', '%hh_size_4_nrml_',
                     #'hh_total_nrml_',
                     ]
        dflt = nrml_dflt
        my_sets = {
            'default': dflt,
            'nrml_default': nrml_dflt,
        }

        def __init__(self, set=None, ):
            Block_Groups._BG_.__init__(self, my_sets=self.my_sets, set=set, dflt=(self.dflt,))
            self.set = set

        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _income_(_BG_):
        nrml_dflt = [
                     'Adoption',
                     'fam_med_income',
                     'median_household_income',
                     'average_household_income',
                     ]
        dflt = [
                 'Adoption',
                 'fam_med_income_nrml_',
                 'median_household_income_nrml_',
                 'average_household_income_nrml_',
                   ]
        my_sets = {
            'default': dflt,
            'nrml_default': nrml_dflt,
        }

        def __init__(self, set=None, ):
            Block_Groups._BG_.__init__(self, my_sets=self.my_sets, set=set, dflt=(self.dflt,))
            self.set = set

        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _All_(_BG_):
        dflt = list_name = [
        "%hh_size_1",
        "%hh_size_2",
        "%hh_size_3",
        "%hh_size_4",
        "Anti_Occup",
        "Gender_Ratio",
        "Green_Travelers",
        "Green_Travelers_log10",
        "Pro_Occup",
        "Adoption",
        #"AvgSres",
        #"PV_HuOwn",
        #"PV_per_100_HuOwn",
        #"SNRaPa",
        #"SNRaPcap",
        #"SNRpcap",
        #"SRaPa",
        #"SRaPcap",
        #"SRpcap",
        #"ST_pcap",
        "Tot_own_mw",
        "Yr_own_mwh",
        "Yr_own_mwh_log10",
        "Yrl_%_$_kwh",
        "Yrl_%_inc",
        "Yrl_savings_$",
        "Yrl_savings_$_log10",
        "active_subsidies",
        "age_10_14_rate",
        "age_15_17_rate",
        "age_18_24_rate",
        "age_25_34_rate",
        "age_25_44_rate",
        "age_25_64_rate",
        "age_35_44_rate",
        "age_45_54_rate",
        "age_55_64_rate",
        "age_55_or_more_rate",
        "age_5_9_rate",
        "age_65_74_rate",
        "age_65_74_rate_log10",
        "age_75_84_rate",
        "age_median",
        "age_median_log10",
        "age_minor_rate",
        "age_minor_rate_log10",
        "age_more_than_85_rate",
        "age_zoomer_rate",
        "al",
        "aqi_90th_percentile",
        "aqi_90th_percentile_description",
        "aqi_max",
        "aqi_max_description",
        "aqi_median",
        "aqi_median_description",
        "ar",
        "area_km2",
        "asian_pct",
        "average_household_income",
        "average_household_size",
        "average_household_size_log10",
        "avg_cbi_usd_p_w",
        "avg_electricity_retail_rate",
        "avg_ibi_pct",
        "avg_inc_ebill_dlrs",
        "avg_monthly_bill_dlrs",
        "avg_monthly_bill_dlrs_log10",
        "avg_monthly_consumption_kwh",
        "avg_monthly_consumption_kwh_log10",
        "avg_months_tenancy",
        "avg_pbi_usd_p_kwh",
        "az",
        "black_pct",
        "ca",
        "cdd",
        "cdd_ci",
        "cdd_log10",
        "cdd_std",
        "cdd_std_log10",
        "centroid_x",
        "climate_zone",
        "climate_zone_description",
        "co",
        "company_na",
        "company_ty",
        "cooling_design_temperature",
        "county_fips",
        "county_name",
        "ct",
        "cust_cnt",
        "daily_solar_radiation",
        "daily_solar_radiation_log10",
        "dc",
        "de",
        "diversity",
        "dlrs_kwh",
        "dlrs_kwh x median_household_income",
        "educated_population_rate",
        "education_bachelor",
        "education_bachelor_or_above_rate",
        "education_bachelor_rate",
        "education_college",
        "education_college_rate",
        "education_doctoral",
        "education_doctoral_rate",
        "education_high_school_graduate",
        "education_high_school_graduate_rate",
        "education_high_school_or_below_rate",
        "education_less_than_high_school",
        "education_less_than_high_school_rate",
        "education_master",
        "education_master_or_above_rate",
        "education_master_rate",
        "education_population",
        "education_professional_school",
        "education_professional_school_rate",
        "eia_id",
        "employ_rate",
        "fam_children_6to17",
        "fam_children_under_6",
        "fam_med_income",
        "female_pct",
        "fips",
        "fl",
        "fmr_2br",
        "ga",
        "geoid",
        "gisjoin",
        "hdd",
        "hdd_ci",
        "hdd_log10",
        "hdd_std",
        "hdd_std_log10",
        "heating_design_temperature",
        "heating_fuel_coal_coke",
        "heating_fuel_coal_coke_rate",
        "heating_fuel_coal_coke_rate_log10",
        "heating_fuel_electricity",
        "heating_fuel_electricity_rate",
        "heating_fuel_electricity_rate_log10",
        "heating_fuel_fuel_oil_kerosene",
        "heating_fuel_fuel_oil_kerosene_rate",
        "heating_fuel_gas",
        "heating_fuel_gas_rate",
        "heating_fuel_housing_unit_count",
        "heating_fuel_none",
        "heating_fuel_none_rate",
        "heating_fuel_other",
        "heating_fuel_other_rate",
        "heating_fuel_solar",
        "heating_fuel_solar_rate",
        "hh_gini_index",
        "hh_med_income",
        "hh_size_1",
        "hh_size_2",
        "hh_size_3",
        "hh_size_4",
        "hh_total",
        "high_hh_rate",
        "high_mf_own_bldg_cnt",
        "high_mf_own_devp_cnt",
        "high_mf_own_devp_m2",
        "high_mf_own_elep_hh",
        "high_mf_own_hh",
        "high_mf_own_mw",
        "high_mf_own_mwh",
        "high_mf_rent_bldg_cnt",
        "high_mf_rent_devp_cnt",
        "high_mf_rent_devp_m2",
        "high_mf_rent_elep_hh",
        "high_mf_rent_hh",
        "high_mf_rent_mw",
        "high_mf_rent_mwh",
        "high_own_Sbldg",
        "high_own_Sbldg_rt",
        "high_own_devp",
        "high_own_devp_log10",
        "high_own_elep_hh",
        "high_own_hh",
        "high_own_mwh",
        "high_own_mwh_log10",
        "high_sf_own_bldg_cnt",
        "high_sf_own_devp_cnt",
        "high_sf_own_devp_m2",
        "high_sf_own_elep_hh",
        "high_sf_own_hh",
        "high_sf_own_mw",
        "high_sf_own_mwh",
        "high_sf_rent_bldg_cnt",
        "high_sf_rent_devp_cnt",
        "high_sf_rent_devp_m2",
        "high_sf_rent_elep_hh",
        "high_sf_rent_hh",
        "high_sf_rent_mw",
        "high_sf_rent_mwh",
        "hispanic_pct",
        "household_count",
        "household_type_family_rate",
        "housing_unit_count",
        "housing_unit_median_gross_rent",
        "housing_unit_median_value",
        "housing_unit_occupied_count",
        "hu_1959toearlier",
        "hu_1959toearlier_pct",
        "hu_1960to1979_pct",
        "hu_1980to1999_pct",
        "hu_2000toafter",
        "hu_2000toafter_pct",
        "hu_med_val",
        "hu_med_val_log10",
        "hu_monthly_owner_costs_greaterthan_1000dlrs",
        "hu_monthly_owner_costs_lessthan_1000dlrs",
        "hu_mortgage",
        "hu_no_mortgage",
        "hu_own",
        "hu_own_pct",
        "hu_rent",
        "hu_vintage_1939toearlier",
        "hu_vintage_1940to1959",
        "hu_vintage_1960to1970",
        "hu_vintage_1980to1999",
        "hu_vintage_2000to2009",
        "hu_vintage_2010toafter",
        "hydro_prod",
        "ia",
        "id",
        "il",
        "in",
        "incent_cnt_res_own",
        "incent_cnt_res_own_log10",
        "incentive_count_nonresidential",
        "incentive_count_residential",
        "incentive_count_residential_log10",
        "incentive_nonresidential_state_level",
        "incentive_residential_state_level",
        "incentive_residential_state_level_log10",
        "ks",
        "ky",
        "la",
        "land_area",
        "lihtc_qualified",
        "locale",
        "locale_dummy",
        "locale_recode",
        "locale_recode(rural)",
        "locale_recode(suburban)",
        "locale_recode(town)",
        "locale_recode(urban)",
        "low_commute_times",
        "low_commute_times_log10",
        "low_hh_rate",
        "low_mf_own_bldg_cnt",
        "low_mf_own_devp_cnt",
        "low_mf_own_devp_m2",
        "low_mf_own_elep_hh",
        "low_mf_own_hh",
        "low_mf_own_mw",
        "low_mf_own_mwh",
        "low_mf_rent_bldg_cnt",
        "low_mf_rent_devp_cnt",
        "low_mf_rent_devp_m2",
        "low_mf_rent_elep_hh",
        "low_mf_rent_hh",
        "low_mf_rent_mw",
        "low_mf_rent_mwh",
        "low_own_Sbldg",
        "low_own_Sbldg_rt",
        "low_own_devp",
        "low_own_devp_log10",
        "low_own_elep_hh",
        "low_own_hh",
        "low_own_mwh",
        "low_own_mwh_log10",
        "low_sf_own_bldg_cnt",
        "low_sf_own_devp_cnt",
        "low_sf_own_devp_m2",
        "low_sf_own_elep_hh",
        "low_sf_own_hh",
        "low_sf_own_mw",
        "low_sf_own_mwh",
        "low_sf_rent_bldg_cnt",
        "low_sf_rent_devp_cnt",
        "low_sf_rent_devp_m2",
        "low_sf_rent_elep_hh",
        "low_sf_rent_hh",
        "low_sf_rent_mw",
        "low_sf_rent_mwh",
        "ma",
        "male_pct",
        "md",
        "me",
        "med_inc_ebill_dlrs",
        "median_household_income",
        "median_household_income_log10",
        "mi",
        "mid_hh_rate",
        "mid_mf_own_bldg_cnt",
        "mid_mf_own_devp_cnt",
        "mid_mf_own_devp_m2",
        "mid_mf_own_hh",
        "mid_mf_own_mw",
        "mid_mf_own_mwh",
        "mid_mf_rent_bldg_cnt",
        "mid_mf_rent_devp_cnt",
        "mid_mf_rent_devp_m2",
        "mid_mf_rent_hh",
        "mid_mf_rent_mw",
        "mid_mf_rent_mwh",
        "mid_own_Sbldg",
        "mid_own_Sbldg_rt",
        "mid_own_devp",
        "mid_own_hh",
        "mid_own_mwh",
        "mid_sf_own_bldg_cnt",
        "mid_sf_own_devp_cnt",
        "mid_sf_own_devp_m2",
        "mid_sf_own_hh",
        "mid_sf_own_mw",
        "mid_sf_own_mwh",
        "mid_sf_rent_bldg_cnt",
        "mid_sf_rent_devp_cnt",
        "mid_sf_rent_devp_m2",
        "mid_sf_rent_hh",
        "mid_sf_rent_mw",
        "mid_sf_rent_mwh",
        "mn",
        "mo",
        "mod_hh_rate",
        "mod_mf_own_bldg_cnt",
        "mod_mf_own_devp_cnt",
        "mod_mf_own_devp_m2",
        "mod_mf_own_elep_hh",
        "mod_mf_own_hh",
        "mod_mf_own_mw",
        "mod_mf_own_mwh",
        "mod_mf_rent_bldg_cnt",
        "mod_mf_rent_devp_cnt",
        "mod_mf_rent_devp_m2",
        "mod_mf_rent_elep_hh",
        "mod_mf_rent_hh",
        "mod_mf_rent_mw",
        "mod_mf_rent_mwh",
        "mod_own_Sbldg",
        "mod_own_Sbldg_rt",
        "mod_own_devp",
        "mod_own_devp_log10",
        "mod_own_elep_hh",
        "mod_own_hh",
        "mod_own_mwh",
        "mod_own_mwh_log10",
        "mod_sf_own_bldg_cnt",
        "mod_sf_own_devp_cnt",
        "mod_sf_own_devp_m2",
        "mod_sf_own_elep_hh",
        "mod_sf_own_hh",
        "mod_sf_own_mw",
        "mod_sf_own_mwh",
        "mod_sf_rent_bldg_cnt",
        "mod_sf_rent_devp_cnt",
        "mod_sf_rent_devp_m2",
        "mod_sf_rent_elep_hh",
        "mod_sf_rent_hh",
        "mod_sf_rent_mw",
        "mod_sf_rent_mwh",
        "moisture_regime",
        "mortgage_with_rate",
        "ms",
        "mt",
        "nc",
        "nd",
        "ne",
        "net_metering",
        "net_metering_bin",
        "net_metering_hu_own",
        "net_metering_hu_own_log10",
        "nh",
        "nj",
        "nm",
        "number_of_solar_system_per_household",
        "number_of_years_of_education",
        "nv",
        "ny",
        "occ_rate",
        "occupancy_owner_rate",
        "occupation_administrative_rate",
        "occupation_agriculture_rate",
        "occupation_arts_rate",
        "occupation_construction_rate",
        "occupation_education_rate",
        "occupation_finance_rate",
        "occupation_information_rate",
        "occupation_manufacturing_rate",
        "occupation_public_rate",
        "occupation_retail_rate",
        "occupation_transportation_rate",
        "occupation_wholesale_rate",
        "oh",
        "ok",
        "or",
        "own_popden",
        "p16_employed",
        "p16_employed_log10",
        "p16_unemployed",
        "pa",
        "pct_eli_hh",
        #"political_ratio",
        "pop25_high_school",
        "pop25_no_high_school",
        "pop25_some_college_plus",
        "pop25_some_college_plus_log10",
        "pop_african_american",
        "pop_african_american_log10",
        "pop_asian",
        "pop_asian_log10",
        "pop_caucasian",
        "pop_caucasian_log10",
        "pop_female",
        "pop_hispanic",
        "pop_hispanic_log10",
        "pop_male",
        "pop_med_age",
        "pop_nat_us_citizen",
        "pop_native_american",
        "pop_non_us_citizen",
        "pop_over_65",
        "pop_total",
        "pop_under_18",
        "pop_us_citizen",
        "population",
        "population_density",
        "population_density_log10",
        "poverty_family_below_poverty_level",
        "poverty_family_below_poverty_level_rate",
        "poverty_family_count",
        "property_tax",
        "property_tax_bin",
        "property_tax_hu_own",
        "property_tax_hu_own_log10",
        "renew_prod",
        "ri",
        "sc",
        "sd",
        "solar_panel_area_divided_by_area",
        "solar_panel_area_per_capita",
        "solar_prod",
        "solar_prod_log10",
        "solar_system_count",
        "solar_system_count_nonresidential",
        "solar_system_count_residential",
        "state",
        "state_abbr",
        "state_fips",
        "state_name",
        "tn",
        "total_area",
        "total_area_log10",
        "total_mf_own_hh",
        "total_own_Sbldg",
        "total_own_Sbldg_log10",
        "total_own_devp",
        "total_own_devp_log10",
        "total_own_elep",
        "total_own_hh",
        "total_panel_area",
        "total_panel_area_nonresidential",
        "total_panel_area_residential",
        "total_sf_own_hh",
        "total_units",
        "tract_fips",
        "transportation_bicycle_rate",
        "transportation_bicycle_rate_log10",
        "transportation_car_alone_rate",
        "transportation_carpool_rate",
        "transportation_home_rate",
        "transportation_home_rate_log10",
        "transportation_motorcycle_rate",
        "transportation_public_rate",
        "transportation_walk_rate",
        "transportation_walk_rate_log10",
        "travel_time_10_19_rate",
        "travel_time_10_19_rate_log10",
        "travel_time_20_29_rate",
        "travel_time_30_39_rate",
        "travel_time_40_59_rate",
        "travel_time_40_89_rate",
        "travel_time_40_89_rate_log10",
        "travel_time_60_89_rate",
        "travel_time_average",
        "travel_time_less_than_10_rate",
        "travel_time_less_than_10_rate_log10",
        "tx",
        "ut",
        "va",
        "very_low_hh_rate",
        "very_low_mf_own_bldg_cnt",
        "very_low_mf_own_devp_cnt",
        "very_low_mf_own_devp_m2",
        "very_low_mf_own_elep_hh",
        "very_low_mf_own_hh",
        "very_low_mf_own_mw",
        "very_low_mf_own_mwh",
        "very_low_mf_rent_bldg_cnt",
        "very_low_mf_rent_devp_cnt",
        "very_low_mf_rent_devp_m2",
        "very_low_mf_rent_elep_hh",
        "very_low_mf_rent_hh",
        "very_low_mf_rent_mw",
        "very_low_mf_rent_mwh",
        "very_low_sf_own_bldg_cnt",
        "very_low_sf_own_devp_cnt",
        "very_low_sf_own_devp_m2",
        "very_low_sf_own_elep_hh",
        "very_low_sf_own_hh",
        "very_low_sf_own_mw",
        "very_low_sf_own_mwh",
        "very_low_sf_rent_bldg_cnt",
        "very_low_sf_rent_devp_cnt",
        "very_low_sf_rent_devp_m2",
        "very_low_sf_rent_elep_hh",
        "very_low_sf_rent_hh",
        "very_low_sf_rent_mw",
        "very_low_sf_rent_mwh",
        "verylow_own_Sbldg",
        "verylow_own_Sbldg_rt",
        "verylow_own_devp",
        "verylow_own_elep_hh",
        "verylow_own_hh",
        "verylow_own_mwh",
        "vt",
        "wa",
        "white_pct",
        "wi",
        "wv",
        "wy",
        ]
        nrml_dflt = dflt
        topp = ['pop_total_scld', 'population_density_scld',  'household_count_scld', ]
        toppA = ['population_density_scld', 'E_DAYPOP_scld', 'household_count_scld', ]

        my_sets = {
                   'default': dflt,
                   }
        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set

        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model
    class _population_(_BG_):
        dflt = [
            'Adoption',
            'pop_total',
            'pop_under_18',
            'hh_total',
            'population_density',
            'household_count',
            'housing_unit_count',
        ]
        nrml_dflt = dflt
        topp = ['pop_total_scld', 'population_density_scld', 'household_count_scld', ]
        toppA = ['population_density_scld', 'E_DAYPOP_scld', 'household_count_scld', ]

        my_sets = {
            'default': dflt,
            'nrml_default': nrml_dflt,
            'top': topp,
            'topA': toppA,
        }


        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set

        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    class _base_(_BG_):
        dflt = []

        topp = ['Anti_Occup', 'diversity', '%female',
                'hh_total', 'average_household_income',
                'education_bachelor_or_above_rate', 'population_density_scld',
                'household_count',
                ]
        toppA = ['Anti_Occup', 'diversity', 'female_pct',
                 'average_household_income',
                 'bachelor_or_above_rate', 'population_density',
                 'household_count',
                 ]
        toppB = ['Anti_Occup', 'diversity', '%female',
                 'average_household_income_scld',
                 'bachelor_or_above_rate', 'population_density_scld', 'E_DAYPOP_scld',
                 'household_count_scld', 'heating_fuel_coal_coke_rate', 'hu_1959toearlier_scld',
                 'hu_med_val_scld', 'hu_mortgage',
                 ]
        toppC = [
            'Anti_Occup',
            'bachelor_or_above_rate', 'population_density_scld', 'E_DAYPOP_scld',
            'household_count_scld', 'heating_fuel_coal_coke_rate', 'hu_1959toearlier_scld',
            'hu_med_val_scld', 'hu_mortgage',
        ]
        toppD = [
            'Anti_Occup', 'diversity',
            'bachelor_or_above_rate', 'population_density_scld', 'E_DAYPOP_scld',
            'heating_fuel_coal_coke_rate', 'hu_1959toearlier_scld',
            'hu_med_val_scld', 'hu_mortgage',
        ]
        toppE = [
            'Anti_Occup',
            'bachelor_or_above_rate', 'population_density_scld', 'E_DAYPOP_scld',
            'heating_fuel_coal_coke_rate', 'hu_1959toearlier_scld',
            'hu_mortgage',
        ]
        toppF = [
            'Anti_Occup',
            'bachelor_or_above_rate', 'population_density_scld', 'E_DAYPOP_scld',
            'heating_fuel_coal_coke_rate', 'hu_1959toearlier_scld',
            'hu_mortgage', 'daily_solar_radiation', 'hdd_scld', 'cdd_scld',
        ]
        toppG = [
            'Anti_Occup', 'Green_Travelers', 'avg_monthly_bill_dlrs',
            'travel_time_average', 'transportation_carpool_rate',
            'bachelor_or_above_rate', 'population_density_scld', 'E_DAYPOP_scld',
            'heating_fuel_coal_coke_rate', 'hu_1959toearlier_scld',
            'hu_mortgage', 'daily_solar_radiation', 'hdd_scld', 'cdd_scld',
        ]
        toppH = [
            'Anti_Occup', 'Green_Travelers', 'avg_monthly_bill_dlrs',
            'travel_time_average', 'transportation_carpool_rate',
            'bachelor_or_above_rate', 'population_density_scld', 'E_DAYPOP_scld',
            'heating_fuel_coal_coke_rate', 'hu_1959toearlier_scld',
            'hu_mortgage', 'hdd_scld',
        ]

        my_sets = {'default': dflt,
                   'top': topp,
                   'topA': toppA,
                   'topB': toppB,
                   'topC': toppC,
                   'topD': toppD,
                   'topE': toppE,
                   'topF': toppF,
                   'topG': toppG,
                   'topH': toppH,
                   }

        def __init__(self, set=None):
            Block_Groups._BG_.__init__(self, self.my_sets, set, dflt=(self.dflt,))
            self.set = set

        def load_set(self, set=None):
            self.model = self.my_sets[set]
            return self.model

    def __init__(self, ):
        self.demographics = self._demographics()
        self.policy = self._policy_()
        self.physical = self._physical_()
        self.habit = self._habit_()
        self.climate = self._climate_()
        self.geography = self._geography_()
        self.Xu_models = self._Xu_Models_()
        self.population = self._population_()
        self.education = self._education_()
        self.age = self._age_()
        self.hh = self._hh_()
        self.energy_use = self._household_energy_use_()
        self.household_energy_use = self._household_energy_use_()
        self.dwelling_characteristics = self._dwelling_characteristics_()
        self.political_affiliation = self._political_affiliation_()
        self.household_size = self._household_size_()
        self.income = self._income_()
        self.income_employment_homeownership = self._income_employment_homeownership_()
        self.base = self._base_()
        self.gender = self._gender_()
        self.RF_Select = self._RF_Select_()
        self._All_ = self._All_()

        self.CT_demos = self._CT_demos_()
        self.Model = None

    def load_model(self, group, set='default', fullsetret=False):
        if group == 'demo':
            self.Model = self.demographics.load_set(set)
        elif group == 'policy':
            self.Model = self.policy.load_set(set)
        elif group == 'physical':
            self.Model = self.physical.load_set(set)
        elif group == 'habit':
            self.Model = self.habit.load_set(set)
        elif group == 'climate':
            self.Model = self.climate.load_set(set)
        elif group == 'geography':
            self.Model = self.geography.load_set(set)
        elif group == 'Xu':
            self.Model = self.Xu_models.load_set(set)
        elif group == 'population':
            self.Model = self.population.load_set(set)
        elif group == 'edu':
            self.Model = self.education.load_set(set)
        elif group == 'age':
            self.Model = self.age.load_set(set)
        elif group == 'hh':
            self.Model = self.hh.load_set(set)
        elif group == 'ct_demo':
            self.Model = self.CT_demos.load_set(set)
        elif group == 'income':
            self.Model = self.income.load_set(set)
        elif group == 'base':
            self.Model = self.base.load_set(set)
        elif group == 'energy':
            self.Model = self.energy_use.load_set(set)
        elif group == 'dwelling':
            self.Model = self.dwelling_characteristics.load_set(set)
        elif group == 'hh_size':
            self.Model = self.household_size.load_set(set)
        elif group == 'political':
            self.Model = self.political_affiliation.load_set(set)
        elif group == 'income employment':
            print('group: ', group)
            self.Model = self.income_employment_homeownership.load_set(set)
        elif group == 'gender':
            self.Model = self.gender.load_set(set)
        elif group == '':
            self.Model = self.base.load_set(set)
        elif group is None:
            self.Model = None
        elif group is 'RF_Select':
            self.Model = self.RF_Select.load_set(set)
        elif group is 'all':
            self.Model = self._All_.load_set(set)
        else:
            print('unknown group')

class Xu_Scalable:
    var_to_scale = ['population_density',
                    'housing_unit_count',
                    'land_area',
                    'hdd',
                    'median_household_income',
                    'dlrs_kwh x median_household_income',
                    ]

class Main_Usecols:
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

    nrel_variables = ['company_na', 'company_ty', 'pct_eli_hh', 'total_units',
                      'eia_id', 'cust_cnt', 'avg_monthly_consumption_kwh', 'avg_monthly_bill_dlrs',
                      'dlrs_kwh', 'hh_size_1', 'hh_size_2', 'hh_size_3', 'hh_size_4', 'fam_med_income',
                      'hh_med_income', 'pop_male', 'pop_female', 'pop25_some_college_plus', 'pop25_high_school',
                      'pop25_no_high_school', 'pop_med_age', 'pop_over_65', 'pop_under_18',
                      'hu_monthly_owner_costs_lessthan_1000dlrs', 'hu_monthly_owner_costs_greaterthan_1000dlrs',
                      'hu_own', 'hu_vintage_2010toafter', 'hu_vintage_2000to2009', 'hu_vintage_1980to1999',
                      'hu_vintage_1960to1970', 'hu_vintage_1940to1959', 'hu_vintage_1939toearlier', 'hu_med_val',
                      'hu_mortgage', 'hdd_std', 'hdd', 'cdd', 'cdd_std', 'climate_zone', 'geoid', 'locale',
                      'active_subsidies',
                      ] + caps_to_sum_own + ann_gen + ["state_fips", 'state_abbr']

    ds_variables = [
        'age_10_14_rate', 'age_15_17_rate', 'age_18_24_rate', 'age_25_34_rate', 'age_35_44_rate',
        'age_45_54_rate', 'age_5_9_rate', 'age_55_64_rate', 'age_65_74_rate', 'age_75_84_rate',
        'age_median', 'age_more_than_85_rate', 'cooling_design_temperature', 'heating_design_temperature',
        'travel_time_10_19_rate', 'travel_time_20_29_rate', 'travel_time_30_39_rate',
        'travel_time_40_59_rate', 'travel_time_60_89_rate', 'travel_time_average',
        'travel_time_less_than_10_rate', 'housing_unit_median_gross_rent', 'education_bachelor',
        'education_bachelor_rate', 'education_college', 'education_college_rate', 'education_doctoral',
        'education_doctoral_rate', 'education_high_school_graduate', 'education_high_school_graduate_rate',
        'education_less_than_high_school', 'education_less_than_high_school_rate', 'education_master',
        'education_master_rate', 'education_population', 'education_professional_school',
        'education_professional_school_rate', 'number_of_years_of_education', 'employ_rate',
        'avg_electricity_retail_rate', 'land_area', 'total_area', 'heating_fuel_coal_coke',
        'heating_fuel_coal_coke_rate', 'heating_fuel_electricity', 'heating_fuel_electricity_rate',
        'heating_fuel_fuel_oil_kerosene', 'heating_fuel_fuel_oil_kerosene_rate', 'heating_fuel_gas',
        'heating_fuel_gas_rate', 'heating_fuel_housing_unit_count', 'heating_fuel_none',
        'heating_fuel_none_rate', 'heating_fuel_other', 'heating_fuel_other_rate', 'heating_fuel_solar',
        'heating_fuel_solar_rate', 'poverty_family_count', 'housing_unit_count',
        'housing_unit_median_value', 'household_count', 'mortgage_with_rate', 'average_household_size',
        'occupancy_owner_rate', 'housing_unit_count', 'housing_unit_occupied_count',
        'household_type_family_rate', 'median_household_income', 'average_household_income', 'state',
        'fips', 'occupation_administrative_rate', 'occupation_agriculture_rate', 'occupation_arts_rate',
        'occupation_construction_rate', 'occupation_education_rate', 'occupation_finance_rate',
        'occupation_information_rate', 'occupation_manufacturing_rate', 'occupation_public_rate',
        'occupation_retail_rate', 'occupation_transportation_rate', 'occupation_wholesale_rate',
        'incentive_count_residential', 'incentive_count_nonresidential',
        'incentive_residential_state_level', 'incentive_nonresidential_state_level', 'net_metering',
        'property_tax', 'voting_2012_dem_percentage', 'voting_2012_gop_percentage', 'population',
        'population_density', 'poverty_family_below_poverty_level',
        'poverty_family_below_poverty_level_rate', 'diversity', 'Adoption', 'solar_system_count',
        'total_panel_area', 'solar_system_count_residential', 'solar_system_count_nonresidential',
        'total_panel_area_residential', 'total_panel_area_nonresidential', 'county',
        'number_of_solar_system_per_household', 'solar_panel_area_divided_by_area',
        'solar_panel_area_per_capita', 'daily_solar_radiation', 'transportation_bicycle_rate',
        'transportation_car_alone_rate', 'transportation_carpool_rate', 'transportation_home_rate',
        'transportation_motorcycle_rate', 'transportation_public_rate', 'transportation_walk_rate',
    ]

    def __init__(self, model='deepsolar'):
        self.model = model
        self.use_cols = self.process_request()

    def get_deepsolar(self):
        # return pd.read_excel(r'C:\Users\gjone\DeepSolar_Convergence\_Data\MainUsecolumns\To_Add_to_model.xlsx', sheet_name='DS')['variables'].values.tolist()
        return self.ds_variables

    def get_nrel(self):
        # return pd.read_excel(r'C:\Users\gjone\DeepSolar_Convergence\_Data\MainUsecolumns\To_Add_to_model.xlsx', sheet_name='NREL')['variables'].values.tolist()
        return self.nrel_variables

    def process_request(self):
        if self.model.lower() in ['DS'.lower(), 'deepsolar'.lower()]:
            return self.get_deepsolar()
        elif self.model.lower() == 'NREL'.lower():
            return self.get_nrel()
        else:
            print('Unknown data set {}'.format(self.model))
            print('options are: DS and NREL')
            quit(-99)


high_slr_low_e =[
    "ok",
    "sc",
    "ga",
    "tn",
    "ks",
    "co",
    "ne",
    "wy",
    "la",
    "al",
    "nc",
    "ms",
    "ar",
    "tx",
    "id",
    "mo",
]
high_slr_high_e = []

low_slr_high_e = [
    "ma",
    "ny",
    "md",
    "me",
    "mi",
    "ri",
    "nj",
    "nh",
    "wi",
    "vt",
    "ct",
]
low_slr_low_e = [
    "il",
    "ia",
    "de",
    "oh",
    "sd",
    "pa",
    "mt",
    "dc",
    "mn",
    "wa",
    "or",
    "ky",
    "nd",
    "in",
    "wv",
    "va",
]

group_labels2 = [
       'population',
       'dwelling',
       'edu',
       'age',
       'inc/empl',
       'gender',
       'climate',
       'geography',
       'habit',
       'policy',
       #'Xu',
       'political',
       #'ALL',
       ]

# rffi selected model for 13 state set
simple_set13 = ['population_density',
                'number_of_years_of_education_scld',
                'heating_fuel_none',
                'total_area',
                'poverty_family_below_poverty_level_rate',
                'occupation_finance_rate',
                'travel_time_10_19_rate',
                'heating_fuel_solar',
                'med_house_val',
                'hu_mortgage',
                'hu_med_val',
                'hu_vintage_1960to1970',
                'hu_1959toearlier_pct',
                'voting_2012_dem_percentage',
                'voting_2012_gop_percentage',
                'hu_own',
                'fam_med_income',
                'pop25_some_college_plus',
                'education_population',
                'education_bachelor',
                'age_55_64_rate',
                'age_median',
                'age_55_or_more_rate',
                'dlrs_kwh',
                'housing_unit_count',
                'cdd',
                'hdd',
                'Green Travelers',
                'travel_time_average',
                'travel_time_60_89_rate',
                'transportation_home_rate',
                'transportation_bicycle_rate',
                'transportation_car_alone_rate',
                'transportation_carpool_rate',
                'avg_monthly_consumption_kwh',
                'travel_time_40_89_rate',
                ]
block_group_2 = [
"education",            # 0
"age",                  # 1
"income",               # 2
'inc/homes',            # 3
"occupation",           # 4
"habit",                # 5
"geography",            # 6
"demo",                 # 7
"demographics",         # 8
"policy",               # 9
"gender",               # 10
"population",           # 11
"suitability",          # 12
"housing",              # 13
"politics",             # 14
"climate",              # 15
"renewables",           # 16
]

class Model_List:
    model_7st_2_2 = '__Data/____Training/DeepSolar_Model_Feb7_scld.xlsx'
    DSMIX_13 = r'_Data/Mixed/thirteen_DeepSolar_2020-03-24-07-08-41_scld_GS.xlsx'
    model_12st_init = '__Data/____Training/DeepSolar_Model_Feb13_adopt.xlsx'

    model_15_init = '__Data/____Training/DeepSolar_Model_15_02-13_scld.xlsx'

    seven_Xiao_set = r'seven_state_set_DeepSolar_scld.xlsx'

    Big_PV_model_2_28 = r'C:\Users\gjone\DeepSolar_Code_Base\__Data\____Training\DeepSolar_Model_13_2020-02-28-08-40-54__GS.xlsx'

    deepsolar_original = r'_Data/DeepSolar/deepsolar_tract_orig_Adoption.xlsx'

class State_Sets:
    tva_seven_state = ['tn', 'al', 'ky', 'ms', 'ga', 'va', 'nc']
    thirteen_state = ['tn', 'al', 'ky', 'ms', 'ga', 'va', 'nc', 'az', 'ca', 'ma', 'ut', 'tx', 'ny']
    fifteen_state1 = ['tn', 'al', 'ky', 'ms', 'ga', 'va', 'nc', 'az', 'ca', 'ma', 'ut', 'tx', 'ny', 'il', 'ia', ]
    southern_states = ['al', 'ar', 'fl', 'ga', 'ky', 'la', 'ms','nc', 'ok', 'sc', 'tn', 'tx', 'va', 'wv', ]
class Drop_Lists:
    """
        Contains several lists of items to drop from the DeepSolar, NREL, and Mixed data sets
    """
    drop_basic = [
        'state', 'fips', 'climate_zone', 'company_na', 'company_ty', 'eia_id',
        'geoid', 'locale', 'cust_cnt', 'cust_cnt_scld', 'FIPS', 'property_tax',
    ]
    drop_full_omega = drop_basic
    solar_drops = [
        'solar_system_count', 'solar_panel_area_divided_by_area', 'solar_panel_area_per_capita',
        'number_of_solar_system_per_household', 'solar_system_count_residential', 'solar_system_count_nonresidential',
        'total_panel_area', 'solar_system_count_nonresidential', 'total_panel_area_residential',
        'total_panel_area_nonresidential', 'Adoption',
    ]
    expanded_solar_drops = solar_drops + ['SNRaPa', 'ST_pcap', 'SNRpcap', 'SRpcap', 'SRaPa', ]

    basic_excludes = [
        'occupation_public_rate', 'age_25_34_rate', 'age_5_9_rate', 'dlrs_kwh', 'occupation_construction_rate',
        'occupation_arts_rate', 'voting_2012_gop_percentage', 'travel_time_less_than_10_rate',
        'incentive_residential_state_level', 'locale', 'occupation_wholesale_rate', 'travel_time_60_89_rate',
        '%female', 'climate_zone', 'heating_fuel_gas_rate', 'education_high_school_graduate_rate',
        'household_type_family_rate', '%hh_size_3', 'locale_recode', 'education_professional_school_rate',
        'education_doctoral_rate', 'occupation_administrative_rate', 'heating_fuel_other_rate', '%hh_size_2',
        '%male', 'heating_fuel_solar_rate', 'property_tax', 'age_65_74_rate', 'incentive_nonresidential_state_level',
        'company_na', 'heating_fuel_fuel_oil_kerosene_rate', 'education_college_rate', 'transportation_home_rate',
        'locale_dummy', 'net_metering', 'occupation_retail_rate', '%hh_size_1', 'transportation_carpool_rate',
        'travel_time_20_29_rate', 'company_ty', 'Ren', 'education_master_rate', 'education_less_than_high_school_rate',
        'voting_2012_dem_percentage', 'heating_fuel_electricity_rate', 'incentive_count_residential', 'state',
        'transportation_walk_rate', 'age_45_54_rate', 'occupation_education_rate', 'transportation_motorcycle_rate',
        'transportation_public_rate', 'eia_id', 'occupation_information_rate', 'heating_fuel_none_rate',
        'travel_time_40_59_rate', 'number_of_solar_system_per_household', 'geoid', 'FIPS', 'age_10_14_rate',
        'incentive_count_nonresidential', 'age_35_44_rate', 'age_75_84_rate', 'education_bachelor_rate',
        'mortgage_with_rate', 'occupation_agriculture_rate', 'age_18_24_rate', 'age_more_than_85_rate',
        'fips', 'poverty_family_below_poverty_level_rate', 'diversity', 'Adoption', '%hh_size_4',
        'occupation_finance_rate', 'travel_time_10_19_rate', 'occupation_manufacturing_rate',
        'occupation_transportation_rate', 'transportation_car_alone_rate', 'age_15_17_rate', 'Pro_Occup',
        'travel_time_30_39_rate', 'heating_fuel_coal_coke_rate', 'Green_Travelers',
        'transportation_bicycle_rate', 'age_55_64_rate',
    ]

    stripped_excludes = [
        'fips',
        'state',
        'locale',
        'locale_recode',
        'climate_zone',
        'company_na',
        'eia_id',
        'geoid',
        'company_ty',
    ]

# ###########################################################
# #######   TODO: Variable Dictionary and conversions
# ###########################################################
solar_conversions_GIS = {'Adoption': 'Adptn',
 'solar_system_count': 'PV_cnt',
 'solar_panel_area_divided_by_area': 'PVarByar',
 'solar_panel_area_per_capita': 'PVarPcap',
 'solar_system_count_residential': 'PV_res',
 'solar_system_count_nonresidential': 'PV_nres',
 'total_panel_area_residential': 'PVarRes',
 'total_panel_area_nonresidential': 'PVarNres',
 'total_panel_area': 'PV_area',
 'number_of_solar_system_per_household': 'PV_hh',
 'SRpcap': 'PVresCap',
 'SNRpcap': 'PvNRCap',
 'ST_pcap': 'PvStCap',
 'SRaPa': 'SRaPa',
 'SNRaPa': 'SNRaPa',
 'SRaPcap': 'SRaPcap',
 'SNRaPcap': 'SNRaPcap'}
# collection of regional data set paths
reg_dict = {
    'tva': Data_set_paths.tva_set,
    '7 State': Data_set_paths.seven_set,
    '13 State': Data_set_paths.thirteen_set,
    'US': Data_set_paths.US_set,
    'cnvrg': Data_set_paths.US_Convergent_Base,
    'p1': Data_set_paths.paperset,
    'p2': Data_set_paths.paperset2,
    'p3': Data_set_paths.paperset3,
    'p4': Data_set_paths.paperset4,
    'p5': Data_set_paths.paperset5,
    'ltt': r"C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\US_set_all_OMEGA_1_24_21_Base.csv",
    'HghSlr': r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\DeepSolar_highSolar.xlsx',
    'T3': r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\DeepSolar_T3.xlsx',
    'Hot':r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\DeepSolar_HotSpots.xlsx',
    'NT3': r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\DeepSolar_NT3.xlsx',
}

climate_zone_dict = {
                     1.0: 'Very Hot', 2.0: 'Hot', 3.0: 'Warm',
                     4.0: 'Mixed', 5.0: 'Cool', 6.0: 'Cold',
                     7.0: 'Very Cold',
                     }

select_drops = [
                #'fips',
                'geoid',
                #'state_fips', 'county_fips',
                'tract_fips', 'high_mf_rent_bldg_cnt',
                'centroid_x', 'housing_unit_median_value', 'land_area', "mid_mf_rent_bldg_cnt",
                "mid_mf_rent_devp_cnt", "mid_mf_rent_devp_m2","mid_mf_rent_hh","mid_mf_rent_mw",
                "mid_mf_rent_mwh", "low_mf_rent_bldg_cnt","low_mf_rent_devp_cnt","low_mf_rent_devp_m2",
                "low_mf_rent_elep_hh","low_mf_rent_hh","low_mf_rent_mw","low_mf_rent_mwh", "high_mf_rent_bldg_cnt",
                "high_mf_rent_devp_cnt", "high_mf_rent_devp_m2", "high_mf_rent_elep_hh",
                "high_mf_rent_hh", "high_mf_rent_mw", "high_mf_rent_mwh", "high_sf_rent_bldg_cnt", "high_sf_rent_devp_cnt",
                "high_sf_rent_devp_m2", "high_sf_rent_elep_hh", "high_sf_rent_hh", "hu_rent",
                "high_sf_rent_mw", "high_sf_rent_mwh", "housing_unit_median_gross_rent", "low_sf_rent_bldg_cnt",
                "low_sf_rent_devp_cnt","low_sf_rent_devp_m2","low_sf_rent_elep_hh","low_sf_rent_hh",
                "low_sf_rent_mw", "low_sf_rent_mwh", "mid_mf_rent_bldg_cnt", "mid_mf_rent_devp_cnt",
                "mid_mf_rent_devp_m2", "mid_mf_rent_hh", "mid_mf_rent_mw", "mid_mf_rent_mwh","mid_sf_rent_bldg_cnt",
                "mid_sf_rent_devp_cnt", "hu_rent",
                "mid_sf_rent_devp_m2","housing_unit_median_gross_rent",
                "mid_sf_rent_hh",'very_low_mf_rent_elep_hh',
                "mid_sf_rent_mw",'low_mf_rent_elep_hh', 'pop_non_us_citizen',
                "mid_sf_rent_mwh","mod_mf_rent_bldg_cnt",
                "mod_mf_rent_devp_cnt",'mod_mf_rent_elep_hh',
                "mod_mf_rent_devp_m2",'high_mf_rent_bldg_cnt',
                "mod_mf_rent_elep_hh",
                "mod_mf_rent_hh",
                "mod_mf_rent_mw", 'incentive_nonresidential_state_level',
                "mod_mf_rent_mwh", "very_low_mf_rent_bldg_cnt",
                "very_low_mf_rent_devp_cnt",
                "very_low_mf_rent_devp_m2",
                "very_low_mf_rent_elep_hh",
                "very_low_mf_rent_hh",
                "very_low_mf_rent_mw",
                "very_low_mf_rent_mwh", "very_low_sf_rent_bldg_cnt",
                "very_low_sf_rent_devp_cnt",
                "very_low_sf_rent_devp_m2",
                "very_low_sf_rent_elep_hh",
                "very_low_sf_rent_hh", 'heating_fuel_gas',
                "very_low_sf_rent_mw",'area_km2',
                "very_low_sf_rent_mwh", #+ ['pop_african_american','pop_non_us_citizen',]
                'incentive_nonresidential_state_level',
               ]

states_by_climate_zone = {
"Very Hot": ['fl'],
"Hot": ['la', 'az', 'fl', 'ga', 'ms', 'tx', 'ca', 'al'],
"Warm": ['nm', 'ut', 'sc', 'nc', 'ar', 'la', 'az', 'nv', 'ga', 'ca', 'tx', 'ms', 'tn', 'ok', 'al'],
"Mixed": ['nc', 'az', 'wa', 'ny', 'ga', 'or', 'tn', 'ks', 'co', 'dc', 'wv', 'ar', 'de', 'nj', 'mo', 'oh', 'nm', 'in', 'pa', 'va', 'md', 'ca', 'tx', 'ok', 'il', 'ky'],
"Cool": ['ut', 'nc', 'wy', 'az', 'wa', 'ny', 'ct', 'or', 'ks', 'co', 'ma', 'wv', 'ne', 'ri', 'nj', 'mo', 'nv', 'ia', 'nh', 'oh', 'nm', 'in', 'pa', 'id', 'md', 'sd', 'ca', 'mi', 'il'],
"Cold": ['ut', 'nd', 'vt', 'wy', 'pa', 'wa', 'me', 'mn', 'id', 'ny', 'mt', 'nh', 'sd', 'ia', 'ca', 'mi', 'co', 'wi'],
"Very Cold": ['nd', 'wy', 'me', 'mn', 'mi', 'co', 'wi'],
}

# #############################################################################
#       TODO: State_model Dictionary ##########################################
# #############################################################################

state_var_dict = {
    "al": ['total_own_hh', 'education_master_rate', 'hu_rent', 'travel_time_average', 
           'Anti_Occup', 'heating_fuel_coal_coke_rate', 'own_popden', 'travel_time_60_89_rate', 'housing_unit_median_gross_rent', 
           'occupation_agriculture_rate', 'hu_vintage_1960to1970', 'occupation_finance_rate', 'hu_vintage_2010toafter', 
           'occupation_administrative_rate', 'transportation_carpool_rate', 'area_km2', 'travel_time_less_than_10_rate', 'age_65_74_rate', 
           'employ_rate', 'pop_african_american', 'age_25_44_rate', 'occupation_arts_rate', 'occupation_education_rate', 
           'transportation_home_rate', 'diversity', '%hh_size_4', '%hh_size_3', 'occupation_public_rate', 'Adoption'],
    "ar_top2": ['poverty_family_below_poverty_level', 'housing_unit_median_gross_rent', 'hu_1959toearlier', 'occupation_information_rate', 
                'age_55_or_more_rate', 'age_65_74_rate', 'pop_non_us_citizen', 'mid_hh_rate', 'pop_hispanic', 'transportation_home_rate', 
                'occupation_finance_rate', 'hu_vintage_1939toearlier', 'transportation_carpool_rate', 'hu_vintage_2010toafter', 
                'low_hh_rate', 'mod_hh_rate', 'pop_asian', 'hu_2000toafter_pct', 'housing_unit_count', 'fam_children_under_6', 
                'occupation_retail_rate', '%hh_size_2', 'occupation_education_rate', 'heating_fuel_coal_coke_rate', 'age_18_24_rate', 
                'age_25_64_rate', 'heating_fuel_electricity_rate', 'mod_sf_rent_elep_hh', 'Pro_Occup', 'transportation_walk_rate', 
                'occupation_wholesale_rate', 'hu_1960to1979_pct', 'housing_unit_median_value', 'occupation_administrative_rate', 
                '%hh_size_4', 'high_mf_rent_bldg_cnt', 'occupation_public_rate', 'household_type_family_rate', 'mod_own_Sbldg_rt', 
                'employ_rate', 'age_5_9_rate', 'Green_Travelers', 'age_more_than_85_rate', 'occupation_arts_rate', 'age_55_64_rate', 
                'age_25_34_rate', 'hu_rent', 'occupation_transportation_rate', 'age_10_14_rate', 'hh_size_3', '%hh_size_3', 
                'average_household_size', 'heating_fuel_gas_rate', 'occupation_agriculture_rate', 'travel_time_average', 
                'occupation_construction_rate', 'travel_time_40_59_rate', 'travel_time_60_89_rate', 'own_popden', 'population_density'],
    "ar2":  ['own_popden', 'travel_time_average', 'occupation_agriculture_rate', 'high_mf_rent_bldg_cnt', 'hu_rent', 
             'heating_fuel_coal_coke_rate', 'pop_asian', 'housing_unit_median_value', 'travel_time_60_89_rate', 'pop_non_us_citizen', 
             'occupation_construction_rate', 'age_25_34_rate', 'Pro_Occup', 'heating_fuel_gas_rate', 'age_65_74_rate', 
             'occupation_arts_rate', 'occupation_administrative_rate', 'household_type_family_rate', 'occupation_information_rate', 
             'age_55_64_rate', 'occupation_retail_rate', 'employ_rate', 'transportation_carpool_rate', 'mod_own_Sbldg_rt', 
             'mod_sf_rent_elep_hh', 'occupation_finance_rate', '%hh_size_2', 'occupation_public_rate', 'hu_vintage_2010toafter', 
             'age_more_than_85_rate', 'age_18_24_rate', 'occupation_transportation_rate', 'hu_1959toearlier', 'mid_hh_rate', 
             'fam_children_under_6', 'low_hh_rate', ],
    "al_top": ['age_25_44_rate', 'housing_unit_median_gross_rent', 'occupation_manufacturing_rate', 'hu_vintage_2010toafter',               
               'occupation_finance_rate', 'pop_african_american', 'pop_total', 'pop_hispanic', 'pop_over_65', 'age_65_74_rate', 
               'hu_vintage_1960to1970', 'occupation_arts_rate', 'high_mf_own_devp_cnt', 'age_25_34_rate', 'transportation_carpool_rate', 
               '%hh_size_4', 'age_18_24_rate', 'age_75_84_rate', 'high_mf_rent_hh', 'hh_gini_index', 'travel_time_40_59_rate', 
               'age_15_17_rate', 'high_mf_rent_bldg_cnt', 'occupation_public_rate', 'age_25_64_rate', 'employ_rate', 'hu_1980to1999_pct', 
               'occupation_administrative_rate', 'age_45_54_rate', 'Green_Travelers', 'age_more_than_85_rate', 'transportation_home_rate', 
               'occupation_retail_rate', '%hh_size_2', 'pop_nat_us_citizen', 'occupation_agriculture_rate', 'age_35_44_rate',             
               '%hh_size_3', 
               'age_10_14_rate', 'avg_inc_ebill_dlrs', 'travel_time_20_29_rate', 'occupation_wholesale_rate', 'diversity', 'hh_size_1', 
               'education_college_rate', 'occupation_education_rate', 'Anti_Occup', 'own_popden', 'heating_fuel_coal_coke_rate', 
               'education_master_rate', 'pop_non_us_citizen', 'travel_time_60_89_rate', 'area_km2', 'travel_time_less_than_10_rate', 
               'hu_rent', 'mid_own_mwh', 'total_own_hh', 'population_density', 'travel_time_average', 'high_own_mwh'],
    "ar": ['population_density', 'travel_time_average', 'total_area', 'occupation_agriculture_rate',
           'heating_fuel_coal_coke_rate', 'pop_female', 'pop_asian', 'housing_unit_median_value',
           'mid_mf_own_devp_m2', 'education_doctoral', 'occupation_construction_rate', 'age_25_34_rate',
           'pop_hispanic', 'Pro_Occup', 'high_sf_own_elep_hh', 'heating_fuel_gas_rate',
           'education_less_than_high_school_rate', 'age_median', 'occupation_arts_rate',
           'household_type_family_rate', 'occupation_administrative_rate', 'occupation_retail_rate',
            'employ_rate', 'mod_own_Sbldg_rt',
            'occupancy_owner_rate', '%hh_size_2'],
    "az": [],
    "ca": ['locale_dummy', 'heating_fuel_coal_coke_rate', 'pop_asian', 'high_own_elep_hh', 'population_density',
            'total_area', 'occupation_agriculture_rate', 'hdd', 'hh_size_3', 'age_55_or_more_rate', 'age_35_44_rate',
            'travel_time_less_than_10_rate', 'occupation_wholesale_rate', 'net_metering_hu_own',
            'occupation_manufacturing_rate', 'Green_Travelers', 'heating_fuel_electricity', '%hh_size_1',
            'occupation_finance_rate', 'occupation_administrative_rate', 'hu_vintage_1940to1959',
            'avg_monthly_bill_dlrs', 'age_25_34_rate', 'age_18_24_rate', 'high_own_Sbldg_rt',
            'transportation_car_alone_rate', 'diversity','high_hh_rate'
            'education_bachelor_rate', 'hh_gini_index', 'travel_time_20_29_rate', 'occupation_public_rate'],
    "co": [],
    "ct": [],
    "dc": [],
    "de": [],
    "fl": [],
    "ga1": ['total_own_hh', 'pop_nat_us_citizen', 'housing_unit_median_value', 'hh_size_1', 'own_popden', 'heating_fuel_coal_coke_rate',  'occupation_finance_rate', 'employ_rate', 'occupation_information_rate', 'occupation_construction_rate', 'transportation_home_rate',  'diversity', 'age_65_74_rate', 'transportation_carpool_rate', 'age_35_44_rate', 'occupation_manufacturing_rate', 'age_75_84_rate',  'occupation_wholesale_rate', 'occupation_transportation_rate', 'travel_time_less_than_10_rate', 'area_km2', 'occupation_arts_rate',  'hh_gini_index', 'hu_vintage_1939toearlier', 'transportation_walk_rate', '%hh_size_1', 'transportation_car_alone_rate'],
    "ga": ['total_own_hh', 'education_master', 'pop_asian', 'fam_children_under_6', 'housing_unit_median_value',
           'hh_size_1', 'heating_fuel_coal_coke_rate',
           'occupation_finance_rate', 'hu_no_mortgage', 'population_density', 'diversity', 'age_65_74_rate'],
       "ga_top": ['age_25_34_rate', 'hu_vintage_1939toearlier', 'transportation_public_rate', 'transportation_walk_rate',  'occupation_administrative_rate', '%hh_size_1', 'high_own_hh', 'age_65_74_rate', 'occupation_information_rate', 'age_18_24_rate',  'education_high_school_graduate_rate', 'transportation_home_rate', 'occupation_education_rate', 'travel_time_60_89_rate',  'hu_vintage_1940to1959', 'employ_rate', 'age_75_84_rate', 'transportation_car_alone_rate', 'Green_Travelers', 'travel_time_40_89_rate',  'age_35_44_rate', 'hh_size_1', 'age_5_9_rate', 'pop_asian', 'occupation_wholesale_rate', 'transportation_carpool_rate', '%hh_size_2',  'fam_children_under_6', 'age_45_54_rate', 'age_10_14_rate', 'occupation_public_rate', 'age_15_17_rate', 'diversity',  'occupation_transportation_rate', 'occupation_retail_rate', 'housing_unit_median_value', 'occupation_arts_rate',  'travel_time_20_29_rate', 'high_own_mwh', 'occupation_manufacturing_rate', 'occupation_construction_rate', 'travel_time_40_59_rate',  '%hh_size_3', 'travel_time_average', 'hh_gini_index', 'occupation_finance_rate', 'heating_fuel_coal_coke_rate', 'pop_nat_us_citizen',  'education_bachelor', 'age_more_than_85_rate', 'pop_non_us_citizen', 'travel_time_less_than_10_rate', 'education_master', 'own_popden',  'area_km2', 'total_area', 'total_own_Sbldg', 'pop_caucasian', 'population_density', 'total_own_hh'],
    
    "ia_top": ['hh_size_4', 'very_low_mf_rent_elep_hh', 'age_more_than_85_rate', 'hu_vintage_1940to1959', 'age_55_64_rate', 'hh_size_3', 'pop_nat_us_citizen', 'employ_rate', 'education_population', 'hh_gini_index', 'age_15_17_rate', '%hh_size_1', 'occupation_public_rate', 'occupation_education_rate', 'age_5_9_rate', '%hh_size_3', 'hu_monthly_owner_costs_lessthan_1000dlrs', 'education_college_rate', 'age_45_54_rate', 'heating_fuel_other_rate', 'travel_time_40_59_rate', 'Green_Travelers', 'transportation_walk_rate', 'area_km2', 'hdd', 'total_area', 'travel_time_60_89_rate', 'transportation_carpool_rate', 'transportation_home_rate', 'hu_mortgage', 'occupation_transportation_rate', 'occupation_arts_rate', 'travel_time_40_89_rate', 'mod_mf_rent_hh', 'transportation_car_alone_rate', 'occupation_administrative_rate', 'low_mf_rent_devp_cnt', 'pop_non_us_citizen', 'total_own_Sbldg', 'total_own_hh', 'low_mf_rent_bldg_cnt', '%hh_size_2', 'occupation_construction_rate', 'travel_time_less_than_10_rate', 'occupation_information_rate', 'education_high_school_graduate_rate', 'occupation_retail_rate', 'mod_mf_rent_bldg_cnt', 'education_bachelor', 'hu_1959toearlier', 'age_10_14_rate', 'travel_time_average', 'travel_time_20_29_rate', 'occupation_wholesale_rate', 'high_mf_own_hh', 'pop_total', 'hh_size_1', 'own_popden', 'population_density'],
    
    "ia1": ['hh_size_1', 'hh_size_3', 'own_popden', 'pop_non_us_citizen', 'travel_time_average', 'area_km2', 'education_high_school_graduate_rate', 'heating_fuel_other_rate', 'occupation_administrative_rate', 'occupation_arts_rate', '%hh_size_2', 'transportation_home_rate', 'hu_vintage_1940to1959', 'age_55_64_rate', 'occupation_construction_rate', 'housing_unit_median_gross_rent', 'Green_Travelers', 'hu_monthly_owner_costs_lessthan_1000dlrs', 'very_low_mf_rent_elep_hh', 'occupation_information_rate', 'occupation_transportation_rate', 'travel_time_60_89_rate', 'travel_time_20_29_rate', '%hh_size_1', 'age_45_54_rate', '%hh_size_3', 'education_college_rate', 'occupation_retail_rate', 'occupation_wholesale_rate', 'hdd'],
    
    "ia": ['hh_size_1', 'total_own_hh', 'heating_fuel_coal_coke_rate', 'heating_fuel_fuel_oil_kerosene_rate',
           'population_density', 'mod_mf_own_bldg_cnt', 'occupation_agriculture_rate', 'hdd',
           'travel_time_average', 'occupation_administrative_rate', 'pop_hispanic', 'diversity', 'pop_over_65',
           'travel_time_40_89_rate', 'occupation_arts_rate', '%hh_size_2', 'transportation_home_rate', 'age_55_64_rate',
           'hu_vintage_1940to1959', 'occupation_construction_rate', 'Green_Travelers', 'occupation_information_rate',
           'occupation_transportation_rate', 'travel_time_20_29_rate', 'hu_1959toearlier', 'occupation_retail_rate',
           'travel_time_less_than_10_rate'],
    "id": [],
    "il": ['female_pct', 'mod_hh_rate', 'average_household_income', 'number_of_years_of_education',
           'occupation_retail_rate', 'hu_2000toafter_pct', 'Green_Travelers',
           'occupation_finance_rate', 'hu_no_mortgage', 'male_pct',
           'heating_fuel_electricity', 'transportation_walk_rate', 'diversity', 'occupation_transportation_rate',
           'Anti_Occup', 'education_master_or_above_rate', 'hu_vintage_2000to2009', 'education_master_rate',
           'hu_1960to1979_pct', 'employ_rate', 'avg_inc_ebill_dlrs', 'education_doctoral', 'age_more_than_85_rate',
           'age_75_84_rate', 'high_mf_own_mw', 'age_15_17_rate', 'hu_2000toafter', 'high_mf_own_devp_m2',
           'travel_time_40_89_rate', 'occupation_arts_rate', 'hu_vintage_1940to1959', 'p16_unemployed', 'hh_size_1',
           'pop_asian', 'hh_size_2', 'education_professional_school', 'travel_time_20_29_rate', 'total_own_Sbldg',
           'poverty_family_count', 'travel_time_less_than_10_rate', 'p16_employed', 'high_own_Sbldg', 'high_own_mwh',
           'education_bachelor', 'education_master', 'total_own_hh', 'high_own_hh', 'total_area', 'population_density',
           'pop_caucasian'],
    
    "il_top": ['education_doctoral', 'hu_vintage_1960to1970', 'travel_time_40_89_rate', 'fam_children_under_6', 'p16_unemployed', 
               'pop_non_us_citizen', 'age_65_74_rate', 'age_25_64_rate', 'pop_african_american', '%hh_size_2', 
               'occupation_administrative_rate', 'occupation_education_rate', 'occupation_manufacturing_rate', 'hh_size_1', 'diversity', 
               'age_55_64_rate', 'age_18_24_rate', 'occupation_construction_rate', 'age_5_9_rate', 'travel_time_60_89_rate', 
               'p16_employed', '%hh_size_3', 'age_35_44_rate', 'age_10_14_rate', 'education_college_rate', 'Anti_Occup', 'age_45_54_rate', 
               'travel_time_40_59_rate', 'occupation_information_rate', 'Green_Travelers', 'occupation_wholesale_rate', 
               'transportation_home_rate', 'hu_1960to1979_pct', 'hh_gini_index', 'occupation_transportation_rate', 
               'occupation_public_rate', 'total_area', 'transportation_carpool_rate', 'area_km2', 'pop_asian', 'age_75_84_rate', 
               'occupation_finance_rate', 'occupation_retail_rate', 'age_more_than_85_rate', 'age_15_17_rate', 'employ_rate', 
               'high_own_Sbldg', 'transportation_walk_rate', 'hu_vintage_1940to1959', 'occupation_arts_rate', 'own_popden', 
               'high_own_mwh', 'pop_caucasian', 'travel_time_less_than_10_rate', 'travel_time_20_29_rate', 'education_bachelor', 
               'population_density', 'education_master', 'high_own_hh', 'total_own_hh'],
    
    "il1": [ 'high_own_hh', 'education_doctoral', 'pop_asian', 'hh_size_1', 'employ_rate', 'p16_unemployed', 'hu_vintage_1960to1970',
              'occupation_finance_rate', 'Anti_Occup', 'travel_time_less_than_10_rate', 'occupation_information_rate', 'area_km2', 
              'occupation_transportation_rate', 'age_18_24_rate', 'pop_non_us_citizen', 'hu_vintage_1940to1959', 
              'occupation_administrative_rate', 'education_college_rate', 'transportation_carpool_rate', 'population_density', 
              'age_more_than_85_rate', 'travel_time_40_89_rate', 'occupation_wholesale_rate', 'transportation_home_rate', 
              'age_45_54_rate', 'travel_time_20_29_rate', '%hh_size_2', 'age_25_64_rate', 'age_10_14_rate',],
    
    
    "in_top": ['cdd', 'occupation_administrative_rate', 'housing_unit_median_gross_rent', 'age_55_64_rate', 'hu_vintage_1960to1970', 'hu_vintage_1939toearlier', 'age_25_34_rate', 'age_75_84_rate', 'hh_gini_index', 'mid_own_Sbldg_rt', 'education_high_school_graduate_rate', 'occupation_agriculture_rate', 'transportation_home_rate', 'employ_rate', 'hu_vintage_1940to1959', 'low_sf_rent_devp_cnt', 'mid_hh_rate', 'diversity', 'transportation_walk_rate', 'travel_time_40_89_rate', 'transportation_carpool_rate', 'high_mf_rent_hh', 'occupation_information_rate', 'age_5_9_rate', 'age_45_54_rate', 'education_college_rate', 'age_35_44_rate', 'occupation_education_rate', 'hu_1960to1979_pct', 'pop_hispanic', 'pop_african_american', 'age_more_than_85_rate', 'age_15_17_rate', 'travel_time_40_59_rate', 'hh_size_1', 'Green_Travelers', '%hh_size_3', 'occupation_finance_rate', 'mod_mf_rent_mwh', 'age_10_14_rate', 'occupation_wholesale_rate', 'age_25_64_rate', 'occupation_construction_rate', 'age_18_24_rate', 'occupation_arts_rate', 'travel_time_60_89_rate', 'occupation_retail_rate', 'area_km2', 'travel_time_20_29_rate', 'occupation_transportation_rate', 'occupation_public_rate', 'own_popden', 'total_area', 'heating_fuel_coal_coke_rate', 'heating_fuel_gas', 'travel_time_average', 'education_bachelor', 'population_density', 'education_master', 'travel_time_less_than_10_rate'],
    "in": ['heating_fuel_gas', 'hh_size_1', 'travel_time_average', 'education_high_school_graduate_rate', 'travel_time_less_than_10_rate', 'heating_fuel_coal_coke_rate', 'pop_hispanic', 'hu_vintage_1960to1970', 'low_sf_rent_devp_cnt', 'occupation_agriculture_rate', 'occupation_transportation_rate', 'own_popden', 'employ_rate', 'occupation_construction_rate', 'occupation_education_rate', 'age_45_54_rate', 'diversity', 'hu_vintage_1940to1959', 'age_more_than_85_rate', 'occupation_information_rate', 'age_55_64_rate', 'occupation_arts_rate', 'age_25_64_rate', 'transportation_carpool_rate', 'age_18_24_rate', 'housing_unit_median_gross_rent', 'occupation_administrative_rate', 'occupation_finance_rate', 'age_10_14_rate', 'occupation_wholesale_rate', 'hu_vintage_1939toearlier', 'occupation_public_rate'],
    "in1": [],
    
    "ks_top": ['pop_non_us_citizen', 'occupation_education_rate', 'mid_mf_rent_devp_cnt', '%hh_size_4', 'mid_own_Sbldg_rt', 'age_55_64_rate', 'hu_1959toearlier', 'age_10_14_rate', 'Green_Travelers', 'high_mf_rent_mwh', 'occupation_public_rate', 'occupation_transportation_rate', 'education_doctoral_rate', 'education_college', 'travel_time_40_89_rate', 'age_35_44_rate', 'age_45_54_rate', 'hh_gini_index', 'Anti_Occup', 'total_area', 'high_mf_rent_devp_cnt', 'pop_female', 'transportation_walk_rate', 'pop_asian', 'travel_time_less_than_10_rate', 'heating_fuel_coal_coke_rate', 'transportation_carpool_rate', 'age_25_64_rate', 'education_less_than_high_school_rate', 'occupation_information_rate', 'occupation_retail_rate', 'area_km2', 'hu_vintage_1940to1959', 'diversity', 'heating_fuel_other_rate', 'employ_rate', 'hu_vintage_1960to1970', 'occupation_arts_rate', 'occupation_wholesale_rate', 'age_5_9_rate', 'hu_monthly_owner_costs_greaterthan_1000dlrs', 'occupation_construction_rate', 'mod_own_elep_hh', '%hh_size_2', 'education_doctoral', 'occupation_manufacturing_rate', 'high_mf_rent_hh', 'pop_nat_us_citizen', 'age_18_24_rate', 'population_density', 'heating_fuel_gas_rate', 'education_bachelor', 'heating_fuel_gas', 'fam_children_under_6', 'travel_time_20_29_rate', 'own_popden', 'high_mf_rent_bldg_cnt', 'travel_time_60_89_rate', 'hu_mortgage', 'p16_employed'],
            
    "ks1": ['p16_employed', 'fam_children_under_6', 'Anti_Occup', 'education_doctoral', 'heating_fuel_other_rate', 'pop_nat_us_citizen', 'heating_fuel_coal_coke_rate', 'hu_vintage_1960to1970', 'own_popden', 'travel_time_60_89_rate', 'education_less_than_high_school_rate', 'occupation_information_rate', 'occupation_construction_rate', 'occupation_retail_rate', 'occupation_education_rate', 'hh_gini_index', 'age_5_9_rate', 'occupation_manufacturing_rate', 'age_18_24_rate', 'pop_non_us_citizen', 'occupation_transportation_rate', 'occupation_arts_rate', 'hu_vintage_1940to1959', 'transportation_walk_rate', 'area_km2', 'heating_fuel_gas_rate', 'age_25_64_rate', 'age_10_14_rate', 'age_35_44_rate', 'age_45_54_rate'],
             
    "ks": [],
            
            
    "ky_top": ['pop_nat_us_citizen', '%hh_size_2', '%hh_size_3', 'very_low_mf_rent_hh', 'age_75_84_rate', 'age_25_34_rate', 'occupation_manufacturing_rate', 'occupation_public_rate', 'poverty_family_below_poverty_level', 'education_professional_school_rate', 'transportation_home_rate', 'transportation_carpool_rate', 'age_45_54_rate', 'pop_asian', 'mod_mf_rent_elep_hh', 'education_professional_school', 'high_mf_own_devp_cnt', 'hdd', 'p16_unemployed', 'occupation_wholesale_rate', 'education_doctoral_rate', 'occupation_retail_rate', 'hu_1960to1979_pct', 'hh_gini_index', 'hu_1959toearlier', 'occupation_construction_rate', 'occupation_administrative_rate', 'education_college', 'Green_Travelers', 'occupation_information_rate', 'high_sf_rent_bldg_cnt', 'mid_own_Sbldg_rt', 'age_25_64_rate', 'age_more_than_85_rate', 'age_15_17_rate', 'travel_time_20_29_rate', 'age_5_9_rate', 'heating_fuel_gas', 'high_mf_rent_hh', 'transportation_walk_rate', 'Anti_Occup', 'housing_unit_count', 'hu_vintage_1960to1970', 'age_18_24_rate', 'hu_rent', 'occupation_education_rate', 'pop_male', 'Pro_Occup', 'occupation_transportation_rate', 'travel_time_less_than_10_rate', 'hh_size_1', 'travel_time_40_89_rate', 'travel_time_40_59_rate', 'age_10_14_rate', 'travel_time_average', 'own_popden', 'pop_over_65', 'heating_fuel_coal_coke_rate', 'education_master_or_above_rate', 'population_density'],
    
    "ky1": ['heating_fuel_coal_coke_rate', 'travel_time_average', 'high_mf_rent_hh', 'population_density', 'heating_fuel_gas', 'Anti_Occup', 'hh_size_1', 'pop_male', 'pop_asian', 'education_master_or_above_rate', 'occupation_construction_rate', 'travel_time_less_than_10_rate', 'occupation_administrative_rate', 'transportation_walk_rate', 'occupation_transportation_rate', 'occupation_education_rate', 'age_45_54_rate', 'hu_1959toearlier', 'transportation_carpool_rate', 'age_more_than_85_rate', 'age_10_14_rate', 'hu_1960to1979_pct', 'hdd', 'mid_own_Sbldg_rt', 'age_25_34_rate', 'occupation_wholesale_rate', 'occupation_information_rate', 'age_15_17_rate', '%hh_size_2', 'mod_mf_rent_elep_hh', '%hh_size_3', 'age_5_9_rate', 'hh_gini_index', 'age_25_64_rate'],
    
    "ky": [],
    
            
    "la": [],
    
    "ma": [],
    "md": [],
    
    "me": [],
    
    "me_top": ['transportation_walk_rate', 'occupation_agriculture_rate', 'heating_fuel_other', 'pop_hispanic', 'travel_time_less_than_10_rate', 'education_professional_school_rate', 'p16_unemployed', 'education_bachelor_or_above_rate', 'heating_fuel_gas_rate', 'hu_2000toafter', 'mod_mf_own_hh', 'transportation_carpool_rate', 'very_low_sf_rent_bldg_cnt', 'average_household_size', 'hu_2000toafter_pct', 'hu_vintage_1960to1970', 'very_low_sf_rent_hh', 'occupation_transportation_rate', 'age_45_54_rate', 'employ_rate', 'hh_gini_index', 'fam_children_under_6', 'age_15_17_rate', 'age_55_or_more_rate', 'age_5_9_rate', 'age_18_24_rate', 'pop_nat_us_citizen', 'age_25_64_rate', 'mod_mf_own_bldg_cnt', 'pop_african_american', 'occupation_retail_rate', 'heating_fuel_coal_coke', 'hu_1960to1979_pct', 'age_65_74_rate', 'occupation_arts_rate', 'occupation_public_rate', 'age_35_44_rate', 'occupation_wholesale_rate', 'age_10_14_rate', 'occupation_information_rate', 'transportation_home_rate', 'age_more_than_85_rate', 'hu_vintage_2000to2009', 'education_college_rate', 'age_75_84_rate', 'occupation_finance_rate', 'age_25_34_rate', 'age_25_44_rate', 'poverty_family_below_poverty_level', 'education_college', 'diversity', 'own_popden', 'age_55_64_rate', 'heating_fuel_electricity_rate', 'hu_1959toearlier_pct', 'population_density', 'heating_fuel_coal_coke_rate', 'hu_rent', 'total_area', 'hu_vintage_1940to1959'],
    
    "me1": ['heating_fuel_coal_coke_rate', 'population_density', 'hu_rent', 'total_area', 'hu_vintage_1940to1959', 'heating_fuel_electricity_rate', 'age_55_64_rate', 'diversity', 'occupation_agriculture_rate', 'pop_nat_us_citizen', 'mod_mf_own_bldg_cnt', 'occupation_finance_rate', 'age_65_74_rate', 'age_18_24_rate', 'education_college', 'age_more_than_85_rate', 'travel_time_less_than_10_rate', 'pop_hispanic', 'hu_2000toafter_pct', 'hh_gini_index', 'age_45_54_rate', 'education_bachelor_or_above_rate', 'occupation_arts_rate', 'occupation_information_rate', 'occupation_transportation_rate', 'hu_1960to1979_pct', 'transportation_home_rate', 'age_25_64_rate', 'average_household_size', 'age_10_14_rate', 'occupation_public_rate', 'employ_rate', 'heating_fuel_other', 'occupation_retail_rate'],
    
    
    "mi_top": ['pop_native_american', 'housing_unit_median_gross_rent', 'age_25_34_rate', 'age_25_44_rate', '%hh_size_3', 'hu_monthly_owner_costs_greaterthan_1000dlrs', 'age_18_24_rate', 'travel_time_40_59_rate', 'high_mf_rent_devp_cnt', 'occupation_education_rate', 'age_75_84_rate', 'hu_1960to1979_pct', 'age_5_9_rate', 'occupation_manufacturing_rate', 'age_45_54_rate', 'travel_time_40_89_rate', 'occupation_wholesale_rate', '%hh_size_2', 'hh_gini_index', 'travel_time_20_29_rate', 'occupation_construction_rate', 'Green_Travelers', 'occupation_transportation_rate', 'occupation_public_rate', 'transportation_carpool_rate', 'age_more_than_85_rate', 'high_own_mwh', 'transportation_home_rate', 'pop_hispanic', 'age_10_14_rate', 'age_55_64_rate', 'occupation_arts_rate', 'age_35_44_rate', 'high_mf_own_mw', 'hh_size_1', 'pop_african_american', 'occupation_administrative_rate', 'occupation_information_rate', 'travel_time_60_89_rate', 'hu_vintage_1940to1959', 'occupation_finance_rate', 'occupation_retail_rate', 'age_15_17_rate', 'travel_time_average', 'pop_asian', 'heating_fuel_gas', 'own_popden', 'education_master', 'education_college_rate', 'travel_time_less_than_10_rate', 'pop_caucasian', 'housing_unit_median_value', 'employ_rate', 'education_bachelor', 'heating_fuel_coal_coke_rate', 'pop_nat_us_citizen', 'area_km2', 'hu_rent', 'total_area', 'population_density'],
    
    "mi1": ['pop_nat_us_citizen', 'education_bachelor', 'own_popden', 'heating_fuel_coal_coke_rate', 'employ_rate', 'hh_size_1', 'pop_hispanic', 'travel_time_average', 'hu_1960to1979_pct', 'occupation_finance_rate', 'housing_unit_median_gross_rent', 'occupation_construction_rate', 'transportation_carpool_rate', 'occupation_transportation_rate', 'occupation_administrative_rate', 'age_25_44_rate', 'education_college_rate', 'occupation_wholesale_rate', 'occupation_education_rate', 'hu_vintage_1940to1959', 'age_55_64_rate', 'occupation_public_rate', '%hh_size_2', '%hh_size_3', 'age_35_44_rate', 'pop_native_american', 'occupation_information_rate', 'age_more_than_85_rate'],
    
    "mi": [],
    
    
    
    "mn_top": ['age_25_64_rate', 'age_55_64_rate', 'hu_vintage_1960to1970', 'hu_vintage_1940to1959', 'hh_gini_index', 'low_mf_rent_hh', 'heating_fuel_gas_rate', 'transportation_carpool_rate', 'education_master', 'education_professional_school_rate', 'travel_time_20_29_rate', 'occupation_transportation_rate', '%hh_size_3', 'occupation_administrative_rate', 'occupation_public_rate', 'transportation_motorcycle_rate', 'hu_2000toafter_pct', 'age_5_9_rate', 'fam_children_6to17', '%hh_size_2', 'hdd', 'age_45_54_rate', 'occupation_finance_rate', 'pop_hispanic', 'occupation_arts_rate', 'age_25_34_rate', 'age_75_84_rate', 'occupation_retail_rate', 'occupation_manufacturing_rate', 'travel_time_less_than_10_rate', 'travel_time_60_89_rate', 'housing_unit_median_gross_rent', 'high_mf_rent_bldg_cnt', 'heating_fuel_coal_coke', 'education_bachelor', 'occupation_information_rate', 'hu_1959toearlier', 'education_master_rate', 'occupation_education_rate', 'transportation_home_rate', 'travel_time_40_89_rate', 'travel_time_40_59_rate', 'Green_Travelers', 'occupation_wholesale_rate', 'occupation_construction_rate', 'employ_rate', 'hu_rent', 'heating_fuel_fuel_oil_kerosene_rate', 'area_km2', 'hu_1960to1979_pct', 'hu_vintage_1939toearlier', 'transportation_bicycle_rate', 'total_area', 'education_college_rate', 'own_popden', 'hh_size_1', 'education_doctoral_rate', 'heating_fuel_coal_coke_rate', 'p16_employed', 'population_density'],
    
    "mn1": ['heating_fuel_coal_coke_rate', 'p16_employed', 'heating_fuel_fuel_oil_kerosene_rate', 'own_popden', 'total_area', 'low_mf_rent_hh', 'education_master_rate', 'occupation_construction_rate', '%hh_size_2', 'pop_hispanic', 'high_mf_rent_bldg_cnt', 'transportation_bicycle_rate', 'age_25_34_rate', 'age_55_64_rate', 'transportation_home_rate', 'hdd', 'occupation_finance_rate', 'occupation_transportation_rate', 'housing_unit_median_gross_rent', 'age_75_84_rate', 'occupation_information_rate', 'age_45_54_rate', 'travel_time_40_89_rate', 'hu_1959toearlier', 'transportation_carpool_rate', 'occupation_education_rate', 'age_25_64_rate', 'occupation_public_rate', 'transportation_motorcycle_rate', 'hu_vintage_1960to1970', 'occupation_arts_rate', 'hh_gini_index', 'occupation_manufacturing_rate', 'Green_Travelers', '%hh_size_3', 'employ_rate'],
    
    
    "mn": [],
    
    
    "mo_top": ['occupation_agriculture_rate', 'occupation_transportation_rate', '%hh_size_4', 'age_10_14_rate', 'education_college_rate', 'hu_vintage_1960to1970', 'high_mf_own_devp_m2', 'occupation_construction_rate', 'occupation_finance_rate', 'age_5_9_rate', 'housing_unit_median_gross_rent', 'age_median', 'diversity', 'occupation_manufacturing_rate', 'occupation_wholesale_rate', 'heating_fuel_gas', 'low_own_elep_hh', 'pop_med_age', 'occupation_arts_rate', 'travel_time_40_59_rate', 'housing_unit_median_value', 'age_18_24_rate', 'high_sf_own_mwh', 'transportation_walk_rate', 'occupation_administrative_rate', 'age_more_than_85_rate', 'transportation_home_rate', 'occupation_public_rate', 'fam_children_under_6', 'hu_1960to1979_pct', 'occupation_education_rate', 'average_household_size', 'Pro_Occup', 'age_75_84_rate', 'transportation_car_alone_rate', 'transportation_carpool_rate', '%hh_size_1', 'occupation_retail_rate', 'travel_time_60_89_rate', 'travel_time_less_than_10_rate', 'age_55_or_more_rate', 'occupation_information_rate', 'age_45_54_rate', 'area_km2', 'heating_fuel_coal_coke', 'total_area', 'high_own_hh', 'employ_rate', '%hh_size_2', 'pop_hispanic', 'travel_time_20_29_rate', 'education_master', 'age_55_64_rate', 'hu_monthly_owner_costs_greaterthan_1000dlrs', 'age_25_34_rate', 'age_65_74_rate', 'heating_fuel_coal_coke_rate', 'population_density', 'own_popden', 'education_bachelor'],
    
    "mo1": ['education_bachelor', 'own_popden', 'heating_fuel_coal_coke_rate', 'occupation_agriculture_rate', 'housing_unit_median_gross_rent', 'pop_hispanic', 'Pro_Occup', 'age_25_34_rate', 'diversity', 'occupation_finance_rate', 'age_55_or_more_rate', 'transportation_car_alone_rate', 'travel_time_60_89_rate', 'occupation_information_rate', 'transportation_carpool_rate', 'employ_rate', 'hu_vintage_1960to1970', 'occupation_construction_rate', 'occupation_arts_rate', '%hh_size_2', 'occupation_transportation_rate', 'transportation_walk_rate', 'occupation_public_rate', 'travel_time_40_59_rate', 'occupation_wholesale_rate', 'education_college_rate', '%hh_size_4', 'age_18_24_rate', 'age_more_than_85_rate', 'age_10_14_rate'],
    
    "mo": [],
    
    "ms_top":['age_18_24_rate', 'education_master_rate', 'age_45_54_rate', 'occupancy_owner_rate', 'travel_time_60_89_rate', 'occupation_wholesale_rate', 'low_hh_rate', 'age_5_9_rate', '%hh_size_1', 'hu_1980to1999_pct', 'transportation_walk_rate', 'high_sf_rent_bldg_cnt', 'occupation_information_rate', 'mod_sf_rent_devp_cnt', 'mid_hh_rate', 'pop_hispanic', 'transportation_carpool_rate', 'Pro_Occup', 'age_more_than_85_rate', 'hu_1960to1979_pct', 'hu_monthly_owner_costs_greaterthan_1000dlrs', 'occupation_arts_rate', 'education_professional_school_rate', '%hh_size_2', 'very_low_mf_rent_elep_hh', 'occupation_transportation_rate', 'cooling_design_temperature', 'area_km2', 'pop_asian', 'travel_time_40_59_rate', 'Green_Travelers', 'very_low_sf_rent_elep_hh', 'high_sf_rent_mwh', 'pop_nat_us_citizen', 'occupation_education_rate', 'cdd', 'transportation_car_alone_rate', 'low_own_elep_hh', 'high_own_mwh', 'travel_time_40_89_rate', 'age_10_14_rate', 'occupation_administrative_rate', 'occupation_public_rate', 'employ_rate', 'occupation_finance_rate', 'diversity', 'low_mf_rent_elep_hh', 'hh_size_2', 'age_15_17_rate', 'mod_sf_rent_elep_hh', 'hu_rent', 'education_high_school_or_below_rate', 'hh_size_3', 'housing_unit_median_value', 'total_area', '%hh_size_3', 'education_bachelor', 'education_high_school_graduate_rate', 'population_density', 'own_popden'],
    
    "ms1": ['own_popden', 'education_high_school_or_below_rate', 'pop_asian', 'area_km2', 'low_mf_rent_elep_hh', 'travel_time_60_89_rate', 'Pro_Occup', 'low_own_elep_hh', 'hu_monthly_owner_costs_greaterthan_1000dlrs', 'hu_rent', 'occupation_administrative_rate', 'occupation_finance_rate', 'employ_rate', 'pop_hispanic', 'travel_time_40_59_rate', 'occupation_arts_rate', 'mod_sf_rent_elep_hh', 'occupation_information_rate', 'mid_hh_rate', 'cooling_design_temperature', 'transportation_carpool_rate', 'low_hh_rate', '%hh_size_3', 'occupation_transportation_rate', 'mod_sf_rent_devp_cnt', '%hh_size_2', 'hu_1960to1979_pct', '%hh_size_1', 'Green_Travelers', 'diversity'],
    
    "ms": [],
    
    
    "mt_top": ['travel_time_less_than_10_rate', 'age_25_64_rate', 'p16_unemployed', 'very_low_mf_own_mwh', 'hu_monthly_owner_costs_lessthan_1000dlrs', 'occupation_education_rate', 'mod_mf_rent_hh', 'occupation_manufacturing_rate', 'occupation_construction_rate', 'very_low_sf_rent_bldg_cnt', 'hu_no_mortgage', 'housing_unit_median_gross_rent', 'p16_employed', 'low_mf_rent_hh', 'transportation_home_rate', 'heating_fuel_other', 'occupation_agriculture_rate', 'area_km2', 'pop_non_us_citizen', 'occupation_public_rate', 'occupation_retail_rate', 'education_doctoral', 'travel_time_60_89_rate', 'age_25_34_rate', 'low_mf_own_devp_cnt', 'age_45_54_rate', 'pop_hispanic', 'travel_time_20_29_rate', 'age_55_64_rate', 'age_18_24_rate', 'cdd', 'age_10_14_rate', 'mid_mf_rent_hh', '%hh_size_2', 'total_area', 'hh_gini_index', 'hdd', 'fam_children_under_6', 'occupation_transportation_rate', 'heating_fuel_fuel_oil_kerosene_rate', 'own_popden', 'heating_fuel_other_rate', 'high_mf_rent_elep_hh', 'travel_time_average', 'heating_fuel_gas_rate', 'age_15_17_rate', 'hu_1959toearlier', 'hu_vintage_1940to1959', 'very_low_sf_rent_hh', 'travel_time_40_59_rate', 'very_low_mf_rent_hh', 'population_density', 'travel_time_40_89_rate', 'very_low_sf_rent_mwh', 'heating_fuel_coal_coke_rate', 'hu_1960to1979_pct', 'hu_vintage_1960to1970', 'heating_fuel_gas', 'age_25_44_rate', 'hu_rent'],
    
    "mt1":['hu_rent', 'hu_vintage_1960to1970', 'population_density', 'heating_fuel_coal_coke_rate', 'travel_time_40_89_rate', 'age_25_44_rate', 'occupation_agriculture_rate', 'hu_vintage_1940to1959', 'heating_fuel_fuel_oil_kerosene_rate', 'low_mf_own_devp_cnt', 'pop_hispanic', 'fam_children_under_6', 'heating_fuel_other_rate', 'age_45_54_rate', 'education_doctoral', 'age_15_17_rate', 'pop_non_us_citizen', 'occupation_education_rate', 'occupation_retail_rate', 'age_18_24_rate', '%hh_size_2', 'housing_unit_median_gross_rent', 'high_mf_rent_elep_hh', 'occupation_manufacturing_rate', 'occupation_public_rate', 'hdd', 'occupation_transportation_rate', 'age_10_14_rate', 'age_25_64_rate', 'occupation_construction_rate'],
    
    "mt": [],
    
    "nc_top": ['hh_gini_index', 'age_55_64_rate', 'education_doctoral_rate', 'occupation_education_rate', 'age_65_74_rate', 'age_5_9_rate', 'mod_mf_own_devp_cnt', 'transportation_walk_rate', 'occupation_administrative_rate', 'pop_non_us_citizen', 'age_10_14_rate', 'heating_fuel_electricity_rate', 'area_km2', 'employ_rate', '%hh_size_3', 'hu_2000toafter_pct', 'age_25_64_rate', 'occupation_wholesale_rate', 'age_45_54_rate', 'hu_vintage_1960to1970', 'occupation_information_rate', 'diversity', 'education_professional_school', 'housing_unit_median_gross_rent', '%hh_size_2', 'occupation_arts_rate', 'occupation_manufacturing_rate', 'number_of_years_of_education', 'occupation_retail_rate', 'age_25_34_rate', 'occupation_public_rate', 'education_college_rate', 'Anti_Occup', 'age_35_44_rate', 'age_15_17_rate', 'age_more_than_85_rate', 'occupation_agriculture_rate', 'travel_time_60_89_rate', 'occupation_transportation_rate', 'high_mf_own_mw', 'occupation_finance_rate', 'transportation_carpool_rate', 'occupation_construction_rate', 'hu_1960to1979_pct', 'education_bachelor_rate', 'travel_time_40_59_rate', 'mid_mf_own_devp_cnt', 'age_18_24_rate', 'hu_1980to1999_pct', 'travel_time_20_29_rate', 'education_doctoral', 'high_mf_own_devp_cnt', 'heating_fuel_coal_coke_rate', 'travel_time_40_89_rate', 'housing_unit_median_value', 'education_bachelor', 'travel_time_less_than_10_rate', 'education_master', 'own_popden', 'population_density'],
    
    "nc1": ['own_popden', 'education_bachelor', 'Anti_Occup', 'mid_mf_own_devp_cnt', 'occupation_agriculture_rate', 'education_doctoral', 'housing_unit_median_value', 'heating_fuel_coal_coke_rate', 'housing_unit_median_gross_rent', 'area_km2', 'occupation_finance_rate', 'travel_time_40_89_rate', 'occupation_administrative_rate', 'occupation_construction_rate', 'pop_non_us_citizen', 'employ_rate', 'transportation_carpool_rate', 'age_65_74_rate', 'occupation_information_rate', 'occupation_transportation_rate', 'hu_2000toafter_pct', 'travel_time_less_than_10_rate', 'age_25_34_rate', 'occupation_public_rate', 'occupation_arts_rate', 'heating_fuel_electricity_rate', 'education_college_rate', 'occupation_wholesale_rate', 'age_45_54_rate', 'age_5_9_rate', 'occupation_education_rate', 'age_35_44_rate'],
    
    "nc": [],
    
    
    
    "nd_top": ['age_more_than_85_rate', 'education_doctoral_rate', 'mod_mf_rent_elep_hh', 'education_high_school_graduate_rate', 'education_master_or_above_rate', 'age_65_74_rate', 'housing_unit_median_value', 'pop_us_citizen', 'very_low_mf_rent_hh', 'age_15_17_rate', 'education_master', 'hu_vintage_2010toafter', 'transportation_walk_rate', 'age_55_or_more_rate', 'p16_employed', 'housing_unit_median_gross_rent', '%hh_size_2', 'age_35_44_rate', 'occupation_manufacturing_rate', 'age_45_54_rate', 'travel_time_60_89_rate', 'occupancy_owner_rate', 'pop_over_65', 'low_sf_own_bldg_cnt', 'travel_time_average', 'male_pct', 'transportation_home_rate', 'female_pct', 'low_own_mwh', 'hu_1980to1999_pct', 'occupation_retail_rate', 'heating_fuel_gas_rate', 'poverty_family_below_poverty_level_rate', 'poverty_family_below_poverty_level', 'hh_size_1', '%hh_size_1', 'occupation_public_rate', 'transportation_bicycle_rate', 'heating_fuel_fuel_oil_kerosene', 'employ_rate', 'education_professional_school_rate', 'hu_1960to1979_pct', 'Green_Travelers', 'pop_nat_us_citizen', 'occupation_education_rate', 'heating_fuel_coal_coke_rate', 'hu_2000toafter_pct', '%hh_size_3', 'occupation_information_rate', 'pop_hispanic', 'occupation_wholesale_rate', 'hu_vintage_1980to1999', 'low_sf_rent_hh', 'heating_fuel_gas', 'hh_gini_index', 'education_college_rate', 'heating_fuel_fuel_oil_kerosene_rate', 'heating_fuel_none', 'fam_children_under_6', 'hu_2000toafter'],
    
    "nd1": ['hu_2000toafter', 'heating_fuel_fuel_oil_kerosene_rate', 'hu_vintage_1980to1999', 'housing_unit_median_value', 'Green_Travelers', 'age_65_74_rate', 'pop_nat_us_citizen', 'heating_fuel_none', 'pop_hispanic', 'heating_fuel_coal_coke_rate', 'low_sf_own_bldg_cnt', 'poverty_family_below_poverty_level', 'very_low_mf_rent_hh', 'education_master_or_above_rate', 'age_15_17_rate', '%hh_size_2', 'occupation_retail_rate', 'heating_fuel_gas_rate', 'occupation_wholesale_rate', 'travel_time_average', 'education_professional_school_rate', 'occupation_information_rate', 'age_45_54_rate', 'travel_time_60_89_rate', '%hh_size_3', 'male_pct', 'hu_1960to1979_pct', 'transportation_bicycle_rate', 'age_more_than_85_rate', 'occupancy_owner_rate', 'occupation_public_rate', 'age_35_44_rate'],
    
    "nd": [],
    
    "ne": [], 
    "ne_top": ['%hh_size_4', 'mod_sf_rent_devp_cnt', 'education_doctoral_rate', 'age_55_64_rate', 'mod_sf_rent_bldg_cnt', 'education_master', 'transportation_walk_rate', 'occupation_administrative_rate', 'hu_vintage_1940to1959', 'occupation_arts_rate', 'education_doctoral', 'hu_mortgage', 'age_25_64_rate', 'occupation_manufacturing_rate', 'travel_time_60_89_rate', 'low_sf_rent_bldg_cnt', 'hu_vintage_1960to1970', 'employ_rate', '%hh_size_3', 'diversity', 'pop_african_american', 'education_professional_school_rate', 'occupation_information_rate', 'education_master_or_above_rate', 'age_15_17_rate', 'age_35_44_rate', 'occupation_construction_rate', 'hu_1960to1979_pct', 'very_low_mf_rent_hh', 'p16_employed', 'p16_unemployed', 'total_area', 'fam_children_under_6', 'Green_Travelers', 'transportation_carpool_rate', 'low_mf_rent_hh', 'population_density', 'travel_time_40_89_rate', 'very_low_sf_rent_devp_cnt', 'occupation_finance_rate', 'hu_vintage_2000to2009', 'transportation_home_rate', 'high_mf_rent_bldg_cnt', 'hh_size_3', 'very_low_mf_rent_devp_cnt', 'education_college_rate', 'mod_mf_rent_devp_cnt', 'area_km2', 'occupation_retail_rate', 'occupation_education_rate', '%hh_size_2', 'low_mf_rent_bldg_cnt', 'travel_time_20_29_rate', 'high_mf_rent_hh', 'travel_time_40_59_rate', 'hh_gini_index', 'hh_size_1', 'occupation_wholesale_rate', 'occupation_agriculture_rate', 'heating_fuel_gas']
,
    "ne1": ['high_mf_rent_hh', 'p16_unemployed', 'hh_size_1', 'mod_mf_rent_devp_cnt', 'travel_time_40_89_rate', 'occupation_agriculture_rate', 'education_master_or_above_rate', 'hu_vintage_1960to1970', 'transportation_walk_rate', 'occupation_education_rate', 'education_professional_school_rate', 'hu_vintage_2000to2009', 'population_density', 'occupation_finance_rate', 'occupation_administrative_rate', 'occupation_information_rate', '%hh_size_3', 'education_college_rate', 'pop_african_american', 'low_sf_rent_bldg_cnt', '%hh_size_2', 'hu_vintage_1940to1959', 'age_15_17_rate', 'occupation_arts_rate', 'transportation_carpool_rate', 'occupation_wholesale_rate', 'employ_rate'],
    
    "nh_top": ['age_55_or_more_rate', 'low_mf_rent_devp_cnt', 'heating_fuel_gas_rate', 'very_low_mf_own_hh', 'diversity', 'age_45_54_rate', 'travel_time_40_59_rate', 'mod_own_Sbldg_rt', 'transportation_walk_rate', 'housing_unit_median_value', 'Anti_Occup', 'occupation_wholesale_rate', 'pop_under_18', 'hh_size_1', 'employ_rate', 'heating_fuel_electricity_rate', 'low_own_Sbldg_rt', '%hh_size_1', 'age_5_9_rate', 'pop_female', 'occupation_public_rate', 'occupation_administrative_rate', 'hu_1959toearlier', 'heating_fuel_fuel_oil_kerosene_rate', 'mod_mf_rent_devp_cnt', 'hu_1960to1979_pct', 'mod_mf_own_bldg_cnt', 'hu_vintage_1939toearlier', 'hu_2000toafter_pct', 'mid_mf_own_bldg_cnt', 'occupation_manufacturing_rate', 'age_25_34_rate', 'travel_time_average', 'hu_rent', 'hu_vintage_1960to1970', 'travel_time_20_29_rate', 'hu_1959toearlier_pct', 'heating_fuel_fuel_oil_kerosene', 'mid_mf_own_hh', 'heating_fuel_coal_coke', 'age_more_than_85_rate', 'housing_unit_median_gross_rent', 'transportation_car_alone_rate', 'occupation_transportation_rate', 'occupation_agriculture_rate', 'hu_vintage_1940to1959', 'pop_nat_us_citizen', 'pop25_high_school', 'education_high_school_graduate', 'age_55_64_rate', 'pop_hispanic', 'heating_fuel_electricity', 'occupation_information_rate', 'total_area', 'age_35_44_rate', 'travel_time_less_than_10_rate', 'p16_unemployed', 'own_popden', 'heating_fuel_coal_coke_rate', 'population_density'],
    "nh1": ['heating_fuel_coal_coke_rate', 'population_density', 'heating_fuel_electricity', 'occupation_agriculture_rate', 'age_55_64_rate', 'p16_unemployed', 'pop_nat_us_citizen', 'heating_fuel_gas_rate', 'pop_hispanic', 'hu_vintage_1940to1959', 'very_low_mf_own_hh', 'diversity', 'mod_mf_own_bldg_cnt', 'age_25_34_rate', 'mid_mf_own_hh', 'travel_time_average', 'hu_1960to1979_pct', 'hu_2000toafter_pct', 'occupation_administrative_rate', 'Anti_Occup', 'employ_rate', 'housing_unit_median_gross_rent', 'occupation_wholesale_rate', '%hh_size_1', 'age_5_9_rate', 'age_45_54_rate', 'occupation_information_rate', 'occupation_public_rate', 'occupation_transportation_rate', 'age_35_44_rate', 'travel_time_20_29_rate', 'hu_vintage_1939toearlier', 'mod_own_Sbldg_rt'],
    
    "nh": [],
    
    
    "nj_top": [],
    "nj1": [],
    "nj": [],
    
    "nm_top": [],
    "nm1": [],
    "nm": [],
    
    "nv_top": [],
    "nv1": [],
    "nv": [],
    
    "ny_top": ['Pro_Occup', 'pop_native_american', 'heating_fuel_electricity', 'employ_rate', 'transportation_walk_rate', 'age_65_74_rate', 'fam_children_under_6', 'hh_gini_index', 'heating_fuel_gas_rate', 'hu_1960to1979_pct', 'hu_vintage_1939toearlier', 'hu_1980to1999_pct', 'pop_nat_us_citizen', 'occupation_finance_rate', 'occupation_arts_rate', 'heating_fuel_electricity_rate', 'pop_non_us_citizen', 'occupation_agriculture_rate', 'transportation_public_rate', 'age_10_14_rate', 'age_45_54_rate', 'occupation_information_rate', 'occupation_administrative_rate', 'diversity', 'Anti_Occup', 'occupation_education_rate', 'age_more_than_85_rate', 'age_35_44_rate', 'occupation_public_rate', 'fam_children_6to17', 'transportation_carpool_rate', 'occupation_construction_rate', 'occupation_retail_rate', 'occupation_manufacturing_rate', 'transportation_car_alone_rate', '%hh_size_2', 'transportation_home_rate', 'occupation_wholesale_rate', 'age_5_9_rate', '%hh_size_3', 'p16_employed', 'age_18_24_rate', 'age_15_17_rate', 'age_75_84_rate', 'hu_vintage_1940to1959', 'high_mf_own_devp_cnt', 'travel_time_40_59_rate', 'occupancy_owner_rate', 'housing_unit_median_value', 'travel_time_20_29_rate', 'travel_time_less_than_10_rate', 'total_area', 'area_km2', 'high_mf_rent_bldg_cnt', 'heating_fuel_coal_coke_rate', 'own_popden', 'very_low_sf_own_elep_hh', 'hu_monthly_owner_costs_greaterthan_1000dlrs', 'population_density', 'high_own_Sbldg'],
    "ny1": [],
    "ny": [],
    
    "oh_top": ['age_25_64_rate', 'fam_children_under_6', 'heating_fuel_fuel_oil_kerosene_rate', 'housing_unit_median_gross_rent', 'avg_inc_ebill_dlrs', 'hu_1960to1979_pct', 'pop_nat_us_citizen', 'occupation_arts_rate', 'low_sf_rent_elep_hh', 'age_65_74_rate', 'Anti_Occup', 'hh_gini_index', 'age_55_64_rate', 'hu_1980to1999_pct', 'age_25_34_rate', 'age_5_9_rate', 'pop_african_american', 'fam_children_6to17', 'age_more_than_85_rate', '%hh_size_2', 'diversity', 'transportation_home_rate', 'hdd', 'transportation_carpool_rate', 'education_college_rate', '%hh_size_3', 'heating_fuel_gas_rate', 'occupation_wholesale_rate', 'occupation_retail_rate', 'age_18_24_rate', 'occupation_manufacturing_rate', 'travel_time_60_89_rate', 'occupation_finance_rate', 'heating_fuel_electricity', 'age_45_54_rate', 'travel_time_20_29_rate', 'travel_time_40_59_rate', 'occupation_administrative_rate', 'own_popden', 'age_35_44_rate', 'occupation_transportation_rate', 'hu_vintage_1940to1959', 'age_15_17_rate', 'age_10_14_rate', 'travel_time_40_89_rate', 'occupation_public_rate', 'occupation_construction_rate', 'occupation_information_rate', 'pop_hispanic', 'area_km2', 'heating_fuel_coal_coke', 'cdd', 'education_master', 'travel_time_average', 'travel_time_less_than_10_rate', 'population_density', 'education_bachelor', 'heating_fuel_coal_coke_rate', 'total_area', 'pop25_some_college_plus'],
    "oh1": ['pop25_some_college_plus', 'travel_time_average', 'avg_inc_ebill_dlrs', 'travel_time_less_than_10_rate', 'heating_fuel_electricity', 'pop_hispanic', 'travel_time_60_89_rate', 'occupation_finance_rate', 'Anti_Occup', 'heating_fuel_coal_coke_rate', 'occupation_administrative_rate', 'hu_vintage_1940to1959', 'hu_1980to1999_pct', 'transportation_carpool_rate', 'hu_1960to1979_pct', 'occupation_information_rate', 'occupation_construction_rate', 'housing_unit_median_gross_rent', 'low_sf_rent_elep_hh', 'occupation_transportation_rate', 'age_more_than_85_rate', 'cdd', 'own_popden', 'heating_fuel_fuel_oil_kerosene_rate', 'age_18_24_rate', 'age_15_17_rate', 'diversity', 'age_10_14_rate', 'age_25_34_rate'],
    "oh": [],
    
    
    "ok_top": [],
    "ok1": ['education_college_rate', 'pop_african_american', 'high_mf_rent_mwh', 'average_household_income', '%hh_size_3', 'age_25_44_rate', 'education_less_than_high_school_rate', 'travel_time_60_89_rate', 'mod_mf_rent_elep_hh', 'occupation_transportation_rate', 'hu_vintage_1980to1999', 'occupation_finance_rate', 'pop_native_american', 'age_75_84_rate', 'Green_Travelers', 'occupation_information_rate', 'low_mf_rent_devp_cnt', 'occupation_administrative_rate', 'education_high_school_graduate_rate', 'cdd', 'population_density', 'pop_nat_us_citizen', 'mid_own_Sbldg_rt', 'age_55_64_rate', 'pop25_some_college_plus', 'very_low_sf_rent_devp_cnt', 'transportation_home_rate', 'age_15_17_rate', 'age_45_54_rate', 'hu_1960to1979_pct', 'hh_gini_index', 'age_25_64_rate', 'travel_time_40_59_rate', 'hu_1980to1999_pct', 'age_10_14_rate', 'male_pct', 'female_pct', 'occupation_manufacturing_rate', 'occupation_wholesale_rate', 'diversity', 'age_25_34_rate', 'occupation_arts_rate', 'age_5_9_rate', 'pop_non_us_citizen', 'high_sf_own_devp_m2', 'age_more_than_85_rate', 'travel_time_less_than_10_rate', 'occupation_retail_rate', 'own_popden', 'high_sf_own_mw', 'high_sf_own_mwh', 'hdd', 'occupation_construction_rate', 'housing_unit_median_value', 'travel_time_average', 'high_own_mwh', 'Tot_own_mw', 'education_bachelor', 'heating_fuel_coal_coke_rate', 'Yr_own_mwh', ]
,
    "ok": [],
    
    "or": [],
    "or_top": [],
    "or1": [],
    
    "pa1": ['high_mf_own_bldg_cnt', 'education_master', 'p16_employed', 'pop_nat_us_citizen', 'pop_caucasian', 'heating_fuel_electricity', 'occupation_finance_rate', 'occupation_administrative_rate', 'employ_rate', 'hu_1960to1979_pct', 'occupation_agriculture_rate', 'occupation_information_rate', 'occupation_transportation_rate', 'Anti_Occup', 'travel_time_average', 'hu_vintage_1940to1959', 'heating_fuel_coal_coke_rate', 'transportation_carpool_rate', 'hu_1959toearlier_pct', 'heating_fuel_fuel_oil_kerosene', 'occupation_construction_rate', 'own_popden', 'occupation_wholesale_rate', 'age_more_than_85_rate', 'transportation_home_rate', 'diversity', 'age_35_44_rate', 'occupation_public_rate', '%hh_size_3', 'education_college_rate', 'age_55_64_rate'],
    "pa_top": ['pop_nat_us_citizen', 'heating_fuel_electricity', 'p16_unemployed', 'occupation_education_rate', 'age_10_14_rate', 'hu_1959toearlier_pct', 'age_35_44_rate', 'transportation_home_rate', 'diversity', 'high_sf_own_mwh', 'heating_fuel_gas_rate', 'hu_vintage_1940to1959', 'Anti_Occup', 'pop_asian', 'age_5_9_rate', 'education_college_rate', 'p16_employed', 'pop_caucasian', 'occupation_manufacturing_rate', 'age_15_17_rate', 'employ_rate', '%hh_size_3', 'occupation_wholesale_rate', 'occupation_construction_rate', 'age_65_74_rate', 'heating_fuel_fuel_oil_kerosene', 'hh_size_1', 'age_25_64_rate', 'occupation_arts_rate', 'occupation_information_rate', 'hu_1960to1979_pct', '%hh_size_2', 'occupation_transportation_rate', 'travel_time_20_29_rate', 'high_mf_rent_bldg_cnt', 'heating_fuel_coal_coke_rate', 'occupation_administrative_rate', 'age_more_than_85_rate', 'age_55_64_rate', 'occupation_public_rate', 'high_mf_own_devp_cnt', 'travel_time_average', 'occupation_finance_rate', 'transportation_carpool_rate', 'occupation_agriculture_rate', 'occupation_retail_rate', 'travel_time_60_89_rate', 'high_sf_own_hh', 'housing_unit_median_value', 'total_area', 'travel_time_40_89_rate', 'travel_time_less_than_10_rate', 'travel_time_40_59_rate', 'own_popden', 'area_km2', 'high_own_mwh', 'education_master', 'high_mf_own_bldg_cnt', 'population_density', 'education_bachelor'],
    "pa": [],
    
    "ri_top": [],
    "ri1": [],
    "ri": [],
    
    
    
    "sc_top":['education_less_than_high_school_rate', 'age_45_54_rate', 'age_25_34_rate', 'hu_1960to1979_pct', 'high_mf_rent_bldg_cnt', 'heating_fuel_electricity', 'occupation_finance_rate', 'age_75_84_rate', 'age_65_74_rate', 'p16_unemployed', 'occupation_wholesale_rate', 'heating_fuel_electricity_rate', 'education_bachelor', 'age_25_64_rate', 'high_mf_rent_devp_cnt', 'occupation_information_rate', 'mid_sf_rent_hh', 'hu_vintage_2010toafter', 'age_55_64_rate', 'age_18_24_rate', 'fam_children_under_6', 'transportation_carpool_rate', 'hu_vintage_2000to2009', 'high_sf_own_devp_cnt', 'mid_mf_rent_bldg_cnt', 'heating_fuel_gas_rate', 'pop_nat_us_citizen', 'age_35_44_rate', '%hh_size_3', 'occupation_agriculture_rate', 'occupation_construction_rate', 'p16_employed', 'heating_fuel_fuel_oil_kerosene_rate', 'occupation_transportation_rate', 'transportation_walk_rate', 'travel_time_60_89_rate', 'occupation_arts_rate', 'occupation_administrative_rate', 'very_low_mf_rent_elep_hh', 'occupation_retail_rate', 'employ_rate', 'education_college_rate', 'travel_time_less_than_10_rate', 'high_mf_rent_elep_hh', 'dlrs_kwh x median_household_income', 'occupation_public_rate', 'low_mf_rent_elep_hh', 'Tot_own_mw', 'high_mf_rent_mwh', 'travel_time_average', 'high_mf_rent_hh', 'mod_mf_rent_elep_hh', 'mid_mf_own_bldg_cnt', 'hu_rent', 'avg_inc_ebill_dlrs', 'age_15_17_rate', 'housing_unit_median_gross_rent', 'mid_mf_rent_hh', 'own_popden', 'population_density'],
    "sc1": ['own_popden', 'mid_mf_rent_hh', 'pop_nat_us_citizen', 'heating_fuel_fuel_oil_kerosene_rate', 'housing_unit_median_gross_rent', 'travel_time_average', 'occupation_agriculture_rate', 'hu_rent', 'employ_rate', 'low_mf_rent_elep_hh', 'occupation_administrative_rate', 'age_65_74_rate', 'hu_vintage_2010toafter', 'age_25_34_rate', 'fam_children_under_6', 'occupation_finance_rate', 'age_55_64_rate', 'transportation_carpool_rate', 'occupation_arts_rate', 'education_college_rate', 'occupation_transportation_rate', 'heating_fuel_gas_rate', 'travel_time_less_than_10_rate', 'occupation_information_rate', 'age_35_44_rate', 'age_45_54_rate', 'age_25_64_rate'],
    "sc": [],
    
    "sd_top": ['Tot_own_mw', 'age_65_74_rate', '%hh_size_4', 'occupation_manufacturing_rate', 'mod_own_hh', 'hu_rent', 'occupation_transportation_rate', 'poverty_family_below_poverty_level_rate', 'pop_non_us_citizen', 'mod_sf_own_mwh', 'Yr_own_mwh', 'employ_rate', 'mod_sf_own_bldg_cnt', 'occupation_construction_rate', 'hu_2000toafter_pct', '%hh_size_3', 'hu_1980to1999_pct', 'hu_2000toafter', 'low_mf_rent_bldg_cnt', 'heating_fuel_other', 'heating_fuel_electricity_rate', 'mod_mf_rent_bldg_cnt', 'education_less_than_high_school_rate', 'heating_fuel_other_rate', 'occupation_information_rate', 'age_15_17_rate', 'hu_1960to1979_pct', 'mid_own_mwh', 'hu_monthly_owner_costs_lessthan_1000dlrs', 'housing_unit_count', 'poverty_family_below_poverty_level', 'mod_mf_rent_elep_hh', 'low_mf_rent_devp_cnt', 'transportation_carpool_rate', '%hh_size_1', 'household_type_family_rate', 'occupation_wholesale_rate', 'age_10_14_rate', 'age_more_than_85_rate', 'hdd', 'average_household_size', 'mid_sf_own_mwh', 'diversity', 'travel_time_average', 'age_25_64_rate', 'occupation_finance_rate', 'mod_own_mwh', 'hu_vintage_2010toafter', 'cdd', 'occupation_administrative_rate', 'travel_time_20_29_rate', 'pop_hispanic', 'age_18_24_rate', 'age_45_54_rate', 'hu_no_mortgage', 'age_5_9_rate', 'travel_time_less_than_10_rate', 'occupation_education_rate', 'hh_size_1', 'heating_fuel_electricity'],
    "sd1": ['heating_fuel_electricity', 'mod_mf_rent_bldg_cnt', 'hu_monthly_owner_costs_lessthan_1000dlrs', 'hu_vintage_2010toafter', 'travel_time_average', 'mod_mf_rent_elep_hh', 'household_type_family_rate', 'occupation_administrative_rate', 'education_less_than_high_school_rate', 'pop_hispanic', 'pop_non_us_citizen', 'travel_time_20_29_rate', 'age_10_14_rate', 'age_15_17_rate', 'occupation_manufacturing_rate', 'age_18_24_rate', 'heating_fuel_other_rate', 'occupation_information_rate', 'heating_fuel_electricity_rate', 'occupation_finance_rate', 'occupation_education_rate', 'hu_1980to1999_pct', 'age_45_54_rate', 'cdd', 'age_5_9_rate', 'age_more_than_85_rate', 'age_65_74_rate', 'poverty_family_below_poverty_level'],
    "sd": [],
    
    "tn_top": ['occupation_arts_rate', 'age_10_14_rate', 'hu_1980to1999_pct', 'age_5_9_rate', 'occupation_manufacturing_rate', 'education_professional_school_rate', 'education_college_rate', 'occupation_education_rate', 'occupation_agriculture_rate', 'age_more_than_85_rate', 'travel_time_60_89_rate', 'age_15_17_rate', 'age_25_34_rate', 'hu_vintage_2010toafter', 'total_own_hh', 'transportation_car_alone_rate', 'age_75_84_rate', 'high_own_hh', 'hu_1959toearlier_pct', 'age_35_44_rate', 'Anti_Occup', 'occupation_retail_rate', 'heating_fuel_gas_rate', 'pop25_some_college_plus', 'occupation_construction_rate', 'hh_size_1', 'transportation_home_rate', 'hh_gini_index', 'occupation_finance_rate', 'age_45_54_rate', 'occupation_public_rate', 'hu_vintage_2000to2009', 'heating_fuel_gas', 'total_area', 'occupation_wholesale_rate', '%hh_size_3', 'age_18_24_rate', 'area_km2', 'hu_1960to1979_pct', 'travel_time_40_59_rate', 'pop_over_65', 'travel_time_20_29_rate', 'travel_time_40_89_rate', 'occupation_administrative_rate', 'transportation_carpool_rate', 'education_professional_school', 'education_bachelor', 'Green_Travelers', 'pop_caucasian', 'travel_time_average', 'own_popden', 'occupation_transportation_rate', 'high_own_mwh', 'diversity', 'occupation_information_rate', 'education_master', 'pop_african_american', 'travel_time_less_than_10_rate', 'heating_fuel_coal_coke_rate', 'population_density'],
    "tn1": ['pop25_some_college_plus', 'education_professional_school', 'heating_fuel_coal_coke_rate', 'own_popden', 'travel_time_average', 'Anti_Occup', 'occupation_agriculture_rate', 'hh_size_1', 'diversity', 'occupation_information_rate', 'occupation_transportation_rate', 'occupation_administrative_rate', 'occupation_construction_rate', 'pop_over_65', 'travel_time_less_than_10_rate', 'occupation_finance_rate', 'transportation_carpool_rate', 'pop_african_american', 'transportation_home_rate', 'education_college_rate', 'occupation_education_rate', 'heating_fuel_gas_rate', 'age_25_34_rate', 'hu_vintage_2010toafter', 'occupation_arts_rate', 'age_more_than_85_rate', 'age_45_54_rate', 'age_35_44_rate', 'hu_1960to1979_pct'],
    "tn": [],
    
    "tx": [],
    "tx": [],
    "tx": [],
    
    "ut": [],
    "ut": [],
    "ut": [],
    
    "va_top": ['heating_fuel_gas_rate', 'occupation_retail_rate', 'hu_1960to1979_pct', 'hu_vintage_2010toafter', 'age_75_84_rate', 'occupation_education_rate', 'heating_fuel_gas', 'heating_fuel_fuel_oil_kerosene_rate', 'hu_1980to1999_pct', 'high_sf_own_bldg_cnt', 'hu_1959toearlier_pct', 'hu_rent', 'transportation_walk_rate', 'employ_rate', 'education_bachelor', 'pop_non_us_citizen', 'occupation_transportation_rate', 'occupation_public_rate', 'high_sf_own_mwh', 'hu_2000toafter_pct', 'hu_vintage_1939toearlier', 'age_65_74_rate', 'area_km2', 'age_5_9_rate', 'travel_time_40_59_rate', 'occupation_construction_rate', 'occupation_information_rate', 'education_college_rate', 'age_18_24_rate', 'education_less_than_high_school_rate', 'occupation_finance_rate', 'transportation_home_rate', 'heating_fuel_fuel_oil_kerosene', 'occupation_wholesale_rate', 'occupation_manufacturing_rate', 'pop_us_citizen', 'travel_time_average', 'age_10_14_rate', 'hu_vintage_1940to1959', 'total_area', 'high_own_hh', 'age_15_17_rate', 'Anti_Occup', 'high_mf_own_devp_cnt', 'age_55_64_rate', 'transportation_carpool_rate', '%hh_size_2', 'age_45_54_rate', 'age_more_than_85_rate', 'occupation_arts_rate', 'Pro_Occup', 'travel_time_20_29_rate', 'travel_time_40_89_rate', 'travel_time_60_89_rate', 'pop_over_65', 'Green_Travelers', 'high_mf_rent_bldg_cnt', 'travel_time_less_than_10_rate', 'pop_caucasian', 'population_density'],
    "va1": ['high_mf_rent_bldg_cnt', 'education_bachelor', 'education_less_than_high_school_rate', 'Anti_Occup', 'pop_non_us_citizen', 'heating_fuel_gas_rate', 'pop_caucasian', 'occupation_finance_rate', 'population_density', 'travel_time_average', 'occupation_construction_rate', 'occupation_information_rate', 'employ_rate', 'area_km2', 'occupation_transportation_rate', 'hu_rent', 'heating_fuel_fuel_oil_kerosene_rate', 'age_55_64_rate', 'transportation_carpool_rate', 'Green_Travelers', 'travel_time_less_than_10_rate', '%hh_size_2', 'transportation_home_rate', 'occupation_public_rate', 'hu_vintage_1939toearlier', 'age_75_84_rate', 'occupation_retail_rate', 'occupation_arts_rate', 'education_college_rate'],
    "va": [],
    
    "vt_top": ['hh_gini_index', 'heating_fuel_fuel_oil_kerosene', 'hdd_std', 'heating_fuel_gas_rate', '%hh_size_3', 'poverty_family_below_poverty_level_rate', 'pop_native_american', 'heating_fuel_gas', 'hu_1960to1979_pct', 'travel_time_less_than_10_rate', 'verylow_own_mwh', 'occupancy_owner_rate', 'travel_time_60_89_rate', 'hdd', 'occupation_arts_rate', 'age_5_9_rate', 'mod_mf_own_hh', 'age_35_44_rate', 'travel_time_average', 'low_own_Sbldg_rt', 'daily_solar_radiation', 'age_55_or_more_rate', 'mid_mf_rent_devp_cnt', 'age_median', 'transportation_public_rate', 'occupation_wholesale_rate', 'pop_med_age', 'occupation_retail_rate', 'heating_fuel_electricity', 'pop_asian', 'mid_hh_rate', 'education_bachelor_rate', 'age_10_14_rate', 'mid_own_Sbldg_rt', 'p16_unemployed', 'age_18_24_rate', 'low_mf_rent_elep_hh', 'education_less_than_high_school_rate', 'transportation_walk_rate', 'occupation_agriculture_rate', 'mod_own_elep_hh', 'low_mf_own_hh', '%hh_size_4', 'transportation_motorcycle_rate', 'transportation_home_rate', 'hu_vintage_1940to1959', 'transportation_carpool_rate', 'travel_time_40_89_rate', 'mod_mf_own_elep_hh', 'travel_time_40_59_rate', 'diversity', 'heating_fuel_fuel_oil_kerosene_rate', 'hu_1959toearlier_pct', 'age_55_64_rate', 'transportation_car_alone_rate', 'pop_nat_us_citizen', 'pop_african_american', 'own_popden', 'population_density', 'heating_fuel_coal_coke_rate'],
    "vt1": ['heating_fuel_coal_coke_rate', 'population_density', 'occupation_agriculture_rate', 'hu_vintage_1940to1959', 'age_55_64_rate', 'pop_african_american', 'transportation_home_rate', 'hu_1959toearlier_pct', 'heating_fuel_gas', 'travel_time_40_89_rate', 'transportation_public_rate', 'verylow_own_mwh', 'transportation_walk_rate', 'travel_time_less_than_10_rate', 'hdd_std', 'poverty_family_below_poverty_level_rate', 'occupation_retail_rate', 'hu_1960to1979_pct', 'heating_fuel_fuel_oil_kerosene', 'low_own_Sbldg_rt', 'age_10_14_rate', 'daily_solar_radiation', 'transportation_carpool_rate', 'occupation_arts_rate', '%hh_size_3', 'education_bachelor_rate', 'education_less_than_high_school_rate', 'hh_gini_index'],
    "vt": [],
    
    "wa": [],
    "wa": [],
    "wa": [],
    
    "wi_top": ['transportation_home_rate', 'transportation_carpool_rate', 'education_high_school_graduate_rate', 'housing_unit_median_gross_rent', 'age_5_9_rate', 'mid_mf_rent_mwh', 'occupation_arts_rate', 'heating_fuel_gas_rate', 'pop_hispanic', 'age_18_24_rate', 'transportation_car_alone_rate', 'hh_gini_index', 'occupation_education_rate', 'Anti_Occup', 'occupation_finance_rate', 'education_master', 'Green_Travelers', 'occupation_public_rate', 'occupation_wholesale_rate', 'age_65_74_rate', 'age_25_64_rate', 'occupation_administrative_rate', 'heating_fuel_electricity_rate', 'diversity', 'hu_1959toearlier', 'travel_time_20_29_rate', 'hu_vintage_1939toearlier', 'age_35_44_rate', 'pop_native_american', 'age_15_17_rate', 'age_55_64_rate', 'hu_1959toearlier_pct', 'travel_time_40_89_rate', 'age_25_34_rate', 'average_household_size', 'age_10_14_rate', 'age_45_54_rate', 'occupation_retail_rate', 'occupation_information_rate', 'occupation_transportation_rate', 'heating_fuel_fuel_oil_kerosene_rate', 'hu_vintage_1940to1959', 'education_doctoral_rate', 'travel_time_60_89_rate', 'transportation_bicycle_rate', 'own_popden', 'education_college_rate', '%hh_size_2', '%hh_size_3', 'household_type_family_rate', 'hu_rent', 'area_km2', 'hh_size_2', 'travel_time_less_than_10_rate', 'heating_fuel_coal_coke_rate', 'total_area', 'heating_fuel_electricity', 'occupation_construction_rate', 'p16_employed', 'population_density'],
    "wi1": ['p16_employed', 'heating_fuel_electricity_rate', 'education_master', 'heating_fuel_coal_coke_rate', 'occupation_construction_rate', 'own_popden', 'heating_fuel_fuel_oil_kerosene_rate', 'Anti_Occup', 'travel_time_60_89_rate', 'hu_vintage_1940to1959', 'transportation_bicycle_rate', 'household_type_family_rate', 'pop_hispanic', 'occupation_administrative_rate', 'education_doctoral_rate', 'age_55_64_rate', 'age_25_34_rate', 'occupation_finance_rate', 'transportation_home_rate', 'travel_time_less_than_10_rate', 'housing_unit_median_gross_rent', 'heating_fuel_gas_rate', 'occupation_transportation_rate', 'occupation_information_rate', 'transportation_car_alone_rate', 'age_35_44_rate', 'diversity', 'age_18_24_rate', 'transportation_carpool_rate', 'pop_native_american', 'age_45_54_rate', 'occupation_retail_rate', 'occupation_wholesale_rate', 'occupation_arts_rate', 'age_15_17_rate', 'occupation_public_rate', 'education_college_rate'],
    "wi": [],
    
    "wv_top": ['cdd_ci', 'age_35_44_rate', 'education_bachelor_rate', 'age_25_44_rate', 'age_25_34_rate', 'travel_time_40_59_rate', 'high_mf_rent_bldg_cnt', 'travel_time_60_89_rate', 'age_25_64_rate', 'age_15_17_rate', 'occupation_information_rate', 'education_college', 'hdd', 'occupation_retail_rate', 'age_10_14_rate', 'hu_no_mortgage', 'occupation_administrative_rate', 'hu_1980to1999_pct', 'age_45_54_rate', 'age_75_84_rate', 'occupation_finance_rate', 'Green_Travelers', 'travel_time_40_89_rate', 'occupancy_owner_rate', 'occupation_arts_rate', 'transportation_home_rate', 'pop_male', 'occupation_construction_rate', 'cdd', 'travel_time_average', 'hh_gini_index', 'age_65_74_rate', 'hu_vintage_1960to1970', 'transportation_carpool_rate', 'age_55_64_rate', 'age_5_9_rate', 'low_mf_rent_hh', 'education_bachelor', 'education_bachelor_or_above_rate', 'pop_asian', 'hu_1959toearlier', 'pop_african_american', 'hu_1960to1979_pct', 'hu_rent', 'occupation_manufacturing_rate', 'pop_nat_us_citizen', 'hh_size_1', 'Pro_Occup', 'heating_fuel_coal_coke_rate', 'travel_time_less_than_10_rate', 'diversity', 'occupation_transportation_rate', 'population_density', 'Anti_Occup', 'pop25_some_college_plus', 'hu_vintage_1940to1959', 'occupation_agriculture_rate', 'education_high_school_or_below_rate', 'education_master', 'own_popden'],
    "wv1": ['own_popden', 'education_high_school_or_below_rate', 'pop25_some_college_plus', 'occupation_agriculture_rate', 'low_mf_rent_hh', 'heating_fuel_coal_coke_rate', 'pop_asian', 'travel_time_average', 'hh_size_1', 'hu_vintage_1940to1959', 'diversity', 'cdd', 'occupation_transportation_rate', 'travel_time_less_than_10_rate', 'occupation_arts_rate', 'age_65_74_rate', 'occupation_information_rate', 'age_55_64_rate', 'age_25_34_rate', 'hu_1980to1999_pct', 'occupation_administrative_rate', 'Green_Travelers', 'occupation_finance_rate', 'occupation_construction_rate', 'hh_gini_index', 'transportation_carpool_rate', 'hu_1960to1979_pct', 'transportation_home_rate', 'age_45_54_rate'],
    "wv": [],
    
   
}


US_Solar_Groupings = {
    'Very Low':['wa', 'vt', 'me', 'nh', 'mi', 'pa', 'mn', 'nd', 'ma', 'wi'],
    'Low':['oh', 'or', 'ct', 'wv', 'ri', 'ny', 'nj', 'in', 'il', 'md'],
    'Moderate':['ia', 'mt', 'de', 'ky', 'sd', 'dc', 'va', 'tn', 'mo', 'ne'],
    'High':['nc', 'ar', 'ks', 'id', 'ga', 'wy', 'al', 'sc', 'ms', 'la'],
    'Very High':['ok', 'co', 'tx', 'ut', 'fl', 'ca', 'nv', 'nm', 'az'],
}

USRegions = {
           'West':  [ 'ca', 'nv' ],        #2
           'N. West': ['wa', 'or', 'id', 'mt', 'wy',],  #5
           'S. West': ['ut', 'az', 'nm', 'co', 'tx', 'ok',], # 6
           'M. West': ['nd', 'sd', 'ne', 'ks', 'mo', 'mn', 'ia', 'il', 'in', 'ky', 'oh', 'mi'],    # 12
           'S. East': ['ar', 'la', 'ms', 'al', 'ga', 'fl', 'sc', 'nc', 'tn',],     # 10
           'Mid_atlantic': ['pa', 'dc', 'de', 'nj', 'ny', 'md', 'wv', 'va',],  # 8
           'N. East': ['ma', 'vt', 'me', 'nh', 'ri', 'ct']  # 6
}

US_4Major = {
    'NT3': Nt3state_abrev,
    'T3':  [ 'ca', 'nv', 'az',],
    'West':  [ 'ca', 'nv', 'wa', 'or', 'id', 'mt', 'wy', 'ut', 'az', 'nm', 'co',],
    'WestNT3':  ['wa', 'or', 'id', 'mt', 'wy', 'ut', 'nm', 'co',],
    'Mid West': ['nd', 'sd', 'ne', 'ks', 'mo', 'mn', 'ia', 'il', 'in','oh', 'mi','wi',],
    'South': ['tx', 'ok','ar', 'la', 'ms', 'al', 'ga', 'fl', 'sc', 'nc', 'tn','wv', 'va','ky',],
    'NorthEast': ['ma','vt', 'me', 'nh', 'ri', 'ct', 'pa', 'dc', 'de', 'nj', 'ny', 'md', ]
}

US_4Broken = {
    'Pacific':['ca','or','wa',],
    'Mountain':['nv','id', 'mt', 'wy', 'ut', 'az', 'nm', 'co',],
    'West N. Central':['nd', 'sd', 'ne', 'ks', 'mo', 'mn', 'ia',],
    'East N. Central':['wi','il', 'in','oh', 'mi',],
    'West South Central':['tx', 'ok','ar', 'la',],
    'East South Central':['ms', 'al','tn','ky',],
    'South Atlantic':['fl', 'sc', 'nc','va','wv','ga','dc','md','de',],
    'Middle Atlantic':['pa','ny','nj',],
    'New England': ['ct', 'me','vt','ri', 'nh','ma',],
}


solr_groupings_percentiles = {
    "low": ['nd', 'pa', 'va', 'me', 'ct', 'il', 'or', 'ri', 'ky', 'in',
            'ny', 'ma', 'nh', 'nj', 'wa', 'ia', 'dc', 'wv', 'vt', 'sd',
            'mn', 'mt', 'md', 'wi', 'de', 'oh', 'mi'],
    "high": ['id', 'ms', 'ut', 'mo', 'al', 'ca', 'co', 'nv', 'ok', 'wy',
             'ks', 'la', 'tx', 'sc', 'nc', 'ne', 'nm', 'az', 'ga', 'tn',
             'ar', 'fl'],
    "high_high": high_slr_high_e,
    "high_low": high_slr_low_e,
    "low_low": low_slr_low_e,
    "low_high": low_slr_high_e,
        }
five_level_solar_g = {
    "very low": ['vt', 'mi', 'pa', 'oh', 'nd', 'mn', 'nh', 'ma', 'wa', 'me', 'wi'],
    "low": ['in', 'ia', 'nj', 'ct', 'ri', 'sd', 'or', 'de', 'mt', 'ny', 'il', 'md', 'ky', 'wv'],
    "moderate": ['mo', 'nc', 'dc', 'ne', 'ks', 'va', 'ar', 'tn'],
    "medium": ['id', 'ga', 'ut', 'al', 'ok', 'co', 'wy', 'ms', 'la', 'tx', 'sc'],
    "high": ['fl', 'ca', 'az', 'nm', 'nv'],
}

variables_by_group = {
"education": ['education_bachelor', 'education_bachelor_or_above_rate', 'education_bachelor_rate', 'education_college',
              'education_college_rate', 'education_doctoral', 'education_doctoral_rate',
              'education_high_school_graduate', 'education_high_school_graduate_rate',
              'education_high_school_or_below_rate', 'education_less_than_high_school',
              'education_less_than_high_school_rate', 'education_master', 'education_master_or_above_rate',
              'education_master_rate', 'education_population', 'education_professional_school',
              'education_professional_school_rate', 'number_of_years_of_education'],
"age": ['age_10_14_rate', 'age_15_17_rate', 'age_18_24_rate', 'age_25_34_rate', 'age_25_44_rate', 'age_25_64_rate',
        'age_35_44_rate', 'age_45_54_rate', 'age_55_64_rate', 'age_55_or_more_rate', 'age_5_9_rate', 'age_65_74_rate',
        'age_75_84_rate', 'age_median', 'age_more_than_85_rate', 'fam_children_6to17', 'fam_children_under_6'],
"policy_ownership": ['net_metering_hu_own'],
"policy_income": ['dlrs_kwh x median_household_income'],
"income": ['average_household_income', 'fam_med_income', 'hh_gini_index',
           'median_household_income'],
"income_housing": ['high_mf_own_bldg_cnt', 'high_mf_own_hh', 'high_mf_rent_bldg_cnt', 'high_mf_rent_hh',
                   'high_sf_own_bldg_cnt', 'high_sf_own_hh', 'high_sf_rent_hh', 'low_mf_own_bldg_cnt',
                   'low_mf_own_hh', 'low_mf_rent_bldg_cnt', 'low_mf_rent_hh', 'low_sf_own_bldg_cnt', 'low_sf_own_hh',
                   'low_sf_rent_bldg_cnt', 'low_sf_rent_hh', 'mid_mf_own_bldg_cnt', 'mid_mf_own_hh',
                   'mid_mf_rent_bldg_cnt', 'mid_mf_rent_hh', 'mid_sf_own_bldg_cnt', 'mid_sf_own_hh',
                   'mid_sf_rent_bldg_cnt', 'mid_sf_rent_hh', 'mod_mf_own_hh', 'mod_mf_rent_bldg_cnt',
                   'mod_mf_rent_hh', 'mod_sf_own_bldg_cnt', 'mod_sf_own_hh', 'mod_sf_rent_bldg_cnt',
                   'mod_sf_rent_hh', 'very_low_mf_own_bldg_cnt', 'very_low_mf_own_hh', 'very_low_mf_rent_bldg_cnt',
                   'very_low_mf_rent_elep_hh', 'very_low_sf_own_bldg_cnt', 'very_low_sf_own_hh',
                   'very_low_sf_rent_bldg_cnt', 'very_low_sf_rent_hh'],
"inc/homes": list(set(['high_mf_own_bldg_cnt', 'high_mf_own_hh', 'high_mf_rent_bldg_cnt', 'high_mf_rent_hh',
                   'high_sf_own_bldg_cnt', 'high_sf_own_hh', 'high_sf_rent_hh', 'low_mf_own_bldg_cnt',
                   'low_mf_own_hh', 'low_mf_rent_bldg_cnt', 'low_mf_rent_hh', 'low_sf_own_bldg_cnt', 'low_sf_own_hh',
                   'low_sf_rent_bldg_cnt', 'low_sf_rent_hh', 'mid_mf_own_bldg_cnt', 'mid_mf_own_hh',
                   'mid_mf_rent_bldg_cnt', 'mid_mf_rent_hh', 'mid_sf_own_bldg_cnt', 'mid_sf_own_hh',
                   'mid_sf_rent_bldg_cnt', 'mid_sf_rent_hh', 'mod_mf_own_hh', 'mod_mf_rent_bldg_cnt',
                   'mod_mf_rent_hh', 'mod_sf_own_bldg_cnt', 'mod_sf_own_hh', 'mod_sf_rent_bldg_cnt',
                   'mod_sf_rent_hh', 'very_low_mf_own_bldg_cnt', 'very_low_mf_own_hh', 'very_low_mf_rent_bldg_cnt',
                   'very_low_mf_rent_elep_hh', 'very_low_sf_own_bldg_cnt', 'very_low_sf_own_hh',
                   'very_low_sf_rent_bldg_cnt', 'very_low_sf_rent_hh'] + l_b_rt + l2_b_c + l3_mwh + l4_elep +
                    own_hh_l+ own_rt_l)),
"energy_income": ['avg_inc_ebill_dlrs'],
"occupation": ['Pro_Occup', 'occupancy_owner_rate', 'occupation_administrative_rate', 'occupation_agriculture_rate',
               'occupation_arts_rate', 'occupation_construction_rate', 'occupation_education_rate',
               'occupation_finance_rate', 'occupation_information_rate', 'occupation_manufacturing_rate',
               'occupation_public_rate', 'occupation_retail_rate', 'occupation_transportation_rate',
               'occupation_wholesale_rate', 'Anti_Occup'],
"habit": ['Green_Travelers', 'avg_monthly_bill_dlrs', 'avg_monthly_consumption_kwh', 'transportation_bicycle_rate',
          'transportation_car_alone_rate', 'transportation_carpool_rate', 'transportation_home_rate',
          'transportation_motorcycle_rate', 'transportation_public_rate', 'transportation_walk_rate',
          'travel_time_10_19_rate', 'travel_time_20_29_rate', 'travel_time_30_39_rate', 'travel_time_40_59_rate',
          'travel_time_40_89_rate', 'travel_time_60_89_rate', 'travel_time_average', 'travel_time_less_than_10_rate'],
"geography": ['Tot_own_mw', 'area_km2', 'land_area', 'locale_dummy', 'locale_recode(rural)', 'locale_recode(suburban)',
              'locale_recode(town)', 'total_area'],
"demo": ['diversity', 'employ_rate', 'household_type_family_rate', 'pct_eli_hh', 'pop_african_american', 'pop_asian',
         'pop_caucasian', 'pop_female', 'pop_hispanic', 'pop_nat_us_citizen', 'pop_native_american',
         'pop_non_us_citizen', 'poverty_family_below_poverty_level', 'poverty_family_below_poverty_level_rate',
         'poverty_family_count'],  # income  # age # occu
'demographics': ['average_household_income', 'average_household_size', 'fam_med_income', 'hh_gini_index',
           'median_household_income'] + ['age_10_14_rate', 'age_15_17_rate', 'age_18_24_rate', 'age_25_34_rate', 'age_25_44_rate', 'age_25_64_rate',
        'age_35_44_rate', 'age_45_54_rate', 'age_55_64_rate', 'age_55_or_more_rate', 'age_5_9_rate', 'age_65_74_rate',
        'age_75_84_rate', 'age_median', 'age_more_than_85_rate', 'fam_children_6to17', 'fam_children_under_6'] + ['Pro_Occup', 'occupancy_owner_rate', 'occupation_administrative_rate', 'occupation_agriculture_rate',
               'occupation_arts_rate', 'occupation_construction_rate', 'occupation_education_rate',
               'occupation_finance_rate', 'occupation_information_rate', 'occupation_manufacturing_rate',
               'occupation_public_rate', 'occupation_retail_rate', 'occupation_transportation_rate',
               'occupation_wholesale_rate'],
"policy": ['avg_electricity_retail_rate', 'dlrs_kwh', 'incentive_count_nonresidential', 'incentive_count_residential',
           'incentive_nonresidential_state_level', 'incentive_residential_state_level', 'net_metering',
           'net_metering_bin', 'property_tax'],
"gender": ['female_pct', 'male_pct', 'pop_male'],
"solar": ['Adoption', 'AvgSres', 'PV_HuOwn', 'SNRaPa', 'SNRaPcap', 'SNRpcap', 'SRaPa', 'SRaPcap', 'SRpcap',
          'ST_pcap', 'number_of_solar_system_per_household', 'property_tax_bin', 'solar_panel_area_divided_by_area',
          'solar_panel_area_per_capita', 'solar_system_count', 'solar_system_count_nonresidential',
          'solar_system_count_residential', 'total_panel_area', 'total_panel_area_nonresidential',
          'total_panel_area_residential'],
"ownership_pop": ['own_popden'],
"drop": ['active_subsidies', 'aqi_90th_percentile', 'aqi_max', 'aqi_median', 'avg_cbi_usd_p_w', 'avg_ibi_pct',
         'avg_pbi_usd_p_kwh', 'centroid_x', 'hdd_ci', 'hdd_std'],
"population": ['cust_cnt', 'household_count', 'housing_unit_count', 'p16_employed', 'p16_unemployed',
               'pop25_high_school', 'pop25_no_high_school', 'pop25_some_college_plus', 'pop_over_65', 'pop_total',
               'pop_under_18', 'pop_us_citizen', 'population', 'population_density', 'total_units'],
"suitability": ['Yr_own_mwh', 'high_mf_own_devp_cnt', 'high_mf_own_devp_m2', 'high_mf_own_elep_hh', 'high_mf_own_mw',
                'high_mf_own_mwh', 'high_mf_rent_devp_cnt', 'high_mf_rent_devp_m2', 'high_mf_rent_elep_hh',
                'high_mf_rent_mw', 'high_mf_rent_mwh', 'high_own_mwh', 'high_sf_own_devp_cnt', 'high_sf_own_devp_m2',
                'high_sf_own_elep_hh', 'high_sf_own_mw', 'high_sf_own_mwh', 'high_sf_rent_bldg_cnt',
                'high_sf_rent_devp_cnt', 'high_sf_rent_devp_m2', 'high_sf_rent_elep_hh', 'high_sf_rent_mw',
                'high_sf_rent_mwh', 'low_mf_own_devp_cnt', 'low_mf_own_devp_m2', 'low_mf_own_elep_hh',
                'low_mf_own_mw', 'low_mf_own_mwh', 'low_mf_rent_devp_cnt', 'low_mf_rent_devp_m2', 'low_mf_rent_elep_hh',
                'low_mf_rent_mw', 'low_mf_rent_mwh', 'low_own_mwh', 'low_sf_own_devp_cnt', 'low_sf_own_devp_m2',
                'low_sf_own_elep_hh', 'low_sf_own_mw', 'low_sf_own_mwh', 'low_sf_rent_devp_cnt', 'low_sf_rent_devp_m2',
                'low_sf_rent_elep_hh', 'low_sf_rent_mw', 'low_sf_rent_mwh', 'mid_mf_own_devp_cnt',
                'mid_mf_own_devp_m2', 'mid_mf_own_mw', 'mid_mf_own_mwh', 'mid_mf_rent_devp_cnt', 'mid_mf_rent_devp_m2',
                'mid_mf_rent_mw', 'mid_mf_rent_mwh', 'mid_own_mwh', 'mid_sf_own_devp_cnt', 'mid_sf_own_devp_m2',
                'mid_sf_own_mw', 'mid_sf_own_mwh', 'mid_sf_rent_devp_cnt', 'mid_sf_rent_devp_m2', 'mid_sf_rent_mw',
                'mid_sf_rent_mwh', 'mod_mf_own_bldg_cnt', 'mod_mf_own_devp_cnt', 'mod_mf_own_devp_m2',
                'mod_mf_own_elep_hh', 'mod_mf_own_mw', 'mod_mf_own_mwh', 'mod_mf_rent_devp_cnt', 'mod_mf_rent_devp_m2',
                'mod_mf_rent_elep_hh', 'mod_mf_rent_mw', 'mod_mf_rent_mwh', 'mod_own_mwh', 'mod_sf_own_devp_cnt',
                'mod_sf_own_devp_m2', 'mod_sf_own_elep_hh', 'mod_sf_own_mw', 'mod_sf_own_mwh', 'mod_sf_rent_devp_cnt',
                'mod_sf_rent_devp_m2', 'mod_sf_rent_elep_hh', 'mod_sf_rent_mw', 'mod_sf_rent_mwh',
                'very_low_mf_own_devp_cnt', 'very_low_mf_own_devp_m2', 'very_low_mf_own_elep_hh', 'very_low_mf_own_mw',
                'very_low_mf_own_mwh', 'very_low_mf_rent_devp_cnt', 'very_low_mf_rent_devp_m2', 'very_low_mf_rent_hh',
                'very_low_mf_rent_mw', 'very_low_mf_rent_mwh', 'very_low_sf_own_devp_cnt', 'very_low_sf_own_devp_m2',
                'very_low_sf_own_elep_hh', 'very_low_sf_own_mw', 'very_low_sf_own_mwh', 'very_low_sf_rent_devp_cnt',
                'very_low_sf_rent_devp_m2', 'very_low_sf_rent_elep_hh', 'very_low_sf_rent_mw', 'very_low_sf_rent_mwh',
                'verylow_own_mwh'],
"population_age": ['pop_med_age'],
"housing": ['%hh_size_1', '%hh_size_2', '%hh_size_3', '%hh_size_4', 'avg_months_tenancy', 'fmr_2br',
            'heating_fuel_coal_coke', 'heating_fuel_coal_coke_rate', 'heating_fuel_electricity',
            'heating_fuel_electricity_rate', 'heating_fuel_fuel_oil_kerosene', 'heating_fuel_fuel_oil_kerosene_rate',
            'heating_fuel_gas', 'heating_fuel_gas_rate', 'heating_fuel_housing_unit_count', 'heating_fuel_none',
            'heating_fuel_none_rate', 'heating_fuel_other', 'heating_fuel_other_rate', 'heating_fuel_solar',
            'heating_fuel_solar_rate', 'hh_size_1', 'hh_size_2', 'hh_size_3', 'hh_size_4', 'hh_total',
            'housing_unit_median_gross_rent', 'housing_unit_median_value', 'housing_unit_occupied_count',
            'hu_1959toearlier', 'hu_1959toearlier_pct', 'hu_1960to1979_pct', 'hu_1980to1999_pct', 'hu_2000toafter',
            'hu_2000toafter_pct', 'hu_med_val', 'hu_monthly_owner_costs_greaterthan_1000dlrs',
            'hu_monthly_owner_costs_lessthan_1000dlrs', 'hu_mortgage', 'hu_no_mortgage', 'hu_own', 'hu_own_pct',
            'hu_rent', 'hu_vintage_1939toearlier', 'hu_vintage_1940to1959', 'hu_vintage_1960to1970',
            'hu_vintage_1980to1999', 'hu_vintage_2000to2009', 'hu_vintage_2010toafter', 'mortgage_with_rate',
            'occ_rate'],
"politics": ['voting_2012_dem_percentage', 'voting_2012_gop_percentage'],
"climate": ['cdd', 'cdd_ci', 'cdd_std', 'climate_zone', 'cooling_design_temperature', 'daily_solar_radiation',
            'hdd', 'heating_design_temperature'],
"occu": ['Anti_Occup'],
"renewables": ['hydro_prod', 'renew_prod', 'solar_prod'],
"income_habit": ['med_inc_ebill_dlrs'],
"policy_mix": ['incent_cnt_res_own'],
}

simple_block_group_dic = {bg:variables_by_group[bg] for bg in block_group_}
simple_block_group_dic2 = {bg:variables_by_group[bg] for bg in block_group_2}

default_RFR_params = {
        'n_estimators': 200,
        'criterion': 'mae',
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
        'n_jobs': 5,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
        'ccp_alpha': 0.0,
        'max_samples': None
    }

default_RF_params = {
        'n_estimators': 100,
        'criterion': 'entropy',  # criterion{“gini”, “entropy”}, default=”gini”
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.0,
        'max_features': None,
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
        'bootstrap': True,
        'oob_score': False,
        'n_jobs': 5,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
        'class_weight': None,
        'ccp_alpha': 0.0,
        'max_samples': None
    }

hot_spot_cmp = {
    'Metric': list(),
    'above two std mean': list(),
    '90th': list(),
    '95th': list(),
}

fips_d = {'fips': None}

fips_dict = {}

regions_list = ['West', 'NorthEast', 'Mid West', 'South'] + ['Hot_Spots_hh','Hot_Spots_hown', 'Hot_Spots_AvgAr',] +['US',]
regions_list += ['High_Solar_Areas', 'Low_Solar_Areas',] + ['URBAN', 'locale_recode(rural)','locale_recode(suburban)',]

vars_Table = [
    # Target variables
    'Adoption',
    "Installations_per_Household",
    'PV_HuOwn',
    'Avg_RPV_(m^2)',

    # identificaton variables
    'State',
    'fips',
    # 'GEOID'

    # Personal Characteristics
    'Racial_Diversity',
    'Income_Diversity',
    'College_Edu',
    'Female_%',
    'Male_%',
    'Age_Median1',
    'Age_Median',
    'Gender_Ratio',
    'Population_Density',

    # Suitability
    'Daily_Solar_Radiation',
    'Cooling_Degree_Days',
    'Cooling_Degree_Days_std',
    'Heating_Degree_Days_std',
    'Heating_Degree_Days',
    'Climate_Zone',
    'Rural',
    'Town',
    'Suburban',
    'Urban_',
    'Newer_Home',
    'Total_owned_Sbldg',
    'Total_Owned_OK_Roof_Cnt',
    'total_Owned_OK_Roof_m2',
    'Popden_x_TotOK_cnt',
    'Popden_x_TotOK_RCnt',
    'Popden_x_TotOK_Rm2',
    'Ownership_x_TotOK_cnt',
    'Ownership_x_TotOK_Rcnt',
    'Ownership_x_TotOK_Rm2',
    # Economic Resource Indicators
    'Average_Household_Size',
    'Home_owner_rt',
    'Occupied_owner_rate',
    'Household_Med_value',
    'Median_Household_income',
    'Monthly_costs_l_1000',
    'Monthly_costs_g_1000',
    ####################################


    # Policy and Financial Incentives
    'Has_Net_metering',
    'own_popden',
    'Has_Property_tax',
    'Estimated_Yearly_savings_$',
    'Energy_Cost($/kWh)',
    'lowincome_tax_credit_bin',
    'Policy_Combo',
    # Behavior Indicators
    'Avg_Monthly_Consumption_kwh',
    'Commutes__40min_%',
    'Average_Commute',
    'Inc_x_Consmpt_kwh',
]

vars_original = [
    # Target variables
    'Adoption',
    "number_of_solar_system_per_household",
    'PV_HuOwn',
    'AvgSres',

    # Identification variables
    'state',
    'fips',
    # 'GEOID',

    # Personal Characteristics
    'diversity',
    'hh_gini_index',
    'education_college_rate',
    'female_pct',
    'male_pct',
    'age_median',
    'pop_med_age',
    'Gender_Ratio',
    'population_density',

    # Suitability
    'daily_solar_radiation',
    'cdd',
    'cdd_std',
    'hdd_std',
    'hdd',
    'climate_zone',
    'locale_recode(rural)',
    'locale_recode(town)',
    'locale_recode(suburban)',
    'locale_recode(urban)',
    'hu_2000toafter',

    'total_own_Sbldg',
    'total_own_devpC',
    'total_own_devp',
    'popden_x_TotOK_cnt',
    'popden_x_TotOK_RCnt',
    'popden_x_TotOK_Rm2',
    'ownership_x_TotOK_cnt',
    'ownership_x_TotOK_Rcnt',
    'ownership_x_TotOK_Rm2',
    ############################

    # Economic Resource Indicators
    'average_household_size',
    'hu_own_pct',
    'occupancy_owner_rate',
    'hu_med_val',
    'median_household_income',
    'hu_monthly_owner_costs_lessthan_1000dlrs',
    'hu_monthly_owner_costs_greaterthan_1000dlrs',

    # Policy and Financial Incentives
    'net_metering_bin',
    'own_popden',
    'property_tax_bin',
    'Yrl_savings_$',
    'dlrs_kwh',
    'lowincome_tax_credit_bin',
    'Policy_Combo',
    # Behavior Indicators
    'avg_monthly_consumption_kwh',
    'travel_time_40_89_rate',
    'travel_time_average',
    'Inc_x_Consmpt_kwh',
]

model_vars1 = [
    #               TODO: Target variables
    'Adoption',
    "number_of_solar_system_per_household",
    'PV_HuOwn',
    'AvgSres',
    # 'state',

    #               TODO: Personal Characteristics
    'diversity',
    'hh_gini_index',
    'education_college_rate',
    'female_pct',
    'male_pct',
    'age_median',
    # 'pop_med_age',
    # 'Gender_Ratio',
    'population_density',

    #               TODO: Suitability
    'daily_solar_radiation',
    'cdd',
    # 'cdd_std',
    # 'hdd_std',
    # 'hdd',
    # 'climate_zone',
    'locale_recode(rural)',
    # 'locale_recode(town)',
    'locale_recode(suburban)',
    # 'locale_recode(urban)',
    'hu_2000toafter',
    # 'total_own_Sbldg',

    #               TODO: Economic Resource Indicators
    'average_household_size',
    'hu_own_pct',
    'occupancy_owner_rate',
    'hu_med_val',
    'median_household_income',
    'hu_monthly_owner_costs_lessthan_1000dlrs',
    'hu_monthly_owner_costs_greaterthan_1000dlrs',
    'total_own_devpC',
    'total_own_devp',
    'popden_x_TotOK_cnt',
    'popden_x_TotOK_RCnt',
    'popden_x_TotOK_Rm2',
    'ownership_x_TotOK_cnt',
    'ownership_x_TotOK_Rcnt',
    'ownership_x_TotOK_Rm2',
    #               TODO: Policy and Financial Incentives
    'net_metering_bin',
    # 'own_popden',
    'property_tax_bin',
    'Yrl_savings_$',
    'dlrs_kwh',
    'lowincome_tax_credit_bin',
    'Policy_Combo',
    #               TODO: Behavior Indicators
    'avg_monthly_consumption_kwh',
    'travel_time_30_39_rate',
    'travel_time_average',
    'Income_x_EnergyCost',

]

model_only = [
    #               TODO: Target variables
    # 'Adoption',
    "number_of_solar_system_per_household",
    'PV_HuOwn',
    'AvgSres',
    # 'state',

    #               TODO: Personal Characteristics
    'diversity',
    # 'hh_gini_index',
    'education_college_rate',
    'female_pct',
    'age_median',
    # 'pop_med_age',
    # 'Gender_Ratio',
    'population_density',

    #               TODO: Suitability
    'daily_solar_radiation',
    'cdd',
    # 'cdd_std',
    # 'hdd_std',
    # 'hdd',
    # 'climate_zone',
    'locale_recode(rural)',
    # 'locale_recode(town)',
    'locale_recode(suburban)',
    # 'locale_recode(urban)',
    'hu_2000toafter',
    # 'total_own_Sbldg',

    #               TODO: Economic Resource Indicators
    'average_household_size',
    'hu_own_pct',
    # 'occupancy_owner_rate',
    'hu_med_val',
    'median_household_income',
    'hu_monthly_owner_costs_lessthan_1000dlrs',
    'hu_monthly_owner_costs_greaterthan_1000dlrs',

    #               TODO: Policy and Financial Incentives
    'net_metering_bin',
    # 'own_popden',
    'property_tax_bin',
    'Yrl_savings_$',
    'dlrs_kwh',

    #               TODO: Behavior Indicators
    'avg_monthly_consumption_kwh',
    'travel_time_40_89_rate',
    'travel_time_average',
    'Inc_x_Consmpt_kwh',
]

to_mm = [
    'age_median',
    'population_density',
    'daily_solar_radiation',
    'cdd',
    'cdd_std',
    'average_household_size',
    'hu_med_val',
    'median_household_income',
    'Yrl_savings_$',
    'avg_monthly_consumption_kwh',
    'hu_monthly_owner_costs_lessthan_1000dlrs',
    'hu_monthly_owner_costs_greaterthan_1000dlrs',
    # 'travel_time_30_39_rate',
    'travel_time_average',
    'hu_2000toafter',
    'Inc_x_Consmpt_kwh',
    'popden_x_TotOK_cnt',
    'popden_x_TotOK_RCnt',
    'popden_x_TotOK_Rm2',
    'ownership_x_TotOK_cnt',
    'ownership_x_TotOK_Rcnt',
    'ownership_x_TotOK_Rm2',
    'total_own_devpC',
    'total_own_devp',
    "Income_x_EnergyCost",
]

to_mm2 = [
"Income_x_EnergyCost",
"Savings_potential",
"Tot_own_mw",
"Yr_own_mwh",
"Yrl_%_inc",
"Yrl_savings_$",
"average_household_size",
"avg_monthly_consumption_kwh",
"cdd",
"cdd_std",
"climate_zone",
"daily_solar_radiation",
"daily_solar_radiation_RF",
"diversity",
"dlrs_kwh",
"hh_gini_index",
"high_own_elep_hh",
"household_count",
"hu_med_val",
"low_own_elep_hh",
"median_household_income",
"mod_own_elep_hh",
"ownership_x_TotOK_Rcnt",
"ownership_x_TotOK_Rm2",
"ownership_x_TotOK_cnt",
"pop25_some_college_plus",
"pop_med_age",
"popden_x_TotOK_RCnt",
"popden_x_TotOK_Rm2",
"popden_x_TotOK_cnt",
"population_density",
"rural_diversity",
"solar_prod",
"suburban_diversity",
"total_own_Sbldg",
"total_own_devp",
"total_own_devpC",
"total_own_elep",
"total_own_hh",
"urban_diversity",
"verylow_own_elep_hh",
'Inc_x_Consmpt_kwh',
"Income_x_Suitable_m2",
'income_x_consumption_energyCost',
]

to_impute = [
    '',
    'diversity',
    'hh_gini_index',
    'education_college_rate',
    'female_pct',
    'age_median',
    'population_density',
    'daily_solar_radiation',
    'cdd',
    # 'cdd_std',
    'average_household_size',
    'hu_own_pct',
    'hu_med_val',
    'median_household_income',
    'Yrl_savings_$',
    'dlrs_kwh',
    'avg_monthly_consumption_kwh',
    'locale_recode(rural)',
    # 'locale_recode(town)',
    'locale_recode(suburban)',
    # 'locale_recode(urban)',
    # 'hu_monthly_owner_costs_lessthan_1000dlrs',
    # 'hu_monthly_owner_costs_greaterthan_1000dlrs',
    'travel_time_30_39_rate',
    # 'travel_time_average',
    'hu_2000toafter',
    'net_metering_bin',
    # 'own_popden',
    'property_tax_bin',
]

# Var Minimal for GIS stuff
vars_Minimal = [
    # Target variables
    # 'Adoption',
    "RPV/HH",
    "RPV/HOw",
    'AvgSize',
    'State',
    'fips',
    'GEOID',
    # 'st_fips',
    # 'st_abbr',

    # Personal Characteristics
    'Diverse%',
    # 'Inc_Div',
    'College%',
    'Female%',
    'Med_Age',
    # 'PmedAge',
    # 'Gndr_Rt',
    # 'Pop_den',

    # Suitability
    'SolarRad',
    'cdd',
    # 'cdd_std',
    # 'hdd_std',
    # 'hdd',
    # 'climateZ',
    'Rural',
    # 'Town',
    'Suburban',
    # 'Urban',
    'HuVge2k',
    # 'Owed_Sb',

    # Economic Resource Indicators
    'AVG_hh_S',
    'HuOwn_rt',
    # 'OccOwnRt',
    'HuMedVal',
    'MedHhInc',
    # 'HuCstG1k',
    # 'HuCstL1k',

    # Policy and Financial Incentives
    'NetMeter',
    # 'own_popden',
    'PrpTax',
    'Pol_Com',
    'EsYrlSav',
    'Savings',
    'dlrs_kwh',
    'Slr_Prd',
    # Behavior Indicators
    'AvgMthCn',
    'CmmtH_Rt',
    # 'AvgCmmt',
    'IncxCons',
    'IncXeCst',
    'IncXm2',
    'Roof_m2',
    'RoofCnt',
    'IcEcnEp',
    'HghCmtT',
    'U_div',
    'R_div',
    'S_div',
]

model_vars_HM = [
    #               TODO: Target variables
    # 'Adoption',
    "number_of_solar_system_per_household",
    'PV_HuOwn',
    'AvgSres',
    'state',
    'fips',
    'GEOID',
    # 'state_fips',
    # 'state_abbr',

    #               TODO: Personal Characteristics
    'diversity',
    # 'hh_gini_index',
    'education_college_rate',
    'female_pct',
    'age_median',
    # 'pop_med_age',
    # 'Gender_Ratio',
    # 'population_density',

    #               TODO: Suitability
    'daily_solar_radiation',
    'cdd',
    # 'cdd_std',
    # 'hdd_std',
    # 'hdd',
    # 'climate_zone',
    'locale_recode(rural)',
    # 'locale_recode(town)',
    'locale_recode(suburban)',
    # 'locale_recode(urban)',
    'hu_2000toafter',
    # 'total_own_Sbldg',

    #               TODO: Economic Resource Indicators
    'average_household_size',
    'hu_own_pct',
    # 'occupancy_owner_rate',
    'hu_med_val',
    'median_household_income',
    # 'hu_monthly_owner_costs_lessthan_1000dlrs',
    # 'hu_monthly_owner_costs_greaterthan_1000dlrs',

    #               TODO: Policy and Financial Incentives
    'net_metering_bin',
    # 'own_popden',
    'property_tax_bin',
    'Policy_Combo',
    'Yrl_savings_$',
    'Savings_potential',
    'dlrs_kwh',
    'solar_prod',
    #               TODO: Behavior Indicators
    'avg_monthly_consumption_kwh',
    'travel_time_30_39_rate',
    # 'travel_time_average',
    'Inc_x_Consmpt_kwh',
    'Income_x_EnergyCost',
    'Income_x_Suitable_m2',
    'total_own_devp',
    'total_own_devpC',
    'income_x_consumption_energyCost',
    'travel_time_40_89_rate',
    'urban_diversity',
    'rural_diversity',
    'suburban_diversity',
]

hm_select = [
    #               TODO: Target variables
    'Adoption',
    "number_of_solar_system_per_household",
    'PV_HuOwn',
    'AvgSres',
    'state',
    'fips',
    'GEOID',
    'state_fips',
    'state_abbr',
    #               TODO: Personal Characteristics
    # 'diversity',
    # 'hh_gini_index',
    # 'education_college_rate',
    # 'female_pct',
    # 'age_median',
    # 'pop_med_age',
    # 'Gender_Ratio',
    # 'population_density',

    #               TODO: Suitability
    # 'daily_solar_radiation',
    # 'cdd',
    # 'cdd_std',
    # 'hdd_std',
    # 'hdd',
    # 'climate_zone',
    'locale_recode(rural)',
    # 'locale_recode(town)',
    'locale_recode(suburban)',
    # 'locale_recode(urban)',
    'hu_2000toafter',
    # 'total_own_Sbldg',

    #               TODO: Economic Resource Indicators
    # 'average_household_size',
    # 'hu_own_pct',
    # 'occupancy_owner_rate',
    # 'hu_med_val',
    'median_household_income',
    # 'hu_monthly_owner_costs_lessthan_1000dlrs',
    # 'hu_monthly_owner_costs_greaterthan_1000dlrs',

    #               TODO: Policy and Financial Incentives
    'net_metering_bin',
    # 'own_popden',
    'property_tax_bin',
    'Policy_Combo',
    # 'Yrl_savings_$',
    # 'dlrs_kwh',

    #               TODO: Behavior Indicators
    'avg_monthly_consumption_kwh',
    'travel_time_30_39_rate',
    # 'travel_time_average',

]

# create a conversion dictionary to make table friendly variable names
conversions_d = {a: b for a, b in zip(vars_original, vars_Table)}
unconversions_d = {b: a for a, b in zip(vars_original, vars_Table)}

#for v in conversions_d:
#    print("'{}': '{}'".format(v, conversions_d[v]))

model_onlyNRM = [conversions_d[v] for v in model_only]
major4_reg = list(US_4Major.keys())

conversions_dHM = {a: b for a, b in zip(model_vars_HM, vars_Minimal)}
# start keeping a list of the things we have added so we can get them later
reg_vars = major4_reg + ['High_Solar_Areas', 'Low_Solar_Areas', 'Hot_Spots_hh', 'Hot_Spots_hown',
                         'Hot_Spots_AvgAr', ] + ['US']
model_varsD = model_vars1 + ['High_Solar_Areas', 'Low_Solar_Areas', 'DS_HighSolar', 'Hot_Spots_hh', 'Hot_Spots_hown',
                             'Hot_Spots_AvgAr', ]
model_varsD += ['URBAN', 'locale_recode(rural)', 'locale_recode(suburban)', ] + ['Inc_x_Consmpt_kwh'] + major4_reg

hm_select += ['High_Solar_Areas', 'Low_Solar_Areas', 'Hot_Spots_hh', 'Hot_Spots_hown', 'Hot_Spots_AvgAr', ]
hm_select += ['URBAN', 'locale_recode(rural)', 'locale_recode(suburban)', ] + ['Inc_x_Consmpt_kwh']
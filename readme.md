# Directory Structure:
* _Data
    * set of empty directories that can be used to store the deep solar and NREL to utilize the merging tool. The notebook titled MergedDataGenerator.ipynb in the _DATAGENTOOLS folder will use this to generate the merged data file with additional variables added. More information on the additional variables can be found in the MixedSetGenerationTools.py where the class uesed to generate the data can be found.
    * Directory Contents:
        * _Mixed: can be used to store the mixed merged data set.
        * Renewable_State_Info: contains file used to add renewable generation information from 2009 from eia into merged set
        * DeepSolar: used by the merged data_generator notebook in _DATAGEN_TOOLS folder to merge the DeepSolar and NREL SEEDSII REPLICA data sets and add/calculate additional variables.
                     User should download, and the Deepsolar data set and add to this folder. Instructions can be found in the DATAGEN_TOOLS folder
          
        * NREL: used by the merged data_generator notebook in _DATAGEN_TOOLS folder to merge the DeepSolar and NREL SEEDSII REPLICA data sets and add/calculate additional variables.
                User should download, and the Deepsolar data set and add to this folder. Instructions can be found in the DATAGEN_TOOLS folder 
        * 

* _DATA_GENERATORS
    * __Make_Mixed_DeepSolar_set.py
    * _CONVERGENT_DATA_GENERATOR.py
    * MergedData_Generator.ipynb
* _DATAGEN_TOOLS
    * _CONVERGENT_DATA_GENERATOR.py
    * _DataConvergenceTools.py
    * Base set Modder-Copy2.ipynb
    * MergedData_Generator.ipynb
    * MixedSetGenerationTools.py
* _products
    * _Data_Manipulation.py
    * __DEEPSOLAR_Resources.py 
    * _ConvergentResources_.py
    * _DEEPSOLAR_.py
    * _interface_tools.py
    * ML_Tools.py
    * performance_metrics.py
    * utility_fnc.py

# Project Description
> Using a merger of the DeepSolar and SEEDSII REPLICA data sets an analysis was made of the entire US, and several subsets 
> of the US based on solar adoption rates, solar radiation in regard to finding the strongest most influental factors 
> relating to per household solar adoption at the census tract level. A mixture of data and assumption driven techniques 
> were employed to find an initial set of powerful predictive  socio-economic, climate, policy, physical suitability, 
> and behavioral factors that drive solar adoption in an statistically significant way. Random Forest Regression was 
> employed to find and rank a base set of variables to build a larger model around. This will contain demonstrations of 
> the tools and code used to perform the analysis and build the model as well as the future home of the paper thw work is used to produce. 
> This space is currently under construction. 


# Generating Merged DeepSolar + SEEDS II REPLICA set
1. Inside the _Data folder are two folders where you need to add the deepsolar and NREL seeds 2 data
   * _Data/DeepSolar
     - add the deepsolar_tract.csv file here. Open the file and save it as an .xlsx file. 
   * _Data/NREL
     - add the seeds_ii_replica.csv file here
2. Inside the _DATAGEN_TOOLS folder there is a notebook titled MergedData_Generator.ipynb. It will 
   generate a merged data set as well as add the additional variables added adn calculated.  It 
   will make the data set if you either add the data sets as described above and run it as is, or 
   you can alter the paths to the DeepSolar and SEEDSII data sets when creating the 
   CONVERGENT_DATA_GEN object in the second code cell. There are instructions in the readme.md in 
   the _DATAGEN_TOOLS folder and the file and the notebook 



# Example programs
  * #### Data Mining With Clustering-- Knn, K-means, DBSCAN.ipynb
    * Jupyter Notebook showing examples of Cluster analysis performed on Mixed DeepSolar and SEEDS II data
    
  * #### RF Regression.ipynb
    * Example program Using Random Forest Regression and feature importance for 
      predicatively powerful variable selection
      
    * ##### DeepSolar
      * DeepSolar Data Set created by Stanford University, link to site for download below
        * [DeepSolar Main Site](https://web.stanford.edu/group/deepsolar/home)
        * Data and metadata csv file down load links are in the related link section
        * Data received from the **"DeepSolar database (census tract level)"** is required to be 
          placed in the ***DeepSolar*** folder in the ***_Data directory***
    
    * ##### NREL SEEDS II data set
        * DeepSolar Data Set created by Stanford University, link to site for down load below
            * [Seeds II REPLICA data download](https://data.nrel.gov/submissions/81)
              * the file seeds_ii_replica.csv is required for data generation to be placed in 
                the ***NREL*** folder in the ***_Data*** directory
    
    
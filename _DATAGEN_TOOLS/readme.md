* _DATAGEN_TOOLS
    * _CONVERGENT_DATA_GENERATOR.py
    * _DataConvergenceTools.py
    * Base set Modder-Copy2.ipynb
    * MergedData_Generator.ipynb
    * MixedSetGenerationTools.py


# Generating Merged DeepSolar + SEEDS II REPLICA set
1. Inside the _Data folder are two folders where you need to add the deepsolar and NREL seeds 2 data
   * _Data/DeepSolar
     - add the deepsolar_tract.csv file here. Open the file and save it as an .xlsx file. 
   * _Data/NREL
     - add the seeds_ii_replica.csv file here
2. Inside this folder there is a notebook titled MergedData_Generator.ipynb. It will 
   generate a merged data set as well as add the additional variables added adn calculated.  It 
   will make the data set if you either add the data sets as described above and run it as is, or 
   you can alter the paths to the DeepSolar and SEEDSII data sets when creating the 
   CONVERGENT_DATA_GEN object in the second code cell. There are instructions in the readme.md in 
   the _DATAGEN_TOOLS folder and the file and the notebook 



# Data Down load sites
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


# Cocry-Pred: A Dynamic Resource Propagation Method Integrating Cocrystal and Substructure Networks for Cocrystal Prediction

## The main framework of this work.
![image](https://github.com/17855461143/fluor_pred/blob/main/figures/2.png?raw=true)

## File Description:
* Data [folder]  
  The folder contains model training data and various files for fingerprint generation networks. Among them, 1_Cocrystal_data.csv is the cocrystal data used by the model, and 2_Cocrystal_molecule.csv contains all the molecular nodes.
    
* External Validation Set [folder]  
  The folder contains two external validation set data files, along with their sources and related evaluation files. In external_validation_set1.csv, both molecules are part of the network; in external_validation_set2.csv, one molecule is from within the network, and the other is an NCE (New Chemical Entity).
* Results [folder]  
  The user interaction code for Fluor-predictor. Once the environment is set up, simply run GUI.py to execute it, and the output data will be saved in the Results folder.  
    
* Software [folder]  
  Single-task models and machine learning code for comparison with the MTATFP model.  
    
* evaluation.py  
  The specific usage method of Fluor-predictor is based on the runtime environment.
  
* Predict.ipynb  
  The usage demonstration of Fluor-predictor: users simply need to replace the target molecules and solvents.

* Software-GUI.ipynb  
  The usage demonstration of Fluor-predictor: users simply need to replace the target molecules and solvents.

* User Guide.pdf  
  The usage demonstration of Fluor-predictor: users simply need to replace the target molecules and solvents.

## Creation of the GUI Usage Environment:  
Please follow these steps to create the working environment.  
* numpy == 1.21.5
* pandas == 1.2.4
* networkx == 2.6.3
* rdkit == 2020.09.1.0

  
## The software interface is shown as follows:
for specific usage, please refer to the User Guide.pdf.
![image](https://github.com/17855461143/fluor_pred/blob/main/figures/1.png?raw=true)

## Data Usage Distribution:
![image](https://github.com/17855461143/fluor_pred/blob/main/figures/3.png?raw=true)

## Atom Weight Visualization:
In the MTATFP folder, we have retained visualization code for all scripts, allowing users to display visualized weights and also to regenerate training files for new data visualization.
![image](https://github.com/17855461143/fluor_pred/blob/main/figures/4.png?raw=true)

## Usage Statement:
Fluor-predictor is a freely available dye database and dye prediction tool. You have the right to install and run the software on your personal computer, as well as to copy and modify the software to meet your personal learning and research needs. However, you are not allowed to use the software for any commercial activities, including but not limited to selling, renting, lending the software or any derivative products of the software, or using the software in any commercial services or products.

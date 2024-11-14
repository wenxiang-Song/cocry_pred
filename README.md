# Cocry-Pred: A Dynamic Resource Propagation Method Integrating Cocrystal and Substructure Networks for Cocrystal Prediction

## File Description:
* Data [folder]  
  The folder contains model training data and various files for fingerprint generation networks. Among them, 1_Cocrystal_data.csv is the cocrystal data used by the model, and 2_Cocrystal_molecule.csv contains all the molecular nodes.
    
* External Validation Set [folder]  
  The folder contains two external validation set data files, along with their sources and related evaluation files. In external_validation_set1.csv, both molecules are part of the network; in external_validation_set2.csv, one molecule is from within the network, and the other is an NCE (New Chemical Entity).
  
* Results [folder]  
  The results of the model predictions are all saved in the Results folder.
    
* Software [folder]  
  The software is a package that can be directly downloaded and saved for local use. Double-clicking on GUI.exe will open the software Cocry-pred.
    
* evaluation.py
  evaluation.py is the tuning code used for selecting the best fingerprints and adjusting hyperparameters.
  
* Predict.ipynb  
  Predict.ipynb is the prediction code, and all results will be saved in the Results folder.
  
* Software-GUI.ipynb  
  Software-GUI.ipynb is the GUI interface we have developed, which users can directly invoke or make certain modifications to.
  
* User Guide.pdf  
  User Guide.pdf is the specific software usage manual.
  
## Creation of the GUI Usage Environment:  
Please follow these steps to create the working environment.  
* numpy == 1.21.5  
* pandas == 1.2.4  
* networkx == 2.6.3  
* rdkit == 2020.09.1.0  
  
## NBI Algorithm Principle Diagram：
![image](https://github.com/17855461143/fluor_pred/blob/main/figures/2.png?raw=true)
  
## Cocry-pred Software Usage Interface:
for specific usage, please refer to the User Guide.pdf.
![image](https://github.com/17855461143/fluor_pred/blob/main/figures/1.png?raw=true)

## Cocrystal Network Diagram:
![image](https://github.com/17855461143/fluor_pred/blob/main/figures/3.png?raw=true)

## Usage Statement:
Cocry-pred is a freely available cocrystal prediction tool. You have the right to install and run the software on your personal computer, as well as to copy and modify the software to meet your personal learning and research needs. However, you are not allowed to use the software for any commercial activities, including but not limited to selling, renting, lending the software or any derivative products of the software, or using the software in any commercial services or products.

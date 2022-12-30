# PLMGraph-Inter
inter-protein contact prediction
![image](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/main/main_fig.jpg)
## Requirements
- #### python3.8
  1. [pytorch1.9](https://pytorch.org/)  
  2. [Biopython](https://biopython.org/)
  3. [esm](https://github.com/facebookresearch/esm)
  4. [numpy](https://numpy.org/)
- #### other packages
  1. [alnstats](https://github.com/psipred/metapsicov/tree/master/src)
  2. [fasta2aln](https://github.com/kad-ecoli/hhsuite2/blob/master/bin/fasta2aln)
  3. [hh-suite](https://github.com/soedinglab/hh-suite)
  4. [CCMpred](https://github.com/soedinglab/CCMpred)

## Installation
### 1. Install PLMGraph-Inter
    git clone https://github.com/ChengfeiYan/PLMGraph-Inter.git
### 2. Modify the path of each tool (CCMpred, alnstats ...) in predict.py
  
### 3. Download the trained models
   Download the trained models from  [trained models](https://drive.google.com/file/d/1Y9eSlIJr-XDG5gREIEeGK4BW_Of0F_UQ/view?usp=sharing), then unzip it into the folder named "model".

## Usage
    python predict.py sequenceA msaA pdbA sequenceB msaB pdbB result_path device
   Where MSA should be derived from Uniref90 or Uniref100 database.

## Example
    python predict.py ./example/1GL1_A.fasta ./example/1GL1_A_uniref100.a3m ./example/1GL1_A.pdb ./example/1GL1_I.fasta ./example/1GL1_I_uniref100.a3m ./example/1GL1_I.pdb ./example/result 'cpu'

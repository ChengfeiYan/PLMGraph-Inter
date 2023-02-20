# PLMGraph-Inter
Inter-protein contact prediction based on protien language model embedded geomteric graphs. 
![image](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/main/main_fig.jpg)
## Requirements
- #### python3.8
  1. [pytorch1.9](https://pytorch.org/)  
  2. [Biopython](https://biopython.org/)
  3. [esm](https://github.com/facebookresearch/esm)
  4. [numpy](https://numpy.org/)
  5. [GVP](https://github.com/drorlab/gvp-pytorch)
  6. [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)  
*[Please note]: To implement protein language models (ESM-1b, ESM-MSA-1b and ESM-IF1 in this study) in [esm](https://github.com/facebookresearch/esm), model weights of these protein language models should be downloaded first from the links provided in the *[Available Models and Datasets] of [esm github](https://github.com/facebookresearch/esm). The paths of these model weights needs to be set in [predict.py](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/main/predict.py#L33) later. 
- #### other tools
  1. [alnstats](https://github.com/psipred/metapsicov/tree/master/src/alnstats) (directly download the executable file, and change its mode to be executable)
  2. [fasta2aln](https://github.com/kad-ecoli/hhsuite2/blob/master/bin/fasta2aln) (directly downloay the executable file, and change its mode to be exetutable)
  3. [hh-suite](https://github.com/soedinglab/hh-suite)
  4. [CCMpred](https://github.com/soedinglab/CCMpred)

## Installation
### 1. Install PLMGraph-Inter
    git clone https://github.com/ChengfeiYan/PLMGraph-Inter.git
### 2. Modify the path of each tool (CCMpred, alnstats ...) and the paths of the model weights of the protein language models (ESM-1b, ESM-MSA-1b, ESM-IF1) in [predict.py](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/main/predict.py#L25)
### 3. Copy the [esm1b_t33_650M_UR50S-contact-regression.pt](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/main/data/regression/esm1b_t33_650M_UR50S-contact-regression.pt) from /data/regression to the location of [ESM-1b's model weights](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/main/predict.py#L33);  Copy the [esm_msa1b_t12_100M_UR50S-contact-regression.pt](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/main/data/regression/esm_msa1b_t12_100M_UR50S-contact-regression.pt) from /data/regression to the location of [ESM-MSA-1b's model weights](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/main/predict.py#L34);
### 4. Download the trained models
   Download the trained models from  [trained models](https://drive.google.com/file/d/1Y9eSlIJr-XDG5gREIEeGK4BW_Of0F_UQ/view?usp=sharing), then unzip it into the folder named "model".

## Usage
    python predict.py sequenceA msaA pdbA sequenceB msaB pdbB result_path device
    1.  sequenceA: fasta file corresponding to target A.
    2.  msaA: a3m file corresponding to target A (multiple sequence alignment).
    3.  pdbA: pdb file corresponding to target A.
    4.  sequenceB: fasta file corresponding to target B.
    5.  msaB: a3m file corresponding to target B (multiple sequence alignment).
    6.  pdbB: pdb file corresponding to target B.
    7.  result_path: [a directory for the output]
    8.  device: cpu, cuda:0, cuda:1, ...
   Where MSA should be derived from Uniref90 or Uniref100 database.

## Example
    python predict.py ./example/1GL1_A.fasta ./example/1GL1_A_uniref100.a3m ./example/1GL1_A.pdb ./example/1GL1_I.fasta ./example/1GL1_I_uniref100.a3m ./example/1GL1_I.pdb ./example/result cpu

## The output of exmaple(1GL1)
![image](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/main/data/plmg.jpg)
It should be noted, we downsampled the MSAs of the example target due to the file size limiation of github. 
The real performance of PLMGraph-Inter for the provided example should be better in real practice. 

## Reference  
Please cite: Protein language model embedded geometric graphs power inter-protein contact prediction.
Yunda Si, Chengfei Yan
bioRxiv 2023.01.07.523121; doi: https://doi.org/10.1101/2023.01.07.523121

If you meet any problem in installing or running the program, please contact chengfeiyan@hust.edu.cn.


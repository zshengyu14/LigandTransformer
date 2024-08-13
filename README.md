# LigandTransformer
Ligand-Transformer is a deep learning framework for predicting the binding affinity between proteins and small molecule ligands, based on the AlphaFold2 transformer architecture. It takes as input the protein sequence and the graph representation of the ligand, and predicts binding affinity as well as inter-residue distances of the protein, inter-atomic distances of the ligand, and atom-residue distances for the protein-ligand complex.

## Installation
The dependency is listed in requirements.txt. The software is tested on 90 Dell PowerEdge XE8545 server consisting of:

2x AMD EPYC 7763 64-Core Processor 1.8GHz (128 cores in total)
1000 GiB RAM
4x NVIDIA A100-SXM-80GB GPUs
Dual-rail Mellanox HDR200 InfiniBand interconnect
and each A100 GPU contains 6912 FP32 CUDA Cores.

The software is run on Rocky Linux 8 and CUDA11.2.

To preprocess the features of protein and ligand, we use the following software:
- [AlphaFold](https://github.com/google-deepmind/alphafold) (Initial release)
- [GraphMVP](https://github.com/chao1224/GraphMVP)




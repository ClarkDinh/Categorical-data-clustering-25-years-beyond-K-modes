# Categorical Data Clustering: 25 Years Beyond K-modes

## Overview
This repository collects Python source codes for clustering categorical data from GitHub. It provides a platform to evaluate and compare various clustering algorithms.

## Algorithms

1. **K-modes (Huang, 1998)**
   - [Link to the paper](https://link.springer.com/article/10.1023/A:1009769707641)
   - [Link to GitHub](https://github.com/nicodv/kmodes)

2. **Fuzzy K-modes (1999)**
   - [Link to the paper](https://ieeexplore.ieee.org/document/784206)
   - [Link to GitHub](https://github.com/srimantacse/ShapleyCategorical/tree/main/src)

3. **ROCK (2000)**
   - [Link to the paper](https://www.sciencedirect.com/science/article/pii/S0306437900000223)
   - [Link to GitHub](https://github.com/annoviko/pyclustering/blob/master/pyclustering/cluster/rock.py)

4. **K-representatives (2004)**
   - [Link to the paper](https://tinyurl.com/3vhthhbr)
   - Link to GitHub: (Not provided)

5. **K-modes Cao (2009)**
   - [Link to the paper](https://www.sciencedirect.com/science/article/pii/S0957417409001043)
   - [Link to GitHub](https://github.com/nicodv/kmodes)

6. **Genetic Fuzzy K-modes (2009)**
   - [Link to the paper](https://www.sciencedirect.com/science/article/pii/S0957417407005957)
   - [Link to GitHub](https://github.com/V-Rang/Fuzzy-clustering)

7. **MGR (2014)**
   - [Link to the paper](https://www.sciencedirect.com/science/article/pii/S0045790617327131)
   - [Link to GitHub](https://github.com/vatsarishabh22/MFk-M-Clustering)

8. **EGA FMC (2018)**
   - [Link to the paper](https://dl.acm.org/doi/10.1504/IJBIC.2018.092801)
   - [Link to GitHub](https://github.com/vatsarishabh22/MFk-M-Clustering)

9. **MFK-means (2018)**
   - [Link to the paper](https://www.sciencedirect.com/science/article/pii/S0045790617327131)
   - [Link to GitHub](https://github.com/medhini/Genetic-Algorithm-Fuzzy-K-Modes)

10. **K-means Like Algorithm (2019)**
    - [Link to the paper](https://link.springer.com/article/10.1007/s12652-019-01445-5)
    - Link to GitHub: (Not provided)

11. **K-PbC (2020)**
    - [Link to the paper](https://link.springer.com/article/10.1007/s10489-020-01677-5)
    - [Link to GitHub](https://github.com/ClarkDinh/k-PbC)

12. **LSH K-representatives (2021)**
    - [Link to the paper](https://www.sciencedirect.com/science/article/pii/S0925231221012340)
    - [Link to GitHub](https://github.com/nmtoan91/lshkrepresentatives)

13. **GT-Kmodes (2021)**
    - [Link to the paper](https://www.sciencedirect.com/science/article/pii/S2666827021000505#sec3)
    - [Link to GitHub](https://github.com/srimantacse/ShapleyCategorical/tree/main/src)

14. **MIS (2022)**
    - [Link to the paper](https://link.springer.com/chapter/10.1007/978-3-031-17114-7_16)
    - [Link to GitHub](https://github.com/c4sgub/MIS_Categorical_Clustering)

## Datasets Evaluated
We confirm the performance of the clustering algorithms using the following commonly used datasets:
- Mushroom
- Soybean
- Zoo
- Congressional Voting Records

## Validation Metrics
We use four external validation metrics for comparison:
- **Accuracy**
- **Purity**
- **Normalized Mutual Information (NMI)**
- **Adjusted Rand Index (ARI)**

## Usage
The results are maintained in the notebook: [compare_clustering_algos_GitHub](https://github.com/ClarkDinh/Categorical-data-clustering-25-years-beyond-K-modes/blob/main/compare_clustering_algos_GitHub.ipynb)

## Installation
Install the required libraries and packages to run the notebook:
```bash
pip install kmodes pyclustering lshkrepresentatives xlsxwriter numpy pandas tqdm matplotlib
```

## Citation

Please cite the following paper:

    Tai Dinh, Wong Hauchi, Philippe Fournier-Viger, Daniil Lisik, Hieu-Chi Dam, Van-Nam Huynh (2024). Categorical Data Clustering: 25 Years Beyond K-modes

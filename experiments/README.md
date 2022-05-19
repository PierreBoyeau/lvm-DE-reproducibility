# lfc_estimation


## Conda installs

On a python 3.7 fresh install with pytorch with version <= 1.7.1:
Install the following packages with conda/mamba


```
mamba install -c conda-forge -c bioconda pandas seaborn scikit-learn tqdm plotnine matplotlib arviz scipy h5py loompy xlrd jupyter nbconvert nbformat ipython hyperopt jupyterlab seaborn scikit-learn statsmodels numba pytables python-igraph leidenalg nodejs nb_conda_kernels plotnine arviz --yes
pip install scanpy matplotlib-venn

mamba install -c conda-forge -c bioconda r-base r-essentials bioconductor-deseq2 bioconductor-edger bioconductor-mast bioconductor-biocparallel  r-blme bioconductor-complexheatmap r-data.table bioconductor-deseq2 r-dplyr bioconductor-edger r-ggplot2 r-glmmtmb bioconductor-limma r-lmertest r-lme4 r-matrix r-matrixstats r-r.methodss3 r-progress r-purrr bioconductor-s4vectors r-scales bioconductor-scater r-sctransform bioconductor-singlecellexperiment bioconductor-summarizedexperiment bioconductor-variancepartition r-viridis r-rcppcnpy r-rcpp tensorboard --yes

mamba install -c conda-forge rpy2 --yes
pip install scikit-misc
pip install "rpy2<=3.4.2"

mamba install -c conda-forge r-stringi libopenblas --yes
```


## Annex packages
```
# Install scVI
cd MYHOMEDIR
cd lvm_scvi
python setup.py install

# R source code


# additional dependencies
cd MYHOMEDIR
cd power_spherical
python setup.py install
```


# Files for reproducibility
First, download the data archive `https://drive.google.com/file/d/1g6tt3V9rXqjWep6Eh1N8J28l0LbXreas/view?usp=sharing` and extract it in  `MYHOMEDIR/experiments`.


In each of these files, replace accordingly `PATH_TO_SCRIPTS`, such that 

```
PATH_TO_SCRIPTS = MYHOMEDIR/scripts
```

# scripts for reproducibility

- `bash run_symsim_all.sh`: SymSim experiments
- `python run_PBMC_active.py`: PBMC experiment
- `bash run_muscat_all.sh`: PBMC: other experiments
- `python run_blish.py`: BLISH experiments
- `pbmcs_compare.ipynb`: PbmcBench experiment
- `symsim_differential_analysis.ipynb`
- `ablation_study.ipynb`
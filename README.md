# IterEmb

The applied Louvain algorithm was downloaded from here: https://github.com/taynaud/python-louvain
node2vec: https://github.com/skojaku/fastnode2vec


```bash
conda create -n iterEmb python=3.9
conda activate iterEmb
conda install -c conda-forge mamba -y
mamba install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge -y
mamba install -y -c bioconda -c conda-forge snakemake -y
mamba install -c conda-forge graph-tool scikit-learn numpy numba scipy pandas networkx seaborn matplotlib gensim ipykernel tqdm black -y
mamba install -c conda-forge numpy==1.23.5
```

Install the in-house packages:
```bash
cd libs/iterEmb && pip install -e .
cd libs/embcom && pip install -e .
```

## Test run

Open Snakefile and change the "DATA_DIR" folder to "test_data":
 ```python
DATA_DIR = "test_data"
```
and run
```bash
snakemake --cores 5 figs
```
which should produce a figure in `figs` folder.


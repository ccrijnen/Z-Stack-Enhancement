## Usage with Conda

To use the virtual environment specifications with conda do the following:
```
conda env create -f environment.yml
```

If you want to use the Conda environment also in Jupyter Notebooks:
```
conda deactivate
conda deactivate
conda activate dl
conda install ipykernel
python -m ipykernel install --user --name ENV_NAME --display-name "Python (ENV_NAME)"
```

# GraphConformal

Installation:

```bash
uv sync --all-groups
```

Then source the virtual environment at `.venv`.

## MIST

Minimal setup (warning: may require a lot of RAM, 70GB+):

```bash
cd mist
sh data_processing/canopus_train/00_download_canopus_data.sh
sh data_processing/canopus_train/01_run_subform.sh

sh data_processing/mol_libraries/pubchem/01_download_smiles.sh
sh data_processing/mol_libraries/pubchem/02_make_formula_subsets.sh

python data_processing/canopus_train/03_retrieval_hdf.py

sh quickstart/model_predictions/00_download_models.sh
```

The files we need in the end are the following:

- `mist/quickstart/pretrained_models/mist_contrastive_canopus_pretrain.ckpt`

- `mist/data/paired_spectra/canopus_train/labels.tsv`
- `mist/data/paired_spectra/canopus_train/spec_files/`

- `mist/data/paired_spectra/canopus_train/splits/canopus_hplus_100_0.tsv`

After running `03_retrieval_hdf.py`:

- `mist/data/paired_spectra/canopus_train/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db.h5`

## Any2Graph

Graph-Valued Regression [Model repo](https://github.com/KrzakalaPaul/Any2Graph)

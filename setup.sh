conda create --name dynamic python==3.13.0
conda activate dynamic

conda install -y -c conda-forge pandas=2.2.3
conda install -y -c conda-forge numpy=2.1.2
conda install -y -c conda-forge scikit-learn=1.5.2
conda install -y -c conda-forge matplotlib=3.9.2
conda install -y -c conda-forge seaborn=0.13.2
conda install -y -c conda-forge scipy=1.14.1
conda install -y -c conda-forge xgboost=2.1.2
conda install -y -c conda-forge catboost=
conda install -y -c conda-forge ipykernel=6.29.5
python -m ipykernel install --user --name=dynamic
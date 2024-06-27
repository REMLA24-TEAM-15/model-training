conda create --name remlapy310 python=3.10
conda activate remlapy310
pip install poetry
poetry install
pip install -i https://test.pypi.org/simple/ libml-URLPhishing
mkdir datasets
dvc repro
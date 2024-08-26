


pyenv local 3.9.19
pyenv virtualenv 3.9.19 gpt-from-scratch-env
pyenv activate gpt-from-scratch-env
pip install -r requirements.txt

-- Jupyter
python -m ipykernel install --user --name=kr_gpt_scratch
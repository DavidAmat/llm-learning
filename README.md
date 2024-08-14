# Python Environment SetUp with Poetry

```{bash}
poetry env use /Users/david.amat/.pyenv/versions/3.9.19/bin/python
poetry shell

# modify the pyproject toml adding your dependencies and versions
poetry update
poetry install

# Ipykernel
poetry add --dev ipykernel
python -m ipykernel install --user --name=kr_llm_rag

# Install hopsworks separately due to its enormous dependencies
pip install hopsworks
```

# Hopswork

Project name: pdfsearchllmrag
# Issues


```{bash}
```




```{bash}

```
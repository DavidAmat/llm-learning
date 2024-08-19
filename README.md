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

# OpenSearch

OpenSearch is a free and open-source search and analytics engine that is commonly used for indexing, searching, and analyzing large datasets. It is a community-driven project that originated as a fork of Elasticsearch and Kibana after changes to Elasticsearch's licensing model. 

OpenSearch has added support for vector search capabilities, which makes it capable of serving as a vector database. 



# Issues


```{bash}
```




```{bash}

```
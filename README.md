# DeepDuo
This is the code repo for Second Language Acquisition Modeling with Attention.

The codebase has experiments for baseline attention and multitask attention. 

## Setup

### Installation
```
virtualenv env
source env/bin/activate
```

Install either tensorflow gpu if available or just tensorflow 
```
pip install tensorflow-gpu
```
or
```
pip install tensorflow
```

Create directories for different data language tracks
```
mkdir data_en_es data_es_en data_fr_en
```

### Download data
Download the [SLAM Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8SWHNO) and unzip each of the language tracks in their respective directories.

### Running the Experiments
For the baseline attention experiments - 

```
python train-en-es.py
```

Do the same for other language pairs. 

For the multitask experiments - 

```
python train.py
```

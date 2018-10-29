# ST_subtype_classification
subtype classification of breast cancer based on ST-data, using markov random fields and neural networks. 

Recommended way of procedure is to clone the directory and the generate a soft link to the ST_subtype_classifier.py file in suitable directory.

1. Clone repository

```bash
git clone https://github.com/almaan/ST_subtype_classification.git
```
2. Generate Soft Link

```bash
sudo ln -s PATH_TO_REPOSITORY/ST_subtype_classifier.py /bin/predict_subtype
```
3. Install Necessary dependencies

```bash
pip3 install numpy pandas tensorflow keras graph-tool seaborn scipy scikit-learn
```

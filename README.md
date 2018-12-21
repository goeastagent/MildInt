# MildInt: Deep learning-based multimodal longitudinal data integration framework

Deep learning-based python package for general purpose data integration framework. 

## Getting Started

### Requirements
MildInt requires the following packages to be installed independently

  * pandas
  * numpy
  * tensorflow
  * sklearn

Python pakcage management system can simply setup the packages
```
pip install pandas
pip install numpy
pip install sklearn
pip install tensorflow
```

### Example code

```
m = MMRNN()

m.append_component('m1', m1.shape[2], m1_hidden, m1.shape[1])
m.append_component('m2', m2.shape[2], m2_hidden, m2.shape[1])
m.append_component('m3', m3.shape[2], m3_hidden, m3.shape[1])
```

Setup each modality. In this example, 3 modalities of data (m1, m2, and m3) will be used.
The code above defines name of the modality, dimension of input, dimension of hidden state, and length of time series. Data m has a shape (#samples, length of time series, size of input dimension). 
```

m.append_data('m1', IDs_m1, m1, y_m1, seqlen_m1)
m.append_data('m2', IDs_m2, m2, y_m2, seqlen_m2)
m.append_data('m3', IDs_m3, m3, y_m3, seqlen_m3)

m.append_test_overlapIDs(testIDs)
m.append_training_overlapIDs(trainIDs)
```
Feeding data to MildInt. Training samples as well test samples should be fed to MildInt. And training samples and test samples are seperated by ID. IDs, data (independent variable), y (dependent variable), and seqlen (time lengths of individual sample) should be arranged in order. 

```
m.build_integrative_network()
m.training(batch_size)

m.evalute_accuracy()
```

Training and test. MildInt provides funtions for measuring accuracy, sensitivity, and specificity. 

The entire example code is given in the file "exmple_code.py"

## Feature extraction

MildInt can be used for feature extraction from single domain of data source. After training the MildInt, 

```
m.single_feature_extraction('m1',m1, seqlen_m1)
```

This gives fixed-size of feature vector from single modality.

## Authors

* **Garam Lee** - *Initial work* - [MildInt](https://github.com/goeastagent/MildInt)

See also the list of [contributors](https://github.com/goeastagent/MildInt/contributors) who participated in this project.
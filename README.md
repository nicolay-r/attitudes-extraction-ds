# attitudes-extraction-ds
![](https://img.shields.io/badge/Python-2.7-brightgreen.svg)
![](https://img.shields.io/badge/TensorFlow-1.4.1-yellowgreen.svg)

Source code for RANLP'2019 paper "Distant Supervision for Sentiment Attitude Extraction"

## Convolutional Neural Networks for Relation Extraction 

Architecture aspects of models refers to the following papers:

1. **Relation Classification via Convolutional Deep Neural Network** 
[[paper]](http://www.aclweb.org/anthology/C14-1220) 
[[code]](https://github.com/roomylee/cnn-relation-extraction) 
[[review]](/relation_extraction/Relation_Classification_via_Convolutional_Deep_Neural_Network.md)
We utilize **Position Feature (PF)** -- is an embedding of distance between a given word towards each entity pair
This feature has been originaly proposed in:
	* Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou and Jun Zhao
	* COLING 2014

2. **Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks**
[[paper]](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf) 
[[review]](/relation_extraction/Distant_Supervision_for_Relation_Extraction_via_Piecewise_Convolutional_Neural_Networks.md) 
[[code]](https://github.com/nicolay-r/sentiment-pcnn)
-- We apply and [implement](networks/context/architectures/pcnn.py) 
the related architecture dubbed as  **Piecewise Convolutional Neural Network** (PCNN).
	* Daojian Zeng, Kang Liu, Yubo Chen and Jun Zhao
	* EMNLP 2015

3. **Relation Extraction with Multi-instance Multi-label Convolutional Neural Networks** 
[[paper]](https://pdfs.semanticscholar.org/8731/369a707046f3f8dd463d1fd107de31d40a24.pdf) 
[[review]](/relation_extraction/Relation_Extraction_with_Multi-instance_Multi-label_Convolutional_Neural_Networks.md)
-- This repository provides an [implementation](networks/mimlre) of the related architecture 
(Figure 2 above) in a way of a framework that allows to train models by matching a context (group of sentences) towards sentiment label. 
It assumes to utilize different sentence encoders: CNN, PCNN, etc.
	* Xiaotian Jiang, Quan Wang, Peng Li, Bin Wang
	* COLING 2016

## Installation

All the related resources were used in experiment presented in `data` folder. 
It is necessary to unpack and download (news embedding), as follows:
```
cd data && ./install.sh
```

## References

TODO.

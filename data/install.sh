#!/bin/bash

# unpack rusentrel
unzip "rusentrel-v1_1.zip"

# unpack rusentiframes
unzip "rusentiframes-v1_0.zip"

# unpack ranlp sources
$ranlp_src="ranlp/sources/"
mkdir -p $ranlp_src
unzip "rsr.zip" -d $ranlp_src
unzip "ruattitudes-v1.0.zip" -d $ranlp_src

# download embedding
$w2v_dir=w2v
mkdir -p $w2v_dir
curl http://rusvectores.org/static/models/rusvectores2/news_mystem_skipgram_1000_20_2015.bin.gz -o $w2v_dir"news_rusvectores2.bin.gz"

#!/bin/bash

modelfolder="../dataset/y1"
binfolder="../bin_y1"

rm -f $binfolder/*.scorer
cp ../build_scorer $binfolder
for f in $modelfolder/*.xml; do
	echo "MAKING SCORER FOR MODEL: $f"
	rm -f ../src/scorer.hpp
	$binfolder/build_scorer $f ../src/scorer.hpp
	g++-4.8 -std=c++11 -march=native -Ofast -fopenmp -Wall ../src/scorer.cpp -o $binfolder/$(basename $f xml)scorer
	ls -l $binfolder/$(basename $f xml)scorer
done


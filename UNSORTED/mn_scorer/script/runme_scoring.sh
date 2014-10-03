#!/bin/bash

odir="../scores/y1"
test="/data/letor-tois/yahoo/test.txt"
bindir="../bin_y1"

mkdir -p $odir
for f in $bindir/*.scorer; do
	echo "$f"
	$f $test $odir/$(basename $f .scorer).txt  > $odir/$(basename $f .scorer).log
	scp $odir/$(basename $f .scorer).txt gabriele@barbera1:/home/gabriele/Code/matrixnet_scorer/scores/y1/
	scp $odir/$(basename $f .scorer).log gabriele@barbera1:/home/gabriele/Code/matrixnet_scorer/scores/y1/
done


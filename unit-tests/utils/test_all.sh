#!/bin/bash

reloaded="tree-reloaded-scores"
compiled="tree-if-scores"

rm $reloaded
rm $compiled

for f in $(ls model.* | sort)
do
    echo $f
    ./bin/quicklearn --test test.bad --scores scores.tmp --model $f
    cat scores.tmp >> $reloaded

    ./bin/quicklearn --dump-code baseline.cc --dump-model $f
    make quickscore RANKER=baseline
    ./bin/quickscore -d test.bad -r 1 -s scores.tmp
    cat scores.tmp >> $compiled
done

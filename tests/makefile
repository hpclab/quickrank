all: 
	make msn.fold1.ranklib.lmart.log
	make msn.fold1.quickrank.lmart.log
	make msn.fold1.quickrank.matrixnet.log


msn.fold1.ranklib.lmart.log: ./other/RankLib.jar
	java -jar ./other/RankLib.jar -ranker 6 \
	-train data/msn1.fold1.train.5k.txt \
	-validate data/msn1.fold1.vali.5k.txt \
	-test data/msn1.fold1.test.5k.txt \
	-tree 100 -leaf 8 -shrinkage 0.10 -tc -1 \
	-metric2t NDCG@10 -metric2T NDCG@10 \
	> msn.fold1.ranklib.lmart.log
	cat msn.fold1.ranklib.lmart.log

msn.fold1.quickrank.lmart.log: ../bin/quickrank
	../bin/quickrank \
	--algo lambdamart \
	--num-trees 100 \
	--shrinkage 0.1 \
	--num-thresholds 0 \
	--num-leaves 8 \
	--min-leaf-support 1 \
	--end-after-rounds 0 \
	--train-metric ndcg \
	--train-cutoff 10 \
	--test-metric ndcg \
	--test-cutoff 10 \
	--partial 0 \
	--train data/msn1.fold1.train.5k.txt \
	--valid data/msn1.fold1.vali.5k.txt \
	--test data/msn1.fold1.test.5k.txt \
	--model msn.fold1.quickrank.lmart.xml \
	> msn.fold1.quickrank.lmart.log
	rm msn.fold1.quickrank.lmart.xml*
	cat msn.fold1.quickrank.lmart.log
	
msn.fold1.quickrank.matrixnet.log: ../bin/quickrank
	../bin/quickrank \
	--algo matrixnet \
	--num-trees 100 \
	--shrinkage 0.1 \
	--num-thresholds 0 \
	--tree-depth 3 \
	--min-leaf-support 1 \
	--end-after-rounds 0 \
	--train-metric ndcg \
	--train-cutoff 10 \
	--test-metric ndcg \
	--test-cutoff 10 \
	--partial 0 \
	--train data/msn1.fold1.train.5k.txt \
	--valid data/msn1.fold1.vali.5k.txt \
	--test data/msn1.fold1.test.5k.txt \
	--model msn.fold1.quickrank.matrixnet.xml \
	> msn.fold1.quickrank.matrixnet.log
	rm msn.fold1.quickrank.matrixnet.xml*
	cat msn.fold1.quickrank.matrixnet.log

# correctness check: synching with ranklib
correct:
	make -C ..
	../bin/qr lm 2 0.1 0 8 1 0 \
	ndcg 10 ndcg 10 0 \
	data/msn1.fold1.train.5k.txt \
	- \
	- \
	- \
	quickrank.lmart.xml \
	> quickrank.lmart.log
	cat quickrank.lmart.log

correctranklib:
	cd ~/docs/LAVORO/coding/ltr/ranklib++; ant
	java -jar ~/docs/LAVORO/coding/ltr/ranklib++/bin/RankLib.jar -ranker 6 \
	-train data/msn1.fold1.train.5k.txt \
	-tree 2 -leaf 8 -shrinkage 0.10 -tc -1 \
	-metric2t NDCG@10 -metric2T NDCG@10 \
	-save ranklib.lmart.xml \
	> ranklib.lmart.log
	cat ranklib.lmart.log

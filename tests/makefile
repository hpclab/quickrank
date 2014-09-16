all: 
	make msn.fold1.ranklib.lmart.log
	make msn.fold1.quickrank.lmart.log
	make msn.fold1.quickrank.matrixnet.log


msn.fold1.ranklib.lmart.log:
	java -jar ./other/RankLib.jar -ranker 6 \
	-train data/msn1.fold1.train.5k.txt \
	-validate data/msn1.fold1.vali.5k.txt \
	-test data/msn1.fold1.test.5k.txt \
	-tree 100 -leaf 8 -shrinkage 0.10 -tc -1 \
	-metric2t NDCG@10 -metric2T NDCG@10 \
	> msn.fold1.ranklib.lmart.log
	cat msn.fold1.ranklib.lmart.log
	
msn.fold1.quickrank.lmart.log:
	../bin/qr lm 100 0.1 0 8 1 0 \
	ndcg 10 ndcg 10 0 \
	data/msn1.fold1.train.5k.txt \
	data/msn1.fold1.vali.5k.txt \
	data/msn1.fold1.test.5k.txt \
	- \
	msn.fold1.quickrank.lmart.xml \
	> msn.fold1.quickrank.lmart.log
	rm msn.fold1.quickrank.lmart.xml*
	cat msn.fold1.quickrank.lmart.log

msn.fold1.quickrank.matrixnet.log:
	../bin/qr mn 100 0.1 0 3 1 0 \
	ndcg 10 ndcg 10 0 \
	data/msn1.fold1.train.5k.txt \
	data/msn1.fold1.vali.5k.txt \
	data/msn1.fold1.test.5k.txt \
	- \
	msn.fold1.quickrank.matrixnet.xml \
	> msn.fold1.quickrank.matrixnet.log
	rm msn.fold1.quickrank.matrixnet.xml*
	cat msn.fold1.quickrank.matrixnet.log

# correctness check: synching with ranklib
correct:
	make -C ../src
	../bin/qr lm 1 0.1 0 8 1 0 \
	ndcg 10 ndcg 10 0 \
	data/msn1.fold1.train.5k.txt \
	- \
	- \
	- \
	quickrank.lmart.xml \
	> quickrank.lmart.log
	cat quickrank.lmart.log

correctranklib:
	cd /Users/claudio/Documents/docs/LAVORO/coding/ltr/ranklib++; ant
	java -jar /Users/claudio/Documents/docs/LAVORO/coding/ltr/ranklib++/bin/RankLib.jar -ranker 6 \
	-train data/msn1.fold1.train.5k.txt \
	-tree 1 -leaf 8 -shrinkage 0.10 -tc -1 \
	-metric2t NDCG@10 -metric2T NDCG@10 \
	-save ranklib.lmart.xml \
	> ranklib.lmart.log
	cat ranklib.lmart.log
	
	
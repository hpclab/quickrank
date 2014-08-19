## valgrind do not support instruction produced by option "-march=native"
# cc=$(shell dpkg -l|tr -s [:blank:] ' '|grep "^ii g++"|cut -f2 -d' '|sort|tail -n1)
pwd=$(shell echo $(PWD))

# added for brew compatibility
cc=
ifndef cc
ifneq ($(shell whereis g++-4.8),)
cc=g++-4.8
endif
endif

ifndef cc
ifneq ($(shell whereis g++-4.9),)
cc=g++-4.9
endif
endif

ifndef cc
#ifneq ($(shell brew ls --versions gcc49),)
ifneq ($(shell /usr/local/Cellar/gcc49/4.9.0/bin/g++-4.9 --version),)
cc=/usr/local/Cellar/gcc49/4.9.0/bin/g++-4.9
endif
endif


all: compile # added for eclipse compatibility

compile:
	@mkdir -p ../bin
	$(cc) -std=c++11 -Wall -pedantic -march=native -Ofast -fopenmp -flto -I "." -o ../bin/qr main.cpp

clean:
	@rm -f *~ ../bin/*

run:
	@../bin/qr || echo "ERROR EXIT CODE $$?"

valgrind:
	$(cc) -std=c++11 -O0 -fopenmp -g -I "." -o ../bin/qr main.cpp
	valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ../bin/qr lm 10 0.1 0 4 1 0 ndcg 10 ndcg 10 0 ../tests/data/msn1.fold1.train.5k.txt - - - test.out
#	$(shell export GLIBCXX_FORCE_NEW;



edit:
	@geany $$(cpptree ./main.cpp) --new-instance > /dev/null 2>&1 &

sync:
	rsync -quaRc --delete . gabriele@node1.novello.isti.cnr.it:$(pwd)
#rsync -quaRc --delete . gabriele@cerino:$(pwd)
#rsync -quaRc --delete . gabriele@node5.novello.isti.cnr.it:$(pwd)
#rsync -quaRc --delete . gabriele@barbera1.isti.cnr.it:$(pwd)

dataset:
	@mkdir -p ../bin
	@mkdir -p ../datasets/rand
	@$(cc) -O3 rds.cpp -o ../bin/rds
	@../bin/rds ../datasets/rand/train.txt 20 100 5 10
	@../bin/rds ../datasets/rand/vali.txt 20 20 5 10
	@../bin/rds ../datasets/rand/test.txt 20 10 5 10

tgz:
	mkdir -p ../backup
	tar -czf ../backup/quickrank_$$(date +%Y%m%d_%H%M%S).tgz .

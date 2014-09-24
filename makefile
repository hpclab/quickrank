QUICKRANK=qr
SRCDIR=src
BINDIR=bin
INCDIR=src

CC=
CCFLAGS=-std=c++11 -Wall -pedantic -march=native -Ofast -fopenmp -I $(INCDIR) 
# added for brew compatibility
CCEXTRA=

ifndef CC
ifneq ($(shell whereis g++-4.8),)
CC=g++-4.8
endif
endif

ifndef CC
ifneq ($(shell whereis g++-4.9),)
CC=g++-4.9
endif
endif

ifndef CC
ifneq ($(shell /usr/local/bin/g++-4.9 --version),)
CC=/usr/local/bin/g++-4.9
CCEXTRA=-Wa,-q
endif
endif

all: compile

compile:
	@mkdir -p $(BINDIR)
	$(CC) $(CCFLAGS) $(CCEXTRA) $(SRCDIR)/main.cpp -o $(BINDIR)/$(QUICKRANK)

clean:
	@rm -f *~ $(BINDIR)/*

## valgrind do not support instruction produced by option "-march=native"
# # TODO (cla): be moved in the test folder
# # Valgrind debugging commands
# valgrind-lm:
# 	@mkdir -p ../bin
# 	$(cc) -std=c++11 -O0 -g -I "." -o ../bin/qr main.cpp
# 	valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
# 	../bin/qr lm 10 0.1 0 8 1 0 ndcg 10 ndcg 10 0 \
# 	../tests/data/msn1.fold1.train.5k.txt ../tests/data/msn1.fold1.vali.5k.txt ../tests/data/msn1.fold1.test.5k.txt - test.out
# #	$(shell export GLIBCXX_FORCE_NEW; # this should avoid stl memory pooling

# valgrind-mn:
# 	@mkdir -p ../bin
# 	$(cc) -std=c++11 -O0 -g -I "." -o ../bin/qr main.cpp
# 	valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
# 	../bin/qr mn 10 0.1 0 4 1 0 ndcg 10 ndcg 10 0 \
# 	../tests/data/msn1.fold1.train.5k.txt ../tests/data/msn1.fold1.vali.5k.txt ../tests/data/msn1.fold1.test.5k.txt - test.out

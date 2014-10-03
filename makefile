QUICKRANK:=qr
SRCDIR:=src
BINDIR:=bin
INCDIRS:=-Iinclude

SRCS:=$(wildcard $(SRCDIR)/*.cc) $(wildcard $(SRCDIR)/*/*.cc) $(wildcard $(SRCDIR)/*/*/*.cc)
DEPS:=$(SRCS:.cc=.d)
OBJS:=$(SRCS:.cc=.o)

CC=
CCFLAGS:=-std=c++11 -Wall -pedantic -march=native -Ofast -fopenmp

# find the compiler
ifneq ($(shell whereis g++-4.8),)
	CC=g++-4.8
else 
	ifneq ($(shell whereis g++-4.9),)
  	CC=g++-4.9
  else
    ifneq ($(shell /usr/local/bin/g++-4.9 --version),)
      CC=/usr/local/bin/g++-4.9
      CCFLAGS+=-Wa,-q
    endif
	endif
endif

all: quickrank

quickrank: $(BINDIR)/$(QUICKRANK)

$(BINDIR)/$(QUICKRANK): $(OBJS)
	@mkdir -p $(BINDIR)
	$(CC) $(CCFLAGS) $(OBJS) -o $(BINDIR)/$(QUICKRANK)
#	strip $@
	
        
clean:
	rm -rf $(OBJS) $(DEPS)

dist-clean: clean
	@rm -rf $(BINDIR)

%.d: %.cc
	@$(CC) $(INCDIRS) $(CCFLAGS) -MM -MT $(@:.d=.o) $< > $@ 

%.o: %.cc
	$(CC) $(CCFLAGS) $(INCDIRS) -c -o $@ $<

-include $(DEPS)

# we should put .o files in a different directory
# we should put .d files in a different directory

 
	
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

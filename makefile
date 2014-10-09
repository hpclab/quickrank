QUICKRANK:=quickrank
SRCDIR:=src
BINDIR:=bin
INCDIRS:=-Iinclude
OBJSDIR:=_build
DEPSDIR:=_deps
DOCDIR:=documentation

SRCS:=$(wildcard $(SRCDIR)/*.cc) $(wildcard $(SRCDIR)/*/*.cc) $(wildcard $(SRCDIR)/*/*/*.cc)
SRCS:=$(filter-out $(SRCDIR)/main.cc, $(SRCS))
DEPS:=$(subst $(SRCDIR),$(DEPSDIR)/$(SRCDIR),$(SRCS:.cc=.d))
OBJS:=$(subst $(SRCDIR),$(OBJSDIR)/$(SRCDIR),$(SRCS:.cc=.o))



CC=
CCFLAGS:=-std=c++11 -Wall -pedantic -march=native -Ofast -fopenmp \
		-lboost_program_options

#		-I/usr/local/boost_1_56_0 -L/usr/local/boost_1_56_0/stage/lib \

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

doc:
	cd $(DOCDIR); doxygen quickrank.doxygen
        
clean:
	rm -rf $(OBJSDIR)
	rm -rf $(DEPSDIR)

dist-clean: clean
	@rm -rf $(BINDIR)

$(DEPSDIR)/%.d: %.cc
	@mkdir -p $(dir $@)
	@$(CC) $(INCDIRS) $(CCFLAGS) -MM -MT $(OBJSDIR)/$(<:.cc=.o) $< > $@ 

$(OBJSDIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCDIRS) -c -o $@ $<

-include $(DEPS)
 
	
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

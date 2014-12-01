QUICKLEARN:=quickrank
QUICKSCORE:=quickscore
SRCDIR:=src
UTESTSDIR:=unit-tests
BINDIR:=bin
INCDIRS:=-Iinclude
OBJSDIR:=_build
DEPSDIR:=_deps
DOCDIR:=documentation

# all sources
SRCS:=$(wildcard $(SRCDIR)/*.cc) $(wildcard $(SRCDIR)/*/*.cc) $(wildcard $(SRCDIR)/*/*/*.cc)
DEPS:=$(subst $(SRCDIR),$(DEPSDIR)/$(SRCDIR),$(SRCS:.cc=.d))
OBJS:=$(subst $(SRCDIR),$(OBJSDIR)/$(SRCDIR),$(SRCS:.cc=.o))

# all test sources
UTESTS:=$(wildcard $(UTESTSDIR)/*.cc) $(wildcard $(UTESTSDIR)/*/*.cc) $(wildcard $(UTESTSDIR)/*/*/*.cc)
UTESTSOBJS:=$(subst $(UTESTSDIR),$(OBJSDIR)/$(UTESTSDIR),$(UTESTS:.cc=.o))


CXX=
CXXFLAGS:=-std=c++11 -Wall -pedantic -march=native -Ofast -fopenmp
LDLIBS:=-lboost_program_options -lboost_system -lboost_filesystem -fopenmp

# find the compiler
ifneq ($(shell whereis g++-4.8),)
	CXX=g++-4.8
else 
	ifneq ($(shell whereis g++-4.9),)
	CXX=g++-4.9
  else
    ifneq ($(shell /usr/local/bin/g++-4.9 --version),)
      CXX=/usr/local/bin/g++-4.9
      CXXFLAGS+=-Wa,-q
    endif
	endif
endif

# builds QuickLearn
all: quicklearn

# builds QuickLeanr
quicklearn: $(OBJS)
	@mkdir -p $(BINDIR)
	$(CXX) \
	$(filter-out $(OBJSDIR)/$(SRCDIR)/$(QUICKSCORE).o,$(OBJS)) \
	$(LDLIBS) -o $(BINDIR)/$(QUICKLEARN)

# creates the documentation
doc:
	cd $(DOCDIR); doxygen quickrank.doxygen

# runs all the unit tests
# to run a single test use make unit-tests TEST=dcg_test
unit-tests: $(BINDIR)/unit-tests
	$(BINDIR)/unit-tests --log_level=test_suite --run_test=$(TEST)

# builds QuickScore
# make quickscore RANKER=modelfile
quickscore: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $(RANKER).cc -o $(RANKER).o 
	$(CXX) \
	$(filter-out $(OBJSDIR)/$(SRCDIR)/$(QUICKLEARN).o,$(OBJS)) \
	$(RANKER).o \
	$(LDLIBS) \
	-o $(BINDIR)/$(QUICKSCORE)

# removes intermediate files
clean:
	rm -rf $(OBJSDIR)
	rm -rf $(DEPSDIR)

# removes everything but the source
dist-clean: clean
	@rm -rf $(BINDIR)
	@rm -rf $(DOCDIR)/html

# build dependency files
$(DEPSDIR)/%.d: %.cc
	@mkdir -p $(dir $@)
	@$(CXX) $(INCDIRS) $(CXXFLAGS) -MM -MT $(OBJSDIR)/$(<:.cc=.o) $< > $@ 

# compilation
$(OBJSDIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c -o $@ $<

# linking
$(BINDIR)/unit-tests: $(OBJS) $(UTESTSOBJS)
	$(CXX) \
	$(filter-out $(OBJSDIR)/$(SRCDIR)/$(QUICKLEARN).o $(OBJSDIR)/$(SRCDIR)/$(QUICKSCORE).o,$(OBJS)) \
	$(UTESTSOBJS) \
	$(LDLIBS) -lboost_unit_test_framework \
	-o $(BINDIR)/unit-tests

#include dependency files
-include $(DEPS)

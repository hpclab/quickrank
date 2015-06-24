#
# QuickRank - A C++ suite of Learning to Rank algorithms
# Webpage: http://quickrank.isti.cnr.it/
# Contact: quickrank@isti.cnr.it
#
# Unless explicitly acquired and licensed from Licensor under another
# license, the contents of this file are subject to the Reciprocal Public
# License ("RPL") Version 1.5, or subsequent versions as allowed by the RPL,
# and You may not copy or use this file in either source code or executable
# form, except in compliance with the terms and conditions of the RPL.
#
# All software distributed under the RPL is provided strictly on an "AS
# IS" basis, WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED, AND
# LICENSOR HEREBY DISCLAIMS ALL SUCH WARRANTIES, INCLUDING WITHOUT
# LIMITATION, ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, QUIET ENJOYMENT, OR NON-INFRINGEMENT. See the RPL for specific
# language governing rights and limitations under the RPL.
#
# Contributor:
#   HPC. Laboratory - ISTI - CNR - http://hpc.isti.cnr.it/
 
QUICKLEARN:=quicklearn
QUICKSCORE:=quickscore
SRCDIR:=src
UTESTSDIR:=unit-tests
BINDIR:=bin
INCDIRS:=-Iinclude -I/usr/local/Cellar/boost/1.58.0/include
OBJSDIR:=_build
DEPSDIR:=_deps
DOCDIR:=documentation
TESTDATA:=quickranktestdata

# all sources
SRCS:=$(wildcard $(SRCDIR)/*.cc) $(wildcard $(SRCDIR)/*/*.cc) $(wildcard $(SRCDIR)/*/*/*.cc)
DEPS:=$(subst $(SRCDIR),$(DEPSDIR)/$(SRCDIR),$(SRCS:.cc=.d))
OBJS:=$(subst $(SRCDIR),$(OBJSDIR)/$(SRCDIR),$(SRCS:.cc=.o))

# all test sources
UTESTS:=$(wildcard $(UTESTSDIR)/*.cc) $(wildcard $(UTESTSDIR)/*/*.cc) $(wildcard $(UTESTSDIR)/*/*/*.cc)
UTESTSOBJS:=$(subst $(UTESTSDIR),$(OBJSDIR)/$(UTESTSDIR),$(UTESTS:.cc=.o))

CXX=
CXXFLAGS:=-std=c++11 -Wall -pedantic -march=native -Ofast -fopenmp
LDLIBS:=-lboost_program_options -lboost_system -lboost_filesystem -fopenmp -L/usr/local/Cellar/boost/1.58.0/lib

# find the compiler
ifneq ($(shell whereis g++-4.9),)
	CXX=g++-4.9
else 
  ifneq ($(shell whereis g++-4.8),)
	CXX=g++-4.8
  else
    ifneq ($(shell /usr/local/bin/g++-4.9 --version),)
      CXX=/usr/local/bin/g++-4.9
      CXXFLAGS+=-Wa,-q
    endif
  endif
endif

# builds QuickLearn
all: quicklearn

# builds QuickLearn
quicklearn: $(OBJS)
	@mkdir -p $(BINDIR)
	$(CXX) $(filter-out $(OBJSDIR)/$(SRCDIR)/$(QUICKSCORE).o,$(OBJS)) $(LDLIBS) -o $(BINDIR)/$(QUICKLEARN)

# builds QuickScore
# make quickscore RANKER=modelfile
quickscore: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $(RANKER).cc -o $(RANKER).o 
	$(CXX) $(filter-out $(OBJSDIR)/$(SRCDIR)/$(QUICKLEARN).o,$(OBJS)) $(RANKER).o $(LDLIBS) -o $(BINDIR)/$(QUICKSCORE)


# creates the documentation
doc:
	cd $(DOCDIR); doxygen quickrank.doxygen

# runs all the unit tests
# to run a single test use make unit-tests TEST=dcg_test
.PHONY: unit-tests
unit-tests:  $(TESTDATA) $(BINDIR)/unit-tests
	$(BINDIR)/unit-tests --log_level=test_suite --run_test=$(TEST)

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

# get sample data for unit-testing
$(TESTDATA):
	git clone http://git.hpc.isti.cnr.it/quickrank/quickranktestdata.git $(TESTDATA)
	
# linking
$(BINDIR)/unit-tests:$(OBJS) $(UTESTSOBJS)
	$(CXX) \
	$(filter-out $(OBJSDIR)/$(SRCDIR)/$(QUICKLEARN).o $(OBJSDIR)/$(SRCDIR)/$(QUICKSCORE).o,$(OBJS)) \
	$(UTESTSOBJS) \
	$(LDLIBS) -lboost_unit_test_framework \
	-o $(BINDIR)/unit-tests

#include dependency files
-include $(DEPS)

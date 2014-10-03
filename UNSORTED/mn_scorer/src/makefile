cc=$(shell dpkg -l|tr -s [:blank:] ' '|grep "^ii g++"|cut -f2 -d' '|sort|tail -n1)
pwd=$(shell echo $(PWD))

compile:
	@mkdir -p ../bin
	$(cc) -std=c++11 -Wall -O3 xmlparser.cpp main.cpp -o ../bin/build_scorer

clean:
	@rm -f *~ ../bin/*

edit:
	@geany main.cpp *.?pp --new-instance > /dev/null 2>&1 &

test:
	rm -f scorer.hpp ../bin/scorer
	../bin/build_scorer ../dataset/model.xml ./scorer.hpp
	$(cc) -std=c++11 -march=native -Ofast -fopenmp -Wall ./scorer.cpp -o ../bin/scorer
	../bin/scorer ../dataset/test.txt ../out/scores.txt

sync:
	rsync -quaRc . gabriele@barbera3.isti.cnr.it:$(pwd)


cc=$(shell dpkg -l|tr -s [:blank:] ' '|grep "^ii g++"|cut -f2 -d' '|sort|tail -n1)
pwd=$(shell echo $(PWD))

compile:
	@mkdir -p ../bin
	$(cc) -std=c++11 -Wall -Wextra -Wno-unused-parameter -Ofast -fopenmp -I "." -o ../bin/ranklib main.cpp

clean:
	@rm -f *~ ../bin/*

run:
	@../bin/ranklib || echo "ERROR EXIT CODE $$?"

edit:
	@geany $$(cpptree ./main.cpp) --new-instance > /dev/null 2>&1 &

sync:
	rsync -quaRc --delete . gabriele@node1.novello.isti.cnr.it:$(pwd)
	rsync -quaRc --delete . gabriele@barbera1.isti.cnr.it:$(pwd)

dataset:
	@mkdir -p ../bin
	@mkdir -p ../datasets/rand
	@$(cc) -O3 rds.cpp -o ../bin/rds
	@../bin/rds ../datasets/rand/train.txt 150 750000 5 100
	@../bin/rds ../datasets/rand/vali.txt 150 150000 5 100
	@../bin/rds ../datasets/rand/test.txt 150 10000 5 100

test:
	@mkdir -p ../bin
	$(cc) -std=c++11 -Wall -Wextra -Wno-unused-parameter -O3 -march=native -ffast-math -flto -fopenmp -I "." -o ../bin/test test.cpp && ../bin/test || echo "ERROR EXIT CODE $$?"

valgrind:
	@valgrind --tool=memcheck --leak-check=full ../bin/ranklib

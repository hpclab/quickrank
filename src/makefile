cc=$(shell dpkg -l|tr -s [:blank:] ' '|grep "^ii g++"|cut -f2 -d' '|sort|tail -n1)
pwd=$(shell echo $(PWD))

compile:
	@mkdir -p ../bin
	$(cc) -std=c++11 -Wall -Wextra -Wno-unused-parameter -Ofast -fopenmp -flto -I "." -o ../bin/quicklib main.cpp

clean:
	@rm -f *~ ../bin/*

run:
	@../bin/quicklib || echo "ERROR EXIT CODE $$?"

edit:
	@geany $$(cpptree ./main.cpp) --new-instance > /dev/null 2>&1 &

sync:
	rsync -quaRc --delete . gabriele@node1.novello.isti.cnr.it:$(pwd)
	rsync -quaRc --delete . gabriele@barbera1.isti.cnr.it:$(pwd)

dataset:
	@mkdir -p ../bin
	@mkdir -p ../datasets/rand
	@$(cc) -O3 rds.cpp -o ../bin/rds
	@../bin/rds ../datasets/rand/train.txt 20 100 5 10
	@../bin/rds ../datasets/rand/vali.txt 20 20 5 10
	@../bin/rds ../datasets/rand/test.txt 20 10 5 10

test:
	@mkdir -p ../bin
	$(cc) -std=c++11 -Wall -Wextra -Wno-unused-parameter -Ofast -fopenmp -I "." -o ../bin/test test.cpp && ../bin/test || echo "ERROR EXIT CODE $$?"

valgrind:
	@valgrind --tool=memcheck --leak-check=full ../bin/quicklib

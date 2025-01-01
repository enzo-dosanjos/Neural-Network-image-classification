.PHONY: cleanall

NN: main.o readFile.o utilsNN.o NN.o makefile
	g++ -ansi -pedantic -Wall -std=c++11 -o NN main.o readFile.o utilsNN.o NN.o
	make clean

%.o: %.cpp %.h
	g++ -c -g $<

clean:
	rm -f *.o

cleanall:
	echo Nettoyage	
	rm -f main *.o
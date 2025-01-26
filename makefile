# Variables de compilation
CXX = g++
CXXFLAGS = -ansi -pedantic -Wall -std=c++11
LDFLAGS = $(shell pkg-config --cflags --libs opencv4)
TARGET = NN
SOURCES = main.cpp readFile.cpp utilsNN.cpp NN.cpp
OBJECTS = $(SOURCES:.cpp=.o)
MAP ?= 0  # Pour activer ou désactiver la MAP, cleanall puis make avec l'option MAP=1 ou MAP=0 (0 par défaut).


$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)
	make clean

readFile.o: readFile.cpp readFile.h
	g++ -c -g readFile.cpp $(LDFLAGS)

%.o: %.cpp %.h
	g++ -c -g $<

clean:
	rm -f *.o

cleanall:
	echo Nettoyage	
	rm -f main *.o

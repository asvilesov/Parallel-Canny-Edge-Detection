all: serial parallel

serial: serial.cpp
	g++ serial.cpp -o output `pkg-config --cflags --libs opencv`

parallel: parallel.cpp
	g++ parallel.cpp -o p_output -fopenmp `pkg-config --cflags --libs opencv`

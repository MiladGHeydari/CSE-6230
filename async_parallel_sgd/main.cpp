#include "common.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <cstdlib> 
#include <ctime>
#include <utility>
#include <stdexcept>
#include <sstream>
#include <string>

// =================
// Helper Functions
// =================

double* read_csv(std::string filename) {
	double* result = new double[585024 * 28];

	std::ifstream myFile(filename);

	if(!myFile.is_open()) throw std::runtime_error("Could not open file");

	std::string line, colname;
    double val;

    if(myFile.good()) {
    	getline(myFile, line);
    }

    int i = 0;
    while(getline(myFile, line)) {
    	std::stringstream ss(line);

    	bool start = true;
        int j = 0;
    	while(ss >> val) {
    		if (start) {
    		    start = false;
    		} else {
    		    result[i*28 + j] = val;
    		    j++;
    		}
    		if(ss.peek() == ',') ss.ignore();
    	}
    	i++;
    }

    myFile.close();

    return result;
}

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    double* dataset = read_csv("async_ds.csv");
    double* w = new double[27];

	for (int i = 0; i < 27; i++) {
		w[i] = 1.0;
	}

    // Algorithm
    auto start_time = std::chrono::steady_clock::now();

    train(dataset, w);

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> timediff = end_time - start_time;
    double seconds = timediff.count();
	std::cout << "Simulation Time = " << seconds << "\n";

	delete[] dataset;
	delete[] w;

	return 0;
}

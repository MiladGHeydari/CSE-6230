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
#include <mpi.h>

// =================
// Helper Functions
// =================

double* read_csv(std::string filename) {
        double* result = new double[500000 * 28];

        std::ifstream myFile(filename);

        if(!myFile.is_open()) throw std::runtime_error("Could not open file");

        std::string line, colname;
    double val;

    if(myFile.good()) {
        getline(myFile, line);
    }

    int i = 0;
    while(i < 500000 && getline(myFile, line)) {
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

    //std::cout << "hi" << std::endl;
    myFile.close();

    return result;
}

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    //double* dataset = read_csv("d.csv");
    double* dataset = new double[500000 * 28];
    double* w = new double[27];
    
    for (int i = 0; i < 27; i++) {
        w[i] = 1.0;
    }

    // Init MPI
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        dataset = read_csv("d.csv");
    }

    MPI_Bcast(dataset, 500000 * 28, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Algorithm
    auto start_time = std::chrono::steady_clock::now();

    train(dataset, w, rank, num_procs);

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    if (rank == 0) {
        std::cout << "Simulation Time = " << seconds << " for " << num_procs << " procs" << "\n";
    }

    delete[] dataset;
    delete[] w;
    MPI_Finalize();
    return 0;
}

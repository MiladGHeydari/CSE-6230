#include <cstdlib> 
#include <ctime>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <utility>
#include <stdexcept>
#include <sstream>
#include <chrono>
#include <omp.h>

using namespace std;

void train(double* dataset, double* w) {
    omp_lock_t weight_lock;
    omp_init_lock(&weight_lock);
    omp_lock_t loss_lock;
    omp_init_lock(&loss_lock);
    double prev_loss = 10.0;
    double diff = 1.0;
    int epoch = 0;

    while (prev_loss > 6.75) {
        double loss = 0.0;
        #pragma omp parallel for
        for (int i = 0; i < 585024; i += 8192) {
            double* grad = new double[27];
            double* w_copy = new double[27];
            double curr_loss = 0.0;
            omp_set_lock(&weight_lock);
            for (int j = 0; j < 27; ++j) {
                grad[j] = 0.0;
                w_copy[j] = w[j];
            }
            omp_unset_lock(&weight_lock);
            for (int ii = i; ii < min(i+8192, 585024); ++ii) {
                double y = dataset[ii*28 + 27];
                double y_hat = 0.0;
                for (int j = 0; j < 27; ++j) {
                    y_hat += dataset[ii*28 + j] * w_copy[j];
                }
                double diff = y_hat - y;
                curr_loss += diff * diff;
                for (int j = 0; j < 27; ++j) {
                    grad[j] += 2 * diff * dataset[ii*28 + j];
                }
            }
            omp_set_lock(&weight_lock);
            for (int j = 0; j < 27; ++j) {
                w[j] -= 0.0000001 * (grad[j] / 8192);
            }
            omp_unset_lock(&weight_lock);
            omp_set_lock(&loss_lock);
            loss += curr_loss;
            omp_unset_lock(&loss_lock);
            delete[] grad;
            delete[] w_copy;
        }
        loss = loss / 585024;
        diff = abs(prev_loss - loss);
        prev_loss = loss;
        if (epoch % 10 == 0) {
            cout << "Epoch: " << epoch << " Loss: " << loss << endl;
        }
        epoch++;
    }
}

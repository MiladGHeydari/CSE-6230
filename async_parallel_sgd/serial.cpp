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

using namespace std;

void train(double* dataset, double* w) {
    double prev_loss = 10.0;
    double diff = 1.0;
    int epoch = 0;
    //std::cout << dataset[0] << std::endl;
    //std::cout << dataset[50*12 + 0] << std::endl;
    //std::cout << dataset[0 + 11] << std::endl;
    //std::cout << dataset[5478*12 + 11] << std::endl;
    //std::cout << dataset[1000000*12 - 1] << std::endl;
    while (prev_loss > 6.75) {
        //for (int i = 0; i < 11; i++) {
            //cout << w[i] << " ";
        //}
        //cout << endl;

        double loss = 0.0;
        
	for (int i = 0; i < 585024; i += 8192) {
            double* grad = new double[27];
	    for (int t = 0; i < 27; i++) {
                grad[t] = 0.0;
            }
            for (int ii = i; ii < min(i+8192, 585024); ++ii) {
                double y = dataset[ii*28 + 27];
	        double y_hat = 0.0;
		for (int j = 0; j < 27; ++j) {
                    y_hat += dataset[ii*28 + j] * w[j];
	        }
		double diff = y_hat - y;
	        loss += diff * diff;
		for (int j = 0; j < 27; ++j) {
		    grad[j] += 2 * diff * dataset[ii*28 + j];
		}
	    }

            //if (i % 1000 == 0) {
                //cout << "Epoch: " << epoch << " Ind: " << i << " Loss: " << loss / (i + 128) << endl;
            //}

	    for (int j = 0; j < 27; ++j) {
                //cout << grad[j] << " ";
	        w[j] -= 0.0000001 * (grad[j] / 8192);
	    }
            //cout << endl;
	    delete[] grad;
	}
	loss = loss / 585024;
	diff = abs(prev_loss - loss);
	prev_loss = loss;
	cout << "Epoch: " << epoch << " Loss: " << loss << endl;
	//cout << "Diff " << diff << endl;
	epoch++;
    }
}

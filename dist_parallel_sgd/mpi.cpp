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
#include <mpi.h>

using namespace std;

void train(double* dataset, double* w, int rank, int num_procs) {
    double train_time = 0;
    double comm_time = 0;
    double prev_loss = 10.0;
    double diff = 1.0;
    int epoch = 0;
    int samples_per_proc = (500000 + num_procs - 1) / num_procs;
    int start_idx = min(rank * samples_per_proc, 500000);
    int end_idx = min(start_idx + samples_per_proc, 500000);
    //cout << rank << " start idx " << start_idx << " end idx " << end_idx << endl;
    //cout << rank << " data[0] " << dataset[0] << endl;
    //cout << rank << " data[153] " << dataset[153 * 28] << endl;
    //cout << "num_procs " << num_procs << endl;
    while (prev_loss > 6.75) {

        auto train_start_time = chrono::steady_clock::now();
        double loss = 0.0;

        for (int i = start_idx; i < end_idx; i += 1) {
            double* grad = new double[27];
            for (int k = 0; k < 27; k++) {
                grad[k] = 0.0;
            }
            for (int ii = i; ii < min(i+1, end_idx); ++ii) {
                double y = dataset[ii*28 + 27];
                double y_hat = 0.0;
                for (int j = 0; j < 27; ++j) {
                    y_hat += dataset[ii*28 + j] * w[j];
                }
                //double prevl = loss;
                double diff = y_hat - y;
                loss += diff * diff;
                //if (rank == 0) {
                    //cout << "ii " << ii << endl;
                    //cout << "y_hat " << y_hat << endl;
                    //cout << "y " << y << endl;
                    //cout << "diff " << diff << endl;
                    //cout << "diff2 " << diff * diff << endl;
                    //cout << "prev loss " << prevl << endl;
                    //cout << "loss " << loss << endl;
                //}
                for (int j = 0; j < 27; ++j) {
                    grad[j] += 2 * diff * dataset[ii*28 + j];
                }
            }
            //if (rank == 0) {
                //cout << "rank " << rank << " ind " << i << " loss " << loss << endl;
            //}

            for (int j = 0; j < 27; ++j) {
                w[j] -= 0.0000001 * (grad[j] / 1);
            }
            delete[] grad;
        }
        if (start_idx < end_idx) {
            loss = loss / 500000;
        }
        auto train_end_time = chrono::steady_clock::now();
        auto comm_start_time = chrono::steady_clock::now();
        double recv_loss = 0.0;
        MPI_Allreduce(&loss, &recv_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double* recv_w = new double[27];
        MPI_Allreduce(w, recv_w, 27, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (int j = 0; j < 27; ++j) {
            w[j] = recv_w[j] / num_procs;
        }
        auto comm_end_time = chrono::steady_clock::now();
        chrono::duration<double> traindiff = train_end_time - train_start_time;
        chrono::duration<double> commdiff = comm_end_time - comm_start_time;
        train_time += traindiff.count();
        comm_time += commdiff.count();
        if (rank == 0) {
            cout << rank << " epoch " << epoch << " loss " << loss << " recv loss " << recv_loss << endl;
        }
        diff = abs(prev_loss - recv_loss);
        prev_loss = recv_loss;
        //diff = abs(prev_loss - loss);
        //prev_loss = loss;
        //if (rank == 0) {
            //cout << "Epoch: " << epoch << " Loss: " << loss << endl;
        //}
        epoch++;
        delete[] recv_w;
    }
    if (rank == 0) {
        cout << "comm time " << comm_time << " train_time " << train_time << " ratio " << comm_time / (comm_time+train_time) << endl;
    }
}

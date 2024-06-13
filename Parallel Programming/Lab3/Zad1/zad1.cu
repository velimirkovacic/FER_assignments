#include <iostream>


using namespace std;



__global__ void findPrim(int* niz, int* result, int size) { 
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x; 
    while(tid < size){
        bool divisorFound = false;
        for(int i = 2; i*i <= niz[tid]; i++) {
            if((niz[tid] / i) * i == niz[tid]) {
                divisorFound = true;
                break;
            }
        }

        if(!divisorFound && niz[tid] != 1) 
            atomicAdd(result, 1);
            //*result += 1;

        tid += blockDim.x * gridDim.x; 
    }
}



int main(int argc, char *argv[]) {
    int N = 100;

    int *h_niz, *h_result;
    h_niz = (int*)malloc(N * sizeof(int));
    h_result = (int*)malloc(sizeof(int));

    for(int i = 0; i < N; i++) h_niz[i] = i + 1;
    *h_result = 0;

    int *d_niz, *d_result;
    cudaMalloc(&d_niz, N * sizeof(int));
    cudaMalloc(&d_result,sizeof(int));


    cudaMemcpy(d_niz, h_niz, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result, sizeof(int), cudaMemcpyHostToDevice);

    
    int threadsPerBlock = 256; // velicina radne grupe grupe dretvi
    int blocksPerGrid = 8; // grid size = velicina skupa dretvi


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    findPrim<<<blocksPerGrid, threadsPerBlock>>>(d_niz, d_result, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);


    cudaMemcpy(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);


    cout << "U skupu brojeva {1, 2, 3, ..., " << N << "} ima ukupno " << *h_result << " prostih brojeva." << endl;
    cout << "Vrijeme izvodenja: " << milliseconds << " ms" << endl;
    cout << "Broj blokova: " << blocksPerGrid << endl;
    cout << "Broj dretvi po bloku: " << threadsPerBlock << endl;
    free(h_niz);
    cudaFree(d_niz);
}

    

    
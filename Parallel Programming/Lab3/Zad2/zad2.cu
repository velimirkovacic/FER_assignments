#include <iostream>
#include <curand_kernel.h>

using namespace std;



__global__ void sampleCircle(unsigned long long* result, unsigned long long* total, unsigned long long tasks) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    curandState state;
    curand_init(321, tid, 0, &state);
    
    for(unsigned long long i = 0; i < tasks; i++) {
        float x =  curand_uniform(&state);
        float y =  curand_uniform(&state);
        if(x*x + y*y <= 1.0)
            atomicAdd(result, 1);
        atomicAdd(total, 1);
    }
    


}



int main(int argc, char *argv[]) {
    unsigned long long N = 1;
    int n;
    cout << "Unesite broj n, (N = 2^n)" << endl << "n = ";
    cin >> n;
    N = N << n;
    cout << "N = " << N << endl;

    unsigned long long *h_krug, *h_uk;
    h_krug = (unsigned long long*)malloc(sizeof(unsigned long long));
    h_uk = (unsigned long long*)malloc(sizeof(unsigned long long));
    *h_krug = 0;
    *h_uk = 0;

    unsigned long long *d_krug, *d_uk;
    cudaMalloc(&d_krug, sizeof(unsigned long long));
    cudaMalloc(&d_uk, sizeof(unsigned long long));

    cudaMemcpy(d_krug, h_krug, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uk, h_uk, sizeof(int), cudaMemcpyHostToDevice);

    unsigned long long threadsPerBlock = 64; // velicina radne grupe grupe dretvi
    unsigned long long blocksPerGrid = 1; // grid size = velicina skupa dretvi
    unsigned long long tasks = N / blocksPerGrid / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    sampleCircle<<<blocksPerGrid, threadsPerBlock>>>(d_krug, d_uk, tasks);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);


    cudaMemcpy(h_krug, d_krug, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uk, d_uk, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cout << "Unutar kruga: " << *h_krug << endl;
    cout << "Ukupno: " << *h_uk << endl;
    cout << "Aproksimacija broja PI: " << 4.0 * (float) *h_krug / (float) *h_uk << endl;
    cout << "Vrijeme izvodenja: " << milliseconds << " ms" << endl;
    cout << "Broj blokova: " << blocksPerGrid << endl;
    cout << "Broj dretvi po bloku: " << threadsPerBlock << endl;
    cout << "Broj zadataka po dretvi: " << tasks << endl;


    free(h_krug);
    free(h_uk);
    cudaFree(d_krug);
    cudaFree(d_uk);
    
}

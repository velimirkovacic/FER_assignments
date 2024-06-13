#include <iostream>
#include <chrono>
#include <random>

using namespace std;



int main() {
    unsigned long long N = 1;
    int n;
    cout << "Unesite broj n, (N = 2^n)" << endl << "n = ";
    cin >> n;
    N = N << n;
    cout << "N = " << N << endl;

    unsigned long long result, total;
    result = 0;
    total = 0;
    std::mt19937_64 generator(321); // Seed the random number generator
    std::uniform_real_distribution<float> distribution(0.0, 1.0);


    auto start = std::chrono::high_resolution_clock::now();

    for (unsigned long long i = 0; i < N; i++) {
        float x =  distribution(generator);
        float y =  distribution(generator);
        if (x * x + y * y <= 1.0)
            result++;
        total++;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    cout << "Aproksimacija broja PI: " << 4.0 * (float) result / (float) total << endl;
    cout << "Vrijeme izvodenja: " << duration.count() << " ms" << endl;



    return 0;
}
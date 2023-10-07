#include<bits/stdc++.h>

using namespace std;

long long mat[1001][1001];
long long best[1001][1001];
long long bestL[1001];
long long bestR[1001];
int n;

void printLR() {
    cout << "L:" << endl;
    for(int j = 0; j < n; j++) {
        
        cout << bestL[j] << " ";
    } cout << endl;
        cout << "R:" << endl;
    for(int j = 0; j < n; j++) {

        cout << bestR[j] << " ";
    } cout << endl;
}

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);

    freopen("input.txt", "r", stdin);

    cin >> n;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            cin >> mat[i][j];
        }
    }

    //1. redak
    best[0][0] = mat[0][0];
    for(int j = 1; j < n; j++) best[0][j] = best[0][j - 1] + mat[0][j];

    //ostali reci
    for(int i = 1; i < n; i++) {
        bestL[0] = best[i - 1][0] + mat[i][0];
        for(int j = 1; j < n; j++) {
            bestL[j] = max(bestL[j - 1], best[i - 1][j]) + mat[i][j];
        }
        bestR[n - 1] = best[i - 1][n - 1] + mat[i][n - 1];
        for(int j = n - 2; j >= 0; j--) {
            bestR[j] = max(best[i - 1][j], bestR[j + 1]) + mat[i][j];
        }
        for(int j = 0; j < n; j++) {
            best[i][j] = max(bestR[j], bestL[j]);
        }

        //printLR();
    }

    cout << best[n - 1][n - 1];

    return 0;
}
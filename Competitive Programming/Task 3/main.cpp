#include<bits/stdc++.h>
#define MOD 1000000007
using namespace std;

long long nMem[1000003];
long long mMem[1000003];
bool nBio[1000003];
bool mBio[1000003];

long long N(int n);


long long M(int n) {
    //if(n > -5) cout << "M(" << n << ")" << endl;
    if(!mBio[n]) {
        mMem[n] = (2 * M(n - 1) + 2 * N(n - 1) + M(n - 2)) % MOD;
        mBio[n] = 1;

    }
    return mMem[n];
}

long long N(int n) {
   // cout << "N(" << n << ")" << endl;
    if(!nBio[n]) {
        nMem[n] = (M(n - 1) + N(n - 1)) % MOD;
        nBio[n] = 1;
    }

    return nMem[n];
}



int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);

    freopen("input.txt", "r", stdin);

    mMem[0] = 1;
    mMem[1] = 2;
    mBio[0] = 1;
    mBio[1] = 1;

    nMem[1] = 1;
    nBio[1] = 1;


    int k, tmp; cin >> k;
    M(1000000);
    for(int i = 0; i < k; i++) {
        cin >> tmp;
        cout << M(tmp) << endl;
    }

    return 0;
}
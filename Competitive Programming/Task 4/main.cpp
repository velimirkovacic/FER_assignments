#include<bits/stdc++.h>

using namespace std;

int niz[100005];
//int pref[100005];
int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);

    freopen("input.txt", "r", stdin);
/*
    for(int l = 1; l <= 5; l++) {
        for(int r = l + 1; r <= 6; r++) {
            cout << "interval: " << l << " - " << r << endl;
            cout << "duljina: " << r - l + 1 << endl;
            cout << "broj jedinica: " << pref[r ] - pref[l - 1] << endl;
            cout << "broj nula: " << r - l + 1 - pref[r] + pref[l - 1] << endl;
        }
    }
*/

    int n, k;
    cin >> n >> k;
    
    for(int i = 0; i < n; i++) {
        char tmp; cin >> tmp;
        niz[i] = tmp - '0';
        //pref[i] = pref[i - 1] + niz[i];
    }

    int left = 0;
    int brojacNula = 0;
    int maxn = 0;
//izvor: https://www.geeksforgeeks.org/longest-subsegment-1s-formed-changing-k-0s/
    for (int right = 0; right < n; right++) {
        if (niz[right] == 0) {
            brojacNula++;
        }

        while (brojacNula > k) {
            if (niz[left] == 0) {
                brojacNula--;
            }
            left++;
        }

        maxn = max(maxn, right - left + 1);
    }

    cout << maxn;

    return 0;
}
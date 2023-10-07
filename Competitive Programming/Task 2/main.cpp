#include<bits/stdc++.h>

using namespace std;

pair<long long, long long> capar[100005];


int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);

    freopen("input.txt", "r", stdin);

    int n, q;
    cin >> n >> q;

    long long atm = 0;

    for(int i = 0; i < n; i++) {
        long long cijena, atmosfera;
        cin >> cijena >> atmosfera;
        capar[i] = make_pair(cijena, atmosfera);
        atm += atmosfera;
    }

    sort(capar, capar + n);

    int cnt = 0;
    for(int i = 1; i < n; i++) {
        if(capar[i].first == capar[i - 1].first) {
            capar[i].second += capar[i - 1].second;
            capar[i - 1].first = 9900000000;
            cnt ++;
        }
    }
    sort(capar, capar + n);
    n -= cnt;


    for(int i = 0; i < n; i++) {
        long long tmp = capar[i].second;
        capar[i].second = atm;
        atm -= tmp;

        //cout << capar[i].first << " " << capar[i].second << endl;
    }

    capar[n].second = 0;

    for(int i = 0; i < q; i++) {
        long long trazi; cin >> trazi;
        if(capar[0].second < trazi || capar[n - 1].second > trazi) {
            cout << -1 << endl;
            continue;
        }

        int kraj = n - 1;
        int pocetak = 0;
        int sredina;
        //cout << "binary: "<<endl;
        while(kraj >= pocetak) {
            
            sredina = (kraj + pocetak)/2;
            //cout << sredina << endl;
            if(capar[sredina].second >= trazi && capar[sredina + 1].second < trazi) {
                cout << capar[sredina].first << endl;
                break;
            }
            if(capar[sredina].second > trazi) pocetak = sredina + 1;
            else kraj = sredina;

        }
        //cout << "binary kraj" << endl;
    }

 
    return 0;
}
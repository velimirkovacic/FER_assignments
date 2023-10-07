#include<iostream>
#include<time.h>
#define MAXN 12
using namespace std;

/*
    Razred koji opisuje potpuni težinski graf s do MAXN vrhova.
    Omogućava 2 vrste pretrage: pohlepnim algoritmom i iscrpnom pretragom u svrhu pronalaženja
    najkraćeg hamilotnovskog ciklusa, to jest rješenja problema trgovačkog putnika.
*/

class PotpuniTezinskiGraf {
    private:
        int n, a, b;
        int distMat[MAXN + 1][MAXN + 1];
        bool posjeceno[MAXN + 1];

        //  Ispunjava listu posjećenih vrhova s false ~ "niti jedan vrh nije posjećen".
        void resetirajPosjecenost() {
            for(int i = 1; i <= MAXN; i++)
                posjeceno[i] = false;
        }

        //  Rekurzija koju poziva funkcija rjesiIscrpnim.
        void rjesiIscrpnimRekurzija(int trenutni, int udaljenost, int &minUdaljenost) {
            bool dno = true;

            for(int i = 2; i <= n; i++) {
                if(!posjeceno[i]) {
                    dno = false;

                    posjeceno[i] = true;
                    rjesiIscrpnimRekurzija(i, udaljenost + distMat[trenutni][i], minUdaljenost);
                    posjeceno[i] = false;
                }
            }
        
            if(dno) {
                udaljenost += distMat[1][trenutni];
                if(udaljenost < minUdaljenost) minUdaljenost = udaljenost;
            }
        }


    public:

        //  Konstruktor stvara matricu udaljenosti vrhova
        PotpuniTezinskiGraf(int n, int a, int b) : a(a), b(b), n(n) {
            for(int i = 1; i <= n; i++) {
                for(int j = i + 1; j <= n; j++) {
                    distMat[j][i] = distMat[i][j] = dist(i, j, a, b);
                }
            }
        }

        //  Funkcija računa udaljenost 2 vrha k i l, k < l
        static int dist(int k, int l, int a, int b) {
            return (a*k + b*l) * (a*k + b*l) + 1;
        }

        /*
            Pohlepni algoritam traži nakraći hamiltonovski ciklus u grafu na sljedeći način:
            1.) Pronađi najkraći brid u grafu (ako ih je više uzmi 1.)
                Taj brid je incidentan s 2 vrha: "lijevi" i "desni".
            2.) S lijevog i desnog vrha pronađi vrh do kojeg je nakraći put s 1 od njih.
            3.) Poveži ih.
                Sada je lijevi (ili desni) vrh pomaknut
            4.) Ponavljajući korake 2 i 3 gradi lanac dok postoje vrhovi koji nisu u lancu.
            5.) Poveži krajeve lanca.

            Udaljenost je duljina tog lanca.
        */
        int rjesiPohlepnim() {
            int lijevi = 1;
            int desni = 2;
            int udaljenost = 0;

            for(int i = 1; i <= n; i++) {
                for(int j = i + 1; j <= n; j++) {
                    if(distMat[i][j] < distMat[lijevi][desni]) {
                        lijevi = i;
                        desni = j;
                    }
                }
            }
            //cout << "lijevi: " << lijevi << " desni:" << desni << endl; 

            resetirajPosjecenost();
            posjeceno[lijevi] = true;
            posjeceno[desni] = true;
            udaljenost += distMat[lijevi][desni];

            for(int i = 1; i <= n - 2; i++) {
                int minVrh = 0;
                int minBrid = 0;
                int odabranDesni = false;

                for(int j = 1; j <= n; j++) {
                    if(!posjeceno[j] && (distMat[lijevi][j] < minBrid || minVrh == 0) ) {
                        minVrh = j;
                        minBrid = distMat[lijevi][j];
                    }
                }
                for(int j = 1; j <= n; j++) {
                    if(!posjeceno[j] && (distMat[desni][j] < minBrid || minVrh == 0) ) {
                        minVrh = j;
                        minBrid = distMat[desni][j];
                        odabranDesni = true;
                    }
                }

                posjeceno[minVrh] = true;

                if(odabranDesni) {
                    udaljenost += distMat[desni][minVrh];
                    desni = minVrh;
                } else {
                    udaljenost += distMat[lijevi][minVrh];
                    lijevi = minVrh;
                }

                //cout << "lijevi: " << lijevi << " desni:" << desni << endl; 
            }
            udaljenost += distMat[lijevi][desni];        
            return udaljenost;
        }

        /*
            Funkcija provodi iscrpno pretraživanje grafa tražeći najkraći put na sljedeći način:
            1.) Fiksira se 1 vrh (npr. vrh '1').
            2.) Gledaju se svi putevi iz njega.
            3.) Rekurzivno pretražuj sve mogućnosti.
            4.) Na dnu rekurzije provjeri je li put kraći od do sada najkraćeg.
            
            Na kraju će se dobiti najkraći mogući put.
        */
        int rjesiIscrpnim(int pohlepnoRjesenje) {
            int minUdaljenost = pohlepnoRjesenje;
            resetirajPosjecenost();
            rjesiIscrpnimRekurzija(1, 0, minUdaljenost);

            return minUdaljenost;
        }
};


//  Glavni program.
int main() {
    int pohlepni, iscrpni;
    int n, a, b;
    
    cout << "Unesite redom, odvojene razmakom, parametre  n,  a  i  b: ";
    cin >> n >> a >> b;

    PotpuniTezinskiGraf* graf = new PotpuniTezinskiGraf(n, a, b);

    pohlepni = graf->rjesiPohlepnim();
    cout << "Pohlepni algoritam nalazi ciklus duljine " << pohlepni << endl;

    //int tajm = time(NULL);

    iscrpni = graf->rjesiIscrpnim(pohlepni);
    cout << "Iscrpni algoritam nalazi ciklus duljine " << iscrpni << endl;

    cout << "Pohlepni algoritam na ovom grafu ";
    if(pohlepni > iscrpni) cout << "ne ";
    cout << "daje optimalno rjesenje!" << endl;
    //cout << "Traje " << (time(NULL)) << " s";

    delete graf;
    return 0;
}
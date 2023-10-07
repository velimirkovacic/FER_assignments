#include<iostream>
#include<vector>
#include<climits>
#include<algorithm>

// MAXS = najveća moguća duljina naziva datoteke u kojoj su spremljeni podaci o grafu
// MAXN = najveći mogući broj vrhova u grafu (+1 jer se 0 ne smatra vrhom)
#define MAXS 100
#define MAXN 20 + 1

using namespace std;


// Funkcija za unos podataka i inicijalizaciju grafa
void init(int &V, vector<int> *graf) {
    char datoteka[MAXS + 1];    // ime datoteke s podacima
    int S_size;                 // broj brojeva u skupu S
    bool S[MAXN];               // S[i] odgovara na pitanje je li i u skupu S

    for(int i = 0; i < MAXN; i++) S[i] = false;

    cout << "Unesite  ime  datoteke: ";
    scanf("%s", datoteka);

    freopen(datoteka, "r", stdin);

    cin >> V >> S_size;

    for(int i = 0; i < S_size; i++) {
        int tmp; cin >> tmp;
        S[tmp] = true;
    }

    // Postavljanje matrice vektora susjedstava
    for(int i = 1; i <= V; i++) {
        for(int j = 1; j <= V; j++) {
            if(i != j && S[abs(i - j)]) {
                graf[i].push_back(j);
            }
        }
    }
}


// Funkcija koja ispisuje s kojim se vrhovima svaki vrh povezan
void povezanostPrint(int &V, vector<int> *graf) {
    for(int i = 1; i <= V; i++) {
        cout << i << " je povezan s: ";
        vector<int>::iterator it;
        for(it = graf[i].begin(); it < graf[i].end(); it++) {
            cout << *it << " ";
        }
        cout << endl;
    }
}

// Rekurzivna funkcija ("backtracking" + "pruning"), koja boja vrh po vrh, za traženje kromatskog broja 
void traziKb(int &V, vector<int> *graf, int trenutniVrh, int *matricaObojanosti, int &maxKb, int &kb, int &brojKoristenih, bool *koristenaBoja) {
    
    // Ako su svi vrhovi iskorišteni ažuriraj kromatski broj
    if(trenutniVrh > V) {
        kb = min(brojKoristenih, kb);
        return;
    } 
    
    // Inače pozovi rekurziju za svaku moguću boju na trenutnom vrhu

    // Iterira po svim bojama
    for(int i = 1; i <= maxKb; i++) {
        bool flag = false;
        vector<int>::iterator it;

        // Provjerava ima li trenutnu boju neki od susjednih vrhova
        for(it = graf[trenutniVrh].begin(); it < graf[trenutniVrh].end(); it++) {
            if(i == matricaObojanosti[*it]) {
                flag = true;
                break;
            }
        }

        // Ako nema susjeda s trenutnom bojom, onda boja trenutni vrh tom bojom
        if(!flag) {
            matricaObojanosti[trenutniVrh] = i;

            // Ako je boja već korištena onda samo poziva rekurziju za sljedeći vrh
            if(koristenaBoja[i]) {
                traziKb(V, graf, trenutniVrh + 1, matricaObojanosti, maxKb, kb, brojKoristenih, koristenaBoja);
            }

            // Inače proverava hoće li dodatak te boje biti veći od najmanjeg pronađenog kromatskog broja
            else if(brojKoristenih + 1 < kb) {
                koristenaBoja[i] = 1;
                brojKoristenih++;
                traziKb(V, graf, trenutniVrh + 1, matricaObojanosti, maxKb, kb, brojKoristenih, koristenaBoja);
                koristenaBoja[i] = 0;
                brojKoristenih--;
            }
        }
    }

    // Skidanje boje prije povratka iz rekurzije
    matricaObojanosti[trenutniVrh] = 0;
}


// Funkcija koja priprema i poziva rekurzivnu funkciju za traženje kromatskog broja
int kromatskiBroj(int &V, vector<int> *graf) {
    if(V == 1 || V == 0) return V;

    // Kromatski broj za jednostavan graf jednak je Δ + 1, ako je Δ najveći stupanj nekog vrha u grafu
    int maxKb = graf[1].size();
    for(int i = 2; i <= V; i++) {
        maxKb = max(maxKb, (int) graf[i].size());
    }
    maxKb = maxKb + 1;

    // Postavljanje liste obojanosti, inicijalno su svi 0 (neobojano), osim vrha 1 koji je 1 (prve boje)
    int matricaObojanosti[MAXN];
    for(int i = 1; i <= V; i++) matricaObojanosti[i] = 0;
    matricaObojanosti[1] = 1;

    // Postavljanje liste iskoristenosti boja, inicijalno su sve 0 (neiskorištene), osim prve boja koja je 1 (iskorištena)
    bool koristenaBoja[MAXN];
    for(int i = 1; i <= V; i++) koristenaBoja[i] = 0;
    koristenaBoja[1] = 1;

    int brojkoristenih = 1;
    int kb = maxKb;

    // Poziv rekurzivne funkcije za traženje kromatskog broja
    traziKb(V, graf, 2, matricaObojanosti, maxKb, kb, brojkoristenih, koristenaBoja);

    return kb;
}


// Glavni program
int main() {
    int V;
    vector<int> graf[MAXN];

    init(V, graf);
    povezanostPrint(V, graf);
    int kb = kromatskiBroj(V, graf);

    cout << "Kromatski broj zadanog grafa je " << kb;

    return 0;
}







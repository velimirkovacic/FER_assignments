/*
Koliko ima peteroznamenkastih brojeva koji imaju najvise jednu znamenku 1 i 
najvise jednu znamenku 2, a ako imaju obje, onda je znamenka 1 lijevo od znamenke 2?
*/


#include<iostream>
#include<string>

#define LEFT_BOUND 10000
#define RIGHT_BOUNT 100000

// Provjerava unutar skupa {LEFT_BOUND, LEFT_BOUND + 1, ... , RIGHT_BOUND - 1}

using namespace std;

bool provjera(string broj) {
    bool br1 = 0;
    bool br2 = 0;

    for(int i = 0; i < broj.length(); i++) {
        if(broj[i] == '1' && br2 == 0 && br1 == 0) br1 = 1;
        else if(broj[i] == '2' && br1 <= 1 && br2 == 0) br2 = 1;
        else if(broj[i] != '1' && broj[i] != '2') continue;
        else return 0;
    }
    return 1;
}

int main() {
    int brojac = 0;
    
    for(int i = LEFT_BOUND; i < RIGHT_BOUNT; i++) {
        brojac += provjera(to_string(i));
    }

    cout << "Takvih brojeva ima: " << brojac << endl;


    return 0;
}
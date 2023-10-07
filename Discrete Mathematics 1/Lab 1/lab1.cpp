#include<iostream>
#include<math.h>

using namespace std;

double izracunFormulom(double lamda1, double lamda2, double a0, double a1, int n) {
    /*  
        Rekurzija:
        a(n) = λ1 * a(n - 1) + λ2 * a(n - 2)
        
        Karakteristicna jednadzba:
        x^2 = λ1 * x + λ2
        x^2 - λ1 * x - λ2 = 0

        Rjesenje karakteristicne jednadzbe
        x1 = (λ1 + sqrt(λ1^2 + 4 * λ2)) / 2    
        x2 = (λ1 - sqrt(λ1^2 + 4 * λ2)) / 2

        Uvrstavanje uvjeta:
            1.) Ako x1 =/= x2
                a(n) = C * x1^n + D * x2^n
                a(0) = C + D
                a(1) = Cx + Dx
            2.) Ako x1 == x2
                a(n) = C * x1^n + D * n * x2^n
                a(0) = C
                a(1) = C + D

        Racunanje nepoznanica:
            1.)
                D = (a1 - a0 * x1) / (x2 - x1)
                C = a0 - D
            2.)
                C = a0
                D = a1 - a0
    */

    double C, D, rez, x1, x2;

    x1 = (lamda1 + sqrt(pow(lamda1, 2) + 4 * lamda2)) / 2;
    x2 = (lamda1 - sqrt(pow(lamda1, 2) + 4 * lamda2)) / 2;
    
    if(x1 == x2) {
        C = a0;
        D = a1 - a0;
        rez = C * pow(x1, n) + D * n * pow(x2, n);
    } else {
        D = (a1 - a0 * x1) / (x2 - x1);
        C = a0 - D;
        rez = C * pow(x1, n) + D * pow(x2, n);
    }

    return rez;
}

double izracunRekurzijom(double lamda1, double lamda2, double a0, double a1, int n) {
    /*  
        Rekurzija:
        a(n) = λ1 * a(n - 1) + λ2 * a(n - 2)

        Uz provjeru n za povetnih uvjeta a0 i a1.
    */
   
    if(n == 0) return a0;
    else if(n == 1) return a1;
    else return lamda1 * izracunRekurzijom(lamda1, lamda2, a0, a1, n - 1) + lamda2 * izracunRekurzijom(lamda1, lamda2, a0, a1, n - 2);
}

int main() {
    double lamda1, lamda2, a0, a1;
    int n;

    cout << "Unesite prvi  koeficijent lambda_1 rekurzivne  relacije : ";
    cin >> lamda1;
    cout << "Unesite  drugi  koeficijent  lambda_2  rekurzivne  relacije: "; 
    cin >> lamda2;
    cout << "Unesite  vrijednost  nultog  clana  niza  a_0: ";
    cin >> a0;
    cout << "Unesite  vrijednost  prvog  clana  niza  a_1: ";
    cin >> a1;
    cout << "Unesite  redni  broj  n  trazenog  clana  niza: ";
    cin >> n;


    cout << "Vrijednost  n-tog  clana  niza  pomocu  formule: " << izracunFormulom(lamda1, lamda2, a0, a1, n) << endl; 
    cout << "Vrijednost  n-tog  clana  niza  iz  rekurzije: " << izracunRekurzijom(lamda1, lamda2, a0, a1, n) << endl;

    return 0;
}
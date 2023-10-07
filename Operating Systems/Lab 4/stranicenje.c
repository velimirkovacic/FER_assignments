#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<time.h>
#include <stdbool.h>

#define MAXN 100
#define MAXM 100
#define INDEX_MASK 0b1111000000
#define POMAK_MASK 0b111111
#define BP_MASK 0b100000
#define LRU_MASK 0b11111
#define OKVIR_MASK 0b1111111111000000


char DISK[MAXN][16][64];            // Simulacija diska, svaki od N procesa ima mjesto za svojih 16 64-oktetnih stranica
char OKVIR[MAXM][64];               // Simulacija RAM-a, ima M 64-oktetnih okvira 
short TABLICA[MAXN][16];            // Tablica prevođenja, ima za svaki od N procesa ima 16 redaka za zapise o 16 stranica
int N, M;                           // N - broj procesa, M - broj okvira u RAM-u
int okvirPripada[MAXM];             // Pomoćna tablica koja govori kojem procesu pripada koji okvir (-1 ako ne pripada ni jednom)
int okvirJeStranica[MAXM];          // Pomoćna tablica koja govori koja je stranica u okviru (broj od 0 do 15)


void ispisOKVIR();                  // Sprema trenutno stanje RAM-a u datoteku okvir.txt
void ispisDISK();                   // Sprema trenutno stanje diska u datoteku disk.txt
void ispisTABLICA();                // Sprema trenutno stanje tablice prevođenja u datoteku tablica.txt
void init(char* argv[]);            // Inicijalizira M, N i okvirPripada 
int index_tbl(int log_adr);         // Indeks retka tablice prevođenja iz logičke adrese
int pomak_okvr(int log_adr);        // Pomak unutar stranice/okvira iz logičke adrese
bool BP(int zapis_tbl);             // Bit prisutnosti iz zapisa tablice prevođenja
int LRU(int zapis_tbl);             // LRU zapis iz zapis tablice prevođenja
int index_okvir(int zapis_tbl);     // Indeks okvira iz zapisa tablice prevođenja
int pronadi_okvir();                // Pronalazi/izbacuje okvir putem LRU strategije
void spremi_na_disk(int okvir);                             // Funckija za spremanje sadržaja stranice na disk
void ucitaj_s_diska(int proc, int str, int okvir);          // Funkcija za učitavanje stranice s diska
int novi_zapis(int okvir, int bp, int lru);                     // Stvara novi zapis za tablicu prevođenja
int povecaj_t(int t, int tr_proc, int tr_str, int tr_okvir);    // Povećava vrijednost sata t



int main(int argc, char *argv[])
{
    srand(time(NULL));
    init(argv);
    char t = 0;

    ispisOKVIR();
    ispisDISK();
    ispisTABLICA();
    
    while(true) {
        for(int proc = 0; proc < N; proc++) {
            int log_adr = rand() % 1024;
            int stranica = index_tbl(log_adr);
            int zapis_tbl = TABLICA[proc][stranica];
            int okvir = index_okvir(zapis_tbl);

            printf("---------------------------\n");
            printf("proces:\t%d\n", proc);
            printf("\tt: %d\n", t);
            printf("\tlog. adresa: 0x%04x\n", log_adr);

            if(!BP(zapis_tbl)) {
                printf("\tPromasaj!\n");
                
                okvir = pronadi_okvir();
                spremi_na_disk(okvir);
                ucitaj_s_diska(proc, stranica, okvir);
                okvirPripada[okvir] = proc;
                okvirJeStranica[okvir] = stranica;

            }
            TABLICA[proc][stranica] = novi_zapis(okvir, 1, t);

            printf("\tfiz. adresa: 0x%04x\n", (int) ((index_okvir(TABLICA[proc][stranica]) << 6) + pomak_okvr(log_adr)));
            printf("\tzapis tablice: 0x%04x\n", TABLICA[proc][stranica]);
            printf("\tsadrzaj adrese: %d\n", (int) OKVIR[okvir][pomak_okvr(log_adr)]);

            OKVIR[okvir][pomak_okvr(log_adr)]++;
            t = povecaj_t(t, proc, stranica, okvir);

            ispisTABLICA();
            ispisOKVIR();
            ispisDISK();

            sleep(3);
        }
    }
    return 0;
}


int povecaj_t(int t, int tr_proc, int tr_str, int tr_okvir) {
    if(t + 1 < 31) return t + 1;

    for(int okv = 0; okv < M; okv++) {
        int proc = okvirPripada[okv];
        if(proc == 0) continue;

        int str = okvirJeStranica[okv];

        TABLICA[proc][str] = novi_zapis(okv, 1, 0);
    }
    TABLICA[tr_proc][tr_str] = novi_zapis(tr_okvir, 1, 1);

    return 1;
}


int novi_zapis(int okvir, int bp, int lru) {
    return (lru + (bp << 5) + (okvir << 6));
}


void spremi_na_disk(int okvir)
{
    int proc = okvirPripada[okvir];
    int str = okvirJeStranica[okvir];

    if(proc == -1)
        return;


    for(int i = 0; i < 64; i++) {
        DISK[proc][str][i] = OKVIR[okvir][i];
    }

    TABLICA[proc][str] = 0;
}


void ucitaj_s_diska(int proc, int str, int okvir)
{
    for(int i = 0; i < 64; i++) {
        OKVIR[okvir][i] = DISK[proc][str][i];
    }

}


int pronadi_okvir() 
{
    int okv = 0;
    int minLru = 32;

    for(int i = 0; i < M; i++) {
        int proc = okvirPripada[i];
        int str = okvirJeStranica[i];

        if(proc == -1) {
            printf("\t\tdodijeljen okvir 0x%04x\n", i);
            return i;
        }

        if(LRU(TABLICA[proc][str]) < minLru) {
            okv = i;
            minLru = LRU(TABLICA[proc][str]);
        }
    }

    printf("\t\tIzbacujem stranicu 0x%04x iz procesa %d\n", okvirJeStranica[okv] << 6, okvirPripada[okv]);
    printf("\t\tlru izbacene stranice: 0x%04x\n", minLru);
    printf("\t\tdodijeljen okvir 0x%04x\n", okv);

    return okv;
}


int index_tbl(int log_adr)
{
    return ((log_adr & INDEX_MASK) >> 6);
}


int pomak_okvr(int log_adr)
{  
    return (log_adr & POMAK_MASK);
}


bool BP(int zapis_tbl)
{
    return ((zapis_tbl & BP_MASK) >> 5);
}


int LRU(int zapis_tbl)
{
    return (zapis_tbl & LRU_MASK);
}


int index_okvir(int zapis_tbl)
{
    return ((zapis_tbl & OKVIR_MASK) >> 6);
}


void ispisTABLICA()
{
    FILE *dat = fopen("tablica.txt", "w");

    for(int i = 0; i < N; i++) {
        fprintf(dat, "Proces %d\n", i);
        for(int j = 0; j < 16; j++) {
            fprintf(dat, "0x%04x\n", TABLICA[i][j]);
        }
        fprintf(dat, "\n");
    }
    fclose(dat);
}


void ispisDISK()
{
    FILE *dat = fopen("disk.txt", "w");

    for(int i = 0; i < N; i++) {
        fprintf(dat, "Proces %d\n", i);
        for(int j = 0; j < 16; j++) {
            for(int k = 0; k < 64; k++) {
                fprintf(dat, "0x%02x ", DISK[i][j][k]);
            }
            fprintf(dat, "\n");
        }
        fprintf(dat, "\n");
    }
    fclose(dat);
}


void ispisOKVIR()
{
    FILE *dat = fopen("okvir.txt", "w");

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < 64; j++) {
            fprintf(dat, "0x%02x ", OKVIR[i][j]);
        }
        fprintf(dat, "\n");
    }
    fclose(dat);
}


void init(char* argv[])
{
    sscanf(argv[1], "%d", &N);
    sscanf(argv[2], "%d", &M);
    for(int i = 0; i < M; i++) {
        okvirPripada[i] = -1;
        okvirJeStranica[i] = -1;
    }
    // sve tablice su već 0-irane
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <signal.h>
#include <sys/syscall.h>

#define FOR(n) for(int i = 0; i < n; i++)		// Ponavljanje n puta.
#define LINUX 0
#define MICROSOFT 1
#define IZGL 10


pthread_mutex_t monitor;
pthread_cond_t red[2];
int unutra[2] = {0, 0};			// Koliko od svake vrste programera je trenutno u restoranu.
int ceka[2] = {0, 0};				// Koliko od svake vrste programera čeka ispred restorana.
int pusten[2] = {0, 0};			// Koliko od svake vrste programera je pusteno u restoran otkada je 1. osoba suprotne vrste došla pred restoran.
const char* stanje_restorana = "stanje.txt";



// Funkcija za dohvat identifikatora dretve TID.
pid_t gettid() 
{
	return syscall(__NR_gettid);

}



// Funkcija generira nasumični cijelu broj u intervalu [min, max].
int randNum(int min, int max)
{
	return rand() % (max - min + 1) + min;
}



// Funkcija koja postavlja da proces spava t sekundi.
void spavaj(int t)
{
	FOR(t) sleep(1);
}



// Funkcija koju poziva programer kada želi ući u restoran.
void Ulaz_u_restac(bool vrsta)
{
	pthread_mutex_lock(&monitor);
	
	while(unutra[!vrsta] > 0 || (ceka[!vrsta] > 0 && pusten[vrsta] >= IZGL)) {
		ceka[vrsta]++;
		pthread_cond_wait(&red[vrsta], &monitor);
		ceka[vrsta]--;
	
	}
	
	unutra[vrsta]++;
	pusten[!vrsta] = 0;
	
	if(ceka[!vrsta] > 0)
		pusten[vrsta]++;
	
	pthread_mutex_unlock(&monitor);
}



// Funkcija koju poziva programer kada želi izaći iz restorana.
void Izlaz_iz_restaca(bool vrsta)
{
	pthread_mutex_lock(&monitor);
	unutra[vrsta]--;
	
	if(!unutra[vrsta])
		pthread_cond_broadcast(&red[!vrsta]);
	
	pthread_mutex_unlock(&monitor);
}



// Funkcija koja ispisuje stanje restorana.
void azuriranje_stanja()
{
	FILE *dat = fopen(stanje_restorana, "w");
	fprintf(dat, "U restoranu:\nLINUX: %d\nMICROSOFT: %d\n\n", unutra[true], unutra[false]);
	fprintf(dat, "ČEKA pred restoranom:\nLINUX: %d\nMICROSOFT: %d\n\n", ceka[true], ceka[false]);
	fprintf(dat, "PUŠTENO u restoran otkako je došla 1. osoba suprotne vrste programera:\nLINUX: %d\nMICROSOFT: %d\n", pusten[true], pusten[false]);
	fclose(dat);
}




// Funkcija koju izvode dretve programeri.
void* Programer(void* arg)
{
	bool vrsta = *((bool*) arg);
	srand(time(NULL));
	
	printf("%s programer %ld je pred vratima restorana\n", (vrsta) ? "LINUX" : "MICROSOFT", (long) gettid());
	
	Ulaz_u_restac(vrsta);
	azuriranje_stanja();
	printf("%s programer %ld je ušao u restoran i započinje s jelom\n", (vrsta) ? "LINUX" : "MICROSOFT", (long) gettid());
	
	spavaj(randNum(7, 15));
	printf("%s programer %ld je završio s jelom\n", (vrsta) ? "LINUX" : "MICROSOFT", (long) gettid());
	
	Izlaz_iz_restaca(vrsta);
	azuriranje_stanja();
	printf("%s programer %ld napušta restoran\n", (vrsta) ? "LINUX" : "MICROSOFT", (long) gettid());
}



// Funkcija koja stvara monitor i redove uvjeta.
void stvori() 
{
	pthread_mutex_init(&monitor, NULL);
	pthread_cond_init(&red[0], NULL);
	pthread_cond_init(&red[1], NULL);
}



// Funkcija koja uništava monitor i redove uvjeta.
void unisti()
{
	pthread_mutex_destroy(&monitor);
	pthread_cond_destroy(&red[0]);
	pthread_cond_destroy(&red[1]);

}



// Glavni program.
int main(void)
{
	stvori();
	srand(time(NULL));
	
	
	if(fork() == 0) {
		while(true) {
			spavaj(randNum(1, 5));
			bool vrsta = randNum(0, 1);
			
			pthread_t thr_id;
			pthread_create(&thr_id, NULL, Programer, &vrsta);
		}
	}
	wait(NULL);
	
	return 0;
}

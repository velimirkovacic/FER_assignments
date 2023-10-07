#include<stdio.h>
#include<stdlib.h>

void A();
void B();
void C();
void S();

void A() {
    printf("A");
    char c = getchar();

    if(c == 'a') {
        return;
    } else if(c == 'b') {
        C();
    } else {
        printf("\nNE\n");
        exit(0);
    }
}

void B() {
    printf("B");
    char c = getchar();

    if(c == 'c') {
        c = getchar();
        if(c == 'c') {
            S();
            c = getchar();
            if(c == 'b') {
                c = getchar();
                if(c == 'c') {
                    return;
                } else {
                    printf("\nNE\n");
                    exit(0);
                }
            } else {
                printf("\nNE\n");
                exit(0);
            }
        } else {
            printf("\nNE\n");
            exit(0);
        }
    } else {
        ungetc(c, stdin);
    }
}

void C() {
    printf("C");
    A();
    A();
}

void S() {
    printf("S");
    char c = getchar();

    if(c == 'a') {
        A();
        B();
    } else if(c == 'b') {
        B();
        A();
    } else {
        printf("\nNE\n");
        exit(0);
    }
}


int main(void) {
    S();
    char c = getchar();
    //printf("(%c)", c);
    if(c == EOF || c == '\n') {
        printf("\nDA\n");
    } else {
        printf("\nNE\n");
    }
    return 0;
}
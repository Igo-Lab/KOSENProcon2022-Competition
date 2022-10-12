#include <stdio.h>

int kass(int *a,int b){
    int i;

    for(i=0;i<3;i++){
        *a = 0;
        printf("C側print: %d\n",*a);
        a++;
    }

    for(i=0;i<3;i++){
        a--;
    }
}
#include <stdio.h>

int kass(int **a,int b){
    int i,j;
    int ss = 1;

    for(i=0;i<3;i++){
        for(j=0;j<2;j++){
        printf("C側print(加工前): %d\n",a[i][j]);
        a[i][j] += ss;
        printf("C側print: %d\n",a[i][j]);
        ss++;
    }
    }
}
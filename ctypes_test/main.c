#include <stdio.h>

int kass(int *a,int b){
    char guide[] = "これが計算結果\n";
    int c = 0;
    for(int i=0;i<3;i++){
        c+=*a+b;
        a++;
    }
    printf("%d\n%s",c,guide);
}
#include <stdio.h>

int kass(int **a){
    int i,j;
    int ss = 1;

    for(i=0;i<5;i++){
        for(j=0;j<2;j++){
        printf("C側print(加工前): %d\n",a[i][j]);
        a[i][j] += ss;
        printf("C側print: %d\n",a[i][j]);
        ss++;
    }
    }
}

void resolver(int *problem,int **src, int problem_len, int *src_length, int ** result){
    for(int i=0;i<5;i++){
        printf("problemの中身\n");
        printf("%d\n",problem[i]);
    }

    for(int i=0;i<5;i++){
        for(int j=0;j<5;j++){
        printf("srcの中身\n");
        printf("%d\n",src[i][j]);
        }
    }

    printf("problem_lenの中身\n%d\n",problem_len);
    
    for(int i=0;i<5;i++){
        printf("src_lengthの中身\n");
        printf("%d\n",src_length[i]);
    }

    //resultの値を変えてやるよ～ん
    kass(result);

}
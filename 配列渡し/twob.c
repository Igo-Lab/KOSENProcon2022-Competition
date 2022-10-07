#include <stdio.h>

void data(int **box,int a){
  
    int sam=0;


    for(int i = 0; i < 2; i++) {
		for(int j = 0; j < 3; j++) {

           sam+=box[i][j]; //a=1

           printf("%d\n",sam);
           printf("%d\n",box[i][j]);
           printf("hello\n");
           
        }

    }

  

}

int main(){
    printf("HELLOOOOFUIGFUIG");
    
}
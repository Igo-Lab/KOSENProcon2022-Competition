int data(int *box,int a){
  
int sam;

    for(int i = 0; i < 2; i++) {
		for(int j = 0; j < 2; j++) {

           sam+=*box+a;

           box++;

        }

    }

    printf("%d",sam);

}
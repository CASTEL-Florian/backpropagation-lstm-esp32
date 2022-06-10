#include "Matlib.h"


float sigmoid_function (float input) {
    return 1/(1+(expf(-input))); // 1/(1+exp(-(input)));
}



void showmat(struct Mat* A){
  if(A->row>0&&A->col>0){
    int k=0;
    printf("[");
    for(int i=1;i<=A->row;i++){
      for (int j=1;j<=A->col;j++){
        if(j<A->col){
          printf("%f\t",A->entries[k++]);
        }else{
          printf("%f",A->entries[k++]);
        }
      }
      if(i<A->row){
        printf("\n");
      }else{
        printf("]\n");
      }
    } 
    printf("\n");
  }else{
    printf("[]");
  }
}


struct Mat* newmat(int r,int c,float d){
  struct Mat* M=(struct Mat*)malloc(sizeof(struct Mat));      
  M->row=r;M->col=c;
  M->entries=(float*)malloc(sizeof(float)*r*c);
  int k=0;
  for(int i=1;i<=M->row;i++){
    for(int j=1;j<=M->col;j++){
      M->entries[k++]=d;
    }
  }
  return M;
}

void freemat(struct Mat* A){
  free(A->entries);
  free(A);
}

struct Mat* eye(int n){
  struct Mat* I=newmat(n,n,0);
  for(int i=1;i<=n;i++){
    I->entries[(i-1)*n+i-1]=1;
  }
  return I;
}
struct Mat* zeros(int r,int c){
  struct Mat* Z=newmat(r,c,0);  
  return Z;
}
struct Mat* ones(int r,int c){
  struct Mat* O=newmat(r,c,1);  
  return O;
}

void fillMat(struct Mat* M, float value){
  int k=0;
  for(int i=1;i<=M->row;i++){
    for(int j=1;j<=M->col;j++){
      M->entries[k++]=value;
    }
  }
}

float meanMat(struct Mat* M){
  float m = 0;
  int k=0;
  for(int i=1;i<=M->row;i++){
    for(int j=1;j<=M->col;j++){
      m += M->entries[k++];
    }
  }
  return m/k;
}

struct Mat* randm(int r,int c,float l,float u){
  struct Mat* R=newmat(r,c,1);  
  int k=0;
  for(int i=1;i<=r;i++){
    for(int j=1;j<=c;j++){
      float r=((float)rand())/((float)RAND_MAX);
      R->entries[k++]=l+(u-l)*r;
    }
  }
  return R;
}
float get(struct Mat* M,int r,int c){
  if(r > M->row || c > M->col || r < 1 || c < 1){
    printf("get Index out of bounds : index (%d, %d), matrix of size (%d, %d)\n", r, c, M->row, M-> col);
  }
  float d=M->entries[(r-1)*M->col+c-1];
  return d;
}
void set(struct Mat* M,int r,int c,float d){
  if(r > M->row || c > M->col || r < 1 || c < 1){
    printf("set Index out of bounds : index (%d, %d), matrix of size (%d, %d)\n", r, c, M->row, M-> col);
  }
  M->entries[(r-1)*M->col+c-1]=d;
}

void scalermultiply(struct Mat* M,float c,struct Mat* res){ 
  // res de dimension (M->row,M->col)
  if((M->row != res->row) || (M->col != res->col)){
    printf("scalermultiply Erreur de dimension (%d,%d) (%d,%d)\n", M->row, M->col, res->row, res->col);
  }
  int k=0;
  for(int i=0;i<M->row;i++){
    for(int j=0;j<M->col;j++){
      res->entries[k]=M->entries[k]*c;
      k+=1;
    }
  }
}
void sum(struct Mat* A,struct Mat* B,struct Mat* res){
  // res de dimension (r,c) soit (A->row,A->col)
  if((A->row != B->row) || (A->col != B->col)){
    printf("sum Erreur de dimension (%d,%d) (%d,%d)\n", A->row, A->col, B->row, B->col);
  }
  int r=A->row;
  int c=A->col;
  int k=0;  
  for(int i=0;i<r;i++){
    for(int j=0;j<c;j++){
      res->entries[k]=A->entries[k]+B->entries[k];
      k+=1;
    }
  }
}
void minus(struct Mat* A,struct Mat* B,struct Mat* res){
  // res de dimension (r,c) soit (A->row,A->col)
  if((A->row != B->row) || (A->col != B->col)){
    printf("minus Erreur de dimension (%d,%d) (%d,%d)\n", A->row, A->col, B->row, B->col);
  }
  int r=A->row;
  int c=A->col;
  int k=0;  
  for(int i=0;i<r;i++){
    for(int j=0;j<c;j++){
      res->entries[k]=A->entries[k]-B->entries[k];
      k+=1;
    }
  }
}
void submat(struct Mat* A,int r1,int r2,int c1,int c2, struct Mat* res){
  // sous matrice composée des lignes r1 à r2 et des colonnes c1 à c2 de A.
  // les bornes sont incluses.
  // res de dimension (r2-r1+1,c2-c1+1)
  if((r2 - r1 + 1 != res->row) || (c2 - c1 + 1 != res->col)){
    printf("submat Erreur de dimension (%d,%d) (%d,%d)\n", A->row, A->col, res->row, res->col);
  }
  if (r1<0 || r2 > A->row || c1 < 0 || c1 > A->col){
    printf("submat Index out of bounds : r1 = %d; r2 = %d; c1 = %d; c2 = %d; matrice dim (%d, %d)", r1, r2, c1, c2, A->row, A->col);
  }
  int k=0;
  for(int i=r1;i<=r2;i++){
    for(int j=c1;j<=c2;j++){
      res->entries[k++]=A->entries[(i-1)*A->col+j-1];
    }
  }
}

void multiply(struct Mat* A,struct Mat* B,struct Mat* res){
  // res de dimension (r1,c2) soit (A->row,B->col)
  if((A->row != res->row) || (A->col != B->row) || (B->col != res->col)){
    printf("multiply Erreur de dimension (%d,%d) (%d,%d) (%d,%d)\n", A->row, A->col, B->row, B->col, res->row, res->col);
  }
  int r1=A->row;
  int r2=B->row;
  int c1=A->col;
  int c2=B->col;
  if (r1==1&&c1==1){
    scalermultiply(B,A->entries[0],res);
    return;
  }else if (r2==1&&c2==1){
    scalermultiply(A,B->entries[0],res);
    return;
  }
  for(int i=1;i<=r1;i++){
    for(int j=1;j<=c2;j++){
      float de=0;
      for(int k=1;k<=r2;k++){
        de+=A->entries[(i-1)*A->col+k-1]*B->entries[(k-1)*B->col+j-1];
      }
      res->entries[(i-1)*res->col+j-1]=de;
    }
  }
}
void elementWiseMultiplication(struct Mat* A,struct Mat* B,struct Mat* res){
  // res de dimension (A->row, a->col)
  if((A->row != B->row) || (A->col != B->col)){
    printf("elementWiseMultiplication Erreur de dimension (%d,%d) (%d,%d)\n", A->row, A->col, B->row, B->col);
  }
  int r=A->row;
  int c=A->col;
  int k=0;  
  for(int i=0;i<r;i++){
    for(int j=0;j<c;j++){
      res->entries[k]=A->entries[k]*B->entries[k];
      k+=1;
    }
  }
}
void matrix_sigmoid(struct Mat* A, struct Mat* res){
  // res de dimension (A->row, a->col)
  if((A->row != res->row) || (A->col != res->col)){
    printf("matrix_sigmoid Erreur de dimension (%d,%d) (%d,%d)\n", A->row, A->col, res->row, res->col);
  }
  int r=A->row;
  int c=A->col;
  int k=0;  
  for(int i=0;i<r;i++){
    for(int j=0;j<c;j++){
      res->entries[k]=sigmoid_function(A->entries[k]);
      k+=1;
    }
  }
}
void matrix_tanh(struct Mat* A,struct Mat* res){
  // res de dimension (A->row, a->col)
  if((A->row != res->row) || (A->col != res->col)){
    printf("matrix_tanh Erreur de dimension (%d,%d) (%d,%d)\n", A->row, A->col, res->row, res->col);
  }
  int r=A->row;
  int c=A->col;
  int k=0;  
  for(int i=0;i<r;i++){
    for(int j=0;j<c;j++){
      res->entries[k]=tanhf(A->entries[k]);
      k+=1;
    }
  }
}
void removerow(struct Mat* A,int r, struct Mat* res){
  // res de dimension (A->row-1,A->col)
  if((A->row - 1 != res->row) || (A->col != res->col)){
    printf("removerow Erreur de dimension (%d,%d) (%d,%d)\n", A->row, A->col, res->row, res->col);
  }
  int k=0;
  for(int i=1;i<=A->row;i++){
    for(int j=1;j<=A->col;j++){
      if(i!=r){
        res->entries[k]=A->entries[(i-1)*A->col+j-1];
        k+=1;
      }
    }
  }
}
void removecol(struct Mat* A,int c,struct Mat* res){
  // res de dimension (A->row,A->col-1)
  if((A->row != res->row) || (A->col - 1 != res->col)){
    printf("removecol Erreur de dimension (%d,%d) (%d,%d)\n", A->row, A->col, res->row, res->col);
  }
  int k=0;
  for(int i=1;i<=A->row;i++){
    for(int j=1;j<=A->col;j++){
      if(j!=c){
        res->entries[k]=A->entries[(i-1)*A->col+j-1];
        k+=1;
      }
    }
  }
}
void transpose(struct Mat* A, struct Mat* res){
  // res de dimension(A->col,A->row)
  if((A->row != res->col) || (A->col != res->row)){
    printf("transpose Erreur de dimension (%d,%d) (%d,%d)\n", A->row, A->col, res->row, res->col);
  }
  int k=0;
  for(int i=1;i<=A->col;i++){
    for(int j=1;j<=A->row;j++){
      res->entries[k]=A->entries[(j-1)*A->row+i-1];
      k+=1;
    }
  }
}

void copyvalue(struct Mat* A,struct Mat* res){
  if((A->row != res->row) || (A->col != res->col)){
    printf("copyvalue Erreur de dimension (%d,%d) (%d,%d)\n", A->row, A->col, res->row, res->col);
  }
  // res de dimension (A->row,A->col)
  int k=0;
  for(int i=1;i<=A->row;i++){
    for(int j=1;j<=A->col;j++){
      res->entries[k]=A->entries[k];
      k++;
    }
  }
}

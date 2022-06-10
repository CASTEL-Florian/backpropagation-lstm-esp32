#ifndef MATLIB_H
#define MATLIB_H

#include "stdlib.h"
#include <stdio.h>
#include "Math.h"


float sigmoid_function (float input);


struct Mat{
  float* entries;
  int row;
  int col;
};


void showmat(struct Mat* A);


struct Mat* newmat(int r,int c,float d);

void freemat(struct Mat* A);

struct Mat* eye(int n);
struct Mat* zeros(int r,int c);
struct Mat* ones(int r,int c);

void fillMat(struct Mat* M, float value);

float meanMat(struct Mat* M);

struct Mat* randm(int r,int c,float l,float u);
float get(struct Mat* M,int r,int c);
void set(struct Mat* M,int r,int c,float d);

void scalermultiply(struct Mat* M,float c,struct Mat* res);
void sum(struct Mat* A,struct Mat* B,struct Mat* res);
void minus(struct Mat* A,struct Mat* B,struct Mat* res);
void submat(struct Mat* A,int r1,int r2,int c1,int c2, struct Mat* res);

void multiply(struct Mat* A,struct Mat* B,struct Mat* res);
void elementWiseMultiplication(struct Mat* A,struct Mat* B,struct Mat* res);
void matrix_sigmoid(struct Mat* A, struct Mat* res);
void matrix_tanh(struct Mat* A,struct Mat* res);
void removerow(struct Mat* A,int r, struct Mat* res);
void removecol(struct Mat* A,int c,struct Mat* res);
void transpose(struct Mat* A, struct Mat* res);

void copyvalue(struct Mat* A,struct Mat* res);

#endif
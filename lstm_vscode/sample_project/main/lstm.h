#ifndef LSTM_H
#define LSTM_H
#include "MatLib.h"


#define H 6
#define INPUT_DIM 1
#define SEQ_LEN 3
#define LEARNING_RATE 0.1
#define OUTPUT_DIM 1
#define TRAIN_LEN TIME_SERIES_LEN - SEQ_LEN
#define TIME_SERIES_LEN 1439


// forward step
struct Mat* a;
struct Mat* prod_h_Wh;
struct Mat* next_c;
struct Mat* next_h;
struct Mat* ig;
struct Mat* tanh_next_c;
struct step_forward* step_forward;

// forward
struct cache* cacheList[SEQ_LEN];
struct Mat* prev_c;
struct Mat* x_cur;
struct forward* forward;

// backward step
struct Mat* d1;
struct Mat* one_matrix_H;
struct Mat* dop;
struct Mat* dprev_c;
struct Mat* dfp;
struct Mat* dip;
struct Mat* dgp;
struct Mat* do_;
struct Mat* df;
struct Mat* di;
struct Mat* dg;
struct Mat* da;
struct Mat* db_step;
struct Mat* WxT;
struct Mat* WhT;
struct Mat* dx;
struct Mat* dprev_h;
struct Mat* xT;
struct Mat* dWx_step;
struct Mat* prev_hT;
struct Mat* dWh_step;
struct step_backward* step_backward;

// backward
struct Mat* dWx;
struct Mat* dh_prev;
struct Mat* dc_prev;
struct Mat* dWh;
struct Mat* db;

// predict
struct Mat* prev_h;
struct Mat* prediction;

// train
struct Mat* dh_states;
struct Mat* Wh;
struct Mat* Wx;
struct Mat* b;
struct Mat* Why;
struct Mat* dWhy;
struct Mat* by;
struct Mat* dby;
struct Mat* WhyT;
struct Mat* scores;
struct Mat* dscores;
struct Mat* dscores2;
struct Mat* temp;
struct Mat* one_matrix_O;
struct backward* backward;

struct cache{
  struct Mat* x;
    struct Mat* prev_c; 
    struct Mat* Wx; 
    struct Mat* Wh; 
    struct Mat* i; 
    struct Mat* f; 
    struct Mat* o; 
    struct Mat* g; 
    struct Mat* next_c;
};

struct cacheList{
  struct cache* cache;
  struct cacheList* next;
};



struct forward{
    struct Mat* h;
};

struct step_backward{
    struct Mat* dx; 
    struct Mat* dprev_h; 
    struct Mat* dprev_c; 
    struct Mat* dWx;
    struct Mat* dWh; 
    struct Mat* db;
};

struct backward{
    struct Mat* dWx; 
    struct Mat* dWh; 
    struct Mat* db;
};

void lstm_step_forward(struct Mat* x, struct Mat* prev_h, struct Mat* prev_c, struct Mat* Wx, struct Mat* Wh, struct Mat* b, int step_number);

struct forward* lstm_forward(struct Mat* x, struct Mat* prev_h, struct Mat* Wx, struct Mat* Wh, struct Mat* b);

struct step_backward* lstm_step_backward(struct Mat* dnext_h, struct Mat* dnext_c, struct cache* cache);

struct backward* lstm_backward(struct Mat* dh, struct cache* cacheList[], int seq_len);

struct Mat* predict(struct Mat* x, struct Mat* Wx, struct Mat* Wh, struct Mat* b, struct Mat* Why, struct Mat* by);

float* train(struct Mat* inputs[], struct Mat* targets[], int input_dim, int hidden_dim, int output_dim, int train_len, int seq_len, float learning_rate, int n_epochs);

#endif
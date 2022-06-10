#include "stdlib.h"
#include <stdio.h>
#include "Math.h"
#include "lstm.h"
#include <time.h>

void free_cache_mat(struct cache* c){
    freemat(c->next_c);
    freemat(c->prev_c);
    freemat(c->f);
    freemat(c->g);
    freemat(c->i);
    freemat(c->o);
    freemat(c->Wh);
    freemat(c->Wx);
    freemat(c->x);
}




void lstm_step_forward(struct Mat* x, struct Mat* prev_h, struct Mat* prev_c, struct Mat* Wx, struct Mat* Wh, struct Mat* b, int step_number){
    /*
    Forward pass for a single timestep of an LSTM.
    The input data has dimension D, the hidden state has dimension H.
    Inputs:
    - x: Input data, of shape (1, input_dim)
    - prev_h: Previous hidden state, of shape (1, hidden_dim)
    - prev_c: previous cell state, of shape (1, hidden_dim)
    - Wx: Input-to-hidden weights, of shape (input_dim, 4*hidden_dim)
    - Wh: Hidden-to-hidden weights, of shape (hidden_dim, 4*hidden_dim)
    - b: Biases, of shape (4*hidden_dim)
    - step_number: identifier of the step
    Modifies the values of:
    - next_h: Next hidden state, of shape (1, H)
    - next_c: Next cell state, of shape (1, H)
    - cacheList: values needed for backward pass.
    */
    struct Mat* i = newmat(1, H, 0);
    struct Mat* f = newmat(1, H, 0);
    struct Mat* o = newmat(1, H, 0);
    struct Mat* g = newmat(1, H, 0);
    multiply(prev_h, Wh, prod_h_Wh);
    multiply(x,Wx,a);

    sum(prod_h_Wh, a, a); 
    sum(a,b,a);
    submat(a,1,1,1,H,i);
    matrix_sigmoid(i,i);
    submat(a,1,1,H+1,2*H,f);
    matrix_sigmoid(f,f);
    submat(a,1,1,2*H+1,3*H,o);
    matrix_sigmoid(o,o);
    submat(a,1,1,3*H+1,4*H,g);

    elementWiseMultiplication(i, g, ig);
    elementWiseMultiplication(f, prev_c, next_c);
    sum(next_c, ig, next_c);


    matrix_tanh(next_c,tanh_next_c);
    elementWiseMultiplication(o,tanh_next_c,next_h);
   
    struct cache* cache = cacheList[step_number];
    cache->x = newmat(1,INPUT_DIM,0);
    cache->prev_c= newmat(1,H,0);
    cache->Wh = newmat(H,4*H,0);
    cache->Wx = newmat(INPUT_DIM,4*H,0);
    cache->next_c = newmat(1,H,0);
    copyvalue(x, cache->x);
    copyvalue(prev_c, cache->prev_c);
    copyvalue(next_c, cache->next_c);
    copyvalue(Wh, cache->Wh);
    copyvalue(Wx, cache->Wx);
    cache->i = i;
    cache->f = f;
    cache->o = o;
    cache->g = g;

    
}

struct forward* lstm_forward(struct Mat* x, struct Mat* prev_h, struct Mat* Wx, struct Mat* Wh, struct Mat* b){
    /*
    Inputs:
    - x: Input data of shape (seq_length, input_dim)
    - h0: Initial hidden state of shape (1, hidden_dim)
    - Wx: Weights for input-to-hidden connections, of shape (input_dim, 4*hidden_dim)
    - Wh: Weights for hidden-to-hidden connections, of shape (hidden_dim, 4*hidden_dim)
    - b: Biases of shape (4*hidden_dim)
    Returns:
    - h: Hidden states at the output of the LSTM, of shape (1, hidden_dim)
    */
    fillMat(next_c, 0);
    for (int i = 0; i < x->row;i++){     // 0 to seq_length-1
        submat(x,i+1, i+1, 1, x->col, x_cur);
        lstm_step_forward(x_cur, prev_h, next_c, Wx, Wh, b, i);
        copyvalue(next_h, prev_h);
    }

    forward->h = prev_h;
    return forward;
}

struct step_backward* lstm_step_backward(struct Mat* dnext_h, struct Mat* dnext_c, struct cache* cache){
    /*
    Backward pass for a single timestep of an LSTM.
    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (1, H)
    - dnext_c: Gradients of next cell state, of shape (1, H)
    - cache: Values from the forward pass
    Returns:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (1, H)
    - dprev_c: Gradient of previous cell state, of shape (1, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    */
    matrix_tanh(cache->next_c, dop);
    elementWiseMultiplication(dop,dop,d1);
    minus(one_matrix_H,d1,d1);
    elementWiseMultiplication(cache->o,d1,d1);
    elementWiseMultiplication(d1,dnext_h,d1);
    sum(d1,dnext_c,d1);
    
    elementWiseMultiplication(cache->f, d1, dprev_c);

    elementWiseMultiplication(dop, dnext_h, dop);

 
    elementWiseMultiplication(cache->prev_c, d1, dfp);
    elementWiseMultiplication(cache->g, d1, dip);
    elementWiseMultiplication(cache->i, d1, dgp);
    minus(one_matrix_H, cache->o, do_);
    elementWiseMultiplication(cache->o, do_, do_);
    elementWiseMultiplication(do_,dop,do_);
    minus(one_matrix_H, cache->f, df);
    elementWiseMultiplication(cache->f, df, df);
    elementWiseMultiplication(df,dfp,df);
    minus(one_matrix_H, cache->i, di);
    elementWiseMultiplication(cache->i, di, di);
    elementWiseMultiplication(di,dip,di);
    elementWiseMultiplication(cache->g, cache->g,dg);
    minus(one_matrix_H,dg,dg);
    elementWiseMultiplication(dg,dgp,dg);
    for (int i = 1; i<=H; i++){
        set(da,1, i, get(di,1,i));
        set(da,1, i+H, get(df,1,i));
        set(da,1, i+2*H, get(do_,1,i));
        set(da,1, i+3*H, get(dg,1,i));
    }
    
    copyvalue(da,db_step);
    transpose(cache->Wx,WxT);
    transpose(cache->Wh,WhT);
    transpose(cache->x,xT);
    multiply(da,WxT,dx);
    multiply(da,WhT,dprev_h);
    multiply(xT,da,dWx_step);
    transpose(dprev_h,prev_hT);
    multiply(prev_hT,da,dWh_step);

    step_backward->dx = dx;
    step_backward->dprev_h = dprev_h;
    step_backward->dprev_c = dprev_c;
    step_backward->dWx = dWx_step;
    step_backward->dWh=dWh_step;
    step_backward->db=db_step;
    
    return step_backward;
}

struct backward* lstm_backward(struct Mat* dh, struct cache* cacheList[], int seq_len){
    /*
    Backward pass for an LSTM over an entire sequence of data.
    Inputs:
    - dh: Upstream gradients of hidden states, of shape (seq_length, hidden_dim)
    - cacheList: Values from the forward pass
    Returns:
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    */
    fillMat(dprev_c, 0);
    for (int i = 0; i<seq_len; i++){
        struct cache* cur_cache = cacheList[seq_len - i - 1];
        struct step_backward* step;
        if (i==0){
            step = lstm_step_backward(dh, dprev_c, cur_cache);
        }else{
            step = lstm_step_backward(dprev_h, dprev_c, cur_cache);
        }
        free_cache_mat(cur_cache);
 
        sum(dWx,step->dWx,dWx);
        sum(dWh,step->dWh,dWh);
        sum(db,step->db,db);
        
    }
    
    backward->dWx = dWx;
    backward->dWh = dWh;
    backward->db = db;
    
    return backward;
}


struct Mat* predict(struct Mat* x, struct Mat* Wx, struct Mat* Wh, struct Mat* b, struct Mat* Why, struct Mat* by){
    /*
    Compute the output of the neural network.
    Inputs:
        - x: Input data
        - seq_length : sequence lenght
        - Wx: LSTM Input-to-hidden weights, of shape (input_dim, 4*hidden_dim)
        - Wh: LSTM Hidden-to-hidden weights, of shape (hidden_dim, 4*hidden_dim)
        - b: LSTM Biases, of shape (4*hidden_dim)
        - Why : Dense weights
        - by : Dense biases
        Returns:
        - prediction : output of the dense layer
    */
    fillMat(prev_h, 0);
    forward = lstm_forward(x, prev_h, Wx, Wh, b);
    
    transpose(Why,WhyT);
    multiply(forward->h, WhyT, prediction);
    sum(prediction,by,prediction);
    matrix_sigmoid(prediction,prediction);
    for(int i = 0; i<SEQ_LEN; i++){
        free_cache_mat(cacheList[i]);
    }
    return prediction;
}

float* train(struct Mat* inputs[], struct Mat* targets[], int input_dim, int hidden_dim, int output_dim, int train_len, int seq_len, float learning_rate, int n_epochs){
    float* epoch_loss_list = (float*)malloc(n_epochs * sizeof(float));
    for (int i = 0; i < n_epochs; i++){
        printf("epoch %d\n", i);
        float epoch_loss = 0;
        for (int k = 0; k < train_len;k++){

            // Feed-forward
            fillMat(prev_h, 0);
            forward = lstm_forward(inputs[k], prev_h, Wx, Wh, b);                        
            transpose(Why,WhyT);
            multiply(forward->h, WhyT, scores);
            sum(scores,by,scores);
            matrix_sigmoid(scores,scores);
            
            // Calculation of the gradient
            minus(scores,targets[k],dscores);
            elementWiseMultiplication(dscores,dscores,dscores2);
            epoch_loss += meanMat(dscores2);
            minus(one_matrix_O,scores,dby);
            elementWiseMultiplication(scores,dby,dby);
            elementWiseMultiplication(dscores,dby,dby);
            multiply(dby, forward->h, dWhy);
            multiply(dscores, Why, dh_states);
            backward = lstm_backward(dh_states, cacheList, seq_len);

            // Gradient update 
            scalermultiply(backward->dWx,learning_rate,backward->dWx);
            scalermultiply(backward->dWh,learning_rate,backward->dWh);
            scalermultiply(backward->db,learning_rate,backward->db);
            scalermultiply(dWhy,learning_rate,dWhy);
            scalermultiply(dby,learning_rate,dby);
            minus(Wx, backward->dWx, Wx);
            minus(Wh, backward->dWh, Wh);
            minus(b, backward->db, b);
            minus(by, dby, by);
            minus(Why, dWhy, Why);
        }
        epoch_loss_list[i] = epoch_loss/train_len;
        
    }
    return epoch_loss_list;
}
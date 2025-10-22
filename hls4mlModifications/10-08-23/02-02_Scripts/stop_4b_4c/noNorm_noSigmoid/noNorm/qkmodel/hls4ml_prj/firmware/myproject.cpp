#include <iostream>

#include "myproject.h"
#include "parameters.h"

namespace hls4ml_model_emu_v3 {

void myproject(
    input_t input_1[10*14],
    result_t layer13_out[1]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_1,layer13_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer3_t layer3_out[10*10];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0

    layer5_t layer5_out[10*10];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    layer6_t layer6_out[10*10];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    layer7_t layer7_out[10*10];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0

    layer8_t layer8_out[10*10];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0

    layer9_t layer9_out[10];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0

    layer10_t layer10_out[10];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0

    layer12_t layer12_out[10];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0

    nnet::pointwise_conv_1d_cl<input_t, layer3_t, config15>(input_1, layer3_out, w3, b3); // q_conv1d

    nnet::relu<layer3_t, layer5_t, relu_config5>(layer3_out, layer5_out); // q_activation

    nnet::pointwise_conv_1d_cl<layer5_t, layer6_t, config16>(layer5_out, layer6_out, w6, b6); // q_conv1d_1

    nnet::linear<layer6_t, layer7_t, linear_config7>(layer6_out, layer7_out); // q_conv1d_1_linear

    nnet::relu<layer7_t, layer8_t, relu_config8>(layer7_out, layer8_out); // q_activation_1

    nnet::global_pooling1d_cl<layer8_t, layer9_t, config9>(layer8_out, layer9_out); // global_average_pooling1d

    nnet::dense<layer9_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10); // q_dense

    nnet::relu<layer10_t, layer12_t, relu_config12>(layer10_out, layer12_out); // q_activation_2

    nnet::dense<layer12_t, result_t, config13>(layer12_out, layer13_out, w13, b13); // q_dense_1

}

}

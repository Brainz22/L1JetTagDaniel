#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_conv1d.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"
#include "nnet_utils/nnet_sepconv1d_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/w15.h"
#include "weights/b15.h"
#include "weights/w16.h"
#include "weights/b16.h"
#include "weights/w9.h"
#include "weights/b9.h"
#include "weights/w12.h"
#include "weights/b12.h"

// hls-fpga-machine-learning insert layer-config
// q_conv1d
struct config15_mult : nnet::dense_config {
    static const unsigned n_in = 14;
    static const unsigned n_out = 10;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 105;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv1d_default_t accum_t;
    typedef bias15_t bias_t;
    typedef weight15_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config15 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 10;
    static const unsigned n_chan = 14;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 10;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 10;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 105;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::pointwise;
    static const unsigned min_width = 10;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 10;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_15<data_T, CONFIG_T>;
    typedef q_conv1d_default_t accum_t;
    typedef bias15_t bias_t;
    typedef weight15_t weight_t;
    typedef config15_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_unscaled<K, S, W>;
};
const ap_uint<config15::filt_width> config15::pixels[] = {0};

// q_activation
struct relu_config4 : nnet::activ_config {
    static const unsigned n_in = 100;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef q_activation_table_t table_t;
};

// q_conv1d_1
struct config16_mult : nnet::dense_config {
    static const unsigned n_in = 10;
    static const unsigned n_out = 10;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 75;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef q_conv1d_1_default_t accum_t;
    typedef bias16_t bias_t;
    typedef weight16_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config16 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 10;
    static const unsigned n_chan = 10;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 10;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 10;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 75;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::pointwise;
    static const unsigned min_width = 10;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 10;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_16<data_T, CONFIG_T>;
    typedef q_conv1d_1_default_t accum_t;
    typedef bias16_t bias_t;
    typedef weight16_t weight_t;
    typedef config16_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_unscaled<K, S, W>;
};
const ap_uint<config16::filt_width> config16::pixels[] = {0};

// q_activation_1
struct relu_config7 : nnet::activ_config {
    static const unsigned n_in = 100;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef q_activation_1_table_t table_t;
};

// global_average_pooling1d
struct config8 : nnet::pooling1d_config {
    static const unsigned n_in = 10;
    static const unsigned n_filt = 10;
    static const nnet::Pool_Op pool_op = nnet::Average;
    static const unsigned reuse_factor = 1;
    typedef model_default_t accum_t;
};

// q_dense
struct config9 : nnet::dense_config {
    static const unsigned n_in = 10;
    static const unsigned n_out = 10;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 93;
    static const unsigned n_nonzeros = 7;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias9_t bias_t;
    typedef weight9_t weight_t;
    typedef layer9_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_activation_2
struct relu_config11 : nnet::activ_config {
    static const unsigned n_in = 10;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef q_activation_2_table_t table_t;
};

// q_dense_1
struct config12 : nnet::dense_config {
    static const unsigned n_in = 10;
    static const unsigned n_out = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 8;
    static const unsigned n_nonzeros = 2;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias12_t bias_t;
    typedef weight12_t weight_t;
    typedef layer12_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// sigmoid
struct sigmoid_config14 : nnet::activ_config {
    static const unsigned n_in = 1;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef sigmoid_table_t table_t;
};


#endif

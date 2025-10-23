#ifndef TOOLLIP_V3_H_
#define TOOLLIP_V3_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

namespace hls4ml_model_emu_v3 {

// Prototype of top level function for C-synthesis
void TOoLLiP_v3(
    input_t input_1[10*14],
    result_t layer13_out[1]
);

// hls-fpga-machine-learning insert emulator-defines

}

#endif

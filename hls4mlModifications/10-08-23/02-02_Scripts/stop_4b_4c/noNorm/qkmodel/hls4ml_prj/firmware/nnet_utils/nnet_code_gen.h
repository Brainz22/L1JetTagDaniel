#ifndef NNET_INSTR_GEN_H_
#define NNET_INSTR_GEN_H_

#include "nnet_helpers.h"
#include <iostream>

namespace nnet {

template <class data_T, typename CONFIG_T> class FillConv1DBuffer {
  public:
    static void fill_buffer(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                            data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_width * CONFIG_T::n_chan],
                            const unsigned partition) {
        // To be implemented in subclasses
    }
};

template <class data_T, typename CONFIG_T> class FillConv2DBuffer {
  public:
    static void
    fill_buffer(data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
                data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
                const unsigned partition) {
        // To be implemented in subclasses
    }
};

// hls4ml insert code
template<class data_T, typename CONFIG_T>
class fill_buffer_15 : public FillConv1DBuffer<data_T, CONFIG_T> {
    public:
    static void fill_buffer(
        data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
        data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_width * CONFIG_T::n_chan],
        const unsigned partition
    ) {
        if (partition ==   0) {
            buffer[0][0] =    data[0]; buffer[0][1] =    data[1]; buffer[0][2] =    data[2]; buffer[0][3] =    data[3]; buffer[0][4] =    data[4]; buffer[0][5] =    data[5]; buffer[0][6] =    data[6]; buffer[0][7] =    data[7]; buffer[0][8] =    data[8]; buffer[0][9] =    data[9]; buffer[0][10] =   data[10]; buffer[0][11] =   data[11]; buffer[0][12] =   data[12]; buffer[0][13] =   data[13];

        }
        if (partition ==   1) {
            buffer[0][0] =   data[14]; buffer[0][1] =   data[15]; buffer[0][2] =   data[16]; buffer[0][3] =   data[17]; buffer[0][4] =   data[18]; buffer[0][5] =   data[19]; buffer[0][6] =   data[20]; buffer[0][7] =   data[21]; buffer[0][8] =   data[22]; buffer[0][9] =   data[23]; buffer[0][10] =   data[24]; buffer[0][11] =   data[25]; buffer[0][12] =   data[26]; buffer[0][13] =   data[27];

        }
        if (partition ==   2) {
            buffer[0][0] =   data[28]; buffer[0][1] =   data[29]; buffer[0][2] =   data[30]; buffer[0][3] =   data[31]; buffer[0][4] =   data[32]; buffer[0][5] =   data[33]; buffer[0][6] =   data[34]; buffer[0][7] =   data[35]; buffer[0][8] =   data[36]; buffer[0][9] =   data[37]; buffer[0][10] =   data[38]; buffer[0][11] =   data[39]; buffer[0][12] =   data[40]; buffer[0][13] =   data[41];

        }
        if (partition ==   3) {
            buffer[0][0] =   data[42]; buffer[0][1] =   data[43]; buffer[0][2] =   data[44]; buffer[0][3] =   data[45]; buffer[0][4] =   data[46]; buffer[0][5] =   data[47]; buffer[0][6] =   data[48]; buffer[0][7] =   data[49]; buffer[0][8] =   data[50]; buffer[0][9] =   data[51]; buffer[0][10] =   data[52]; buffer[0][11] =   data[53]; buffer[0][12] =   data[54]; buffer[0][13] =   data[55];

        }
        if (partition ==   4) {
            buffer[0][0] =   data[56]; buffer[0][1] =   data[57]; buffer[0][2] =   data[58]; buffer[0][3] =   data[59]; buffer[0][4] =   data[60]; buffer[0][5] =   data[61]; buffer[0][6] =   data[62]; buffer[0][7] =   data[63]; buffer[0][8] =   data[64]; buffer[0][9] =   data[65]; buffer[0][10] =   data[66]; buffer[0][11] =   data[67]; buffer[0][12] =   data[68]; buffer[0][13] =   data[69];

        }
        if (partition ==   5) {
            buffer[0][0] =   data[70]; buffer[0][1] =   data[71]; buffer[0][2] =   data[72]; buffer[0][3] =   data[73]; buffer[0][4] =   data[74]; buffer[0][5] =   data[75]; buffer[0][6] =   data[76]; buffer[0][7] =   data[77]; buffer[0][8] =   data[78]; buffer[0][9] =   data[79]; buffer[0][10] =   data[80]; buffer[0][11] =   data[81]; buffer[0][12] =   data[82]; buffer[0][13] =   data[83];

        }
        if (partition ==   6) {
            buffer[0][0] =   data[84]; buffer[0][1] =   data[85]; buffer[0][2] =   data[86]; buffer[0][3] =   data[87]; buffer[0][4] =   data[88]; buffer[0][5] =   data[89]; buffer[0][6] =   data[90]; buffer[0][7] =   data[91]; buffer[0][8] =   data[92]; buffer[0][9] =   data[93]; buffer[0][10] =   data[94]; buffer[0][11] =   data[95]; buffer[0][12] =   data[96]; buffer[0][13] =   data[97];

        }
        if (partition ==   7) {
            buffer[0][0] =   data[98]; buffer[0][1] =   data[99]; buffer[0][2] =  data[100]; buffer[0][3] =  data[101]; buffer[0][4] =  data[102]; buffer[0][5] =  data[103]; buffer[0][6] =  data[104]; buffer[0][7] =  data[105]; buffer[0][8] =  data[106]; buffer[0][9] =  data[107]; buffer[0][10] =  data[108]; buffer[0][11] =  data[109]; buffer[0][12] =  data[110]; buffer[0][13] =  data[111];

        }
        if (partition ==   8) {
            buffer[0][0] =  data[112]; buffer[0][1] =  data[113]; buffer[0][2] =  data[114]; buffer[0][3] =  data[115]; buffer[0][4] =  data[116]; buffer[0][5] =  data[117]; buffer[0][6] =  data[118]; buffer[0][7] =  data[119]; buffer[0][8] =  data[120]; buffer[0][9] =  data[121]; buffer[0][10] =  data[122]; buffer[0][11] =  data[123]; buffer[0][12] =  data[124]; buffer[0][13] =  data[125];

        }
        if (partition ==   9) {
            buffer[0][0] =  data[126]; buffer[0][1] =  data[127]; buffer[0][2] =  data[128]; buffer[0][3] =  data[129]; buffer[0][4] =  data[130]; buffer[0][5] =  data[131]; buffer[0][6] =  data[132]; buffer[0][7] =  data[133]; buffer[0][8] =  data[134]; buffer[0][9] =  data[135]; buffer[0][10] =  data[136]; buffer[0][11] =  data[137]; buffer[0][12] =  data[138]; buffer[0][13] =  data[139];

        }
    }
};
template<class data_T, typename CONFIG_T>
class fill_buffer_16 : public FillConv1DBuffer<data_T, CONFIG_T> {
    public:
    static void fill_buffer(
        data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
        data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_width * CONFIG_T::n_chan],
        const unsigned partition
    ) {
        if (partition ==   0) {
            buffer[0][0] =    data[0]; buffer[0][1] =    data[1]; buffer[0][2] =    data[2]; buffer[0][3] =    data[3]; buffer[0][4] =    data[4]; buffer[0][5] =    data[5]; buffer[0][6] =    data[6]; buffer[0][7] =    data[7]; buffer[0][8] =    data[8]; buffer[0][9] =    data[9];

        }
        if (partition ==   1) {
            buffer[0][0] =   data[10]; buffer[0][1] =   data[11]; buffer[0][2] =   data[12]; buffer[0][3] =   data[13]; buffer[0][4] =   data[14]; buffer[0][5] =   data[15]; buffer[0][6] =   data[16]; buffer[0][7] =   data[17]; buffer[0][8] =   data[18]; buffer[0][9] =   data[19];

        }
        if (partition ==   2) {
            buffer[0][0] =   data[20]; buffer[0][1] =   data[21]; buffer[0][2] =   data[22]; buffer[0][3] =   data[23]; buffer[0][4] =   data[24]; buffer[0][5] =   data[25]; buffer[0][6] =   data[26]; buffer[0][7] =   data[27]; buffer[0][8] =   data[28]; buffer[0][9] =   data[29];

        }
        if (partition ==   3) {
            buffer[0][0] =   data[30]; buffer[0][1] =   data[31]; buffer[0][2] =   data[32]; buffer[0][3] =   data[33]; buffer[0][4] =   data[34]; buffer[0][5] =   data[35]; buffer[0][6] =   data[36]; buffer[0][7] =   data[37]; buffer[0][8] =   data[38]; buffer[0][9] =   data[39];

        }
        if (partition ==   4) {
            buffer[0][0] =   data[40]; buffer[0][1] =   data[41]; buffer[0][2] =   data[42]; buffer[0][3] =   data[43]; buffer[0][4] =   data[44]; buffer[0][5] =   data[45]; buffer[0][6] =   data[46]; buffer[0][7] =   data[47]; buffer[0][8] =   data[48]; buffer[0][9] =   data[49];

        }
        if (partition ==   5) {
            buffer[0][0] =   data[50]; buffer[0][1] =   data[51]; buffer[0][2] =   data[52]; buffer[0][3] =   data[53]; buffer[0][4] =   data[54]; buffer[0][5] =   data[55]; buffer[0][6] =   data[56]; buffer[0][7] =   data[57]; buffer[0][8] =   data[58]; buffer[0][9] =   data[59];

        }
        if (partition ==   6) {
            buffer[0][0] =   data[60]; buffer[0][1] =   data[61]; buffer[0][2] =   data[62]; buffer[0][3] =   data[63]; buffer[0][4] =   data[64]; buffer[0][5] =   data[65]; buffer[0][6] =   data[66]; buffer[0][7] =   data[67]; buffer[0][8] =   data[68]; buffer[0][9] =   data[69];

        }
        if (partition ==   7) {
            buffer[0][0] =   data[70]; buffer[0][1] =   data[71]; buffer[0][2] =   data[72]; buffer[0][3] =   data[73]; buffer[0][4] =   data[74]; buffer[0][5] =   data[75]; buffer[0][6] =   data[76]; buffer[0][7] =   data[77]; buffer[0][8] =   data[78]; buffer[0][9] =   data[79];

        }
        if (partition ==   8) {
            buffer[0][0] =   data[80]; buffer[0][1] =   data[81]; buffer[0][2] =   data[82]; buffer[0][3] =   data[83]; buffer[0][4] =   data[84]; buffer[0][5] =   data[85]; buffer[0][6] =   data[86]; buffer[0][7] =   data[87]; buffer[0][8] =   data[88]; buffer[0][9] =   data[89];

        }
        if (partition ==   9) {
            buffer[0][0] =   data[90]; buffer[0][1] =   data[91]; buffer[0][2] =   data[92]; buffer[0][3] =   data[93]; buffer[0][4] =   data[94]; buffer[0][5] =   data[95]; buffer[0][6] =   data[96]; buffer[0][7] =   data[97]; buffer[0][8] =   data[98]; buffer[0][9] =   data[99];

        }
    }
};

} // namespace nnet

#endif

//Numpy array shape [10]
//Min -1.250000000000
//Max 0.625000000000
//Number of zeros 2

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[10];
#else
bias16_t b16[10] = {-0.21875, 0.62500, -0.50000, 0.18750, -0.15625, -1.25000, -0.15625, 0.00000, 0.00000, -0.31250};
#endif

#endif

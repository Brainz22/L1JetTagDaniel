//Numpy array shape [10]
//Min -0.312500000000
//Max 0.312500000000
//Number of zeros 3

#ifndef B3_H_
#define B3_H_

#ifndef __SYNTHESIS__
bias3_t b3[10];
#else
bias3_t b3[10] = {0.03125, 0.00000, 0.03125, 0.00000, -0.15625, -0.03125, 0.00000, 0.31250, -0.31250, 0.12500};

#endif

#endif

//Numpy array shape [10]
//Min -1.125000000000
//Max 0.593750000000
//Number of zeros 6

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[10];
#else
bias16_t b16[10] = {-0.09375, -1.12500, 0.59375, 0.00000, -1.12500, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000};
#endif

#endif

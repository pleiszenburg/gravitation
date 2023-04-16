/* -*- coding: utf-8 -*- */

/*

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/_libcc2/lib.c: C single-thread core, SIMD

	Copyright (C) 2019-2023 Sebastian M. Ernst <ernst@pleiszenburg.de>

<LICENSE_BLOCK>
The contents of this file are subject to the GNU General Public License
Version 2 ("GPL" or "License"). You may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt
https://github.com/pleiszenburg/gravitation/blob/master/LICENSE

Software distributed under the License is distributed on an "AS IS" basis,
WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License for the
specific language governing rights and limitations under the License.
</LICENSE_BLOCK>

*/

#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CDTYPE double
#define VLEN 4

typedef struct univ {

    CDTYPE *x, *y, *z;
    CDTYPE *ax, *ay, *az;
    CDTYPE *m;

    CDTYPE g;

    size_t n;

} univ;

static CDTYPE inline *_aligned_alloc(size_t n) {

    return (CDTYPE*)aligned_alloc(32, n * sizeof(CDTYPE));

}

void univ_alloc(univ *self)
{

    self->x = _aligned_alloc(self->n);
    self->y = _aligned_alloc(self->n);
    self->z = _aligned_alloc(self->n);

    self->ax = _aligned_alloc(self->n);
    self->ay = _aligned_alloc(self->n);
    self->az = _aligned_alloc(self->n);

    self->m = _aligned_alloc(self->n);

}

void univ_free(univ *self)
{

    free(self->x);
    free(self->y);
    free(self->z);

    free(self->ax);
    free(self->ay);
    free(self->az);

    free(self->m);

}

static __m256d inline _load_256d(size_t diff, size_t i, CDTYPE *d){

    if (diff == 4) {
        return _mm256_load_pd(&d[i]);
    }
    if (diff == 3) {
        return _mm256_set_pd(0.0, d[i+2], d[i+1], d[i]);
    }
    if (diff == 2) {
        return _mm256_set_pd(0.0, 0.0, d[i+1], d[i]);
    }
    return _mm256_set_pd(0.0, 0.0, 0.0, d[i]);

}

static double inline _sum_256d(__m256d d, size_t diff){

    if (diff == 4) {
        return d[0] + d[1] + d[2] + d[3];
    }
    if (diff == 3) {
        return d[0] + d[1] + d[2];
    }
    if (diff == 2) {
        return d[0] + d[1];
    }
    return d[0];
}

static void inline _sub_256d(
    __m256d x, __m256d y, __m256d z,
    size_t i, size_t diff,
    CDTYPE *dx, CDTYPE *dy, CDTYPE *dz
) {

    size_t l;

    for (size_t k = 0; k < diff; k ++) {
        l = i + k;
        dx[l] -= x[k];
        dy[l] -= y[k];
        dz[l] -= z[k];
    }

}

static void inline _print_256d(__m256d d){
    printf(" AVX: %e, %e, %e, %e \n", d[0], d[1], d[2], d[3]);
}
static void inline _print_a4(CDTYPE *d, size_t i){
    printf(" MEM: %e, %e, %e, %e \n", d[i], d[i+1], d[i+2], d[i+3]);
}
// static void assert_nan(CDTYPE x, CDTYPE y, CDTYPE z) {
//     if (isnan(x) || isnan(y) || isnan(z)) {
//         exit(1);
//     }
// }

static void inline _univ_update_pair(univ *self, size_t i, size_t j, __m256d g)
{

    // printf("\n");

    size_t diff = j - i;
    if (diff > VLEN) {
        diff = VLEN;
    }

    // printf(" i == %ld | j == %ld | diff == %ld \n", i, j, diff);

    __m256d xi = _load_256d(diff, i, self->x);
    __m256d yi = _load_256d(diff, i, self->y);
    __m256d zi = _load_256d(diff, i, self->z);
    __m256d mi = _load_256d(diff, i, self->m);

    __m256d xj = _mm256_set1_pd(self->x[j]);
    __m256d yj = _mm256_set1_pd(self->y[j]);
    __m256d zj = _mm256_set1_pd(self->z[j]);
    __m256d mj = _mm256_set1_pd(self->m[j]);

    // printf(" data \n");
    // _print_256d(xi);
    // _print_256d(yi);
    // _print_256d(zi);
    // _print_256d(xj);
    // _print_256d(yj);
    // _print_256d(zj);

    __m256d dx = _mm256_sub_pd(xi, xj);
    __m256d dy = _mm256_sub_pd(yi, yj);
    __m256d dz = _mm256_sub_pd(zi, zj);

    // printf(" diff \n");
    // _print_256d(dx);
    // _print_256d(dy);
    // _print_256d(dz);

    __m256d dxx = _mm256_mul_pd(dx,  dx);
    __m256d dyy = _mm256_mul_pd(dy,  dy);
    __m256d dzz = _mm256_mul_pd(dz,  dz);

    // printf("TT %ld \n", i);
    // _print_256d(g);

    // printf(" diff^2 \n");
    // _print_256d(dxx);
    // _print_256d(dyy);
    // _print_256d(dzz);

    __m256d dxyz = _mm256_add_pd(_mm256_add_pd(dxx, dyy), dzz);

    // printf(" dxyz \n");
    // _print_256d(dxyz);

    __m256d dxyzg = _mm256_div_pd(g, dxyz);

    // printf(" dxyzg \n");
    // _print_256d(dxyzg);

    __m256d aj = _mm256_mul_pd(dxyzg, mi);
    __m256d ai = _mm256_mul_pd(dxyzg, mj);

    // printf(" ai \n");
    // _print_256d(ai);
    // printf(" aj \n");
    // _print_256d(aj);

    // printf(" dxyz \n");
    // _print_256d(dxyz);

    dxyz = _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(dxyz));

    // printf(" rsqrt(dxyz) \n");
    // _print_256d(dxyz);

    dx = _mm256_mul_pd(dxyz, dx);
    dy = _mm256_mul_pd(dxyz, dy);
    dz = _mm256_mul_pd(dxyz, dz);

    // _print_256d(dx);
    // _print_256d(dy);
    // _print_256d(dz);

    // printf(" aj[v]: %e, %e, %e \n", self->ax[j], self->ay[j], self->az[j]);

    self->ax[j] += _sum_256d(_mm256_mul_pd(aj, dx), diff);
    self->ay[j] += _sum_256d(_mm256_mul_pd(aj, dy), diff);
    self->az[j] += _sum_256d(_mm256_mul_pd(aj, dz), diff);

    // printf(" aj[n]: %e, %e, %e \n", self->ax[j], self->ay[j], self->az[j]);
    // assert_nan(self->ax[j], self->ay[j], self->az[j]);

    dx = _mm256_mul_pd(ai, dx);
    dy = _mm256_mul_pd(ai, dy);
    dz = _mm256_mul_pd(ai, dz);

    size_t l;
    for (size_t k = 0; k < diff; k++) {
        l = i + k;
        // printf(" ai[v]: %e, %e, %e (%ld, %ld) \n", self->ax[l], self->ay[l], self->az[l], i, k);
        self->ax[l] -= dx[k];
        self->ay[l] -= dy[k];
        self->az[l] -= dz[k];
        // printf(" ai[n]: %e, %e, %e (%ld, %ld) \n", self->ax[l], self->ay[l], self->az[l], i, k);
        // assert_nan(self->ax[l], self->ay[l], self->az[l]);
    }

}

void univ_step_stage1(univ *self)
{

    size_t n_mem = sizeof(CDTYPE)*self->n;

    memset(self->ax, 0, n_mem);
    memset(self->ay, 0, n_mem);
    memset(self->az, 0, n_mem);

    __m256d g = _mm256_set1_pd(self->g);

    // printf(" n == %ld \n", self->n);

    for(size_t j = 1; j < self->n; j++){

        for(size_t i = 0; i < j; i += VLEN){

            _univ_update_pair(self, i, j, g);

        }

    }

}

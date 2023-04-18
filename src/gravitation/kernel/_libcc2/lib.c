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

    CDTYPE *rx, *ry, *rz;
    CDTYPE *ax, *ay, *az;
    CDTYPE *m;

    CDTYPE g;

    size_t n;

} univ;

typedef struct __m256d_3d {

    __m256d x, y, z;

} v3;

static CDTYPE inline *_aligned_alloc(size_t n) {

    return (CDTYPE*)aligned_alloc(32, n * sizeof(CDTYPE));

}

void univ_alloc(univ *self)
{

    self->rx = _aligned_alloc(self->n);
    self->ry = _aligned_alloc(self->n);
    self->rz = _aligned_alloc(self->n);

    self->ax = _aligned_alloc(self->n);
    self->ay = _aligned_alloc(self->n);
    self->az = _aligned_alloc(self->n);

    self->m = _aligned_alloc(self->n);

}

void univ_free(univ *self)
{

    free(self->rx);
    free(self->ry);
    free(self->rz);

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

static __m256d inline _mask_256d(__m256d d, size_t diff){

    if (diff == 4) {
        return d;
    }
    if (diff <= 3) {
        d[3] = 0.0;
    }
    if (diff <= 2) {
        d[2] = 0.0;
    }
    if (diff == 1) {
        d[1] = 0.0;
    }
    return d;
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

    for (size_t k = 0; k < diff; k++) {
        l = i + k;
        dx[l] -= x[k];
        dy[l] -= y[k];
        dz[l] -= z[k];
    }

}

static struct __m256d_3d inline _univ_update_pair(
    univ *self, size_t i, size_t j,
    __m256d rxi, __m256d ryi, __m256d rzi,
    __m256d axi, __m256d ayi, __m256d azi,
    __m256d mi, __m256d g
)
{

    size_t j_diff = j - i;
    if (j_diff > VLEN) {
        j_diff = VLEN;
    }

    __m256d rxj = _mm256_set1_pd(self->rx[j]);
    __m256d ryj = _mm256_set1_pd(self->ry[j]);
    __m256d rzj = _mm256_set1_pd(self->rz[j]);
    __m256d mj = _mm256_set1_pd(self->m[j]);

    __m256d dx = _mm256_sub_pd(rxi, rxj);
    __m256d dy = _mm256_sub_pd(ryi, ryj);
    __m256d dz = _mm256_sub_pd(rzi, rzj);

    __m256d dxx = _mm256_mul_pd(dx,  dx);
    __m256d dyy = _mm256_mul_pd(dy,  dy);
    __m256d dzz = _mm256_mul_pd(dz,  dz);

    __m256d dxyz = _mm256_add_pd(_mm256_add_pd(dxx, dyy), dzz);

    __m256d dxyzg = _mm256_div_pd(g, dxyz);

    __m256d aj = _mm256_mul_pd(dxyzg, mi);
    __m256d ai = _mm256_mul_pd(dxyzg, mj);

    dxyz = _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(dxyz));

    dx = _mm256_mul_pd(dxyz, dx);
    dy = _mm256_mul_pd(dxyz, dy);
    dz = _mm256_mul_pd(dxyz, dz);

    self->ax[j] += _sum_256d(_mm256_mul_pd(aj, dx), j_diff);
    self->ay[j] += _sum_256d(_mm256_mul_pd(aj, dy), j_diff);
    self->az[j] += _sum_256d(_mm256_mul_pd(aj, dz), j_diff);

    axi = _mm256_add_pd(axi, _mask_256d(_mm256_mul_pd(ai, dx), j_diff));
    ayi = _mm256_add_pd(ayi, _mask_256d(_mm256_mul_pd(ai, dy), j_diff));
    azi = _mm256_add_pd(azi, _mask_256d(_mm256_mul_pd(ai, dz), j_diff));

    return (struct __m256d_3d){axi, ayi, azi};

}

void univ_step_stage1(univ *self)
{

    size_t n_mem = sizeof(CDTYPE)*self->n;

    memset(self->ax, 0, n_mem);
    memset(self->ay, 0, n_mem);
    memset(self->az, 0, n_mem);

    size_t i, j, n_diff;

    __m256d rxi, ryi, rzi;
    __m256d axi, ayi, azi;
    __m256d mi;

    __m256d g = _mm256_set1_pd(self->g);

    struct __m256d_3d ai;

    for(i = 0; i < self->n - 1; i += VLEN){

        n_diff = self->n - i;
        if (n_diff > VLEN) {
            n_diff = VLEN;
        }

        rxi = _load_256d(n_diff, i, self->rx);
        ryi = _load_256d(n_diff, i, self->ry);
        rzi = _load_256d(n_diff, i, self->rz);
        mi = _load_256d(n_diff, i, self->m);

        axi = _mm256_set1_pd(0.0);
        ayi = _mm256_set1_pd(0.0);
        azi = _mm256_set1_pd(0.0);

        for(j = i + 1; j < self->n; j++){

            ai = _univ_update_pair(
                self, i, j,
                rxi, ryi, rzi,
                axi, ayi, azi,
                mi, g
            );

            axi = ai.x;
            ayi = ai.y;
            azi = ai.z;

        }

        _sub_256d(
            axi, ayi, azi,
            i, n_diff,
            self->ax, self->ay, self->az
        );

    }

}

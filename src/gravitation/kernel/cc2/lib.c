/* -*- coding: utf-8 -*- */

/*

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/_libcc2/lib.c: C single-thread core

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

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct univ_f4 {

    float *rx, *ry, *rz;
    float *ax, *ay, *az;
    float *m;

    float g;

    size_t n;

} univ_f4;

typedef struct univ_f8 {

    double *rx, *ry, *rz;
    double *ax, *ay, *az;
    double *m;

    double g;

    size_t n;

} univ_f8;

static float inline *_aligned_alloc_f4(size_t n) {

    return (float*)aligned_alloc(32, n * sizeof(float));

}

static double inline *_aligned_alloc_f8(size_t n) {

    return (double*)aligned_alloc(32, n * sizeof(double));

}

void univ_alloc_f4(univ_f4 *self)
{

    self->rx = _aligned_alloc_f4(self->n);
    self->ry = _aligned_alloc_f4(self->n);
    self->rz = _aligned_alloc_f4(self->n);

    self->ax = _aligned_alloc_f4(self->n);
    self->ay = _aligned_alloc_f4(self->n);
    self->az = _aligned_alloc_f4(self->n);

    self->m = _aligned_alloc_f4(self->n);

}

void univ_alloc_f8(univ_f8 *self)
{

    self->rx = _aligned_alloc_f8(self->n);
    self->ry = _aligned_alloc_f8(self->n);
    self->rz = _aligned_alloc_f8(self->n);

    self->ax = _aligned_alloc_f8(self->n);
    self->ay = _aligned_alloc_f8(self->n);
    self->az = _aligned_alloc_f8(self->n);

    self->m = _aligned_alloc_f8(self->n);

}

void univ_free_f4(univ_f4 *self)
{

    free(self->rx);
    free(self->ry);
    free(self->rz);

    free(self->ax);
    free(self->ay);
    free(self->az);

    free(self->m);

}

void univ_free_f8(univ_f8 *self)
{

    free(self->rx);
    free(self->ry);
    free(self->rz);

    free(self->ax);
    free(self->ay);
    free(self->az);

    free(self->m);

}

static void inline _univ_update_pair_f4(univ_f4 *self, size_t i, size_t j)
{

    float dx = self->rx[i] - self->rx[j];
    float dy = self->ry[i] - self->ry[j];
    float dz = self->rz[i] - self->rz[j];

    float dxx = dx * dx;
    float dyy = dy * dy;
    float dzz = dz * dz;

    float dxyz = dxx + dyy + dzz;

    float dxyzg = self->g / dxyz;

    float aj = dxyzg * self->m[i];
    float ai = dxyzg * self->m[j];

    dxyz = (float)1.0 / (float)sqrt(dxyz);

    dx *= dxyz;
    dy *= dxyz;
    dz *= dxyz;

    self->ax[j] += aj * dx;
    self->ay[j] += aj * dy;
    self->az[j] += aj * dz;

    self->ax[i] -= ai * dx;
    self->ay[i] -= ai * dy;
    self->az[i] -= ai * dz;

}

static void inline _univ_update_pair_f8(univ_f8 *self, size_t i, size_t j)
{

    double dx = self->rx[i] - self->rx[j];
    double dy = self->ry[i] - self->ry[j];
    double dz = self->rz[i] - self->rz[j];

    double dxx = dx * dx;
    double dyy = dy * dy;
    double dzz = dz * dz;

    double dxyz = dxx + dyy + dzz;

    double dxyzg = self->g / dxyz;

    double aj = dxyzg * self->m[i];
    double ai = dxyzg * self->m[j];

    dxyz = (double)1.0 / (double)sqrt(dxyz);

    dx *= dxyz;
    dy *= dxyz;
    dz *= dxyz;

    self->ax[j] += aj * dx;
    self->ay[j] += aj * dy;
    self->az[j] += aj * dz;

    self->ax[i] -= ai * dx;
    self->ay[i] -= ai * dy;
    self->az[i] -= ai * dz;

}

void univ_iterate_stage1_f4(univ_f4 *self)
{

    size_t n_mem = sizeof(float)*self->n;

    memset(self->ax, 0, n_mem);
    memset(self->ay, 0, n_mem);
    memset(self->az, 0, n_mem);

    for(size_t i = 0; i < self->n - 1; i++){
        for(size_t j = i + 1; j < self->n; j++){
            _univ_update_pair_f4(self, i, j);
        }
    }

}

void univ_iterate_stage1_f8(univ_f8 *self)
{

    size_t n_mem = sizeof(double)*self->n;

    memset(self->ax, 0, n_mem);
    memset(self->ay, 0, n_mem);
    memset(self->az, 0, n_mem);

    for(size_t i = 0; i < self->n - 1; i++){
        for(size_t j = i + 1; j < self->n; j++){
            _univ_update_pair_f8(self, i, j);
        }
    }

}

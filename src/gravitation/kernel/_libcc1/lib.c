/* -*- coding: utf-8 -*- */

/*

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/_libcc1/lib.c: C single-thread core

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

#define CDTYPE double

typedef struct univ {

    CDTYPE *rx, *ry, *rz;
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

static void inline _univ_update_pair(univ *self, size_t i, size_t j)
{

    CDTYPE dx = self->rx[i] - self->rx[j];
    CDTYPE dy = self->ry[i] - self->ry[j];
    CDTYPE dz = self->rz[i] - self->rz[j];

    CDTYPE dxx = dx * dx;
    CDTYPE dyy = dy * dy;
    CDTYPE dzz = dz * dz;

    CDTYPE dxyz = dxx + dyy + dzz;

    CDTYPE dxyzg = self->g / dxyz;

    CDTYPE aj = dxyzg * self->m[i];
    CDTYPE ai = dxyzg * self->m[j];

    dxyz = (CDTYPE)1.0 / (CDTYPE)sqrt(dxyz);

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

void univ_step_stage1(univ *self)
{

    size_t n_mem = sizeof(CDTYPE)*self->n;

    memset(self->ax, 0, n_mem);
    memset(self->ay, 0, n_mem);
    memset(self->az, 0, n_mem);

    for(size_t i = 0; i < self->n - 1; i++){
        for(size_t j = i + 1; j < self->n; j++){
            _univ_update_pair(self, i, j);
        }
    }

}

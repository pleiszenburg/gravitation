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

typedef struct mass_f4 {

    float rx, ry, rz;
    float ax, ay, az;
    float m;

} mass_f4;

typedef struct mass_f8 {

    double rx, ry, rz;
    double ax, ay, az;
    double m;

} mass_f8;

void univ_alloc_f4(mass_f4 **masses, size_t n)
{

    *masses = (mass_f4*)aligned_alloc(32, n * sizeof(mass_f4));

}

void univ_alloc_f8(mass_f8 **masses, size_t n)
{

    *masses = (mass_f8*)aligned_alloc(32, n * sizeof(mass_f8));

}

void univ_free_f4(mass_f4 **masses)
{

    free(*masses);

}

void univ_free_f8(mass_f8 **masses)
{

    free(*masses);

}

static void inline _univ_update_pair_f4(mass_f4 *pm1, mass_f4 *pm2, float g)
{

    float dx = pm1->rx - pm2->rx;
    float dy = pm1->ry - pm2->ry;
    float dz = pm1->rz - pm2->rz;

    float dxx = dx * dx;
    float dyy = dy * dy;
    float dzz = dz * dz;

    float dxyz = dxx + dyy + dzz;

    float dxyzg = g / dxyz;

    float aj = dxyzg * pm1->m;
    float ai = dxyzg * pm2->m;

    dxyz = (float)1.0 / (float)sqrt(dxyz);

    dx *= dxyz;
    dy *= dxyz;
    dz *= dxyz;

    pm2->ax += aj * dx;
    pm2->ay += aj * dy;
    pm2->az += aj * dz;

    pm1->ax -= ai * dx;
    pm1->ay -= ai * dy;
    pm1->az -= ai * dz;

}

static void inline _univ_update_pair_f8(mass_f8 *pm1, mass_f8 *pm2, double g)
{

    double dx = pm1->rx - pm2->rx;
    double dy = pm1->ry - pm2->ry;
    double dz = pm1->rz - pm2->rz;

    double dxx = dx * dx;
    double dyy = dy * dy;
    double dzz = dz * dz;

    double dxyz = dxx + dyy + dzz;

    double dxyzg = g / dxyz;

    double aj = dxyzg * pm1->m;
    double ai = dxyzg * pm2->m;

    dxyz = (double)1.0 / (double)sqrt(dxyz);

    dx *= dxyz;
    dy *= dxyz;
    dz *= dxyz;

    pm2->ax += aj * dx;
    pm2->ay += aj * dy;
    pm2->az += aj * dz;

    pm1->ax -= ai * dx;
    pm1->ay -= ai * dy;
    pm1->az -= ai * dz;

}

void univ_step_stage1_f4(mass_f4 *masses, float g, size_t n)
{

    for(size_t i = 0; i < n; i++) {
        masses[i].ax = (float)0.0;
        masses[i].ay = (float)0.0;
        masses[i].az = (float)0.0;
    }

    for(size_t i = 0; i < n - 1; i++){
        for(size_t j = i + 1; j < n; j++){
            _univ_update_pair_f4(&masses[i], &masses[j], g);
        }
    }

}

void univ_step_stage1_f8(mass_f8 *masses, double g, size_t n)
{

    for(size_t i = 0; i < n; i++) {
        masses[i].ax = (double)0.0;
        masses[i].ay = (double)0.0;
        masses[i].az = (double)0.0;
    }

    for(size_t i = 0; i < n - 1; i++){
        for(size_t j = i + 1; j < n; j++){
            _univ_update_pair_f8(&masses[i], &masses[j], g);
        }
    }

}

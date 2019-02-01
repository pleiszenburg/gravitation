/* -*- coding: utf-8 -*- */

/*

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/_lib1_/lib.c: C single-thread core

	Copyright (C) 2019 Sebastian M. Ernst <ernst@pleiszenburg.de>

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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Header für erweiterte Integer-Datentypen
#include <stdint.h>

#define UNIVERSUM_DATATYPE float
#define COUNTER_DATATYPE long

struct univ {

	// Gravitation INIT
	UNIVERSUM_DATATYPE *X, *Y, *Z;
	UNIVERSUM_DATATYPE *AX, *AY, *AZ;
	UNIVERSUM_DATATYPE *M;

	UNIVERSUM_DATATYPE G;

	// Anzahl der Massen
	COUNTER_DATATYPE N;

};


void step_stage1(struct univ *self)
{

	// Iteration und Segmentierung
	COUNTER_DATATYPE i, j, k, seg_len;

	// Vektor für Abstand
	UNIVERSUM_DATATYPE dx, dy, dz;
	// Vektor für normalisierten Abstand
	UNIVERSUM_DATATYPE dnx, dny, dnz;
	// Hilfsvariblen für Betrag
	UNIVERSUM_DATATYPE dxx, dyy, dzz, dxyz, dxyzs;
	// Gravitation Hilfsvarible
	UNIVERSUM_DATATYPE PHY_Gdxyz;
	// Beschleunigung Hilfsvarible
	UNIVERSUM_DATATYPE Ai, Aj;

	// Segmentierung steuern
	i = 1;
	j = 0;
	seg_len = ((*self).N * ((*self).N - 1)) / 2;

	// STUFE 1: Beschleunigug
	for(k = 0; k < seg_len; k++)
	{

		// Abstand der beiden Punkte vektoriell berechnen
		dx = (*self).X[i] - (*self).X[j];
		dy = (*self).Y[i] - (*self).Y[j];
		dz = (*self).Z[i] - (*self).Z[j];

		// Quadrate ausrechnen
		dxx = dx * dx;
		dyy = dy * dy;
		dzz = dz * dz;

		// Quadrate summieren (Quadrat des Betrags des Vektors)
		dxyz = dxx + dyy + dzz;

		// Gravitationskonstante durch Quadrats des Betrags des Vektors
		PHY_Gdxyz = (*self).G / dxyz;

		// Betrag der Beschleunigung(en) ausrechnen
		Aj = PHY_Gdxyz * (*self).M[i];
		Ai = PHY_Gdxyz * (*self).M[j];

		// Wurzel ziehen um Betrag zu bekommen
		dxyzs = (UNIVERSUM_DATATYPE)1.0 / (UNIVERSUM_DATATYPE)sqrt(dxyz);

		// Abstand normalisieren
		dnx = dx * dxyzs;
		dny = dy * dxyzs;
		dnz = dz * dxyzs;

		// Beschleunigung vektoriell ausrechnen (j)
		(*self).AX[j] = (*self).AX[j] + Aj * dnx;
		(*self).AY[j] = (*self).AY[j] + Aj * dny;
		(*self).AZ[j] = (*self).AZ[j] + Aj * dnz;

		// Beschleunigung vektoriell ausrechnen (i)
		(*self).AX[i] = (*self).AX[i] - Ai * dnx;
		(*self).AY[i] = (*self).AY[i] - Ai * dny;
		(*self).AZ[i] = (*self).AZ[i] - Ai * dnz;

		// Segmentierung steuern
		j++;
		if(j == i)
		{
			i++;
			j = 0;
		}

	}

}

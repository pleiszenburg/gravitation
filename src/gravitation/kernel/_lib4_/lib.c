/* -*- coding: utf-8 -*- */

/*

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/_lib4_/lib.c: C SSE2 SIMD multi-thread core

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

// Header für SSE(1)
#include <xmmintrin.h>

// Header für openMP
#include <omp.h>

// Wie viele Operationen gleichzeitig?
#define SSEI_OP 4

// Datentyp für SSE
#define UNIVERSUM_DATATYPE_SSE __m128

struct univ {

	// Gravitation INIT
	UNIVERSUM_DATATYPE *X, *Y, *Z;
	UNIVERSUM_DATATYPE *AX, *AY, *AZ;
	UNIVERSUM_DATATYPE *M;

	UNIVERSUM_DATATYPE G;

	// Anzahl der Massen
	COUNTER_DATATYPE N;

	UNIVERSUM_DATATYPE *AXmp, *AYmp, *AZmp;
	COUNTER_DATATYPE *j_min, *j_max;
	COUNTER_DATATYPE seg_len, OPENMP_threadsmax;

};


// Vektor aus vier Floats wird um ein Element nach links verschoben. Am Ende wird eine Null aufgefüllt. Eine Asm-Instruktion
static inline UNIVERSUM_DATATYPE_SSE SSEI_m128shift(UNIVERSUM_DATATYPE_SSE p)
{
	return (UNIVERSUM_DATATYPE_SSE)_mm_srli_si128(_mm_castps_si128(p), 4);
}


void step_stage1_segmentation(struct univ *self)
{

	COUNTER_DATATYPE speicher, n, iter, j, z, zs;

	(*self).OPENMP_threadsmax = omp_get_max_threads();
	(*self).seg_len = ((*self).N * ((*self).N - 1)) / 2;

	// Größe der Vektoren für Sprungmarken berechnen
	speicher = (COUNTER_DATATYPE)sizeof(COUNTER_DATATYPE) * (COUNTER_DATATYPE)(*self).OPENMP_threadsmax;

	// Speicher allozieren
	(*self).j_min = (COUNTER_DATATYPE *)calloc(speicher, 1);
	(*self).j_max = (COUNTER_DATATYPE *)calloc(speicher, 1);

	// Paare je Thread
	iter = (COUNTER_DATATYPE)floor((float)(*self).seg_len / (float)(*self).OPENMP_threadsmax);

	// Threads zählen
	n = 0;
	// Paare je Thread zählen
	z = 0;
	// Alle Paare zählen
	zs = 0;
	// Erster Startpunkt
	(*self).j_min[0] = 0;

	// Sprunkpunkte ausrechnen
	for(j = 0; j < (*self).N; j += SSEI_OP)
	{

		// Paare entsprechend SSEI_OP Schleifendurchläufen addieren
		z = z + (SSEI_OP * ((*self).N - 1)) - (SSEI_OP * j) + ((SSEI_OP * (SSEI_OP - 1)) / 2);

		// Genug Paare für den Thread?
		if(z >= iter)
		{

			// Obergrenze des letzten Threads speichern
			(*self).j_max[n] = j;

			// Thread hochzählen
			n++;

			// Untergrenze des nächsten Threads speichern
			(*self).j_min[n] = j;

			// Zähler für alle Paare erhöhen
			zs += z;

			// Zähler auf Null
			z = 0;

			// Letzter Thread?
			if(n + 1 == (*self).OPENMP_threadsmax)
			{

				// Letzte Obergrenze
				(*self).j_max[n] = (*self).N;

				break;

			}

		}

	}

}


void step_stage1_calc(struct univ *self)
{

	// Iteration und Segmentierung
	COUNTER_DATATYPE i, j, j_f, i_f, i_g, f, m, tn;

	// Vektor für Abstand
	UNIVERSUM_DATATYPE_SSE dx, dy, dz;
	// Vektor für normalisierten Abstand
	UNIVERSUM_DATATYPE_SSE dnx, dny, dnz;
	// Hilfsvariblen für Betrag
	UNIVERSUM_DATATYPE_SSE dxx, dyy, dzz, dxyz, dxyzs;
	// Gravitation Hilfsvarible
	UNIVERSUM_DATATYPE_SSE PHY_Gdxyz;
	// Beschleunigung Hilfsvarible
	UNIVERSUM_DATATYPE_SSE Ai, Aj;

	// Vektoren für SSEI: X, Y, Z
	UNIVERSUM_DATATYPE_SSE Xi, Yi, Zi, Xj, Yj, Zj;
	// Vektoren für SSEI: AX, AY, AZ
	UNIVERSUM_DATATYPE_SSE AXi, AYi, AZi, AXj, AYj, AZj;
	// Vektoren für SSEI: M
	UNIVERSUM_DATATYPE_SSE Mi, Mj;

	// Parallelisierung
	COUNTER_DATATYPE m_i;

	// Index für letztes Element der Vektoren
	const COUNTER_DATATYPE I_E = SSEI_OP - 1;

	const UNIVERSUM_DATATYPE_SSE PHY_G_SSE = {(*self).G, (*self).G, (*self).G, (*self).G};

	#pragma omp parallel \
		default(none) \
		private(m,tn,m_i,i,j,j_f,i_f,i_g,f,dx,dy,dz,dxx,dyy,dzz,dxyz,dxyzs,dnx,dny,dnz,PHY_Gdxyz,Ai,Aj,Xi,Yi,Zi,Xj,Yj,Zj,AXi,AYi,AZi,AXj,AYj,AZj,Mi,Mj) \
		shared(self)
	{

		// Thread-Nummer
		tn = (COUNTER_DATATYPE)omp_get_thread_num();

		// Segmentierung basierend auf Thread-Index
		m = tn * (*self).N;

		// ZEILEN (j)
		for(j = (*self).j_min[tn]; j < (*self).j_max[tn]; j += SSEI_OP)
		{

			// Initialisierung eiens Durchlaufes der äußeren Schleife (Zeilen). Nimmt an, dass N durch SSEI_OP teilbar ist.
			for(f = 0; f < SSEI_OP; f++)
			{

				// j + f ergibt Index im Speicher für j-Vektoren, j_f
				j_f = j + f;

				// Position und Masse für in j-Vektor schreiben (Vektor mit vier diagonalen initialisieren)
				Xj[f] = (*self).X[j_f];
				Yj[f] = (*self).Y[j_f];
				Zj[f] = (*self).Z[j_f];
				Mj[f] = (*self).M[j_f];

				// j-Beschleunigungen aus Null setzen
				AXj[f] = (UNIVERSUM_DATATYPE)0.0;
				AYj[f] = (UNIVERSUM_DATATYPE)0.0;
				AZj[f] = (UNIVERSUM_DATATYPE)0.0;

				// j_f + 1 ergibt Index im Speicher für i-Vektoren, i_f
				i_f = j_f + 1;

				// Wenn N durch vier teilbar ist, entsteht hier IMMER ein leeres Element am Ende, was abgefangen werden muss
				if(i_f < (*self).N)
				{

					// Position und Masse in i-Vektor schreiben
					Xi[f] = (*self).X[i_f];
					Yi[f] = (*self).Y[i_f];
					Zi[f] = (*self).Z[i_f];
					Mi[f] = (*self).M[i_f];

				} else {

					// Position und Masse in i-Vektor schreiben
					Xi[f] = (UNIVERSUM_DATATYPE)0.0;
					Yi[f] = (UNIVERSUM_DATATYPE)0.0;
					Zi[f] = (UNIVERSUM_DATATYPE)0.0;
					Mi[f] = (UNIVERSUM_DATATYPE)0.0;

				}

				// i-Beschleunigungen aus Null setzen
				AXi[f] = (UNIVERSUM_DATATYPE)0.0;
				AYi[f] = (UNIVERSUM_DATATYPE)0.0;
				AZi[f] = (UNIVERSUM_DATATYPE)0.0;

			}

			// SPALTEN (i)
			for(i = j + 1; i < (*self).N; i++)
			{

				// Abstand der beiden Punkte vektoriell berechnen
				dx = _mm_sub_ps(Xi, Xj);
				dy = _mm_sub_ps(Yi, Yj);
				dz = _mm_sub_ps(Zi, Zj);

				// Quadrate ausrechnen
				dxx = _mm_mul_ps(dx, dx);
				dyy = _mm_mul_ps(dy, dy);
				dzz = _mm_mul_ps(dz, dz);

				// Quadrate summieren (Quadrat des Betrags des Vektors)
				dxyz = _mm_add_ps(dxx, _mm_add_ps(dyy, dzz));

				// Gravitationskonstante durch Quadrats des Betrags des Vektors
				PHY_Gdxyz = _mm_div_ps(PHY_G_SSE, dxyz);

				// Betrag der Beschleunigung(en) ausrechnen
				Aj = _mm_mul_ps(PHY_Gdxyz, Mi);
				Ai = _mm_mul_ps(PHY_Gdxyz, Mj);

				// Wurzel ziehen um Betrag zu bekommen
				dxyzs = _mm_rsqrt_ps(dxyz);

				// Abstand normalisieren
				dnx = _mm_mul_ps(dx, dxyzs);
				dny = _mm_mul_ps(dy, dxyzs);
				dnz = _mm_mul_ps(dz, dxyzs);

				// Beschleunigung vektoriell ausrechnen (j)
				AXj = _mm_add_ps(AXj, _mm_mul_ps(Aj, dnx));
				AYj = _mm_add_ps(AYj, _mm_mul_ps(Aj, dny));
				AZj = _mm_add_ps(AZj, _mm_mul_ps(Aj, dnz));

				// Beschleunigung vektoriell ausrechnen (i)
				AXi = _mm_add_ps(AXi, _mm_mul_ps(Ai, dnx));
				AYi = _mm_add_ps(AYi, _mm_mul_ps(Ai, dny));
				AZi = _mm_add_ps(AZi, _mm_mul_ps(Ai, dnz));

				// Zieladdresse für i-Vektor
				m_i = i + m;

				// Vor Shift Beschleunigug aus erstem Element des i-Vektors holen
				(*self).AXmp[m_i] -= AXi[0];
				(*self).AYmp[m_i] -= AYi[0];
				(*self).AZmp[m_i] -= AZi[0];

				// i-Vektoren verschlieben
				Xi = SSEI_m128shift(Xi);
				Yi = SSEI_m128shift(Yi);
				Zi = SSEI_m128shift(Zi);
				Mi = SSEI_m128shift(Mi);
				AXi = SSEI_m128shift(AXi);
				AYi = SSEI_m128shift(AYi);
				AZi = SSEI_m128shift(AZi);

				// Falls es etwas zum nachladen gibt, jetzt nachladen
				if(i + SSEI_OP < (*self).N)
				{

					// Speicher-Index des Elementes, was nachgeladen werden soll
					i_g = i + SSEI_OP;

					// Nachladen
					Xi[I_E] = (*self).X[i_g];
					Yi[I_E] = (*self).Y[i_g];
					Zi[I_E] = (*self).Z[i_g];
					Mi[I_E] = (*self).M[i_g];

				}

			}

			// Abschluss eines Durchlaufes der äußeren Schleife (Zeilen)
			for(f = 0; f < SSEI_OP; f++)
			{

				// j + f + m ergibt Index im Speicher für j-Vektoren, j_f
				j_f = j + f + m;

				// j-Beschleunigungen zurückschreiben
				(*self).AXmp[j_f] += AXj[f];
				(*self).AYmp[j_f] += AYj[f];
				(*self).AZmp[j_f] += AZj[f];

			}

		}

	}

}


void step_stage1_reduction(struct univ *self)
{

	// Iteration und Segmentierung
	COUNTER_DATATYPE m, mm, i, mi;

	// Reduktion: Parallele Speicherbereiche zusammenführen
	for(m = 0; m < (*self).OPENMP_threadsmax; m++)
	{

		// Sprungweite brechnen
		mm = m * (*self).N;

		// Einmal je Thread durch alle Massen laufen
		for(i = 0; i < (*self).N; i++)
		{

			// Sprung-Index in parallelem Speicher
			mi = i + mm;

			// Beschleunigungen aus unterschiedlichen Speicherbereichen summieren
			(*self).AX[i] += (*self).AXmp[mi];
			(*self).AY[i] += (*self).AYmp[mi];
			(*self).AZ[i] += (*self).AZmp[mi];

			// Parallele Speicherbereiche auf Null setzen
			(*self).AXmp[mi] = 0;
			(*self).AYmp[mi] = 0;
			(*self).AZmp[mi] = 0;

		}

	}

}

void step_stage1(struct univ *self)
{
	step_stage1_calc(self);
	step_stage1_reduction(self);
}

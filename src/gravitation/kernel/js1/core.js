/* -*- coding: utf-8 -*- */

/*

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

    src/gravitation/kernel/js1/core.js: js1 kernel javascript core

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

/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
CLASSES
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

class universe
{
	constructor(mass_list)
	{
		this._mass_list = [];
		this._a_list = [];
		for(let m of mass_list)
		{
			this._mass_list.push({
				"r": null,
				"a": new Array(__SIM_DIM__).fill(0.0),
				"m": m
			});
			this._a_list.push(null);
		}
	}
	update_pair(pm1, pm2)
	{
		let relative_r = [];
		for(let i = 0; i < __SIM_DIM__; i++)
			relative_r.push(pm1.r[i] - pm2.r[i]);
		let distance_sq = 0.0;
		for(let i = 0; i < __SIM_DIM__; i++)
			distance_sq += relative_r[i] * relative_r[i];
		let distance_inv = 1.0 / Math.sqrt(distance_sq);
		for(let i = 0; i < __SIM_DIM__; i++)
			relative_r[i] *= distance_inv;
		let a_factor = __G__ / distance_sq;
		let a1 = a_factor * pm2.m;
		let a2 = a_factor * pm1.m;
		for(let i = 0; i < __SIM_DIM__; i++)
			pm1.a[i] -= relative_r[i] * a1;
		for(let i = 0; i < __SIM_DIM__; i++)
			pm2.a[i] += relative_r[i] * a2;
	}
	step_stage1(r)
	{
		for(let i = 0; i < this._mass_list.length; i++)
			this._mass_list[i].r = r[i];
		for(let i = 0; i < this._mass_list.length; i++)
		{
			for(let j = 0; j < __SIM_DIM__; j++)
			{
				this._mass_list[i].a[j] = 0.0;
			}
		}
		for(let i = 0; i < this._mass_list.length - 1; i++)
		{
			for(let j = i + 1; j < this._mass_list.length; j++)
			{
				this.update_pair(this._mass_list[i], this._mass_list[j]);
			}
		}
		for(let i = 0; i < this._mass_list.length; i++)
			this._a_list[i] = this._mass_list[i].a;
		return this._a_list;
	}
}

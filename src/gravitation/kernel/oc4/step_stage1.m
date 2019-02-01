% -*- coding: utf-8 -*-

%{

GRAVITATION
n-body-simulation performance test suite
https://github.com/pleiszenburg/gravitation

	src/gravitation/kernel/oc4/step_stage1.m: oc4 kernel octave core

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

%}

% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% ROUTINES
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

function a = step_stage1(i_start, i_end, r)

    global m;
    global G;
    global SIM_DIM;
    global MASS_LEN;

    a = zeros(MASS_LEN, SIM_DIM, 'float');

    for i = i_start:i_end

        k = MASS_LEN - i;

        % np.subtract(self.mass_r_array[i,:], self.mass_r_array[i+1:,:], out = self.relative_r[:k])
        relative_r = r(i,:) - r(i+1:MASS_LEN,:);

        % np.multiply(self.relative_r[:k], self.relative_r[:k], out = self.distance_sqv[:k])
        distance_sqv = relative_r .* relative_r;

        % np.add.reduce(self.distance_sqv[:k], axis = 1, out = self.distance_sq[:k])
        distance_sq = sum(distance_sqv, 2);

        % np.sqrt(self.distance_sq[:k], out = self.distance_inv[:k])
        % np.divide(1.0, self.distance_inv[:k], out = self.distance_inv[:k])
        distance_inv = 1 ./ sqrt(distance_sq);

        % np.multiply(self.relative_r[:k], self.distance_inv[:k].reshape(k, 1), out = self.relative_r[:k])
        relative_r .*= distance_inv;

        % np.divide(self._G, self.distance_sq[:k], out = self.a_factor[:k])
        a_factor = G ./ distance_sq;

        % np.multiply(self.a_factor[:k], self.mass_m_array[i+1:], out = self.a1[:k])
        a1 = a_factor .* m(i+1:MASS_LEN).';

        % np.multiply(self.a_factor[:k], self.mass_m_array[i], out = self.a2[:k])
        a2 = a_factor .* m(i);

        % np.multiply(self.relative_r[:k], self.a1[:k].reshape(k, 1), out = self.a1r[:k])
        a1r = relative_r .* a1;

        % np.add.reduce(self.a1r[:k], axis = 0, out = self.a1v)
        a1v = sum(a1r, 1);

        % np.subtract(self.mass_a_array[i,:], self.a1v, out = self.mass_a_array[i,:])
        a(i,:) -= a1v;

        % np.multiply(self.relative_r[:k], self.a2[:k].reshape(k, 1), out = self.a2r[:k])
        a2r = relative_r .* a2;

        % np.add(self.mass_a_array[i+1:,:], self.a2r[:k], out = self.mass_a_array[i+1:,:])
        a(i+1:MASS_LEN,:) += a2r;

    end

end

"""This package consists of modules containing the definitions for four different
classes of parameters:

    amplitude,
    hybrid,
    functional,
    spatial

Each set of parameters measures a family of related properties describing a 
surface. You may refer to each class' respective module for a description of
the class as a whole and for documentation for the functions implementing each
parameter.

Before calculating any surface parameters, it's recommended to take a first-order
fit of the surface and then subtract that from the surface itself. `dr.utils.fit_and_subtract`
implements a helper function for doing so. Some parameters rely on other helper 
functions, which will be either imported from external libraries or again from 
`dr.utils`.

For more comprehensive information on each parameter, see: 
http://www.imagemet.com/WebHelp6/Default.htm#RoughnessParameters/Roughness_Parameters.htm
"""

from .amplitude import S_a, S_q, S_sk, S_ku, S_z, S_10z, S_v, S_p, S_mean
from .hybrid import S_sc, S_dq, S_dq6, S_dr, S_2a, S_3a
from .functional import S_bi, S_ci, S_vi, S_pk, S_k, S_vk, S_dc
from .spatial import S_ds, S_td, S_tdi, S_rw, S_rwi, S_hw, S_fd, S_cl, S_tr

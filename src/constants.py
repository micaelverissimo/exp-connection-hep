
__all__ = ['GeV', 'et_bins', 'eta_bins', 'et_str_bins', 'eta_str_bins']
import numpy as np

GeV = 1e3
et_bins  = [[15., 30], [30., 50.], [50., np.inf]]
eta_bins = [[0.0, 0.8], [0.8, 1.37], [1.37, 1.54], [1.54, 2.37], [2.37, 2.50]]

et_str_bins  = ['[15, 30[ GeV', '[30, 50[ GeV', r'$E_T \geq$ 50 GeV']
eta_str_bins = [f'[{l_eta}, {h_eta}[' for l_eta, h_eta in eta_bins] 
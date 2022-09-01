import os
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter as gf
import pdb

from colormaps import batlow,batlow_r,jet3,jet3_r
from multicolor import MultiColor

#======================================================================

### params
#   flds
#   prtl
#   spec

def _get_fname(var, path):
    _fnames = {'param' : os.path.join(path, 'output/params.{}'),
               'field' : os.path.join(path, 'output/flds.tot.{}'),
               'parts' : os.path.join(path, 'output/prtl.tot.{}'),
               'spec' : os.path.join(path, 'output/spec.tot.{}')}
    return _fnames[var]
                

def load_trist(vars='all', path='./', num=None, verbose=False):
    if type(vars) == str:
        if vars == 'all':
            _ftypes = ['param', 'field', 'parts', 'spec']
        else:
            _ftypes = vars.split()
    else:
        _ftypes = vars

    if 'param' not in _ftypes:
        _ftypes.append('param')

    ad = {}

    choices = get_output_times(path)
    while num not in choices:
        _ =  'Select from the following possible movie numbers: '\
             '\n{0} '.format(choices)
        num = int(input(_))

    num = '{:05d}'.format(num)
    for ft in _ftypes:
        fname = _get_fname(ft, path).format(num)
        try:
            with h5py.File(fname, 'r') as f:
                for k in f.keys():
                    if verbose:
                        print('loading {} from {}'.format(k, fname))
                    ad[k] = f[k][:]
        except:
            print("File not found?", fname)

    ad['xx'] = np.arange(1, 1 + ad['grid:mx0'][0], ad['output:istep'])
    ad['yy'] = np.arange(1, 1 + ad['grid:my0'][0], ad['output:istep'])

    ad['xx'] = (ad['xx'] + .5)/(ad['plasma:c_omp'][0])
    ad['yy'] = (ad['yy'] + .5)/(ad['plasma:c_omp'][0])

    for k in ad:
        try:
            ad[k] = np.squeeze(ad[k])
        except:
            pass

    #print(ad['v3x'].shape[2])
    #print(ad['xx'])
    #print(ad['c_omp'])

    return ad

#======================================================================

def get_output_times(path='./', var='field'):
    import glob
    import os
     
    #dpath = os.path.join(path, _get_fname(var, path).format('*'))
    dpath = _get_fname(var, path).format('*')
    choices = glob.glob(dpath)
    # Now there are fies that end in .xdmf, and we gotta get rid of them
    choices = [int(c[-3:]) for c in choices if c[-5:] != '.xdmf']
    choices.sort()
    return np.array(choices)

#======================================================================

def odp(d, v, ax=None, **kwargs):
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    if type(v) ==  str:
        v = d[v]

    return ax.plot(d['xx'], np.mean(np.squeeze(v), axis=0), **kwargs)

#======================================================================

def calc_gamma(d):
    keys = d.keys()
    for k in keys:
        if k[:2] == 'u_':
            n = k[2:]
            _g = 1./np.sqrt(1. - (d['u_'+n]**2 + d['v_'+n]**2 + d['w_'+n]**2))
            d['g_'+n] = _g

    return None
#======================================================================

def calc_psi(f):
    """ Calculated the magnetic scaler potential for a 2D simulation
    Args:
        d (dict): Dictionary containing the fields of the simulation
            d must contain bx, by, xx and yy
    Retruns:
        psi (numpy.array(len(d['xx'], len(d['yy']))) ): Magnetic scaler
            potential
    """

    bx = np.squeeze(f['bx'])
    by = np.squeeze(f['by'])
    dx = dy = 1./f['c_omp']

    psi = 0.0*bx
    psi[1:,0] = np.cumsum(bx[1:,0])*dy
    psi[:,1:] = (psi[:,0] - np.cumsum(by[:,1:], axis=1).T*dx).T

    return psi.T

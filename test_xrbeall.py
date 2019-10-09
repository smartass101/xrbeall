"""Test the Beall(1982) method using a linear dispersion relation"""

import numpy as np
import xarray

import xrbeall


def test_beal_fft():
    N = 2**8
    t, dt = np.linspace(0, 10, N**2, retstep=True)
    f = (np.arange(N) - N//2) * 0.5/dt
    test_ds = xarray.Dataset(coords={'time':t, 'frequency': f})
    test_ds.attrs['v_phase'] = 1
    test_ds['k'] = 2*np.pi * test_ds.frequency / test_ds.v_phase
    test_ds.attrs['Dx'] = 2*np.pi / test_ds.k.max().item() / 2 # K_nyq = 0.5/Dx
    test_ds['phases'] = (['frequency'],
                    np.random.random(test_ds.frequency.size) * 2*np.pi)
    t, f, k, p = (test_ds[k] for k in ['time', 'frequency', 'k', 'phases'])
    Dx = test_ds.Dx
    test_ds['sig1'] = (np.cos(2*np.pi*f*t + p)).mean(dim='frequency')
    test_ds['sig2'] = (np.cos(2*np.pi*f*t + k*Dx + p)).mean(dim='frequency')
    ds = xrbeall.beall_fft(test_ds.sig1, test_ds.sig2, test_ds.Dx,
                      test_ds.k.size, nperseg=test_ds.frequency.size*2-1)
    recovered_k = ds.wavenumber[ds.S_K_f.argmax(dim='wavenumber')]
    # TODO f=0,f_max does not work, maybe detrending or DC?

    #np.testing.assert_allclose(recovered_k[2:], test_ds.k[2:], rtol=0.50)
    stat_disp_ds = xrbeall.statistical_dispersion_relation(ds)
    np.testing.assert_allclose(stat_disp_ds.K_mean, test_ds.k, rtol=0.3, atol=20)



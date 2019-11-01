"""Test the Beall(1982) method using a linear dispersion relation"""

import numpy as np
import xarray

import xrbeall


def test_beal_fft():
    N = 2**8
    t, dt = np.linspace(0, 10, N**2, retstep=True)
    f = np.fft.fftfreq(N, dt)
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
                           test_ds.k.size,
                           avg_dim=None,
                           nperseg=test_ds.frequency.size,
                           return_onesided=False,
    )
    S_K_f = ds.S_K_f.mean(dim='time')
    counts = S_K_f.count(dim='wavenumber')
    assert counts.min() > 0

    stat_disp_ds = xrbeall.statistical_dispersion_relation(S_K_f, ds.S.mean(dim='time'))
    f_nyq = stat_disp_ds.frequency.max()
    # close to f=0 or f=f_nyq it becomes inaccurate
    mask = (1e3 < np.abs(stat_disp_ds.frequency)) & (np.abs(stat_disp_ds.frequency) < f_nyq * 0.9)
    np.testing.assert_allclose(stat_disp_ds.K_mean[mask], test_ds.k[mask], rtol=0.05)



"""Wavenumber and frequency spectrum S(K,f) estimation based on Beall(1982)"""
import numpy as np              # for linspace construction
import numba                    # for optimized for loops
import xarray                   # for data container construction
import xarray_dsp as dsp        # for spectrogram


@numba.jit
def average_close_K(S, K_l, K):
    """Return S_K_f[k,j] = average over i: S[j,i] if K_l[j,i] close to K[k] else 0"""
    Nf, Nt = S.shape            # same as K_l
    NK = K.shape[0]
    bin_half_width = (K[1] - K[0]) / 2.0
    S_K_f = np.zeros((NK, Nf))
    for j in range(Nf):
        for k in range(NK):
            for i in range(Nt):
                if abs(K_l[j,i] - K[k]) < bin_half_width:
                    S_K_f[k,j] += S[j,i]
            S_K_f[k,j] /= Nt
    return S_K_f


def beall(S, H, Dx, NK):
    """Beall (1982) method of S(K,f) estimation from 2-point measurements

    Estimates the local wavenumber K and frequency f spectrum S(K,f)
    given the signal spectra and cross-spectrum and the distance between points

    Parameters
    ----------
    S : xarray.DataArray, (Nf,Nt) -> ('frequency', 'time')
        spectrum ansamble (i.e. spectrogram) S(f,t)
        should be average of spectra from both points
    H : array_like, (Nf,Nt)
        cross-spectrum ansamble (i.e. cross-spectrogram) H(f,t)
    Dx : float
        distance between the 2 measurement points
    NK : int
        requested number of points in local wavenumber dimension

    Returns
    -------
    S_K_f : xarray.DataArray, (NK, Nf) -> ('wavenumber', 'frequency')
        calculated S(K, f) spectrum
    """
    K_max = np.pi / Dx  # Nyquist-like maximum unambiguously resolvable wavenumber
    K, DK = np.linspace(-K_max, K_max, NK, retstep=True)  # wavenumber range
    # local wavenumber approximated by central phase differentiation
    K_l = np.angle(H) / Dx        # cross-phase over distance
    K_l = np.clip(K_l, -K_max, K_max)             # remove ambiguous wavenumbers
    S_K_f = average_close_K(S.values, K_l, K)     # spectrum averaging in bin selection
    S_K_f = xarray.DataArray(S_K_f, coords=[('wavenumber', K), ('frequency', S.frequency)])
    return S_K_f


def beall_fft(sig1, sig2, Dx, NK, **spectral_kw):
    """Wrapper around :func:`beall` which uses FFT for spectra estimation

    Parameters
    ----------
    sig1 : xarray.DataArray, (Nt) -> ('time')
         1. signal to calculate spectra from
    sig2 : xarray.DataArray, (Nt) -> ('time')
         2. signal to calculate spectra from
    Dx : float
        distance between the 2 measurement points
    NK : int
        requested number of points in local wavenumber dimension
    spectral_kw : keyword arguments, optional
        extra keyword arguments will passed on to :func:`xarray_dsp.spectrogram`
        by default noverlap=0 to be consistent with Beall (1982)
        seglen=1/f_res may be of interest

    Returns
    -------
    ds : xarray.Dataset({S_K_f, S_f, S1, S2, H})
        S_K_f : S(K,f) wavenumber and frequency spectrum returned by :func:`beall`
        S_f : average S(f) spectrum used to calculate or normalize S(K,f)
        S1 : spectrogram of sig1
        S2 : spectrogram of sig2
        H : cross-spectrogram of sig1 and sig2
    """
    spectral_kw = spectral_kw.copy()  # will modify copy
    spectral_kw.setdefault('noverlap', 0)  # to be consistent with Beall
    # ansamble spectra
    S1, S2 = (dsp.spectrogram(s, **spectral_kw) for s in (sig1, sig2))
    S = 0.5 * (S1 + S2)         # average ansamble spectra
    # cross spectra ansamble
    H = dsp.crossspectrogram(sig1, sig2, **spectral_kw)
    S_K_f = beall(S, H, Dx, NK)
    ds = xarray.Dataset({'S_K_f': S_K_f, 'S_f': S.mean(dim='time'),
                         'S1': S1, 'S2': S2, 'H': H})
    return ds


def statistical_dispersion_relation(beall_ds):
    s_K_f = beall_ds.S_K_f / beall_ds.S_f  # conditional spectrum
    K = s_K_f.wavenumber
    K_mean = (K * s_K_f).sum(dim='wavenumber')
    K_std = np.sqrt( ( (K - K_mean)**2 * s_K_f ).sum(dim='wavenumber') )
    ds = xarray.Dataset({'s_K_f': s_K_f, 'K_mean': K_mean, 'K_std': K_std})
    return ds

"""Wavenumber and frequency spectrum S(K,f) estimation based on Beall(1982)"""
import numpy as np              # for linspace construction
import xarray as xr             # for data container construction


def beall(S, H, Dx, NK, chunks={'frequency': 1}, avg_dim='time',
          wavenumber_dim='wavenumber'):
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
    chunks : dict, optional
        argument to :py:meth:`xarray.DataArray.chunk` to distribute
        broadcasted K-f computation
    avg_dim : str, optional
        dimension over which to average over with mean. If None,
        no averaging is done and is left to the user.
    wavenumber_dim : str, optional
        name of the wavenumber dimension

    Returns
    -------
    S_K_f : xarray.DataArray, (NK, Nf) -> ('wavenumber', 'frequency')
        S(K, f) spectrum, wrapping dask array (uncomputed)
        if avg_dim is None, will also contain the 'time' dimension
    """
    K_max = np.pi / Dx  # Nyquist-like maximum unambiguously resolvable wavenumber
    K, DK = np.linspace(-K_max, K_max, NK, retstep=True)  # wavenumber range
    K = xr.DataArray(K, coords=[(wavenumber_dim, K)], name=wavenumber_dim)
    # local wavenumber approximated by central phase differentiation
    K_l = xr.apply_ufunc(np.angle, H) / Dx  # cross-phase over distance
    # chunk to a Dask array
    K_ld = K_l.chunk(chunks)
    in_bin = np.abs(K_ld - K) < DK/2
    S_K_f_toavg = S.where(in_bin)
    if avg_dim is None:
        return S_K_f_toavg
    else:
        S_K_f = S_K_f_toavg.mean(dim=avg_dim)
        return S_K_f


def beall_fft(sig1, sig2, Dx, NK, chunks={'frequency': 1}, avg_dim='time',
              wavenumber_dim='wavenumber', **spectral_kw):
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
    chunks : dict, optional
        argument to :py:meth:`xarray.DataArray.chunk` to distribute
        broadcasted K-f computation
    avg_dim : str, optional
        dimension over which to average over with mean. If None,
        no averaging is done and is left to the user.
    wavenumber_dim : str, optional
        name of the wavenumber dimension
    spectral_kw : keyword arguments, optional
        extra keyword arguments will passed on to :py:func:`xarray_dsp.spectrogram`
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
    import xrscipy.signal as dsp
    spectral_kw = spectral_kw.copy()  # will modify copy
    spectral_kw.setdefault('noverlap', 0)  # to be consistent with Beall
    # ansamble spectra
    S1, S2 = (dsp.spectrogram(s, **spectral_kw) for s in (sig1, sig2))
    S = 0.5 * (S1 + S2)         # average ansamble spectra
    # cross spectra ansamble
    H = dsp.crossspectrogram(sig1, sig2, **spectral_kw)
    S_K_f = beall(S, H, Dx, NK, chunks, avg_dim, wavenumber_dim)
    ds = xr.Dataset({'S_K_f': S_K_f, 'S': S,
                         'S1': S1, 'S2': S2, 'H': H})
    return ds


def statistical_dispersion_relation(S_K_f, S_f, wavenumber_dim='wavenumber'):
    r"""Estimate the statistical dispersion relation


    Calculates the mean and standard deviation of the wavenumber with respect
    to the other dimensions as

    .. math::

        s(K, f) = S(K, f) / S(f)
        \bar{K} = \int s(K,f) \cdot K dK
        \mathrm{sd}(K) = \sqrt{\int s(K,f) \cdot (K - \bar{K})^2  dK}


    Parameters
    ----------
    S_K_f : xarray.DataArray
        estimated S(K, f; ...) wavenumber-frequency spectrum
        must contain *wavenumber_dim* dimension
        other dimensions (frequency, time if not fully averaged)
        must broadcast with *S_f*
    S_f : xarray.DataArray
        frequency spectrum
        must broadcast with *S_K_f*
    wavenumber_dim : str, optional
        name of the wavenumber dimension in *S_K_f*

    Returns
    -------
    ds : xarray.Dataset
        s_K_f : conditional spectrum
        K_mean : mean K
        K_std : std. deviation about K_mean
    """
    s_K_f = S_K_f / S_f  # conditional spectrum
    K = s_K_f.coords[wavenumber_dim]
    K_mean = (K * s_K_f).sum(dim=wavenumber_dim)
    K_std = np.sqrt( ( (K - K_mean)**2 * s_K_f ).sum(dim=wavenumber_dim) )
    ds = xr.Dataset({'s_K_f': s_K_f, 'K_mean': K_mean, 'K_std': K_std})
    return ds

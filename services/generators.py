import numpy as np


# =========================
#  Generátory časových řad
# =========================

def generate_logistic_map(length, r=3.9, x0=0.2, burn=500):
    """
    Logistická mapa: x_{n+1} = r * x_n * (1 - x_n)
    Vrací posledních `length` hodnot po zahozní burn-in části.

    Parameters
    ----------
    length : int
        Počet vzorků v návratové sérii (po burn-in).
    r : float
        Parametr mapy (typicky 3.5–4.0 pro chaotické chování).
    x0 : float
        Počáteční podmínka v (0, 1).
    burn : int
        Počet počátečních iterací, které zahodíme (burn-in).

    Returns
    -------
    np.ndarray tvaru (length,)
    """
    N = length + burn
    x = np.empty(N)
    x[0] = x0
    for i in range(1, N):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x[burn:]


def generate_henon_map(length, a=1.4, b=0.3, x0=0.1, y0=0.0, burn=200):
    """
    Henonova mapa:
        x_{n+1} = 1 - a x_n^2 + y_n
        y_{n+1} = b x_n
    Vrací x-sérii po burn-in.

    Parameters
    ----------
    length : int
        Počet vzorků v návratové x-sérii (po burn-in).
    a, b : float
        Parametry Henonovy mapy (klasicky a=1.4, b=0.3).
    x0, y0 : float
        Počáteční podmínky.
    burn : int
        Počet počátečních kroků, které jsou zahozena.

    Returns
    -------
    np.ndarray tvaru (length,)
        Pouze x-složka po burn-in.
    """
    N = length + burn
    xs = np.empty(N)
    ys = np.empty(N)
    xs[0] = x0
    ys[0] = y0
    for n in range(1, N):
        xs[n] = 1 - a * xs[n - 1] ** 2 + ys[n - 1]
        ys[n] = b * xs[n - 1]
    return xs[burn:]


def generate_lorenz_x(
    length,
    dt=0.01,
    sigma=10.0,
    rho=28.0,
    beta=8 / 3,
    x0=1.0,
    y0=1.0,
    z0=1.0,
    burn=1000,
):
    """
    Lorenzův systém integrován jednoduchým Eulerovým schématem.
    Vrací pouze x-sérii po burn-in.

    dx/dt = sigma (y - x)
    dy/dt = x (rho - z) - y
    dz/dt = x y - beta z

    Parameters
    ----------
    length : int
        Počet vzorků v návratové x-sérii (po burn-in).
    dt : float
        Krok integrace.
    sigma, rho, beta : float
        Parametry Lorenzova systému.
    x0, y0, z0 : float
        Počáteční podmínky.
    burn : int
        Počet počátečních kroků, které zahodíme.

    Returns
    -------
    np.ndarray tvaru (length,)
        x-složka Lorenzova systému po burn-in.
    """
    N = length + burn
    xs = np.empty(N)
    ys = np.empty(N)
    zs = np.empty(N)
    xs[0], ys[0], zs[0] = x0, y0, z0

    for i in range(1, N):
        dx = sigma * (ys[i - 1] - xs[i - 1])
        dy = xs[i - 1] * (rho - zs[i - 1]) - ys[i - 1]
        dz = xs[i - 1] * ys[i - 1] - beta * zs[i - 1]

        xs[i] = xs[i - 1] + dx * dt
        ys[i] = ys[i - 1] + dy * dt
        zs[i] = zs[i - 1] + dz * dt

    return xs[burn:]


def generate_pink_noise(length):
    """
    1/f šum (pink noise) vytvořený ve frekvenční doméně.

    Postup:
        - vygenerujeme náhodné fáze
        - amplitudu zvolíme ~ 1/sqrt(f) (pro f > 0)
        - provedeme iFFT a ořízneme na požadovanou délku
        - normalizujeme na nulový průměr a jednotkovou směrodatnou odchylku

    Parameters
    ----------
    length : int
        Počet vzorků v návratovém signálu.

    Returns
    -------
    np.ndarray tvaru (length,)
        Normalizovaný 1/f šum.
    """
    # nejbližší mocnina 2 >= length kvůli FFT
    N = int(2 ** np.ceil(np.log2(length)))
    freqs = np.fft.rfftfreq(N)
    phases = np.random.uniform(0, 2 * np.pi, len(freqs))

    # amplituda ~ 1/sqrt(f), f=0 nastavíme na 0
    amplitude = np.where(freqs == 0, 0.0, 1.0 / np.sqrt(freqs))
    spectrum = amplitude * (np.cos(phases) + 1j * np.sin(phases))

    signal = np.fft.irfft(spectrum, n=N)
    signal = signal[:length]

    # normalizace (nulový průměr, jednotková směrodatná odchylka)
    signal = (signal - signal.mean()) / signal.std()
    return signal

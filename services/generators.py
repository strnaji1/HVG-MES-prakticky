import numpy as np

# =========================
#  Generátory časových řad
# =========================

def generate_logistic_map(length, r=3.9, x0=0.2, burn=500):
    # Logistická mapa: x_{n+1} = r * x_n * (1 - x_n)
    N = length + burn
    x = np.empty(N)
    x[0] = x0
    for i in range(1, N):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x[burn:] # Oříznutí o náběh (burn-in)


def generate_henon_map(length, a=1.4, b=0.3, x0=0.1, y0=0.0, burn=200):
    # Dvourozměrná Henonova mapa, vrací pouze složku X
    N = length + burn
    xs = np.empty(N)
    ys = np.empty(N)
    xs[0], ys[0] = x0, y0
    for n in range(1, N):
        xs[n] = 1 - a * xs[n - 1] ** 2 + ys[n - 1]
        ys[n] = b * xs[n - 1]
    return xs[burn:] # Stabilizovaná x-série


def generate_lorenz_x(length, dt=0.01, sigma=10.0, rho=28.0, beta=8/3, x0=1.0, y0=1.0, z0=1.0, burn=1000):
    # Lorenzův atraktor řešený Eulerovou metodou
    N = length + burn
    xs, ys, zs = np.empty(N), np.empty(N), np.empty(N)
    xs[0], ys[0], zs[0] = x0, y0, z0

    for i in range(1, N):
        # Diferenciální rovnice systému
        dx = sigma * (ys[i - 1] - xs[i - 1])
        dy = xs[i - 1] * (rho - zs[i - 1]) - ys[i - 1]
        dz = xs[i - 1] * ys[i - 1] - beta * zs[i - 1]

        # Numerický integrační krok
        xs[i] = xs[i - 1] + dx * dt
        ys[i] = ys[i - 1] + dy * dt
        zs[i] = zs[i - 1] + dz * dt

    return xs[burn:] # Návrat chaotické x-složky


def generate_pink_noise(length):
    # 1/f šum generovaný spektrální syntézou
    N = int(2 ** np.ceil(np.log2(length))) # Zarovnání na mocninu 2 pro FFT
    freqs = np.fft.rfftfreq(N)
    phases = np.random.uniform(0, 2 * np.pi, len(freqs))

    # Definice amplitudy (1/sqrt(f)) a fázový posun
    amplitude = np.where(freqs == 0, 0.0, 1.0 / np.sqrt(freqs))
    spectrum = amplitude * (np.cos(phases) + 1j * np.sin(phases))

    # Přechod do časové oblasti a ořez na délku
    signal = np.fft.irfft(spectrum, n=N)
    signal = signal[:length]

    # Z-score normalizace signálu
    return (signal - signal.mean()) / signal.std()
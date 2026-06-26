import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import PyMieScatt as ps


#load-dieltric-data

location = r"[your location]"

au_file = location + r"\Permittivity_Gold_JohnsonChristy.txt"
ag_file = location + r"\Permittivity_Silver_JohnsonChristy.txt"

au_data = np.loadtxt(au_file)
ag_data = np.loadtxt(ag_file)

#File-format:
#Energy(eV)   eps1   eps2

au_data = np.loadtxt(au_file)
ag_data = np.loadtxt(ag_file)

#Au-data

energy_au = au_data[:,0]
eps1_au  = au_data[:,1]
eps2_au  = au_data[:,2]

#Ag-data

energy_ag = ag_data[:,0]
eps1_ag  = ag_data[:,1]
eps2_ag  = ag_data[:,2]

#Energy-to-Wavelength

wavelength_au = 1240 / energy_au
wavelength_ag = 1240 / energy_ag

#Sorting

idx_au = np.argsort(wavelength_au)
idx_ag = np.argsort(wavelength_ag)

wavelength_au = wavelength_au[idx_au]
eps1_au = eps1_au[idx_au]
eps2_au = eps2_au[idx_au]

wavelength_ag = wavelength_ag[idx_ag]
eps1_ag = eps1_ag[idx_ag]
eps2_ag = eps2_ag[idx_ag]


#Wavelenght range


wavelength = np.linspace(
    350,
    750,
    800
)


#Interpolation

eps1_au_interp = interp1d(
    wavelength_au,
    eps1_au,
    kind='cubic',
    fill_value='extrapolate'
)(wavelength)

eps2_au_interp = interp1d(
    wavelength_au,
    eps2_au,
    kind='cubic',
    fill_value='extrapolate'
)(wavelength)

eps1_ag_interp = interp1d(
    wavelength_ag,
    eps1_ag,
    kind='cubic',
    fill_value='extrapolate'
)(wavelength)

eps2_ag_interp = interp1d(
    wavelength_ag,
    eps2_ag,
    kind='cubic',
    fill_value='extrapolate'
)(wavelength)


#complex-refractive-index

m_au = np.sqrt(
    eps1_au_interp
    +
    1j * eps2_au_interp
)

m_ag = np.sqrt(
    eps1_ag_interp
    +
    1j * eps2_ag_interp
)

#Core-size

core_radius = 28

shell_thicknesses = (
    list(range(0,11,1))
)

#Peak-data

peak_wavelengths = []
peak_shells = []

#Spectrum
fig, ax = plt.subplots(figsize=(8,6))

colors = plt.cm.plasma(
    np.linspace(
        0.15,
        0.95,
        len(shell_thicknesses)
    )
)

for idx, shell in enumerate(shell_thicknesses):

    total_radius = core_radius + shell

    extinction = []

    #Loop-wavelength

    for i in range(len(wavelength)):

        result = ps.MieQCoreShell(
            mCore=m_au[i],
            mShell=m_ag[i],
            wavelength=wavelength[i],
            dCore=2 * core_radius,
            dShell=2 * total_radius,
            nMedium=1.333
        )

        Qext = result[0]

        Cext = Qext * np.pi * total_radius**2

        extinction.append(Cext)

    extinction = np.array(extinction)

    #Peak-detector

    peak_idx = np.argmax(extinction)

    peak_wavelength = wavelength[peak_idx]

    print(f"Shell {shell} nm --> Peak = {peak_wavelength:.1f} nm")

    peak_shells.append(shell)
    peak_wavelengths.append(peak_wavelength)

    #Plot-spectrum

    ax.plot(
        wavelength,
        extinction,
        lw=2,
        color=colors[idx],
        label=f"{shell} nm"
    )

#Graph setting

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel(r"Extinction Cross Section (nm$^2$)")

ax.set_xlim(375,625)
ax.set_xticks([400,450,500,550,600])

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((4,4))
ax.yaxis.set_major_formatter(formatter)

ax.tick_params(
    direction='out',
    length=5,
    width=1
)

ax.legend(
    title="Ag Shell",
    fontsize=9,
    ncol=2
)

#Peak tren

fig2, ax2 = plt.subplots(figsize=(6,5))

ax2.plot(
    peak_shells,
    peak_wavelengths,
    '-',
    color='gray',
    lw=1.5
)

ax2.scatter(
    peak_shells,
    peak_wavelengths,
    c=colors,
    s=80,
    zorder=3
)

ax2.set_xlabel("Ag Shell Thickness (nm)")
ax2.set_ylabel("Peak Wavelength (nm)")

ax2.set_xlim(-0.5,10.5)

ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()
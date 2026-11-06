import contextlib
import io
import numpy
import matplotlib.pyplot as plt

import _env
import preprocessing
import processing
import compute_alpha
from demo_control_polycopie2024 import our_optimization_procedure_multiple, compute_objective_function

N = 50
level = 0
c0 = 340.0

betas = [0.1, 0.25, 0.4, 0.6, 0.8]
nb_resonances = 5
numb_iter = 20

f_min, f_max = 270.0, 1600.0
prix_m2 = 36.0

material_name = 'Melamine'
material = {'phi': 0.99, 'sigma': 14000.0, 'alpha_h': 1.02}


alphas_calcules = {}


def Alpha_code(k):
    if k not in alphas_calcules:
        with contextlib.redirect_stdout(io.StringIO()):
            alpha, erreur = compute_alpha.compute_alpha(k * c0, material, material_name)
        alphas_calcules[k] = -alpha
    return alphas_calcules[k]


def energie(k, chi):
    A = Alpha_code(k) if numpy.any(chi) else 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        u = processing.solve_helmholtz(domain_omega, spacestep, k, f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, A * chi)
    return compute_objective_function(domain_omega, u, spacestep)


def energie_moyenne(chi):
    return numpy.mean([energie(k, chi) for k in k_plage])


M = 2 * N
spacestep = 1.0 / N
beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)
f_dir[:, :] = 0.0
f_dir[0, 0:N] = 1.0

mur = (domain_omega == _env.NODE_ROBIN)
chi_nu = numpy.zeros((M, N))
chi_plein = mur.astype(float)
facteur_surface = mur.sum() / N

k_plage = numpy.linspace(2 * numpy.pi * f_min / c0, 2 * numpy.pi * f_max / c0, 60)


k_fin = numpy.linspace(k_plage[0], k_plage[-1], 200)
J_nu = numpy.array([energie(k, chi_nu) for k in k_fin])
pics = [i for i in range(1, len(k_fin) - 1) if J_nu[i] > J_nu[i - 1] and J_nu[i] > J_nu[i + 1]]
pics = sorted(pics, key=lambda i: -J_nu[i])[:nb_resonances]     # les plus gros pics
k_resonances = sorted(k_fin[pics])

Alphas_resonances = [Alpha_code(k) for k in k_resonances]
for k in k_plage:
    Alpha_code(k)

J_plein = energie_moyenne(chi_plein)
resultats_beta = []
resultats_dB = []
resultats_chi = []

for beta in betas:
    chi0 = preprocessing.set2zero(preprocessing._set_chi(M, N, x, y), domain_omega)
    with contextlib.redirect_stdout(io.StringIO()):
        chi_opt, energy, u_list, grad = our_optimization_procedure_multiple(
            domain_omega, spacestep, k_resonances, f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
            Alphas_resonances, 5.0, chi0, beta, numb_iter=numb_iter)
    gain = 10 * numpy.log10(J_plein / energie_moyenne(chi_opt))
    resultats_beta.append(beta)
    resultats_dB.append(gain)
    resultats_chi.append(chi_opt)

resultats_beta.append(1.0)
resultats_dB.append(0.0)
resultats_chi.append(chi_plein)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(resultats_beta, resultats_dB, 'o-')
ax.axhline(0, color='gray', ls='--', lw=1)
ax.text(0.02, 0.1, 'mur entièrement couvert', color='gray', transform=ax.get_yaxis_transform())
ax.set_xlabel('β : part du mur couverte de poreux')
ax.set_ylabel('Gain par rapport au mur entièrement couvert (dB)')
ax.set_title(f'Coût - performance, {material_name}, mur '
             f'{"plat" if level == 0 else "préfractal niveau " + str(level)}')
ax.grid(alpha=0.3)
ax2 = ax.secondary_xaxis('top', functions=(lambda b: b * facteur_surface * prix_m2,
                                           lambda c: c / (facteur_surface * prix_m2)))
ax2.set_xlabel('Coût du matériau (€ par m² de mur)')
plt.tight_layout()
nom_figure = f'cout_performance_{material_name}_level{level}.png'
plt.savefig(nom_figure, dpi=130)
print(f'\nfigure enregistrée: {nom_figure}')

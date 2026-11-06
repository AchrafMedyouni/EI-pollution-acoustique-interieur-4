import contextlib
import io
import numpy
import scipy.signal
import matplotlib.pyplot as plt

import _env
import preprocessing
import processing
import compute_alpha
from demo_control_polycopie2024 import our_optimization_procedure_multiple, compute_objective_function

N = 50
level = 2
c0 = 340.0
epaisseur = 0.01

betas = [0.1, 0.25, 0.4, 0.6, 0.8]
nb_resonances = 5
numb_iter = 20
f_min, f_max = 270.0, 1600.0


MATERIALS = {
    'Mélamine': {'phi': 0.99, 'sigma': 14000.0, 'alpha_h': 1.02, 'f_emission': 80.4},
    'Polyuréthane': {'phi': 0.98, 'sigma': 45000.0, 'alpha_h': 2.01, 'f_emission': 3.22 * 35.0},
    'Pin': {'phi': 0.671, 'sigma': 35540.0, 'alpha_h': 1.22, 'f_emission': 24.0},
}
reference = 'Mélamine'     # le mur plein de ce matériau sert de référence (gain = 0 dB)
PRIX = {
    'Mélamine': 8.5,
    'Polyuréthane': 2.9,
    'Pin': 6.9
}

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
chi_depart = preprocessing.set2zero(preprocessing._set_chi(M, N, x, y), domain_omega)
facteur_surface = mur.sum() / N

k_plage = numpy.linspace(2 * numpy.pi * f_min / c0, 2 * numpy.pi * f_max / c0, 60)


def alpha_code(k, nom):

    with contextlib.redirect_stdout(io.StringIO()):
        alpha, erreur = compute_alpha.compute_alpha(k * c0, MATERIALS[nom], nom)
    return -alpha


def energie(domain, k, chi, A=0.0):

    u = processing.solve_helmholtz(domain, spacestep, k, f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, A * chi)
    return compute_objective_function(domain, u, spacestep)


def energie_moyenne(domain, chi, alphas):

    return numpy.mean([energie(domain, k, chi, A) for k, A in zip(k_plage, alphas)])


def impact_carbone(nom, beta):

    volume = beta * facteur_surface * epaisseur
    return volume * MATERIALS[nom]['f_emission']

def cout(nom, beta):
    """Prix (€) de matériau pour 1 m de mur : surface couverte (m2) x prix au m2"""
    surface = beta * facteur_surface            # m2 de poreux pour 1 m de mur
    return surface * PRIX[nom]


# =====================================================================
# ÉTAPE 1 : fréquences de résonance du mur nu (les mêmes pour tous les matériaux)
# =====================================================================
print('Étape 1 : recherche des résonances du mur nu...')
k_fin = numpy.linspace(k_plage[0], k_plage[-1], 200)
J_nu = numpy.array([energie(domain_omega, k, chi_nu) for k in k_fin])
pics, _ = scipy.signal.find_peaks(J_nu)
pics = sorted(pics, key=lambda i: -J_nu[i])[:nb_resonances]
k_resonances = sorted(k_fin[pics])

alphas_plage = {nom: [alpha_code(k, nom) for k in k_plage] for nom in MATERIALS}

domain_plat, _, _, _, _ = preprocessing._set_geometry_of_domain(M, N, 0)
chi_plein_plat = (domain_plat == _env.NODE_ROBIN).astype(float)
J_ref = energie_moyenne(domain_plat, chi_plein_plat, alphas_plage[reference])

betas_tracees = betas + [1.0]
resultats = {}
for nom in MATERIALS:
    alphas_resonances = [alpha_code(k, nom) for k in k_resonances]
    gains = []
    for beta in betas:
        with contextlib.redirect_stdout(io.StringIO()):
            chi_opt, energy, u_list, grad = our_optimization_procedure_multiple(
                domain_omega, spacestep, k_resonances, f, f_dir, f_neu, f_rob,
                beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                alphas_resonances, 5.0, chi_depart, beta, numb_iter=numb_iter)
        gains.append(10 * numpy.log10(J_ref / energie_moyenne(domain_omega, chi_opt, alphas_plage[nom])))
    gains.append(10 * numpy.log10(J_ref / energie_moyenne(domain_omega, chi_plein, alphas_plage[nom])))
    resultats[nom] = gains

forme = 'plat' if level == 0 else 'préfractal niveau ' + str(level)


def tracer(abscisses, xlabel, fichier):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nom, gains in resultats.items():
        ax.plot(abscisses(nom), gains, 'o-', label=nom)
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'Gain par rapport au mur plat plein de {reference} (dB)')
    ax.set_title(f'Mur {forme}')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fichier, dpi=130)


tracer(lambda nom: betas_tracees, 'β : part du mur couverte de poreux', f'perf_beta_materiaux_level{level}.png')
tracer(lambda nom: [impact_carbone(nom, b) for b in betas_tracees],
       'Impact carbone : kg CO$_2$ eq par m de mur', f'perf_carbone_materiaux_level{level}.png')
tracer(lambda nom: [cout(nom, b) for b in betas_tracees],
       'Coût : € par m de mur', f'perf_prix_materiaux_level{level}.png')
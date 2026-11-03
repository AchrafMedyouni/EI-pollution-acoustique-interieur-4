# -*- coding: utf-8 -*-
"""
Courbe coût - performance pour un matériau et une forme de mur donnés.

Principe (simple) :
  1. on trouve les fréquences de résonance du mur NU (les plus gros pics de J(f)) ;
  2. pour chaque quantité de poreux beta, on optimise la répartition chi
     sur ces fréquences (somme des J, fonction our_optimization_procedure) ;
  3. on mesure la performance = réduction de bruit en dB par rapport au mur nu,
     moyennée sur toute la plage de la voix ;
  4. on trace la performance en fonction de beta et du coût.
"""

import numpy
import matplotlib.pyplot as plt

import _env
import preprocessing
import processing
from demo_control_polycopie2024 import our_optimization_procedure_multiple, compute_objective_function


# =====================================================================
# PARAMÈTRES À CHOISIR
# =====================================================================
N = 50                     # points de la grille (domaine de 1 m x 1 m)
level = 0                  # 0 = mur plat, 1 ou 2 = mur préfractal
c0 = 343.0                 # vitesse du son (m/s)

betas = [0.1, 0.25, 0.4, 0.6, 0.8]   # quantités de poreux testées (beta = 0 et 1 sont ajoutés)
nb_resonances = 5          # nombre de fréquences de résonance utilisées pour optimiser
numb_iter = 20             # itérations de la descente de gradient par beta

f_min, f_max = 270.0, 1600.0         # plage de la voix étudiée (Hz)
prix_m2 = 36.0             # prix de la mélamine (€/m²), environ, achat grand public

# matériau : mélamine (porosité, résistivité, tortuosité)
phi, sigma, alpha_h = 0.99, 14000.0, 1.02


# =====================================================================
# FONCTIONS UTILES
# =====================================================================
def compute_alpha(omega):
    """alpha(omega) du matériau, convention du poly (dn u + alpha u = 0)."""
    gamma_p, rho0 = 1.4, 1.2
    eta1 = phi / alpha_h
    xi1 = phi * gamma_p / c0 ** 2
    a = sigma * phi ** 2 * gamma_p / (c0 ** 2 * rho0 * alpha_h)
    lam1 = numpy.sqrt(-(omega ** 2 * xi1 + 1j * a * omega) / eta1 + 0j)
    if lam1.real < 0:
        lam1 = -lam1
    return eta1 * lam1


def Alpha_code(k):
    """Alpha à donner au code : processing.py utilise la convention opposée."""
    return -compute_alpha(k * c0)


def energie(k, chi):
    """Énergie J pour un nombre d'onde k et une répartition chi."""
    u = processing.solve_helmholtz(domain_omega, spacestep, k, f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
                                   Alpha_code(k) * chi)
    return compute_objective_function(domain_omega, u, spacestep)


def energie_moyenne(chi):
    """Moyenne de J sur toute la plage de la voix (grille régulière)."""
    return numpy.mean([energie(k, chi) for k in k_plage])


# =====================================================================
# MISE EN PLACE (comme dans le code fourni)
# =====================================================================
M = 2 * N
spacestep = 1.0 / N
beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)
f_dir[:, :] = 0.0
f_dir[0, 0:N] = 1.0                       # onde plane venant du haut

mur = (domain_omega == _env.NODE_ROBIN)
chi_nu = numpy.zeros((M, N))              # mur nu : pas de poreux
chi_plein = mur.astype(float)             # mur entièrement couvert
facteur_surface = mur.sum() / N           # longueur du mur / largeur (1 pour un mur plat)

# grille de fréquences pour mesurer la performance
k_plage = numpy.linspace(2 * numpy.pi * f_min / c0, 2 * numpy.pi * f_max / c0, 40)


# =====================================================================
# ÉTAPE 1 : fréquences de résonance du mur nu
# =====================================================================
print('Étape 1 : recherche des résonances du mur nu...')
k_fin = numpy.linspace(k_plage[0], k_plage[-1], 200)
J_nu = numpy.array([energie(k, chi_nu) for k in k_fin])
pics = [i for i in range(1, len(k_fin) - 1) if J_nu[i] > J_nu[i - 1] and J_nu[i] > J_nu[i + 1]]
pics = sorted(pics, key=lambda i: -J_nu[i])[:nb_resonances]     # les plus gros pics
k_resonances = sorted(k_fin[pics])
print('   résonances retenues (Hz) :', [round(k * c0 / (2 * numpy.pi)) for k in k_resonances])


# =====================================================================
# ÉTAPE 2 et 3 : optimisation et performance pour chaque beta
# =====================================================================
J_ref = energie_moyenne(chi_nu)           # référence : le mur nu
resultats_beta = [0.0]
resultats_dB = [0.0]
resultats_chi = [chi_nu]

for beta in betas:
    print(f'Étape 2 : optimisation pour beta = {beta} ...')
    chi0 = preprocessing.set2zero(preprocessing._set_chi(M, N, x, y), domain_omega)  # départ du code
    chi_opt, energy, u_list, grad = our_optimization_procedure_multiple(
        domain_omega, spacestep, k_resonances, f, f_dir, f_neu, f_rob,
        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
        [Alpha_code(k) for k in k_resonances], 5.0, chi0, beta, numb_iter=numb_iter)
    gain = 10 * numpy.log10(J_ref / energie_moyenne(chi_opt))
    print(f'   beta = {beta} : réduction moyenne = {gain:.1f} dB')
    resultats_beta.append(beta)
    resultats_dB.append(gain)
    resultats_chi.append(chi_opt)

gain_plein = 10 * numpy.log10(J_ref / energie_moyenne(chi_plein))
resultats_beta.append(1.0)
resultats_dB.append(gain_plein)
resultats_chi.append(chi_plein)


# =====================================================================
# ÉTAPE 4 : tableau et courbe
# =====================================================================
cout = [b * facteur_surface * prix_m2 for b in resultats_beta]   # € de matériau par m² de mur
print('\n beta | coût (€/m² de mur) | réduction (dB) | % du mur entièrement couvert')
for b, c, g in zip(resultats_beta, cout, resultats_dB):
    print(f' {b:4.2f} | {c:18.1f} | {g:14.1f} | {100 * g / gain_plein:5.0f} %')

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(resultats_beta, resultats_dB, 'o-')
ax.set_xlabel('β : part du mur couverte de poreux')
ax.set_ylabel('Réduction du bruit par rapport au mur nu (dB)')
ax.set_title(f'Coût - performance, mélamine, mur {"plat" if level == 0 else "préfractal niveau " + str(level)}')
ax.grid(alpha=0.3)
ax2 = ax.secondary_xaxis('top', functions=(lambda b: b * facteur_surface * prix_m2,
                                           lambda c: c / (facteur_surface * prix_m2)))
ax2.set_xlabel('Coût du matériau (€ par m² de mur)')
plt.tight_layout()
plt.savefig(f'cout_performance_level{level}.png', dpi=130)
print(f'\nFigure enregistrée : cout_performance_level{level}.png')

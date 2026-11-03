# -*- coding: utf-8 -*-
"""
Courbe coût - performance pour un matériau et une forme de mur donnés.

Principe (simple) :
  1. on trouve les fréquences de résonance du mur NU (les plus gros pics de J(f)) ;
  2. pour chaque quantité de poreux beta, on optimise la répartition chi
     sur ces fréquences (somme des J, fonction our_optimization_procedure) ;
  3. on mesure la performance = gain en dB par rapport au mur ENTIÈREMENT couvert,
     à partir de l'énergie moyenne sur toute la plage de la voix ;
     (on ne prend pas le mur nu comme référence : ses résonances sont des pics
      quasi infinis, et leur moyenne dépend trop de la grille de fréquences) ;
  4. on trace la performance en fonction de beta et du coût.

Le coefficient alpha(omega) du matériau est calculé avec compute_alpha.py (fichier du cours).
"""

import contextlib
import io
import numpy
import matplotlib.pyplot as plt

import _env
import preprocessing
import processing
import compute_alpha
from demo_control_polycopie2024 import our_optimization_procedure_multiple, compute_objective_function


# =====================================================================
# PARAMÈTRES À CHOISIR
# =====================================================================
N = 50                     # points de la grille (domaine de 1 m x 1 m)
level = 0                  # 0 = mur plat, 1 ou 2 = mur préfractal
c0 = 340.0                 # vitesse du son (m/s), la même que dans compute_alpha.py

betas = [0.1, 0.25, 0.4, 0.6, 0.8]   # quantités de poreux testées (beta = 0 et 1 sont ajoutés)
nb_resonances = 5          # nombre de fréquences de résonance utilisées pour optimiser
numb_iter = 20             # itérations de la descente de gradient par beta

f_min, f_max = 270.0, 1600.0         # plage de la voix étudiée (Hz)
prix_m2 = 36.0             # prix du matériau (€/m²), environ, mélamine achat grand public

# matériau (porosité, résistivité, tortuosité) : mélamine
material_name = 'Melamine'
material = {'phi': 0.99, 'sigma': 14000.0, 'alpha_h': 1.02}


# =====================================================================
# FONCTIONS UTILES
# =====================================================================
alphas_calcules = {}        # pour ne pas recalculer alpha deux fois à la même fréquence


def Alpha_code(k):
    """Alpha à donner au code pour le nombre d'onde k.

    compute_alpha.py donne alpha dans la convention du poly (dn u + alpha u = 0) ;
    processing.py utilise la convention opposée, d'où le signe moins.
    """
    if k not in alphas_calcules:
        with contextlib.redirect_stdout(io.StringIO()):          # compute_alpha affiche beaucoup de texte
            alpha, erreur = compute_alpha.compute_alpha(k * c0, material, material_name)
        alphas_calcules[k] = -alpha
    return alphas_calcules[k]


def energie(k, chi):
    """Énergie J pour un nombre d'onde k et une répartition chi."""
    A = Alpha_code(k) if numpy.any(chi) else 0.0                 # mur nu : pas besoin d'alpha
    with contextlib.redirect_stdout(io.StringIO()):
        u = processing.solve_helmholtz(domain_omega, spacestep, k, f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, A * chi)
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
k_plage = numpy.linspace(2 * numpy.pi * f_min / c0, 2 * numpy.pi * f_max / c0, 60)


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
print('Calcul de alpha(omega) avec compute_alpha.py...')
Alphas_resonances = [Alpha_code(k) for k in k_resonances]
for k in k_plage:
    Alpha_code(k)

J_plein = energie_moyenne(chi_plein)      # référence : le mur entièrement couvert (beta = 1)
resultats_beta = []
resultats_dB = []
resultats_chi = []

for beta in betas:
    print(f'Étape 2 : optimisation pour beta = {beta} ...')
    chi0 = preprocessing.set2zero(preprocessing._set_chi(M, N, x, y), domain_omega)  # départ du code
    with contextlib.redirect_stdout(io.StringIO()):                # cache l'affichage de l'optimisation
        chi_opt, energy, u_list, grad = our_optimization_procedure_multiple(
            domain_omega, spacestep, k_resonances, f, f_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
            Alphas_resonances, 5.0, chi0, beta, numb_iter=numb_iter)
    # gain > 0 : moins de bruit qu'en couvrant tout le mur ; gain < 0 : plus de bruit
    gain = 10 * numpy.log10(J_plein / energie_moyenne(chi_opt))
    print(f'   beta = {beta} : gain par rapport au mur plein = {gain:+.1f} dB')
    resultats_beta.append(beta)
    resultats_dB.append(gain)
    resultats_chi.append(chi_opt)

resultats_beta.append(1.0)
resultats_dB.append(0.0)
resultats_chi.append(chi_plein)


# =====================================================================
# ÉTAPE 4 : tableau et courbe
# =====================================================================
cout = [b * facteur_surface * prix_m2 for b in resultats_beta]   # € de matériau par m² de mur
print('\n beta | coût (€/m² de mur) | gain par rapport au mur plein (dB)')
for b, c, g in zip(resultats_beta, cout, resultats_dB):
    print(f' {b:4.2f} | {c:18.1f} | {g:+.1f}')

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
print(f'\nFigure enregistrée : {nom_figure}')

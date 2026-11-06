# -*- coding: utf-8 -*-
"""
Courbes performance - impact carbone pour plusieurs matériaux poreux.

Pour chaque matériau et chaque quantité de poreux beta :
  1. on optimise la répartition chi sur les fréquences de résonance du mur nu ;
  2. performance = gain en dB de l'énergie moyenne sur la plage de la voix,
     par rapport au mur entièrement couvert du matériau de référence ;
  3. impact = kg CO2 eq de matériau pour 1 m de mur.

Le coefficient alpha(omega) est calculé avec compute_alpha.py
"""

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


# parametres
N = 50                     # points de la grille (domaine de 1 m x 1 m)
level = 0                  # 0 = mur plat, 1 ou 2 = mur préfractal
c0 = 340.0                 # vitesse du son (m/s), la même que dans compute_alpha.py
epaisseur = 0.01           # épaisseur du matériau (m), la même que L dans compute_alpha.py

betas = [0.1, 0.25, 0.4, 0.6, 0.8]   # quantités de poreux testées (beta = 1 est ajouté)
nb_resonances = 5          # nombre de fréquences de résonance utilisées pour optimiser
numb_iter = 20             # itérations de la descente de gradient par beta
f_min, f_max = 270.0, 1600.0         # plage de la voix étudiée (Hz)

# Matériaux : propriétés acoustiques (cours) et facteur d'émission (kg CO2 eq par m3, fabrication)
MATERIALS = {
    'Mélamine': {'phi': 0.99, 'sigma': 14000.0, 'alpha_h': 1.02, 'f_emission': 80.4},  # EPD BASF Basotect G
    'Polyuréthane': {'phi': 0.98, 'sigma': 45000.0, 'alpha_h': 2.01, 'f_emission': 3.22 * 35.0},  # Europur : 3,22 kg CO2 eq/kg, 35 kg/m3
    'Pin': {'phi': 0.671, 'sigma': 35540.0, 'alpha_h': 1.22, 'f_emission': 24.0},  # EPD EGGER, résineux séché
}
reference = 'Mélamine'     # le mur plein de ce matériau sert de référence (gain = 0 dB)
PRIX = {
    'Mélamine': 8.5,
    'Polyuréthane': 2.9,
    'Pin': 6.9
}

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
chi_depart = preprocessing.set2zero(preprocessing._set_chi(M, N, x, y), domain_omega)  # départ du code
facteur_surface = mur.sum() / N           # longueur du mur / largeur (1 pour un mur plat)

# fréquences où l'on mesure la performance (nombres d'onde k = omega / c0)
k_plage = numpy.linspace(2 * numpy.pi * f_min / c0, 2 * numpy.pi * f_max / c0, 60)


# =====================================================================
# FONCTIONS UTILES
# =====================================================================
def alpha_code(k, nom):
    """Alpha à donner au code pour le nombre d'onde k et le matériau nom

    compute_alpha.py donne alpha dans la convention du poly (dn u + alpha u = 0) ;
    processing.py utilise la convention opposée, d'où le signe moins
    """
    with contextlib.redirect_stdout(io.StringIO()):          # compute_alpha affiche beaucoup de texte
        alpha, erreur = compute_alpha.compute_alpha(k * c0, MATERIALS[nom], nom)
    return -alpha


def energie(domain, k, chi, A=0.0):
    """Énergie J pour une géométrie domain, un nombre d'onde k, une répartition chi et un coefficient alpha A"""
    u = processing.solve_helmholtz(domain, spacestep, k, f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, A * chi)
    return compute_objective_function(domain, u, spacestep)


def energie_moyenne(domain, chi, alphas):
    """Moyenne de J sur la plage de la voix, avec les alpha calculés pour chaque k de k_plage"""
    return numpy.mean([energie(domain, k, chi, A) for k, A in zip(k_plage, alphas)])


def impact_carbone(nom, beta):
    """kg CO2 eq de matériau pour 1 m de mur"""
    volume = beta * facteur_surface * epaisseur        # m3 de matériau
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
pics = sorted(pics, key=lambda i: -J_nu[i])[:nb_resonances]     # les plus gros pics
k_resonances = sorted(k_fin[pics])
print('   résonances retenues (Hz) :', [round(k * c0 / (2 * numpy.pi)) for k in k_resonances])


# =====================================================================
# ÉTAPE 2 : alpha de chaque matériau, puis référence commune
# =====================================================================
alphas_plage = {nom: [alpha_code(k, nom) for k in k_plage] for nom in MATERIALS}

# référence commune à tous les niveaux : mur plat (niveau 0) entièrement couvert de mélamine
domain_plat, _, _, _, _ = preprocessing._set_geometry_of_domain(M, N, 0)
chi_plein_plat = (domain_plat == _env.NODE_ROBIN).astype(float)
J_ref = energie_moyenne(domain_plat, chi_plein_plat, alphas_plage[reference])


# =====================================================================
# ÉTAPE 3 : optimisation et performance pour chaque matériau et chaque beta
# =====================================================================
betas_tracees = betas + [1.0]
resultats = {}     # nom -> liste des gains en dB (un par beta, beta = 1 en dernier)
for nom in MATERIALS:
    alphas_resonances = [alpha_code(k, nom) for k in k_resonances]
    gains = []
    for beta in betas:
        print(f'Optimisation : {nom}, beta = {beta} ...')
        with contextlib.redirect_stdout(io.StringIO()):            # cache l'affichage de l'optimisation
            chi_opt, energy, u_list, grad = our_optimization_procedure_multiple(
                domain_omega, spacestep, k_resonances, f, f_dir, f_neu, f_rob,
                beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                alphas_resonances, 5.0, chi_depart, beta, numb_iter=numb_iter)
        # gain > 0 : moins de bruit que le mur de référence ; gain < 0 : plus de bruit
        gains.append(10 * numpy.log10(J_ref / energie_moyenne(domain_omega, chi_opt, alphas_plage[nom])))
    # beta = 1 : mur entièrement couvert de ce matériau
    gains.append(10 * numpy.log10(J_ref / energie_moyenne(domain_omega, chi_plein, alphas_plage[nom])))
    resultats[nom] = gains

# =====================================================================
# ÉTAPE 4 : tableau et courbes
# =====================================================================
print('\nmatériau | beta | impact (kg CO2 eq / m de mur) | gain (dB)')
for nom, gains in resultats.items():
    for b, g in zip(betas_tracees, gains):
        print(f'{nom:12s} | {b:4.2f} | {impact_carbone(nom, b):10.3f} | {g:+.1f}')

forme = 'plat' if level == 0 else 'préfractal niveau ' + str(level)


def tracer(abscisses, xlabel, fichier):
    """Trace le gain de chaque matériau ; abscisses(nom) donne la liste des x du matériau"""
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
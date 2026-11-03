# -*- coding: utf-8 -*-
import numpy
import _env
import processing


def compute_objective_function(domain_omega, u, spacestep):
    """J(u) = somme sur les noeuds intérieurs de |u|^2 h^2."""
    mask = (domain_omega == _env.NODE_INTERIOR)
    return numpy.sum(numpy.abs(u[mask]) ** 2) * spacestep ** 2


def our_optimization_procedure_multiple(domain_omega, spacestep, wavenumbers, f, f_dir, f_neu, f_rob,
                               beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                               Alphas, mu, chi, V_obj, numb_iter=60, normaliser=True):
    """Optimisation de la densité chi sur PLUSIEURS fréquences à la fois.

    On minimise J_tot(chi) = somme_i w_i * J(chi, k_i), dont le gradient est la
    somme des gradients de chaque fréquence (méthode du « vote »).

    Paramètres (en plus de ceux de solve_helmholtz) :
        wavenumbers : liste des nombres d'onde k_i (une valeur par fréquence) ;
        Alphas      : liste des coefficients Alpha(k_i), un par fréquence
                      (ou un seul nombre complexe, utilisé pour toutes) ;
                      ATTENTION convention du code : Alpha = -alpha_materiau ;
        mu          : pas initial de la descente de gradient ;
        V_obj       : fraction du mur couverte de poreux (contrainte de volume) ;
        normaliser  : si True, chaque fréquence est divisée par son énergie de départ,
                      pour qu'un pic très fort ne domine pas la somme.
    """
    wavenumbers = list(wavenumbers)
    nf = len(wavenumbers)
    if numpy.isscalar(Alphas):
        Alphas = [Alphas] * nf
    Alphas = list(Alphas)

    (M, N) = numpy.shape(domain_omega)
    robin = (domain_omega == _env.NODE_ROBIN)
    S = numpy.sum(robin)
    V_target = V_obj * S                      # quantité totale de poreux à respecter

    def P_l(chi, l):
        """Troncature dans [0, 1] sur le mur, 0 ailleurs."""
        return numpy.where(robin, numpy.maximum(0, numpy.minimum(1, chi + l)), 0)

    def projection(chi_tmp):
        """Trouve l par dichotomie pour que sum(P_l(chi_tmp, l)) = V_target."""
        lo, hi = -1.0 - numpy.max(chi_tmp), 1.0 - numpy.min(chi_tmp)
        for _ in range(100):
            l = 0.5 * (lo + hi)
            if numpy.sum(P_l(chi_tmp, l)) > V_target:
                hi = l
            else:
                lo = l
            if hi - lo < 1e-10:
                break
        return P_l(chi_tmp, 0.5 * (lo + hi))

    def resoudre(k, Alpha, chi):
        return processing.solve_helmholtz(domain_omega, spacestep, k, f, f_dir, f_neu, f_rob,
                                          beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
                                          Alpha * chi)

    def energies(chi):
        """Liste des énergies J(chi, k_i) et des solutions u_i."""
        us = [resoudre(k, A, chi) for k, A in zip(wavenumbers, Alphas)]
        return [compute_objective_function(domain_omega, u, spacestep) for u in us], us

    # -- poids de chaque fréquence
    chi = projection(chi)
    J0, _ = energies(chi)
    if normaliser:
        poids = [1.0 / J for J in J0]          # chaque fréquence compte autant
    else:
        poids = [1.0] * nf

    def energie_totale(J_list):
        return sum(w * J for w, J in zip(poids, J_list))

    energy = numpy.zeros((numb_iter + 1, 1), dtype=numpy.float64)
    k = 0
    while k < numb_iter and mu > 10 ** (-5):
        print('---- iteration number = ', k)
        grad = numpy.zeros((M, N), dtype=numpy.float64)
        J_list = []
        u_list = []
        for (kw, Alpha, w) in zip(wavenumbers, Alphas, poids):
            # 1. problème direct pour cette fréquence
            u = resoudre(kw, Alpha, chi)
            # 2. problème adjoint pour cette fréquence
            p = processing.solve_helmholtz(domain_omega, spacestep, kw, -2 * numpy.conj(u), 0 * f_dir,
                                           f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu,
                                           beta_rob, Alpha * chi)
            # 3. énergie pour cette fréquence
            J_list.append(compute_objective_function(domain_omega, u, spacestep))
            u_list.append(u)
            # 4. gradient pour cette fréquence, ajouté au total (le « vote »)
            for i in range(M):
                for j in range(N):
                    if robin[i, j]:
                        vals = [numpy.real(Alpha * u[a, b] * p[a, b])
                                for (a, b) in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
                                if 0 <= a < M and 0 <= b < N and domain_omega[a, b] == _env.NODE_INTERIOR]
                        if vals:
                            grad[i, j] += w * numpy.mean(vals)
        energy[k, 0] = energie_totale(J_list)
        print('   énergie totale =', energy[k, 0])

        # -- un seul pas de gradient projeté, avec le gradient total
        ene = energy[k, 0]
        while ene >= energy[k, 0] and mu > 10 ** (-5):
            chi_new = projection(chi - mu * grad)
            J_new, u_new = energies(chi_new)
            ene = energie_totale(J_new)
            if ene < energy[k, 0]:
                mu = mu * 1.1        # l'énergie baisse : on accepte et on agrandit le pas
            else:
                mu = mu / 2          # l'énergie monte : on réduit le pas
        if ene < energy[k, 0]:
            chi = chi_new.copy()
            u_list = u_new
        k += 1

    # énergie finale
    J_fin, u_list = energies(chi)
    energy[k, 0] = energie_totale(J_fin)
    return chi, energy[:k + 1], u_list, grad
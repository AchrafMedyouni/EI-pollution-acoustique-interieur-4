# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import os
import scipy

# MRG packages
import _env
import preprocessing
import processing
import postprocessing
import compute_alpha
#import solutions


def plot_energy_responses(wavenumbers, curves):
    """Trace plusieurs réponses en énergie sur la même figure (échelle log)

    curves : dictionnaire {légende: tableau d'énergies}
    """
    freqs = wavenumbers * 343.0 / (2 * numpy.pi)  # k -> fréquence en Hz (pièce de 1 m)
    matplotlib.pyplot.figure(figsize=(9, 5))
    for label, energies in curves.items():
        matplotlib.pyplot.semilogy(freqs, energies, label=label)
    matplotlib.pyplot.xlabel('Frequency (Hz)')
    matplotlib.pyplot.ylabel('Energy')
    matplotlib.pyplot.title('Energy Response')
    matplotlib.pyplot.grid(True, which='both', alpha=0.3)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()


def our_optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj):
    """This function return the optimized density.

    Parameter:
        cf solvehelmholtz's remarks
        Alpha: complex, it corresponds to the absorbtion coefficient;
        mu: float, it is the initial step of the gradient's descent;
        V_obj: float, it characterizes the volume constraint on the density chi.
    """

    k = 0
    (M, N) = numpy.shape(domain_omega)
    numb_iter = 60
    energy = numpy.zeros((numb_iter+1, 1), dtype=numpy.float64)
    S = numpy.sum(domain_omega == _env.NODE_ROBIN)
    V_target = V_obj * S 
    def P_l(chi, l): 
        return numpy.where(domain_omega == _env.NODE_ROBIN, numpy.maximum(0, numpy.minimum(1, chi + l)), 0)

    while k < numb_iter and mu > 10**(-5):
        print('---- iteration number = ', k)
        print('1. computing solution of Helmholtz problem, i.e., u')
        u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alpha*chi)
        print('2. computing solution of adjoint problem, i.e., p')
        p = processing.solve_helmholtz(domain_omega, spacestep, omega, -2*numpy.conj(u), 0*f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alpha*chi)
        print('3. computing objective function, i.e., energy')
        energy[k] = compute_objective_function(domain_omega, u, spacestep)
        print('4. computing parametric gradient')
        #Alpha = preprocessing.set2zero(Alpha, domain_omega)
        grad = numpy.zeros((M, N), dtype=numpy.float64)
        for i in range(M):
            for j in range(N):
                if domain_omega[i, j] == _env.NODE_ROBIN:
                    vals = [numpy.real(Alpha*u[a, b]*p[a, b])
                            for (a, b) in ((i-1, j), (i+1, j), (i, j-1), (i, j+1))
                            if 0 <= a < M and 0 <= b < N and domain_omega[a, b] == _env.NODE_INTERIOR]
                    if vals:
                        grad[i, j] = numpy.mean(vals)
        ene = energy[k]
        while ene >= energy[k] and mu > 10 ** -5:
            print('    a. computing gradient descent')
            l = 0 
            print('    b. computing projected gradient')
            chi_new = P_l(chi-mu*grad, l)
            S = numpy.sum(domain_omega == _env.NODE_ROBIN)
            beta = V_obj*S
            integral = numpy.abs(numpy.sum(chi_new) - beta) ## a modifier si besoin 
            while integral > 10**(-3):
                if numpy.sum(chi_new)*(spacestep)**2 > beta*(spacestep)**2:
                    l -= integral/S
                else: 
                    l += integral/S
                chi_new = P_l(chi-mu*grad, l)
                integral = numpy.abs(numpy.sum(chi_new) - beta) ## a modifier si besoin 

            print('    c. computing solution of Helmholtz problem, i.e., u')
            u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alpha*chi_new)
            print('    d. computing objective function, i.e., energy (E)')
            ene = compute_objective_function(domain_omega, u, spacestep)
            bool_a = ene < energy[k]
            if bool_a:
                # The step is increased if the energy decreased
                mu = mu * 1.1
            else:
                # The step is decreased is the energy increased
                mu = mu / 2
        if ene < energy[k]:
            chi = chi_new.copy()
        k += 1

    print('end. computing solution of Helmholtz problem, i.e., u')
    u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alpha*chi)
    energy[k] = compute_objective_function(domain_omega, u, spacestep)
    return chi, energy[:k+1], u, grad


def compute_objective_function(domain_omega, u, spacestep):
    """
    This function compute the objective function:
    J(u,domain_omega)= \int_{domain_omega}||u||^2 

    Parameter:
        domain_omega: Matrix (NxP), it defines the domain and the shape of the
        Robin frontier;
        u: Matrix (NxP), it is the solution of the Helmholtz problem, we are
        computing its energy;
        spacestep: float, it corresponds to the step used to solve the Helmholtz
        equation.
    """
    mask = domain_omega == _env.NODE_INTERIOR

    energy = numpy.sum(numpy.real(u[mask]) ** 2 + numpy.imag(u[mask]) ** 2) * (spacestep**2)

    return energy

def compute_energy_response(domain_omega, spacestep, freqs, npoints, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, chi, alphas):
    """
    This function compute the energy response of the system for a given range of
    frequencies.

    Parameter:
        domain_omega: Matrix (NxP), it defines the domain and the shape of the
        Robin frontier;
        spacestep: float, it corresponds to the step used to solve the Helmholtz
        equation;
        omegas: list of floats, it is the range of frequencies for which we want
        to compute the energy response;
        npoints: int, it is the number of points in the frequency range;
        cf solvehelmholtz's remarks.
    """
    wavenumbers = (2.0 * numpy.pi * freqs[0]/340, 2.0 * numpy.pi * freqs[1]/340)
    #alphas = compute_alpha.run_compute_alpha(material, 'Birch LT', npoints, freqs=freqs)
    k = numpy.linspace(wavenumbers[0], wavenumbers[1], npoints)
    energies = numpy.zeros((npoints,), dtype=numpy.float64)
    for i in range(npoints):
        nbr = k[i]
        print('Computing energy response for frequency = ', k[i]*340/(2.0*numpy.pi), 'Hz')
        alpha = alphas[i]
        u = processing.solve_helmholtz(domain_omega, spacestep, nbr, f, f_dir,
                                       f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir,
                                       beta_neu, beta_rob, alpha*chi)
        energies[i] = compute_objective_function(domain_omega, u, spacestep)

    return energies

def recherche_max(energie, frequencies, nmax):
    """
    This function compute the maximum of the energy response of the system for a given range of
    frequencies.

    Parameter:
        energie: list of floats, it is the energy response of the system for
        each frequency in omegas;
        omegas: list of floats, it is the range of frequencies for which we want
        to compute the energy response;
        nmax: int, it is the number of maximums to be found.
    """
    pics = [i for i in range(1, len(energie) - 1) if energie[i] > energie[i-1] and energie[i] > energie[i+1]]
    max_indices = numpy.array(sorted(pics, key=lambda i: -energie[i])[:nmax])
    max_frequencies = frequencies[max_indices]
    max_energies = energie[max_indices]

    return max_frequencies, max_energies


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
    
def plot_energy_response(omegas, energies):
    """
    This function plot the energy response of the system for a given range of
    frequencies.

    Parameter:
        omegas: list of floats, it is the range of frequencies for which we want
        to compute the energy response;
        energies: list of floats, it is the energy response of the system for
        each frequency in omegas.
    """
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(omegas, energies)
    matplotlib.pyplot.xlabel('Frequency (Hz)')
    matplotlib.pyplot.ylabel('Energy')
    matplotlib.pyplot.yscale('log')
    matplotlib.pyplot.title('Energy Response')
    matplotlib.pyplot.grid()
    matplotlib.pyplot.show()

if __name__ == '__main__':

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry


    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumber = 2*numpy.pi*2000/340.0

    N = int(10*wavenumber / (2*numpy.pi)) # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 0 # level of the fractal
    spacestep = 1.0 / N  # mesh size
    # 1/N = 2pi/5k
    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- define boundary conditions
    # planar wave defined on top
    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0
    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)

    # -- define absorbing material
    #Alpha = 15.0 - 25.0 * 1j

    # -- this is the function you have written during your project
    #import compute_alpha
    phi = 0.529  # porosity
    sigma = 151429.0  # resitivity
    alpha_h = 1.37  # tortuosity
    material = {'phi': phi, 'sigma': sigma, 'alpha_h': alpha_h}
    material_name = 'Birch LT'
    Alpha = compute_alpha.compute_single_alpha(material, material_name, wavenumber*340/(2*numpy.pi))
    alpha_rob = Alpha * chi
    #----------------------------------------------------
    # -- set parameters for optimization
    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # initial volume of the domain
    V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
    mu = 5  # initial gradient step
    mu1 = 10**(-4)#10**(-5)  # parameter of the volume functional

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()
    
    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization
    # energy = numpy.zeros((100+1, 1), dtype=numpy.float64)
    # chi, energy, u, grad = our_optimization_procedure(...)
    # chi, energy, u, grad = our_optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
    #                     beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
    #                     Alpha, mu, chi, V_obj)
    # --- en of optimization

    npoints = 1000
    try :
        input_filename = './data/dta_alpha_' + str(material_name) + '.mtx'
        alphas = scipy.io.mmread(input_filename)
    except: 
        alphas = compute_alpha.run_compute_alpha(material, 'Birch LT', npoints)
    # il faut garder en tête que Alpha change avec la fréquemce
    # energy_optimised = compute_energy_response(domain_omega, spacestep, (250, 2000), npoints, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,chi, alphas)
    freqs = numpy.linspace(250, 2000, npoints)
    freqs = numpy.round(freqs, 8)
    #sans chi
    # energy = compute_energy_response(domain_omega, spacestep, (250, 2000), npoints, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, 0*chi, alphas)
    energy_chi = compute_energy_response(domain_omega, spacestep, (250, 2000), npoints, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,0*chi0, alphas)
    # changement du signe des alphas pour avoir la même convention que le cours 
    #alphas = -alphas 
    max_freqs, max_energies = recherche_max(energy_chi, freqs, 5)
    indexes = [freqs.tolist().index(max_freqs[i]) for i in range(len(max_freqs))]
    alphas_max = [alphas[i] for i in indexes]


    chi, energy, u, grad = our_optimization_procedure_multiple(domain_omega, spacestep, [2*numpy.pi*freq/340.0 for freq in max_freqs], f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                        alphas_max, mu, chi, V_obj, numb_iter=60, normaliser=True)
    
    energy_optimised = compute_energy_response(domain_omega, spacestep, (250, 2000), npoints, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, chi, alphas)

    energy_chi = compute_energy_response(domain_omega, spacestep, (250, 2000), npoints, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, chi0, alphas)

    chi_rand, _energy, u, grad = our_optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                        Alpha, mu, chi0.copy(), V_obj)
    energy = compute_energy_response(domain_omega, spacestep, (250, 2000), npoints, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, chi_rand, alphas)

    #Plotting the energy response
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(freqs, energy, label="optimisation sur la fréquence 2000Hz", color='blue')
    matplotlib.pyplot.plot(freqs, energy_optimised, label="optimisation plusieurs fréquences", color='red')
    matplotlib.pyplot.plot(freqs, energy_chi, label="pas d'optimisation", color='green')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel('Frequency (Hz)')
    matplotlib.pyplot.ylabel('Energy')
    matplotlib.pyplot.yscale('log')
    matplotlib.pyplot.title('Energy Response')
    matplotlib.pyplot.grid()
    matplotlib.pyplot.show()


    chin = chi.copy()
    un = u.copy()


    #-- plot chi, u, and energy
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(un, chin)
    err = un - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)

    print('End.')

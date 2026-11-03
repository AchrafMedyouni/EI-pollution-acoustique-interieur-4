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
    energy[-1] = compute_objective_function(domain_omega, u, spacestep)
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

def recherche_max(energie, omegas, nmax):
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
    max_indices = numpy.argsort(energie)[-nmax:][::-1]
    max_frequencies = omegas[max_indices]
    max_energies = energie[max_indices]

    return max_frequencies, max_energies
    
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
    wavenumber = 50.0

    N = int(7*wavenumber / (2*numpy.pi)) # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 3 # level of the fractal
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
    Alpha = 15.0 - 25.0 * 1j
    # -- this is the function you have written during your project
    #import compute_alpha
    #Alpha = compute_alpha.compute_alpha(...)
    alpha_rob = Alpha * chi

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
    phi = 0.529  # porosity
    sigma = 151429.0  # resitivity
    alpha_h = 1.37  # tortuosity
    material = {'phi': phi, 'sigma': sigma, 'alpha_h': alpha_h}
    material_name = 'Birch LT'
    try :
        input_filename = './data/dta_alpha_' + str(material_name) + '.mtx'
        alphas = scipy.io.mmread(input_filename)
    except: 
        alphas = compute_alpha.run_compute_alpha(material, 'Birch LT', npoints)
    # il faut garder en tête que Alpha change avec la fréquemce
    # energy_optimised = compute_energy_response(domain_omega, spacestep, (250, 2000), npoints, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,chi, alphas)
    freqs = numpy.linspace(250, 2000, npoints)
    #sans chi
    # energy = compute_energy_response(domain_omega, spacestep, (250, 2000), npoints, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, 0*chi, alphas)

    # energy_chi = compute_energy_response(domain_omega, spacestep, (250, 2000), npoints, f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,chi0, alphas)

    # max_omegas, max_energies = recherche_max(energy_chi, freqs, 5)

    max_omegas = [428.67867868, 765.01501502, 868.36836837, 824.57457457, 1914.16416416] # a enlever

    print("Max frequencies:", max_omegas)

    # Plotting the energy response
    # matplotlib.pyplot.figure()
    # matplotlib.pyplot.plot(omegas, energy, label="sans Matériaux absorbant", color='blue')
    # # matplotlib.pyplot.plot(omegas, energy_optimised, label="avec Matériaux absorbant", color='red')
    # # matplotlib.pyplot.plot(omegas, energy_chi, label="avec Matériaux absorbant initial", color='green')
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.xlabel('Frequency (Hz)')
    # matplotlib.pyplot.ylabel('Energy')
    # matplotlib.pyplot.yscale('log')
    # matplotlib.pyplot.title('Energy Response')
    # matplotlib.pyplot.grid()
    # matplotlib.pyplot.show()


    chin = chi.copy()
    un = u.copy()


    # -- plot chi, u, and energy
    # postprocessing._plot_uncontroled_solution(u0, chi0)
    # postprocessing._plot_controled_solution(un, chin)
    # err = un - u0
    # postprocessing._plot_error(err)
    # postprocessing._plot_energy_history(energy)

    print('End.')

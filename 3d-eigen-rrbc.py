

"""Dedalus simulation of 3d Rayleigh benard rotating convection

Usage:
    3d-rrbc.py --ra=<rayleigh> --ek=<ekman> --N=<resolution> --max_dt=<Maximum_dt> --init_dt=<Initial_dt> [--pr=<prandtl>] [--mesh=<mesh>]
    3d-rrbc.py -h | --help

Options:
    -h --help   Display this help message
    --ra=<rayliegh>        Rayleigh number
    --ek=<ekman>           Ekman number
    --N=<resolution>       Nx=Ny=2Nz
    --max_dt=<Maximum_dt>  Maximum Time Step
    --init_dt=<Initial_dt> Initial Time Step
    --pr=<prandtl>         Prandtl number [default: 7]
    --mesh=<mesh>          Parallel mesh [default: None]
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as de
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)


# Global parameters
Nz = 64

# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
z_basis = de.Chebyshev('z', Nz, interval=(-1/2, 1/2))
domain = de.Domain([z_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)

# 2D Boussinesq hydrodynamics, with no-slip boundary conditions
# Use substitutions for x and t derivatives

def main(Rayleigh, Ekman, kx_range = [0,3,100], Prandtl = 1):
    
    kx_global = np.linspace(kx_range[0], kx_range[1], kx_range[2])
    problem = de.EVP(domain, variables=['p','T','u','v','w','Tz','uz','wz','vz'], eigenvalue='omega')
    problem.parameters['Pr'] = Prandtl
    problem.parameters['Ra'] = Rayleigh
    problem.parameters['Ek'] = Ekman
    problem.parameters['kx'] = 1
    problem.parameters['ky'] = 0.1
    problem.substitutions['dy(A)'] = "1j*ky*A"
    problem.substitutions['dx(A)'] = "1j*kx*A"
    problem.substitutions['dt(A)'] = "-1j*omega*A"


    problem.add_equation("dx(u) + dy(v) + wz = 0")
    problem.add_equation("dt(T) - (dx(dx(T)) + dy(dy(T)) + dz(Tz)) = w -(u*dx(T) + v*dy(T) + w*Tz)")
    problem.add_equation("dt(u) + dx(p) - Pr*(dx(dx(u)) + dy(dy(u)) + dz(uz)) - (Pr/Ek)*v  = -(u*dx(u) + v*dy(u) + w*uz)")
    problem.add_equation("dt(v) + dy(p) - Pr*(dx(dx(v)) + dy(dy(v)) + dz(vz)) + (Pr/Ek)*u  = -(u*dx(v) + v*dy(v) + w*vz)")
    problem.add_equation("dt(w) + dz(p) - Pr*(dx(dx(w)) + dy(dy(w)) + dz(wz)) - Ra*Pr*T = -(u*dx(w) + v*dy(w) +w*wz)")
    #problem.add_equation("dt(u) + dx(p) - Pr*(dx(dx(u)) + dy(dy(u)) + dz(uz))  = -(u*dx(u) + v*dy(u) + w*uz)") ## Non rotating
    #problem.add_equation("dt(v) + dy(p) - Pr*(dx(dx(v)) + dy(dy(v)) + dz(vz))  = -(u*dx(v) + v*dy(v) + w*vz)") ## Non rotating



    problem.add_equation("Tz - dz(T) = 0")
    problem.add_equation("uz - dz(u) = 0")
    problem.add_equation("vz - dz(v) = 0")
    problem.add_equation("wz - dz(w) = 0")
    problem.add_bc("left(T) = 0")
    problem.add_bc("left(u) = 0")
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(T) = 0")
    problem.add_bc("right(u) = 0")
    problem.add_bc("right(w) = 0")
    problem.add_bc("right(v) = 0")
    problem.add_bc("left(v) = 0")
    solver = problem.build_solver()

    # Create function to compute max growth rate for given kx
    def max_growth_rate(kx):
        # Change kx parameter
        problem.namespace['kx'].value = kx
        # Solve for eigenvalues with sparse search near zero, rebuilding NCCs
        solver.solve_sparse(solver.pencils[0], N=10, target=0, rebuild_coeffs=True)
        # Return largest imaginary part
        return np.max(solver.eigenvalues.imag)

    # Compute growth rate over local wavenumbers
    kx_local = kx_global[CW.rank::CW.size]
    t1 = time.time()
    growth_local = np.array([max_growth_rate(kx) for kx in kx_local])
    t2 = time.time()

    # Reduce growth rates to root process
    growth_global = np.zeros_like(kx_global)
    growth_global[CW.rank::CW.size] = growth_local
    if CW.rank == 0:
        CW.Reduce(MPI.IN_PLACE, growth_global, op=MPI.SUM, root=0)
    else:
        CW.Reduce(growth_global, growth_global, op=MPI.SUM, root=0)


    for idx,lines in enumerate(growth_global):
   
        if lines > 0:
            if CW.rank==0:
                print('Growth rate of mode {:.2f} is {:.2f}.'.format(kx_global[idx], lines))
    return growth_global, kx_global

def check(array):
   
   unstable = False
   indexes = []
   for idx,elements in enumerate(array):
       if elements > 0:
           unstable = True
           indexes.append(idx)
   return unstable,indexes



Ekman = 1e-5

for Rayleigh in np.linspace(2e7, 4e7, 100):
    
    growth_rates, kx = main(Rayleigh = Rayleigh, Ekman = Ekman, kx_range = [0,64,100])
    instability, indexes = check(growth_rates)
    if instability == True:
        if CW.rank==0:
            print('Unstable mode found at {}, with Ra: {:.3e}, Ek: {:.3e}.'.format(indexes[0], Rayleigh, Ekman))
    else:
        if CW.rank==0:
            print('No instabilities found, with Ra: {:.3e}, Ek: {:.3e}.'.format(Rayleigh, Ekman))

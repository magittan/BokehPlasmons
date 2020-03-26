import Plasmon_Modeling as PM
import matplotlib.pyplot as plt
import numpy as np

def main():
    sample = PM.RectangularSample(100,100)
    sample.placeRectangularSource(0,10,5,10,0)

    sigma = PM.S()
    omega = PM.O()
    sigma.set_sigma_values_RQ(4,100)
    omega.set_omega_values(1,1)
    lam = float(10000)
    phi = float(90)*np.pi/180

    sample.eigenvalue_target_solve(20,sigma,number_extracted = 5, to_plot = True, density = 100, _lam=lam, _phi=phi)
    """
    sample.run(omega,sigma,density = int(100), _lam=lam,_phi=phi)

    results = sample.cast_solution_to_Array()

    plt.figure()
    plt.imshow(results[0], origin='lower')
    plt.show()
    """

if __name__ == '__main__': main()

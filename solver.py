import matplotlib.pyplot as plt

class Solver:

    def __init__(self, method = "euler") -> None:
        self._method = method
        self._solver_list = ['euler', 'midpoint', 'rk4', 'adaptive']

    def get_solvers(self) -> list:
        return self._solver_list

    def set_solver(self, type) -> None:
        if type in self._solver_list:
            self._method = type
        else:
            return ValueError

    def solve(self, *args, **kwargs) -> list:

        if self._method == 'euler':
            soln = self.euler_solver(*args, **kwargs)
        elif self._method == 'midpoint':
            soln = self.midpoint_solver(*args, **kwargs)
        elif self._method == 'RK4':
            soln = self.rk4_solver(*args, **kwargs)
        elif self._method == 'adaptive':
            soln = self.adaptive_solver(*args, **kwargs)
        else:
            return ValueError
        return soln

    ######################################## solvers ####################################################

    def euler_solver(self, ode, initial_value:list, integration_limit:int=5, stepsize:float = 0.01) -> list:

        # ODE solver using euler's method with constant step size

        t, y = initial_value
        tf = integration_limit
        iterations = abs(( tf - t ) / stepsize)
        soln = list()
        i = 0
        soln.append((y,t))

        while True: # This loop will run until the number of iterations are completed
            y = y + ode(t, y)*stepsize
            t += stepsize
            soln.append((y,t))
            i+= 1
            if i == iterations:
                break
        return soln

    def midpoint_solver(self, ode, initial_value:list, integration_limit:int=5, stepsize:float = 0.01) -> list:

        # ODE solver using midpoint type second order Runge-Kutta method with constant step size

        t, y = initial_value
        tf = integration_limit
        iterations = abs(( tf - t ) / stepsize)
        soln = list()
        i = 0
        soln.append((y,t))

        while True: # This loop will run until the number of iterations are completed

            k1 = ode(t, y) #K1
            k2 = ode(t+(stepsize/2),y+(k1*stepsize/2)) #K2
            y += (k2*stepsize) #Midpoint formula to update y
            t += stepsize
            i += 1
            soln.append((y,t))
            if i == iterations:
                break
        return soln

    def rk4_solver(self, ode, initial_value:list, integration_limit:int=5, stepsize:float = 0.01):

        # ODE solver using fourth order Runge-Kutta method with constant step size

        t, y = initial_value
        tf = integration_limit
        iterations = abs(( tf - t ) / stepsize)
        soln = list()
        i = 0
        soln.append((y,t))

        while True: # This loop will run until the number of iterations are completed

            k1 = stepsize * ode(t, y)
            k2 = stepsize * ode(t + 0.5*stepsize, y + 0.5*k1)
            k3 = stepsize * ode(t + 0.5*stepsize, y + 0.5*k2)
            k4 = stepsize * ode(t + stepsize, y + k3)

            y += (1.0/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            t += stepsize
            i += 1
            soln.append((y,t))
            if i == iterations:
                break
        return soln


    def adaptive_solver(self, ode, initial_value:list, integration_limit:int=5, stepsize:float = 0.01, tolerance:float = 0.001, iter_limit:int = 100) -> list:

        # Broken !!! variable step size is breaking things, error calculation is WIP
        # Adaptive Runge - Kutta solver including estimate of the local truncation error of a single Runge–Kutta step

        t, y = initial_value
        tf = integration_limit
        iterations = abs(( tf - t ) / stepsize)
        soln = list()
        i = 0
        soln.append((y,t))

        while True: # This loop will run until the number of iterations are completed

            k1 = stepsize * ode(t, y)
            k2 = stepsize * ode(t + 0.5*stepsize, y + 0.5*k1)

            k3 = stepsize * ode(t + 0.5*stepsize, y + 0.5*k2)
            k4 = stepsize * ode(t + stepsize, y + k3)

            y_RK  = y + (1.0/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            y_mid = y + (k2*stepsize)

            iter = 0
            while True:
                if abs(y_RK - y_mid) > tolerance:

                    stepsize = stepsize/2

                    k2 = stepsize * ode(t + 0.5*stepsize, y + 0.5*k1)
                    k3 = stepsize * ode(t + 0.5*stepsize, y + 0.5*k2)
                    k4 = stepsize * ode(t + stepsize, y + k3)

                    y_RK  = y + (1.0/6.0)*(k1 + 2*k2 + 2*k3 + k4)
                    y_mid = y + (k2*stepsize)

                    iter += 1
                elif abs(y_RK - y_mid) <= tolerance or iter > iter_limit:
                    break
                else:
                    exit()

            y = y_RK
            t += stepsize
            i += 1
            soln.append((y,t))
            if i == iterations:
                break
        return soln

############################### End of Class ##################################################

def plot_soln(soln:list, system:bool=False) -> None:
    xlist, ylist = list(), list()
    for obj in soln:
        t,y = obj
        xlist.append(t)
        ylist.append(y)
    plt.plot(ylist, xlist)
    if not system:
        plt.show()

def plot_list(soln:list) -> None:
    for dylist in soln:
        plot_soln(dylist, system=True)
    plt.show()

############ test #############
def main():

    import numpy as np
    def example_ode(t, y) -> float:
        dydt = float(2 - 2*y - np.e**(-4*t))
        return dydt

    sol = Solver()
    sol.set_solver('RK4')
    solsys = []

    soln1 = sol.rk4_solver(example_ode, [0,1], 1)
    #solsys.append(soln1)

    soln2 = sol.adaptive_solver(example_ode, [0,1], 1)
    solsys.append(soln2)

    #solsys = solve_system(odelist=[example_ode, example_ode, example_ode], initlist=[(1,1), (0,1), (-1,2)] )
    plot_list(solsys)

main()

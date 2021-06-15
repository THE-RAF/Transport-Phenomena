from autograd import grad
import autograd.numpy as np


def derivative(func, x):
	grad_f = grad(func)
	return grad_f(x)

def function_zero(func, initial_guess, tolerance=0.0001, maxiter=100):
	x = initial_guess
	x_last = x + 2*tolerance

	i = 0
	while abs(x-x_last) >= tolerance:
		x_last = x
		x = x - func(x)/derivative(func, x)

		i += 1
		if i >= maxiter:
			raise Exception('the method failed.')
			exit()
	return x


if __name__ == '__main__':
	f = lambda x: np.sqrt(x-np.tanh(1-x))+2*np.exp(-x*np.cos(1/x))-x*np.sin(x)

	print(function_zero(f, 7.))

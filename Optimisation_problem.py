#Project in TMA4180 Optimisation
#Given a set of red/blue points in R^2, find the ellipse containing
#as many red points and as few blue points as possible
#using optimisation algorithms.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import line_search

newparams = {'figure.figsize': (8.0, 4.0), 'axes.grid': True,
             'lines.markersize': 8, 'lines.linewidth': 2,
             'font.size': 14}
plt.rcParams.update(newparams)

def make_A(x):
    '''
    Make a 2x2 matrix from x in R^5
    '''
    A = np.zeros((2,2))
    A[0,0] = x[0]
    A[0,1] = x[1]
    A[1,0] = x[1]
    A[1,1] = x[2]
    return A

def make_c(x):
    '''
    Make c (or b) from x
    '''
    c = np.zeros(2)
    c[0] = x[3]
    c[1] = x[4]
    return c

def make_x(A,c):
    '''
    Make x from A and c (or b)
    '''
    A = A.flatten()
    x = np.zeros(5)
    x[0] = A[0]
    x[1] = A[1]
    x[2] = A[-1]
    x[3] = c[0]
    x[4] = c[1]
    return x
    
def r(x,z,version):
    '''
    Calculates r_i for every z_i, given x.
    version = 1,2. Switches between model 1 and 2.
    '''
    A = make_A(x)
    c = make_c(x)
    r_vec = np.zeros(len(z))
    if version==1:
        for i in range(len(z)):
            t = z[i]-c
            r_vec[i] = t.T@A@t-1
    elif version==2:
        for i in range(len(z)):
            r_vec[i] = z[i].T@A@z[i] - z[i].T@c - 1  
    return r_vec

def f(x,z,w,version):
    '''
    version=1,2 switches between f_1 and f_2
    '''
    func = 0
    r_vec = r(x,z,version)
    for i in range(N):
        if w[i]: #red points (a)
            func += max(r_vec[i],0)**2
        else:    #blue points (b)
            func += min(r_vec[i],0)**2
    return func

def gradf(x,z,w,version):
    '''
    Gradient of f
    '''
    gradf_a = np.zeros((2,2))
    gradf_c = np.zeros(2)
    r_vec = r(x,z,version)
    A = make_A(x)
    c = make_c(x)
    for i in range(N):
        if version == 1:
            gradr_a = np.outer(z[i]-c, z[i]-c)
            gradr_c= -2 * A @ (z[i] - c)
        else:
            gradr_a = np.outer(z[i], z[i])
            gradr_c = -z[i]
        if w[i] == 1:
            gradf_a += 2*max(r_vec[i],0)*gradr_a
            gradf_c += 2*max(r_vec[i],0)*gradr_c
        else:
            gradf_a += 2*min(r_vec[i],0)*gradr_a
            gradf_c += 2*min(r_vec[i],0)*gradr_c
    grad = make_x(gradf_a, gradf_c)
    return grad

def search_direction(x,z,w,version):
    '''
    -grad f(x)
    '''
    return -gradf(x,z,w,version)

def step_length(x,z,w,p,version=1):
    '''
    Step length calculated with backtracking Armijo line search
    '''
    alpha = 1.0
    initial_descent = gradf(x,z,w,version).T @ p
    initial_value = f(x,z,w,version)
    n_step = 0
    while f(x+alpha*p,z,w,version) > initial_value + c1*(alpha*initial_descent) and n_step < max_it:
        alpha *= rho
        n_step += 1
    return alpha

def gradient_descent_step(x,z,w,version):
    '''
    One step of gradient descent with Backtracking
    '''
    p = search_direction(x,z,w,version)
    alpha = step_length(x,z,w,p,version)
    x += alpha*p
    return x,p

def gradient_descent(x,z,w,version):
    '''
    Gradient descent method with backtracking
    '''
    not_converged = True
    n_step = 0
    while not_converged:
        n_step += 1
        if (n_step %20) == 0:
            print(n_step)
            print(x[3:])
            print(x[:3])
        x,p = gradient_descent_step(x,z,w,version)
        not_converged = (np.linalg.norm(p) > grad_min) and (n_step < max_n_steps)
    print("||gradf||:", np.linalg.norm(p))
    return x

def conjugate_gradient_step(x,z,w,gradf_old,p,version):
    '''
    One step of Conjugate gradient method with strong Wolfe conditions
    '''
    my_tuple = line_search(f,gradf,x,p,c1=c1,c2=c2,args=(z,w,version))
    alpha = my_tuple[0]
    if alpha == None:
        alpha = step_length(x,z,w,p,version)
    x += alpha*p
    gradf_new = gradf(x,z,w,version)
    beta = (gradf_new.T@gradf_new)/(gradf_old.T@gradf_old)
    p=-gradf_new+beta*p
    gradf_old=gradf_new
    return x, gradf_old, p

def conjugate_gradient(x,z,w,version):
    '''
    Conjugate gradient with strong Wolfe conditions
    '''
    gradf_old=gradf(x,z,w,version)
    p=-gradf_old
    n_step=0
    while np.linalg.norm(gradf_old) > grad_min and n_step < max_n_steps:
        x, gradf_old, p = conjugate_gradient_step(x,z,w,gradf_old,p,version)
        n_step += 1       
        if (n_step %20) == 0:
            print(n_step)
#            print("grad:",gradf_old)
            print(x[3:])
            print(x[:3])
    return x

def step_length_curvature(x,z,w,p,version=1):
    '''
    Finds an alpha satisfying Wolfe conditions. Used for BFGS
    '''
    alpha = 1.0
    amin=0
    amax=np.inf
    initial_descent = gradf(x,z,w,version).T @ p
    initial_value = f(x,z,w,version)
    n_step = 0
    l=f(x+alpha*p,z,w,version)
    r=initial_value + c1*alpha*initial_descent
    while (l > r or gradf(x+alpha*p,z,w,version).T@p<c2*initial_descent) and n_step < max_it:
        if l > r:
            amax=alpha
            alpha=(amin+amax)/2
        else:
            amin=alpha
            if amax==np.inf:
                alpha=2*alpha
            else:
                alpha=(amin+amax)/2
        l=f(x+alpha*p,z,w,version)
        r=initial_value + c1*alpha*initial_descent
        n_step += 1 
    return alpha

def gradient_descent_firststep(x,z,w,version):
    '''
    One step of gradient descent method, used for BFGS.
    '''
    p = search_direction(x,z,w,version)
    alpha = step_length_curvature(x,z,w,p,version)
    gradfk = gradf(x,z,w,version)
    x += alpha*p
    gradfk1 = gradf(x,z,w,version)
    y = gradfk1-gradfk 
    H0 = y.T@(alpha*p)/(y.T@y)*np.eye(5)
    return x, H0
    
def BFGS(x,z,w,version):
    '''
    BFGS with gradient descent
    '''
    x_old, H = gradient_descent_firststep(x,z,w,version)
    gradf_old = gradf(x_old,z,w,version)
    n_step=0
    not_converged=True
    e = []
    while not_converged and n_step < max_n_steps:
        p = -H@gradf_old
        if (gradf_old.T)@p >=0:
            p = -gradf_old
            print("OBS", gradf_old.T@p)
        alpha = step_length_curvature(x_old,z,w,p,version) 
        x_new = x_old + alpha*p
        gradf_new = gradf(x_new,z,w,version)
        s = x_new - x_old 
        y = gradf_new - gradf_old
        gradf_old=gradf_new
        x_old = x_new
        if s.T@y > epsilon*np.linalg.norm(s)*np.linalg.norm(y): 
            rho = 1/(y.T@s)
            Hy=H@y
            H= H-rho*(np.outer(s,H@y.T)+np.outer(Hy,s))+(rho**2*y.T@Hy+rho)*np.outer(s,s)
        n_step += 1
        ei = np.linalg.norm(gradf_old)
        e.append(ei)
        not_converged = ei > grad_min
        if n_step % 20 == 0 :
           print("Iter:", n_step)
    return x_new, e
    
def convergence_CG(N,it,version):
    '''
    Make convergence plot for conjugate gradient
    '''
    x = np.array([1.,-1,5,-1,2])
    z,w,xtest = test1(N, version, perturb=True)
    e = np.zeros(it)
    gradf_old = gradf(x,z,w,version)
    p = - gradf_old
    for i in range(it):
        if gradf_old[0] == 0:
            break
        x,gradf_old,p = conjugate_gradient_step(x,z,w,gradf_old,p,version)
        e[i] = np.linalg.norm(gradf_old)
        if i%20==0:
            print("Iter:",i)
    plt.semilogy(np.arange(0,it,1),e,label="Conjugate gradient")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

def convergence_GD(N,it,version):
    '''
    Make convergence plot for gradient descent
    '''
    x = np.array([1.,-1,5,-1,2])
    z,w,xtest = test1(N, version, perturb=True)
    e = np.zeros(it)
    for i in range(it):
        x,p = gradient_descent_step(x,z,w,version)
        e[i] = np.linalg.norm(p)
        if i%20==0:
            print("Iter:",i)
    plt.semilogy(np.arange(0,it,1),e, label="Gradient descent")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
    
def c(x,i):
    '''
    Constraints
    '''
    if i==1: 
        return (x[0]-gamma1)
    if i==2:
        return (gamma2-x[0])
    if i==3:
        return (x[2]-gamma1)
    if i==4:
        return (gamma2-x[2])
    if i==5:
        l=x[0] * x[2]
        if l < 0:
            s = -1
        else:
            r=gamma1**2+x[1]**2
            s=np.sqrt(l)-np.sqrt(r)
        return s

def c_sum(x):
    '''
    Sum of constraints
    '''
    s=0
    for i in range(1,5+1):
        ci=c(x,i)
        if ci<=0:
            s-=np.inf
        else:
            s+=np.log(ci)
    return s

def B(x,z,w,beta,version):
    '''
    Barrier function
    '''
    return f(x,z,w,version)-beta*c_sum(x)
    
def gradB(x,z,w,beta, version=1):
    '''
    Gradient of Barrier function
    '''
    s=0
    for i in range(1,5+1):
        s+=gradc(x,i)/c(x,i)
    return gradf(x,z,w,version)-beta*s

def gradc(x,i):
    '''
    Gradient of constraints
    '''
    if i == 1:
        return [1,0,0,0,0]
    if i == 2:
        return [-1,0,0,0,0]
    if i == 3:
        return [0,0,1,0,0]
    if i == 4: 
        return [0,0,-1,0,0]
    if i == 5:
        return np.array([1/2*np.sqrt(x[2]/x[0]),-x[1]/(np.sqrt(gamma1**2+x[1]**2)),1/2*np.sqrt(x[0]/x[2]),0,0])

    
def step_length_Barrier(x,z,w,p,beta,version=1):
    '''
    Calculates step length satisfying Armijo conditions
    rejecting directions that are not feasible
    '''
    alpha = 1.0
    initial_descent = gradB(x,z,w,beta,version).T @ p
    initial_value = B(x,z,w,beta,version)
    n_step = 0
    while (B(x+alpha*p,z,w,beta, version) > initial_value + c1*(alpha*initial_descent) or x[0]+alpha*p[0]<0 or x[2]+alpha*p[2]<0) and n_step < max_it:
        alpha *= rho
        n_step += 1 
    return alpha
    
def search_direction_Barrier(x,z,w,beta,version=1):
    '''
    Search direction for the barrier function
    '''
    return -gradB(x,z,w,beta,version)
    
def gradient_descent_Barrier(x,z,w,beta,version=1):
    '''
    Gradient descent for barrier function
    '''
    not_converged = True
    n_step = 0
    e = []
    while not_converged:
        p = search_direction_Barrier(x,z,w,beta,version)
        alpha = step_length_Barrier(x,z,w,p,beta,version)
        x += alpha*p
        n_step += 1
        ei = np.linalg.norm(p)
        e.append(ei)
        not_converged = (ei > grad_min) and (n_step < max_n_steps)
    return x, e

def Barrier(x,z,w,version=1):
    '''
    Runs gradient descent multiple times on barrier function, varying beta
    '''
    beta=1
    n_step=0
    e = []
    while np.linalg.norm(gradB(x,z,w,beta,version)) > grad_min and n_step < max_n_steps:
        x,ei=gradient_descent_Barrier(x,z,w,beta,version)
        beta=0.4*beta
        n_step+=1
        print(n_step)
        e += ei
    return x, e

def test1(N, version, perturb=True):
    '''
    First test case. Ellipse surrounded by blue points
    '''
    grid = [0,5]
    xtest = np.array([.5,0,.5,0,0])
    z,w = general_test(xtest,N,0.5,grid,version,perturb)
    return z,w,xtest

def test2(N, version, perturb=True):
    '''
    Second test case. Solution is a hyperbola
    '''
    grid = [0,5]
    xtest = np.array([.1,.3,.2,0,0])
    z,w = general_test(xtest,N,0.5,grid,version,perturb)
    return z,w,xtest

def test3(N, version, perturb=True):
    '''
    Third test case. Red and blue points completely splitted
    '''
    grid = [0,5]
    xtest = np.array([1,0,1.,10,10])
    z,w = general_test(xtest,N,0.5,grid,1,perturb=False)
    for i in range(N):
        if z[i,0] > 1:
            w[i] = True
            z[i,0] += .5
    return z,w,xtest

def test4(N, version, perturb=True):
    '''
    Fourth test case. blue|red|blue
    '''
    grid = [0,5]
    xmin, xmax = grid
    z = (np.random.uniform(size=(N,2))*(xmax-xmin))-(xmax-xmin)/2
    w = np.array([-1 < z[i,0] < 1 for i in range(N)])
    return z,w,xtest

def general_test(x,N,eps,grid,version=1,perturb=True):
    '''
    Given grid=[0,5]:
    Generates points on [-2.5,2.5]x[-2.5,2.5]
    Gives labels based on the ellipse defined by x
    Perturbs the points if perturb=True
    '''
    xmin, xmax = grid
    z = (np.random.uniform(size=(N,2))*(xmax-xmin))-(xmax-xmin)/2
    r_vec = r(x,z,version)
    w = np.array([r_vec[i] <= 0 for i in range(N)])
    if perturb:
        z += np.random.uniform(size=(N,2))*eps-eps/2
    return z, w

def plot_ellipse(x,z,w,version):
    '''
    Plots the points z and the ellipse defined by x
    '''
    A = make_A(x)
    c = make_c(x)
    fig, ax = plt.subplots()
    M=100
    xmin = np.amin(z[:,0])
    xmax = np.amax(z[:,0])
    ymin = np.amin(z[:,1])
    ymax = np.amax(z[:,1])
    xs = np.linspace(xmin-1,xmax+1,M)
    ys = np.linspace(ymin-1,ymax+1,M)
    X,Y = np.meshgrid(xs,ys)
    Z = np.zeros((M,M))
    
    if version==1:
        for i in range(M):
            for j in range(M):
                zij = np.array([X[i,j],Y[i,j]])-c
                Z[i,j] = zij.T @ A @ zij - 1
    if version==2:
         for i in range(M):
            for j in range(M):
                zij = np.array([X[i,j],Y[i,j]])
                Z[i,j] = zij.T @ A @ zij - zij.T @ c - 1
    
#    CS=ax.contour(X,Y,Z, [0,0.1])
    CS=ax.contour(X,Y,Z, [0])
#    ax.clabel(CS, inline=1, fontsize=10)
    ax.scatter(z[w==1,0],z[w==1,1], color='r')
    ax.scatter(z[w==0,0],z[w==0,1], color='b')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    plt.show()
    
def evaluate_result(x,z,w,version):
    '''
    Prints an evaluation of the result
    '''
    na = len(w[w])
    nb = N-na
    correct_inside = 0
    wrong_inside = 0
    correct_outside = 0
    wrong_outside = 0
    r_vec = r(x,z,version)
    for i in range(N):
        if r_vec[i] <= 0:
            if w[i]:
                correct_inside += 1
            else:
                wrong_inside += 1
        else:
            if w[i]:
                wrong_outside += 1
            else:
                correct_outside += 1
    print("Red points inside:\t", correct_inside, "/", na)
    print("Red points outside:\t", wrong_outside, "/", na)    
    print("Blue points outside:", correct_outside, "/", nb)
    print("Blue points inside:\t", wrong_inside, "/", nb)
    print("Misclassification:", (wrong_inside+wrong_outside),"/",N,"=",np.round((wrong_inside+wrong_outside)/N*100),"%")
    
'''
Set parameters
'''

max_it = 50         #Max iterations of main algorithm
max_n_steps = 200   #Max iterations of line search
c1 = 0.01           #Constant for Wolfe line search
c2 = 0.4            #Constant for Wolfe line search
rho = 0.25          #Decrease parameter in line search
grad_min = 1.0e-5   #Tolerance of error
epsilon = 1.0e-2    #Tolerance for updating H in BFGS
gamma1 = 0.01       #Constraint parameter
gamma2 = 20         #Constraint parameter
N = 200             #Number of points
np.random.seed(0)   #Seed for recreating test cases
version = 1         #Model 1 or 2
x0 = np.array([2.,0,2,-1,1])    #Initial guess. Must be float

'''
Generate test case
'''
z, w, xtest = test1(N,1) #Regular case
#z, w, xtest = test2(N,1) #Hyberbola

'''
Choose algorithm and solve problem
'''
x = gradient_descent(x0.copy(),z,w,version)
#x = conjugate_gradient(x0.copy(),z,w,version)
#x, e = BFGS(x0.copy(),z,w,version)
#x, e = Barrier(x0.copy(),z,w,version)     #NB! Decrease max_it and max_n_step before running Barrier

'''
Plot convergence of BFGS or Barrier
'''
#plt.semilogy(np.arange(0, len(e), 1), e, label="BFGS")
#plt.semilogy(np.arange(0, len(e), 1), e, label="Barrier")

'''
Plot and print result
'''
plot_ellipse(x,z,w,version)
evaluate_result(x,z,w,version)
'''
Run gradient descent and/or conjugate gradient and plot convergence
'''
#convergence_GD(N,max_n_steps,version)
#convergence_CG(N,max_n_steps,version)
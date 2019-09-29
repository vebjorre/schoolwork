import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt


def make_circle_problem(n,nx,PLOT):
    # This python-script uses the following three input parameters:
    #   n       - Number of points.
    #   nx      - Resolution of the plotting.
    #   PLOT    - Boolean variable for plotting.

    # Defining function handles.
    transform_domain = lambda r : 2*r-1
    rad = lambda x1,x2 : np.sqrt(x1**2+x2**2)

    # Initializing essential parameters.
    r = np.linspace(0,1,nx)
    x = transform_domain(r)
    dx = 2/nx
    x1,x2 = np.meshgrid(x,x)

    # Creating the data structure 'problem' in terms of dictionaries.
    problem = {'domain':{'x1':x,'x2':x},'classes':[None,None]}
    group1 = {'mean_rad':0,'sigma':0.1,'prob_unscaled':lambda x1,x2: 0,'prob':lambda x1,x2: 0,'density':0}
    group1['prob_unscaled'] = lambda x,y : np.exp(-(rad(x,y)-group1['mean_rad'])**2/(2*group1['sigma']**2))
    density_group1 = group1['prob_unscaled'](x1,x2)
    int_density_group1 = (dx**2)*sum(sum(density_group1))
    group1['density'] = density_group1/int_density_group1
    group2 = {'mean_rad':0.5,'sigma':0.1,'prob_unscaled':lambda x1,x2: 0,'prob':lambda x1,x2: 0,'density':0}
    group2['prob_unscaled'] = lambda x,y : np.exp(-(rad(x,y)-group2['mean_rad'])**2/(2*group2['sigma']**2))
    density_group2 = group2['prob_unscaled'](x1,x2)
    int_density_group2 = (dx**2)*sum(sum(density_group2))
    group2['density'] = density_group2/int_density_group2
    problem['classes'][0] = group1
    problem['classes'][1] = group2

    # Creating the arrays x1 and x2.
    x1 = np.zeros((n,2))
    x2 = np.zeros((n,2))
    count = 0
    for i in range(0,n):
        count += 1
        N1 = 'x1_'+str(count)+'.png'
        N2 = 'x2_'+str(count)+'.png'
        x1[i,0],x1[i,1] = pinky(problem['domain']['x1'],problem['domain']['x2'],problem['classes'][0]['density'],PLOT,N1)
        x2[i,0],x2[i,1] = pinky(problem['domain']['x1'],problem['domain']['x2'],problem['classes'][1]['density'],PLOT,N2)

    # Creating the data structure 'data' in terms of dictionaries.
    x = np.concatenate((x1[0:n,:],x2[0:n,:]),axis=0)
    y = np.concatenate((np.ones((n,1)),2*np.ones((n,1))),axis=0)
    i = rnd.permutation(2*n)
    data = {'x':x[i,:],'y':y[i]}

    return data, problem


def pinky(Xin,Yin,dist_in,PLOT,NAME):
    # Checking the input.
    if len(np.shape(dist_in)) > 2:
        print("The input must be a N x M matrix.")
        return
    sy,sx = np.shape(dist_in)
    if (len(Xin) != sx) or (len(Yin) != sy):
        print("Dimensions of input vectors and input matrix must match.")
        return
    for i in range(0,sy):
        for j in range(0,sx):
            if dist_in[i,j] < 0:
                print("All input probability values must be positive.")
                return

    # Create column distribution. Pick random number.
    col_dist = np.sum(dist_in,1)
    col_dist /= sum(col_dist)
    Xin2 = Xin
    Yin2 = Yin

    # Generate random value index and saving first value.
    ind1 = gendist(col_dist,1,1,PLOT,NAME)
    ind1 = np.array(ind1,dtype="int")
    x0 = Xin2[ind1]

    # Find corresponding indices and weights in the other dimension.
    A = (x0-Xin)**2
    val_temp = np.sort(A)
    ind_temp = np.array([i[0] for i in sorted(enumerate(A), key=lambda x:x[1])])
    eps = 2**-52
    if val_temp[0] < eps:
        row_dist = dist_in[:,ind_temp[0]]
    else:
        low_val = min(ind_temp[0:2])
        high_val = max(ind_temp[0:2])
        Xlow = Xin[low_val]
        Xhigh = Xin[high_val]
        w1 = 1-(x0-Xlow)/(Xhigh-Xlow)
        w2 = 1-(Xhigh-x0)/(Xhigh-Xlow)
        row_dist = w1*dist_in[:,low_val]+w2*dist_in[:,high_val]
    row_dist = row_dist/sum(row_dist)
    ind2 = gendist(row_dist,1,1,PLOT,NAME)
    y0 = Yin2[ind2]

    return x0,y0


def gendist(P,N,M,PLOT,NAME):
    # Checking input.
    if min(P) < 0:
        print('All elements of first argument, P, must be positive.')
        return
    if (N < 1) or (M < 1):
        print('Output matrix dimensions must be greater than or equal to one.')
        return

    # Normalizing P and creating cumlative distribution.
    Pnorm = np.concatenate([[0],P],axis=0)/sum(P)
    Pcum = np.cumsum(Pnorm)

    # Creating random matrix.
    R = rnd.rand()

    # Calculate output matrix T.
    V = np.linspace(0,len(P)-1,len(P))
    hist,inds = np.histogram(R,Pcum)
    hist = np.argmax(hist)
    T = int(V[hist])

    # Plotting graphs.
    if PLOT == True:
        Pfreq = (N*M*P)/sum(P)
        LP = len(P)
        fig,ax = plt.subplots()
        ax.hist(T,np.linspace(1,LP,LP))
        ax.plot(Pfreq,color='red')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('P-vector Index')
        fig.savefig(NAME)

    return T

def eta(x): #hypothesis function, acts element-wise
    return np.exp(x) / (np.exp(x) + 1)

def etader(x): #The differentiated hypothesis function, acts element-wise
    return np.exp(x) / (np.exp(x) + 1) ** 2

def sigma(Y): #activation function, acts element-wise
    return np.tanh(Y)

def onestep(h,Y0,K0): #The numerical solution after one step of the Euler method with step size h
    Y1 = Y0 + h * sigma(Y0 @ K0)
    return Y1

def Euler(M,h,K,Y0): #The numerical solution after M steps of the Euler method with step size h
    YM = Y0
    for i in range(M):
        YM = onestep(h,YM,K[i]) 
    return YM

def ObjFunc(M,h,K,Y0,C): #The value of the objective function
    Y_M = Euler(M,h,K,Y0)
    J = .5 * np.linalg.norm(eta(Y_M @ K[-1]) - C) ** 2
    return J

def GradCalcNumerical(M,h,K,Y0,C,eps): #The numerical approximation of the gradient of J.
    deltaJ = [None]*M
    deltaJw = np.empty((4,1))
    W = K[-1]
    J1 = ObjFunc(M,h,K,Y0,C)
    
    #Compute the K-variation
    for m in range(M):
        deltaJ_m = np.empty((4,4))
        for i in range(4):
            for j in range(4):
                Km = np.copy(K[m])
                Km[i,j] += eps
                K2 = K.copy()
                K2[m] = Km
                J2 = ObjFunc(M,h,K2,Y0,C)
                deltaJ_m[i,j] = 1 / eps * (J2 - J1)
        deltaJ[m] = deltaJ_m
        
    #Compute the W-variation
    for i in range(4):
        W1 = np.copy(W)
        W1[i] += eps
        K2 = K.copy()
        K2[-1] = W1
        J2 = ObjFunc(M,h,K2,Y0,C)
        deltaJw[i] = 1/eps*(J2-J1)
        
    return deltaJ, deltaJw

def GradCalcAnalytical(M,h,K,Y0,C,N): #The analytical gradient of J.
    deltaJk = [None]*M        #Variation of J with respect to K_i
    deltaJw = np.empty((4,1)) #Variation of J with respect to W
    W = K[-1]
    ab = np.empty((N,1))
    YM = Y0
    Yms = {}
    Yms[0] = onestep(h,Y0,K[0])
    
    #Compute W-variation
    for m in range(1,M): 
        Yms[m] = onestep(h,Yms[m-1],K[m])
    YM = Yms[M-1]
    a = eta(YM @ W) - C
    b = etader(YM @ W)
    for i in range(N):
        ab[i] = a[i] * b[i]
    deltaJw = YM.T @ ab
    
    #Compute K-variation:
    deltaJY = ab @ W.T  #Variation of J_0 with respect to Y_M
    #Variation of Y_M with respect to K_m
    for m in range(M):
        dery = deltaJY + h * ((1 - sigma(Yms[M-1] @ K[M-1]) ** 2) * deltaJY) @ K[M-1].T 
        Ym = Yms[m]
        Km = K[m]
        if m==(M-1):
            derk = h * Ym.T @ ((1 - sigma(Ym @ Km) ** 2) * deltaJY)
            deltaJk[m] = derk
            break
        for j in range(M-2, m, -1):
            Yj = Yms[j]
            Kj = K[j]
            dery = dery + h*((1 - sigma(Yj @ Kj) ** 2) * dery) @ Kj.T
        derk = h * Ym.T @ ((1 - sigma(Ym @ Km) ** 2) * dery)
        deltaJk[m] = derk
        
    return deltaJw, deltaJk

def main(): 
    h=0.1           #The stepsize to be used by the Euler method 
    M=20            #number of steps to be taken (depth of neural network)
    e=0.00001       #differentiation parameter if using GradCalcNumerical(M,h,K,Y0,C,eps)
    tau=0.25        #gradient descent parameter
    TOL=0.5         #tolerance for the residual of the optimisation problem
    MAX_ITER=5000   #maximum number of iterations
    N=200           #number of points
    n=int(N/2)
    nx=100
    data, problem = make_circle_problem(n,nx,False) #generate test data
    print("Learning data generated\nStarting network training\nIterations:")
    Y=data['x']     #points (x^i,y^i), i=1,...,N
    C=data['y']-1   #color (c^i)
    Z=(Y**2)        #z^i and w^i depends on (x^i,y^i), otherwise there will be no correlation with the color C
    Y0=np.concatenate((Y,Z),axis=1) #Y0 = (x^i,y^i,z^i,w^i)
    K0=np.identity(4) #initial guess for K
    K=[None]*(M+1)    #A structure containing values for the M 4x4 matrices K_0,...,K_{M-1} and W
    for i in range(M):
        K[i]=K0       #Set all K[i], i=1,...,N, equal to the 4x4 identity matrix.
    W=np.ones((4,1))  #initial guess for W
    K[M]=W           
    J=ObjFunc(M,h,K,Y0,C) #The residual. The value of the objective function
    teller=0          #counts number of iterations 
    
    while J > TOL and teller < MAX_ITER: #stops when it has reached 5000 iterations or J is within the tolerance
        DJw, DJ = GradCalcAnalytical(M,h,K,Y0,C,N) #Compute the analytical gradient of the objective function J
        #DJw,DJ = GradCalcNumerical(M,h,K,Y0,C,eps) #The numerical approximation of the gradient of J.
        for i in range(M):
            K[i] = K[i] - tau * DJ[i] #updating the structure K
        K[-1] = K[-1] - tau * DJw     #updating W
        J = ObjFunc(M,h,K,Y0,C)       #computing the residual J(K)
        if teller % 100 == 0:         #printing iteration count
            print(teller, "J =", J)
        teller += 1
    print("Finished on iteration", teller, "with J =", J)
    
    #Validation: use the same K but get new data points from make_circle_problem. 
    print("Validation with 10 new data sets:")
    SUM = 0 
    for i in range(10):
        data, problem = make_circle_problem(n,nx,False) #get new data points
        Y = data['x']
        C = data['y'] - 1
        Z = (Y**2)
        Y0 = np.concatenate((Y,Z), axis=1)
        #record the percentage of cases where it classifies the points correctly
        Y_M = Euler(M,h,K,Y0) #Compute Y_M
        egenC = eta(Y_M @ K[-1]) #find the color
        riktig = 0 #counts the cases where it classifies the points correctly
        for j in range(N):
            if abs(C[j] - egenC[j]) < 0.5: #if it classifies the points correctly 
                riktig += 1
        print(i+1, ":", riktig, "/", N, "=", riktig/N*100, "%")
        SUM += riktig
    print("Average: ", "{0:.2f}".format(SUM/(10*N)*100), "%")

#Run the program
main()
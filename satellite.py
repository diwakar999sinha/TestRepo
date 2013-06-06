class SatelliteD(object):
    ''' Satellite(object) - The states are arranged as follows:  [x xdot y ydot]

    '''

    def __init__(self, sde=None, IC=None, **kwargs):
        #### Expected inputs
        self.sde        = sde
        self.IC         = IC

        #### Kwarg Inputs
        self._mu     = kwargs.get('mu', 398601.2)
  
        
    def A(self,X):
        n       =  len(X)
#        #print len(self.f(X))
        m       =  len(self.f(X))
        self._A =  np.zeros((m,n))

        self._A[1,0]   = 3*self._mu*X[0]**2/float((X[0]**2 + X[2]**2)**2.5) - self._mu/float((X[0]**2 + X[2]**2)**1.5)
        self._A[3,0]   = 3*self._mu*X[0]*X[2]/float((X[0]**2 + X[2]**2)**2.5)
        self._A[0,1]   = 1
        self._A[1,2]   = 3*self._mu*X[0]*X[2]/float((X[0]**2 + X[2]**2)**2.5)
        self._A[3,2]   = 3*self._mu*X[2]**2/float((X[0]**2 + X[2]**2)**2.5) - self._mu/float((X[0]**2 + X[2]**2)**1.5)
        self._A[2,3]   = 1

        return self._A    


    def f(self, X):
        dX1 = X[1] 
        dX2 = self.f_1(X) 
        dX3 = X[3]
        dX4 = self.f_2(X)
        return np.array([dX1, dX2, dX3, dX4])

    def f_1(self, X):
        return ((self._mu) * ( -X[0] ) /( (X[0])**2 + (X[2])**2)**(1.5))


    def f_2(self, X):
        return ((self._mu) * ( -X[2] ) / ( (X[0])**2 + (X[2])**2)**(1.5))
    
    def Df(self, X, P):  # X here is the convariance matrix
        
        return np.dot(self.jacobian(X), P) # self._A is the jacobian of the function
    
    def DTf(self, X, P):
        return np.dot(P, self.jacobian(X).T)
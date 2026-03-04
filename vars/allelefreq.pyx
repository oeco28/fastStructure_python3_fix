# cython: language_level=3
import numpy as np
cimport numpy as np
cimport admixprop as ap
from cpython cimport bool
from scipy.special import digamma, gammaln, polygamma
import scipy.optimize as opt
import utils
from functools import reduce
ctypedef np.uint8_t uint8_t
cdef extern from "allelefreq.h":
    void P_update_simple( uint8_t* G, double* zetabeta, double* zetagamma, double* xi, double* beta, double* gamma, double* var_beta, double* var_gamma, long N, long L, long K )
    void P_update_logistic( double* Dvarbeta, double* Dvargamma, double* mu, double* Lambda, double* var_beta, double* var_gamma, double mintol, long L, long K)
cdef class AlleleFreq:
    def __cinit__(self, long L, long K, str prior):
        self.L = L
        self.K = K
        self.prior = prior
        if self.prior=='simple':
            self.beta = np.ones((self.L,self.K))
            self.gamma = np.ones((self.L,self.K))
        elif self.prior=='logistic':
            self.mu = np.zeros((self.L,1))
            self.Lambda = np.ones((self.K,))
            self.mintol = 1e-1
        self.var_beta = np.ones((L,K)) + 0.1*np.random.rand(L,K)
        self.var_gamma = 10*np.ones((L,K)) + 0.1*np.random.rand(L,K)
        self.zetabeta = np.exp(digamma(self.var_beta) - digamma(self.var_beta+self.var_gamma))
        self.zetagamma = np.exp(digamma(self.var_gamma) - digamma(self.var_beta+self.var_gamma))
        self.oldvar_beta = []
        self.oldvar_gamma = []
        self.require()
    cdef copy(self):
        cdef AlleleFreq newinstance
        newinstance = AlleleFreq(self.L, self.K, prior=self.prior)
        newinstance.var_beta = self.var_beta.copy()
        newinstance.zetabeta = self.zetabeta.copy()
        newinstance.var_gamma = self.var_gamma.copy()
        newinstance.zetagamma = self.zetagamma.copy()
        if self.prior=='logistic':
            newinstance.mu = self.mu
            newinstance.Lambda = self.Lambda
        newinstance.require()
        return newinstance
    cdef require(self):
        self.var_beta = np.require(self.var_beta, dtype=np.float64, requirements='C')
        self.var_gamma = np.require(self.var_gamma, dtype=np.float64, requirements='C')
        self.zetabeta = np.require(self.zetabeta, dtype=np.float64, requirements='C')
        self.zetagamma = np.require(self.zetagamma, dtype=np.float64, requirements='C')
        if self.prior=='simple':
            self.beta = np.require(self.beta, dtype=np.float64, requirements='C')
            self.gamma = np.require(self.gamma, dtype=np.float64, requirements='C')
        elif self.prior=='logistic':
            self.mu = np.require(self.mu, dtype=np.float64, requirements='C')
            self.Lambda = np.require(self.Lambda, dtype=np.float64, requirements='C')
    cdef _update_simple(self, np.ndarray[np.uint8_t, ndim=2] G, ap.AdmixProp psi):
        cdef np.ndarray[np.float64_t, ndim=2] zetabeta
        cdef np.ndarray[np.float64_t, ndim=2] zetagamma
        cdef np.ndarray[np.float64_t, ndim=2] xi
        cdef np.ndarray[np.float64_t, ndim=2] beta
        cdef np.ndarray[np.float64_t, ndim=2] gamma
        cdef np.ndarray[np.float64_t, ndim=2] var_beta
        cdef np.ndarray[np.float64_t, ndim=2] var_gamma
        self.var_beta = np.zeros((self.L,self.K),dtype=np.float64)
        self.var_gamma = np.zeros((self.L,self.K),dtype=np.float64)
        self.require()
        zetabeta  = np.ascontiguousarray(self.zetabeta)
        zetagamma = np.ascontiguousarray(self.zetagamma)
        xi        = np.ascontiguousarray(psi.xi)
        beta      = np.ascontiguousarray(self.beta)
        gamma     = np.ascontiguousarray(self.gamma)
        var_beta  = np.ascontiguousarray(self.var_beta)
        var_gamma = np.ascontiguousarray(self.var_gamma)
        P_update_simple(&G[0,0], &zetabeta[0,0], &zetagamma[0,0], &xi[0,0], &beta[0,0], &gamma[0,0], &var_beta[0,0], &var_gamma[0,0], psi.N, self.L, self.K)
        self.var_beta  = var_beta
        self.var_gamma = var_gamma
        if np.isnan(self.var_beta).any():
            self.var_beta = self.oldvar_beta[-1]
        if np.isnan(self.var_gamma).any():
            self.var_gamma = self.oldvar_gamma[-1]
        self.zetabeta  = np.exp(digamma(self.var_beta)  - digamma(self.var_beta+self.var_gamma))
        self.zetagamma = np.exp(digamma(self.var_gamma) - digamma(self.var_beta+self.var_gamma))
        self.require()
    cdef _update_logistic(self, np.ndarray[np.uint8_t, ndim=2] G, ap.AdmixProp psi):
        cdef np.ndarray[np.float64_t, ndim=2] beta
        cdef np.ndarray[np.float64_t, ndim=2] Dvarbeta
        cdef np.ndarray[np.float64_t, ndim=2] Dvargamma
        cdef np.ndarray[np.float64_t, ndim=2] var_beta
        cdef np.ndarray[np.float64_t, ndim=2] var_gamma
        cdef np.ndarray bad_beta, bad_gamma
        cdef np.ndarray[np.float64_t, ndim=2] zetabeta
        cdef np.ndarray[np.float64_t, ndim=2] zetagamma
        cdef np.ndarray[np.float64_t, ndim=2] xi
        beta      = np.zeros((self.L,self.K), dtype=np.float64)
        Dvarbeta  = np.require(self.var_beta.copy(),  dtype=np.float64, requirements='C')
        Dvargamma = np.require(self.var_gamma.copy(), dtype=np.float64, requirements='C')
        zetabeta  = np.ascontiguousarray(self.zetabeta)
        zetagamma = np.ascontiguousarray(self.zetagamma)
        xi        = np.ascontiguousarray(psi.xi)
        P_update_simple(&G[0,0], &zetabeta[0,0], &zetagamma[0,0], &xi[0,0], &beta[0,0], &beta[0,0], &Dvarbeta[0,0], &Dvargamma[0,0], psi.N, self.L, self.K)
        var_beta, var_gamma = self._unconstrained_solver(Dvarbeta, Dvargamma)
        bad_beta  = reduce(utils.OR, [(var_beta<=0),  np.isnan(var_beta)])
        bad_gamma = reduce(utils.OR, [(var_gamma<=0), np.isnan(var_gamma)])
        var_beta[bad_beta]   = self.var_beta[bad_beta]
        var_gamma[bad_gamma] = self.var_gamma[bad_gamma]
        self.var_beta  = var_beta
        self.var_gamma = var_gamma
        self.zetabeta  = np.exp(digamma(self.var_beta)  - digamma(self.var_beta+self.var_gamma))
        self.zetagamma = np.exp(digamma(self.var_gamma) - digamma(self.var_beta+self.var_gamma))
        self.require()
    cdef _unconstrained_solver(self, np.ndarray[np.float64_t, ndim=2] Dvarbeta,
                                     np.ndarray[np.float64_t, ndim=2] Dvargamma):
        cdef np.ndarray[np.float64_t, ndim=2] mu
        cdef np.ndarray[np.float64_t, ndim=2] Lambda
        cdef np.ndarray[np.float64_t, ndim=2] lvar_beta
        cdef np.ndarray[np.float64_t, ndim=2] lvar_gamma
        cdef np.ndarray[np.float64_t, ndim=2] var_beta
        cdef np.ndarray[np.float64_t, ndim=2] var_gamma
        var_beta  = np.require(self.var_beta.copy(),  dtype=np.float64, requirements='C')
        var_gamma = np.require(self.var_gamma.copy(), dtype=np.float64, requirements='C')
        mu        = np.ascontiguousarray(self.mu)
        Lambda    = np.ascontiguousarray(np.atleast_2d(self.Lambda))
        lvar_beta  = np.ascontiguousarray(var_beta)
        lvar_gamma = np.ascontiguousarray(var_gamma)
        P_update_logistic(&Dvarbeta[0,0], &Dvargamma[0,0], &mu[0,0], &Lambda[0,0], &lvar_beta[0,0], &lvar_gamma[0,0], self.mintol, self.L, self.K)
        return lvar_beta, lvar_gamma
    cdef update(self, np.ndarray[np.uint8_t, ndim=2] G, ap.AdmixProp psi):
        if self.prior=='simple':
            self._update_simple(G, psi)
        elif self.prior=='logistic':
            self._update_logistic(G, psi)
    cdef square_update(self, np.ndarray[np.uint8_t, ndim=2] G, ap.AdmixProp psi):
        cdef long step
        cdef bool a_ok
        cdef np.ndarray R_beta, R_gamma, V_beta, V_gamma
        cdef double a
        self.oldvar_beta  = [self.var_beta.copy()]
        self.oldvar_gamma = [self.var_gamma.copy()]
        for step in range(0, 2):
            self.update(G, psi)
            self.oldvar_beta.append(self.var_beta.copy())
            self.oldvar_gamma.append(self.var_gamma.copy())
        R_beta  = self.oldvar_beta[1]  - self.oldvar_beta[0]
        R_gamma = self.oldvar_gamma[1] - self.oldvar_gamma[0]
        V_beta  = self.oldvar_beta[2]  - self.oldvar_beta[1]  - R_beta
        V_gamma = self.oldvar_gamma[2] - self.oldvar_gamma[1] - R_gamma
        a = -1.*np.sqrt(((R_beta*R_beta).sum()+(R_gamma*R_gamma).sum())
                /((V_beta*V_beta).sum()+(V_gamma*V_gamma).sum()))
        if a>-1:
            a = -1.
        a_ok = False
        while not a_ok:
            self.var_beta  = (1+a)**2*self.oldvar_beta[0]  - 2*a*(1+a)*self.oldvar_beta[1]  + a**2*self.oldvar_beta[2]
            self.var_gamma = (1+a)**2*self.oldvar_gamma[0] - 2*a*(1+a)*self.oldvar_gamma[1] + a**2*self.oldvar_gamma[2]
            if (self.var_beta<=0).any() or (self.var_gamma<=0).any():
                a = (a-1)/2.
                if np.abs(a+1)<1e-4:
                    a = -1.
            else:
                a_ok = True
        if np.isnan(self.var_beta).any() or np.isnan(self.var_gamma).any():
            self.var_beta  = self.oldvar_beta[1]
            self.var_gamma = self.oldvar_gamma[1]
        self.zetabeta  = np.exp(digamma(self.var_beta)  - digamma(self.var_beta+self.var_gamma))
        self.zetagamma = np.exp(digamma(self.var_gamma) - digamma(self.var_beta+self.var_gamma))
        self.require()
    cdef update_hyperparam(self, bool nolambda):
        cdef np.ndarray dat, C
        if self.prior=='logistic':
            dat = digamma(self.var_beta)-digamma(self.var_gamma)
            self.mu = utils.insum(self.Lambda*dat,[1]) / self.Lambda.sum()
            diff = dat-self.mu
            if not nolambda:
                C = 1./(self.L) * (utils.outsum(diff**2) + utils.outsum(polygamma(1,self.var_beta)+polygamma(1,self.var_gamma))).ravel()
                self.Lambda = 1./C

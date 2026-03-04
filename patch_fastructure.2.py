#!/usr/bin/env python3
"""
patch_faststructure.py  -  One-shot Python 3 / Cython 3 patch for fastStructure.

Run from the ROOT of a fresh fastStructure clone:

    git clone https://github.com/rajanil/fastStructure
    cd fastStructure
    python patch_faststructure.py

Then build:

    cd vars && python setup.py build_ext --inplace && cd ..
    python setup.py build_ext --inplace

Then run (always use the wrapper so vars/ is on the path):

    bash run_structure.sh  -K 3 --input=test/testdata --output=test/out
    bash run_chooseK.sh    --input=test/out
    bash run_distruct.sh   -K 3 --input=test/out --output=test/out.svg

What this script fixes
----------------------
1.  setup.py (root)
      - distutils -> setuptools
      - Cython.Distutils.build_ext -> Cython.Build.cythonize
      - Adds compiler_directives language_level=3
      - Adds include_path=['vars/'] so .pxd headers are found
      - Adds GSL library linkage

2.  vars/setup.py
      - Same distutils -> setuptools swap
      - Each extension now lists its companion C_*.c file as an
        additional source so the C symbols (P_update_simple etc.)
        are compiled and linked into the .so directly
      - Adds GSL include/library dirs

3.  vars/admixprop.pyx   (complete rewrite of fixed version)
4.  vars/allelefreq.pyx  (complete rewrite of fixed version)

5.  vars/*.pxd + vars/utils.pyx + vars/marglikehood.pyx
      - Adds  # cython: language_level=3

6.  parse_bed.pyx, parse_str.pyx, fastStructure.pyx
      - Adds  # cython: language_level=3
      - print "..."  ->  print("...")
      - Integer division X/Y -> X//Y on assignment lines
      - from functools import reduce  where needed

7.  structure.py, chooseK.py, distruct.py
      - print "..."  ->  print("...")  (all variants including % and >>)

8.  Writes run_structure.sh, run_chooseK.sh, run_distruct.sh wrappers
    that set PYTHONPATH=vars:. so bare  import allelefreq  etc. work
"""

import re
import sys
import os
from pathlib import Path

ROOT = Path(".")
VARS = ROOT / "vars"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def read(p):
    return Path(p).read_text(encoding="utf-8")

def write(p, text):
    Path(p).write_text(text, encoding="utf-8")
    print("  wrote: {}".format(p))

def add_language_level(src):
    if "language_level" not in src:
        return "# cython: language_level=3\n" + src
    return src

def add_functools_reduce(src):
    if "reduce(" not in src or "from functools import reduce" in src:
        return src
    lines = src.splitlines(keepends=True)
    last = max((i for i, l in enumerate(lines)
                if l.lstrip().startswith(("import ", "from ", "cimport "))), default=0)
    lines.insert(last + 1, "from functools import reduce\n")
    return "".join(lines)

def fix_has_key(src):
    # dict.has_key(x)  ->  x in dict
    return re.sub(r'(\w+)\.has_key\(([^)]+)\)', r'\2 in \1', src)

def fix_print(src):
    def rep(m):
        return "{}print({})".format(m.group(1), m.group(2).rstrip())
    # print >> sys.stderr, "msg"
    src = re.sub(r'^([ \t]*)print\s*>>\s*(\S+?),\s*(.+)',
                 lambda m: '{}print({}, file={})\n'.format(
                     m.group(1), m.group(3).rstrip(), m.group(2)),
                 src, flags=re.MULTILINE)
    # print "..." or print "..." % x
    src = re.sub(r'^([ \t]*)print\s+(.+)', rep, src, flags=re.MULTILINE)
    return src

def fix_int_division(src):
    result = []
    for line in src.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith('#'):
            result.append(line)
            continue
        if re.search(r'["\'].*?/.*?["\']', line):
            result.append(line)
            continue
        if '=' in line and re.search(r'\w\s*/\s*\w', line):
            line = re.sub(r'(\w)(\s*)/(\s*)(\w)',
                          lambda m: m.group(1)+m.group(2)+'//'+m.group(3)+m.group(4),
                          line)
        result.append(line)
    return "".join(result)

def patch_simple_pyx(path):
    """Apply language_level, reduce, print and int-div fixes to a .pyx/.pxd file."""
    p = Path(path)
    if not p.exists():
        print("  SKIP (not found): {}".format(path))
        return
    src = read(p)
    orig = src
    src = add_language_level(src)
    src = add_functools_reduce(src)
    src = fix_has_key(src)
    src = fix_print(src)
    src = fix_int_division(src)
    if src != orig:
        write(p, src)
    else:
        print("  no changes: {}".format(path))

def patch_py(path):
    """Fix print statements in .py files."""
    p = Path(path)
    if not p.exists():
        print("  SKIP (not found): {}".format(path))
        return
    src = read(p)
    orig = src
    src = fix_has_key(src)
    src = fix_print(src)
    if src != orig:
        write(p, src)
    else:
        print("  no changes: {}".format(path))

# ---------------------------------------------------------------------------
# File contents written verbatim
# (These are the fully corrected versions developed through iterative fixing)
# ---------------------------------------------------------------------------

VARS_SETUP = r"""from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

gsl_include = os.environ.get('GSL_INCLUDE', '/usr/local/include')
gsl_lib     = os.environ.get('GSL_LIB',     '/usr/local/lib')

include_dirs = [numpy.get_include(), '.', gsl_include]
library_dirs = [gsl_lib]
libraries    = ['gsl', 'gslcblas']

# Each extension includes its companion C_*.c file so that the C symbols
# (Q_update, P_update_simple, P_update_logistic, etc.) are compiled and
# linked directly into the .so — no separate shared library needed.
ext_modules = [
    Extension("utils",
              ["utils.pyx"],
              include_dirs=include_dirs,
              library_dirs=library_dirs,
              libraries=libraries),
    Extension("admixprop",
              ["admixprop.pyx", "C_admixprop.c"],
              include_dirs=include_dirs,
              library_dirs=library_dirs,
              libraries=libraries),
    Extension("allelefreq",
              ["allelefreq.pyx", "C_allelefreq.c"],
              include_dirs=include_dirs,
              library_dirs=library_dirs,
              libraries=libraries),
    Extension("marglikehood",
              ["marglikehood.pyx", "C_marglikehood.c"],
              include_dirs=include_dirs,
              library_dirs=library_dirs,
              libraries=libraries),
]

setup(
    name='fastStructure_vars',
    ext_modules=cythonize(ext_modules,
                          include_path=['.'],
                          compiler_directives={'language_level': '3'}),
)
"""

ROOT_SETUP = r"""from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

gsl_include = os.environ.get('GSL_INCLUDE', '/usr/local/include')
gsl_lib     = os.environ.get('GSL_LIB',     '/usr/local/lib')

include_dirs = [numpy.get_include(), '.', 'vars/', gsl_include]
library_dirs = [gsl_lib]
libraries    = ['gsl', 'gslcblas']

setup(
    name='fastStructure',
    ext_modules=cythonize(
        [
            Extension("parse_bed",
                      ["parse_bed.pyx"],
                      include_dirs=include_dirs,
                      library_dirs=library_dirs,
                      libraries=libraries),
            Extension("parse_str",
                      ["parse_str.pyx"],
                      include_dirs=include_dirs,
                      library_dirs=library_dirs,
                      libraries=libraries),
            Extension("fastStructure",
                      ["fastStructure.pyx"],
                      include_dirs=include_dirs,
                      library_dirs=library_dirs,
                      libraries=libraries),
        ],
        include_path=['.', 'vars/'],
        compiler_directives={'language_level': '3'},
    ),
)
"""

ADMIXPROP_PYX = r"""# cython: language_level=3
import numpy as np
cimport numpy as np
cimport allelefreq as af
from scipy.special import digamma
from cpython cimport bool
import utils
ctypedef np.uint8_t uint8_t
cdef extern from "admixprop.h":
    void Q_update( uint8_t* G, double* zetabeta, double* zetagamma, double* xi, double* new_var, long N, long L, long K )
cdef class AdmixProp:
    def __cinit__(self, long N, long K):
        self.N = N
        self.K = K
        self.alpha = 1./K*np.ones((1,K))
        self.var = np.ones((N,K)) + 0.1*np.random.rand(N,K)
        self.xi = np.exp(digamma(self.var)-digamma(utils.insum(self.var,[1])))
        self.oldvar = []
    cdef copy(self):
        cdef AdmixProp newinstance
        newinstance = AdmixProp(self.N, self.K)
        newinstance.var = self.var.copy()
        newinstance.xi = self.xi.copy()
        newinstance.oldvar = []
        return newinstance
    cdef require(self):
        self.var = np.require(self.var, dtype=np.float64, requirements='C')
        self.xi = np.require(self.xi, dtype=np.float64, requirements='C')
    cdef update(self, np.ndarray[np.uint8_t, ndim=2] G, af.AlleleFreq pi):
        cdef np.ndarray[np.float64_t, ndim=2] zetabeta
        cdef np.ndarray[np.float64_t, ndim=2] zetagamma
        cdef np.ndarray[np.float64_t, ndim=2] xi
        cdef np.ndarray[np.float64_t, ndim=2] var
        self.var = np.zeros((self.N,self.K), dtype=np.float64)
        zetabeta = np.ascontiguousarray(pi.zetabeta)
        zetagamma = np.ascontiguousarray(pi.zetagamma)
        xi = np.ascontiguousarray(self.xi)
        var = np.ascontiguousarray(self.var)
        Q_update(&G[0,0], &zetabeta[0,0], &zetagamma[0,0], &xi[0,0], &var[0,0], self.N, pi.L, self.K)
        self.var = var
        if np.isnan(self.var).any():
            self.var = self.oldvar[-1]
        else:
            self.var = self.alpha + self.var
            self.xi = np.exp(digamma(self.var)-digamma(utils.insum(self.var,[1])))
        self.require()
    cdef square_update(self, np.ndarray[np.uint8_t, ndim=2] G, af.AlleleFreq pi):
        cdef long step
        cdef bool a_ok
        cdef np.ndarray R, V
        cdef double a
        self.oldvar = [self.var.copy()]
        for step in range(0, 2):
            self.update(G, pi)
            self.oldvar.append(self.var.copy())
        R = self.oldvar[1] - self.oldvar[0]
        V = self.oldvar[2] - self.oldvar[1] - R
        a = -1.*np.sqrt((R*R).sum()/(V*V).sum())
        if a>-1:
            a = -1.
        a_ok = False
        while not a_ok:
            self.var = (1+a)**2*self.oldvar[0] - 2*a*(1+a)*self.oldvar[1] + a**2*self.oldvar[2]
            if (self.var<=0).any():
                a = (a-1)/2.
                if np.abs(a+1)<1e-4:
                    a = -1.
            else:
                a_ok = True
        if np.isnan(self.var).any():
            self.var = self.oldvar[1]
        self.xi = np.exp(digamma(self.var)-digamma(utils.insum(self.var,[1])))
        self.require()
"""

ALLELEFREQ_PYX = r"""# cython: language_level=3
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
"""

PARSE_BED_PYX = r"""# cython: language_level=3
import numpy as np
cimport numpy as np
import struct
import sys
cdef dict genomap = dict([('00',0),('01',1),('11',2),('10',3)])
def load(file):
    cdef int n, l, i, Nindiv, Nsnp, Nbytes
    cdef bytes line
    cdef str checkA, checkB, checkC, bytestr
    cdef np.ndarray genotype
    handle = open(file+'.fam','r')
    for i,line in enumerate(handle):
        pass
    Nindiv = i+1
    Nbytes = Nindiv//4+(Nindiv%4>0)*1
    handle = open(file+'.bim','r')
    for i,line in enumerate(handle):
        pass
    Nsnp = i+1
    tobit = lambda x: ''.join([bin(b)[2:].zfill(8)[::-1] for b in struct.unpack('<%sB'%Nbytes, x)])
    genotype = np.zeros((Nindiv,Nsnp),dtype='uint8')
    handle = open(file+'.bed','rb')
    line = handle.read(1)
    checkA = bin(struct.unpack('<B', line)[0])[2:].zfill(8)[::-1]
    line = handle.read(1)
    checkB = bin(struct.unpack('<B', line)[0])[2:].zfill(8)[::-1]
    line = handle.read(1)
    checkC = bin(struct.unpack('<B', line)[0])[2:].zfill(8)[::-1]
    if checkA!="00110110" or checkB!="11011000":
        print("This is not a valid bed file")
        handle.close()
        sys.exit(2)
    for l in range(Nsnp):
        line = handle.read(Nbytes)
        bytestr = tobit(line)
        for n in range(Nindiv):
            genotype[n,l] = genomap[bytestr[2*n:2*n+2]]
    handle.close()
    return genotype
"""

RUN_WRAPPER = """#!/usr/bin/env bash
# Wrapper for {script} — sets PYTHONPATH so bare imports like
# "import allelefreq" resolve to vars/allelefreq.so
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="${{SCRIPT_DIR}}/vars:${{SCRIPT_DIR}}:${{PYTHONPATH}}" \\
    python3 "${{SCRIPT_DIR}}/{script}" "$@"
"""

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    if not (ROOT / "parse_bed.pyx").exists():
        print("ERROR: run this from the root of a fastStructure clone.")
        sys.exit(1)

    print("\n=== Writing setup.py files ===")
    write(ROOT / "setup.py",      ROOT_SETUP)
    write(VARS  / "setup.py",     VARS_SETUP)

    print("\n=== Writing corrected vars/*.pyx ===")
    write(VARS / "admixprop.pyx",  ADMIXPROP_PYX)
    write(VARS / "allelefreq.pyx", ALLELEFREQ_PYX)

    print("\n=== Writing corrected root *.pyx ===")
    write(ROOT / "parse_bed.pyx", PARSE_BED_PYX)

    print("\n=== Patching vars/*.pxd and simple .pyx files ===")
    for name in ["admixprop.pxd", "allelefreq.pxd",
                 "marglikehood.pxd", "utils.pxd",
                 "utils.pyx", "marglikehood.pyx"]:
        patch_simple_pyx(VARS / name)

    print("\n=== Patching root *.pyx ===")
    for name in ["parse_bed.pyx", "parse_str.pyx", "fastStructure.pyx"]:
        patch_simple_pyx(ROOT / name)

    print("\n=== Patching Python scripts ===")
    for name in ["structure.py", "chooseK.py", "distruct.py"]:
        patch_py(ROOT / name)

    print("\n=== Writing run wrappers ===")
    for script in ["structure.py", "chooseK.py", "distruct.py"]:
        stem = script.replace(".py", "")
        wrapper = ROOT / "run_{}.sh".format(stem)
        write(wrapper, RUN_WRAPPER.format(script=script))
        wrapper.chmod(0o755)

    # Remove stale eggs that shadow local .so files
    print("\n=== Removing stale fastStructure_vars eggs from site-packages ===")
    import shutil, site
    for sp in site.getsitepackages():
        sp = Path(sp)
        for egg in sp.glob("fastStructure_vars*.egg"):
            print("  removing: {}".format(egg))
            shutil.rmtree(egg) if egg.is_dir() else egg.unlink()
        pth = sp / "easy-install.pth"
        if pth.exists():
            lines = pth.read_text().splitlines()
            cleaned = [l for l in lines if "fastStructure_vars" not in l]
            if len(cleaned) != len(lines):
                pth.write_text("\n".join(cleaned) + "\n")
                print("  cleaned easy-install.pth in {}".format(sp))

    print("""
=== All patches applied! ===

Build steps:
  cd vars && python setup.py build_ext --inplace && cd ..
  python setup.py build_ext --inplace

Run:
  bash run_structure.sh  -K 3 --input=test/testdata --output=test/out --full --seed=100
  bash run_chooseK.sh    --input=test/out
  bash run_distruct.sh   -K 3 --input=test/out --output=test/out.svg
""")

if __name__ == "__main__":
    main()

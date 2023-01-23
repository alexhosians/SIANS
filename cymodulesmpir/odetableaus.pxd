cdef int SelectSolver(str method, double[:,:] A, double[:] C, double[:] B, double[:] E, double[:] params, double u)
cdef int Getstages(str method, double[:] u)
cdef int ArrayChecks(double[:,:] A, double[:] C, double[:] B, double[:] E, int n_stages)

cdef int RungeKutta4(double[:,:] A, double[:] C, double[:] B, double[:] E)

cdef int DormandPrince54(double[:,:] A, double[:] C, double[:] B, double[:] E)
cdef int Tsitouras54(double[:,:] A, double[:] C, double[:] B, double[:] E)

cdef int Verner65(double[:,:] A, double[:] C, double[:] B, double[:] E)

cdef int Verner76(double[:,:] A, double[:] C, double[:] B, double[:] E)
cdef int Verner76_dense8(double[:,:] A, double[:] C, double[:] B, double[:] E, double u)

cdef int DormandPrince86(double[:,:] A, double[:] C, double[:] B, double[:] E)
cdef int Verner87(double[:,:] A, double[:] C, double[:] B, double[:] E)

cdef int Verner98(double[:,:] A, double[:] C, double[:] B, double[:] E)
cdef int Tsitouras98(double[:,:] A, double[:] C, double[:] B, double[:] E)
cdef int Verner98_dense8(double[:,:] A, double[:] C, double[:] B, double[:] E, double u)
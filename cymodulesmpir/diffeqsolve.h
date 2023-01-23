extern int ode_solver_2body(double t, double* y, double* dfdt, double* params_input, double* semiaxes, double itol[3], double *vertices1D,
	int *face_ids, double* moment_inertia, int triggers[3]);

extern int ode_solver_Nbody(double t, double* y, double* dfdt, double* params_input, double* semiaxes, double itol[3], double *vertices1D,
  	int *face_ids, double *moment_inertia, int triggers[3]);
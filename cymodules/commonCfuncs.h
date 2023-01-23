extern double absolute_value_vector_two(double r1[3], double r2[3]);
extern double absolute_value_vector(double r[3]);
extern double dot_product(double r1[3], double r2[3]);
extern int cross_product(double r1[3], double r2[3], double cr[3]);

extern double Matrix_determinant_3x3(double A[3][3]);
extern void Rotation_matrix_components(double phi, double theta, double psi, double R_matrix[3][3], int transpose);
extern int Rotation_matrix_Euler_param(double e0, double e1, double e2, double e3, double R_matrix[3][3], int transpose);

extern double angular_acceleration_ode(double a, double b, double c, double rho, double p1, double p2, double torq, int comp);
extern double moment_intertia_ellipsoid(double a, double b, double c, double rho, int i);

extern int Grav_rot_matrix(double phi_1, double theta_1, double psi_1, double phi_2, double theta_2, double psi_2, double Rot_grav[3][3]);
extern int Grav_rot_matrix_euler(double e0_A, double e1_A, double e2_A, double e3_A, 
								 double e0_B, double e1_B, double e2_B, double e3_B, double Rot_grav[3][3]);

extern int Step_solution_RK_pointer(double *y_new, double *y, double **K, double *B, int n_stages, int numvars, double dt);

extern int Rotation_matrix_components_new(double phi, double theta, double psi, double R_matrix[3][3], int transpose, int trigger);
extern double angular_speed_ode_new(double p, double q, double r, double phi, double theta, double psi, int comp, int trigger);
extern double angular_speed_ode_new_v2(double p, double q, double r, double phi, double theta, double psi, double res[3], int trigger);
extern int moment_intertia_tetrahedron(double rho, double volume, double xc, double yc, double zc, double **vertices,
									   double I[3][3], int nfaces, int body);
extern int moment_inertia_polyhedron(double *vertices, int* index_combo, int N_faces, int N_vertices, double mass, int id_fac, double I[9]);
extern int * Face_id_combinations(double* vertices, int N_vertices, int * N_faces);
extern int ellipsoid_intersect_check(double semiaxes[6], double input_values[11], double positions[6], int eulerparam);
extern int polyhedron_sphere_intersection_simple(double **vertices, int n_vertices, double pos_B[3], double radius_B);
extern double potential_spheroid(double x_sq, double y_sq, double z_sq, double a_sq, double c_sq, double mb, double ka, double a, double c);
extern double potential_spheroid_oblate(double x_sq, double y_sq, double z_sq, double a_sq, double c_sq, double ka, double c);
extern double potential_spheroid_prolate(double x_sq, double y_sq, double z_sq, double a_sq, double c_sq, double ka, double c);
extern double potential_sphere(double x, double y, double z);

extern double get_potential_ellipsoid(double x_sq, double y_sq, double z_sq, double a_sq, double b_sq, double c_sq, double kappa);
extern double potential_ellipsoid(double x_sq, double y_sq, double z_sq, double a_sq, double b_sq, double c_sq, double kappa);

extern double kappa_value(double a_sq, double c_sq, double x_sq, double y_sq, double z_sq);
extern double kappa_ellipsoid(double a_sq, double b_sq, double c_sq, double x_sq, double y_sq, double z_sq);

extern double grav_field_ellipsoid(double a_sq, double b_sq, double c_sq, double kappa, int component);
extern int get_gravfield_ellipsoid(double a_sq, double b_sq, double c_sq, double kappa, double g_field[3]);

extern double potential_polyhedron(double r[3], double **vertices, int nfacesB, int* face_ids_other);
extern double grav_field_polyhedron(double r[3], double **vertices, int nfacesB, int* face_ids_other, double g_field[3], int use_omp);

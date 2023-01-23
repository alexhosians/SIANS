extern double Force_ellipsoid(double input[11], double itol[3], double semiaxes[6], double **vertices, int FM_check,
  int component, int eulerparam, double prefac, int* face_ids_other, int vertexids[7]);

extern double Force_polyhedron(double input[11], double itol[3], double semiaxes[6], double **vertices,
 int FM_check, int component, int eulerparam, int vertex_combo[3], double prefac, int vertexids[7], int *face_ids_other);

extern int Force_point_mass(double input[11], double semiaxes[6], int eulerparam, double **vertices, int vertexids[7],
 int *face_ids_other, double result[3], double mass_array[4]);

extern double mutual_potential_ellipsoid(double limits[4], double semiaxes[6], double input_values[11],
 double itols[3], double *vertices1D, int vertexids[6], int* face_ids, double massdens[4], int eulerparam);

extern double mutual_potential_polyhedron(double semiaxes[6], double input_values[11], double itols[3], double *vertices1D, 
	 int vertex_combo[3], double massdens[4], int* face_ids, int vertexids[6], int eulerparam);

extern double mutual_potential_point_mass(double semiaxes[6], double input_values[11], double *vertices1D,
 double massdens[4], int* face_ids_other, int vertexids[6], int eulerparam);

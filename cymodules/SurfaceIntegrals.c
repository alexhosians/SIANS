#define _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include "commonCfuncs.h"
#include "potentials.h"
#include <omp.h>

// Strucutres required to send data through surface integrals.
struct F_integration_params2{
  // Datatypes for polyhedron integration
  double u, prefac;
  double xc, yc, zc;
  double a_B, b_B, c_B;
  int FM_check, component, nvertices;
  double Rot_A[3][3];
  double Rot_B_T[3][3];
  double **vertices;
  int *face_ids_other;
  int vertex_combo[4];
  int nfaces_arr[4];
  int bodyA_faces, bodyB_faces;
};

struct F_integration_params1{
  // Datatypes for polyhedron integration
  double prefac;
  double xc, yc, zc;
  double a_B, b_B, c_B;
  int FM_check, component, nvertices;
  double Rot_A[3][3];
  double Rot_B_T[3][3];
  double itols[3];
  double **vertices;
  int *face_ids_other;
  int vertex_combo[4];
  int nfaces_arr[4];
  int bodyA_faces, bodyB_faces;
};

struct ellip_integration_params2{
  // Datatypes for ellipsoid integration
  double z, prefac;
  double xc, yc, zc;
  double a_A, b_A, c_A;
  double a_B, b_B, c_B;
  int FM_check, component;
  double Rot_A[3][3];
  double Rot_B_T[3][3];
  double **vertices;
  int *face_ids_other;
  int nvertices_A, nvertices_B, nfaces_B, bodyB_ID;
};

struct ellip_integration_params1{
  // Datatypes for ellipsoid integraiton
  double prefac;
  double xc, yc, zc;
  double a_A, b_A, c_A;
  double a_B, b_B, c_B;
  int FM_check, component, nfaces;
  double Rot_A[3][3];
  double Rot_B_T[3][3];
  double itols[3];
  double **vertices;
  int *face_ids_other;
  int nvertices_A, nvertices_B, nfaces_B, bodyB_ID;
};

struct potint_tetra_params1{
  // Parameters for integartion over z for the graviational potential
  double t_low;
  double t_up;
  double xc, yc, zc;
  double Rot_A[3][3];
  double Rot_B_T[3][3];
  double Rot_grav[3][3];
  double rho_A, rho_B;
  double iabstol, ireltol, quadval;
  double a_div_c, b_div_c, c_sq;
  double a_B, b_B, c_B, mb;
  double prefac;
  double **vertices;
  int *face_ids_other;
  int nvertices_A, nvertices_B, nfaces_B, bodyB_ID;
};

struct potint_tetra_params2{
  // Parameters for integartion over t for the graviational potential
  double z;
  double xc, yc, zc;
  double Rot_A[3][3];
  double Rot_B_T[3][3];
  double Rot_grav[3][3];
  double rho_A, rho_B;
  double a_div_c, b_div_c, c_sq;
  double a_B, b_B, c_B, mb;
  double prefac;
  double **vertices;
  int *face_ids_other;
  int nvertices_A, nvertices_B, nfaces_B, bodyB_ID;
};

struct potint_ellip_params2{
  // Basic properties of the spheroids
  double u, G_grav, rho_A, rho_B, mB;
  double xc, yc, zc;
  double a_B, b_B, c_B;
  int FM_check, component, nvertices;
  double Rot_A[3][3];
  double Rot_B_T[3][3];
  double Rot_grav[3][3];
  int nvert_A;
  int nvert_B;
  int nfaces_A;
  int nfaces_B;
  int bodyB_ID;
  int total_vertices;
  double **vertices;
  int *face_ids_other;
  int vertex_combo[4];
};

struct potint_ellip_params1{
  // Basic properties of the spheroids
  double G_grav, rho_A, rho_B, mB;
  double xc, yc, zc;
  double a_B, b_B, c_B;
  int FM_check, component, nvertices;
  double Rot_A[3][3];
  double Rot_B_T[3][3];
  double Rot_grav[3][3];
  double itols[3];
  int nvert_A;
  int nvert_B;
  int nfaces_A;
  int nfaces_B;
  int bodyB_ID;
  int total_vertices;
  double **vertices;
  int *face_ids_other;
  int vertex_combo[4];
};


/* 
===========================================
Force and Torque surface integral.
Integrates over an ellipsoid.
===========================================
*/

double Integrate_ellipsoid_surface(double alpha, void *prms){
  // Unpack parameters
  struct ellip_integration_params2 * params = (struct ellip_integration_params2 *) prms;
  double z = (params->z);  
  double xc = (params->xc);
  double yc = (params->yc);
  double zc = (params->zc);
  double a_A = (params->a_A);
  double b_A = (params->b_A);
  double c_A = (params->c_A);
  double a_B = (params->a_B);
  double b_B = (params->b_B);
  double c_B = (params->c_B);
  
  double prefac = (params->prefac);
  int nfaces_B = (params->nfaces_B);
  int nvertices_A = (params->nvertices_A);
  int nvertices_B = (params->nvertices_B);
  int bodyB_ID = (params->bodyB_ID);
  double Rot_A[3][3];
  double Rot_B_T[3][3];
  for (int i = 0; i < 3; i++){
    for (int j = 0; j < 3; j++){
      Rot_A[i][j] = (params->Rot_A[i][j]);
      Rot_B_T[i][j] = (params->Rot_B_T[i][j]);
    }
  }
  int FM_check = (params->FM_check);
  int component = (params->component);

  int* face_ids_other = (int*)malloc(nfaces_B*3*sizeof(int));
  for (int i=0; i<3*nfaces_B; i++){
    face_ids_other[i] = (params->face_ids_other[i]);
  }
  double a_div_c = a_A/c_A;
  double b_div_c = b_A/c_A;
  double c_sq = c_A*c_A;
  double c_min_z = c_sq - z*z;
  double aprime = a_div_c*sqrt(c_min_z);
  double bprime = b_div_c*sqrt(c_min_z);

  double x = aprime*cos(alpha);
  double y = bprime*sin(alpha);

  double RB_ro1 = x*Rot_A[0][0] + y*Rot_A[0][1] + z*Rot_A[0][2] - xc;
  double RB_ro2 = x*Rot_A[1][0] + y*Rot_A[1][1] + z*Rot_A[1][2] - yc;
  double RB_ro3 = x*Rot_A[2][0] + y*Rot_A[2][1] + z*Rot_A[2][2] - zc;
  
  double newx = Rot_B_T[0][0]*RB_ro1 + Rot_B_T[0][1]*RB_ro2 + Rot_B_T[0][2]*RB_ro3;
  double newy = Rot_B_T[1][0]*RB_ro1 + Rot_B_T[1][1]*RB_ro2 + Rot_B_T[1][2]*RB_ro3;
  double newz = Rot_B_T[2][0]*RB_ro1 + Rot_B_T[2][1]*RB_ro2 + Rot_B_T[2][2]*RB_ro3;
  double phi;
  if (nvertices_B > 0){
    double **vertices_in = (double **)malloc(nvertices_B * sizeof(double *));
    for (int i=0; i<nvertices_B; i++){
      vertices_in[i] = (double*)malloc(3 * sizeof(double));
      vertices_in[i][0] = (params->vertices[i+nvertices_A*bodyB_ID][0]);
      vertices_in[i][1] = (params->vertices[i+nvertices_A*bodyB_ID][1]);
      vertices_in[i][2] = (params->vertices[i+nvertices_A*bodyB_ID][2]);
    }
    double r[3];
    r[0] = newx;
    r[1] = newy;
    r[2] = newz;
    phi = (prefac/2)*potential_polyhedron(r, vertices_in, nfaces_B, face_ids_other);
    for (int i=0; i<nvertices_B; i++){
      free(vertices_in[i]);
    }
    free(vertices_in);
  }
  else{
    double x1_sq = newx*newx;
    double y1_sq = newy*newy;
    double z1_sq = newz*newz;
    double a_sq = a_B*a_B;
    double b_sq = b_B*b_B;
    double c_sq = c_B*c_B;
    // Compute potential of ellipsoid
    double ka;
    if (fabs(a_B - b_B) < 1e-15){
      ka = kappa_value(a_sq, c_sq, x1_sq, y1_sq, z1_sq);
      if (a_B > c_B){
        phi = M_PI*a_B*b_B*c_B*prefac*potential_spheroid_oblate(x1_sq, y1_sq, z1_sq, a_sq, c_sq, ka, c_B);
      }
      else if (a_B < c_B){
        phi = M_PI*a_B*b_B*c_B*prefac*potential_spheroid_prolate(x1_sq, y1_sq, z1_sq, a_sq, c_sq, ka, c_B);
      }
      else{
        phi = prefac*potential_sphere(newx, newy, newz);
      }
    } 
    else if (fabs(a_B - c_B) < 1e-15){
      ka = kappa_ellipsoid(a_sq, b_sq, c_sq, x1_sq, y1_sq, z1_sq);
      if (a_B > b_B){
        phi = M_PI*a_B*b_B*c_B*prefac*potential_spheroid_oblate(x1_sq, z1_sq, y1_sq, a_sq, b_sq, ka, b_B);
      }
      else if (a_B < b_B){
        phi = M_PI*a_B*b_B*c_B*prefac*potential_spheroid_prolate(x1_sq, z1_sq, y1_sq, a_sq, b_sq, ka, b_B);
      }
      else{
        phi = prefac*potential_sphere(newx, newy, newz);
      }
    } 
    else if (fabs(b_B - c_B) < 1e-15){
      ka = kappa_ellipsoid(a_sq, b_sq, c_sq, x1_sq, y1_sq, z1_sq);
      if (b_B > a_B){
        phi = M_PI*a_B*b_B*c_B*prefac*potential_spheroid_oblate(y1_sq, z1_sq, x1_sq, b_sq, a_sq, ka, c_B);
      }
      else if (b_B < a_B){
        phi = M_PI*a_B*b_B*c_B*prefac*potential_spheroid_prolate(y1_sq, z1_sq, x1_sq, b_sq, a_sq, ka, c_B);
      }
      else{
        phi = prefac*potential_sphere(newx, newy, newz);
      }
    } 
    else{
      ka = kappa_ellipsoid(a_sq, b_sq, c_sq, x1_sq, y1_sq, z1_sq);
      phi = M_PI*a_B*b_B*c_B*prefac*get_potential_ellipsoid(x1_sq, y1_sq, z1_sq, a_sq, b_sq, c_sq, ka);
    }    
  }
  double integrand, ncomp;
  if (FM_check == 1){
    // Computes the force components
    if(component == 1){
      ncomp = bprime*cos(alpha);
      integrand = phi*ncomp;
    }
    else if(component == 2){
      ncomp = aprime*sin(alpha);
      integrand = phi*ncomp;
    }
    else if(component == 3){
      ncomp = a_div_c*b_div_c*z;
      integrand = phi*ncomp;
    }
  }
  else if (FM_check == 2){
    // Computes the torque components
    if (component == 1){
      ncomp = (aprime/c_sq)*(c_sq - b_A*b_A)*z*sin(alpha);
      integrand = -phi*ncomp;
    }
    else if(component == 2){
      ncomp = -(bprime/c_sq)*(c_sq - a_A*a_A)*z*cos(alpha);
      integrand = -phi*ncomp;
    }
    else if (component == 3){
      ncomp = ((b_A*b_A - a_A*a_A)/c_sq)*c_min_z*cos(alpha)*sin(alpha);
      integrand = -phi*ncomp;
    } 
  }
  else{
    printf("Parameter FM_check not properly set!");
    exit(1);
  }
  free(face_ids_other);
  return integrand;
}

double Ellipsoid_integration_step1(double z, void *prms){
  struct ellip_integration_params1 * params_in = (struct ellip_integration_params1 *) prms;
  struct ellip_integration_params2 params;
  double itol[3];
  itol[0] = (params_in->itols[0]);
  itol[1] = (params_in->itols[1]);
  itol[2] = (params_in->itols[2]);
  params.z = z;
  params.xc = (params_in->xc);
  params.yc = (params_in->yc);
  params.zc = (params_in->zc);
  params.a_A = (params_in->a_A);
  params.b_A = (params_in->b_A);
  params.c_A = (params_in->c_A);
  params.a_B = (params_in->a_B);
  params.b_B = (params_in->b_B);
  params.c_B = (params_in->c_B);
  
  params.nfaces_B = (params_in->nfaces_B);
  params.nvertices_A = (params_in->nvertices_A);
  params.nvertices_B = (params_in->nvertices_B);
  params.bodyB_ID = (params_in->bodyB_ID);
  params.prefac = (params_in->prefac);
  int i, j;
  for (i = 0; i < 3; i++){
    for (j = 0; j < 3; j++){
      params.Rot_A[i][j] = (params_in->Rot_A[i][j]);
      params.Rot_B_T[i][j] = (params_in->Rot_B_T[i][j]);
    }
  }
  params.FM_check = (params_in->FM_check);
  params.component = (params_in->component);
  params.vertices = malloc((params_in->nvertices_B)*sizeof(double*));
  for (i=0; i<(params_in->nvertices_B); i++){
    params.vertices[i] = malloc(3*sizeof(double));
    params.vertices[i][0] = (params_in->vertices[i][0]);
    params.vertices[i][1] = (params_in->vertices[i][1]);
    params.vertices[i][2] = (params_in->vertices[i][2]); 
  }
  params.face_ids_other = malloc((params_in->nfaces_B)*3*sizeof(int));
  for (i=0; i<3*(params_in->nfaces_B); i++){
    params.face_ids_other[i] = (params_in->face_ids_other[i]);
  }
  // Sets up GSL functions
  double result, error;
  gsl_function F;
  F.function = &Integrate_ellipsoid_surface;
  F.params = &params;
  gsl_integration_workspace * w = gsl_integration_workspace_alloc(2000);
  gsl_integration_qag(&F, 0, 2*M_PI, itol[0], itol[1], 2000, itol[2], w, &result, &error);
  gsl_integration_workspace_free(w);

  for (i=0; i<(params_in->nvertices_B); i++){
    free(params.vertices[i]);
  }
  free(params.vertices);
  free(params.face_ids_other);
  return result;
}

double Force_ellipsoid(double input[11], double itol[3], double semiaxes[6], double **vertices, int FM_check,
  int component, int eulerparam, double prefac, int* face_ids_other, int vertexids[7]){
  // Semiaxes of body B
  double a_A = semiaxes[0];
  double b_A = semiaxes[1];
  double c_A = semiaxes[2];
  double a_B = semiaxes[3];
  double b_B = semiaxes[4];
  double c_B = semiaxes[5];
  
  // Difference in centroid
  double xc = input[0];
  double yc = input[1];
  double zc = input[2];
  // Angles and rotation matrices
  double Rot_A[3][3];
  double Rot_B_T[3][3];
  if (eulerparam){
    double e0_A = input[3];
    double e1_A = input[4];
    double e2_A = input[5];
    double e3_A = input[6];
    double e0_B = input[7];
    double e1_B = input[8];
    double e2_B = input[9];
    double e3_B = input[10];
    Rotation_matrix_Euler_param(e0_A, e1_A, e2_A, e3_A, Rot_A, 0);
    Rotation_matrix_Euler_param(e0_B, e1_B, e2_B, e3_B, Rot_B_T, 1); 
  }
  else{
    double phi_A = input[3];
    double theta_A = input[4];
    double psi_A = input[5];
    double phi_B = input[6];
    double theta_B = input[7];
    double psi_B = input[8];
    Rotation_matrix_components(phi_A, theta_A, psi_A, Rot_A, 0);
    Rotation_matrix_components(phi_B, theta_B, psi_B, Rot_B_T, 1); 
  }
  struct ellip_integration_params1 params;
  params.xc = xc;
  params.yc = yc;
  params.zc = zc;
  params.a_A = a_A;
  params.b_A = b_A;
  params.c_A = c_A;
  params.a_B = a_B;
  params.b_B = b_B;
  params.c_B = c_B;
  params.prefac = prefac;
  params.nfaces_B = vertexids[1];
  params.nvertices_A = vertexids[4];
  params.nvertices_B = vertexids[5];
  params.bodyB_ID = vertexids[3];
  params.itols[0] = itol[0];
  params.itols[1] = itol[1];
  params.itols[2] = itol[2];
  params.FM_check = FM_check;
  params.component = component;
  int i, j;
  for (i = 0; i < 3; i++){
    for (j = 0; j < 3; j++){
      params.Rot_A[i][j] = Rot_A[i][j];
      params.Rot_B_T[i][j] = Rot_B_T[i][j];
    }
  }
  params.vertices = malloc(vertexids[5]*sizeof(double*));
  for (i=0; i<vertexids[5]; i++){
    params.vertices[i] = malloc(3*sizeof(double));
    params.vertices[i][0] = vertices[i][0];
    params.vertices[i][1] = vertices[i][1];
    params.vertices[i][2] = vertices[i][2]; 
  } 
  params.face_ids_other = malloc(3*vertexids[1]*sizeof(int));
  for (i=0; i<3*vertexids[1]; i++){
    params.face_ids_other[i] = face_ids_other[i];
  }
  // Sets up GSL functions
  double result;
  gsl_function F;
  F.function = &Ellipsoid_integration_step1;
  F.params = &params;
  //gsl_set_error_handler_off();
  // Fixed Legendre quadrature, faster than qag for low n, but less accurate in some situations
  // WIll give errors in e.g. z-mototion in co-planar motion.
  // Use standard QAG in release version. 
  /*
  const gsl_integration_fixed_type * T = gsl_integration_fixed_legendre;
  gsl_integration_fixed_workspace * w = gsl_integration_fixed_alloc(T, itol[3], -c_A, c_A, 0.0, 0.0);
  gsl_integration_fixed(&F, &result, w);
  gsl_integration_fixed_free (w);
  */
  
  double error;
  gsl_integration_workspace * w = gsl_integration_workspace_alloc(2000);
  gsl_integration_qag(&F, -c_A, c_A, itol[0], itol[1], 2000, itol[2], w, &result, &error);
  gsl_integration_workspace_free(w);

  for (i=0; i<vertexids[5]; i++){
    free(params.vertices[i]);
  }
  free(params.vertices);
  free(params.face_ids_other);
  return result;
}


/* 
===========================================
Force and Torque surface integral.
Integrates over a polyhedron.
===========================================
*/

double Integrate_tetrahedron_surface(double v, void *prms){
  /*
  Integrates over the surface of a tetrahedron.
  Considers a potential from an ellipsoid.
  */
  // Unpack parameters
  int i;
  struct F_integration_params2 * params = (struct F_integration_params2 *) prms;
  double u = (params->u);  
  double xc = (params->xc);
  double yc = (params->yc);
  double zc = (params->zc);
  double a_B = (params->a_B);
  double b_B = (params->b_B);
  double c_B = (params->c_B);
  double prefac = (params->prefac);
  int nvertices = (params->nvertices);

  double Rot_A[3][3];
  double Rot_B_T[3][3];
  for (i = 0; i < 3; i++){
    for (int j = 0; j < 3; j++){
      Rot_A[i][j] = (params->Rot_A[i][j]);
      Rot_B_T[i][j] = (params->Rot_B_T[i][j]);
    }
  }

  int FM_check = (params->FM_check);
  int component = (params->component);
  int bodyA_ID = (params->bodyA_faces);
  int bodyB_ID = (params->bodyB_faces);
  int faces_A = (params->nfaces_arr[0]);
  int faces_B = (params->nfaces_arr[1]);
  int nvertices_A = (params->nfaces_arr[2]);
  int nvertices_B = (params->nfaces_arr[3]);
  int* face_ids_other = (int*)malloc(faces_B*3*sizeof(int));
  for (i=0; i<3*faces_B; i++){
    face_ids_other[i] = (params->face_ids_other[i]);
  }
  int v1 = (params->vertex_combo[0]);
  int v2 = (params->vertex_combo[1]);
  int v3 = (params->vertex_combo[2]);
  
  double x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4;
  double x, y, z;
  double nxt2, nyt2, nzt2;
  int ok_normal_vector = 0;
  while (ok_normal_vector <= 0){
    int outer = (params->vertex_combo[3]);
    x1 = (params->vertices[v1][0]);
    x2 = (params->vertices[v2][0]);
    x3 = (params->vertices[v3][0]);
    y1 = (params->vertices[v1][1]);
    y2 = (params->vertices[v2][1]);
    y3 = (params->vertices[v3][1]);
    z1 = (params->vertices[v1][2]);
    z2 = (params->vertices[v2][2]);
    z3 = (params->vertices[v3][2]);
    x4 = (params->vertices[outer][0]);
    y4 = (params->vertices[outer][1]);
    z4 = (params->vertices[outer][2]);

    x = (1-v)*x1 + v*u*x2 + v*(1-u)*x3;
    y = (1-v)*y1 + v*u*y2 + v*(1-u)*y3;
    z = (1-v)*z1 + v*u*z2 + v*(1-u)*z3;
    double nxt = v*((y2-y3)*(-z1 + u*z2 + (1-u)*z3) - (z2-z3)*(-y1 + u*y2 + (1-u)*y3));
    double nyt = v*((z2-z3)*(-x1 + u*x2 + (1-u)*x3) - (x2-x3)*(-z1 + u*z2 + (1-u)*z3));
    double nzt = v*((x2-x3)*(-y1 + u*y2 + (1-u)*y3) - (y2-y3)*(-x1 + u*x2 + (1-u)*x3));
    double n_norm = sqrt(nxt*nxt + nyt*nyt + nzt*nzt);
    
    double dcheck = nxt*(x4-x1) + nyt*(y4-y1) + nzt*(z4-z1);
    if (dcheck == 0){
      (params->vertex_combo[3]) = outer + 1;
    }
    else{
      if (dcheck < 0){
        nxt2 = nxt;
        nyt2 = nyt;
        nzt2 = nzt;
      }
      else{
        nxt2 = -nxt;
        nyt2 = -nyt;
        nzt2 = -nzt;
      }
      ok_normal_vector = 1;
    }
  }

  double RB_ro1 = x*Rot_A[0][0] + y*Rot_A[0][1] + z*Rot_A[0][2] - xc;
  double RB_ro2 = x*Rot_A[1][0] + y*Rot_A[1][1] + z*Rot_A[1][2] - yc;
  double RB_ro3 = x*Rot_A[2][0] + y*Rot_A[2][1] + z*Rot_A[2][2] - zc;
  
  double newx = Rot_B_T[0][0]*RB_ro1 + Rot_B_T[0][1]*RB_ro2 + Rot_B_T[0][2]*RB_ro3;
  double newy = Rot_B_T[1][0]*RB_ro1 + Rot_B_T[1][1]*RB_ro2 + Rot_B_T[1][2]*RB_ro3;
  double newz = Rot_B_T[2][0]*RB_ro1 + Rot_B_T[2][1]*RB_ro2 + Rot_B_T[2][2]*RB_ro3;
  
  double phi;
  if (faces_B > 0){
    double **vertices_in = (double **)malloc(nvertices_B * sizeof(double *));
    for (i=0; i<nvertices_B; i++){
      vertices_in[i] = (double*)malloc(3 * sizeof(double));
      vertices_in[i][0] = (params->vertices[i+nvertices_A*bodyB_ID][0]);
      vertices_in[i][1] = (params->vertices[i+nvertices_A*bodyB_ID][1]);
      vertices_in[i][2] = (params->vertices[i+nvertices_A*bodyB_ID][2]);
    }
    double r[3];
    r[0] = newx;
    r[1] = newy;
    r[2] = newz;
    phi = (prefac/2)*potential_polyhedron(r, vertices_in, faces_B, face_ids_other);
    for (i=0; i<nvertices_B; i++){
      free(vertices_in[i]);
    }
    free(vertices_in);
  }
  else{
    double x1_sq = newx*newx;
    double y1_sq = newy*newy;
    double z1_sq = newz*newz;
    double a_sq = a_B*a_B;
    double b_sq = b_B*b_B;
    double c_sq = c_B*c_B;

    // Compute potential of ellipsoid
    double ka;
    if (fabs(a_B - b_B) < 1e-15){
      ka = kappa_value(a_sq, c_sq, x1_sq, y1_sq, z1_sq);
      if (a_B > c_B){
        phi = M_PI*a_B*b_B*c_B*prefac*potential_spheroid_oblate(x1_sq, y1_sq, z1_sq, a_sq, c_sq, ka, c_B);
      }
      else if (a_B < c_B){
        phi = M_PI*a_B*b_B*c_B*prefac*potential_spheroid_prolate(x1_sq, y1_sq, z1_sq, a_sq, c_sq, ka, c_B);
      }
      else{
        phi = prefac*potential_sphere(newx, newy, newz);
      }
    } 
    else{
      ka = kappa_ellipsoid(a_sq, b_sq, c_sq, x1_sq, y1_sq, z1_sq);
      phi = M_PI*a_B*b_B*c_B*prefac*get_potential_ellipsoid(x1_sq, y1_sq, z1_sq, a_sq, b_sq, c_sq, ka);
    }    
  }
  double integrand;
  if (FM_check == 1){
    // Computes the force components
    if(component == 1){
      integrand = phi*nxt2;       
    }
    else if(component == 2){
      integrand = phi*nyt2;
    }
    else if(component == 3){
      integrand = phi*nzt2;
    }
    else{
      printf("Parameter component not properly set!");
      exit(1);
    }
  }
  else if (FM_check == 2){
    // Computes the torque components
    if (component == 1){
      integrand = -phi*(nyt2*z - nzt2*y);
    }
    else if(component == 2){
      integrand = -phi*(nzt2*x - nxt2*z);
    }
    else if (component == 3){
      integrand = -phi*(nxt2*y - nyt2*x);
    } 
    else{
      printf("Parameter component not properly set!");
      exit(1);
    }
  }
  else{
    printf("Parameter FM_check not properly set!");
    exit(1);
  }
  free(face_ids_other);
  return integrand;
}

double Tetrahedron_integration_step1(double u, void *prms){
  struct F_integration_params1 * params_in = (struct F_integration_params1 *) prms;
  struct F_integration_params2 params;
  double itol[3];
  itol[0] = (params_in->itols[0]);
  itol[1] = (params_in->itols[1]);
  itol[2] = (params_in->itols[2]);
  int nvertices = (params_in->nvertices);
  params.u = u;
  params.xc = (params_in->xc);
  params.yc = (params_in->yc);
  params.zc = (params_in->zc);
  params.a_B = (params_in->a_B);
  params.b_B = (params_in->b_B);
  params.c_B = (params_in->c_B);
  
  params.nvertices = nvertices;
  params.prefac = (params_in->prefac);
  int i, j;
  for (i = 0; i < 3; i++){
    for (j = 0; j < 3; j++){
      params.Rot_A[i][j] = (params_in->Rot_A[i][j]);
      params.Rot_B_T[i][j] = (params_in->Rot_B_T[i][j]);
    }
  }
  params.FM_check = (params_in->FM_check);
  params.component = (params_in->component);
  
  params.bodyA_faces = (params_in->bodyA_faces);
  params.bodyB_faces = (params_in->bodyB_faces);
  params.nfaces_arr[0] = (params_in->nfaces_arr[0]);
  params.nfaces_arr[1] = (params_in->nfaces_arr[1]);
  params.nfaces_arr[2] = (params_in->nfaces_arr[2]);
  params.nfaces_arr[3] = (params_in->nfaces_arr[3]);
  params.vertices = malloc(nvertices*3*sizeof(double*));
  for (i=0; i<nvertices; i++){
    params.vertices[i] = malloc(3*sizeof(double));
    params.vertices[i][0] = (params_in->vertices[i][0]);
    params.vertices[i][1] = (params_in->vertices[i][1]);
    params.vertices[i][2] = (params_in->vertices[i][2]); 
  }
  params.vertex_combo[0] = (params_in->vertex_combo[0]);
  params.vertex_combo[1] = (params_in->vertex_combo[1]);
  params.vertex_combo[2] = (params_in->vertex_combo[2]);
  params.vertex_combo[3] = (params_in->vertex_combo[3]);
  params.face_ids_other = malloc((params_in->nfaces_arr[1])*3*sizeof(int));
  for (i=0; i<3*(params_in->nfaces_arr[1]); i++){
    params.face_ids_other[i] = (params_in->face_ids_other[i]);
  }
  // Sets up GSL functions
  double result, error;
  gsl_function F;
  F.function = &Integrate_tetrahedron_surface;
  F.params = &params;
  
  gsl_integration_workspace * w = gsl_integration_workspace_alloc(2000);
  gsl_integration_qag(&F, 0, 1, itol[0], itol[1], 2000, itol[2], w, &result, &error);
  gsl_integration_workspace_free(w);
  for (i=0; i<nvertices; i++){
    free(params.vertices[i]);
  }
  free(params.vertices);
  free(params.face_ids_other);
  return result;
}

double Force_polyhedron(double input[11], double itol[3], double semiaxes[6], double **vertices,
 int FM_check, int component, int eulerparam, int vertex_combo[3], double prefac, int vertexids[7], int *face_ids_other){
  /*
  Computes force on a tetrahedron by integrating over the tetrahedron surface.
  */
  // Total number of faces
  int bodyA = vertexids[2];
  int bodyB = vertexids[3];
  int nvertices = vertexids[6];

  int v1 = vertex_combo[0];
  int v2 = vertex_combo[1];
  int v3 = vertex_combo[2];
  int outer;

  for (outer = vertexids[5]*vertexids[2]; outer < vertexids[4] + vertexids[5]*vertexids[2] - 1; outer ++){
    if (outer != v1 && outer != v2 && outer != v3){
      break;
    }
  }

  // Semiaxes of body B
  double a_B = semiaxes[3];
  double b_B = semiaxes[4];
  double c_B = semiaxes[5];
  // Difference in centroid
  double xc = input[0];
  double yc = input[1];
  double zc = input[2];
  
  // Angles and rotation matrices
  double Rot_A[3][3];
  double Rot_B_T[3][3];
  if (eulerparam){
    double e0_A = input[3];
    double e1_A = input[4];
    double e2_A = input[5];
    double e3_A = input[6];
    double e0_B = input[7];
    double e1_B = input[8];
    double e2_B = input[9];
    double e3_B = input[10];
    Rotation_matrix_Euler_param(e0_A, e1_A, e2_A, e3_A, Rot_A, 0);
    Rotation_matrix_Euler_param(e0_B, e1_B, e2_B, e3_B, Rot_B_T, 1); 
  }
  else{
    double phi_A = input[3];
    double theta_A = input[4];
    double psi_A = input[5];
    double phi_B = input[6];
    double theta_B = input[7];
    double psi_B = input[8];
    Rotation_matrix_components(phi_A, theta_A, psi_A, Rot_A, 0);
    Rotation_matrix_components(phi_B, theta_B, psi_B, Rot_B_T, 1); 
  }
  // Rotate vertices
  int i, j;
  struct F_integration_params1 params;
  params.xc = xc;
  params.yc = yc;
  params.zc = zc;
  params.a_B = a_B;
  params.b_B = b_B;
  params.c_B = c_B;
  params.prefac = prefac;
  params.nvertices = nvertices;
  params.itols[0] = itol[0];
  params.itols[1] = itol[1];
  params.itols[2] = itol[2];
  params.FM_check = FM_check;
  params.component = component;
  for (i = 0; i < 3; i++){
    for (j = 0; j < 3; j++){
      params.Rot_A[i][j] = Rot_A[i][j];
      params.Rot_B_T[i][j] = Rot_B_T[i][j];
    }
  }
  params.bodyA_faces = bodyA;
  params.bodyB_faces = bodyB;
  params.nfaces_arr[0] = vertexids[0];
  params.nfaces_arr[1] = vertexids[1];
  params.nfaces_arr[2] = vertexids[4];
  params.nfaces_arr[3] = vertexids[5];
  params.vertices = malloc(nvertices*sizeof(double*));
  for (i=0; i<nvertices; i++){
    params.vertices[i] = malloc(3*sizeof(double));
    params.vertices[i][0] = vertices[i][0];// - centroid[0];
    params.vertices[i][1] = vertices[i][1];// - centroid[1];
    params.vertices[i][2] = vertices[i][2];// - centroid[2]; 
  } 

  params.vertex_combo[0] = v1;
  params.vertex_combo[1] = v2;
  params.vertex_combo[2] = v3;
  params.vertex_combo[3] = outer;
  params.face_ids_other = malloc(vertexids[1]*3*sizeof(int));
  for (i=0; i<3*vertexids[1]; i++){
    params.face_ids_other[i] = face_ids_other[i];
  }
  // Sets up GSL functions
  double result;
  gsl_function F;
  F.function = &Tetrahedron_integration_step1;
  F.params = &params;
  //gsl_set_error_handler_off();
  
  // Fixed Legendre quadrature, faster than qag for low n, but less accurate in some situations
  // WIll give errors in e.g. z-mototion in co-planar motion.
  // Use standard QAG in release version. 
  /*
  const gsl_integration_fixed_type * T = gsl_integration_fixed_legendre;
  gsl_integration_fixed_workspace * w = gsl_integration_fixed_alloc(T, itol[3], 0, 1, 0.0, 0.0);
  gsl_integration_fixed(&F, &result, w);
  gsl_integration_fixed_free (w);
  */
  
  double error;
  gsl_integration_workspace * w = gsl_integration_workspace_alloc(2000);
  gsl_integration_qag(&F, 0, 1, itol[0], itol[1], 2000, itol[2], w, &result, &error);
  gsl_integration_workspace_free(w);
  for (i=0; i<nvertices; i++){
    free(params.vertices[i]);
  }
  free(params.vertices);
  free(params.face_ids_other);
  return result;
}


/* 
===========================================
Computes force over a point mass. 
Expression reduces to Newtonian form.
===========================================
*/

int Force_point_mass(double input[11], double semiaxes[6], int eulerparam, double **vertices, int vertexids[7],
 int *face_ids_other, double result[3], double mass_array[4]){
  /* 
  Computes force of a point mass in a gravitational field of any body.
  Instead of doing a surface integral, the force is the gravitational field of the opposing body
  multiplied with the mass of the point mass.
  */
   // Difference in centroid
  double xc = input[0];
  double yc = input[1];
  double zc = input[2];
  // Angles and rotation matrix of B
  double Rot_B[3][3]; 
  double Rot_B_T[3][3];
  if (eulerparam){
    double e0_B = input[7];
    double e1_B = input[8];
    double e2_B = input[9];
    double e3_B = input[10];
    Rotation_matrix_Euler_param(e0_B, e1_B, e2_B, e3_B, Rot_B, 0);
    Rotation_matrix_Euler_param(e0_B, e1_B, e2_B, e3_B, Rot_B_T, 1);
  }
  else{
    double phi_B = input[6];
    double theta_B = input[7];
    double psi_B = input[8];
    Rotation_matrix_components(phi_B, theta_B, psi_B, Rot_B, 0);
    Rotation_matrix_components(phi_B, theta_B, psi_B, Rot_B_T, 1);
  }
  // Semiaxes of body B
  double a_B = semiaxes[3];
  double b_B = semiaxes[4];
  double c_B = semiaxes[5];
  // Unpack mass array
  double mass_A = mass_array[0];
  double mass_B = mass_array[1];
  double rho_B = mass_array[2];
  double G_grav = mass_array[3];
  double prefac = mass_A*G_grav;
  
  double newx = Rot_B_T[0][0]*xc + Rot_B_T[0][1]*yc + Rot_B_T[0][2]*zc;
  double newy = Rot_B_T[1][0]*xc + Rot_B_T[1][1]*yc + Rot_B_T[1][2]*zc;
  double newz = Rot_B_T[2][0]*xc + Rot_B_T[2][1]*yc + Rot_B_T[2][2]*zc;

  int nfaces_B = vertexids[1];
  int bodyB_ID = vertexids[3];
  int nvertices_A = vertexids[4];
  int nvertices_B = vertexids[5];
  double gx_n, gy_n, gz_n;
  if (nfaces_B > 0){
    double **vertices_in = (double **)malloc(nvertices_B * sizeof(double *));
    for (int i=0; i<nvertices_B; i++){
      vertices_in[i] = (double*)malloc(3 * sizeof(double));
      vertices_in[i][0] = vertices[i+nvertices_A*bodyB_ID][0];
      vertices_in[i][1] = vertices[i+nvertices_A*bodyB_ID][1];
      vertices_in[i][2] = vertices[i+nvertices_A*bodyB_ID][2];
      //printf("vv = [%.3f, %.3f, %.3f]\n", vertices_in[i][0], vertices_in[i][1], vertices_in[i][2]);
    }
    double r[3];
    r[0] = newx;
    r[1] = newy;
    r[2] = newz;
    double g_field[3];
    g_field[0] = 0;
    g_field[1] = 0;
    g_field[2] = 0;
    double datastuff[16]= {0};
    datastuff[5] = xc;
    datastuff[6] = yc;
    datastuff[7] = zc;
    datastuff[14] = -1;
    datastuff[15] = -1;
    int use_comp = 0;
    if (nfaces_B > 100){
      use_comp = 1;  
    }
    double phi = (prefac/2)*grav_field_polyhedron(r, vertices_in, nfaces_B, face_ids_other, g_field, use_comp);
    gx_n = -prefac*g_field[0]*rho_B;
    gy_n = -prefac*g_field[1]*rho_B;
    gz_n = -prefac*g_field[2]*rho_B;
    for (int i=0; i<nvertices_B; i++){
      free(vertices_in[i]);
    }
    free(vertices_in);
  }
  else{
    double x_sq = newx*newx;
    double y_sq = newy*newy;
    double z_sq = newz*newz;
    double a_sq = a_B*a_B;
    double b_sq = b_B*b_B;
    double c_sq = c_B*c_B;
    if (fabs(a_B - b_B) < 1e-15){
      if (a_B > c_B){
        // Oblate
        double ka = kappa_value(a_sq, c_sq, x_sq, y_sq, z_sq);
        double factor = 2*M_PI*a_B*b_B*c_B*prefac*rho_B;
        double csqrtka = sqrt(c_sq + ka);
        double a_min_c = a_sq - c_sq;
        double f1 = csqrtka/(a_min_c*(a_sq+ka));
        double f2 = asin(sqrt(a_min_c/(a_sq+ka)))/(a_min_c*sqrt(a_min_c));
        double f3 = 1.0/(a_min_c*csqrtka);
        gx_n = factor*newx*(f1 - f2);
        gy_n = factor*newy*(f1 - f2);
        gz_n = 2.0*factor*newz*(f2 - f3);
      }
      else if (a_B < c_B){
        // Prolate
        double ka = kappa_value(a_sq, c_sq, x_sq, y_sq, z_sq);
        double factor = 2*M_PI*a_B*b_B*c_B*prefac*rho_B;
        double csqrtka = sqrt(c_sq + ka);
        double a_min_c = c_sq - a_sq;
        double f1 = csqrtka/(a_min_c*(a_sq+ka));
        double f2 = asinh(sqrt(a_min_c/(a_sq+ka)))/(a_min_c*sqrt(a_min_c));
        double f3 = 1.0/(a_min_c*csqrtka);
        gx_n = factor*newx*(f2 - f1);
        gy_n = factor*newy*(f2 - f1);
        gz_n = 2.0*factor*newz*(f3 - f2);
      }
      else{
        // Sphere
        double denominator = (x_sq + y_sq + z_sq)*sqrt(x_sq + y_sq + z_sq);
        double f1 = newx/denominator;
        double f2 = newy/denominator;
        double f3 = newz/denominator;
        gx_n = -prefac*f1*mass_B;
        gy_n = -prefac*f2*mass_B;
        gz_n = -prefac*f3*mass_B;
      }
    } 
    else{
      // Ellipsoid
      double g_field[3];
      double factor = 2*M_PI*a_B*b_B*c_B*prefac*rho_B;
      double ka = kappa_ellipsoid(a_sq, b_sq, c_sq, x_sq, y_sq, z_sq);
      get_gravfield_ellipsoid(a_sq, b_sq, c_sq, ka, g_field);
      gx_n = factor*newx*g_field[0];
      gy_n = factor*newy*g_field[1];
      gz_n = factor*newz*g_field[2];
    }
  }

  result[0] = Rot_B[0][0]*gx_n + Rot_B[0][1]*gy_n + Rot_B[0][2]*gz_n;
  result[1] = Rot_B[1][0]*gx_n + Rot_B[1][1]*gy_n + Rot_B[1][2]*gz_n;
  result[2] = Rot_B[2][0]*gx_n + Rot_B[2][1]*gy_n + Rot_B[2][2]*gz_n;
  return 0;
}


/* 
===========================================
Mutual potential surface integral.
Integrates over an ellipsoidal body
===========================================
*/

double integrate_potint_ellipsoid_surface(double t, void *p){
  /*
  Computes the mutual gravitational potential of an ellipsoid in a tetrahedron field.
  */
  // Unpacking parameters
  struct potint_tetra_params2 * params = (struct potint_tetra_params2 *) p;
  double z = (params->z);
  double xc = (params->xc);
  double yc = (params->yc);
  double zc = (params->zc);
  double c_sqa = (params->c_sq);
  double a_div_c = (params->a_div_c);
  double b_div_c = (params->b_div_c);
  double a_B = (params->a_B);
  double b_B = (params->b_B);
  double c_B = (params->c_B);
  
  int nfaces_B = (params->nfaces_B);
  int nvertices_A = (params->nvertices_A);
  int nvertices_B = (params->nvertices_B);
  int bodyB_ID = (params->bodyB_ID);
  double rho_B = (params->rho_B);
  double prefac = (params->prefac); // This is rho_A*G_grav
  double mB = (params->mb);

  // Rotation matrices
  double Rot_A[3][3];
  double Rot_B_T[3][3];
  double Rot_grav[3][3];
  int i;
  for (i = 0; i < 3; i++){
    for (int j = 0; j < 3; j++){
      Rot_A[i][j] = (params->Rot_A[i][j]);
      Rot_B_T[i][j] = (params->Rot_B_T[i][j]);
      Rot_grav[i][j] = (params->Rot_grav[i][j]);
    }
  }

  int* face_ids_other = (int*)malloc(nfaces_B*3*sizeof(int));
  for (i=0; i<3*nfaces_B; i++){
    face_ids_other[i] = (params->face_ids_other[i]);
  }

  double cz_sqrt = sqrt(c_sqa - z*z);
  double aprime = a_div_c*cz_sqrt;
  double bprime = b_div_c*cz_sqrt;
  double cost = cos(t);
  double sint = sin(t);

  // Surface points of spheroid body 0
  double x = aprime*cost;
  double y = bprime*sint;
  // Computes rotated global coordinate points. See notes
  double RB_ro1 = x*Rot_A[0][0] + y*Rot_A[0][1] + z*Rot_A[0][2] - xc;
  double RB_ro2 = x*Rot_A[1][0] + y*Rot_A[1][1] + z*Rot_A[1][2] - yc;
  double RB_ro3 = x*Rot_A[2][0] + y*Rot_A[2][1] + z*Rot_A[2][2] - zc;
  
  double newx = Rot_B_T[0][0]*RB_ro1 + Rot_B_T[0][1]*RB_ro2 + Rot_B_T[0][2]*RB_ro3;
  double newy = Rot_B_T[1][0]*RB_ro1 + Rot_B_T[1][1]*RB_ro2 + Rot_B_T[1][2]*RB_ro3;
  double newz = Rot_B_T[2][0]*RB_ro1 + Rot_B_T[2][1]*RB_ro2 + Rot_B_T[2][2]*RB_ro3;
  
  double nx = bprime*cost;
  double ny = aprime*sint;
  double nz = z*a_div_c*b_div_c;
  
  double phi;
  double gx_n, gy_n, gz_n;
  double g_field[3];
  double pre_ggrav;
  if (nvertices_B > 0){
    pre_ggrav = rho_B;
    double **vertices_in = (double **)malloc(nvertices_B * sizeof(double *));
    for (i=0; i<nvertices_B; i++){
      vertices_in[i] = (double*)malloc(3 * sizeof(double));
      vertices_in[i][0] = (params->vertices[i+nvertices_A*bodyB_ID][0]);
      vertices_in[i][1] = (params->vertices[i+nvertices_A*bodyB_ID][1]);
      vertices_in[i][2] = (params->vertices[i+nvertices_A*bodyB_ID][2]);
    }
    double r[3];
    r[0] = newx;
    r[1] = newy;
    r[2] = newz;
  
    g_field[0] = 0;
    g_field[1] = 0;
    g_field[2] = 0;
    phi = (rho_B/2)*grav_field_polyhedron(r, vertices_in, nfaces_B, face_ids_other, g_field, 0);
    
    //double term1, term2, ptint;
    gx_n = -g_field[0];
    gy_n = -g_field[1];
    gz_n = -g_field[2];

    for (i=0; i<nvertices_B; i++){
      free(vertices_in[i]);
    }
    free(vertices_in);
  }
  else{
    // Squaring
    double x_sq = newx*newx;
    double y_sq = newy*newy;
    double z_sq = newz*newz;
    double a_sq = a_B*a_B;
    double b_sq = b_B*b_B;
    double c_sq = c_B*c_B;
    double prefac_2 = M_PI*a_B*b_B*c_B*rho_B;
    pre_ggrav = 2.0*prefac_2; 
    if (fabs(a_B - b_B) < 1e-15){
        double ka = kappa_value(a_sq, c_sq, x_sq, y_sq, z_sq);
        if (a_B > c_B){
          // Oblate
          phi = prefac_2*potential_spheroid_oblate(x_sq, y_sq, z_sq, a_sq, c_sq, ka, c_B);
          double csqrtka = sqrt(c_sq + ka);
          double a_min_c = a_sq - c_sq;
          double f1 = csqrtka/(a_min_c*(a_sq+ka));
          double f2 = asin(sqrt(a_min_c/(a_sq+ka)))/(a_min_c*sqrt(a_min_c));
          double f3 = 1.0/(a_min_c*csqrtka);
          gx_n = newx*(f1 - f2);
          gy_n = newy*(f1 - f2);
          gz_n = 2.0*newz*(f2 - f3);
        }
        else if (a_B < c_B){
          // Prolate
          phi = prefac_2*potential_spheroid_prolate(x_sq, y_sq, z_sq, a_sq, c_sq, ka, c_B);
          double csqrtka = sqrt(c_sq + ka);
          double a_min_c = c_sq - a_sq;
          double f1 = csqrtka/(a_min_c*(a_sq+ka));
          double f2 = asinh(sqrt(a_min_c/(a_sq+ka)))/(a_min_c*sqrt(a_min_c));
          double f3 = 1.0/(a_min_c*csqrtka);
          gx_n = newx*(f2 - f1);
          gy_n = newy*(f2 - f1);
          gz_n = 2.0*newz*(f3 - f2);
        }
        else{
          // Sphere
          // If this is the case, prefac is mB*G_grav
          phi = mB*potential_sphere(newx, newy, newz);
          double denominator = (x_sq + y_sq + z_sq)*sqrt(x_sq + y_sq + z_sq);
          double f1 = newx/denominator;
          double f2 = newy/denominator;
          double f3 = newz/denominator;
          gx_n = -f1*mB;
          gy_n = -f2*mB;
          gz_n = -f3*mB;
          pre_ggrav = 1;
        }
      } 
      else{
        // Ellipsoid
        double ka = kappa_ellipsoid(a_sq, b_sq, c_sq, x_sq, y_sq, z_sq);
        phi = prefac_2*get_potential_ellipsoid(x_sq, y_sq, z_sq, a_sq, b_sq, c_sq, ka);
        get_gravfield_ellipsoid(a_sq, b_sq, c_sq, ka, g_field);
        gx_n = newx*g_field[0];
        gy_n = newy*g_field[1];
        gz_n = newz*g_field[2];
      }
  }

  double gx = Rot_grav[0][0]*gx_n + Rot_grav[0][1]*gy_n + Rot_grav[0][2]*gz_n;
  double gy = Rot_grav[1][0]*gx_n + Rot_grav[1][1]*gy_n + Rot_grav[1][2]*gz_n;
  double gz = Rot_grav[2][0]*gx_n + Rot_grav[2][1]*gy_n + Rot_grav[2][2]*gz_n;

  // Gravitational field components
  double term1 = phi*(x*nx + y*ny + z*nz);
  double term2 = 0.5*(x*x + y*y + z*z)*(gx*nx + gy*ny + gz*nz)*pre_ggrav;
  double ptint = prefac*(term1 - term2)/3.0;

  free(face_ids_other);
  return ptint;
}

double potint_ellipsoid_step1(double z, void *p){
  /*
  Do the first integration of the gravitational potential energy.
  */
  // Unpacking parameters
  struct potint_tetra_params1 * params = (struct potint_tetra_params1 *) p;
  double t_low = (params->t_low);
  double t_up = (params->t_up);
  double iabstol = (params->iabstol);
  double ireltol = (params->ireltol);
  double quadval = (params->quadval);
  // Packs parameters for second integral
  struct potint_tetra_params2 params2;
  params2.z = z;
  params2.xc = (params->xc);
  params2.yc = (params->yc);
  params2.zc = (params->zc);
  int i;
  for (i = 0; i < 3; i++){
    for (int j = 0; j < 3; j++){
      params2.Rot_A[i][j] = (params->Rot_A[i][j]);
      params2.Rot_B_T[i][j] = (params->Rot_B_T[i][j]);
      params2.Rot_grav[i][j] = (params->Rot_grav[i][j]);
    }
  }
  params2.c_sq = (params->c_sq);
  params2.a_div_c = (params->a_div_c);
  params2.b_div_c = (params->b_div_c);
  params2.a_B = (params->a_B);
  params2.b_B = (params->b_B);
  params2.c_B = (params->c_B);
  params2.nfaces_B = (params->nfaces_B);
  params2.nvertices_A = (params->nvertices_A);
  params2.nvertices_B = (params->nvertices_B);
  params2.bodyB_ID = (params->bodyB_ID);
  params2.prefac = (params->prefac);
  params2.rho_B = (params->rho_B);
  params2.mb = (params->mb);

  params2.vertices = malloc((params->nvertices_B)*sizeof(double*));
  for (i=0; i<(params->nvertices_B); i++){
    params2.vertices[i] = malloc(3*sizeof(double));
    params2.vertices[i][0] = (params->vertices[i][0]);
    params2.vertices[i][1] = (params->vertices[i][1]);
    params2.vertices[i][2] = (params->vertices[i][2]); 
  }
  params2.face_ids_other = malloc((params->nfaces_B)*3*sizeof(int));
  for (i=0; i<3*(params->nfaces_B); i++){
    params2.face_ids_other[i] = (params->face_ids_other[i]);
  }

  // Calls integration
  gsl_integration_workspace * w = gsl_integration_workspace_alloc(2000);
  double result, error;
  gsl_function F;
  F.function = &integrate_potint_ellipsoid_surface;
  //F.function = &potintegrand_spheroid_testcase;
  F.params = &params2;
  // Integrate over t
  //gsl_set_error_handler_off(); // Turn off error handling. Must be used with caution!
  gsl_integration_qag(&F, t_low, t_up, iabstol, ireltol, 2000, quadval, w, &result, &error);
  gsl_integration_workspace_free(w);

  for (i=0; i<(params->nvertices_B); i++){
    free(params2.vertices[i]);
  }
  free(params2.vertices);
  free(params2.face_ids_other);
  return result;
}

double mutual_potential_ellipsoid(double limits[4], double semiaxes[6], double input_values[11],
 double itols[3], double *vertices1D, int vertexids[6], int* face_ids, double massdens[4], int eulerparam){
  /*
  Calls the double integration of the mutual gravitational potential.
  For two bodies A, and B, integrates over the surface of A in the potential field of B.
  Applied for tetrahedron case, integrating over ellipsoid surface in tetrahedron field.
  */
  // Unpacking variables
  int total_vertices = vertexids[0] + vertexids[1];
  double **vertices = (double **)malloc(total_vertices*sizeof(double *));
  for (int i=0; i<total_vertices; i++){
    vertices[i] = (double*)malloc(3*sizeof(double));
    for (int j=0; j<3; j++){
      vertices[i][j] = vertices1D[3*i+j];
    }
  }
  // Limits
  double z_low = limits[0];
  double z_up = limits[1];
  double t_low = limits[2];
  double t_up = limits[3];
  // Semiaxes
  double aA = semiaxes[0];
  double bA = semiaxes[1];
  double cA = semiaxes[2];
  double aB = semiaxes[3];
  double bB = semiaxes[4];
  double cB = semiaxes[5];
  // Difference in centroid
  double xc = input_values[0];
  double yc = input_values[1];
  double zc = input_values[2];
    // Angles and rotation matrices
  double Rot_A[3][3];
  double Rot_B_T[3][3];
  double Rot_grav[3][3];
  if (eulerparam){
    double e0_A = input_values[3];
    double e1_A = input_values[4];
    double e2_A = input_values[5];
    double e3_A = input_values[6];
    double e0_B = input_values[7];
    double e1_B = input_values[8];
    double e2_B = input_values[9];
    double e3_B = input_values[10];
    Rotation_matrix_Euler_param(e0_A, e1_A, e2_A, e3_A, Rot_A, 0);
    Rotation_matrix_Euler_param(e0_B, e1_B, e2_B, e3_B, Rot_B_T, 1); 
    Grav_rot_matrix_euler(e0_B, e1_B, e2_B, e3_B, e0_A, e1_A, e2_A, e3_A, Rot_grav);
  }
  else{
    double phi_A = input_values[3];
    double theta_A = input_values[4];
    double psi_A = input_values[5];
    double phi_B = input_values[6];
    double theta_B = input_values[7];
    double psi_B = input_values[8];
    Rotation_matrix_components(phi_A, theta_A, psi_A, Rot_A, 0);
    Rotation_matrix_components(phi_B, theta_B, psi_B, Rot_B_T, 1); 
    Grav_rot_matrix(phi_B, theta_B, psi_B, phi_A, theta_A, psi_A, Rot_grav);
  }
  
  // Other constants
  double rho_A = massdens[0];
  double rho_B = massdens[1];
  double mB = massdens[2];
  double G_grav = massdens[3];
  
  // Packing new params array for second integration
  struct potint_tetra_params1 params;
  params.t_low = t_low;
  params.t_up = t_up;
  params.xc = xc;
  params.yc = yc;
  params.zc = zc;
  for (int i = 0; i < 3; i++){
    for (int j = 0; j < 3; j++){
      params.Rot_A[i][j] = Rot_A[i][j];
      params.Rot_B_T[i][j] = Rot_B_T[i][j];
      params.Rot_grav[i][j] = Rot_grav[i][j];
    }
  }
  params.iabstol = itols[0];
  params.ireltol = itols[1];
  params.quadval = itols[2];
  params.c_sq = cA*cA;
  params.a_div_c = aA/cA;
  params.b_div_c = bA/cA;
  params.a_B = aB;
  params.b_B = bB;
  params.c_B = cB;
  params.nfaces_B = vertexids[3];
  params.nvertices_A = vertexids[0];
  params.nvertices_B = vertexids[1];
  params.bodyB_ID = vertexids[5];
  params.prefac = rho_A*G_grav;
  params.rho_B = rho_B;
  params.mb = mB;
  // Centroid moved outside of function.
  params.vertices = malloc(total_vertices*sizeof(double*));
  for (int i=0; i<total_vertices; i++){
    params.vertices[i] = malloc(3*sizeof(double));
    params.vertices[i][0] = vertices[i][0];
    params.vertices[i][1] = vertices[i][1];
    params.vertices[i][2] = vertices[i][2];
  } 
  params.face_ids_other = malloc(vertexids[3]*3*sizeof(int));
  for (int i=0; i<3*vertexids[3]; i++){
    params.face_ids_other[i] = face_ids[i];
  }
  // Calls integration
  double result, error;
  gsl_function F;
  F.function = &potint_ellipsoid_step1;
  F.params = &params;
  //gsl_set_error_handler_off(); // Turn off error handling. Must be used with caution!
  gsl_integration_workspace * w = gsl_integration_workspace_alloc(2000);
  gsl_integration_qag(&F, z_low, z_up, itols[0], itols[1], 2000, itols[2], w, &result, &error);
  gsl_integration_workspace_free(w);
  /*
  // Fixed Legendre quadrature, faster than qag for low n
  const gsl_integration_fixed_type * T = gsl_integration_fixed_legendre;
  gsl_integration_fixed_workspace * w = gsl_integration_fixed_alloc(T, itols[3], z_low, z_up, 0.0, 0.0);
  gsl_integration_fixed(&F, &result, w);
  gsl_integration_fixed_free (w);
  */
  
  for (int i=0; i<total_vertices; i++){
    free(params.vertices[i]);
    free(vertices[i]);
  }
  free(params.vertices);
  free(vertices);
  free(params.face_ids_other);
  return result;
}

/* 
===========================================
Mutual potential surface integral.
Integrates over a polyhedron body
===========================================
*/

double integrate_potint_polyhedron_surface(double v, void *p){
  /*
  Computes the mutual gravitational potential of an ellipsoid in a tetrahedron field.
  */
  // Unpacking parameters
  struct potint_ellip_params2 * params = (struct potint_ellip_params2 *) p;
  double u = (params->u);  
  double xc = (params->xc);
  double yc = (params->yc);
  double zc = (params->zc);
  double a_B = (params->a_B);
  double b_B = (params->b_B);
  double c_B = (params->c_B);
  double G_grav = (params->G_grav);
  double rho_A = (params->rho_A);
  double rho_B = (params->rho_B);
  double mB = (params->mB);
  int total_vertices = (params->total_vertices);
  int i, j;
  // Rotation matrices
  double Rot_A[3][3];
  double Rot_B_T[3][3];
  double Rot_grav[3][3];
  for (i = 0; i < 3; i++){
    for (j = 0; j < 3; j++){
      Rot_A[i][j] = (params->Rot_A[i][j]);
      Rot_B_T[i][j] = (params->Rot_B_T[i][j]);
      Rot_grav[i][j] = (params->Rot_grav[i][j]);
    }
  }
  int nvert_A = (params->nvert_A);
  int nvert_B = (params->nvert_B);
  int nfaces_B = (params->nfaces_B);
  int bodyB_ID = (params->bodyB_ID);

  int* face_ids_other = (int*)malloc(nfaces_B*3*sizeof(int));
  for (i=0; i<3*nfaces_B; i++){
    face_ids_other[i] = (params->face_ids_other[i]);
  }
  int v1 = (params->vertex_combo[0]);
  int v2 = (params->vertex_combo[1]);
  int v3 = (params->vertex_combo[2]);
  int outer = (params->vertex_combo[3]);
  
  double x1 = (params->vertices[v1][0]);
  double x2 = (params->vertices[v2][0]);
  double x3 = (params->vertices[v3][0]);
  double y1 = (params->vertices[v1][1]);
  double y2 = (params->vertices[v2][1]);
  double y3 = (params->vertices[v3][1]);
  double z1 = (params->vertices[v1][2]);
  double z2 = (params->vertices[v2][2]);
  double z3 = (params->vertices[v3][2]);
  double x4 = (params->vertices[outer][0]);
  double y4 = (params->vertices[outer][1]);
  double z4 = (params->vertices[outer][2]);
  
  
  double x = (1-v)*x1 + v*u*x2 + v*(1-u)*x3;
  double y = (1-v)*y1 + v*u*y2 + v*(1-u)*y3;
  double z = (1-v)*z1 + v*u*z2 + v*(1-u)*z3;
  
  // Computes rotated global coordinate points. See notes
  double RB_ro1 = x*Rot_A[0][0] + y*Rot_A[0][1] + z*Rot_A[0][2] - xc;
  double RB_ro2 = x*Rot_A[1][0] + y*Rot_A[1][1] + z*Rot_A[1][2] - yc;
  double RB_ro3 = x*Rot_A[2][0] + y*Rot_A[2][1] + z*Rot_A[2][2] - zc;
  
  double newx = Rot_B_T[0][0]*RB_ro1 + Rot_B_T[0][1]*RB_ro2 + Rot_B_T[0][2]*RB_ro3;
  double newy = Rot_B_T[1][0]*RB_ro1 + Rot_B_T[1][1]*RB_ro2 + Rot_B_T[1][2]*RB_ro3;
  double newz = Rot_B_T[2][0]*RB_ro1 + Rot_B_T[2][1]*RB_ro2 + Rot_B_T[2][2]*RB_ro3;
  
  // Normal vector for integration
  double nx,ny,nz;
  double nxt = v*((y2-y3)*(-z1 + u*z2 + (1-u)*z3) - (z2-z3)*(-y1 + u*y2 + (1-u)*y3));
  double nyt = v*((z2-z3)*(-x1 + u*x2 + (1-u)*x3) - (x2-x3)*(-z1 + u*z2 + (1-u)*z3));
  double nzt = v*((x2-x3)*(-y1 + u*y2 + (1-u)*y3) - (y2-y3)*(-x1 + u*x2 + (1-u)*x3));
  //double normn = sqrt(nxt*nxt + nyt*nyt + nzt*nzt);
  double dcheck = nxt*(x4-x1) + nyt*(y4-y1) + nzt*(z4-z1);
  if (dcheck < 0){
    nx = nxt;
    ny = nyt;
    nz = nzt;
  }
  else{
    nx = -nxt;
    ny = -nyt;
    nz = -nzt;
  }

  // Compute kappa, grav potential and grav field
  double g_field[3];
  double phi;
  double gx_n, gy_n, gz_n;
  double prefac, pre_ggrav;
  if (nvert_B > 0){
    prefac = rho_B;
    pre_ggrav = prefac;
    double r[3];
    r[0] = newx;
    r[1] = newy;
    r[2] = newz;
    // Compute tetrahedron potential
    double **vertices_in = (double **)malloc(nvert_B * sizeof(double *));
    for (i=0; i<nvert_B; i++){
      vertices_in[i] = (double*)malloc(3 * sizeof(double));
      vertices_in[i][0] = (params->vertices[i+nvert_A*bodyB_ID][0]);
      vertices_in[i][1] = (params->vertices[i+nvert_A*bodyB_ID][1]);
      vertices_in[i][2] = (params->vertices[i+nvert_A*bodyB_ID][2]);
    }
    
    g_field[0] = 0;
    g_field[1] = 0;
    g_field[2] = 0;

    phi = (prefac/2)*grav_field_polyhedron(r, vertices_in, nfaces_B, face_ids_other, g_field, 0);
    //double term1, term2, ptint;
    gx_n = -g_field[0];
    gy_n = -g_field[1];
    gz_n = -g_field[2];

    for (i=0; i<nvert_B; i++){
      free(vertices_in[i]);
    }
    free(vertices_in);
  }
  else{
    // Squaring
    double x_sq = newx*newx;
    double y_sq = newy*newy;
    double z_sq = newz*newz;
    double a_sq = a_B*a_B;
    double b_sq = b_B*b_B;
    double c_sq = c_B*c_B;
    prefac = M_PI*a_B*b_B*c_B*rho_B;
    pre_ggrav = 2.0*prefac; 
    if (fabs(a_B - b_B) < 1e-15){
        double ka = kappa_value(a_sq, c_sq, x_sq, y_sq, z_sq);
        if (a_B > c_B){
          // Oblate
          phi = prefac*potential_spheroid_oblate(x_sq, y_sq, z_sq, a_sq, c_sq, ka, c_B);
          double csqrtka = sqrt(c_sq + ka);
          double a_min_c = a_sq - c_sq;
          double f1 = csqrtka/(a_min_c*(a_sq+ka));
          double f2 = asin(sqrt(a_min_c/(a_sq+ka)))/(a_min_c*sqrt(a_min_c));
          double f3 = 1.0/(a_min_c*csqrtka);
          gx_n = newx*(f1 - f2);
          gy_n = newy*(f1 - f2);
          gz_n = 2.0*newz*(f2 - f3);
        }
        else if (a_B < c_B){
          // Prolate
          phi = prefac*potential_spheroid_prolate(x_sq, y_sq, z_sq, a_sq, c_sq, ka, c_B);
          double csqrtka = sqrt(c_sq + ka);
          double a_min_c = c_sq - a_sq;
          double f1 = csqrtka/(a_min_c*(a_sq+ka));
          double f2 = asinh(sqrt(a_min_c/(a_sq+ka)))/(a_min_c*sqrt(a_min_c));
          double f3 = 1.0/(a_min_c*csqrtka);
          gx_n = newx*(f2 - f1);
          gy_n = newy*(f2 - f1);
          gz_n = 2.0*newz*(f3 - f2);
        }
        else{
          // Sphere
          // If this is the case, prefac is mB*G_grav
          phi = mB*potential_sphere(newx, newy, newz);
          double denominator = (x_sq + y_sq + z_sq)*sqrt(x_sq + y_sq + z_sq);
          double f1 = newx/denominator;
          double f2 = newy/denominator;
          double f3 = newz/denominator;
          gx_n = -f1*mB;
          gy_n = -f2*mB;
          gz_n = -f3*mB;
          pre_ggrav = 1;
        }
      } 
      else{
        // Ellipsoid
        double ka = kappa_ellipsoid(a_sq, b_sq, c_sq, x_sq, y_sq, z_sq);
        phi = prefac*get_potential_ellipsoid(x_sq, y_sq, z_sq, a_sq, b_sq, c_sq, ka);
        get_gravfield_ellipsoid(a_sq, b_sq, c_sq, ka, g_field);
        gx_n = newx*g_field[0];
        gy_n = newy*g_field[1];
        gz_n = newz*g_field[2];
      }
  }
  double gx = Rot_grav[0][0]*gx_n + Rot_grav[0][1]*gy_n + Rot_grav[0][2]*gz_n;
  double gy = Rot_grav[1][0]*gx_n + Rot_grav[1][1]*gy_n + Rot_grav[1][2]*gz_n;
  double gz = Rot_grav[2][0]*gx_n + Rot_grav[2][1]*gy_n + Rot_grav[2][2]*gz_n;

  // Gravitational field components
  double term1 = phi*(x*nx + y*ny + z*nz);
  double term2 = 0.5*(x*x + y*y + z*z)*(gx*nx + gy*ny + gz*nz)*pre_ggrav;
  double ptint = rho_A*G_grav*(term1 - term2)/3.0;
  free(face_ids_other);
  return ptint;
}

double potint_polyhedron_step1(double u, void *prms){
  struct potint_ellip_params1 * params_in = (struct potint_ellip_params1 *) prms;
  struct potint_ellip_params2 params;
  double itol[3];
  itol[0] = (params_in->itols[0]);
  itol[1] = (params_in->itols[1]);
  itol[2] = (params_in->itols[2]);
  int total_vertices = (params_in->total_vertices);
  params.u = u;
  params.xc = (params_in->xc);
  params.yc = (params_in->yc);
  params.zc = (params_in->zc);
  params.a_B = (params_in->a_B);
  params.b_B = (params_in->b_B);
  params.c_B = (params_in->c_B);
  
  params.total_vertices = total_vertices;
  params.G_grav = (params_in->G_grav);
  params.rho_A = (params_in->rho_A);
  params.rho_B = (params_in->rho_B);
  params.mB = (params_in->mB);
  int i, j;
  for (i = 0; i < 3; i++){
    for (j = 0; j < 3; j++){
      params.Rot_A[i][j] = (params_in->Rot_A[i][j]);
      params.Rot_B_T[i][j] = (params_in->Rot_B_T[i][j]);
      params.Rot_grav[i][j] = (params_in->Rot_grav[i][j]);
    }
  }
  params.nvert_A = (params_in->nvert_A);
  params.nvert_B = (params_in->nvert_B);
  params.nfaces_A = (params_in->nfaces_A);
  params.nfaces_B = (params_in->nfaces_B);
  params.bodyB_ID = (params_in->bodyB_ID);
  params.vertices = malloc(total_vertices*sizeof(double*));
  for (i=0; i<total_vertices; i++){
    params.vertices[i] = malloc(3*sizeof(double));
    params.vertices[i][0] = (params_in->vertices[i][0]);
    params.vertices[i][1] = (params_in->vertices[i][1]);
    params.vertices[i][2] = (params_in->vertices[i][2]); 
  }
  params.vertex_combo[0] = (params_in->vertex_combo[0]);
  params.vertex_combo[1] = (params_in->vertex_combo[1]);
  params.vertex_combo[2] = (params_in->vertex_combo[2]);
  params.vertex_combo[3] = (params_in->vertex_combo[3]);
  params.face_ids_other = malloc((params_in->nfaces_B)*3*sizeof(int));
  for (i=0; i<3*(params_in->nfaces_B); i++){
    params.face_ids_other[i] = (params_in->face_ids_other[i]);
  }
  // Sets up GSL functions
  double result, error;
  gsl_function F;
  F.function = &integrate_potint_polyhedron_surface;
  F.params = &params;
  
  gsl_integration_workspace * w = gsl_integration_workspace_alloc(2000);
  gsl_integration_qag(&F, 0, 1, itol[0], itol[1], 2000, itol[2], w, &result, &error);
  gsl_integration_workspace_free(w);

  for (i=0; i<total_vertices; i++){
    free(params.vertices[i]);
  }
  free(params.vertices);
  free(params.face_ids_other);
  return result;
}

double mutual_potential_polyhedron(double semiaxes[6], double input_values[11], double itols[3], double *vertices1D,
 int vertex_combo[3], double massdens[4], int* face_ids, int vertexids[6], int eulerparam){
   /*
  Calls the double integration of the mutual gravitational potential.
  For two bodies A, and B, integrates over the surface of A in the potential field of B.
  Applied for tetrahedron case, integrating over tetrahedron surface in an ellipsoid field.
  */
  int total_vertices = vertexids[0] + vertexids[1];
  double **vertices = (double **)malloc(total_vertices*sizeof(double *));
  for (int i=0; i<total_vertices; i++){
    vertices[i] = (double*)malloc(3*sizeof(double));
    for (int j=0; j<3; j++){
      vertices[i][j] = vertices1D[3*i+j];
    }
  }
  int v1 = vertex_combo[0];
  int v2 = vertex_combo[1];
  int v3 = vertex_combo[2];
  int outer;
  for (outer = vertexids[1]*vertexids[4]; outer < vertexids[0] + vertexids[1]*vertexids[4] - 1; outer ++){
    if (outer != v1 && outer != v2 && outer != v3){
      break;
    }
  }
  // Difference in centroid
  double xc = input_values[0];
  double yc = input_values[1];
  double zc = input_values[2];
    // Angles and rotation matrices
  double Rot_A[3][3];
  double Rot_B_T[3][3];
  double Rot_grav[3][3];
  if (eulerparam){
    double e0_A = input_values[3];
    double e1_A = input_values[4];
    double e2_A = input_values[5];
    double e3_A = input_values[6];
    double e0_B = input_values[7];
    double e1_B = input_values[8];
    double e2_B = input_values[9];
    double e3_B = input_values[10];
    Rotation_matrix_Euler_param(e0_A, e1_A, e2_A, e3_A, Rot_A, 0);
    Rotation_matrix_Euler_param(e0_B, e1_B, e2_B, e3_B, Rot_B_T, 1); 
    Grav_rot_matrix_euler(e0_B, e1_B, e2_B, e3_B, e0_A, e1_A, e2_A, e3_A, Rot_grav);
  }
  else{
    double phi_A = input_values[3];
    double theta_A = input_values[4];
    double psi_A = input_values[5];
    double phi_B = input_values[6];
    double theta_B = input_values[7];
    double psi_B = input_values[8];
    Rotation_matrix_components(phi_A, theta_A, psi_A, Rot_A, 0);
    Rotation_matrix_components(phi_B, theta_B, psi_B, Rot_B_T, 1); 
    Grav_rot_matrix(phi_B, theta_B, psi_B, phi_A, theta_A, psi_A, Rot_grav);
  }

  // Semiaxes of body B
  double a_B = semiaxes[3];
  double b_B = semiaxes[4];
  double c_B = semiaxes[5];
  // Mass and densities
  double rho_A = massdens[0];
  double rho_B = massdens[1];
  double mB = massdens[2];
  double G_grav = massdens[3];
  
  // Rotate vertices
  int i, j;
  struct potint_ellip_params1 params;
  params.xc = xc;
  params.yc = yc;
  params.zc = zc;
  params.a_B = a_B;
  params.b_B = b_B;
  params.c_B = c_B;
  params.G_grav = G_grav;
  params.rho_A = rho_A;
  params.rho_B = rho_B;
  params.mB = mB;
  params.total_vertices = total_vertices;
  params.itols[0] = itols[0];
  params.itols[1] = itols[1];
  params.itols[2] = itols[2];
  for (i = 0; i < 3; i++){
    for (j = 0; j < 3; j++){
      params.Rot_A[i][j] = Rot_A[i][j];
      params.Rot_B_T[i][j] = Rot_B_T[i][j];
      params.Rot_grav[i][j] = Rot_grav[i][j];
    }
  }
  params.nvert_A = vertexids[0];
  params.nvert_B = vertexids[1];
  params.nfaces_A = vertexids[2];  
  params.nfaces_B = vertexids[3];
  params.bodyB_ID = vertexids[5];
  params.vertices = malloc(total_vertices*sizeof(double*));
  for (i=0; i<total_vertices; i++){
    params.vertices[i] = malloc(3*sizeof(double));
    params.vertices[i][0] = vertices[i][0];
    params.vertices[i][1] = vertices[i][1];
    params.vertices[i][2] = vertices[i][2]; 
  } 
  params.vertex_combo[0] = v1;
  params.vertex_combo[1] = v2;
  params.vertex_combo[2] = v3;
  params.vertex_combo[3] = outer;
  params.face_ids_other = malloc(vertexids[3]*3*sizeof(int));
  for (i=0; i<3*vertexids[3]; i++){
    params.face_ids_other[i] = face_ids[i];
  }
  // Sets up GSL functions
  double result, error;
  gsl_function F;
  F.function = &potint_polyhedron_step1;
  F.params = &params;
  //gsl_set_error_handler_off();
  // Fixed Legendre quadrature, faster than qag for low n, but less accurate in some situations
  // WIll give errors in e.g. z-mototion in co-planar motion.
  // Use standard QAG in release version. 
  /*
  const gsl_integration_fixed_type * T = gsl_integration_fixed_legendre;
  gsl_integration_fixed_workspace * w = gsl_integration_fixed_alloc(T, itols[3], 0, 1, 0.0, 0.0);
  //gsl_integration_fixed_workspace * w = gsl_integration_fixed_alloc(T, itol[3], -c_B, c_B, 0.0, 0.0);
  gsl_integration_fixed(&F, &result, w);
  gsl_integration_fixed_free (w);
  */
  
  gsl_integration_workspace * w = gsl_integration_workspace_alloc(2000);
  gsl_integration_qag(&F, 0, 1, itols[0], itols[1], 2000, itols[2], w, &result, &error);
  gsl_integration_workspace_free(w);
  for (i=0; i<total_vertices; i++){
    free(params.vertices[i]);
    free(vertices[i]);
  }
  free(params.vertices);
  free(params.face_ids_other);
  free(vertices);
  return result;
}

/* 
===========================================
Mutual potential by "integrating" over a point mass.
Reduces to Newtonian expression.
===========================================
*/


double mutual_potential_point_mass(double semiaxes[6], double input_values[11], double *vertices1D,
 double mass_array[4], int* face_ids_other, int vertexids[6], int eulerparam){
  /* 
  Compute potential energy for a point mass.
  Potential energy is U = m_A*G*Phi_B.
  */
  // Difference in centroid
  double xc = input_values[0];
  double yc = input_values[1];
  double zc = input_values[2];
  // Angles and rotation matrix of B
  double Rot_B_T[3][3]; 
  if (eulerparam){
    double e0_B = input_values[7];
    double e1_B = input_values[8];
    double e2_B = input_values[9];
    double e3_B = input_values[10];
    Rotation_matrix_Euler_param(e0_B, e1_B, e2_B, e3_B, Rot_B_T, 1); 
  }
  else{
    double phi_B = input_values[6];
    double theta_B = input_values[7];
    double psi_B = input_values[8];
    Rotation_matrix_components(phi_B, theta_B, psi_B, Rot_B_T, 1); 
  }

  // Semiaxes of body B
  double a_A = semiaxes[0];
  double b_A = semiaxes[1];
  double c_A = semiaxes[2];
  double a_B = semiaxes[3];
  double b_B = semiaxes[4];
  double c_B = semiaxes[5];
  // Mass and densities
  double mass_A = mass_array[0];
  double mass_B = mass_array[1];
  double rho_B = mass_array[2];
  double G_grav = mass_array[3];
  double prefac = mass_A*G_grav;

  double newx = Rot_B_T[0][0]*xc + Rot_B_T[0][1]*yc + Rot_B_T[0][2]*zc;
  double newy = Rot_B_T[1][0]*xc + Rot_B_T[1][1]*yc + Rot_B_T[1][2]*zc;
  double newz = Rot_B_T[2][0]*xc + Rot_B_T[2][1]*yc + Rot_B_T[2][2]*zc;

  int nvertices_A = vertexids[0];
  int nvertices_B = vertexids[1];
  int nfaces_B = vertexids[3];
  int bodyB_ID = vertexids[5];

  double Epot;
  if (nfaces_B > 0){
    // Polyhedron
    double **vertices_in = (double **)malloc(nvertices_B * sizeof(double *));
    for (int i=0; i<nvertices_B; i++){
      vertices_in[i] = (double*)malloc(3 * sizeof(double));
      vertices_in[i][0] = vertices1D[3*(i+nvertices_A*bodyB_ID)];
      vertices_in[i][1] = vertices1D[3*(i+nvertices_A*bodyB_ID)+1];
      vertices_in[i][2] = vertices1D[3*(i+nvertices_A*bodyB_ID)+2];
    }
    double r[3];
    r[0] = newx;
    r[1] = newy;
    r[2] = newz;
    Epot = (prefac*rho_B/2)*potential_polyhedron(r, vertices_in, nfaces_B, face_ids_other);
    
    for (int i=0; i<nvertices_B; i++){
      free(vertices_in[i]);
    }
    free(vertices_in);
  }
  else{
    double x_sq = newx*newx;
    double y_sq = newy*newy;
    double z_sq = newz*newz;
    double a_sq = a_B*a_B;
    double b_sq = b_B*b_B;
    double c_sq = c_B*c_B;
    if (fabs(a_B - b_B) < 1e-15){
      if (a_B > c_B){
        // Oblate
        double ka = kappa_value(a_sq, c_sq, x_sq, y_sq, z_sq);
        double factor = M_PI*a_B*b_B*c_B*prefac*rho_B;
        Epot = factor*potential_spheroid_oblate(x_sq, y_sq, z_sq, a_sq, c_sq, ka, c_B);
      }
      else if (a_B < c_B){
        // Prolate
        double ka = kappa_value(a_sq, c_sq, x_sq, y_sq, z_sq);
        double factor = M_PI*a_B*b_B*c_B*prefac*rho_B;
        Epot = factor*potential_spheroid_prolate(x_sq, y_sq, z_sq, a_sq, c_sq, ka, c_B);
      }
      else{
        // Sphere
        Epot = mass_B*prefac*potential_sphere(newx, newy, newz);
      }
    } 
    else{
      // Ellipsoid
      double factor = M_PI*a_B*b_B*c_B*prefac*rho_B;
      double ka = kappa_ellipsoid(a_sq, b_sq, c_sq, x_sq, y_sq, z_sq);
      Epot = factor*get_potential_ellipsoid(x_sq, y_sq, z_sq, a_sq, b_sq, c_sq, ka); 
    }
  }
  return Epot;
}

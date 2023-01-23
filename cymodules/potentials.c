#define _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>
#include "commonCfuncs.h"
#include <gsl/gsl_poly.h>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_sf_result.h>

double kappa_value(double a_sq, double c_sq, double x_sq, double y_sq, double z_sq){
  // Computes kappa value from MacMillan. See notes
  double B = a_sq + c_sq - x_sq - y_sq - z_sq;
  double C = a_sq*(c_sq - z_sq) - c_sq*(x_sq + y_sq);
  double kappa = 0.5*(-B + sqrt(B*B - 4.0*C));
  return kappa;
}

double kappa_ellipsoid(double a_sq, double b_sq, double c_sq, double x_sq, double y_sq, double z_sq){
  /*
  Computes the kappa value for a general ellipsoid.
  Returns the largest root of the cubic equation.

  The cubic equation takes the form
  A*kappa^3 + B*kappa^2 + C*kappa + D = 0
  See notes for the coefficients A, B, C, D.
  */
  double ab_sq = a_sq*b_sq;
  double ac_sq = a_sq*c_sq;
  double bc_sq = b_sq*c_sq;
  // Coefficients of the cubic equation
  double B = a_sq + b_sq + c_sq - (x_sq + y_sq + z_sq);
  double C = ab_sq + ac_sq + bc_sq - x_sq*(b_sq + c_sq) - y_sq*(a_sq + c_sq) - z_sq*(a_sq + b_sq);
  double D = ab_sq*c_sq - x_sq*bc_sq - y_sq*ac_sq - z_sq*ab_sq;
  // Solves the cubic equation
  double sol1[1], sol2[1], sol3[1];
  gsl_poly_solve_cubic(B, C, D, sol1, sol2, sol3);
  double solutions[3] = {sol1[0], sol2[0], sol3[0]};

  // Find the maximum (real) value of the solutions and returns as kappa
  double kappa = 0; 
  for (int i = 0; i < 3; i++){
    if (solutions[i] > kappa){
      kappa = solutions[i];
    }
  }
  return kappa;
}

double potential_spheroid(double x_sq, double y_sq, double z_sq, double a_sq, double c_sq, double mb, double ka, double a, double c){
  /* 
  Computes the value of the potential for a spheroid.
  See MacMillan 1930 for the mathematical expression.
  Includes the Newtonian potential if a sphere is considered.

  Does not include the density factor in the expressions. Multiply this after integration!
  */
  double a_min_c;
  double xy_sq = x_sq + y_sq;
  double v1a,v1b,v1c,v1;
  double v2a,v2b,v2;
  double v3a,v3b,v3;
  double prefac;
  double prefactor = M_PI*a_sq*c;
  if(a > c){
    // For an oblate spheroid
    a_min_c = a_sq - c_sq;
    prefac = sqrt(c_sq + ka);
    v1a = 2.0*prefactor/sqrt(a_min_c);
    v1b = 1.0 - (xy_sq - 2.0*z_sq)/(2.0*a_min_c);
    v1c = asin(sqrt(a_min_c/(a_sq + ka)));
    v1 = v1a*v1b*v1c;

    v2a = prefactor*prefac/a_min_c;
    v2b = xy_sq/(a_sq + ka);
    v2 = v2a*v2b;

    v3a = prefactor/a_min_c;
    v3b = 2.0*z_sq/prefac;
    v3 = -v3a*v3b;
  }
  else if(a < c){
    // For a prolate spheroid
    a_min_c = c_sq - a_sq;
    prefac = sqrt(c_sq + ka);
    v1a = 2.0*prefactor/sqrt(a_min_c);
    v1b = 1.0 + (xy_sq - 2.0*z_sq)/(2.0*a_min_c);
    v1c = asinh(sqrt(a_min_c/(a_sq + ka)));
    v1 = v1a*v1b*v1c;

    v2a = prefactor*prefac/(a_min_c);
    v2b = xy_sq/(a_sq + ka);
    v2 = -v2a*v2b;

    v3a = prefactor/(a_min_c);
    v3b = 2.0*z_sq/prefac;
    v3 = v3a*v3b;
  }
  else{
    // For a sphere
    v1 = 0.0;
    v2 = 0.0;
    v3 = mb/sqrt(xy_sq + z_sq);
  }
  double Phi = v1 + v2 + v3;
  return Phi;
}

double potential_spheroid_oblate(double x_sq, double y_sq, double z_sq, double a_sq, double c_sq, double ka, double c){
  // Gravitational potential for an oblate spheroid. See MacMillan 1930.
  double xy_sq = x_sq + y_sq;
  
  double a_min_c = a_sq - c_sq;
  double prefac = sqrt(c_sq + ka);
  double v1a = 2.0/sqrt(a_min_c);
  double v1b = 1.0 - (xy_sq - 2.0*z_sq)/(2.0*a_min_c);
  double v1c = asin(sqrt(a_min_c/(a_sq + ka)));
  double v1 = v1a*v1b*v1c;

  double v2a = prefac/a_min_c;
  double v2b = xy_sq/(a_sq + ka);
  double v2 = v2a*v2b;

  double v3 = -2*z_sq/(prefac*a_min_c);
  double Phi = v1 + v2 + v3;
  return Phi;
}

double potential_spheroid_prolate(double x_sq, double y_sq, double z_sq, double a_sq, double c_sq, double ka, double c){
  // Gravitational potential for a prolate spheroid. See MacMillan 1930.
  double xy_sq = x_sq + y_sq;

  double a_min_c = c_sq - a_sq;
  double prefac = sqrt(c_sq + ka);
  double v1a = 2.0/sqrt(a_min_c);
  double v1b = 1.0 + (xy_sq - 2.0*z_sq)/(2.0*a_min_c);
  double v1c = asinh(sqrt(a_min_c/(a_sq + ka)));
  double v1 = v1a*v1b*v1c;

  double v2a = prefac/(a_min_c);
  double v2b = xy_sq/(a_sq + ka);
  double v2 = -v2a*v2b;

  double v3 = 2*z_sq/(prefac*a_min_c);
  //printf("v1,v2,v3 = %.5f, %.5f, %.5f\n", a_min_c, a_sq + ka, sqrt(a_min_c/(a_sq + ka)));
  double Phi = v1 + v2 + v3;
  return Phi;
}


double potential_sphere(double x, double y, double z){
  // Gravitational constant for a sphere: Uses Newtonian gravitational potential.
  double Phi = 1/sqrt(x*x + y*y + z*z);
  return Phi;
}

double potential_ellipsoid(double x_sq, double y_sq, double z_sq, double a_sq, double b_sq, double c_sq, double kappa){
  /*
  Computes the value of the potential for a general ellipsoid.
  See MacMillan 1930 for the mathematical expression.  
  Does not include the density factor in the expressions. Multiply this after integration!
  Does also not multiply the a*b*c factor.
  Uses the formulation of elliptic integrals
  */
  double omega_kappa = asin(sqrt((a_sq - c_sq)/(a_sq + kappa)));
  double k = sqrt((a_sq-b_sq)/(a_sq-c_sq));
  double F_result = gsl_sf_ellint_F(omega_kappa, k , GSL_PREC_DOUBLE);
  double E_result = gsl_sf_ellint_E(omega_kappa, k , GSL_PREC_DOUBLE);
  double result = (2/sqrt(a_sq - c_sq))*((1 - x_sq/(a_sq-b_sq) + y_sq/(a_sq-b_sq))*F_result
    + (x_sq/(a_sq-b_sq) - ((a_sq-c_sq)*y_sq)/((a_sq-b_sq)*(b_sq-c_sq)) + z_sq/(b_sq-c_sq))*E_result
    + (y_sq*(c_sq + kappa)/(b_sq-c_sq) - z_sq*(b_sq + kappa)/(b_sq-c_sq))*(sqrt(a_sq-c_sq)/sqrt((a_sq+kappa)*(b_sq+kappa)*(c_sq+kappa))));

  return result;
}

double get_potential_ellipsoid(double x_sq, double y_sq, double z_sq, double a_sq, double b_sq, double c_sq, double kappa){
  /*
  Checks semiaxes conditions and computes the ellipsoid gravitational potential properly.
  */
  double fi;
  if ((a_sq > b_sq) && (b_sq > c_sq)){
    fi = potential_ellipsoid(x_sq, y_sq, z_sq, a_sq, b_sq, c_sq, kappa);
  }
  else if ((b_sq > a_sq) && (a_sq > c_sq)){
    fi = potential_ellipsoid(y_sq, x_sq, z_sq, b_sq, a_sq, c_sq, kappa);
  }
  else if ((c_sq > b_sq) && (b_sq > a_sq)){
    fi = potential_ellipsoid(z_sq, y_sq, x_sq, c_sq, b_sq, a_sq, kappa);
  }
  else if ((a_sq > c_sq) && (c_sq > b_sq)){
    fi = potential_ellipsoid(x_sq, z_sq, y_sq, a_sq, c_sq, b_sq, kappa);
  }
  else if ((b_sq > c_sq) && (c_sq > a_sq)){
    fi = potential_ellipsoid(y_sq, z_sq, x_sq, b_sq, c_sq, a_sq, kappa);
  }
  else if ((c_sq > a_sq) && (a_sq > b_sq)){
    fi = potential_ellipsoid(z_sq, x_sq, y_sq, c_sq, a_sq, b_sq, kappa);
  }
  else{
    printf("Potential semiaxes conditions not met! \n");
    exit(1);
  }
  return fi;
}


double grav_field_ellipsoid(double a_sq, double b_sq, double c_sq, double kappa, int component){
  /*
  Computes the gravitational field of a general ellipsoid.
  See MacMillan 1930 for the mathematical expressons.
  The prefactor is not included. Only computes the integral.
  */
  double omega_kappa = asin(sqrt((a_sq - c_sq)/(a_sq + kappa)));
  double k = sqrt((a_sq-b_sq)/(a_sq-c_sq));
  gsl_sf_result F_result, E_result;
  gsl_sf_ellint_F_e(omega_kappa, k, GSL_PREC_DOUBLE, &F_result);
  gsl_sf_ellint_E_e(omega_kappa, k, GSL_PREC_DOUBLE, &E_result);
  double result;
  if (component == 1){
    // Returns x component
    result = (2/sqrt(a_sq-c_sq))*(E_result.val - F_result.val)/(a_sq-b_sq);
  }
  else if (component == 2){
    // Returns y component
    result = (2/sqrt(a_sq-c_sq))*((F_result.val/(a_sq-b_sq)) - (a_sq-c_sq)*E_result.val/((a_sq-b_sq)*(b_sq-c_sq))
            + ((c_sq+kappa)/(b_sq-c_sq))*(sqrt(a_sq-c_sq)/sqrt((a_sq+kappa)*(b_sq+kappa)*(c_sq+kappa))));
  }
  else if(component == 3){
    // Returns z component
    result = (2/sqrt(a_sq-c_sq))*(E_result.val/(b_sq-c_sq)
            - ((b_sq+kappa)/(b_sq-c_sq))*(sqrt(a_sq-c_sq)/sqrt((a_sq+kappa)*(b_sq+kappa)*(c_sq+kappa))));
  }
  else{
    printf("Component invalid. Try component=1,2,3");
    exit(1);
  }
  return result;
}

int get_gravfield_ellipsoid(double a_sq, double b_sq, double c_sq, double kappa, double g_field[3]){
  /* Checks semiaxes conditions and computes the ellipsoid gravitational field properly. */
  if ((a_sq > b_sq) && (b_sq > c_sq)){
    g_field[0] = grav_field_ellipsoid(a_sq, b_sq, c_sq, kappa, 1);
    g_field[1] = grav_field_ellipsoid(a_sq, b_sq, c_sq, kappa, 2);
    g_field[2] = grav_field_ellipsoid(a_sq, b_sq, c_sq, kappa, 3);
  }
  else if ((b_sq > a_sq) && (a_sq > c_sq)){
    g_field[0] = grav_field_ellipsoid(b_sq, a_sq, c_sq, kappa, 2);
    g_field[1] = grav_field_ellipsoid(b_sq, a_sq, c_sq, kappa, 1);
    g_field[2] = grav_field_ellipsoid(b_sq, a_sq, c_sq, kappa, 3);
  }
  else if ((c_sq > b_sq) && (b_sq > a_sq)){
    g_field[0] = grav_field_ellipsoid(c_sq, b_sq, a_sq, kappa, 3);
    g_field[1] = grav_field_ellipsoid(c_sq, b_sq, a_sq, kappa, 2);
    g_field[2] = grav_field_ellipsoid(c_sq, b_sq, a_sq, kappa, 1);
  }
  else if ((a_sq > c_sq) && (c_sq > b_sq)){
    g_field[0] = grav_field_ellipsoid(a_sq, c_sq, b_sq, kappa, 1);
    g_field[1] = grav_field_ellipsoid(a_sq, c_sq, b_sq, kappa, 3);
    g_field[2] = grav_field_ellipsoid(a_sq, c_sq, b_sq, kappa, 2);
  }
  else if ((b_sq > c_sq) && (c_sq > a_sq)){
    g_field[0] = grav_field_ellipsoid(b_sq, c_sq, a_sq, kappa, 2);
    g_field[1] = grav_field_ellipsoid(b_sq, c_sq, a_sq, kappa, 3);
    g_field[2] = grav_field_ellipsoid(b_sq, c_sq, a_sq, kappa, 1);
  }
  else if ((c_sq > a_sq) && (a_sq > b_sq)){
    g_field[0] = grav_field_ellipsoid(c_sq, a_sq, b_sq, kappa, 3);
    g_field[1] = grav_field_ellipsoid(c_sq, a_sq, b_sq, kappa, 1);
    g_field[2] = grav_field_ellipsoid(c_sq, a_sq, b_sq, kappa, 2);
  }
  else{
    printf("Potential field semiaxes conditions not met! \n");
    exit(1);
  }
  return 0;
}


int normal_vector_face_righthand(double **vertices, int face_ids[3], int v4, double n_i[3]){
  /*Computes normal vector of a triangle*/
  double r2r1[3];
  double r3r1[3];
  double r4r1[3];
  int v1 = face_ids[0];
  int v2 = face_ids[1];
  int v3 = face_ids[2];
  int tick = 0;
  while (tick == 0){
    for (int i=0; i<3; i++){
      //printf("vertices_x = [%.4f, %.4f, %.4f, %.4f]\n", vertices[v1][i], vertices[v2][i], vertices[v3][i], vertices[v4][i]);
      r2r1[i] = vertices[v2][i] - vertices[v1][i];
      r3r1[i] = vertices[v3][i] - vertices[v1][i];
      r4r1[i] = vertices[v4][i] - vertices[v1][i];
    }
    double cross_prod[3];
    cross_product(r3r1, r2r1, cross_prod);
    double norm_cproduct = absolute_value_vector(cross_prod);
    double ni_temp[3];
    ni_temp[0] = cross_prod[0]/norm_cproduct;
    ni_temp[1] = cross_prod[1]/norm_cproduct;
    ni_temp[2] = cross_prod[2]/norm_cproduct;

    double d = dot_product(ni_temp, r4r1);
    if (d < 0){
      n_i[0] = ni_temp[0];
      n_i[1] = ni_temp[1];
      n_i[2] = ni_temp[2];
      tick = 1;
    }
    else if (d == 0){
      // Discard perpendicular faces. Chooses new v4.
      return -1;
    }
    else{
      v1 = face_ids[2];
      v3 = face_ids[0];
    }
  }
  face_ids[0] = v1;
  face_ids[1] = v2;
  face_ids[2] = v3;
  return 0;
}

int sgn_func(double x){
  /* Sign function of a value x */
  if (x > 0){
    return 1;
  }
  else if (x < 0){
    return -1;
  }
  else{
    return 0;
  }
}

double K_factor(double r[3], double r_i1[3], double r_i2[3], double n_i[3], double r_i[3]){
  /* Singularity term for the polyhedron potential. */
  double r_p[3];
  double dotprod_nirij = dot_product(n_i, r_i1);
  double cprod1[3];
  double cprod2[3];
  cross_product(n_i, r, cprod1);
  cross_product(n_i, cprod1, cprod2);
  r_p[0] = dotprod_nirij*n_i[0] - cprod2[0];
  r_p[1] = dotprod_nirij*n_i[1] - cprod2[1];
  r_p[2] = dotprod_nirij*n_i[2] - cprod2[2];

  double ri1_rp[3];
  double ri2_rp[3];
  ri1_rp[0] = r_i1[0] - r_p[0];
  ri1_rp[1] = r_i1[1] - r_p[1];
  ri1_rp[2] = r_i1[2] - r_p[2];
  ri2_rp[0] = r_i2[0] - r_p[0];
  ri2_rp[1] = r_i2[1] - r_p[1];
  ri2_rp[2] = r_i2[2] - r_p[2];
  double cross_prod[3];
  cross_product(ri1_rp, ri2_rp, cross_prod);
  double bb = dot_product(n_i, cross_prod);
  int sign_factor = sgn_func(bb);
  double f1 = absolute_value_vector(ri1_rp);
  double f2 = absolute_value_vector(ri2_rp);
  double acosfactor = dot_product(ri1_rp, ri2_rp);
  double in_acos = acosfactor/(f1*f2);
  if (in_acos > 1){
    in_acos = 1;
  }
  else if (in_acos < -1){
    in_acos = -1;
  }
  double theta_ij = sign_factor*acos(in_acos);
  double K_prefactor = -fabs(n_i[0]*(r[0] - r_i1[0]) + n_i[1]*(r[1] - r_i1[1]) + n_i[2]*(r[2] - r_i1[2]));
  double K_ij = K_prefactor*theta_ij;
  return K_ij;
}

double grav_potential_face_edge(double r[3], double r_i1[3], double r_i2[3], double n_i[3]){
  /*
  Gravitational potential contribution from one segment of a face.
  Loop this over vertices j.
  */
  // Some prefactors needed for the a,b,c,d constants
  double r_ri1[3];
  double ri2_ri1[3];
  r_ri1[0] = r[0] - r_i1[0];
  r_ri1[1] = r[1] - r_i1[1];
  r_ri1[2] = r[2] - r_i1[2];
  ri2_ri1[0] = r_i2[0] - r_i1[0];
  ri2_ri1[1] = r_i2[1] - r_i1[1];
  ri2_ri1[2] = r_i2[2] - r_i1[2];
  double abs_a = absolute_value_vector(r_ri1);
  double ri2_ri1_abs = absolute_value_vector(ri2_ri1);
  double dot_prod_b = dot_product(r_ri1, ri2_ri1);
  double dot_prod_c = dot_product(n_i, r_ri1);
  double n_cross_rri[3];
  cross_product(n_i, r_ri1, n_cross_rri);
  double dot_prod_d = dot_product(n_cross_rri, ri2_ri1);

  // The constants a_ij, b_ij, c_ij, d_ij and prefactors K1, K2
  double a = abs_a/ri2_ri1_abs;
  double b = dot_prod_b/(ri2_ri1_abs*ri2_ri1_abs);
  double c = dot_prod_c/ri2_ri1_abs;
  double d = -dot_prod_d/ri2_ri1_abs;
  double K1 = sqrt(fabs(a*a - b*b - c*c));
  double K2 = sqrt(1 + a*a - 2*b);

  if (a*a - b*b - c*c <= 0){
    printf("Polyhedron potential becomes singular. Consider enabling mpir. Terminating...\n");
    exit(0);
  }

  if ((1 + a*a - 2*b) < 0){
    printf("Polyhedron potential becomes imaginary. Consider enabling mpir. Terminating...\n");
    exit(0);
  }
  // Integral contribution of one segment
  double I = d*((c/K1)*(atan2(c*(1 - b), K1*K2) + atan2(c*b, a*K1)) + log((1 - b + K2)/(a-b)));
  return I;
}

double grav_potential_face_array(double r[3], double r_i[3], double n_i[3], double vertices[3][3]){
  /* Computes the gravitational potential of one face of the polyhedron */
  double I = 0;
  double K = 0;
  double r_i1[3];
  double r_i2[3];

  double I1, I2, I3;
  double K1, K2, K3;
  r_i1[0] = vertices[0][0];
  r_i1[1] = vertices[0][1];
  r_i1[2] = vertices[0][2];
  r_i2[0] = vertices[1][0];
  r_i2[1] = vertices[1][1];
  r_i2[2] = vertices[1][2];
  I1 = grav_potential_face_edge(r, r_i1, r_i2, n_i);
  K1 = K_factor(r, r_i1, r_i2, n_i, r_i);
  
  r_i1[0] = vertices[1][0];
  r_i1[1] = vertices[1][1];
  r_i1[2] = vertices[1][2];
  r_i2[0] = vertices[2][0];
  r_i2[1] = vertices[2][1];
  r_i2[2] = vertices[2][2];
  I2 = grav_potential_face_edge(r, r_i1, r_i2, n_i);
  K2 = K_factor(r, r_i1, r_i2, n_i, r_i);
  
  r_i1[0] = vertices[2][0];
  r_i1[1] = vertices[2][1];
  r_i1[2] = vertices[2][2];
  r_i2[0] = vertices[0][0];
  r_i2[1] = vertices[0][1];
  r_i2[2] = vertices[0][2];
  I3 = grav_potential_face_edge(r, r_i1, r_i2, n_i);
  K3 = K_factor(r, r_i1, r_i2, n_i, r_i);
  
  I = I1 + I2 + I3;
  K = K1 + K2 + K3;
 
  // Compute the gravitational potential
  double ri_r[3];
  ri_r[0] = r_i[0] - r[0];
  ri_r[1] = r_i[1] - r[1];
  ri_r[2] = r_i[2] - r[2];
  // Prefactor n*(r_i - r) 
  double dot_prod_n = dot_product(n_i, ri_r);
  double Phi = dot_prod_n*(I + K);
  if (isnan(Phi)){
    printf("Potential becomes NaN. Consider enabling mpir. Terminating... \n");
    exit(0);
  }
  if (isinf(Phi) > 0){
    printf("Potential is infinite. Consider enabling mpir. Terminating...\n");
    exit(0);
  }
  return Phi;
}

double grav_field_face_array(double r[3], double r_i[3], double n_i[3], double vertices[3][3], double g_field[3]){
  /* Computes the gravitational potential of one face of the polyhedron */
  double I = 0;
  double K = 0;
  double r_i1[3];
  double r_i2[3];

  double I1, I2, I3;
  double K1, K2, K3;
  r_i1[0] = vertices[0][0];
  r_i1[1] = vertices[0][1];
  r_i1[2] = vertices[0][2];
  r_i2[0] = vertices[1][0];
  r_i2[1] = vertices[1][1];
  r_i2[2] = vertices[1][2];
  I1 = grav_potential_face_edge(r, r_i1, r_i2, n_i);
  K1 = K_factor(r, r_i1, r_i2, n_i, r_i);
  
  r_i1[0] = vertices[1][0];
  r_i1[1] = vertices[1][1];
  r_i1[2] = vertices[1][2];
  r_i2[0] = vertices[2][0];
  r_i2[1] = vertices[2][1];
  r_i2[2] = vertices[2][2];
  I2 = grav_potential_face_edge(r, r_i1, r_i2, n_i);
  K2 = K_factor(r, r_i1, r_i2, n_i, r_i);

  r_i1[0] = vertices[2][0];
  r_i1[1] = vertices[2][1];
  r_i1[2] = vertices[2][2];
  r_i2[0] = vertices[0][0];
  r_i2[1] = vertices[0][1];
  r_i2[2] = vertices[0][2];
  I3 = grav_potential_face_edge(r, r_i1, r_i2, n_i);
  K3 = K_factor(r, r_i1, r_i2, n_i, r_i);

  I = I1 + I2 + I3;
  K = K1 + K2 + K3;

  double ri_r[3];
  ri_r[0] = r_i[0] - r[0];
  ri_r[1] = r_i[1] - r[1];
  ri_r[2] = r_i[2] - r[2];
  // Prefactor n*(r_i - r) 
  double dot_prod_n = dot_product(n_i, ri_r);
  double Phi = dot_prod_n*(I + K);
  if (isnan(Phi)){
    printf("Potential becomes NaN. Consider enabling mpir. Terminating... \n");
    exit(0);
  }
  if (isinf(Phi) > 0){
    printf("Potential is infinite. Consider enabling mpir. Terminating...\n");
    exit(0);
  }
  // Compute the gravitational field
  g_field[0] = n_i[0]*(I + K);
  g_field[1] = n_i[1]*(I + K);
  g_field[2] = n_i[2]*(I + K);
  return Phi;
}


double potential_polyhedron(double r[3], double **vertices, int nfacesB, int* face_ids_other){
  /* Computes the gravitational potential of one face of the polyhedron */
  int i, k;

  double Phi = 0;

  double n_i[3];
  double r_i[3];
  double vertices_in[3][3];
  int face_ids[3];
  // Input vertices must be oriented with the right hand orientation
  for (i=0; i<nfacesB; i++){
    k = 0;
    face_ids[0] = face_ids_other[3*i];
    face_ids[1] = face_ids_other[3*i+1];
    face_ids[2] = face_ids_other[3*i+2];
    r_i[0] = vertices[face_ids[0]][0];
    r_i[1] = vertices[face_ids[0]][1];
    r_i[2] = vertices[face_ids[0]][2];
    while (k>=0){
      if (k != face_ids[0] && k != face_ids[1] && k != face_ids[2]){
        int check = normal_vector_face_righthand(vertices, face_ids, k, n_i);
        if (check < 0){
          k += 1;
        }
        else{
          k = -1;
        }
      }
      else{
        k += 1;
      }
    }
    for (int ij = 0; ij < 3; ij++){
      vertices_in[0][ij] = vertices[face_ids[2]][ij];
      vertices_in[1][ij] = vertices[face_ids[1]][ij];
      vertices_in[2][ij] = vertices[face_ids[0]][ij];
    }    
    Phi += grav_potential_face_array(r, r_i, n_i, vertices_in);    
  }

  return Phi;
}


double grav_field_polyhedron(double r[3], double **vertices, int nfacesB, int* face_ids_other, double g_field[3], int use_omp){
  /* Computes the gravitational potential and field of one face of the polyhedron */
  int i, k;
  double Phi = 0;
  double gx = 0;
  double gy = 0;
  double gz = 0;
  
  if (use_omp == 1){
    #pragma omp parallel for reduction(+:Phi, gx, gy, gz)
    for (i=0; i<nfacesB; i++){
      double n_i[3];
      double r_i[3];
      double vertices_in[3][3];
      double g_field_in[3];
      int face_ids[3];
      k = 0;
      face_ids[0] = face_ids_other[3*i];
      face_ids[1] = face_ids_other[3*i+1];
      face_ids[2] = face_ids_other[3*i+2];
      r_i[0] = vertices[face_ids[0]][0];
      r_i[1] = vertices[face_ids[0]][1];
      r_i[2] = vertices[face_ids[0]][2];
      while (k>=0){
        if (k != face_ids[0] && k != face_ids[1] && k != face_ids[2]){
          int check = normal_vector_face_righthand(vertices, face_ids, k, n_i);
          if (check < 0){
            k += 1;
          }
          else{
            k = -1;
          }
        }
        else{
          k += 1;
        }
      }
      for (int ij = 0; ij < 3; ij++){
        vertices_in[0][ij] = vertices[face_ids[2]][ij];
        vertices_in[1][ij] = vertices[face_ids[1]][ij];
        vertices_in[2][ij] = vertices[face_ids[0]][ij];
      }    
      Phi += grav_field_face_array(r, r_i, n_i, vertices_in, g_field_in);
      gx += g_field_in[0];
      gy += g_field_in[1];
      gz += g_field_in[2];
    }
  }
  else{
    for (i=0; i<nfacesB; i++){
      double n_i[3];
      double r_i[3];
      double vertices_in[3][3];
      double g_field_in[3];
      int face_ids[3];
      k = 0;
      face_ids[0] = face_ids_other[3*i];
      face_ids[1] = face_ids_other[3*i+1];
      face_ids[2] = face_ids_other[3*i+2];
      r_i[0] = vertices[face_ids[0]][0];
      r_i[1] = vertices[face_ids[0]][1];
      r_i[2] = vertices[face_ids[0]][2];
      while (k>=0){
        if (k != face_ids[0] && k != face_ids[1] && k != face_ids[2]){
          int check = normal_vector_face_righthand(vertices, face_ids, k, n_i);
          if (check < 0){
            k += 1;
          }
          else{
            k = -1;
          }
        }
        else{
          k += 1;
        }
      }
      for (int ij = 0; ij < 3; ij++){
        vertices_in[0][ij] = vertices[face_ids[2]][ij];
        vertices_in[1][ij] = vertices[face_ids[1]][ij];
        vertices_in[2][ij] = vertices[face_ids[0]][ij];
      }    
      Phi += grav_field_face_array(r, r_i, n_i, vertices_in, g_field_in);
      gx += g_field_in[0];
      gy += g_field_in[1];
      gz += g_field_in[2];
    }
  }
  g_field[0] = gx;
  g_field[1] = gy;
  g_field[2] = gz;
  return Phi;
}

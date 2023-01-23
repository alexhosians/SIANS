#define _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>
#include "commonCfuncs.h"
#include <gsl/gsl_poly.h>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_sf_result.h>
#include <mpir.h>


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

    mpf_set_default_prec(512);
    mpf_t r2r1x, r2r1y, r2r1z;
    mpf_init_set_d(r2r1x, r2r1[0]);
    mpf_init_set_d(r2r1y, r2r1[1]);
    mpf_init_set_d(r2r1z, r2r1[2]);
    mpf_t r3r1x, r3r1y, r3r1z;
    mpf_init_set_d(r3r1x, r3r1[0]);
    mpf_init_set_d(r3r1y, r3r1[1]);
    mpf_init_set_d(r3r1z, r3r1[2]);
    mpf_t t1, t2;
    mpf_init(t1);
    mpf_init(t2);
    mpf_t cprodx, cprody, cprodz;
    mpf_init(cprodx);
    mpf_init(cprody);
    mpf_init(cprodz);
    mpf_mul(t1, r3r1y, r2r1z);
    mpf_mul(t2, r3r1z, r2r1y);
    mpf_sub(cprodx, t1, t2);
    mpf_mul(t1, r3r1z, r2r1x);
    mpf_mul(t2, r3r1x, r2r1z);
    mpf_sub(cprody, t1, t2);
    mpf_mul(t1, r3r1x, r2r1y);
    mpf_mul(t2, r3r1y, r2r1x);
    mpf_sub(cprodz, t1, t2);
    mpf_t cx, cy, cz;
    mpf_init(cx);
    mpf_init(cy);
    mpf_init(cz);
    mpf_mul(cx, cprodx, cprodx);
    mpf_mul(cy, cprody, cprody);
    mpf_mul(cz, cprodz, cprodz);
    mpf_add(t1, cx, cy);
    mpf_add(t2, t1, cz);
    mpf_t normcprod;
    mpf_init(normcprod);
    mpf_sqrt(normcprod, t2);
    mpf_t nix, niy, niz;
    mpf_init(nix);
    mpf_init(niy);
    mpf_init(niz);
    mpf_div(nix, cprodx, normcprod);
    mpf_div(niy, cprody, normcprod);
    mpf_div(niz, cprodz, normcprod);
      
    double nx = mpf_get_d(nix);
    double ny = mpf_get_d(niy);
    double nz = mpf_get_d(niz);
    mpf_clear(r2r1x);
    mpf_clear(r2r1y);
    mpf_clear(r2r1z);
    mpf_clear(r3r1x);
    mpf_clear(r3r1y);
    mpf_clear(r3r1z);
    mpf_clear(t1);
    mpf_clear(t2);
    mpf_clear(cprodx);
    mpf_clear(cprody);
    mpf_clear(cprodz);
    mpf_clear(cx);
    mpf_clear(cy);
    mpf_clear(cz); 
    mpf_clear(normcprod);
    mpf_clear(nix);
    mpf_clear(niy);
    mpf_clear(niz);
    double d = dot_product(ni_temp, r4r1);
    if (d < 0){
      n_i[0] = nx;
      n_i[1] = ny;
      n_i[2] = nz;;
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

int new_position_polyhedron_mpir(double datastuff[15], double Rot_A[3][3], double Rot_B_T[3][3], mpf_t newx_t, mpf_t newy_t, mpf_t newz_t){
  double u = datastuff[0];
  double v = datastuff[1];
  double xc = datastuff[2];
  double yc = datastuff[3];
  double zc = datastuff[4];
  double x1 = datastuff[5];
  double x2 = datastuff[6];
  double x3 = datastuff[7];
  double y1 = datastuff[8];
  double y2 = datastuff[9];
  double y3 = datastuff[10];
  double z1 = datastuff[11];
  double z2 = datastuff[12];
  double z3 = datastuff[13];

  mpf_set_default_prec(512);
  mpf_t vt, ut;
  mpf_init(vt);
  mpf_init(ut);
  mpf_set_d(ut, u);
  mpf_set_d(vt, v);
  mpf_t p1, p2;
  mpf_init(p1);
  mpf_init(p2);
  mpf_set_d(p1, 1-v);
  mpf_set_d(p2, 1-u);
  mpf_t x1t, x2t, x3t, y1t, y2t, y3t, z1t, z2t, z3t;
  mpf_init(x1t);
  mpf_init(x2t);
  mpf_init(x3t);
  mpf_init(y1t);
  mpf_init(y2t);
  mpf_init(y3t);
  mpf_init(z1t);
  mpf_init(z2t);
  mpf_init(z3t);
  mpf_set_d(x1t, x1);
  mpf_set_d(x2t, x2);
  mpf_set_d(x3t, x3);
  mpf_set_d(y1t, y1);
  mpf_set_d(y2t, y2);
  mpf_set_d(y3t, y3);
  mpf_set_d(z1t, z1);
  mpf_set_d(z2t, z2);
  mpf_set_d(z3t, z3);
  mpf_t xc_t, yc_t, zc_t;
  mpf_init(xc_t);
  mpf_init(yc_t);
  mpf_init(zc_t);
  mpf_set_d(xc_t, xc);
  mpf_set_d(yc_t, yc);
  mpf_set_d(zc_t, zc);
  mpf_t x_t, y_t, z_t;
  mpf_t t1, t21, t2, t31, t3, t4;
  mpf_init(x_t);
  mpf_init(y_t);
  mpf_init(z_t);
  mpf_init(t1);
  mpf_init(t2);
  mpf_init(t3);
  mpf_init(t21);
  mpf_init(t31);
  mpf_init(t4);
  // x
  mpf_mul(t1, p1, x1t);
  mpf_mul(t21, vt, ut);
  mpf_mul(t2, t21, x2t);
  mpf_mul(t31, vt, p2);
  mpf_mul(t3, t31, x3t);
  mpf_add(t4, t1, t2);
  mpf_add(x_t, t4, t3);
  // y
  mpf_mul(t1, p1, y1t);
  mpf_mul(t21, vt, ut);
  mpf_mul(t2, t21, y2t);
  mpf_mul(t31, vt, p2);
  mpf_mul(t3, t31, y3t);
  mpf_add(t4, t1, t2);
  mpf_add(y_t, t4, t3);
  // z
  mpf_mul(t1, p1, z1t);
  mpf_mul(t21, vt, ut);
  mpf_mul(t2, t21, z2t);
  mpf_mul(t31, vt, p2);
  mpf_mul(t3, t31, z3t);
  mpf_add(t4, t1, t2);
  mpf_add(z_t, t4, t3);

  mpf_t R00, R01, R02, R10, R11, R12, R20, R21, R22;
  mpf_init(R00);
  mpf_init(R01);
  mpf_init(R02);
  mpf_init(R10);
  mpf_init(R11);
  mpf_init(R12);
  mpf_init(R20);
  mpf_init(R21);
  mpf_init(R22);
  mpf_set_d(R00, Rot_A[0][0]);
  mpf_set_d(R01, Rot_A[0][1]);
  mpf_set_d(R02, Rot_A[0][2]);
  mpf_set_d(R10, Rot_A[1][0]);
  mpf_set_d(R11, Rot_A[1][1]);
  mpf_set_d(R12, Rot_A[1][2]);
  mpf_set_d(R20, Rot_A[2][0]);
  mpf_set_d(R21, Rot_A[2][1]);
  mpf_set_d(R22, Rot_A[2][2]);
  mpf_t f1, f2, f3, f4, f5, f6, f7, f8, f9;
  mpf_init(f1);
  mpf_init(f2);
  mpf_init(f3);
  mpf_init(f4);
  mpf_init(f5);
  mpf_init(f6);
  mpf_init(f7);
  mpf_init(f8);
  mpf_init(f9);
  mpf_mul(f1, x_t, R00);
  mpf_mul(f2, y_t, R01);
  mpf_mul(f3, z_t, R02);
  mpf_mul(f4, x_t, R10);
  mpf_mul(f5, y_t, R11);
  mpf_mul(f6, z_t, R12);
  mpf_mul(f7, x_t, R20);
  mpf_mul(f8, y_t, R21);
  mpf_mul(f9, z_t, R22);
  mpf_t ro1, ro2, ro3;
  mpf_t t1_, t2_;
  mpf_init(ro1);
  mpf_init(ro2);
  mpf_init(ro3);
  mpf_init(t1_);
  mpf_init(t2_);
  mpf_add(t1_, f1, f2);
  mpf_add(t2_, t1_, f3);
  mpf_sub(ro1, t2_, xc_t);
  mpf_add(t1_, f4, f5);
  mpf_add(t2_, t1_, f6);
  mpf_sub(ro2, t2_, yc_t);
  mpf_add(t1_, f7, f8);
  mpf_add(t2_, t1_, f9);
  mpf_sub(ro3, t2_, zc_t);
  mpf_set_d(R00, Rot_B_T[0][0]);
  mpf_set_d(R01, Rot_B_T[0][1]);
  mpf_set_d(R02, Rot_B_T[0][2]);
  mpf_set_d(R10, Rot_B_T[1][0]);
  mpf_set_d(R11, Rot_B_T[1][1]);
  mpf_set_d(R12, Rot_B_T[1][2]);
  mpf_set_d(R20, Rot_B_T[2][0]);
  mpf_set_d(R21, Rot_B_T[2][1]);
  mpf_set_d(R22, Rot_B_T[2][2]);
  
  mpf_mul(f1, ro1, R00);
  mpf_mul(f2, ro2, R01);
  mpf_mul(f3, ro3, R02);
  mpf_mul(f4, ro1, R10);
  mpf_mul(f5, ro2, R11);
  mpf_mul(f6, ro3, R12);
  mpf_mul(f7, ro1, R20);
  mpf_mul(f8, ro2, R21);
  mpf_mul(f9, ro3, R22);
  mpf_add(t1_, f1, f2);
  mpf_add(newx_t, t1_, f3);
  mpf_add(t1_, f4, f5);
  mpf_add(newy_t, t1_, f6);
  mpf_add(t1_, f7, f8);
  mpf_add(newz_t, t1_, f9);

  mpf_clear(vt);
  mpf_clear(ut);
  mpf_clear(p1);
  mpf_clear(p2);
  mpf_clear(x1t);
  mpf_clear(x2t);
  mpf_clear(x3t);
  mpf_clear(y1t);
  mpf_clear(y2t);
  mpf_clear(y3t);
  mpf_clear(z1t);
  mpf_clear(z2t);
  mpf_clear(z3t);
  mpf_clear(x_t);
  mpf_clear(y_t);
  mpf_clear(z_t);
  mpf_clear(xc_t);
  mpf_clear(yc_t);
  mpf_clear(zc_t);
  mpf_clear(t1);
  mpf_clear(t2);
  mpf_clear(t3);
  mpf_clear(t21);
  mpf_clear(t31);
  mpf_clear(t4);
  mpf_clear(R00);
  mpf_clear(R01);
  mpf_clear(R02);
  mpf_clear(R10);
  mpf_clear(R11);
  mpf_clear(R12);
  mpf_clear(R20);
  mpf_clear(R21);
  mpf_clear(R22);
  mpf_clear(f1);
  mpf_clear(f2);
  mpf_clear(f3);
  mpf_clear(f4);
  mpf_clear(f5);
  mpf_clear(f6);
  mpf_clear(f7);
  mpf_clear(f8);
  mpf_clear(f9);
  mpf_clear(ro1);
  mpf_clear(ro2);
  mpf_clear(ro3);
  mpf_clear(t1_);
  mpf_clear(t2_);
  return 0;
}

int new_position_ellipsoid_mpir(double datastuff[15], double Rot_A[3][3], double Rot_B_T[3][3], mpf_t newx_t, mpf_t newy_t, mpf_t newz_t){
  double z = datastuff[0];
  double alpha = datastuff[1];
  double a_A = datastuff[2];
  double b_A = datastuff[3];
  double c_A = datastuff[4];
  double xc = datastuff[5];
  double yc = datastuff[6];
  double zc = datastuff[7];
  double calpha = cos(alpha);
  double salpha = sin(alpha);
  mpf_set_default_prec(512);
  mpf_t a_t, b_t, c_t, z_t, salpha_t, calpha_t;
  mpf_init_set_d(a_t, a_A);
  mpf_init_set_d(b_t, b_A);
  mpf_init_set_d(c_t, c_A);
  mpf_init_set_d(z_t, z);
  mpf_init_set_d(salpha_t, salpha);
  mpf_init_set_d(calpha_t, calpha);

  mpf_t a_div_c, b_div_c, c_sq, c_min_z, p1, p2;
  mpf_init(a_div_c);
  mpf_init(b_div_c);
  mpf_init(c_sq);
  mpf_init(c_min_z);
  mpf_init(p1);
  mpf_init(p2);
  mpf_div(a_div_c, a_t, c_t);
  mpf_div(b_div_c, b_t, c_t);
  mpf_mul(c_sq, c_t, c_t);
  mpf_mul(p1, z_t, z_t);
  mpf_sub(c_min_z, c_sq, p1);
  mpf_sqrt(p2, c_min_z);
  mpf_t aprime, bprime;
  mpf_init(aprime);
  mpf_init(bprime);
  mpf_mul(aprime, a_div_c, p2);
  mpf_mul(bprime, b_div_c, p2);
  mpf_t x_t, y_t;
  mpf_init(x_t);
  mpf_init(y_t);
  mpf_mul(x_t, aprime, calpha_t);
  mpf_mul(y_t, bprime, salpha_t);

  mpf_t xc_t, yc_t, zc_t;
  mpf_init_set_d(xc_t, xc);
  mpf_init_set_d(yc_t, yc);
  mpf_init_set_d(zc_t, zc);

  mpf_t R00, R01, R02, R10, R11, R12, R20, R21, R22;
  mpf_init(R00);
  mpf_init(R01);
  mpf_init(R02);
  mpf_init(R10);
  mpf_init(R11);
  mpf_init(R12);
  mpf_init(R20);
  mpf_init(R21);
  mpf_init(R22);
  mpf_set_d(R00, Rot_A[0][0]);
  mpf_set_d(R01, Rot_A[0][1]);
  mpf_set_d(R02, Rot_A[0][2]);
  mpf_set_d(R10, Rot_A[1][0]);
  mpf_set_d(R11, Rot_A[1][1]);
  mpf_set_d(R12, Rot_A[1][2]);
  mpf_set_d(R20, Rot_A[2][0]);
  mpf_set_d(R21, Rot_A[2][1]);
  mpf_set_d(R22, Rot_A[2][2]);
  mpf_t f1, f2, f3, f4, f5, f6, f7, f8, f9;
  mpf_init(f1);
  mpf_init(f2);
  mpf_init(f3);
  mpf_init(f4);
  mpf_init(f5);
  mpf_init(f6);
  mpf_init(f7);
  mpf_init(f8);
  mpf_init(f9);
  mpf_mul(f1, x_t, R00);
  mpf_mul(f2, y_t, R01);
  mpf_mul(f3, z_t, R02);
  mpf_mul(f4, x_t, R10);
  mpf_mul(f5, y_t, R11);
  mpf_mul(f6, z_t, R12);
  mpf_mul(f7, x_t, R20);
  mpf_mul(f8, y_t, R21);
  mpf_mul(f9, z_t, R22);
  mpf_t ro1, ro2, ro3;
  mpf_t t1_, t2_;
  mpf_init(ro1);
  mpf_init(ro2);
  mpf_init(ro3);
  mpf_init(t1_);
  mpf_init(t2_);
  mpf_add(t1_, f1, f2);
  mpf_add(t2_, t1_, f3);
  mpf_sub(ro1, t2_, xc_t);
  mpf_add(t1_, f4, f5);
  mpf_add(t2_, t1_, f6);
  mpf_sub(ro2, t2_, yc_t);
  mpf_add(t1_, f7, f8);
  mpf_add(t2_, t1_, f9);
  mpf_sub(ro3, t2_, zc_t);
  mpf_set_d(R00, Rot_B_T[0][0]);
  mpf_set_d(R01, Rot_B_T[0][1]);
  mpf_set_d(R02, Rot_B_T[0][2]);
  mpf_set_d(R10, Rot_B_T[1][0]);
  mpf_set_d(R11, Rot_B_T[1][1]);
  mpf_set_d(R12, Rot_B_T[1][2]);
  mpf_set_d(R20, Rot_B_T[2][0]);
  mpf_set_d(R21, Rot_B_T[2][1]);
  mpf_set_d(R22, Rot_B_T[2][2]);
  
  mpf_mul(f1, ro1, R00);
  mpf_mul(f2, ro2, R01);
  mpf_mul(f3, ro3, R02);
  mpf_mul(f4, ro1, R10);
  mpf_mul(f5, ro2, R11);
  mpf_mul(f6, ro3, R12);
  mpf_mul(f7, ro1, R20);
  mpf_mul(f8, ro2, R21);
  mpf_mul(f9, ro3, R22);
  mpf_add(t1_, f1, f2);
  mpf_add(newx_t, t1_, f3);
  mpf_add(t1_, f4, f5);
  mpf_add(newy_t, t1_, f6);
  mpf_add(t1_, f7, f8);
  mpf_add(newz_t, t1_, f9);

  mpf_clear(a_t);
  mpf_clear(b_t);
  mpf_clear(c_t);
  mpf_clear(salpha_t);
  mpf_clear(calpha_t);
  mpf_clear(a_div_c);
  mpf_clear(b_div_c);
  mpf_clear(c_sq);
  mpf_clear(c_min_z);
  mpf_clear(aprime);
  mpf_clear(bprime);
  mpf_clear(p1);
  mpf_clear(p2);
  mpf_clear(x_t);
  mpf_clear(y_t);
  mpf_clear(z_t);
    
  mpf_clear(xc_t);
  mpf_clear(yc_t);
  mpf_clear(zc_t);
  mpf_clear(R00);
  mpf_clear(R01);
  mpf_clear(R02);
  mpf_clear(R10);
  mpf_clear(R11);
  mpf_clear(R12);
  mpf_clear(R20);
  mpf_clear(R21);
  mpf_clear(R22);
  mpf_clear(f1);
  mpf_clear(f2);
  mpf_clear(f3);
  mpf_clear(f4);
  mpf_clear(f5);
  mpf_clear(f6);
  mpf_clear(f7);
  mpf_clear(f8);
  mpf_clear(f9);
  mpf_clear(ro1);
  mpf_clear(ro2);
  mpf_clear(ro3);
  mpf_clear(t1_);
  mpf_clear(t2_);
  return 0;
}

int new_position_sphere_mpir(double datastuff[15], double Rot_A[3][3], double Rot_B_T[3][3], mpf_t newx_t, mpf_t newy_t, mpf_t newz_t){
  double xc = datastuff[5];
  double yc = datastuff[6];
  double zc = datastuff[7];

  mpf_t xc_t, yc_t, zc_t;
  mpf_init_set_d(xc_t, xc);
  mpf_init_set_d(yc_t, yc);
  mpf_init_set_d(zc_t, zc);
  mpf_t R00, R01, R02, R10, R11, R12, R20, R21, R22;
  mpf_init_set_d(R00, Rot_B_T[0][0]);
  mpf_init_set_d(R01, Rot_B_T[0][1]);
  mpf_init_set_d(R02, Rot_B_T[0][2]);
  mpf_init_set_d(R10, Rot_B_T[1][0]);
  mpf_init_set_d(R11, Rot_B_T[1][1]);
  mpf_init_set_d(R12, Rot_B_T[1][2]);
  mpf_init_set_d(R20, Rot_B_T[2][0]);
  mpf_init_set_d(R21, Rot_B_T[2][1]);
  mpf_init_set_d(R22, Rot_B_T[2][2]);
  mpf_t ro1, ro2, ro3, ro4;
  mpf_init(ro1);
  mpf_init(ro2);
  mpf_init(ro3);
  mpf_init(ro4);
  mpf_mul(ro1, R00, xc_t);
  mpf_mul(ro2, R01, yc_t);
  mpf_mul(ro3, R02, zc_t);
  mpf_add(ro4, ro1, ro2);
  mpf_add(newx_t, ro3, ro4);
  mpf_mul(ro1, R10, xc_t);
  mpf_mul(ro2, R11, yc_t);
  mpf_mul(ro3, R12, zc_t);
  mpf_add(ro4, ro1, ro2);
  mpf_add(newy_t, ro3, ro4);
  mpf_mul(ro1, R20, xc_t);
  mpf_mul(ro2, R21, yc_t);
  mpf_mul(ro3, R22, zc_t);
  mpf_add(ro4, ro1, ro2);
  mpf_add(newz_t, ro3, ro4);
  
  mpf_clear(xc_t);
  mpf_clear(yc_t);
  mpf_clear(zc_t);
  mpf_clear(R00);
  mpf_clear(R01);
  mpf_clear(R02);
  mpf_clear(R10);
  mpf_clear(R11);
  mpf_clear(R12);
  mpf_clear(R20);
  mpf_clear(R21);
  mpf_clear(R22);
  mpf_clear(ro1);
  mpf_clear(ro2);
  mpf_clear(ro3);
  mpf_clear(ro4);
  return 0;
}

double grav_pot_gmp(double r_i1[3], double ri2_ri1[3], double n_i[3], double datastuff[15], double Rot_A[3][3], double Rot_B_T[3][3]){
  mpf_t rx, ry, rz;
  mpf_init(rx);
  mpf_init(ry);
  mpf_init(rz);
  if (datastuff[14] >= 0){
    new_position_polyhedron_mpir(datastuff, Rot_A, Rot_B_T, rx, ry, rz);
  }
  else{
    if (datastuff[15] >= 0){
      new_position_ellipsoid_mpir(datastuff, Rot_A, Rot_B_T, rx, ry, rz);
    }
    else{
      new_position_sphere_mpir(datastuff, Rot_A, Rot_B_T, rx, ry, rz);
    }
  }
  mpf_set_default_prec(512);
  mpf_t n_x, n_y, n_z;
  mpf_init(n_x);
  mpf_init(n_y);
  mpf_init(n_z);
  mpf_set_d(n_x, n_i[0]);
  mpf_set_d(n_y, n_i[1]);
  mpf_set_d(n_z, n_i[2]);
  mpf_t ri1x, ri1y, ri1z;
  mpf_init(ri1x);
  mpf_init(ri1y);
  mpf_init(ri1z);
  mpf_set_d(ri1x, r_i1[0]);
  mpf_set_d(ri1y, r_i1[1]);
  mpf_set_d(ri1z, r_i1[2]);
  mpf_t r_ri1x, r_ri1y, r_ri1z;
  mpf_init(r_ri1x);
  mpf_init(r_ri1y);
  mpf_init(r_ri1z);
  mpf_sub(r_ri1x, rx, ri1x);
  mpf_sub(r_ri1y, ry, ri1y);
  mpf_sub(r_ri1z, rz, ri1z);
  mpf_t r_ri1x_sq, r_ri1y_sq, r_ri1z_sq;
  mpf_init(r_ri1x_sq);
  mpf_init(r_ri1y_sq);
  mpf_init(r_ri1z_sq);
  mpf_mul(r_ri1x_sq, r_ri1x, r_ri1x);
  mpf_mul(r_ri1y_sq, r_ri1y, r_ri1y);
  mpf_mul(r_ri1z_sq, r_ri1z, r_ri1z);
  mpf_t ri2_ri1x, ri2_ri1y, ri2_ri1z;
  mpf_init(ri2_ri1x);
  mpf_init(ri2_ri1y);
  mpf_init(ri2_ri1z); 
  mpf_set_d(ri2_ri1x, ri2_ri1[0]);
  mpf_set_d(ri2_ri1y, ri2_ri1[1]);
  mpf_set_d(ri2_ri1z, ri2_ri1[2]);
  mpf_t ri2_ri1x_sq, ri2_ri1y_sq, ri2_ri1z_sq;
  mpf_init(ri2_ri1x_sq);
  mpf_init(ri2_ri1y_sq);
  mpf_init(ri2_ri1z_sq);
  mpf_mul(ri2_ri1x_sq, ri2_ri1x, ri2_ri1x);
  mpf_mul(ri2_ri1y_sq, ri2_ri1y, ri2_ri1y);
  mpf_mul(ri2_ri1z_sq, ri2_ri1z, ri2_ri1z);
  // Common denominator squared
  mpf_t ri2_ri1_abs_f, ri2_ri1_abs_f_sq, pre_f1, pre_f2, ri2_ri1_abs_f_sqrt;
  mpf_init(ri2_ri1_abs_f);
  mpf_init(ri2_ri1_abs_f_sq);
  mpf_init(ri2_ri1_abs_f_sqrt);
  mpf_init(pre_f1);
  mpf_init(pre_f2);
  mpf_add(pre_f1, ri2_ri1x_sq, ri2_ri1y_sq);
  mpf_add(ri2_ri1_abs_f, pre_f1, ri2_ri1z_sq);
  mpf_mul(ri2_ri1_abs_f_sq, ri2_ri1_abs_f, ri2_ri1_abs_f);
  mpf_sqrt(ri2_ri1_abs_f_sqrt, ri2_ri1_abs_f);
  // For a_ij^2
  mpf_t abs_a_f, a_f, a_f_sq;
  mpf_init(abs_a_f);
  mpf_init(a_f);
  mpf_init(a_f_sq);
  mpf_add(pre_f1, r_ri1x_sq, r_ri1y_sq);
  mpf_add(pre_f2, pre_f1, r_ri1z_sq);
  mpf_sqrt(abs_a_f, pre_f2);
  mpf_div(a_f, abs_a_f, ri2_ri1_abs_f_sqrt);
  mpf_div(a_f_sq, pre_f2, ri2_ri1_abs_f);
  // For b_ij^2
  mpf_t dot_prod_b_f, dbx, dby, dbz, b_f, b_f_sq;
  mpf_init(dot_prod_b_f);
  mpf_init(dbx);
  mpf_init(dby);
  mpf_init(dbz);
  mpf_init(b_f);
  mpf_init(b_f_sq);
  mpf_mul(dbx, r_ri1x, ri2_ri1x);
  mpf_mul(dby, r_ri1y, ri2_ri1y);
  mpf_mul(dbz, r_ri1z, ri2_ri1z);
  mpf_add(pre_f1, dbx, dby);
  mpf_add(dot_prod_b_f, pre_f1, dbz);
  mpf_div(b_f, dot_prod_b_f, ri2_ri1_abs_f);
  mpf_mul(b_f_sq, b_f, b_f);
  // For c_ij
  mpf_t dot_prod_c_f, c_f, c_f_sq;
  mpf_init(dot_prod_c_f);
  mpf_init(c_f);
  mpf_init(c_f_sq);
  mpf_mul(dbx, n_x, r_ri1x);
  mpf_mul(dby, n_y, r_ri1y);
  mpf_mul(dbz, n_z, r_ri1z);
  mpf_add(pre_f1, dbx, dby);
  mpf_add(dot_prod_c_f, pre_f1, dbz);
  mpf_div(c_f, dot_prod_c_f, ri2_ri1_abs_f_sqrt);
  mpf_mul(c_f_sq, c_f, c_f);
  // For d_ij
  mpf_t nxr_x, nxr_y, nxr_z;
  mpf_init(nxr_x);
  mpf_init(nxr_y);
  mpf_init(nxr_z);
  //// components of the cross product
  mpf_t nxr_x_t1, nxr_x_t2;
  mpf_init(nxr_x_t1);
  mpf_init(nxr_x_t2);
  mpf_mul(nxr_x_t1, n_y, r_ri1z);
  mpf_mul(nxr_x_t2, n_z, r_ri1y);
  mpf_sub(nxr_x, nxr_x_t1, nxr_x_t2);
  mpf_t nxr_y_t1, nxr_y_t2;
  mpf_init(nxr_y_t1);
  mpf_init(nxr_y_t2);
  mpf_mul(nxr_y_t1, n_z, r_ri1x);
  mpf_mul(nxr_y_t2, n_x, r_ri1z);
  mpf_sub(nxr_y, nxr_y_t1, nxr_y_t2);
  mpf_t nxr_z_t1, nxr_z_t2;
  mpf_init(nxr_z_t1);
  mpf_init(nxr_z_t2);
  mpf_mul(nxr_z_t1, n_x, r_ri1y);
  mpf_mul(nxr_z_t2, n_y, r_ri1x);
  mpf_sub(nxr_z, nxr_z_t1, nxr_z_t2);
  // Actual d_ij
  mpf_t dot_prod_d_f, d_f;
  mpf_init(dot_prod_d_f);
  mpf_init(d_f);
  mpf_mul(dbx, nxr_x, ri2_ri1x);
  mpf_mul(dby, nxr_y, ri2_ri1y);
  mpf_mul(dbz, nxr_z, ri2_ri1z);
  mpf_add(pre_f1, dbx, dby);
  mpf_add(pre_f2, pre_f1, dbz);
  mpf_neg(dot_prod_d_f, pre_f2);
  mpf_div(d_f, dot_prod_d_f, ri2_ri1_abs_f_sqrt);
  double a = mpf_get_d(a_f);
  double b = mpf_get_d(b_f);
  double c = mpf_get_d(c_f);
  double d = mpf_get_d(d_f);
  // For K1
  mpf_t ksum1, ksum2;
  mpf_init(ksum1);
  mpf_init(ksum2);
  mpf_sub(ksum1, a_f_sq, b_f_sq);
  mpf_sub(ksum2, ksum1, c_f_sq);
  double ksum_get = mpf_get_d(ksum2);
  // Clear all variables
  mpf_clear(rx);
  mpf_clear(ry);
  mpf_clear(rz);
  mpf_clear(n_x);
  mpf_clear(n_y);
  mpf_clear(n_z);
  mpf_clear(ri1x);
  mpf_clear(ri1y);
  mpf_clear(ri1z);
  mpf_clear(r_ri1x);
  mpf_clear(r_ri1y);
  mpf_clear(r_ri1z);
  mpf_clear(r_ri1x_sq);
  mpf_clear(r_ri1y_sq);
  mpf_clear(r_ri1z_sq);
  mpf_clear(ri2_ri1x);
  mpf_clear(ri2_ri1y);
  mpf_clear(ri2_ri1z);
  mpf_clear(ri2_ri1x_sq);
  mpf_clear(ri2_ri1y_sq);
  mpf_clear(ri2_ri1z_sq);
  mpf_clear(ri2_ri1_abs_f);
  mpf_clear(ri2_ri1_abs_f_sq);
  mpf_clear(ri2_ri1_abs_f_sqrt);
  mpf_clear(pre_f1);
  mpf_clear(pre_f2);
  mpf_clear(abs_a_f);
  mpf_clear(a_f);
  mpf_clear(a_f_sq);
  mpf_clear(dot_prod_b_f);
  mpf_clear(dbx);
  mpf_clear(dby);
  mpf_clear(dbz);
  mpf_clear(b_f);
  mpf_clear(b_f_sq);
  mpf_clear(dot_prod_c_f);
  mpf_clear(c_f);
  mpf_clear(c_f_sq);
  mpf_clear(nxr_x);
  mpf_clear(nxr_y);
  mpf_clear(nxr_z);
  mpf_clear(nxr_x_t1);
  mpf_clear(nxr_x_t2);
  mpf_clear(nxr_y_t1);
  mpf_clear(nxr_y_t2);
  mpf_clear(nxr_z_t1);
  mpf_clear(nxr_z_t2);
  mpf_clear(dot_prod_d_f);
  mpf_clear(d_f);
  mpf_clear(ksum1);
  if (fabs(d) <= 0){
    mpf_clear(ksum2);
    return 0;
  }
  double K1;
  if (ksum_get <= 0 && fabs(d) > 0){
    K1 = sqrt(a*a - b*b - c*c);
    mpf_clear(ksum2);
  }
  else{
    mpf_t K1t;
    mpf_init(K1t);
    mpf_sqrt(K1t, ksum2);
    K1 = mpf_get_d(K1t);
    mpf_clear(K1t);
    mpf_clear(ksum2); 
  }
  //double K1 = sqrt(a*a - b*b - c*c);
  double K2 = sqrt(1 + a*a - 2*b);

  double I = d*((c/K1)*(atan2(c*(1 - b), K1*K2) + atan2(c*b, a*K1)) + log((1 - b + K2)/(a-b)));
  return I;
}

double grav_potential_face_edge_mpir(double r[3], double r_i1[3], double r_i2[3], double n_i[3], int trigger[1], double datastuff[15],
 double Rot_A[3][3], double Rot_B_T[3][3]){
  /*
  Gravitational potential contribution from one segment of a face.
  Loop this over vertices j.
  */
  // Some prefactors needed for the a,b,c,d constants
  double ri2_ri1[3];
  ri2_ri1[0] = r_i2[0] - r_i1[0];
  ri2_ri1[1] = r_i2[1] - r_i1[1];
  ri2_ri1[2] = r_i2[2] - r_i1[2];
  double r_ri1[3];
  r_ri1[0] = r[0] - r_i1[0];
  r_ri1[1] = r[1] - r_i1[1];
  r_ri1[2] = r[2] - r_i1[2];
  double abs_a = absolute_value_vector(r_ri1);
  double ri2_ri1_abs = absolute_value_vector(ri2_ri1);
  double dot_prod_b = dot_product(r_ri1, ri2_ri1);
  double dot_prod_c = dot_product(n_i, r_ri1);
  //printf("r-ri = [%.3f, %.3f, %.3f] \n", r_ri1[0], r_ri1[1], r_ri1[2]);
  double n_cross_rri[3];
  cross_product(n_i, r_ri1, n_cross_rri);
  double dot_prod_d = dot_product(n_cross_rri, ri2_ri1);

  // The constants a_ij, b_ij, c_ij, d_ij and prefactors K1, K2
  double a = abs_a/ri2_ri1_abs;
  double b = dot_prod_b/(ri2_ri1_abs*ri2_ri1_abs);
  double c = dot_prod_c/ri2_ri1_abs;
  double d = -dot_prod_d/ri2_ri1_abs;
  double I;
  if (a*a - b*b - c*c <= 0 && trigger[0] == 0){
    I = grav_pot_gmp(r_i1, ri2_ri1, n_i, datastuff, Rot_A, Rot_B_T);
    trigger[0] = 1;
  }
  else{
    //double K1 = sqrt(fabs(a*a - b*b - c*c));
    if (trigger[0] == 1){
      I = grav_pot_gmp(r_i1, ri2_ri1, n_i, datastuff, Rot_A, Rot_B_T); 
    }
    else{
      double K1 = sqrt(a*a - b*b - c*c);
      double K2 = sqrt(1 + a*a - 2*b);
      I = d*((c/K1)*(atan2(c*(1 - b), K1*K2) + atan2(c*b, a*K1)) + log((1 - b + K2)/(a-b)));
    }
  }
  
  if ((1 + a*a - 2*b) < 0){
    printf("Polyhedron potential becomes imaginary! Terminating... \n");
    exit(0);
  }
  return I;
}

double grav_potential_face_array_mpir(double r[3], double r_i[3], double n_i[3], double vertices[3][3], int trigger[1], double datastuff[15], 
  double Rot_A[3][3], double Rot_B_T[3][3]){
  /* Computes the gravitational potential of one face of the polyhedron */
  int nvertices = 3; // Set this as input argument in the future for general polyhedron
  double r_i1[3];
  double r_i2[3];

  r_i1[0] = vertices[0][0];
  r_i1[1] = vertices[0][1];
  r_i1[2] = vertices[0][2];
  r_i2[0] = vertices[1][0];
  r_i2[1] = vertices[1][1];
  r_i2[2] = vertices[1][2];
  double I1 = grav_potential_face_edge_mpir(r, r_i1, r_i2, n_i, trigger, datastuff, Rot_A, Rot_B_T);
  double K1 = K_factor(r, r_i1, r_i2, n_i, r_i);
  
  r_i1[0] = vertices[1][0];
  r_i1[1] = vertices[1][1];
  r_i1[2] = vertices[1][2];
  r_i2[0] = vertices[2][0];
  r_i2[1] = vertices[2][1];
  r_i2[2] = vertices[2][2];
  double I2 = grav_potential_face_edge_mpir(r, r_i1, r_i2, n_i, trigger, datastuff, Rot_A, Rot_B_T);
  double K2 = K_factor(r, r_i1, r_i2, n_i, r_i);
  
  r_i1[0] = vertices[2][0];
  r_i1[1] = vertices[2][1];
  r_i1[2] = vertices[2][2];
  r_i2[0] = vertices[0][0];
  r_i2[1] = vertices[0][1];
  r_i2[2] = vertices[0][2];
  double I3 = grav_potential_face_edge_mpir(r, r_i1, r_i2, n_i, trigger, datastuff, Rot_A, Rot_B_T);
  double K3 = K_factor(r, r_i1, r_i2, n_i, r_i);

  double I = I1 + I2 + I3;
  double K = K1 + K2 + K3;

 
  // Compute the gravitational potential
  double ri_r[3];
  ri_r[0] = r_i[0] - r[0];
  ri_r[1] = r_i[1] - r[1];
  ri_r[2] = r_i[2] - r[2];
  // Prefactor n*(r_i - r) 
  double dot_prod_n = dot_product(n_i, ri_r);
  double Phi = dot_prod_n*(I + K);
  if (isnan(Phi)){
    printf("Potential becomes NaN. Terminating... \n");
    exit(0);
  }
  if (isinf(Phi) > 0){
    printf("Potential is infinite. Terminating... \n");
    exit(0);
  }
  return Phi;
}


double grav_field_face_array_mpir(double r[3], double r_i[3], double n_i[3], double vertices[3][3], double g_field[3], 
  int trigger[1], double datastuff[15], double Rot_A[3][3], double Rot_B_T[3][3]){
  /* Computes the gravitational potential of one face of the polyhedron */
  double r_i1[3];
  double r_i2[3];
  r_i1[0] = vertices[0][0];
  r_i1[1] = vertices[0][1];
  r_i1[2] = vertices[0][2];
  r_i2[0] = vertices[1][0];
  r_i2[1] = vertices[1][1];
  r_i2[2] = vertices[1][2];
  double I1 = grav_potential_face_edge_mpir(r, r_i1, r_i2, n_i, trigger, datastuff, Rot_A, Rot_B_T);
  double K1 = K_factor(r, r_i1, r_i2, n_i, r_i);
  r_i1[0] = vertices[1][0];
  r_i1[1] = vertices[1][1];
  r_i1[2] = vertices[1][2];
  r_i2[0] = vertices[2][0];
  r_i2[1] = vertices[2][1];
  r_i2[2] = vertices[2][2];
  double I2 = grav_potential_face_edge_mpir(r, r_i1, r_i2, n_i, trigger, datastuff, Rot_A, Rot_B_T);
  double K2 = K_factor(r, r_i1, r_i2, n_i, r_i);
  r_i1[0] = vertices[2][0];
  r_i1[1] = vertices[2][1];
  r_i1[2] = vertices[2][2];
  r_i2[0] = vertices[0][0];
  r_i2[1] = vertices[0][1];
  r_i2[2] = vertices[0][2];
  double I3 = grav_potential_face_edge_mpir(r, r_i1, r_i2, n_i, trigger, datastuff, Rot_A, Rot_B_T);
  double K3 = K_factor(r, r_i1, r_i2, n_i, r_i);
  double I = I1 + I2 + I3;
  double K = K1 + K2 + K3;
 
  // Compute the gravitational field
  g_field[0] = n_i[0]*(I + K);
  g_field[1] = n_i[1]*(I + K);
  g_field[2] = n_i[2]*(I + K);
  // Compute gravitational potential
  double ri_r[3];
  ri_r[0] = r_i[0] - r[0];
  ri_r[1] = r_i[1] - r[1];
  ri_r[2] = r_i[2] - r[2];
  // Prefactor n*(r_i - r) 
  double dot_prod_n = dot_product(n_i, ri_r);
  double Phi = dot_prod_n*(I + K);
  return Phi;
}


double potential_polyhedron_mpir(double r[3], double **vertices, int nfacesB, int* face_ids_other, int trigger[1],
  double datastuff[14], double Rot_A[3][3], double Rot_B_T[3][3]){
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
    Phi += grav_potential_face_array_mpir(r, r_i, n_i, vertices_in, trigger, datastuff, Rot_A, Rot_B_T);    
  }

  return Phi;
}

double grav_field_polyhedron_mpir(double r[3], double **vertices, int nfacesB, int* face_ids_other, double g_field[3], int trigger[1],
  double datastuff[14], double Rot_A[3][3], double Rot_B_T[3][3], int use_omp){
  /* Computes the gravitational potential and field of one face of the polyhedron */
  int i, k;
  double Phi = 0;
  // Input vertices must be oriented with the right hand orientation
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
      Phi += grav_field_face_array_mpir(r, r_i, n_i, vertices_in, g_field_in, trigger, datastuff, Rot_A, Rot_B_T);
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
      Phi += grav_field_face_array_mpir(r, r_i, n_i, vertices_in, g_field_in, trigger, datastuff, Rot_A, Rot_B_T);
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


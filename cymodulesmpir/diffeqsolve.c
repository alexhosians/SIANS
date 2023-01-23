/*
This C file contains the functions required to solve the differential equations.
This includes force and torque computations and ODEs for angular velocity and angles.
Functions for a 2-body and N-body are included.
*/
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include "commonCfuncs.h"
#include "SurfaceIntegrals.h"
#include <time.h>
#include <omp.h>
#include <mpir.h>

typedef struct{
  double c6[6];
} array6size;

typedef struct{
  double c11[11];
} array11size;

int angular_acceleration_ode_ellipsoid(double I[3][3], double omega_x, double omega_y, double omega_z, 
  double torq_x, double torq_y, double torq_z, double results[3]){
  /*
  Ordinary differential equations for the angular velocity.
  Function specific for ellipsoids.
  */
  double I_11 = I[0][0];
  double I_22 = I[1][1];
  double I_33 = I[2][2];
  results[0] = (torq_x - (I_33 - I_22)*omega_y*omega_z)/I_11;
  results[1] = (torq_y - (I_11 - I_33)*omega_x*omega_z)/I_22;
  results[2] = (torq_z - (I_22 - I_11)*omega_x*omega_y)/I_33;
  return 0;
}

int angular_acceleration_ode_general(double I[3][3], double omega_x, double omega_y, double omega_z, 
  double torq_x, double torq_y, double torq_z, double results[3]){
  /*
  Ordinary differential equations for the angular velocity.
  Applies for any general body with a moment of inertia with non-zero non-diagonal elements.
  Mainly used for polyhedra
  */
  double detI = Matrix_determinant_3x3(I);
  double I_00 = I[0][0];
  double I_01 = I[0][1];
  double I_02 = I[0][2];
  double I_10 = I[1][0];
  double I_11 = I[1][1];
  double I_12 = I[1][2];
  double I_20 = I[2][0];
  double I_21 = I[2][1];
  double I_22 = I[2][2];
  
  // Inverse matrix
  double invI_00 = (I_11*I_22 - I_12*I_21);
  double invI_01 = (I_02*I_21 - I_01*I_22);
  double invI_02 = (I_01*I_12 - I_02*I_11);
  double invI_10 = (I_12*I_20 - I_10*I_22);
  double invI_11 = (I_00*I_22 - I_02*I_20);
  double invI_12 = (I_02*I_10 - I_00*I_12);
  double invI_20 = (I_10*I_21 - I_11*I_20);
  double invI_21 = (I_01*I_20 - I_00*I_21);
  double invI_22 = (I_00*I_11 - I_01*I_10);
  
  double A_x = I_00*omega_x + I_01*omega_y + I_02*omega_z;
  double A_y = I_10*omega_x + I_11*omega_y + I_12*omega_z;
  double A_z = I_20*omega_x + I_21*omega_y + I_22*omega_z;

  double Q_x = torq_x - omega_y*A_z + omega_z*A_y;
  double Q_y = torq_y - omega_z*A_x + omega_x*A_z;
  double Q_z = torq_z - omega_x*A_y + omega_y*A_x;
  // Return derivative of angular velocities
  results[0] = (invI_00*Q_x + invI_01*Q_y + invI_02*Q_z)/detI;
  results[1] = (invI_10*Q_x + invI_11*Q_y + invI_12*Q_z)/detI;
  results[2] = (invI_20*Q_x + invI_21*Q_y + invI_22*Q_z)/detI;
  return 0;
}

double angular_speed_ode(double p, double q, double r, double phi, double theta, int comp){
  /* 
  ODE for the angular speeds of phi, theta and psi 
  p, q, r represents local angular velocity in around the x,y and z axes respectively.
  */
  double rhs_ode;
  if (comp == 1){
    rhs_ode = p + (q*sin(phi) + r*cos(phi))*tan(theta);
  }
  else if (comp == 2){
    rhs_ode = q*cos(phi) - r*sin(phi);
  }
  else if (comp == 3){
    rhs_ode = (q*sin(phi) + r*cos(phi))/cos(theta);
  }
  else{
    printf("Component of angular_speed_ode() not set properly! \n");
    exit(1);
  }
  return rhs_ode;
}

double angular_speed_euler_param(double p, double q, double r, double e0, double e1, double e2, double e3, int comp){
  /* 
  ODE for the angular speeds of the euler parameters e0, e1, e2 and e3 
  p, q, r represents local angular velocity in around the x, y and z axes respectively.
  */
  double rhs_ode;
  if (comp == 1){
    rhs_ode = 0.5*(-e1*p - e2*q - e3*r);
  }
  else if (comp == 2){
    rhs_ode = 0.5*(e0*p - e3*q + e2*r);
  }
  else if (comp == 3){
    rhs_ode = 0.5*(e3*p + e0*q - e1*r);
  }
  else if (comp == 4){
    rhs_ode = 0.5*(-e2*p + e1*q + e0*r);
  }
  else{
    printf("Component of angular_speed_euler_param() not set properly! \n");
    exit(1);
  }
  return rhs_ode;
}

int ode_solver_2body(double t, double* y, double* dfdt, double* params_input, double* semiaxes, double itol[3], double *vertices1D,
  int *face_ids, double *moment_inertia, int triggers[3]){
  /*
  Two-body solver
  */
  // Other properties
  double mass_A = params_input[0];
  double mass_B = params_input[1];
  double rho_A = params_input[2];
  double rho_B = params_input[3];
  double radius_A = params_input[4];
  double radius_B = params_input[5]; 
  int nfaces_A = params_input[6];
  int nfaces_B = params_input[7];
  int nvertices_A = params_input[8];
  int nvertices_B = params_input[9];
  int nvertices = nvertices_A + nvertices_B;
  int nfaces = nfaces_A + nfaces_B;
  double G_grav = params_input[10];
  
  // Velocities
  double vx_A = y[0];
  double vy_A = y[1];
  double vz_A = y[2];
  double vx_B = y[3];
  double vy_B = y[4];
  double vz_B = y[5];
  // Positions
  double x_A = y[6];
  double y_A = y[7];
  double z_A = y[8];
  double x_B = y[9];
  double y_B = y[10];
  double z_B = y[11];
  double xc_BA = x_B - x_A;
  double yc_BA = y_B - y_A;
  double zc_BA = z_B - z_A;
  double xc_AB = x_A - x_B;
  double yc_AB = y_A - y_B;
  double zc_AB = z_A - z_B;
  // Angular speeds
  double p_A = y[12];
  double q_A = y[13];
  double r_A = y[14];
  double p_B = y[15];
  double q_B = y[16];
  double r_B = y[17];
  // Semiaxes
  double a_A = semiaxes[0];
  double b_A = semiaxes[1];
  double c_A = semiaxes[2];
  double a_B = semiaxes[3];
  double b_B = semiaxes[4];
  double c_B = semiaxes[5];
  // For force computation
  array6size semiaxes_in1;
  array6size semiaxes_in2;
  semiaxes_in1 = (array6size){a_A, b_A, c_A, a_B, b_B, c_B};
  semiaxes_in2 = (array6size){a_B, b_B, c_B, a_A, b_A, c_A};
  // Other triggers
  int eulerparam = triggers[0];
  int include_sun = triggers[1];
  // Rotation angles and rotation matrix of body A
  double Rot_A[3][3];
  array11size input1;
  array11size input2;
  // Also computes angular speed ODE here
  double angular_speeds_derivative[8];
  if (eulerparam){
    double e0_A = y[18];
    double e1_A = y[19];
    double e2_A = y[20];
    double e3_A = y[21];
    double e0_B = y[22];
    double e1_B = y[23];
    double e2_B = y[24];
    double e3_B = y[25];
    Rotation_matrix_Euler_param(e0_A, e1_A, e2_A, e3_A, Rot_A, 0);
    input1 = (array11size){xc_BA, yc_BA, zc_BA, e0_A, e1_A, e2_A, e3_A, e0_B, e1_B, e2_B, e3_B};
    input2 = (array11size){xc_AB, yc_AB, zc_AB, e0_B, e1_B, e2_B, e3_B, e0_A, e1_A, e2_A, e3_A};
    
    angular_speeds_derivative[0] = angular_speed_euler_param(p_A, q_A, r_A, e0_A, e1_A, e2_A, e3_A, 1);
    angular_speeds_derivative[1] = angular_speed_euler_param(p_A, q_A, r_A, e0_A, e1_A, e2_A, e3_A, 2);
    angular_speeds_derivative[2] = angular_speed_euler_param(p_A, q_A, r_A, e0_A, e1_A, e2_A, e3_A, 3);
    angular_speeds_derivative[3] = angular_speed_euler_param(p_A, q_A, r_A, e0_A, e1_A, e2_A, e3_A, 4);
    angular_speeds_derivative[4] = angular_speed_euler_param(p_B, q_B, r_B, e0_B, e1_B, e2_B, e3_B, 1);
    angular_speeds_derivative[5] = angular_speed_euler_param(p_B, q_B, r_B, e0_B, e1_B, e2_B, e3_B, 2);
    angular_speeds_derivative[6] = angular_speed_euler_param(p_B, q_B, r_B, e0_B, e1_B, e2_B, e3_B, 3);
    angular_speeds_derivative[7] = angular_speed_euler_param(p_B, q_B, r_B, e0_B, e1_B, e2_B, e3_B, 4);
  }
  else{
    double phi_A = y[18];
    double theta_A = y[19];
    double psi_A = y[20];
    double phi_B = y[21];
    double theta_B = y[22];
    double psi_B = y[23];
    Rotation_matrix_components(phi_A, theta_A, psi_A, Rot_A, 0);
    input1 = (array11size){xc_BA, yc_BA, zc_BA, phi_A, theta_A, psi_A, phi_B, theta_B, psi_B, 0, 0};
    input2 = (array11size){xc_AB, yc_AB, zc_AB, phi_B, theta_B, psi_B, phi_A, theta_A, psi_A, 0, 0};

    angular_speeds_derivative[0] = angular_speed_ode(p_A, q_A, r_A, phi_A, theta_A, 1);
    angular_speeds_derivative[1] = angular_speed_ode(p_A, q_A, r_A, phi_A, theta_A, 2);
    angular_speeds_derivative[2] = angular_speed_ode(p_A, q_A, r_A, phi_A, theta_A, 3);
    angular_speeds_derivative[3] = angular_speed_ode(p_B, q_B, r_B, phi_B, theta_B, 1);
    angular_speeds_derivative[4] = angular_speed_ode(p_B, q_B, r_B, phi_B, theta_B, 2);
    angular_speeds_derivative[5] = angular_speed_ode(p_B, q_B, r_B, phi_B, theta_B, 3);
  }
  int i, j;
  // Sort vertices into a 2D matrix
  double **vertices = (double **)malloc(nvertices * sizeof(double *));
  for (i=0; i<nvertices; i++){
    vertices[i] = (double*)malloc(3 * sizeof(double));
    for (j=0; j<3; j++){
      vertices[i][j] = vertices1D[3*i+j];
    }
  }
  
  double prefac_A;
  double prefac_B;
  int A_is_sphere = 0;
  int B_is_sphere = 0;
  if (nvertices_B > 0){
    // If body B is a tetrahedron
    prefac_A = rho_A*rho_B*G_grav;
  }
  else{
    // If body B is a sphere
    if (fabs(a_B - b_B) < 1e-15 && fabs(a_B - c_B) < 1e-15){
      prefac_A = rho_A*mass_B*G_grav;
      B_is_sphere = 1;
    }
    // If body B is an ellipsoid
    else{
      prefac_A = rho_A*rho_B*G_grav;
    }
  }

  if (nvertices_A > 0){
    prefac_B = prefac_A;
  }
  else{
    // If body A is a sphere
    if (fabs(a_A - b_A) < 1e-15 && fabs(a_A - c_A) < 1e-15){
      A_is_sphere = 1;
      prefac_B = mass_A*rho_B*G_grav;
      Rotation_matrix_components(0, 0, 0, Rot_A, 0);
    }
    // If body A is an ellipsoid
    else{
      prefac_B = prefac_A;
    }
  }
  if (nfaces_A > 0){
    A_is_sphere = 0;
  }
  if (nfaces_B > 0){
    B_is_sphere = 0;
  }
  int collide = 0;
  int return_value = 0;
  if (sqrt(xc_BA*xc_BA + yc_BA*yc_BA + zc_BA*zc_BA) <= (radius_A + radius_B)){
    if (nvertices_A > 0 && B_is_sphere){
      // Collision check for a primary polyhedron and secondary sphere
      double position_B[3] = {xc_BA, yc_BA, zc_BA};
      collide = polyhedron_sphere_intersection_simple(vertices, nvertices_A, position_B, radius_B);
    }
    else if (nvertices_B > 0 && A_is_sphere){
      // Collision check for a secondary polyhedron and primary sphere
      double position_A[3] = {xc_BA, yc_BA, zc_BA};
      collide = polyhedron_sphere_intersection_simple(vertices, nvertices_B, position_A, radius_A);
    }
    else if (nvertices_A <= 0 && nvertices_B >= 0){
      // Collision check for any general ellipsoid
      double positions[6] = {x_A, y_A, z_A, x_B, y_B, z_B};
      collide = ellipsoid_intersect_check(semiaxes_in1.c6, input1.c11, positions, eulerparam);
    }
    else{
      // General collision solution for any arbitrary body.
      // No collision implementation for two polyhedra or for polyhedron and ellipsoid.
      collide = 1;
    }
    if (collide){
      return_value = -1;
    }
    else{
      // Secondary inside the primary
      if (sqrt(xc_BA*xc_BA + yc_BA*yc_BA + zc_BA*zc_BA) < radius_A){
        return_value = -2;
      }

    }
  }
  
  double Fx = 0;
  double Fy = 0;
  double Fz = 0;
  double MxA = 0;
  double MyA = 0;
  double MzA = 0;
  double MxB = 0;
  double MyB = 0;
  double MzB = 0;

  // Force and torque of body A
  int vertexids[7];
  vertexids[0] = nfaces_A;
  vertexids[1] = nfaces_B;
  vertexids[2] = 0;
  vertexids[3] = 1;
  vertexids[4] = nvertices_A;
  vertexids[5] = nvertices_B;
  vertexids[6] = nvertices;
  int *face_ids_other = (int*)malloc(3*nfaces_B * sizeof(int));
  for (i=0; i<nfaces_B; i++){
    face_ids_other[3*i] = face_ids[3*i+3*nfaces_A];
    face_ids_other[3*i+1] = face_ids[3*i+1+3*nfaces_A];
    face_ids_other[3*i+2] = face_ids[3*i+2+3*nfaces_A];
  }

  if (nfaces_A > 0){
    if (nfaces_B > 0 && nfaces_A > 100){
      // Run in parallel if two polyhedron and body A has more than 100 faces
      #pragma omp parallel for reduction(+:Fx,Fy,Fz,MxA,MyA,MzA)
      for (i=0; i<nfaces_A; i++){
        int vertex_combo[3];
        vertex_combo[0] = face_ids[3*i];
        vertex_combo[1] = face_ids[3*i+1];
        vertex_combo[2] = face_ids[3*i+2];
        Fx += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 1, 1, eulerparam, vertex_combo, prefac_A, vertexids, face_ids_other);
        Fy += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 1, 2, eulerparam, vertex_combo, prefac_A, vertexids, face_ids_other);
        Fz += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 1, 3, eulerparam, vertex_combo, prefac_A, vertexids, face_ids_other); 
        MxA += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 2, 1, eulerparam, vertex_combo, prefac_A, vertexids,face_ids_other);
        MyA += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 2, 2, eulerparam, vertex_combo, prefac_A, vertexids,face_ids_other);
        MzA += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 2, 3, eulerparam, vertex_combo, prefac_A, vertexids,face_ids_other);
      }
    }
    else{
      for (i=0; i<nfaces_A; i++){
        int vertex_combo[3];
        vertex_combo[0] = face_ids[3*i];
        vertex_combo[1] = face_ids[3*i+1];
        vertex_combo[2] = face_ids[3*i+2];
        Fx += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 1, 1, eulerparam, vertex_combo, prefac_A, vertexids, face_ids_other);
        Fy += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 1, 2, eulerparam, vertex_combo, prefac_A, vertexids, face_ids_other);
        Fz += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 1, 3, eulerparam, vertex_combo, prefac_A, vertexids, face_ids_other); 
        MxA += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 2, 1, eulerparam, vertex_combo, prefac_A, vertexids,face_ids_other);
        MyA += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 2, 2, eulerparam, vertex_combo, prefac_A, vertexids,face_ids_other);
        MzA += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 2, 3, eulerparam, vertex_combo, prefac_A, vertexids,face_ids_other); 
      }
    }
  }
  else{
    if (A_is_sphere){
      double force_result[3];
      double mass_array[4];
      mass_array[0] = mass_A;
      mass_array[1] = mass_B;
      mass_array[2] = rho_B;
      mass_array[3] = G_grav;
      Force_point_mass(input1.c11, semiaxes_in1.c6, eulerparam, vertices, vertexids, face_ids_other, force_result, mass_array);
      Fx = -force_result[0];
      Fy = -force_result[1];
      Fz = -force_result[2];
      MxA = 0;
      MyA = 0;
      MzA = 0;
    }
    else{
      Fx = Force_ellipsoid(input1.c11, itol, semiaxes_in1.c6, vertices, 1, 1, eulerparam, prefac_A, face_ids_other, vertexids);
      Fy = Force_ellipsoid(input1.c11, itol, semiaxes_in1.c6, vertices, 1, 2, eulerparam, prefac_A, face_ids_other, vertexids);
      Fz = Force_ellipsoid(input1.c11, itol, semiaxes_in1.c6, vertices, 1, 3, eulerparam, prefac_A, face_ids_other, vertexids);
      MxA = Force_ellipsoid(input1.c11, itol, semiaxes_in1.c6, vertices, 2, 1, eulerparam, prefac_A, face_ids_other, vertexids);
      MyA = Force_ellipsoid(input1.c11, itol, semiaxes_in1.c6, vertices, 2, 2, eulerparam, prefac_A, face_ids_other, vertexids);
      MzA = Force_ellipsoid(input1.c11, itol, semiaxes_in1.c6, vertices, 2, 3, eulerparam, prefac_A, face_ids_other, vertexids);
    }
  }  

  vertexids[0] = nfaces_B;
  vertexids[1] = nfaces_A;
  vertexids[2] = 1;
  vertexids[3] = 0;
  vertexids[4] = nvertices_B;
  vertexids[5] = nvertices_A;
  face_ids_other = (int*)realloc(face_ids_other, 3*nfaces_A * sizeof(int));
  for (i=0; i<nfaces_A; i++){
    face_ids_other[3*i] = face_ids[3*i];
    face_ids_other[3*i+1] = face_ids[3*i+1];
    face_ids_other[3*i+2] = face_ids[3*i+2];
  }

  if (nfaces_B > 0){
    if (nfaces_A > 0 && nfaces_B > 100){
      // Run in parallel if two polyhedron and one has many faces
      #pragma omp parallel for reduction(+:MxB,MyB,MzB)
      for (i=0; i<nfaces_B; i++){
        int vertex_combo[3];
        vertex_combo[0] = face_ids[3*i+3*nfaces_A] + nvertices_A;
        vertex_combo[1] = face_ids[3*i+1+3*nfaces_A] + nvertices_A;
        vertex_combo[2] = face_ids[3*i+2+3*nfaces_A] + nvertices_A;
        MxB += Force_polyhedron(input2.c11, itol, semiaxes_in2.c6, vertices, 2, 1, eulerparam, vertex_combo, prefac_B,vertexids,face_ids_other);
        MyB += Force_polyhedron(input2.c11, itol, semiaxes_in2.c6, vertices, 2, 2, eulerparam, vertex_combo, prefac_B,vertexids,face_ids_other);
        MzB += Force_polyhedron(input2.c11, itol, semiaxes_in2.c6, vertices, 2, 3, eulerparam, vertex_combo, prefac_B,vertexids,face_ids_other);
      }
    }
    else{
      for (i=0; i<nfaces_B; i++){
        int vertex_combo[3];
        vertex_combo[0] = face_ids[3*i+3*nfaces_A] + nvertices_A;
        vertex_combo[1] = face_ids[3*i+1+3*nfaces_A] + nvertices_A;
        vertex_combo[2] = face_ids[3*i+2+3*nfaces_A] + nvertices_A;
        MxB += Force_polyhedron(input2.c11, itol, semiaxes_in2.c6, vertices, 2, 1, eulerparam, vertex_combo, prefac_B,vertexids,face_ids_other);
        MyB += Force_polyhedron(input2.c11, itol, semiaxes_in2.c6, vertices, 2, 2, eulerparam, vertex_combo, prefac_B,vertexids,face_ids_other);
        MzB += Force_polyhedron(input2.c11, itol, semiaxes_in2.c6, vertices, 2, 3, eulerparam, vertex_combo, prefac_B,vertexids,face_ids_other);
      }
    }
  }
  else{
    if (B_is_sphere){
      MxB = 0;
      MyB = 0;
      MzB = 0;
    }
    else{
      MxB = Force_ellipsoid(input2.c11, itol, semiaxes_in2.c6, vertices, 2, 1, eulerparam, prefac_B, face_ids_other, vertexids);
      MyB = Force_ellipsoid(input2.c11, itol, semiaxes_in2.c6, vertices, 2, 2, eulerparam, prefac_B, face_ids_other, vertexids);
      MzB = Force_ellipsoid(input2.c11, itol, semiaxes_in2.c6, vertices, 2, 3, eulerparam, prefac_B, face_ids_other, vertexids);
    }
  }
  free(face_ids_other);
  double Fx_new = Rot_A[0][0]*Fx + Rot_A[0][1]*Fy + Rot_A[0][2]*Fz;
  double Fy_new = Rot_A[1][0]*Fx + Rot_A[1][1]*Fy + Rot_A[1][2]*Fz;
  double Fz_new = Rot_A[2][0]*Fx + Rot_A[2][1]*Fy + Rot_A[2][2]*Fz;
  
  // Angular accelerations
  // Local
  double I_tetra_A[3][3];
  double I_tetra_B[3][3];
  for (i=0; i<3; i++){
    for (j=0; j<3; j++){
      I_tetra_A[i][j] = moment_inertia[3*i + j];
      I_tetra_B[i][j] = moment_inertia[3*(i+3) + j];
    }
  }
  double domegadt_A[3];
  double domegadt_B[3];
  if (nfaces_A > 0){
    angular_acceleration_ode_general(I_tetra_A, p_A, q_A, r_A, MxA, MyA, MzA, domegadt_A);
  }
  else{
    if (A_is_sphere){
      domegadt_A[0] = 0;
      domegadt_A[1] = 0;
      domegadt_A[2] = 0;
    }
    else{
      angular_acceleration_ode_ellipsoid(I_tetra_A, p_A, q_A, r_A, MxA, MyA, MzA, domegadt_A);
      
    }
  }
  if (nfaces_B > 0){
    //moment_intertia_tetrahedron(rho_B, mass_B/rho_B, 0, 0, 0, vertices, I_tetra, nfaces_B, 1);
    angular_acceleration_ode_general(I_tetra_B, p_B, q_B, r_B, MxB, MyB, MzB, domegadt_B);
  }
  else{
    if (B_is_sphere){
      domegadt_B[0] = 0;
      domegadt_B[1] = 0;
      domegadt_B[2] = 0;
    }
    else{
      angular_acceleration_ode_ellipsoid(I_tetra_B, p_B, q_B, r_B, MxB, MyB, MzB, domegadt_B);
    }
  }

  // Return values
  // Forces to get velocities
  if (include_sun){
    // Add the gravitational force from the Sun, mass of Sun = 1
    double r_norm_A = x_A*x_A + y_A*y_A + z_A*z_A;
    double r_norm_B = x_B*x_B + y_B*y_B + z_B*z_B;
    
    double Sun_force_A = -G_grav*mass_A/(sqrt(r_norm_A)*r_norm_A);
    double Sun_force_B = -G_grav*mass_B/(sqrt(r_norm_B)*r_norm_B);
    double Fs_Ax = Sun_force_A*x_A;
    double Fs_Ay = Sun_force_A*y_A;
    double Fs_Az = Sun_force_A*z_A;

    double Fs_Bx = Sun_force_B*x_B;
    double Fs_By = Sun_force_B*y_B;
    double Fs_Bz = Sun_force_B*z_B;

    dfdt[0] = (Fx_new + Fs_Ax)/mass_A;
    dfdt[1] = (Fy_new + Fs_Ay)/mass_A;
    dfdt[2] = (Fz_new + Fs_Az)/mass_A;
    dfdt[3] = (-Fx_new + Fs_Bx)/mass_B;
    dfdt[4] = (-Fy_new + Fs_By)/mass_B;
    dfdt[5] = (-Fz_new + Fs_Bz)/mass_B;
  }
  else{
    dfdt[0] = Fx_new/mass_A;
    dfdt[1] = Fy_new/mass_A;
    dfdt[2] = Fz_new/mass_A;
    dfdt[3] = -Fx_new/mass_B;
    dfdt[4] = -Fy_new/mass_B;
    dfdt[5] = -Fz_new/mass_B;
  }
  // Velocities to get positions
  dfdt[6] = vx_A;
  dfdt[7] = vy_A;
  dfdt[8] = vz_A;
  dfdt[9] = vx_B;
  dfdt[10] = vy_B;
  dfdt[11] = vz_B;
  // Angular accelerations to get angular speeds
  dfdt[12] = domegadt_A[0];
  dfdt[13] = domegadt_A[1];
  dfdt[14] = domegadt_A[2];
  dfdt[15] = domegadt_B[0];
  dfdt[16] = domegadt_B[1];
  dfdt[17] = domegadt_B[2];
  // Angular speed to get rotation angles
  if (eulerparam){
    dfdt[18] = angular_speeds_derivative[0];
    dfdt[19] = angular_speeds_derivative[1];
    dfdt[20] = angular_speeds_derivative[2];
    dfdt[21] = angular_speeds_derivative[3];
    dfdt[22] = angular_speeds_derivative[4];
    dfdt[23] = angular_speeds_derivative[5];
    dfdt[24] = angular_speeds_derivative[6];
    dfdt[25] = angular_speeds_derivative[7];
  }
  else{
    dfdt[18] = angular_speeds_derivative[0];
    dfdt[19] = angular_speeds_derivative[1];
    dfdt[20] = angular_speeds_derivative[2];
    dfdt[21] = angular_speeds_derivative[3];
    dfdt[22] = angular_speeds_derivative[4];
    dfdt[23] = angular_speeds_derivative[5];
  }
  // Free vertices pointer
  for (i=0; i<nvertices; i++){
    free(vertices[i]);
  }
  free(vertices);
  return return_value;
}


int ode_solver_Nbody(double t, double* y, double* dfdt, double* params_input, double* semiaxes, double itol[3], double *vertices1D,
  int *face_ids, double *moment_inertia, int triggers[3]){
  /*
  N-body solver.
  */
  // For force computation
  array6size semiaxes_in1;
  array6size semiaxes_in2;
  array11size input1;
  array11size input2;
  // Other triggers
  int eulerparam = triggers[0];
  int include_sun = triggers[1];
  int N_bodies = triggers[2];
  // Gravitational constant
  double G_grav = params_input[N_bodies*5];
  // Rotation angles and rotation matrix of body A
  double Rot_A[3][3];
  // Counters to pick out parameters
  int N2 = 2*N_bodies;
  int N3 = 3*N_bodies;
  int N4 = 4*N_bodies;
  int N6 = 6*N_bodies;
  int N9 = 9*N_bodies;
  int N10 = 10*N_bodies;
  int return_value = 0;
  
  double *force_comps = calloc((N_bodies*3), sizeof(double));
  double *torque_comps = calloc((N_bodies*3), sizeof(double));

  double angular_speeds_derivative[4];
  double domegadt[3];
  double angle1, angle2, angle3, angle4;
  
  for (int i=0; i<N_bodies; i++){
    int iC = 3*i;
    int iD = 4*i;
    // Obtain parameters of body A
    double a_A = semiaxes[iC];
    double b_A = semiaxes[iC+1];
    double c_A = semiaxes[iC+2];  
    double mass_A = params_input[i];
    double rho_A = params_input[N_bodies+i];
    double radius_A = params_input[N2+i];
    int nfaces_A = params_input[N3+i];
    int nvertices_A = params_input[N4+i];
    
    int A_is_sphere = 0;
    if (fabs(a_A - b_A) < 1e-15 && fabs(a_A - c_A) < 1e-15){
      A_is_sphere = 1;
    }
    if (nfaces_A > 0){
      A_is_sphere = 0;
    }
    
    double vx_A = y[iC];
    double vy_A = y[iC+1];
    double vz_A = y[iC+2];
    
    double x_A = y[N3+iC];
    double y_A = y[N3+iC+1];
    double z_A = y[N3+iC+2];
    
    double omegax_A = y[N6+iC];
    double omegay_A = y[N6+iC+1];
    double omegaz_A = y[N6+iC+2];
    if (eulerparam){
      angle1 = y[N9+iD];
      angle2 = y[N9+iD+1];
      angle3 = y[N9+iD+2];
      angle4 = y[N9+iD+3];
      Rotation_matrix_Euler_param(angle1, angle2, angle3, angle4, Rot_A, 0);
      angular_speeds_derivative[0] = angular_speed_euler_param(omegax_A, omegay_A, omegaz_A, angle1, angle2, angle3, angle4, 1);
      angular_speeds_derivative[1] = angular_speed_euler_param(omegax_A, omegay_A, omegaz_A, angle1, angle2, angle3, angle4, 2);
      angular_speeds_derivative[2] = angular_speed_euler_param(omegax_A, omegay_A, omegaz_A, angle1, angle2, angle3, angle4, 3);
      angular_speeds_derivative[3] = angular_speed_euler_param(omegax_A, omegay_A, omegaz_A, angle1, angle2, angle3, angle4, 4);
      dfdt[N9+iD] = angular_speeds_derivative[0];
      dfdt[N9+iD+1] = angular_speeds_derivative[1];
      dfdt[N9+iD+2] = angular_speeds_derivative[2];
      dfdt[N9+iD+3] = angular_speeds_derivative[3];
    }
    else{
      angle1 = y[N9+iC];
      angle2 = y[N9+iC+1];
      angle3 = y[N9+iC+2];
      Rotation_matrix_components(angle1, angle2, angle3, Rot_A, 0);
      angular_speeds_derivative[0] = angular_speed_ode(omegax_A, omegay_A, omegaz_A, angle1, angle2, 1);
      angular_speeds_derivative[1] = angular_speed_ode(omegax_A, omegay_A, omegaz_A, angle1, angle2, 2);
      angular_speeds_derivative[2] = angular_speed_ode(omegax_A, omegay_A, omegaz_A, angle1, angle2, 3);
      dfdt[N9+iC] = angular_speeds_derivative[0];
      dfdt[N9+iC+1] = angular_speeds_derivative[1];
      dfdt[N9+iC+2] = angular_speeds_derivative[2];
    }
    for (int j=i+1; j<N_bodies; j++){
      int jC = j*3;
      int jD = j*4;
      double a_B = semiaxes[jC];
      double b_B = semiaxes[jC+1];
      double c_B = semiaxes[jC+2];
      double mass_B = params_input[j];
      double rho_B = params_input[N_bodies+j];
      double radius_B = params_input[N2+j];
      int nfaces_B = params_input[N3+j];
      int nvertices_B = params_input[N4+j];
      
      double vx_B = y[jC];
      double vy_B = y[jC+1];
      double vz_B = y[jC+2];
      
      double x_B = y[N3+jC];
      double y_B = y[N3+jC+1];
      double z_B = y[N3+jC+2];
      
      double omegax_B = y[N6+jC];
      double omegay_B = y[N6+jC+1];
      double omegaz_B = y[N6+jC+2];
      int nvertices = nvertices_A + nvertices_B;
      int nfaces = nfaces_A + nfaces_B;

      double xc_BA = x_B - x_A;
      double yc_BA = y_B - y_A;
      double zc_BA = z_B - z_A;
      double xc_AB = x_A - x_B;
      double yc_AB = y_A - y_B;
      double zc_AB = z_A - z_B;
      if (eulerparam){
        double e0_B = y[N9+jD];
        double e1_B = y[N9+jD+1];
        double e2_B = y[N9+jD+2];
        double e3_B = y[N9+jD+3];
        input1 = (array11size){xc_BA, yc_BA, zc_BA, angle1, angle2, angle3, angle4, e0_B, e1_B, e2_B, e3_B};
        input2 = (array11size){xc_AB, yc_AB, zc_AB, e0_B, e1_B, e2_B, e3_B, angle1, angle2, angle3, angle4};
      }
      else{
        double phi_B = y[N9+jC];
        double theta_B = y[N9+jC+1];
        double psi_B = y[N9+jC+2];
        input1 = (array11size){xc_BA, yc_BA, zc_BA, angle1, angle2, angle3, phi_B, theta_B, psi_B, 0, 0};
        input2 = (array11size){xc_AB, yc_AB, zc_AB, phi_B, theta_B, psi_B, angle1, angle2, angle3, 0, 0};
      }
      // Sort vertices into a 2D matrix
      double **vertices = (double **)malloc(nvertices * sizeof(double *));
      for (int k=0; k<nvertices; k++){
        vertices[k] = (double*)malloc(3 * sizeof(double));
        for (int m=0; m<3; m++){
          vertices[k][m] = vertices1D[3*k+m];
        }
      }
      semiaxes_in1 = (array6size){a_A, b_A, c_A, a_B, b_B, c_B};
      semiaxes_in2 = (array6size){a_B, b_B, c_B, a_A, b_A, c_A};
      // Determines pre-factor used for surface integration
      double prefac_A;
      double prefac_B;
      int B_is_sphere = 0;
      if (nvertices_B > 0){
        // If body B is a tetrahedron
        prefac_A = rho_A*rho_B*G_grav;
      }
      else{
        // If body B is a sphere
        if (fabs(a_B - b_B) < 1e-15 && fabs(a_B - c_B) < 1e-15){
          prefac_A = rho_A*mass_B*G_grav;
          B_is_sphere = 1;
        }
        // If body B is an ellipsoid
        else{
          prefac_A = rho_A*rho_B*G_grav;
        }
      }

      if (nvertices_A > 0){
        prefac_B = prefac_A;
      }
      else{
        // If body A is a sphere
        if (A_is_sphere){
          prefac_B = mass_A*rho_B*G_grav;
          Rotation_matrix_components(0, 0, 0, Rot_A, 0);
        }
        // If body A is an ellipsoid
        else{
          prefac_B = prefac_A;
        }
      }
      if (nfaces_B > 0){
        B_is_sphere = 0;
      }
      // Check collision between bodies
      int collide = 0;
      if (sqrt(xc_BA*xc_BA + yc_BA*yc_BA + zc_BA*zc_BA) <= (radius_A + radius_B)){
        if (nvertices_A > 0 && B_is_sphere){
          // Collision check for a primary polyhedron and secondary sphere
          double position_B[3] = {xc_BA, yc_BA, zc_BA};
          collide = polyhedron_sphere_intersection_simple(vertices, nvertices_A, position_B, radius_B);
        }
        else if (nvertices_B > 0 && A_is_sphere){
          // Collision check for a secondary polyhedron and primary sphere
          double position_A[3] = {xc_BA, yc_BA, zc_BA};
          collide = polyhedron_sphere_intersection_simple(vertices, nvertices_B, position_A, radius_A);
        }
        else if (nvertices_A <= 0 && nvertices_B >= 0){
          // Collision check for any general ellipsoid
          double positions[6] = {x_A, y_A, z_A, x_B, y_B, z_B};
          collide = ellipsoid_intersect_check(semiaxes_in1.c6, input1.c11, positions, eulerparam);
        }
        else{
          // General collision solution for any arbitrary body.
          // No collision implementation for two polyhedra or for polyhedron and ellipsoid.
          collide = 1;
        }
        if (collide){
          return_value = -1;
        }
        else{
          // Secondary inside the primary
          if (sqrt(xc_BA*xc_BA + yc_BA*yc_BA + zc_BA*zc_BA) < radius_A){
            return_value = -2;
          }

        }
      }
      double Fx = 0;
      double Fy = 0;
      double Fz = 0;
      double MxA = 0;
      double MyA = 0;
      double MzA = 0;
      double MxB = 0;
      double MyB = 0;
      double MzB = 0;
      
      // Sort vertices and face ids
      int vertexids[7];
      vertexids[0] = nfaces_A;
      vertexids[1] = nfaces_B;
      vertexids[2] = 0;
      vertexids[3] = 1;
      vertexids[4] = nvertices_A;
      vertexids[5] = nvertices_B;
      vertexids[6] = nvertices;
      int *face_ids_other = (int*)malloc(3*nfaces_B * sizeof(int));
      for (int k=0; k<nfaces_B; k++){
        face_ids_other[3*k] = face_ids[3*k+3*nfaces_A];
        face_ids_other[3*k+1] = face_ids[3*k+1+3*nfaces_A];
        face_ids_other[3*k+2] = face_ids[3*k+2+3*nfaces_A];
      }
      // Force and torque of body A
      if (nfaces_A > 0){
        if (nfaces_B > 0 && nfaces_A > 100){
          int n;
          // Run in parallel if two polyhedron and body A has more than 100 faces
          #pragma omp parallel for reduction(+:Fx,Fy,Fz,MxA,MyA,MzA)
          for (n=0; n<nfaces_A; n++){
            int vertex_combo[3];
            vertex_combo[0] = face_ids[3*n];
            vertex_combo[1] = face_ids[3*n+1];
            vertex_combo[2] = face_ids[3*n+2];
            Fx += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 1, 1,
                     eulerparam, vertex_combo, prefac_A, vertexids, face_ids_other);
            Fy += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 1, 2,
                     eulerparam, vertex_combo, prefac_A, vertexids, face_ids_other);
            Fz += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 1, 3,
                     eulerparam, vertex_combo, prefac_A, vertexids, face_ids_other); 
            MxA += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 2, 1,
                     eulerparam, vertex_combo, prefac_A, vertexids,face_ids_other);
            MyA += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 2, 2,
                     eulerparam, vertex_combo, prefac_A, vertexids,face_ids_other);
            MzA += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 2, 3,
                     eulerparam, vertex_combo, prefac_A, vertexids,face_ids_other);
          }
        }
        else{
          for (int k=0; k<nfaces_A; k++){
            int vertex_combo[3];
            vertex_combo[0] = face_ids[3*k];
            vertex_combo[1] = face_ids[3*k+1];
            vertex_combo[2] = face_ids[3*k+2];
            Fx += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 1, 1,
                    eulerparam, vertex_combo, prefac_A, vertexids, face_ids_other);
            Fy += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 1, 2,
                    eulerparam, vertex_combo, prefac_A, vertexids, face_ids_other);
            Fz += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 1, 3,
                    eulerparam, vertex_combo, prefac_A, vertexids, face_ids_other); 
            MxA += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 2, 1,
                    eulerparam, vertex_combo, prefac_A, vertexids,face_ids_other);
            MyA += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 2, 2,
                    eulerparam, vertex_combo, prefac_A, vertexids,face_ids_other);
            MzA += Force_polyhedron(input1.c11, itol, semiaxes_in1.c6, vertices, 2, 3,
                    eulerparam, vertex_combo, prefac_A, vertexids,face_ids_other); 
          }
        }
      }
      else{
        if (A_is_sphere){
          double force_result[3];
          double mass_array[4];
          mass_array[0] = mass_A;
          mass_array[1] = mass_B;
          mass_array[2] = rho_B;
          mass_array[3] = G_grav;
          Force_point_mass(input1.c11, semiaxes_in1.c6, eulerparam, vertices, vertexids, face_ids_other, force_result, mass_array);
          Fx = -force_result[0];
          Fy = -force_result[1];
          Fz = -force_result[2];
          MxA = 0;
          MyA = 0;
          MzA = 0;
        }
        else{
          Fx = Force_ellipsoid(input1.c11, itol, semiaxes_in1.c6, vertices, 1, 1, eulerparam, prefac_A, face_ids_other, vertexids);
          Fy = Force_ellipsoid(input1.c11, itol, semiaxes_in1.c6, vertices, 1, 2, eulerparam, prefac_A, face_ids_other, vertexids);
          Fz = Force_ellipsoid(input1.c11, itol, semiaxes_in1.c6, vertices, 1, 3, eulerparam, prefac_A, face_ids_other, vertexids);
          MxA = Force_ellipsoid(input1.c11, itol, semiaxes_in1.c6, vertices, 2, 1, eulerparam, prefac_A, face_ids_other, vertexids);
          MyA = Force_ellipsoid(input1.c11, itol, semiaxes_in1.c6, vertices, 2, 2, eulerparam, prefac_A, face_ids_other, vertexids);
          MzA = Force_ellipsoid(input1.c11, itol, semiaxes_in1.c6, vertices, 2, 3, eulerparam, prefac_A, face_ids_other, vertexids);
        }
      }  
      // Resort vertices and face ids for torque computation of body B
      vertexids[0] = nfaces_B;
      vertexids[1] = nfaces_A;
      vertexids[2] = 1;
      vertexids[3] = 0;
      vertexids[4] = nvertices_B;
      vertexids[5] = nvertices_A;
      face_ids_other = (int*)realloc(face_ids_other, 3*nfaces_A * sizeof(int));
      for (int k=0; k<nfaces_A; k++){
        face_ids_other[3*k] = face_ids[3*k];
        face_ids_other[3*k+1] = face_ids[3*k+1];
        face_ids_other[3*k+2] = face_ids[3*k+2];
      }

      if (nfaces_B > 0){
        if (nfaces_A > 0 && nfaces_B > 100){
          int n;
          // Run in parallel if two polyhedron and one has many faces
          #pragma omp parallel for reduction(+:MxB,MyB,MzB)
          for (n=0; n<nfaces_B; n++){
            int vertex_combo[3];
            vertex_combo[0] = face_ids[3*n+3*nfaces_A] + nvertices_A;
            vertex_combo[1] = face_ids[3*n+1+3*nfaces_A] + nvertices_A;
            vertex_combo[2] = face_ids[3*n+2+3*nfaces_A] + nvertices_A;
            MxB += Force_polyhedron(input2.c11, itol, semiaxes_in2.c6, vertices, 2, 1,
                   eulerparam, vertex_combo, prefac_B,vertexids,face_ids_other);
            MyB += Force_polyhedron(input2.c11, itol, semiaxes_in2.c6, vertices, 2, 2,
                   eulerparam, vertex_combo, prefac_B,vertexids,face_ids_other);
            MzB += Force_polyhedron(input2.c11, itol, semiaxes_in2.c6, vertices, 2, 3,
                   eulerparam, vertex_combo, prefac_B,vertexids,face_ids_other);
          }
        }
        else{
          for (int k=0; k<nfaces_B; k++){
            int vertex_combo[3];
            vertex_combo[0] = face_ids[3*k+3*nfaces_A] + nvertices_A;
            vertex_combo[1] = face_ids[3*k+1+3*nfaces_A] + nvertices_A;
            vertex_combo[2] = face_ids[3*k+2+3*nfaces_A] + nvertices_A;
            MxB += Force_polyhedron(input2.c11, itol, semiaxes_in2.c6, vertices, 2, 1,
                   eulerparam, vertex_combo, prefac_B,vertexids,face_ids_other);
            MyB += Force_polyhedron(input2.c11, itol, semiaxes_in2.c6, vertices, 2, 2,
                   eulerparam, vertex_combo, prefac_B,vertexids,face_ids_other);
            MzB += Force_polyhedron(input2.c11, itol, semiaxes_in2.c6, vertices, 2, 3,
                   eulerparam, vertex_combo, prefac_B,vertexids,face_ids_other);
          }
        }
      }
      else{
        if (B_is_sphere){
          MxB = 0;
          MyB = 0;
          MzB = 0;
        }
        else{
          MxB = Force_ellipsoid(input2.c11, itol, semiaxes_in2.c6, vertices, 2, 1, eulerparam, prefac_B, face_ids_other, vertexids);
          MyB = Force_ellipsoid(input2.c11, itol, semiaxes_in2.c6, vertices, 2, 2, eulerparam, prefac_B, face_ids_other, vertexids);
          MzB = Force_ellipsoid(input2.c11, itol, semiaxes_in2.c6, vertices, 2, 3, eulerparam, prefac_B, face_ids_other, vertexids);
        }
      }
      free(face_ids_other);
      // Moves force to global frame
      double Fx_new = Rot_A[0][0]*Fx + Rot_A[0][1]*Fy + Rot_A[0][2]*Fz;
      double Fy_new = Rot_A[1][0]*Fx + Rot_A[1][1]*Fy + Rot_A[1][2]*Fz;
      double Fz_new = Rot_A[2][0]*Fx + Rot_A[2][1]*Fy + Rot_A[2][2]*Fz;
      force_comps[iC] += Fx_new;
      force_comps[iC+1] += Fy_new;
      force_comps[iC+2] += Fz_new;
      force_comps[jC] -= Fx_new;
      force_comps[jC+1] -= Fy_new;
      force_comps[jC+2] -= Fz_new;

      torque_comps[iC] += MxA;
      torque_comps[iC+1] += MyA;
      torque_comps[iC+2] += MzA;
      torque_comps[jC] += MxB;
      torque_comps[jC+1] += MyB;
      torque_comps[jC+2] += MzB;
      
      // Free vertices pointer
      for (int k=0; k<nvertices; k++){
        free(vertices[k]);
      }
      free(vertices);
    }
    // Compute angular velocity ODE
    double I_tensor[3][3];
    for (int k=0; k<3; k++){
      for (int m=0; m<3; m++){
        I_tensor[k][m] = moment_inertia[3*(k+iC) + m];
      }
    }

    if (A_is_sphere){
      domegadt[0] = 0;
      domegadt[1] = 0;
      domegadt[2] = 0;
    }
    else{
      if (nfaces_A > 0){
        angular_acceleration_ode_general(I_tensor, omegax_A, omegay_A, omegaz_A, 
                torque_comps[iC], torque_comps[iC+1], torque_comps[iC+2], domegadt);
      }
      else{
        angular_acceleration_ode_ellipsoid(I_tensor, omegax_A, omegay_A, omegaz_A,
                 torque_comps[iC], torque_comps[iC+1], torque_comps[iC+2], domegadt);
      }
    }
    if (include_sun){
      // Add the gravitational force from the Sun, mass of Sun = 1
      double r_norm_A = x_A*x_A + y_A*y_A + z_A*z_A;
      double Sun_force = -G_grav*mass_A/(sqrt(r_norm_A)*r_norm_A);
      double Fs_x = Sun_force*x_A;
      double Fs_y = Sun_force*y_A;
      double Fs_z = Sun_force*z_A;
      force_comps[iC] += Fs_x;
      force_comps[iC+1] += Fs_y;
      force_comps[iC+2] += Fs_z;
      }
    // Return values
    // Forces to get velocities
    dfdt[iC] = force_comps[iC]/mass_A;
    dfdt[iC+1] = force_comps[iC+1]/mass_A;
    dfdt[iC+2] = force_comps[iC+2]/mass_A;
    // Velocities to get positions
    dfdt[N3+iC] = vx_A;
    dfdt[N3+iC+1] = vy_A;
    dfdt[N3+iC+2] = vz_A;
    // Angular accelerations to get angular speeds
    dfdt[N6+iC] = domegadt[0];
    dfdt[N6+iC+1] = domegadt[1];
    dfdt[N6+iC+2] = domegadt[2];
  }
  return return_value;
}

int force_comp_omp(double input[11], double itol[6], double saxes[6], double **vertices, int F_compute, int eulerparam,
  double prefac, int vertexids[7], int *fids, int* face_ids, int nfaces, int face_step, int vertex_step,
  double force_res[3], double torque_res[3]){
  /* Computes force/torque with OpenMP parallelization */
  double Fx = 0;
  double Fy = 0;
  double Fz = 0;
  double Mx = 0;
  double My = 0;
  double Mz = 0;
  int k;
  if (F_compute){
    #pragma omp parallel for reduction(+:Fx,Fy,Fz,Mx,My,Mz)
    for (k=0; k<nfaces; k++){
      int vertex_combo[3];
      vertex_combo[0] = face_ids[3*k + 3*face_step] + vertex_step;
      vertex_combo[1] = face_ids[3*k + 1 + 3*face_step] + vertex_step;
      vertex_combo[2] = face_ids[3*k + 2 + 3*face_step] + vertex_step;
      Fx += Force_polyhedron(input, itol, saxes, vertices, 1, 1, eulerparam, vertex_combo, prefac, vertexids, fids);
      Fy += Force_polyhedron(input, itol, saxes, vertices, 1, 2, eulerparam, vertex_combo, prefac, vertexids, fids);
      Fz += Force_polyhedron(input, itol, saxes, vertices, 1, 3, eulerparam, vertex_combo, prefac, vertexids, fids); 
      Mx += Force_polyhedron(input, itol, saxes, vertices, 2, 1, eulerparam, vertex_combo, prefac, vertexids, fids);
      My += Force_polyhedron(input, itol, saxes, vertices, 2, 2, eulerparam, vertex_combo, prefac, vertexids, fids);
      Mz += Force_polyhedron(input, itol, saxes, vertices, 2, 3, eulerparam, vertex_combo, prefac, vertexids, fids); 
    }
  }
  else{
    #pragma omp parallel for reduction(+:Mx,My,Mz)
    for (k=0; k<nfaces; k++){
      int vertex_combo[3];
      vertex_combo[0] = face_ids[3*k + 3*face_step] + vertex_step;
      vertex_combo[1] = face_ids[3*k + 1 + 3*face_step] + vertex_step;
      vertex_combo[2] = face_ids[3*k + 2 + 3*face_step] + vertex_step;
      Mx += Force_polyhedron(input, itol, saxes, vertices, 2, 1, eulerparam, vertex_combo, prefac, vertexids, fids);
      My += Force_polyhedron(input, itol, saxes, vertices, 2, 2, eulerparam, vertex_combo, prefac, vertexids, fids);
      Mz += Force_polyhedron(input, itol, saxes, vertices, 2, 3, eulerparam, vertex_combo, prefac, vertexids, fids);
    }
  }
  force_res[0] = Fx;
  force_res[1] = Fy;
  force_res[2] = Fz;
  torque_res[0] = Mx;
  torque_res[1] = My;
  torque_res[2] = Mz;
  return 0;
}

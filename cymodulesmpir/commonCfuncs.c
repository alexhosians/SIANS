/*
C file that contains some commonly used functions.
*/
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_eigen.h>

int ay[1] = {0};  // Used as a loop counter

// Strcutures containing constant parameters used for integration
struct ellipsoid_params{
  // Parameters for integration of the ellipsoid potential
  double asq, bsq, csq;
  double xsq, ysq, zsq;
};

struct gravfield_ellipsoid_params{
  // Parameters for integration of the ellipsoid gravitational field
  double asq, bsq, csq;
  double semiax_sq;
};

double absolute_value_vector_two(double r1[3], double r2[3]){
  double x = r1[0] - r2[0];
  double y = r1[1] - r2[1];
  double z = r1[2] - r2[2];
  double abs_value = sqrt(x*x + y*y + z*z);
  return abs_value;
}

double absolute_value_vector(double r[3]){
  double abs_value = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
  return abs_value;
}

double dot_product(double r1[3], double r2[3]){
  double dot_prod = r1[0]*r2[0] + r1[1]*r2[1] + r1[2]*r2[2];
  return dot_prod;
}

int cross_product(double r1[3], double r2[3], double cr[3]){
  cr[0] = r1[1]*r2[2] - r1[2]*r2[1];
  cr[1] = r1[2]*r2[0] - r1[0]*r2[2];
  cr[2] = r1[0]*r2[1] - r1[1]*r2[0];
  return 0;
}


void Rotation_matrix_components(double phi, double theta, double psi, double R_matrix[3][3], int transpose){
  // A function that calculates the rotation matrix and its transpose.
  double sinphi = sin(phi);
  double cosphi = cos(phi);
  double sintheta = sin(theta);
  double costheta = cos(theta);
  double sinpsi = sin(psi);
  double cospsi = cos(psi);
  if (transpose == 0){
  	double sinpsint = sinphi*sintheta;
  	double cospsint = cosphi*sintheta;
    R_matrix[0][0] = cospsi*costheta;
    R_matrix[1][0] = sinpsi*costheta;
    R_matrix[2][0] = -sintheta;
    R_matrix[0][1] = -sinpsi*cosphi + cospsi*sinpsint;
    R_matrix[1][1] = cospsi*cosphi + sinpsi*sinpsint;
    R_matrix[2][1] = costheta*sinphi;
    R_matrix[0][2] = sinpsi*sinphi + cospsi*cospsint;
    R_matrix[1][2] = -cospsi*sinphi + sinpsi*cospsint;
    R_matrix[2][2] = costheta*cosphi;
  }
  else{
  	double cospsint = cospsi*sintheta;
  	double sinpsint = sinpsi*sintheta;
    R_matrix[0][0] = cospsi*costheta;
    R_matrix[1][0] = -sinpsi*cosphi + cospsint*sinphi;
    R_matrix[2][0] = sinpsi*sinphi + cospsint*cosphi;
    R_matrix[0][1] = sinpsi*costheta;
    R_matrix[1][1] = cospsi*cosphi + sinpsint*sinphi;
    R_matrix[2][1] = -cospsi*sinphi + sinpsint*cosphi;
    R_matrix[0][2] = -sintheta;
    R_matrix[1][2] = costheta*sinphi;
    R_matrix[2][2] = costheta*cosphi;
  }
}


int Rotation_matrix_Euler_param(double e0, double e1, double e2, double e3, double R_matrix[3][3], int transpose){
  if (transpose == 0){
    R_matrix[0][0] = 2*(e0*e0 + e1*e1 - 0.5);
    R_matrix[0][1] = 2*(e1*e2 - e0*e3);
    R_matrix[0][2] = 2*(e1*e3 + e0*e2);
    
    R_matrix[1][0] = 2*(e1*e2 + e0*e3);
    R_matrix[1][1] = 2*(e0*e0 + e2*e2 - 0.5);
    R_matrix[1][2] = 2*(e2*e3 - e0*e1);

    R_matrix[2][0] = 2*(e1*e3 - e0*e2);
    R_matrix[2][1] = 2*(e2*e3 + e0*e1);
    R_matrix[2][2] = 2*(e0*e0 + e3*e3 - 0.5);
  }
  else{
    R_matrix[0][0] = 2*(e0*e0 + e1*e1 - 0.5);
    R_matrix[0][1] = 2*(e1*e2 + e0*e3);
    R_matrix[0][2] = 2*(e1*e3 - e0*e2);

    R_matrix[1][0] = 2*(e1*e2 - e0*e3);
    R_matrix[1][1] = 2*(e0*e0 + e2*e2 - 0.5);
    R_matrix[1][2] = 2*(e2*e3 + e0*e1);

    R_matrix[2][0] = 2*(e1*e3 + e0*e2);
    R_matrix[2][1] = 2*(e2*e3 - e0*e1);
    R_matrix[2][2] = 2*(e0*e0 + e3*e3 - 0.5);
  }
  return 0;
}

int Grav_rot_matrix(double phi_1, double theta_1, double psi_1, double phi_2, double theta_2, double psi_2, double Rot_grav[3][3]){
	/* 
	Computes the rotation matrix that rotates the gravitational field
	Used for the gravitational potential energy.
	*/
	// Two rotation matrices
	double R_1[3][3];
	double R_2_T[3][3];

	// Some precomputed values
	double cospsi1 = cos(psi_1);
	double sinpsi1 = sin(psi_1);
	double cosphi1 = cos(phi_1);
	double sinphi1 = sin(phi_1);
	double costheta1 = cos(theta_1);
	double sintheta1 = sin(theta_1);
	
	double cospsi2 = cos(psi_2);
	double sinpsi2 = sin(psi_2);
	double cosphi2 = cos(phi_2);
	double sinphi2 = sin(phi_2);
	double costheta2 = cos(theta_2);
	double sintheta2 = sin(theta_2);

	// Defining the matrices explicitly
	R_1[0][0] = cospsi1*costheta1;
	R_1[0][1] = -sinpsi1*cosphi1 + cospsi1*sintheta1*sinphi1;
	R_1[0][2] = sinpsi1*sinphi1 + cospsi1*sintheta1*cosphi1;
	R_1[1][0] = sinpsi1*costheta1;
	R_1[1][1] = cospsi1*cosphi1 + sinpsi1*sintheta1*sinphi1;
	R_1[1][2] = -cospsi1*sinphi1 + sinpsi1*sintheta1*cosphi1;
	R_1[2][0] = -sintheta1;
	R_1[2][1] = costheta1*sinphi1;
	R_1[2][2] = costheta1*cosphi1;

	R_2_T[0][0] = cospsi2*costheta2;
	R_2_T[0][1] = sinpsi2*costheta2;
	R_2_T[0][2] = -sintheta2;
	R_2_T[1][0] = -sinpsi2*cosphi2 + cospsi2*sintheta2*sinphi2;
	R_2_T[1][1] = cospsi2*cosphi2 + sinpsi2*sintheta2*sinphi2;
	R_2_T[1][2] = costheta2*sinphi2;
	R_2_T[2][0] = sinpsi2*sinphi2 + cospsi2*sintheta2*cosphi2;
	R_2_T[2][1] = -cospsi2*sinphi2 + sinpsi2*sintheta2*cosphi2;
	R_2_T[2][2] = costheta2*cosphi2;

	// Do the matrix rotation
	double sumval;
	int i,j,k;
	for (i = 0; i < 3; i = i + 1){
		for (j = 0; j < 3; j = j + 1){
			sumval = 0.0;
			for (k = 0; k < 3; k = k + 1){
				sumval = sumval + R_2_T[i][k]*R_1[k][j];
			}
			Rot_grav[i][j] = sumval;
		}
	}
  return 0;
}

int Grav_rot_matrix_euler(double e0_A, double e1_A, double e2_A, double e3_A, 
  double e0_B, double e1_B, double e2_B, double e3_B, double Rot_grav[3][3]){
  /* 
  Computes the rotation matrix that rotates the gravitational field
  Used for the gravitational potential energy.
  */
  // Two rotation matrices
  double R_A[3][3];
  double R_B_T[3][3];
  Rotation_matrix_Euler_param(e0_A, e1_A, e2_A, e3_A, R_A, 0);
  Rotation_matrix_Euler_param(e0_B, e1_B, e2_B, e3_B, R_B_T, 1);
  
  // Multiply the matrices
  double sumval;
  int i,j,k;
  for (i = 0; i < 3; i = i + 1){
    for (j = 0; j < 3; j = j + 1){
      sumval = 0.0;
      for (k = 0; k < 3; k = k + 1){
        sumval = sumval + R_B_T[i][k]*R_A[k][j];
      }
      Rot_grav[i][j] = sumval;
    }
  }
  return 0;
}

double moment_intertia_spheroid(double a, double c, double rho, int i){
	/* Values o fthe moment of intertia for spheroids */
	//double pi_ = atan(1)*4;
	double M = (4.0/15.0)*M_PI*rho*a*a*c;
	double mom_inert;
	if (i == 1){
		mom_inert = M*(a*a+c*c);
	}
	else if (i == 2){
		mom_inert = M*(a*a+c*c);
	}
	else if (i == 3){
		mom_inert = 2.0*M*a*a;
	}
	else{
		printf("Component of Moment of intertia function not set properly!");
		exit(1);
	}
	return mom_inert;
}

double moment_intertia_ellipsoid(double a, double b, double c, double rho, int i){
	/* Values o fthe moment of intertia for spheroids */
	//double pi_ = atan(1)*4;
	double M = (4.0/15.0)*M_PI*rho*a*b*c;
	double mom_inert;
	if (i == 1){
		mom_inert = M*(b*b+c*c);
	}
	else if (i == 2){
		mom_inert = M*(c*c+a*a);
	}
	else if (i == 3){
		mom_inert = M*(a*a+b*b);
	}
	else{
		printf("Component of Moment of intertia function not set properly!");
		exit(1);
	}
	return mom_inert;
}


double angular_acceleration_ode(double a, double b, double c, double rho, double p1, double p2, double torq, int comp){
	/*
	The right hand side of the differential equation of p,q,r.
	*/

	double I_11 = moment_intertia_ellipsoid(a,b,c,rho,1);
	double I_22 = moment_intertia_ellipsoid(a,b,c,rho,2);
	double I_33 = moment_intertia_ellipsoid(a,b,c,rho,3);
	
	double rhs_ode;
	if (comp == 1){
		rhs_ode = (torq - (I_33 - I_22)*p1*p2)/I_11;
	}
	else if (comp == 2){
		rhs_ode = (torq - (I_11 - I_33)*p1*p2)/I_22;
	}
	else if (comp == 3){
		rhs_ode = (torq - (I_22 - I_11)*p1*p2)/I_33;
	}
	else{
		printf("Component of angular_acceleration_ode() not set properly! \n");
		exit(1);
	}
	return rhs_ode;
}

int Step_solution_RK_pointer(double *y_new, double *y, double **K, double *B, int n_stages, int numvars, double dt){
  /*
  Function that steps the solution based on the K-parameters from the Runge-Kutta solver.
  Use this for a x20 speed-up, but it is currently bugged.
  */
  double new_steps = 0.0;
  int i, j;
  for (i = 0; i<numvars; i++){
    new_steps = 0.0;
    for (j = 0; j<n_stages; j++){
      new_steps += K[j][i]*B[j];
    }
    y_new[i] = y[i] + new_steps*dt;
  }
  return 0;
}

int Rotation_matrix_components_new(double phi, double theta, double psi, double R_matrix[3][3], int transpose, int trigger){
  // A function that calculates the rotation matrix and its transpose.
  double sinphi = sin(phi);
  double cosphi = cos(phi);
  double sintheta = sin(theta);
  double costheta = cos(theta);
  double sinpsi = sin(psi);
  double cospsi = cos(psi);
  double sinpsint;
  double cospsint;
  if (trigger == 0){
    // For Z-Y-X rotation -- original
    if (transpose == 0){
      sinpsint = sinphi*sintheta;
      cospsint = cosphi*sintheta;
      R_matrix[0][0] = cospsi*costheta;
      R_matrix[1][0] = sinpsi*costheta;
      R_matrix[2][0] = -sintheta;
      R_matrix[0][1] = -sinpsi*cosphi + cospsi*sinpsint;
      R_matrix[1][1] = cospsi*cosphi + sinpsi*sinpsint;
      R_matrix[2][1] = costheta*sinphi;
      R_matrix[0][2] = sinpsi*sinphi + cospsi*cospsint;
      R_matrix[1][2] = -cospsi*sinphi + sinpsi*cospsint;
      R_matrix[2][2] = costheta*cosphi;
    }
    else{
      cospsint = cospsi*sintheta;
      sinpsint = sinpsi*sintheta;
      R_matrix[0][0] = cospsi*costheta;
      R_matrix[1][0] = -sinpsi*cosphi + cospsint*sinphi;
      R_matrix[2][0] = sinpsi*sinphi + cospsint*cosphi;
      R_matrix[0][1] = sinpsi*costheta;
      R_matrix[1][1] = cospsi*cosphi + sinpsint*sinphi;
      R_matrix[2][1] = -cospsi*sinphi + sinpsint*cosphi;
      R_matrix[0][2] = -sintheta;
      R_matrix[1][2] = costheta*sinphi;
      R_matrix[2][2] = costheta*cosphi;
    }
  }
  else if (trigger == 1){
    // For X-Z-Y rotation
    if (transpose == 0){
      R_matrix[0][0] = cospsi*costheta;
      R_matrix[1][0] = cosphi*sinpsi*costheta + sinphi*sintheta;
      R_matrix[2][0] = sinphi*sinpsi*costheta - cosphi*sintheta;
      R_matrix[0][1] = -sinpsi;
      R_matrix[1][1] = cosphi*cospsi;
      R_matrix[2][1] = sinphi*cospsi;
      R_matrix[0][2] = cospsi*sintheta;
      R_matrix[1][2] = cosphi*sinpsi*sintheta - sinphi*costheta;
      R_matrix[2][2] = sinphi*sinpsi*sintheta + cosphi*costheta;
    }
    else{
      R_matrix[0][0] = cospsi*costheta;
      R_matrix[0][1] = cosphi*sinpsi*costheta + sinphi*sintheta;
      R_matrix[0][2] = sinphi*sinpsi*costheta - cosphi*sintheta;
      R_matrix[1][0] = -sinpsi;
      R_matrix[1][1] = cosphi*cospsi;
      R_matrix[1][2] = sinphi*cospsi;
      R_matrix[2][0] = cospsi*sintheta;
      R_matrix[2][1] = cosphi*sinpsi*sintheta - sinphi*costheta;
      R_matrix[2][2] = sinphi*sinpsi*sintheta + cosphi*costheta;
    }
  }
  else if (trigger == 2){
    // For Y-X-Z rotation
    if (transpose == 0){
      R_matrix[0][0] = costheta*cospsi + sintheta*sinphi*sinpsi;
      R_matrix[1][0] = cosphi*sinpsi;
      R_matrix[2][0] = -sintheta*cospsi + costheta*sinphi*sinpsi;
      R_matrix[0][1] = -costheta*sinpsi + sintheta*sinphi*cospsi;
      R_matrix[1][1] = cosphi*cospsi;
      R_matrix[2][1] = sintheta*sinpsi + costheta*sinphi*cospsi;
      R_matrix[0][2] = sintheta*cosphi;
      R_matrix[1][2] = -sinphi;
      R_matrix[2][2] = costheta*cosphi;
    }
    else{
      R_matrix[0][0] = costheta*cospsi + sintheta*sinphi*sinpsi;
      R_matrix[0][1] = cosphi*sinpsi;
      R_matrix[0][2] = -sintheta*cospsi + costheta*sinphi*sinpsi;
      R_matrix[1][0] = -costheta*sinpsi + sintheta*sinphi*cospsi;
      R_matrix[1][1] = cosphi*cospsi;
      R_matrix[1][2] = sintheta*sinpsi + costheta*sinphi*cospsi;
      R_matrix[2][0] = sintheta*cosphi;
      R_matrix[2][1] = -sinphi;
      R_matrix[2][2] = costheta*cosphi;
    }
  }
  else{
    printf("Wrong trigger input. \n");
    exit(1);
  }
  return 0;
}


double angular_speed_ode_new(double p, double q, double r, double phi, double theta, double psi, int comp, int trigger){
  /* ODE for the angular speeds of phi, theta and psi */
  double rhs_ode;
  if (trigger == 0){
    // For Z-Y-X rotation -- original
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
  }
  else if (trigger == 1){
    // For X-Z-Y rotation
    if (comp == 1){
      rhs_ode = (p*cos(theta) + r*sin(theta))/cos(psi);
    }
    else if (comp == 2){
      rhs_ode = q + (p*cos(theta) + r*sin(theta))*tan(psi);
    }
    else if (comp == 3){
      rhs_ode = r*cos(theta) - p*sin(theta);
    }
    else{
      printf("Component of angular_speed_ode() not set properly! \n");
      exit(1);
    }
  }
  else if (trigger == 2){
    // For Y-X-Z rotation
    if (comp == 1){
      rhs_ode = p*cos(psi) - q*sin(psi);
    }
    else if (comp == 2){
      rhs_ode = (p*sin(psi) + q*cos(psi))/cos(phi);
    }
    else if (comp == 3){
      rhs_ode = r + (p*sin(psi) + q*cos(psi))*tan(phi);
    }
    else{
      printf("Component of angular_speed_ode() not set properly! \n");
      exit(1);
    }
  }
  else{
    printf("Trigger not properly set! \n");
    exit(1);
  }
  return rhs_ode;
}


double angular_speed_ode_new_v2(double p, double q, double r, double phi, double theta, double psi, double res[3], int trigger){
  /* ODE for the angular speeds of phi, theta and psi */
  double phidot, thetadot, psidot;
  phidot = p + (q*sin(phi) + r*cos(phi))*tan(theta);
  thetadot = q*cos(phi) - r*sin(phi);
  psidot = (q*sin(phi) + r*cos(phi))/cos(theta);
  
  double rhs_ode;
  if (trigger == 0){
    res[0] = phidot;
    res[1] = thetadot;
    res[2] = psidot;
  }
  /*
  else if (trigger == 1){
    // For X-Z-Y rotation
    if (comp == 1){
      rhs_ode = (p*cos(theta) + r*sin(theta))/cos(psi);
    }
    else if (comp == 2){
      rhs_ode = q + (p*cos(theta) + r*sin(theta))*tan(psi);
    }
    else if (comp == 3){
      rhs_ode = r*cos(theta) - p*sin(theta);
    }
    else{
      printf("Component of angular_speed_ode() not set properly! \n");
      exit(1);
    }
  }
  else if (trigger == 2){
    // For Y-X-Z rotation
    if (comp == 1){
      rhs_ode = p*cos(psi) - q*sin(psi);
    }
    else if (comp == 2){
      rhs_ode = (p*sin(psi) + q*cos(psi))/cos(phi);
    }
    else if (comp == 3){
      rhs_ode = r + (p*sin(psi) + q*cos(psi))*tan(phi);
    }
    else{
      printf("Component of angular_speed_ode() not set properly! \n");
      exit(1);
    }
  }
  else{
    printf("Trigger not properly set! \n");
    exit(1);
  }
  */
  return rhs_ode;
}

double Matrix_determinant_3x3(double A[3][3]){
  /* Computes determinant of a 3x3 matrix*/
  double det = A[0][0]*A[1][1]*A[2][2] + A[0][1]*A[1][2]*A[2][0] + A[0][2]*A[1][0]*A[2][1] 
             - A[0][2]*A[1][1]*A[2][0] - A[0][1]*A[1][0]*A[2][2] - A[0][0]*A[1][2]*A[2][1];
  return det;
}

int moment_intertia_tetrahedron(double rho, double volume, double xc, double yc, double zc, double **vertices,
 double I[3][3], int nfaces, int body){
  /* 
  Moment of inertia for a tetrahedron with given vertices
  Expressions from F. Tonon (2004).
  xc, yc, zc are centroid coordinates of the tetrahedron.
  */
  // Vertex points
  int index = nfaces*body;
  double x_1 = vertices[index][0] - xc;
  double y_1 = vertices[index][1] - yc;
  double z_1 = vertices[index][2] - zc;
  double x_2 = vertices[index+1][0] - xc;
  double y_2 = vertices[index+1][1] - yc;
  double z_2 = vertices[index+1][2] - zc;
  double x_3 = vertices[index+2][0] - xc;
  double y_3 = vertices[index+2][1] - yc;
  double z_3 = vertices[index+2][2] - zc;
  double x_4 = vertices[index+3][0] - xc;
  double y_4 = vertices[index+3][1] - yc;
  double z_4 = vertices[index+3][2] - zc;
  
  double prefac = 6*rho*volume;
  double a = prefac*(y_1*y_1 + y_1*y_2 + y_2*y_2 + y_1*y_3 + y_2*y_3 + y_3*y_3 + y_1*y_4 
                  + y_2*y_4 + y_3*y_4 + y_4*y_4 + z_1*z_1 + z_1*z_2 + z_2*z_2 + z_1*z_3 
                  + z_2*z_3 + z_3*z_3 + z_1*z_4 + z_2*z_4 + z_3*z_4 + z_4*z_4)/60;
  double b = prefac*(x_1*x_1 + x_1*x_2 + x_2*x_2 + x_1*x_3 + x_2*x_3 + x_3*x_3 + x_1*x_4 
                  + x_2*x_4 + x_3*x_4 + x_4*x_4 + z_1*z_1 + z_1*z_2 + z_2*z_2 + z_1*z_3 
                  + z_2*z_3 + z_3*z_3 + z_1*z_4 + z_2*z_4 + z_3*z_4 + z_4*z_4)/60;
  double c = prefac*(x_1*x_1 + x_1*x_2 + x_2*x_2 + x_1*x_3 + x_2*x_3 + x_3*x_3 + x_1*x_4 
                  + x_2*x_4 + x_3*x_4 + x_4*x_4 + y_1*y_1 + y_1*y_2 + y_2*y_2 + y_1*y_3 
                  + y_2*y_3 + y_3*y_3 + y_1*y_4 + y_2*y_4 + y_3*y_4 + y_4*y_4)/60;
  double a_prime = prefac*(2*y_1*z_1 + y_2*z_1 + y_3*z_1 + y_4*z_1 + y_1*z_2 + 2*y_2*z_2
                  + y_3*z_2 + y_4*z_2 + y_1*z_3 + y_2*z_3 + 2*y_3*z_3 + y_4*z_3
                  + y_1*z_4 + y_2*z_4 + y_3*z_4 + 2*y_4*z_4)/120;
  double b_prime = prefac*(2*x_1*z_1 + x_2*z_1 + x_3*z_1 + x_4*z_1 + x_1*z_2 + 2*x_2*z_2
                  + x_3*z_2 + x_4*z_2 + x_1*z_3 + x_2*z_3 + 2*x_3*z_3 + x_4*z_3
                  + x_1*z_4 + x_2*z_4 + x_3*z_4 + 2*x_4*z_4)/120;
  double c_prime = prefac*(2*x_1*y_1 + x_2*y_1 + x_3*y_1 + x_4*y_1 + x_1*y_2 + 2*x_2*y_2
                  + x_3*y_2 + x_4*y_2 + x_1*y_3 + x_2*y_3 + 2*x_3*y_3 + x_4*y_3
                  + x_1*y_4 + x_2*y_4 + x_3*y_4 + 2*x_4*y_4)/120;
  I[0][0] = a;
  I[0][1] = -b_prime;
  I[0][2] = -c_prime;
  I[1][0] = -b_prime;
  I[1][1] = b;
  I[1][2] = -a_prime;
  I[2][0] = -c_prime;
  I[2][1] = -a_prime;
  I[2][2] = c;
  return 0;
}

int moment_inertia_polyhedron(double *vertices, int* index_combo, int N_faces, int N_vertices, double mass, int id_fac, double I[9]){
  /* Moment of inertia for any general polyhedron */
  double *volume_elements = (double*)malloc(N_faces*sizeof(double*));
  double volume = 0;
  for (int i=0; i<N_faces; i++){
    int v1 = index_combo[3*i];
    int v2 = index_combo[3*i+1];
    int v3 = index_combo[3*i+2];
    double n_i[3];
    double vertices_in[4][3];
    for (int l=0; l<N_vertices; l++){
      if (l != v1 && l != v2 && l != v3){
        for (int j=0; j<3; j++){
          vertices_in[0][j] = vertices[id_fac+3*v1+j];
          vertices_in[1][j] = vertices[id_fac+3*v2+j];
          vertices_in[2][j] = vertices[id_fac+3*v3+j];
          vertices_in[3][j] = vertices[id_fac+3*l+j];
        }
      normal_vector_triangular_face_nonorm(vertices_in, n_i);
      break;
      }
    }
    volume_elements[i] = (n_i[0]*vertices[id_fac+3*v1] + n_i[1]*vertices[id_fac+3*v1+1] + n_i[2]*vertices[id_fac+3*v1+2])/6;
    volume += volume_elements[i];
  }
  printf("volume = %.5f\n", volume);
  double rho = mass/volume;
  double P[9];
  for (int i=0; i<9; i++){
    P[i] = 0;
  }
  double D[3];
  double E[3];
  double F[3];
  for (int i=0; i<N_faces; i++){
    double prefac = rho*volume_elements[i]/20;
    int v1 = index_combo[3*i];
    int v2 = index_combo[3*i+1];
    int v3 = index_combo[3*i+2];
    for (int l=0; l<3; l++){
      D[l] = vertices[id_fac + 3*v1 + l];
      E[l] = vertices[id_fac + 3*v2 + l];
      F[l] = vertices[id_fac + 3*v3 + l];
    }
    for (int j=0; j<3; j++){
      for (int k=0; k<3; k++){
        P[3*j + k] += prefac*(2*(D[j]*D[k] + E[j]*E[k] + F[j]*F[k]) + D[j]*E[k] + D[k]*E[j] 
                  + D[j]*F[k] + D[k]*F[j] + E[j]*F[k] + E[k]*F[j]);
      }
    }
  } 
  I[0] = P[4] + P[8];
  // 01 and 02 swapped here...
  I[1] = -P[2];
  I[2] = -P[1];
  // 10 and 20 swapped here...
  I[3] = -P[6];
  I[4] = P[0] + P[8];
  I[5] = -P[5];
  // and here
  I[6] = -P[3];
  I[7] = -P[7];
  I[8] = P[4] + P[0];

  free(volume_elements);
  return 0;
}


int normal_vector_triangular_face(double r_vectors[4][3], double n_i[3]){
  /*Computes normal vector of a triangle*/
  double r2r1[3];
  double r3r1[3];
  double r4r1[3];
  for (int i=0; i<3; i++){
    r2r1[i] = r_vectors[1][i] - r_vectors[0][i];
    r3r1[i] = r_vectors[2][i] - r_vectors[0][i];
    r4r1[i] = r_vectors[3][i] - r_vectors[0][i];
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
  }
  else{
    n_i[0] = -ni_temp[0];
    n_i[1] = -ni_temp[1];
    n_i[2] = -ni_temp[2];
  }
  return 0;
}

int normal_vector_triangular_face_nonorm(double r_vectors[4][3], double n_i[3]){
  /*Computes normal vector of a triangle*/
  double r2r1[3];
  double r3r1[3];
  double r4r1[3];
  for (int i=0; i<3; i++){
    r2r1[i] = r_vectors[1][i] - r_vectors[0][i];
    r3r1[i] = r_vectors[2][i] - r_vectors[0][i];
    r4r1[i] = r_vectors[3][i] - r_vectors[0][i];
  }
  double cross_prod[3];
  cross_product(r3r1, r2r1, cross_prod);
  double norm_cproduct = absolute_value_vector(cross_prod);
  double ni_temp[3];
  ni_temp[0] = cross_prod[0];
  ni_temp[1] = cross_prod[1];
  ni_temp[2] = cross_prod[2];
  
  double d = dot_product(ni_temp, r4r1);
  if (d < 0){
    n_i[0] = ni_temp[0];
    n_i[1] = ni_temp[1];
    n_i[2] = ni_temp[2];
  }
  else{
    n_i[0] = -ni_temp[0];
    n_i[1] = -ni_temp[1];
    n_i[2] = -ni_temp[2];
  }
  return 0;
}


int normal_vector_triangle_righthand(int indices[4], double *vertices, double n_i[3]){
  /* 
  Computes normal vector of a triangular face.
  Also reorders vertices to maintain counter clockwise orientation.
  */
  int setup = 0;
  int flip = 0;
  int v1 = indices[0];
  int v2 = indices[1];
  int v3 = indices[2];
  int v4 = indices[3]; 
  double r2r1[3];
  double r3r1[3];
  double r4r1[3];
  while (setup == 0){
    for (int i=0; i<3; i++){
      r2r1[i] = vertices[3*v2 + i] - vertices[3*v1 + i];
      r3r1[i] = vertices[3*v3 + i] - vertices[3*v1 + i];
      r4r1[i] = vertices[3*v4 + i] - vertices[3*v1 + i]; 
    }
    double cross_prod[3];
    cross_product(r3r1, r2r1, cross_prod);
    double ni_temp[3];
    ni_temp[0] = cross_prod[0];
    ni_temp[1] = cross_prod[1];
    ni_temp[2] = cross_prod[2];
    double d = dot_product(ni_temp, r4r1);
    if (d <= 0){
      setup = 1;
      n_i[0] = ni_temp[0];
      n_i[1] = ni_temp[1];
      n_i[2] = ni_temp[2];
       
    }
    else{
      flip = 1;
      v1 = indices[2];
      v3 = indices[0];
    }
  }
  if (flip == 0){
    indices[0] = v3;
    indices[2] = v1;
  }
  return 0;
}

int normal_dot_check(int M, double **N_faces){
  double dot_prod_normal_vecs;
  for (int m=0; m<M; m++){
    for (int n=0; n<M-1; n++){
      if (m > n){
        dot_prod_normal_vecs = N_faces[m][0]*N_faces[n][0] + N_faces[m][1]*N_faces[n][1] + N_faces[m][2]*N_faces[n][2];
        if (dot_prod_normal_vecs <= 0){
          return 0;
        }
      }
    }
  }
  return 1;
}

int unique_indices(int input_ids[4]){
  for (int i=0; i<3; i++){
    for (int j=i+1; j < 4; j++){
      if (input_ids[i] == input_ids[j]){
        return 0;
      }
    }
  }
  return 1;
}

int * Face_id_combinations(double* vertices, int N_vertices, int * N_faces){
  int i, j, k;
  int M = N_vertices - 3;
  int N_extra_normals = (M-1)*M/2;
  int N_faces_local = 0;
  int *face_ids;
  if (N_extra_normals == 0){
    *N_faces = 4;
    int count = 0;
    face_ids = (int*)malloc(12 * sizeof(int));
    for (i=0; i<2; i++){
      for (j=1; j<3; j++){
        for (k=2; k<4; k++){
          if (i<j && j<k){
            face_ids[3*count] = i;
            face_ids[3*count+1] = j;
            face_ids[3*count+2] = k;
            count += 1; 
          }
        }
      }
    }
  }
  else{
    int l;
    double **N_vectors = (double **)malloc(M * sizeof(double *));
    for (i=0; i<M; i++){
      N_vectors[i] = (double*)malloc(3 * sizeof(double *));
    }
    face_ids = (int*)malloc(3 * sizeof(int));
    int allocate = 0;
    int is_segment;
    int counter;
    int indices[4];
    int unique_ids;
    double normal_vector[3];
    //double r_vectors[4][3];
    for (i=0; i<N_vertices; i++){
      for (j=0; j<N_vertices; j++){
        for (k=0; k<N_vertices; k++){
          counter = 0;
          if (i < j && j < k){
            for (l=0; l<N_vertices; l++){
              indices[0] = i;
              indices[1] = j;
              indices[2] = k;
              indices[3] = l;
              unique_ids = unique_indices(indices);
              if (unique_ids == 1){
                /*
                for (m=0; m<4; m++){
                  for (n=0; n<3; n++){
                    r_vectors[m][n] = vertices[3*m + n];
                  }
                }
                printf("%d %d %d %d\n", i,j,k,l);
                normal_vector_triangular_face(r_vectors, normal_vector);
                */
                normal_vector_triangle_righthand(indices, vertices, normal_vector);
                N_vectors[counter][0] = normal_vector[0];
                N_vectors[counter][1] = normal_vector[1];
                N_vectors[counter][2] = normal_vector[2];
                counter += 1;
              }
            }
            is_segment = normal_dot_check(M, N_vectors);
            if (is_segment == 1){
              if (allocate == 1){
                //printf("Reallocing with %d \n", N_faces_local);
                face_ids = (int *)realloc(face_ids, 3*(N_faces_local+1) * sizeof(int));
              }
              face_ids[3*N_faces_local] = indices[0];
              face_ids[3*N_faces_local+1] = indices[1];
              face_ids[3*N_faces_local+2] = indices[2];
              //printf("face ids set as %d %d %d\n", indices[0], indices[1], indices[2]);
              N_faces_local += 1;
              allocate = 1;
            }
          }
        }
      }
    }

    for (i=0; i<M; i++){
      free(N_vectors[i]);
    }
    free(N_vectors);
    *N_faces = N_faces_local;
  }
  return face_ids;
}

int get_centroid_tetrahedron(double centroid[3], double **vertices, int nvertices){
  /* Centroid for a tetrahedron */
  double x_sum = 0;
  double y_sum = 0;
  double z_sum = 0;
  for (int i = 0; i < nvertices; i++){
    x_sum += vertices[i][0];
    y_sum += vertices[i][1];
    z_sum += vertices[i][2]; 
  }
  centroid[0] = x_sum/4;
  centroid[1] = y_sum/4;
  centroid[2] = z_sum/4;
  return 0;
}


int gluInvertMatrix(double A[4][4], double invOut[4][4]){
  /* See https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix */ 
  double m[16];
  int i, j;
  for (i=0; i<4; i++){
    for (j=0; j<4; j++){
      m[4*i+j] = A[i][j];
    }
  }
  double inv[16];

  inv[0] = m[5]  * m[10] * m[15] - 
           m[5]  * m[11] * m[14] - 
           m[9]  * m[6]  * m[15] + 
           m[9]  * m[7]  * m[14] +
           m[13] * m[6]  * m[11] - 
           m[13] * m[7]  * m[10];

  inv[4] = -m[4]  * m[10] * m[15] + 
            m[4]  * m[11] * m[14] + 
            m[8]  * m[6]  * m[15] - 
            m[8]  * m[7]  * m[14] - 
            m[12] * m[6]  * m[11] + 
            m[12] * m[7]  * m[10];

  inv[8] = m[4]  * m[9] * m[15] - 
           m[4]  * m[11] * m[13] - 
           m[8]  * m[5] * m[15] + 
           m[8]  * m[7] * m[13] + 
           m[12] * m[5] * m[11] - 
           m[12] * m[7] * m[9];

  inv[12] = -m[4]  * m[9] * m[14] + 
             m[4]  * m[10] * m[13] +
             m[8]  * m[5] * m[14] - 
             m[8]  * m[6] * m[13] - 
             m[12] * m[5] * m[10] + 
             m[12] * m[6] * m[9];

  inv[1] = -m[1]  * m[10] * m[15] + 
            m[1]  * m[11] * m[14] + 
            m[9]  * m[2] * m[15] - 
            m[9]  * m[3] * m[14] - 
            m[13] * m[2] * m[11] + 
            m[13] * m[3] * m[10];

  inv[5] = m[0]  * m[10] * m[15] - 
           m[0]  * m[11] * m[14] - 
           m[8]  * m[2] * m[15] + 
           m[8]  * m[3] * m[14] + 
           m[12] * m[2] * m[11] - 
           m[12] * m[3] * m[10];

  inv[9] = -m[0]  * m[9] * m[15] + 
            m[0]  * m[11] * m[13] + 
            m[8]  * m[1] * m[15] - 
            m[8]  * m[3] * m[13] - 
            m[12] * m[1] * m[11] + 
            m[12] * m[3] * m[9];

  inv[13] = m[0]  * m[9] * m[14] - 
            m[0]  * m[10] * m[13] - 
            m[8]  * m[1] * m[14] + 
            m[8]  * m[2] * m[13] + 
            m[12] * m[1] * m[10] - 
            m[12] * m[2] * m[9];

  inv[2] = m[1]  * m[6] * m[15] - 
           m[1]  * m[7] * m[14] - 
           m[5]  * m[2] * m[15] + 
           m[5]  * m[3] * m[14] + 
           m[13] * m[2] * m[7] - 
           m[13] * m[3] * m[6];

  inv[6] = -m[0]  * m[6] * m[15] + 
            m[0]  * m[7] * m[14] + 
            m[4]  * m[2] * m[15] - 
            m[4]  * m[3] * m[14] - 
            m[12] * m[2] * m[7] + 
            m[12] * m[3] * m[6];

  inv[10] = m[0]  * m[5] * m[15] - 
            m[0]  * m[7] * m[13] - 
            m[4]  * m[1] * m[15] + 
            m[4]  * m[3] * m[13] + 
            m[12] * m[1] * m[7] - 
            m[12] * m[3] * m[5];

  inv[14] = -m[0]  * m[5] * m[14] + 
             m[0]  * m[6] * m[13] + 
             m[4]  * m[1] * m[14] - 
             m[4]  * m[2] * m[13] - 
             m[12] * m[1] * m[6] + 
             m[12] * m[2] * m[5];

  inv[3] = -m[1] * m[6] * m[11] + 
            m[1] * m[7] * m[10] + 
            m[5] * m[2] * m[11] - 
            m[5] * m[3] * m[10] - 
            m[9] * m[2] * m[7] + 
            m[9] * m[3] * m[6];

  inv[7] = m[0] * m[6] * m[11] - 
           m[0] * m[7] * m[10] - 
           m[4] * m[2] * m[11] + 
           m[4] * m[3] * m[10] + 
           m[8] * m[2] * m[7] - 
           m[8] * m[3] * m[6];

  inv[11] = -m[0] * m[5] * m[11] + 
             m[0] * m[7] * m[9] + 
             m[4] * m[1] * m[11] - 
             m[4] * m[3] * m[9] - 
             m[8] * m[1] * m[7] + 
             m[8] * m[3] * m[5];

  inv[15] = m[0] * m[5] * m[10] - 
            m[0] * m[6] * m[9] - 
            m[4] * m[1] * m[10] + 
            m[4] * m[2] * m[9] + 
            m[8] * m[1] * m[6] - 
            m[8] * m[2] * m[5];

  double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

  if (det == 0){
    printf("Matrix for invertion is singular. \n");
    exit(0);
  }

  double det_inv = 1.0 / det;

  for (i=0; i<4; i++){
    for (j=0; j<4; j++){
      invOut[i][j] = inv[4*i+j] * det_inv;
    }
  }
  return 0;
}

int matrix_transpose_4x4(double A[4][4], double result[4][4]){
  /* Transpose of a 4x4 matrix*/
  for (int i=0; i<4; i++){
    for (int j=0; j<4; j++){
      result[i][j] = A[j][i];
    }
  }
  return 0;
}

int Rotation_matrix_components_4x4(double phi, double theta, double psi, double result[4][4]){
  // A function that calculates the rotation matrix and its transpose.
  double sinphi = sin(phi);
  double cosphi = cos(phi);
  double sintheta = sin(theta);
  double costheta = cos(theta);
  double sinpsi = sin(psi);
  double cospsi = cos(psi);
  double sinpsint;
  double cospsint;

  double Rx[4][4] = {{1, 0, 0, 0}, 
                     {0, cosphi, -sinphi, 0}, 
                     {0, sinphi, cosphi, 0}, 
                     {0, 0, 0, 1}};
  double Ry[4][4] = {{costheta, 0, sintheta, 0},
                     {0, 1, 0, 0},
                     {-sintheta, 0, costheta, 0},
                     {0, 0, 0, 1}};
  double Rz[4][4] = {{cospsi, -sinpsi, 0, 0},
                     {sinpsi, cospsi, 0, 0},
                     {0, 0, 1, 0},
                     {0, 0, 0, 1}};

  double multiple1[4][4];
  int i, j, k;
  double sum;
  // Multiply Rz*Ry
  for (i=0; i<4; i++){
    for (j=0; j<4; j++){
      sum = 0;
      for (k=0; k<4; k++){
        sum += Rz[i][k]*Ry[k][j];
      }
      multiple1[i][j] = sum;
    }
  }
  // Final multiplication
  for (i=0; i<4; i++){
    for (j=0; j<4; j++){
      sum = 0;
      for (k=0; k<4; k++){
        sum += multiple1[i][k]*Rx[k][j];
      }
      result[i][j] = sum;
    }
  }
  return 0;
}

int Rotation_matrix_Euler_param_4x4(double e0, double e1, double e2, double e3, double R_matrix[4][4]){
  R_matrix[0][0] = 2*(e0*e0 + e1*e1 - 0.5);
  R_matrix[0][1] = 2*(e1*e2 - e0*e3);
  R_matrix[0][2] = 2*(e1*e3 + e0*e2);
  R_matrix[0][3] = 0;

  R_matrix[1][0] = 2*(e1*e2 + e0*e3);
  R_matrix[1][1] = 2*(e0*e0 + e2*e2 - 0.5);
  R_matrix[1][2] = 2*(e2*e3 - e0*e1);
  R_matrix[1][3] = 0;

  R_matrix[2][0] = 2*(e1*e3 - e0*e2);
  R_matrix[2][1] = 2*(e2*e3 + e0*e1);
  R_matrix[2][2] = 2*(e0*e0 + e3*e3 - 0.5);
  R_matrix[2][3] = 0;

  R_matrix[3][0] = 0;
  R_matrix[3][1] = 0;
  R_matrix[3][2] = 0;
  R_matrix[3][3] = 1;
  return 0;
}



int S_matrix(double a, double b, double c, double x0, double y0, double z0, double R[4][4], double result[4][4]){
  /* 4x4 matrix used for ellipsoid intersection check */

  // Rotate centroid
  double newx = x0*R[0][0] + y0*R[1][0] + z0*R[2][0] + R[3][0];
  double newy = x0*R[0][1] + y0*R[1][1] + z0*R[2][1] + R[3][1];
  double newz = x0*R[0][2] + y0*R[1][2] + z0*R[2][2] + R[3][2];

  double a_sq = a*a;
  double b_sq = b*b;
  double c_sq = c*c;
  double G = -newx/a_sq;
  double H = -newy/b_sq;
  double J = -newz/c_sq;
  double K = newx*newx/a_sq + newy*newy/b_sq + newz*newz/c_sq - 1;
  double S[4][4] = {{1/a_sq, 0, 0, G},
                    {0, 1/b_sq, 0, H}, 
                    {0, 0, 1/c_sq, J}, 
                    {G, H, J, K}};
  double R_T[4][4];
  matrix_transpose_4x4(R, R_T);
  double multiple1[4][4];
  double sum;
  int i, j, k;
  // First matrix multiplcation
  for (i=0; i<4; i++){
    for (j=0; j<4; j++){
      sum = 0;
      for (k=0; k<4; k++){
        sum += R[i][k]*S[k][j];
      }
      multiple1[i][j] = sum;
    }
  }
  // Second matrix multiplication, final result
  for (i=0; i<4; i++){
    for (j=0; j<4; j++){
      sum = 0;
      for (k=0; k<4; k++){
        sum += multiple1[i][k]*R_T[k][j];
      }
      result[i][j] = sum;
    }
  }
  return 0;
}

int ellipsoid_intersect_check(double semiaxes[6], double input[11], double positions[6], int eulerparam){
  /* 
  Check whether or not two ellipsoids intersect.
  Intersect/collide when eigevalues are complex or when the real components are (nearly) equal.
  */
  // Semiaxes
  double a_A = semiaxes[0];
  double b_A = semiaxes[1];
  double c_A = semiaxes[2];
  double a_B = semiaxes[3];
  double b_B = semiaxes[4];
  double c_B = semiaxes[5];
  // Positions
  double x_A = positions[0];
  double y_A = positions[1];
  double z_A = positions[2];
  double x_B = positions[3];
  double y_B = positions[4];
  double z_B = positions[5];

  double Rot_A[4][4];
  double Rot_B[4][4];
  if (eulerparam){
    double e0_A = input[3];
    double e1_A = input[4];
    double e2_A = input[5];
    double e3_A = input[6];
    double e0_B = input[7];
    double e1_B = input[8];
    double e2_B = input[9];
    double e3_B = input[10];
    Rotation_matrix_Euler_param_4x4(e0_A, e1_A, e2_A, e3_A, Rot_A);
    Rotation_matrix_Euler_param_4x4(e0_B, e1_B, e2_B, e3_B, Rot_B); 
  }
  else{
    double phi_A = input[3];
    double theta_A = input[4];
    double psi_A = input[5];
    double phi_B = input[6];
    double theta_B = input[7];
    double psi_B = input[8];
    Rotation_matrix_components_4x4(phi_A, theta_A, psi_A, Rot_A);
    Rotation_matrix_components_4x4(phi_B, theta_B, psi_B, Rot_B); 
  }
  /*
  printf("[%.5f, %.5f, %.5f, %.5f]]\n", Rot_A[0][0], Rot_A[0][1], Rot_A[0][2], Rot_A[0][3]);
  printf("[%.5f, %.5f, %.5f, %.5f]]\n", Rot_A[1][0], Rot_A[1][1], Rot_A[1][2], Rot_A[1][3]);
  printf("[%.5f, %.5f, %.5f, %.5f]]\n", Rot_A[2][0], Rot_A[2][1], Rot_A[2][2], Rot_A[2][3]);
  printf("[%.5f, %.5f, %.5f, %.5f]]\n", Rot_A[3][0], Rot_A[3][1], Rot_A[3][2], Rot_A[3][3]);
  exit(0);
  */
  double A_ellip[4][4];
  double B_ellip[4][4];
  S_matrix(a_B, b_B, c_B, x_B, y_B, z_B, Rot_B, B_ellip);
  S_matrix(a_A, b_A, c_A, x_A, y_A, z_A, Rot_A, A_ellip);

  double A_inv[4][4];
  gluInvertMatrix(A_ellip, A_inv);
  // Solve eigenvalue problem
  double matrix_eigen[16];
  double sum;
  int i;
  for (i=0; i<4; i++){
    for (int j=0; j<4; j++){
      sum = 0;
      for (int k=0; k<4; k++){
        sum += A_inv[i][k]*B_ellip[k][j];
      }
      matrix_eigen[4*i+j] = sum;
    }
  }
  
  /*
  printf("[%.5f, %.5f, %.5f, %.5f]]\n", A_ellip[0][0], A_ellip[0][1], A_ellip[0][2], A_ellip[0][3]);
  printf("[%.5f, %.5f, %.5f, %.5f]]\n", A_ellip[1][0], A_ellip[1][1], A_ellip[1][2], A_ellip[1][3]);
  printf("[%.5f, %.5f, %.5f, %.5f]]\n", A_ellip[2][0], A_ellip[2][1], A_ellip[2][2], A_ellip[2][3]);
  printf("[%.5f, %.5f, %.5f, %.5f]]\n", A_ellip[3][0], A_ellip[3][1], A_ellip[3][2], A_ellip[3][3]);
  printf("\n");
  printf("[%.5f, %.5f, %.5f, %.5f]]\n", B_ellip[0][0], B_ellip[0][1], B_ellip[0][2], B_ellip[0][3]);
  printf("[%.5f, %.5f, %.5f, %.5f]]\n", B_ellip[1][0], B_ellip[1][1], B_ellip[1][2], B_ellip[1][3]);
  printf("[%.5f, %.5f, %.5f, %.5f]]\n", B_ellip[2][0], B_ellip[2][1], B_ellip[2][2], B_ellip[2][3]);
  printf("[%.5f, %.5f, %.5f, %.5f]]\n", B_ellip[3][0], B_ellip[3][1], B_ellip[3][2], B_ellip[3][3]);
   
  printf("\n");
  printf("[%.5f, %.5f, %.5f, %.5f]\n", matrix_eigen[0], matrix_eigen[1], matrix_eigen[2], matrix_eigen[3]);
  printf("[%.5f, %.5f, %.5f, %.5f]\n", matrix_eigen[4], matrix_eigen[5], matrix_eigen[6], matrix_eigen[7]);
  printf("[%.5f, %.5f, %.5f, %.5f]\n", matrix_eigen[8], matrix_eigen[9], matrix_eigen[10], matrix_eigen[11]);
  printf("[%.5f, %.5f, %.5f, %.5f]\n", matrix_eigen[12], matrix_eigen[13], matrix_eigen[14], matrix_eigen[15]);
  */
  gsl_matrix_view m = gsl_matrix_view_array (matrix_eigen, 4, 4);

  gsl_vector_complex *eval = gsl_vector_complex_alloc (4);
  gsl_matrix_complex *evec = gsl_matrix_complex_alloc (4, 4);

  gsl_eigen_nonsymmv_workspace * w = gsl_eigen_nonsymmv_alloc (4);
  gsl_eigen_nonsymmv (&m.matrix, eval, evec, w);
  gsl_eigen_nonsymmv_free (w);

  gsl_eigen_nonsymmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC);
  int return_val = 0;
  for (i = 0; i < 4; i++){
    gsl_complex eval_i = gsl_vector_complex_get(eval, i);
    //printf("%.16f, %.16f\n", GSL_REAL(eval_i), GSL_IMAG(eval_i));
    if (fabs(GSL_IMAG(eval_i)) > 0){
      return_val = 1;
    }
  }

  double r1 = GSL_REAL(gsl_vector_complex_get(eval, 0));
  double r2 = GSL_REAL(gsl_vector_complex_get(eval, 1));
  double r3 = GSL_REAL(gsl_vector_complex_get(eval, 2));
  double r4 = GSL_REAL(gsl_vector_complex_get(eval, 3));
  //printf("%.16e, %.16e\n", fabs(r1-r2), fabs(r3-r4));
  if (fabs(r1-r2) < 1e-14){
    return_val = 1;
  }
  else if (fabs(r3-r4) < 1e-14){
    return_val = 1;
  }

  gsl_vector_complex_free (eval);
  gsl_matrix_complex_free (evec);

  return return_val;
}

int polyhedron_sphere_intersection_simple(double **vertices, int n_vertices, double pos_B[3], double radius_B){
  /*
  Simple implementation of a polyhedron and sphere intersection check.
  Checks if any of the vertices of the polyhedron are inside the sphere.
  Collision is met if there exists at least one vertex inside the sphere.
  Computed relative to the primary.
  */
  int collision = 0;
  for (int i=0; i<n_vertices; i++){
    double xc = pos_B[0] - vertices[i][0];
    double yc = pos_B[1] - vertices[i][1];
    double zc = pos_B[2] - vertices[i][2];
    double vertex_distance = sqrt(xc*xc + yc*yc + zc*zc);
    if (vertex_distance < radius_B){
      collision = 1;
      break;
    }
  }
  return collision;
}
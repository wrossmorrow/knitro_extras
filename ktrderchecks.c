/*
 *  ktrextras.c
 *  
 *
 *  Created by W. Ross Morrow on 12/1/11.
 *  Copyright 2011 Iowa State University. All rights reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <vecLib/cblas.h>

#include <knitro.h>

#include "ktrextras.h"

#if !defined(MAX)
#  define MAX(a,b)  ((a) > (b) ? (a) : (b))
#endif

#if !defined(MIN)
#  define MIN(a,b)  ((a) < (b) ? (a) : (b))
#endif

#if !defined(ABS)
#  define ABS(a)    ((a) > 0 ? (a) : -(a))
#endif

#define MAX_S 16

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Check first derivatives for a KNITRO callback routine
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void ktr_evalga_check(int			N,			// number of variables
					  int			M,			// number of constraints
					  KTR_callback  func_call,	// callback to evaluate objective and constraint values
					  KTR_callback  grad_call,	// callback routine to evaluate objective gradient and constraint Jacobian
					  int			Jnnz,		// (constant) constraint Jacobian number of nonzeros
					  int*			Jrows,		// (constant) constraint Jacobian structure - row indices
					  int*			Jcols,		// (constant) constraint Jacobian structure - column indices
					  double*		x,			// point at which we want to check first derivatives
					  double*		xLoBnds,	// lower bounds on variables
					  double*		xUpBnds,	// upper bounds on variables
					  void*			userParams, // user parameters
					  double*		info)		// information (can be NULL)
{
	int base = 0;
	
	double curr_obj;
	double new_obj;
	
	double* new_x;
	
	double* curr_cns;
	double* new_cns;
	
	double* curr_grad;
	double* curr_cjac;
	
	double* fdgrad;
	double* fdcjac;
	
	double h, hinv, H, Hinv;
	
	double graddiffnorm[MAX_S] = { 0.0 };
	double cjacdiffnorm[MAX_S] = { 0.0 };
	
	int s, m, n, e;
	
	int flag;
	
	if( func_call == NULL || grad_call == NULL || x == NULL ) { return; }
	
	// allocate memory needed
	new_x      = (double*)calloc( N    , sizeof(double) );
	curr_cns   = (double*)calloc( M    , sizeof(double) );
	new_cns    = (double*)calloc( M    , sizeof(double) );
	curr_grad  = (double*)calloc( N    , sizeof(double) );
	curr_cjac  = (double*)calloc( Jnnz , sizeof(double) );
	fdgrad     = (double*)calloc( N    , sizeof(double) );
	fdcjac     = (double*)calloc( M*N  , sizeof(double) );
	
	printf("\n\n");
	printf("KTR_EVALGA_CHECK:: There are %i variables and %i constraints\n",N,M);
	printf("KTR_EVALGA_CHECK:: Constraint Jacobian is %i x %i ",M,N);
	printf("with %i nonzeros (%0.4f %% dense)\n",Jnnz,100.0*(double)Jnnz/((double)(M*N)));
	// for( e = 0 ; e < Gnnz ; e++ ) {
	//	printf("                      element %i gives (%i,%i)\n",e,Grows[e],Gcols[e]);
	//}
	
	// ensure x is within bounds (using Euclidean projection)
	for( n = 0 ; n < N ; n++ ) {
		if( x[n] < xLoBnds[n] ) { x[n] = xLoBnds[n]; }
		if( x[n] > xUpBnds[n] ) { x[n] = xUpBnds[n]; }
	}
	
	printf("KTR_EVALGA_CHECK:: Current Point:\n");
	printf("  x = [ %0.6f ",x[0]);
	for( n = 1 ; n < N ; n++ ) { printf(", %0.6f ",x[n]); }
	printf("]\n");
	
	// get current objective function and constraint function values
	func_call(KTR_RC_EVALFC, N, M, 0, 0, x, NULL, &curr_obj, curr_cns, NULL, NULL, NULL, NULL, userParams);
	
	// get current objective gradient and constraint Jacobian values
	grad_call(KTR_RC_EVALGA, N, M, Jnnz, 0, x, NULL, NULL, NULL, curr_grad, curr_cjac, NULL, NULL, userParams);
	
	// finite differences
	for( s = 0 ; s < MAX_S ; s++ ) {
		
		// stepsize
		h = pow( 10.0 , - ( s - base ) );
		hinv = 1.0 / h;
		
		// for each direction...
		for( n = 0 ; n < N ; n++ ) {
			
			// perturb variables in each coordinate direction
			cblas_dcopy( N , x , 1 , new_x , 1 );
			
			H = ( 1.0 + fabs(x[n]) ) * h;
			Hinv = 1.0 / H;
			
			// respect bounds, with new point
			if( x[n] + H <= xUpBnds[n] ) {
				
				// use a * forward * difference (implicitly safeguarded to lie within bounds)
				
				// new_x <- min{ x + H * e_n , xUpBnds[n] }
				new_x[n] += H;
				
				// get * new * objective function and constraint function values
				func_call(KTR_RC_EVALFC, N, M, 0, 0, new_x, NULL, &new_obj, new_cns, NULL, NULL, NULL, NULL, userParams);
				
				// form finite differences
				
				// fdgrad[n] = ( new_obj - curr_obj ) / H
				fdgrad[n] = Hinv * ( new_obj - curr_obj );
				
				// fdcjac[:,n] = ( new_cns - curr_cns ) / H
				cblas_dcopy( M , new_cns , 1 , fdcjac+M*n , 1 );
				cblas_daxpy( M , -1.0 , curr_cns , 1 , fdcjac+M*n , 1 );
				cblas_dscal( M , Hinv , fdcjac+M*n , 1 );
				
			} else {
				
				// we assume (or assert) that 
				//
				//		xLoBnds[n] <= x[n] <= xUpBnds[n]
				//
				// thus, if the if statement above has been violated, 
				// we must have that x[n] + H > xUpBnds[n]
				
				// here use a * backward * difference, safeguarded to lie within bounds
				
				// new_x <- max{ x - H * e_n , xLoBnds[n] }
				new_x[n] -= H;
				if( new_x[n] < xLoBnds[n] ) { new_x[n] = xLoBnds[n]; }
				
				// get * new * objective function and constraint function values
				func_call(KTR_RC_EVALFC, N, M, 0, 0, new_x, NULL, &new_obj, new_cns, NULL, NULL, NULL, NULL, userParams);
				
				// form * backward * finite differences
				
				// fdgrad[n] = ( curr_obj - new_obj ) / H
				fdgrad[n] = Hinv * ( curr_obj - new_obj );
				
				// fdcjac[:,n] = ( curr_cns - new_cns ) / H
				cblas_dcopy( M , curr_cns , 1 , fdcjac+M*n , 1 );
				cblas_daxpy( M , -1.0 , new_cns , 1 , fdcjac+M*n , 1 );
				cblas_dscal( M , Hinv , fdcjac+M*n , 1 );
				
			}
			
		}
		
		// finite difference gradient and constraint Jacobian have been formed
		// compare to computed gradient and constraint Jacobian
		
		// computed gradient
		
		// graddiffnorm <- || fdgrad - grad ||_inf
		cblas_daxpy( N , -1.0 , curr_grad , 1 , fdgrad , 1 );
		graddiffnorm[s] = 0.0;
		for( n = 0 ; n < N ; n++ ) { 
			if( graddiffnorm[s] < fabs( fdgrad[n] ) ) { 
				graddiffnorm[s] = fabs( fdgrad[n] );
			}
		}
		
		// constraint Jacobian
		// first assess sparsity pattern
		for( m = 0 ; m < M ; m++ ) {
			for( n = 0 ; n < N ; n++ ) {
				if( fdcjac[ m + M*n ] != 0.0 ) {
					// make sure this entry is included in the sparse structure
					flag = 0;
					for( e = 0 ; e < Jnnz ; e++ ) {
						if( Jrows[e] == m && Jcols[e] == n ) { flag = 1; break; }
					}
					if( flag == 0 ) {
						printf("KTR_EVALGA_CHECK WARNING:: element [%i,%i] in the constraint Jacobian has a nonzero finite difference,\n",m,n);
						printf("                           but is not included in the given sparsity structure.\n");
					}
				}
			}
		}
		
		// then compute differences (we can overwrite elements of finite difference)
		cjacdiffnorm[s] = 0.0;
		for( e = 0 ; e < Jnnz ; e++ ) {
			fdcjac[ Jrows[e] + M * Jcols[e] ] -= curr_cjac[e];
			if( cjacdiffnorm[s] < fabs( fdcjac[ Jrows[e] + M * Jcols[e] ] ) ) {
				cjacdiffnorm[s] = fabs( fdcjac[ Jrows[e] + M * Jcols[e] ] );
			}
		}
		
	}
	
	printf("KTR_EVALGA_CHECK:: Objective Gradient:\n");
	for( s = 0 ; s < MAX_S ; s++ ) {
		h = pow( 10.0 , - ( s - base ) );
		printf("  h = %0.16f, || diff ||_inf = %0.16f\n",h,graddiffnorm[s]);
	}
	
	printf("KTR_EVALGA_CHECK:: Constraint Jacobian:\n");
	for( s = 0 ; s < MAX_S ; s++ ) {
		h = pow( 10.0 , - ( s - base ) );
		printf("  h = %0.16f, || diff ||_inf = %0.16f\n",h,cjacdiffnorm[s]);
	}
	
	// free memory allocated above
	free(new_x);
	free(curr_cns);
	free(new_cns);
	free(curr_grad);
	free(curr_cjac);
	
	free(fdgrad);
	free(fdcjac);
	
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Check first derivatives for a KNITRO callback routine, long version
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void ktr_evalga_check_l(int				N, // number of variables
						int				M, // number of constraints
						KTR_callback	func_call, // callback to evaluate objective and constraint values
						KTR_callback	grad_call, // callback routine to evaluate objective gradient and constraint Jacobian
						int				Jnnz, // (constant) constraint Jacobian number of nonzeros
						int*			Jrows, // (constant) constraint Jacobian structure - row indices
						int*			Jcols, // (constant) constraint Jacobian structure - column indices
						double*			x,		// point at which we want to check first derivatives
						double*			xLoBnds,  // lower bounds on variables
						double*			xUpBnds,  // upper bounds on variables
						void*			userParams, // user parameters
						double*			info)		// information (can be NULL)
{
	int base = 0;
	
	double curr_obj;				// current objective value, double precision
	double new_obj;					// new objective value, double precision
	
	double * new_x;					// perturbed x values, double precision
	long double * new_x_l;			// perturbed x values, extended precision
	
	double * curr_cns;				// current constraint values, double precision
	double * new_cns;				// new constraint values, double precision
	
	double * curr_grad;				// current computed objective gradient values
	double * curr_cjac;				// current computed constraint Jacobian values
	
	long double * fdgrad_l;			// finite difference objective gradient, extended precision
	
	int fdJnnz = 0;					// sparse finite difference constraint Jacobian nonzeros
	int * spfdjac_rows;				// sparse finite difference constraint Jacobian row elements
	int * spfdjac_cols;				// sparse finite difference constraint Jacobian column elements
	long double * spfdjac_data_l;	// sparse finite difference constraint Jacobian data, extended precision
	
	int * spfdjac_col_rows;
	long double * spfdjac_col_data_l;
	
	long double diff_l;				// temporary storage for differences computed below
	long double h, H, Hinv;			// step sizes and scalings for finite differences
	
	long double ztol_l = (long double)(1e-14); // tolerance for declaring jacobian elements zero
	
	double graddiffnorm[MAX_S] = { 0.0 };
	double cjacdiffnorm[MAX_S] = { 0.0 };
	
	int s, m, n, e, i, j; // loop indices
	
	int found; // flag for comparing given constraint Jacobian structure and that determined by finite differences
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	if( func_call == NULL || grad_call == NULL || x == NULL ) { return; }
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	new_x      = (double*)calloc( N , sizeof(double) );
	new_x_l    = (long double*)calloc( N , sizeof(long double) );
	
	curr_cns   = (double*)calloc( M    , sizeof(double) );
	new_cns    = (double*)calloc( M    , sizeof(double) );
	
	curr_grad  = (double*)calloc( N    , sizeof(double) );
	curr_cjac  = (double*)calloc( Jnnz , sizeof(double) );
	
	fdgrad_l   = (long double *)calloc( N , sizeof(long double) );
	
	spfdjac_col_rows   = (int*)calloc( M , sizeof(int) );
	spfdjac_col_data_l = (long double *)calloc( M , sizeof(long double) );
	
	// Jnnz is a size ESTIAMATE (hopefully correct); this will be resized below, 
	// if needed, but that signals a structure error. No warning is given, but
	// warnings given for specific structure errors
	spfdjac_rows   = (int*)calloc( Jnnz , sizeof(int) );
	spfdjac_cols   = (int*)calloc( Jnnz , sizeof(int) );
	spfdjac_data_l = (long double *)calloc( Jnnz , sizeof(long double) );
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	printf("\n\n");
	printf("KTR_EVALGA_CHECK_L:: There are %i variables and %i constraints\n",N,M);
	printf("KTR_EVALGA_CHECK_L:: Constraint Jacobian is %i x %i ",M,N);
	printf("with %i nonzeros (%0.4f %% dense)\n",Jnnz,100.0*(double)Jnnz/((double)(M*N)));
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// ensure x is within bounds (using Euclidean projection)
	for( n = 0 ; n < N ; n++ ) {
		if( x[n] < xLoBnds[n] ) { x[n] = xLoBnds[n]; }
		if( x[n] > xUpBnds[n] ) { x[n] = xUpBnds[n]; }
	}
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	printf("KTR_EVALGA_CHECK_L:: Current Point:\n");
	printf("  x = [ %0.6f ",x[0]);
	for( n = 1 ; n < N ; n++ ) { printf(", %0.6f ",x[n]); }
	printf("]\n");
	
	// get current objective function and constraint function values
	func_call(KTR_RC_EVALFC, N, M, 0, 0, x, NULL, &curr_obj, curr_cns, NULL, NULL, NULL, NULL, userParams);
	
	// get current objective gradient and constraint Jacobian values
	grad_call(KTR_RC_EVALGA, N, M, Jnnz, 0, x, NULL, NULL, NULL, curr_grad, curr_cjac, NULL, NULL, userParams);
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// finite differences
	for( s = 0 ; s < MAX_S ; s++ ) {
		
		// stepsize
		h = pow( (long double)(10.0) , - ( s - base ) );
		
		// reset finite-difference nonzeros here!
		fdJnnz = 0;
		
		// for each direction...
		for( n = 0 ; n < N ; n++ ) {
			
			// perturb variables in each coordinate direction (in extended precision)
			// with a relative perturbation
			for( i = 0 ; i < N ; i++ ) { new_x_l[i] = (long double)(x[i]); }
			H = ( (long double)(1.0) + fabs((long double)(x[n])) ) * h;
			Hinv = (long double)(1.0) / H;
			
			/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
			 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
			 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
			
			// respect bounds, with new point
			if( x[n] + (double)H <= xUpBnds[n] ) {
				
				// 
				// use a * forward * difference (implicitly safeguarded to lie within bounds)
				//
				
				// new_x <- min{ x + H * e_n , xUpBnds[n] }
				new_x_l[n] += H;
				
				// cast new_x_l as double in new_x (round off error only)
				// for call to given function for objective and constraint function
				// values
				for( i = 0 ; i < N ; i++ ) { new_x[i] = (double)(new_x_l[i]); }
				
				// get * new * objective function and constraint function values
				func_call(KTR_RC_EVALFC, N, M, 0, 0, new_x, NULL, &new_obj, new_cns, NULL, NULL, NULL, NULL, userParams);
				
				// 
				// form finite differences (in extended precision)
				// 
				
				fdgrad_l[n] = Hinv * ( (long double)(new_obj) - (long double)(curr_obj) );
				
				j = 0;
				for( i = 0 ; i < M ; i++ ) {
					
					// first, form scaled differences in extended precision
					diff_l = ( (long double)(new_cns[i]) - (long double)(curr_cns[i]) ) * Hinv;
				
					// then, if scaled difference is not approximately zero, store
					// in sparse data structure (double precision)
					if( diff_l > ztol_l ) {
						spfdjac_col_rows[j]   = i;		// i is the ROW index of this entry
														// (column index is n)
						spfdjac_col_data_l[j] = diff_l; // diff_l gives the entry value
						j++;
					}
					
				}
				
			} else {
				
				// we assume (or assert) that 
				//
				//		xLoBnds[n] <= x[n] <= xUpBnds[n]
				//
				// thus, if the if statement above has been violated, 
				// we must have that x[n] + H > xUpBnds[n]
				
				// 
				// here use a * backward * difference, safeguarded to lie within bounds
				// 
				
				// new_x <- min{ x + H * e_n , xUpBnds[n] }
				new_x_l[n] -= H;
				if( new_x_l[n] < (long double)(xLoBnds[n]) ) { new_x_l[n] = (long double)(xLoBnds[n]); }
				
				// cast new_x_l as double in new_x (round off error only)
				for( i = 0 ; i < N ; i++ ) { new_x[i] = (double)(new_x_l[i]); }
				
				// get * new * objective function and constraint function values
				func_call(KTR_RC_EVALFC, N, M, 0, 0, new_x, NULL, &new_obj, new_cns, NULL, NULL, NULL, NULL, userParams);
				
				// 
				// form * backward * finite differences (in extended precision)
				// 
				
				fdgrad_l[n] = Hinv * ( (long double)(curr_obj) - (long double)(new_obj) );
				
				j = 0;
				for( i = 0 ; i < M ; i++ ) {
					
					// first, form scaled differences in extended precision
					diff_l = ( (long double)(curr_cns[i]) - (long double)(new_cns[i]) ) * Hinv;
					
					// then, if scaled difference is not approximately zero, store
					// in sparse data structure (extended precision)
					if( diff_l > ztol_l ) {
						spfdjac_col_rows[j]   = i;		// i is the ROW index of this entry
														// (column index is n)
						spfdjac_col_data_l[j] = diff_l; // diff_l gives the entry value
						j++;
					}
					
				}
				
			}
			
			/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
			 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
			 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
			
			// add the current sparse column to the sparse finite difference 
			// constraint Jacobian
			
			if( fdJnnz + j > Jnnz ) {
				
				// need more space than our original estimate! 
				
				spfdjac_rows   = (int*)realloc( spfdjac_rows , (fdJnnz+j) * sizeof(int) );
				spfdjac_cols   = (int*)realloc( spfdjac_cols , (fdJnnz+j) * sizeof(int) );
				spfdjac_data_l = (long double*)realloc( spfdjac_data_l , (fdJnnz+j) * sizeof(long double) );
				if( spfdjac_rows == NULL || spfdjac_cols == NULL || spfdjac_data_l == NULL ) {
					printf("memory allocation error\n");
					exit(-1);
				}
				
			}
			
			// copy in new data
			for( e = 0 ; e < j ; e++ ) {
				spfdjac_rows  [ fdJnnz + e ] = spfdjac_col_rows[e];
				spfdjac_cols  [ fdJnnz + e ] = n;
				spfdjac_data_l[ fdJnnz + e ] = spfdjac_col_data_l[e];
			}
			
			// increment nonzeros in sparse finite difference 
			// constraint Jacobian
			fdJnnz += j;
			
		}
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
		// 
		// finite difference gradient and constraint Jacobian have been formed
		// compare to computed gradient and constraint Jacobian
		// 
		
		// computed gradient
		
		// graddiffnorm <- || fdgrad - grad ||_inf
		graddiffnorm[s] = 0.0;
		for( n = 0 ; n < N ; n++ ) {
			diff_l = fabs( fdgrad_l[n] - (long double)(curr_grad[n]) );
			if( graddiffnorm[s] < (double)(diff_l) ) { 
				graddiffnorm[s] = (double)(diff_l);
			}
		}
		
		// constraint Jacobian
		
		// then compute differences
		cjacdiffnorm[s] = 0.0;
		for( i = 0 ; i < fdJnnz ; i++ ) {
			
			// spfdjac element i: fdJac[ rows[i] , cols[i] ] = data[i]
			
			// we * should * find a corresponding Jacobian element
			found = 0;
			
			for( e = 0 ; e < Jnnz ; e++ ) {
				
				// J[ Jrows[e] , Jcols[e] ] = curr_cjac[e]
				if( Jrows[e] == spfdjac_rows[i] && Jcols[e] == spfdjac_cols[i] ) {
					
					found = 1;
					
					diff_l = fabs( spfdjac_data_l[i] - (long double)(curr_cjac[e]) );
					if( cjacdiffnorm[s] < (double)(diff_l) ) {
						cjacdiffnorm[s] = (double)(diff_l);
					}
					break;
					
				}
			}
			
			if( found == 0 ) {
				printf("KTR_EVALGA_CHECK_L WARNING:: element [%i,%i] in the constraint Jacobian has a nonzero finite difference,\n",spfdjac_rows[i],spfdjac_cols[i]);
				printf("                             but is not included in the given sparsity structure.\n");
			}
			
		}
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
	}
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	printf("KTR_EVALGA_CHECK_L:: Objective Gradient:\n");
	for( s = 0 ; s < MAX_S ; s++ ) {
		h = pow( 10.0 , - ( s - base ) );
		printf("  h = %0.16f, || diff ||_inf = %0.16f\n",(double)h,graddiffnorm[s]);
		if( info != NULL ) {
			if( s == 0 ) { 
				info[0] = graddiffnorm[s]; 
				info[1] = (double)h;
			} else { 
				if( info[0] > graddiffnorm[s] ) {
					info[0] = graddiffnorm[s]; 
					info[1] = (double)h;
				}
			}
		}
	}
	
	printf("KTR_EVALGA_CHECK_L:: Constraint Jacobian:\n");
	for( s = 0 ; s < MAX_S ; s++ ) {
		h = pow( 10.0 , - ( s - base ) );
		printf("  h = %0.16f, || diff ||_inf = %0.16f\n",(double)h,cjacdiffnorm[s]);
		if( info != NULL ) {
			if( s == 0 ) { 
				info[2] = cjacdiffnorm[s]; 
				info[3] = (double)h;
			} else { 
				if( info[2] > cjacdiffnorm[s] ) {
					info[2] = cjacdiffnorm[s]; 
					info[3] = (double)h;
				}
			}
		}
	}
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	free(new_x);
	free(new_x_l);
	
	free(curr_cns);
	free(new_cns);
	
	free(curr_grad);
	free(curr_cjac);
	
	free(fdgrad_l);
	
	free(spfdjac_col_rows);
	free(spfdjac_col_data_l);
	
	free(spfdjac_rows  ); 
	free(spfdjac_cols  );
	free(spfdjac_data_l);
	
}


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Check first derivatives for a KNITRO callback routine
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void ktr_evalhess_check(int			N,			// number of variables
						int			M,			// number of constraints
						KTR_callback grad_call,	// callback routine to evaluate objective gradient and constraint Jacobian
						KTR_callback hess_call,	// callback routine to evaluate Lagrangian Hessian
						int			Jnnz,		// (constant) constraint Jacobian number of nonzeros
						int*		Jrows,		// (constant) constraint Jacobian structure - row indices
						int*		Jcols,		// (constant) constraint Jacobian structure - column indices
						int			Hnnz,		// (constant) Hessian number of nonzeros
						int*		Hrows,		// (constant) Hessian structure - row indices
						int*		Hcols,		// (constant) Hessian structure - column indices
						double*		x,			// point at which we want to check Hessian
						double*		xLoBnds,	// lower bounds on variables
						double*		xUpBnds,	// upper bounds on variables
						double*		lambda,		// lambda multipliers to use
						void*		userParams, // user parameters
						double*		info)		// information (can be NULL)
{
	int base = 0;
	
	double* new_x;
	
	double* curr_grad;
	double* curr_cjac;
	
	double* new_grad;
	double* new_cjac;
	
	double* curr_hess;
	
	double* fdhess;
	
	double h, hinv, H, Hinv;
	
	double hessdiffnorm[MAX_S] = { 0.0 };
	
	int s, m, n, k, e, hbase;
	
	int flag;
	
	int Nsq;
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	if( grad_call == NULL || hess_call == NULL || x == NULL ) { return; }
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	Nsq = 0.5 * N * (N+1);
	
	printf("\n\n");
	printf("KTR_EVALHESS_CHECK:: There are %i variables and %i constraints\n",N,M);
	printf("KTR_EVALHESS_CHECK:: Hessian has %i nonzeros",Hnnz);
	printf(" (%0.4f %% dense)\n", 100.0 * (double)Hnnz / (0.5*(N*(N+1))) );
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// allocate memory needed
	new_x      = (double*)calloc( N    , sizeof(double) );
	
	curr_grad  = (double*)calloc( N    , sizeof(double) );
	new_grad   = (double*)calloc( N    , sizeof(double) );
	
	curr_cjac  = (double*)calloc( Jnnz , sizeof(double) );
	new_cjac   = (double*)calloc( Jnnz , sizeof(double) );
	
	curr_hess  = (double*)calloc( Hnnz , sizeof(double) );
	fdhess     = (double*)calloc( Nsq  , sizeof(double) );
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// ensure x is within bounds (using Euclidean projection)
	for( n = 0 ; n < N ; n++ ) {
		x[n] = MAX( xLoBnds[n] , x[n] );
		x[n] = MIN( xUpBnds[n] , x[n] );
	}
	
	printf("KTR_EVALHESS_CHECK:: Current Point:\n");
	printf("  x = [ %0.6f ",x[0]);
	for( n = 1 ; n < N ; n++ ) { printf(", %0.6f ",x[n]); }
	printf("]\n");
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// get current objective gradient and constraint Jacobian values
	grad_call(KTR_RC_EVALGA, 
			  N, 
			  M, 
			  Jnnz, 
			  0, 
			  x, 
			  NULL, 
			  NULL, 
			  NULL, 
			  curr_grad, 
			  curr_cjac, 
			  NULL, 
			  NULL, 
			  userParams);
	
	printf("got grad & jac: \n");
	
	// get current Hessian values
	hess_call(KTR_RC_EVALH, 
			  N, 
			  M, 
			  0, 
			  Hnnz, 
			  x, 
			  lambda, 
			  NULL, 
			  NULL, 
			  NULL, 
			  NULL, 
			  curr_hess, 
			  NULL, 
			  userParams);
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// finite differences
	for( s = 0 ; s < MAX_S ; s++ ) {
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
		// stepsize
		h = pow( 10.0 , - ( s - base ) );
		hinv = 1.0 / h;
		
		hbase = 0;
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
		// for each direction...
		for( n = 0 ; n < N ; n++ ) {
			
			// perturb variables in each coordinate direction
			cblas_dcopy( N , x , 1 , new_x , 1 );
			
			H = ( 1.0 + fabs(x[n]) ) * h;
			Hinv = 1.0 / H;
			
			// respect bounds, with new point
			if( x[n] + H <= xUpBnds[n] ) {
				
				// use a * forward * difference (implicitly safeguarded to lie within bounds)
				
				// new_x <- min{ x + H * e_n , xUpBnds[n] }
				new_x[n] += H;
				
				// get * new * objective gradient and constraint Jacobian values
				grad_call(KTR_RC_EVALGA, 
						  N, 
						  M, 
						  Jnnz, 
						  0, 
						  new_x, 
						  NULL, 
						  NULL, 
						  NULL, 
						  new_grad, 
						  new_cjac, 
						  NULL, 
						  NULL, 
						  userParams);
				
				// form finite differences: objective component
				
				// H(1:n,n) <- ( new_grad(1:n) - curr_grad(1:n) ) / H
				for( k = 0 ; k <= n ; k++ ) {
					fdhess[hbase+k] = Hinv * ( new_grad[k] - curr_grad[k] );
				}
				
				// form finite differences: constraint component
				
				for( e = 0 ; e < Jnnz ; e++ ) {
					
					m = Jrows[e]; // constraint index
					k = Jcols[e]; // derivative index
					
					if( k <= n ) { // only accummulate if in upper triangle
						fdhess[hbase+k] += Hinv * lambda[m] * ( new_cjac[e] - curr_cjac[e] );
					}
					
				}
				
				// increment base indexer for upper-triangular fdhess matrix
				hbase += n+1;
				
			} else {
				
				// we assume (or assert) that 
				//
				//		xLoBnds[n] <= x[n] <= xUpBnds[n]
				//
				// thus, if the if statement above has been violated, 
				// we must have that x[n] + H > xUpBnds[n]
				
				// here use a * backward * difference, safeguarded to lie within bounds
				
				// new_x <- max{ x - H * e_n , xLoBnds[n] }
				new_x[n] -= H;
				if( new_x[n] < xLoBnds[n] ) { new_x[n] = xLoBnds[n]; }
				
				// get * new * objective gradient and constraint Jacobian values
				grad_call(KTR_RC_EVALGA, 
						  N, 
						  M, 
						  Jnnz, 
						  0, 
						  new_x, 
						  NULL, 
						  NULL, 
						  NULL, 
						  new_grad, 
						  new_cjac, 
						  NULL, 
						  NULL, 
						  userParams);
				
				
				// form * backward * finite differences: objective component
				
				// H(1:n,n) <- ( new_grad(1:n) - curr_grad(1:n) ) / H
				for( k = 0 ; k <= n ; k++ ) {
					fdhess[hbase+k] = Hinv * ( curr_grad[k] - new_grad[k] );
				}
				
				// form * backward * finite differences: constraint component
				
				for( e = 0 ; e < Jnnz ; e++ ) {
					
					m = Jrows[e]; // constraint index
					k = Jcols[e]; // derivative index
					
					if( k <= n ) { // only accummulate if in upper triangle
						fdhess[hbase+k] += Hinv * lambda[m] * ( curr_cjac[e] - new_cjac[e] );
					}
					
				}
				
				// increment base indexer for upper-triangular fdhess matrix
				hbase += n+1;
				
			}
			
		}
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
		// finite difference Hessian has been formed; compare to computed version
		
		hbase = 0;
		
		for( n = 0 ; n < N ; n++ ) {
			for( k = 0 ; k <= n ; k++ ) {
				
				flag = 0;
				for( e = 0 ; e < Hnnz ; e++ ) {
					// printf("    H(%i,%i)\n",Hrows[e]+1,Hcols[e]+1);
					
					if( Hrows[e] == k && Hcols[e] == n ) { 
						flag = 1;
						fdhess[hbase+k] -= curr_hess[e];
						hessdiffnorm[s] = MAX( hessdiffnorm[s] , ABS( fdhess[hbase+k] ) );
						break; 
					}
					
				}
				
				if( flag == 0 && ABS( fdhess[hbase+k] ) > 0.0 ) {
					printf("KTR_EVALHESS_CHECK WARNING:: element [%i,%i] in the Lagrangian Hessian has a nonzero finite difference,\n",n,k);
					printf("                           but is not included in the given sparsity structure.\n");
				}
			}
			
			hbase += n+1;
			
		}
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
	}
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	printf("KTR_EVALHESS_CHECK::\n");
	for( s = 0 ; s < MAX_S ; s++ ) {
		h = pow( 10.0 , - ( s - base ) );
		printf("  h = %0.16f, || diff ||_inf = %0.16f\n",h,hessdiffnorm[s]);
	}
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	
	// free memory allocated above
	free(new_x);
	free(curr_grad);
	free(curr_cjac);
	free(curr_hess);
	
	free(new_grad);
	free(new_cjac);
	free(fdhess);
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
}




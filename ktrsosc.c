/*
 *  ktrsosc.c
 *  
 *
 *  Created by W. Ross Morrow on 10/12/11.
 *  Copyright 2011 Iowa State University. All rights reserved.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <vecLib/cblas.h>

#include "cpdt.h"

#include "ktrextras.h"

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * 
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// type of Hessian used in KNITRO SOSC check
#define KTR_SOSC_EXACT_HESSIAN		0
#define KTR_SOSC_HESSIAN_MULTIPLY	1
#define KTR_SOSC_FINITE_DIFF		2

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * 
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// data needed to use Hessian-multiply functions and directional finite differences
typedef struct {
	
	int type;
	
	int		N;
	int		M;
	
	double* x;
	double* newx;
	double* lambda;
	
	int		objGoal;
	
	KTR_callback* ktr_gradjac;
	KTR_callback* ktr_HessMult;
	
	double*	objGrad;
	double*	newobjGrad;
	
	int		rowmajorMMF;
	int		Jnnz;
	int*	Jrows;
	int*	Jcols;
	double* Jdata;
	double* newJdata;
	
	int		account;
	int*	cType;
	int*	actCons;
	
	double	steplength;
	
	void* userParams;
	
} KTR_HFSOSC_DATA;

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * Wrapper for using a Hessian-multiply function Hmult defined in 
 * cpdtests/defs.h
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// this function "translates" calls to CHL Hmult routine 
// must thus satisfy the prototype for an "Hmult" object
void KTR_HFSOSC_HessMult(int	 N, 
						 double* v, 
						 double* z, 
						 void*   params)
{
	KTR_HFSOSC_DATA* data;
	
	if( v == NULL || z == NULL || params == NULL ) {
		printf("KTR_HFSOSC_HessMult():: invalid argument(s).\n");
		return;
	}
	
	// cast params as a FDHESS data structure
	data = (KTR_HFSOSC_DATA*)params;
	
	if( data->N != N ) {
		printf("KTR_HFSOSC_HessMult():: size passed in inconsistent with KTR_HFSOSC_DATA.\n");
		return;
	}
	
	if( data->type != KTR_SOSC_HESSIAN_MULTIPLY || data->ktr_HessMult == NULL ) {
		printf("KTR_HFSOSC_HessMult():: must be given Hessian-vector multiply function.\n");
		return;
	}
	
	if( data->x == NULL || data->lambda == NULL ) {
		printf("KTR_HFSOSC_HessMult():: not given valid current point and multipliers.\n");
		return;
	}
	
	// v <- H * v using Hessian-vector multiply function. This doesn't
	// restrict the Hessian to the active constraints, so we should make
	// sure that the lambda multipliers are zero for non-active constraints
	// (bounds are irrelevant if this function has been implemented
	// correctly)
	if( data->objGoal == KTR_OBJGOAL_MAXIMIZE ) {
		
		// for maximization, we must apply test to Hessian for 
		// -obj, not the given objective. The routine we are passed
		// does
		//
		//		(H obj)v  + sum_m lambda(m) (H c_m)v
		//
		// To get
		//
		//		(H -obj)v + sum_m lambda(m) (H c_m)v
		//
		// from this routine we use the equality
		//
		//		(H -obj)v + sum_m lambda(m) (H c_m)v
		//			= -(H obj)v + sum_m lambda(m) (H c_m)v
		//			= -(H obj)v - sum_m (-lambda(m)) (H c_m)v
		//			= - [ (H obj)v + sum_m (-lambda(m)) (H c_m)v ]
		//
		// the bracketed term here can be computed with the given
		// routine, after negating the multipliers. The result
		// must also be negated. 
		
		// negate multipliers
		cblas_dscal(data->N + data->M, -1.0, data->lambda, 1);
		
		// do multiply
		data->ktr_HessMult(KTR_RC_EVALHV,
						   data->N,
						   data->M,
						   0,
						   0,
						   data->x,
						   data->lambda,
						   NULL,
						   NULL,
						   NULL,
						   NULL,
						   NULL,
						   v,
						   data->userParams);
		
		// z <- - v to match structure of CHL Hmult calls
		cblas_dcopy(data->N, v, 1, z, 1);
		cblas_dscal(data->N, -1.0, z, 1);
		
	} else { // minimizing
		
		data->ktr_HessMult(KTR_RC_EVALHV,
						   data->N,
						   data->M,
						   0,
						   0,
						   data->x,
						   data->lambda,
						   NULL,
						   NULL,
						   NULL,
						   NULL,
						   NULL,
						   v,
						   data->userParams);
		
		// copy v into z to match structure of CHL Hmult calls
		cblas_dcopy(data->N, v, 1, z, 1);
		
	}
	
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * Wrapper for using directional finite differences in Hmult, as defined in 
 * cpdtests/defs.h
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void KTR_HFSOSC_DFDhess(int		N, 
						double* v, 
						double* z, 
						void*   params)
{
	KTR_HFSOSC_DATA* data;
	
	int i, j;
	
	if( v == NULL || z == NULL || params == NULL ) {
		printf("KTR_HFSOSC_DFDhess():: invalid argument(s).\n");
		return;
	}
	
	// cast params as a FDHESS data structure
	data = (KTR_HFSOSC_DATA*)params;
	
	if( data->N != N ) {
		printf("KTR_HFSOSC_DFDhess():: size passed in inconsistent with KTR_HFSOSC_DATA.\n");
		return;
	}
	
	if( data->type != KTR_SOSC_FINITE_DIFF || data->ktr_gradjac == NULL ) {
		printf("KTR_HFSOSC_DFDhess():: must be given gradients.\n");
		return;
	}
	
	if( data->x == NULL || data->lambda == NULL ) {
		printf("KTR_HFSOSC_DFDhess():: not given valid current point and multipliers.\n");
		return;
	}
	
	// default steplength
	if( data->steplength <= 0 ) { data->steplength = 1e-10; }
	
	// compute finite difference: 
	// 
	//		z <- g(x+sv) - g(x) + ( Dc_A(x+sv) - Dc_A(x) )' * lambda_A
	//		z <- z / s
	//
	
	// take step: newx <- x + steplength * v
	cblas_dcopy(data->N, data->x, 1, data->newx, 1);
	cblas_daxpy(data->N, data->steplength, v, 1, data->newx, 1);
	
	// compute *new* objective gradient and constraint Jacobian
	data->ktr_gradjac(KTR_RC_EVALGA,
					  data->N,
					  data->M,
					  data->Jnnz,
					  0,
					  data->newx,
					  data->lambda,
					  NULL,
					  NULL,
					  data->newobjGrad,
					  data->newJdata,
					  NULL,
					  NULL,
					  data->userParams);
	
	// compute difference of two Jacobians:
	//
	//		newJdata <- Dc(x+sv) - Dc(x)
	//
	cblas_daxpy(data->Jnnz, -1.0, data->Jdata, 1, data->newJdata, 1);
	
	// objective gradient component
	if( data->objGoal == KTR_OBJGOAL_MAXIMIZE ) { 
		
		// maximizing; have to use "-obj" in positive-definiteness test
		
		// z <- - newobjGrad + objGrad
		cblas_dcopy(data->N, data->objGrad, 1, z, 1);
		cblas_daxpy(data->N, -1.0, data->newobjGrad, 1, z, 1);
		
	} else { // minimizing
		
		// z <- newobjGrad - objgrad
		cblas_dcopy(data->N, data->newobjGrad, 1, z, 1);
		cblas_daxpy(data->N, -1.0, data->objGrad, 1, z, 1);
		
	}
	
	// constraint component: sum of rows of Dc_A(x+sv) - Dc_A(x) (transposed)
	// with coefficients from lambda_A
	if( data->account > 0 ) {
		
		if( data->rowmajorMMF ) {
			
			j = 0;
			for( i = 0 ; i < data->account ; i++ ) {
				
				// ignore linear constraints
				if( data->cType[ data->actCons[i] ] != KTR_CONTYPE_LINEAR ) {
					
					// z <- z + lambda[ actCons[i] ] * newJdata[ actCons[i] , : ]
					
					// find first element s such that Jrows[s] = actCons[i]
					// because Jacobian is stored in "row-major MMF", the
					// row indices are sequential:
					//
					//		0,...,0,1,...,1,2,... etc
					//
					// so we can start where we left off the last search
					while( j < data->Jnnz ) {
						if( data->Jrows[j] < data->actCons[i] ) { j++; }
						else { break; }
					}
					
					// now we have either Jrow[j] >= actCons[i] or j == Jnnz
					
					// if Jrow[j] == actCons[i] && j < Jnnz, we need to 
					// add in all the entries in the constraint 
					// gradient difference to the finite difference
					if( j < data->Jnnz ) {
						
						if( data->Jrows[j] == data->actCons[i] ) {
							
							// z[n] <- z[n] + lambda[ actCons[i] ] * newJdata[ actCons[i] , n ]
							while( j < data->Jnnz ) {
								if( data->Jrows[j] == data->actCons[i] ) { 
									z[data->Jcols[j]] 
										= z[data->Jcols[j]] 
												+ data->lambda[data->actCons[i]] * data->newJdata[j];
									j++;
								} else { break; }
							}
							
						} else {
							
							// Jrow[j] > actCons[i], and some active constraint
							// has no elements in the constraint Jacobian! This
							// seems a bad situation... in that it means some
							// constraint is literally a constant or the programs 
							// are not correct
							
							printf("WARNING:: Constraint %i/%i is said to be active, but has no Jacobian components.\n",data->actCons[i],data->M);
							
						}
						
					} else {
						
						// if j == Jnnz, then some active constraint has no
						// elements in the constraint Jacobian! This seems a 
						// bad situation... in that it means some constraint 
						// is literally a constant or the programs are not
						// correct
						
						printf("WARNING:: Constraint %i/%i is said to be active, but has no Jacobian components.\n",data->actCons[i],data->M);
						
					}
					
				}
				
			}
			
		} else {
			
			// ordering not known, or not useful. Here we should probably 
			// traverse the Jacobian data first. 
			for( i = 0 ; i < data->Jnnz ; i++ ) {
				
				// if Jdata[i] != 0 and the corresponding row is not a linear constraint... 
				if( data->Jdata[i] != 0.0 && data->cType[data->Jrows[i]] != KTR_CONTYPE_LINEAR ) {
					
					for( j = 0 ; j < data->account ; j++ ) {
						
						// and if the corresponding row is active, 
						if( data->actCons[j] == data->Jrows[i] ) {
							
							// add this component into z
							z[data->Jcols[i]] = z[data->Jcols[i]] + data->lambda[data->actCons[j]] * data->newJdata[i];
							break;
							
						}
					}
					
				}
				
			}
			
		}
		
	}
	
	// scale z <- z / steplength
	cblas_dscal(data->N, 1.0/(data->steplength), z, 1);
	
}

// identify active bounds
int ktr_sosc_getActBnds(int			N,
						double*		x,
						double*		lambda, // just x multipliers
						double*		xLoBnds,
						double*		xUpBnds,
						int*		abcount,
						int**		actBnds,
						double		abtol,
						double		sctol)
{
	int n;
	int flag = 1;
	
	if( *actBnds != NULL ) {
		printf("WARNING:: active bounds list is not NULL. Allocating memory may cause a memory leak.\n");
	}
	
	*actBnds = (int*)calloc(N, sizeof(int));
	
	*abcount = 0;
	for( n = 0 ; n < N ; n++ ) {
		
		// printf("  ActBnds:: %i, [%e,%e,%e]\n", n, xLoBnds[n], x[n], xUpBnds[n]);
		
		// ignore if x[n] is unbounded
		if( xLoBnds[n] > - KTR_INFBOUND || xUpBnds[n] < KTR_INFBOUND ) {
			
			// some bound is finite
			if( xLoBnds[n] > - KTR_INFBOUND ) {
				
				// if the lower bound constraint is active...
				if( fabs( x[n] - xLoBnds[n] ) <= abtol ) {
					
					// add to active bounds list
					(*actBnds)[*abcount] = n;
					(*abcount)++;
					
					// check complementarity
					if( fabs( lambda[n] ) <=  sctol ) {
						printf("Trial point is not (numerically) strictly complementary.\n");
						printf("Test is not currently applicable at such points.\n");
						flag = 0;
						break;
					}
					
				} else {
					
					// if there is an upper bound...
					if( xUpBnds[n] < KTR_INFBOUND ) {
						
						// and it is active...
						if( fabs( x[n] - xUpBnds[n] ) <= abtol ) {
							
							// add to active bounds list
							(*actBnds)[*abcount] = n;
							(*abcount)++;
							
							// and verify strict complementarity
							if( fabs( lambda[n] ) <=  sctol ) {
								printf("Trial point is not (numerically) strictly complementary.\n");
								printf("Test is not currently applicable at such points.\n");
								flag = 0;
								break;
							}
							
						}
						
					}
					
				}
				
			} else { // xUpBnds[n] < KTR_INFBOUND must hold
				
				// if the upper bound constraint is active...
				if( fabs( x[n] - xUpBnds[n] ) <= abtol ) {
					
					// add to active bounds list
					(*actBnds)[*abcount] = n;
					(*abcount)++;
					
					// check complementarity
					if( fabs( lambda[n] ) <=  sctol ) {
						printf("Trial point is not (numerically) strictly complementary.\n");
						printf("Test is not currently applicable at such points.\n");
						flag = 0;
						break;
					}
					
				}
				
			}
			
		}
		
	}
	
	// resize arrays, or delete
	if( abcount > 0 ) { 
		*actBnds = (int*)realloc(*actBnds, (*abcount)*sizeof(int));
	} else { 
		free(*actBnds); 
		*actBnds = NULL;
	}
	
	return flag;
}

// identify active constraints
int ktr_sosc_getActCons(int			M,
						double*		cns,
						double*		lambda, // just x multipliers
						double*		cLoBnds,
						double*		cUpBnds,
						int*		account,
						int**		actCons,
						double		actol,
						double		sctol)
{
	int m;
	int flag = 1;
	
	if( M <= 0 ) { return 0; }
	
	if( *actCons != NULL ) {
		printf("WARNING:: active constraints list is not NULL. Allocating memory may cause a memory leak.\n");
	}
	
	*actCons = (int*)calloc(M, sizeof(int));
	
	*account = 0;
	for( m = 0 ; m < M ; m++ ) {
		
		// ignore if constraint is unbounded (this should not occur)
		if( cLoBnds[m] > - KTR_INFBOUND || cUpBnds[m] < KTR_INFBOUND ) {
			
			// automatically include if equality constraint
			if( cLoBnds[m] == cUpBnds[m] ) {
				
				// add to index list
				(*actCons)[*account] = m;
				(*account)++;
				
				// check complementarity
				if( fabs( lambda[m] ) <= sctol ) {
					printf("Trial point is not (numerically) strictly complementary.\n");
					printf("Test is not currently applicable at such points.\n");
					flag = 0;
					break;
				}
				
			} else { // inequality constraint, some bound (possibly both) is finite
				
				// if lower constraint bound is finite ...
				if( cLoBnds[m] > - KTR_INFBOUND ) {
					
					// and if the lower bound constraint is active...
					if( fabs( cns[m] - cLoBnds[m] ) <= actol ) {
						
						// add to index list
						(*actCons)[*account] = m;
						(*account)++;
						
						// check complementarity
						if( fabs( lambda[m] ) <= sctol ) {
							printf("Trial point is not (numerically) strictly complementary.\n");
							printf("Test is not currently applicable at such points.\n");
							flag = 0;
							break;
						}
						
					} else {
						
						// if upper bound is finite...
						if( cUpBnds[m] < KTR_INFBOUND ) {
							
							// and active...
							if( fabs( cns[m] - cUpBnds[m] ) <= actol ) {
								
								// add to index list
								(*actCons)[*account] = m;
								(*account)++;
								
								// check complementarity
								if( fabs( lambda[m] ) <= sctol ) {
									printf("Trial point is not (numerically) strictly complementary.\n");
									printf("Test is not currently applicable at such points.\n");
									flag = 0;
									break;
								}
								
							} 
							
						}
						
					}
					
				} else { // (cLoBnds[m],cUpBnds[m]) = (-KTR_INFBOUND,cUpBnds[m]) < (-KTR_INFBOUND,KTR_INFBOUND)
					
					// if the upper bound constraint is active...
					if( fabs( cns[m] - cUpBnds[m] ) <= actol ) {
						
						// add to index list
						(*actCons)[*account] = m;
						(*account)++;
						
						// check complementarity
						if( fabs( lambda[m] ) <= sctol ) {
							printf("Trial point is not (numerically) strictly complementary.\n");
							printf("Test is not currently applicable at such points.\n");
							flag = 0;
							break;
						}
						
					} 
					
				}
				
			}
			
		}
		
	}
	
	// resize array, or delete
	if( *account > 0 ) { 
		*actCons = (int*)realloc(*actCons, (*account)*sizeof(int));
	} else { 
		free(*actCons); 
		*actCons = NULL; 
	}
	
	return flag;
}

// compute active constraint Jacobian
int ktr_sosc_getActConsJac(int		Jnnz,
						   int*		Jrows,
						   int*		Jcols,
						   double*  Jdata,
						   int		rowmajorMMF,
						   int		abcount, // number of active bounds
						   int*		actBnds, // active bounds
						   int		account, // number of active constraints
						   int*		actCons, // active constraints
						   int*		Annz,
						   int**	Arows,
						   int**	Acols,
						   double** Adata)
{
	int i, j;
	int flag = 1;
	int newAnnz;
	
	// we now have lists of active bounds and constraints; form *active*
	// constraint Jacobian (in sparse format) including bounds. Note that 
	// signs on the rows are irrelevant in determining the null space of
	// the active constraint Jacobian. Our format is: 
	//
	//		[ active bound constraints ]
	//		[ active constraints	   ]
	// 
	// there will be *one* element for each active bound, but we don't yet
	// know the number of elements in the active inequality/equality constraint 
	// Jacobian because it may be a submatrix of the whole Jacobian. We can 
	// allocate enough space first and resize after. 
	
	// this is certainly enough space...
	*Annz  = abcount + Jnnz;
	*Arows =    (int*)calloc(*Annz, sizeof(int));
	*Acols =    (int*)calloc(*Annz, sizeof(int));
	*Adata = (double*)calloc(*Annz, sizeof(double));
	if( *Arows == NULL || *Acols == NULL || *Adata == NULL ) {
		printf("memory allocation failure.\n");
		return 0;
	}
	
	// insert ones for each active bound (sign irrelevant in the A matrix)
	for( i = 0 ; i < abcount ; i++ ) {
		
		// A[ i , actBnd[i] ] = 1.0
		(*Arows)[i] = i;
		(*Acols)[i] = actBnds[i];
		(*Adata)[i] = 1.0;
		
	}
	
	// re-set Annz to be accurate... will be incremented below because
	// this is used as a counter
	newAnnz = abcount;
	
	// insert rows of the constraint Jacobian corresponding to the active 
	// constraints. We could save time if we knew a priori something about the
	// entry ordering in Jacobian data, particularly if it is "row-major MMF". 
	if( rowmajorMMF ) {
		
		// traverse active constraints first
		j = 0;
		for( i = 0 ; i < account ; i++ ) {
			
			// find first index j with Jrow[j] >= actCons[i]
			while( j < Jnnz ) {
				if( Jrows[j] < actCons[i] ) { j++; }
				else { break; }
			}
			
			// now Jrow[j] >= actCons[i], or j == Jnnz
			if( j < Jnnz ) {
				
				if( Jrows[j] == actCons[i] ) {
					
					// add in all entries with Jrow[j] = actCons[i]
					while( j < Jnnz ) {
						if( Jrows[j] == actCons[i] ) {
							(*Arows)[newAnnz] = Jrows[j] + abcount;
							(*Acols)[newAnnz] = Jcols[j];
							(*Adata)[newAnnz] = Jdata[j];
							newAnnz++;
							j++;
						} else { break; }
					}
					
					// now Jrow[j] != actCons[i]; if Jrow[j] < actCons[i]
					// the matrix is not, in fact, rowmajorMMF
					
				} else {
					
					// some active constraint does not have any 
					// Jacobian entries! This would result in 
					// a singular active constraint Jacobian
					printf("WARNING:: Constraint %i is said to be active, but has no Jacobian components.\n",actCons[i]);
					
				}
				
			} else {
				
				// some active constraint does not have any 
				// Jacobian entries! This would result in 
				// a singular active constraint Jacobian
				printf("WARNING:: Constraint %i is said to be active, but has no Jacobian components.\n",actCons[i]);
				
			}
			
		}
		
	} else {
		
		// traverse Jacobian data first
		for( i = 0 ; i < Jnnz ; i++ ) {
			
			// if Jrows[i] is in actCons, add the appropriate row, column, and data
			// to A. 
			for( j = 0 ; j < account ; j++ ) {
				
				if( Jrows[i] == actCons[j] ) {
					
					(*Arows)[newAnnz] = Jrows[i] + abcount;
					(*Acols)[newAnnz] = Jcols[i];
					(*Adata)[newAnnz] = Jdata[i];
					newAnnz++;
					break;
					
				}
				
			}
			
		}
		
	}
	
	// resize Arows, Acols, Adata if needed
	if( newAnnz < *Annz ) {
		
		// reduce size
		*Arows =    (int*)realloc(*Arows, newAnnz*sizeof(int));
		*Acols =    (int*)realloc(*Acols, newAnnz*sizeof(int));
		*Adata = (double*)realloc(*Adata, newAnnz*sizeof(double));
		
		// reset Annz for use below
		*Annz = newAnnz;
	}
	
	return flag;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * KNITRO version of the sparse CPD-SOSC check
 *	
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * This algorithm is faster if the constraint Jacobian is stored in "row-
 * major MMF" format, by which we mean that the elements of the Jacobian
 * in the MMF data structure are stored with successive blocks of entries
 * corresponding to rows of the Jacobian, in order of increasing row
 * index. That is, 
 * 
 *		Jrows ~ [ 0 , ... , 0 , 1 , ... , 1 , 2 , ... , 2 , ... ]
 * 
 * the column ordering is irrelevant. 
 * 
 * This code does not have the capacity to sort the Jacobian entries 
 * to match this pattern given any ordering, because that would only be 
 * marginally useful: if an explicit Hessian or a Hessian-multiply function
 * were used, this sorting would only be used in construction of the 
 * active constraint Jacobian, and it is not clear that pre-sorting 
 * (relatively) unordered Jacobian entries would ultimately speed up this
 * task. If only the gradient and Jacobian are provided, this sorting
 * would have to be done for every function evaluation and that would
 * certainly be time consuming. Thus, the user should implement the 
 * Jacobian evaluation function to return results in "row-major MMF" if
 * they wish the code to be as fast as possible. (The speed-up is not 
 * likely to be large relative to the test itself.)
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * info[0] = detailed termination flag
 * info[1] = number of active bounds
 * info[2] = number of active constraints
 * info[3] = rank estimate of the constraint Jacobian (from SPQR if "test"
 *			 is SOSC_CHL or SOSC_DIA and from MA57 if "test" is SOSC_MIT)
 * info[4] = if "test" is SOSC_MIT, number of negative eigenvalues of KKT 
 *			 matrix, otherwise not used
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
int ktr_sosc_check(int		N,			// number of variables
				   int		M,			// number of constraints
				   int		ktrObjGoal, // MAXIMIZE or MINIMIZE
				   double*  x,			// trial point (ostensibly a stationary point)
				   double*	xLoBnds,	// lower bounds on x
				   double*	xUpBnds,	// upper bounds on x
				   double*  lambda,		// trial multipliers (ostensibly a stationary point)
				   double*  obj,		// current objective value (optional; can be NULL)
				   int*		cType,		// constraint types (LINEAR, QUADRATIC, or GENERAL)
				   double*	cLoBnds,	// lower bounds on constraints
				   double*	cUpBnds,	// upper bounds on constraints
				   double*	cns,		// current constraint values, if available (can be NULL)
				   double*  objGrad,	// current objective gradient, if available (can be NULL)
				   int		Jnnz,		// number of nonzeros in constraint Jacobian
				   int*		Jrows,		// constraint Jacobian structure, rows
				   int*		Jcols,		// constraint Jacobian structure, cols
				   double*	Jdata,		// current constraint Jacobian values, if available (can be NULL)
				   int		rowmajorMMF,// flag identifying whether Jacobian is stored "row-major MMF" (likely to speed up computations)
				   int		Hnnz,		// number of nonzeros in Hessian of the Lagrangian (can be zero)
				   int*		Hrows,		// Hessian of the Lagrangian structure, rows (can be NULL)
				   int*		Hcols,		// Hessian of the Lagrangian structure, cols (can be NULL)
				   double*	Hdata,		// current Hessian of the Lagrangian values, if available (can be NULL)
				   KTR_callback* ktr_fc,// KTR_callback style function that can evaluate constraint values
				   KTR_callback* ktr_gj,// KTR_callback style function that can evaluate constraint Jacobian values
				   KTR_callback* ktr_eh,// KTR_callback style function that can evaluate Hessian matrix (can be NULL)
				   KTR_callback* ktr_hm,// KTR_callback style function that can evaluate Hessian-vector multiplies (can be NULL)
				   void*	userParams, // user parameters that should be passed to *any* KNITRO callback functions
				   int		CPDT_method,// method to use (CPDT package flag)
				   int*		info)		// information about SOSC check
{
	// count and indices of active bounds
	int  abcount = 0;
	int* actBnds = NULL;
	
	// count and indices of active constraints
	int  account = 0;
	int* actCons = NULL;
	
	double abtol = 1.0e-10; // active bound tolerance
	double actol = 1.0e-10; // active constraint tolerance
	double sctol = 1.0e-10; // strict complementarity tolerance
	
	// "free" flags for memory that might be allocated
	// here for pointers passed in
	int freecns		= 0;
	int freejac		= 0;
	int freegrad	= 0;
	int freehessian = 0;
	
	// number of active constraints (including bounds)
	int MA;
	
	// active constraint Jacobian
	int		Annz  = 0; 
	int*	Arows = NULL;
	int*	Acols = NULL;
	double* Adata = NULL;
	
	// required for Hessian-free implementation
	int HessType;
	KTR_HFSOSC_DATA data;
	
	CPDT_ContextPtr c;
	
	// test result:
	//
	//		1 == success
	//		0 == no information
	//		< 0  failure (of some sort)
	//
	int flag = 1;
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * VALIDATE ARGUMENTS  * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// must be given valid problem sizes
	if( N <= 0 || M < 0 ) { 
		printf("Invalid problem size.\n");
		return 0;
	}
	
	// must be given current point, and if there are constraints, must be 
	// given multipliers
	if( x == NULL || ( M > 0 && lambda == NULL ) ) { 
		printf("Must have current point and, if constrained, multipliers.\n");
		return 0;
	}
	
	// must be given either constraint data or constraint evaluation function, 
	// if there are constraints
	if( M > 0 && cns == NULL && ktr_fc == NULL ) {
		printf("Must be given either current constraint values or a function to evaluate them.\n");
		return 0;
	}
	
	// must be given constraint Jacobian structure, if there are constraints
	if( M > 0 && ( Jnnz <= 0 || Jrows == NULL || Jcols == NULL ) ) {
		printf("Must be given constraint Jacobian structure.\n");
		return 0;
	}
	
	// must be given either constraint Jacobian data or the evaluation function
	if( M > 0 && Jdata == NULL && ktr_gj == NULL ) {
		printf("Must be given either constraint Jacobian values or a function to evaluate them.\n");
		return 0;
	}
	
	// if we're not given an exact Hessian, we must have either a Hessian
	// multiply function or gradient/Jacobian function
	if( ktr_eh == NULL && Hdata == NULL && ktr_hm == NULL && ktr_gj == NULL ) {
		printf("Must be given exact Hessian, Hessian-vector multiply function, or gradient/Jacobian evaluation function.\n");
		return 0;
	}
	
	// if we're given an exact Hessian, must have structure. data is optional. 
	if( ( ktr_eh != NULL || Hdata != NULL ) && ( Hnnz <= 0 || Hrows == NULL || Hcols == NULL ) ) {
		printf("To use exact Hessian, must be given Hessian structure.\n");
		return 0;
	} else { 
		// declare use of Hessian for below
		HessType = KTR_SOSC_EXACT_HESSIAN;
	}
	
	// finish with identification of hessian type
	if( ktr_hm != NULL ) { HessType = KTR_SOSC_HESSIAN_MULTIPLY; }
	else { HessType = KTR_SOSC_FINITE_DIFF; }
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	printf("KTR SOSC - %i Variables and %i Constraints\n",N,M);
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * EVALUATE WHICH BOUNDS ARE ACTIVE AND CHECK STRICT COMPLEMENTARITY * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	flag = ktr_sosc_getActBnds(N, 
							   x, 
							   lambda+M, 
							   xLoBnds, 
							   xUpBnds, 
							   &abcount, // pointer so that data can be returned
							   &actBnds, // pointer so that data can be returned
							   abtol,
							   sctol);
	
	if( info != NULL ) { info[1] = abcount; }
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * EVALUATE WHICH CONSTRAINTS ARE ACTIVE * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	if( flag > 0 ) {
		
		// make sure we have current constraint values
		if( cns == NULL && M > 0 ) {
			
			if( obj == NULL ) { 
				
				// allocate space for current constraint values
				// + 1 so we can store objective value as well, 
				// as required by KNITRO-style callback routine
				cns = (double*)calloc(M+1, sizeof(double));
				if( cns == NULL ) {
					
					printf("memory allocation failure.\n");
					return 0;
					
				} else {
					
					// get current constraint values (and objective)
					ktr_fc(KTR_RC_EVALFC,
								N,
								M,
								0,
								0,
								x,
								lambda,
								cns+M, // (objective stored at the end of cns array)
								cns,
								NULL,
								NULL, 
								NULL,
								NULL,
								userParams);
					
					// declare that we need to delete cns later
					freecns = 1;
					
				}
				
			} else {
				
				// allocate space for current constraint values
				cns = (double*)calloc(M, sizeof(double));
				if( cns == NULL ) {
					
					printf("memory allocation failure.\n");
					flag = 0;
					
				} else {
					
					// get current constraint values (and objective)
					ktr_fc(KTR_RC_EVALFC,
								N,
								M,
								0,
								0,
								x,
								lambda,
								obj,
								cns,
								NULL,
								NULL, 
								NULL,
								NULL,
								userParams);
					
					// declare that we need to delete cns later
					freecns = 1;
					
				}
				
			}
			
		}
		
		// get active constraints list
		flag = ktr_sosc_getActCons(M, 
								   cns,
								   lambda,
								   cLoBnds, 
								   cUpBnds,
								   &account,
								   &actCons,
								   actol,
								   sctol);
		
		if( info != NULL ) { info[2] = account; }
		
		// we won't need the current constraint values any more
		if( freecns ) { free( cns ); cns = NULL; } 
		
	}
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * GET TOTAL NUMBER OF ACTIVE BOUNDS AND CONSTRAINTS * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	MA = abcount + account;
	
	printf("           %i Active Bounds and %i Active Constraints\n",abcount,account);
	
	if( MA > N ) {
		printf("KTR SOSC ERROR:: Too many active constraints (%i) for number of variables (%i).\n",MA,N);
		flag = -1;
	}
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * CHECK SOSC  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	if( flag > 0 ) {
		
		if( MA == N ) {
			
			// if there are as many active constraints (including bounds) as
			// variables, and the current KKT point is strictly complementary, 
			// then this "fully constrained" case breaks down to verifying 
			// multiplier signs. Trusting KNITRO to terminate only with
			// multipliers having the right signs, we can ignore this check. 
			// However, it is easy to do, so possibly worth checking. 
			
			printf("           KKT Point is fully constrained\n");
			
		} else { // 0 <= MA < N: use CPDT library routines
			
			printf("           N = %i < %i = MA; Calling CPDT Routines\n",N,MA);
			
			/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
			 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
			 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
			
			// define new CPDT context for active constraints
			c = CPDT_new_context( N , MA );
			
			if( c == NULL ) {
				printf("KTR SOSC - ERROR; could not initialize CPDT.\n");
			}
			
			/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
			 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
			 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
			
			// define Constraints Matrix for CPDT
			// this is the active constraint Jacobian (including active bounds)
			
			// make sure we have constraint Jacobian
			if( Jdata == NULL ) {
				
				// if no valid objective gradient pointer has been passed
				if( objGrad == NULL ) {
					
					// allocate space; + N because we need to be able to
					// store the objective gradient as well, as KNITRO
					// callback will assume we can do that. 
					Jdata   = (double*)calloc(Jnnz, sizeof(double));
					objGrad = (double*)calloc(N,    sizeof(double));
					if( Jdata == NULL || objGrad == NULL ) {
						
						printf("memory allocation failure.\n");
						flag = 0;
						
					} else { 
						
						// evaluate Jacobian (and objective gradient)
						ktr_gj(KTR_RC_EVALGA,
							   N,
							   M,
							   Jnnz,
							   0,
							   x,
							   lambda,
							   NULL,
							   NULL,
							   objGrad, // objective gradient
							   Jdata, // constraint Jacobian
							   NULL,
							   NULL,
							   userParams);
						
						// identify that we need to free this space later
						freegrad = 1;
						freejac  = 1;
						
					}
					
				} else {
					
					// allocate space for constraint Jacobian only
					Jdata = (double*)calloc(Jnnz, sizeof(double));
					if( Jdata == NULL ) {
						
						printf("memory allocation failure.\n");
						flag = 0;
						
					} else {
						
						// evaluate Jacobian (and objective gradient)
						ktr_gj(KTR_RC_EVALGA,
							   N,
							   M,
							   Jnnz,
							   0,
							   x,
							   lambda,
							   NULL,
							   NULL,
							   objGrad, // objective gradient
							   Jdata, // constraint Jacobian
							   NULL,
							   NULL,
							   userParams);
						
						// identify that we need to free this space later
						freejac  = 1;
						
					}
					
				}
				
			} else { // Jdata != NULL
				
				// if Jdata was provided, but objGrad was not, - and - we are using
				// finite-difference Hessians, then we have to compute gradients 
				// anyway. Using an exact Hessian or Hessian-vector multiply function
				// does not require current objective gradient. 
				if( HessType == KTR_SOSC_FINITE_DIFF && objGrad == NULL ) {
					
					// allocate space for objective gradient only
					objGrad = (double*)calloc( N , sizeof(double));
					if( objGrad == NULL ) {
						
						printf("memory allocation failure.\n");
						flag = 0;
						
					} else {
						
						// evaluate Jacobian (and objective gradient)
						ktr_gj(KTR_RC_EVALGA,
							   N,
							   M,
							   Jnnz,
							   0,
							   x,
							   lambda,
							   NULL,
							   NULL,
							   objGrad, // objective gradient
							   Jdata, // constraint Jacobian
							   NULL,
							   NULL,
							   userParams);
						
						// identify that we need to free this space later
						freegrad = 1;
						
					}
					
				}
				
			}
			
			printf("           Constructing Active Constraint Jacobian\n");
			
			// construct active constraint Jacobian (constraints
			// and bounds)
			flag = ktr_sosc_getActConsJac(Jnnz,
										  Jrows,
										  Jcols,
										  Jdata,
										  rowmajorMMF,
										  abcount,
										  actBnds,
										  account,
										  actCons,
										  &Annz,
										  &Arows,
										  &Acols,
										  &Adata);
			
			// assign constraint matrix to the CPDT context
			CPDT_setSparseConstraintMatrix_MMF(c ,
											   MA , 
											   N , 
											   Annz , 
											   Arows , 
											   Acols , 
											   Adata , 
											   CPDT_INDEX_STYLE_C );
			
			/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
			 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
			 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
			
			// define Hessian matrix for CPDT: 
			//  if exact hessian is given, that is used
			//  if a hessian multiply is given, that is used
			//  otherwise, finite differences are used
			switch( HessType ) {
					
				case KTR_SOSC_EXACT_HESSIAN:
					
					// use exact hessian provided; Hnnz, Hrows, and Hcols are valid. 
					
					// first, ensure sure we have the Hessian data
					if( Hdata == NULL ) {
						
						Hdata = (double*)calloc(Hnnz, sizeof(double));
						if( Hdata == NULL ) {
							
							printf("memory allocation failure.\n");
							flag = 0;
							
						} else { 
							
							// get current Hessian data
							ktr_eh(KTR_RC_EVALH,
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
								   Hdata,
								   NULL,
								   userParams);
							
							freehessian = 1;
							
						}
						
					}
					
					// assign Hessian matrix to the CPDT context
					CPDT_setSparseHessian_MMF(c ,
											  N , 
											  Hnnz , 
											  Hrows , 
											  Hcols , 
											  Hdata ,
											  CPDT_INDEX_STYLE_C );
					
					break;
				
				case KTR_SOSC_HESSIAN_MULTIPLY:
					
					// use Hessian-multiply function provided. To do this, 
					// we use the KTR_SOSC_DATA structure created above
					data.type		  = KTR_SOSC_HESSIAN_MULTIPLY;
					
					data.N			  = N;
					data.M			  = M;			 // - not - MA; this is passed to ktr_hm function
					data.x			  = x;
					data.lambda		  = lambda;
					data.ktr_gradjac  = NULL;		 // even if provided, we ignore
					data.ktr_HessMult = ktr_hm;
					data.userParams	  = userParams;
					
					data.objGoal	  = ktrObjGoal;  // objective type (maximize or minimize)
					
					// assign hessian multiply function (and parameters) to the CPDT context
					CPDT_setHessMultFcn(c , 
										&KTR_HFSOSC_HessMult , 
										(void*)(&data) );
					
					break;
					
				default:
					
					// directional finite differences using given objective
					// gradient and constraint Jacobian routine
					
					// use directional finite differences: 
					// 
					//		Hv ~ (1/s) * [ g(x+sv) - g(x)
					//							( Dc_A(x+sv) - Dc_A(x) )' * lambda_A ]
					// 
					// where Dc_A and lambda_A denote the Jacobian of the active
					// constraints (including bounds) and the corresponding multipliers
					// if the multipliers for all inactive constraints are zero (as
					// they should be) then we don't have to consider activity. 
					
					// To do this, we use the KTR_SOSC_DATA structure created 
					// above but have to store some extra data
					data.type		  = KTR_SOSC_FINITE_DIFF;
					
					data.N			  = N;
					data.M			  = M;			// *not* MA
					data.x			  = x;			// current point
					data.lambda		  = lambda;		// multipliers
					data.ktr_gradjac  = ktr_gj;		// gradient evaluation function
					data.ktr_HessMult = NULL;		// not provided; otherwise, would use
					data.userParams   = userParams;	// always passed
					
					data.objGoal	  = ktrObjGoal; // objective type (maximize or minimize)
					
					// objective gradient
					data.objGrad	  = objGrad;	// - current - objective gradient
					
					// constraint Jacobian
					data.Jnnz		  = Jnnz;		// number of nonzeros in constraint Jacobian
					data.Jrows		  = Jrows;		// constraint Jacobian structure, row indices
					data.Jcols		  = Jcols;		// constraint Jacobian structure, column indices
					data.Jdata		  = Jdata;		// - current - constraint Jacobian
					
					data.rowmajorMMF  = rowmajorMMF;// flag identifying whether Jacobian is stored "row-major MMF"
					
					// active constraints (bounds irrelevant in Hessian, as are linear constraints)
					data.account	  = account;	// number of active constraints
					data.cType		  = cType;		// constraint types
					data.actCons	  = actCons;	// indices of active constraints
					
					// finite difference steplength
					data.steplength   = 1e-6;
					
					// storage - we allocate here and store in the data structure 
					// so that we don't have to allocate and free in every evaluation
					// of the objective gradient and constraint Jacobian
					data.newx		  = (double*)calloc(N, sizeof(double)); // new point
					data.newobjGrad   = (double*)calloc(N, sizeof(double)); // new objective gradient
					data.newJdata	  = (double*)calloc(Jnnz, sizeof(double)); // new constraint Jacobian
					
					// assign hessian multiply function (and parameters) to the CPDT context
					CPDT_setHessMultFcn(c , 
										&KTR_HFSOSC_DFDhess , 
										(void*)(&data) );
				
					break;
				
			}
			
			/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
			 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
			 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
			
			printf("           Calling CPD Test\n");
			
			// set any options; particularly the test method
			CPDT_setTestMethod( c , CPDT_method );
			
			// test for constrained positive definiteness
			flag = CPDT_test( c );
			
			// get information about the check
			
			info[3] = CPDT_getConstraintMatrixRank( c );
			
			if( CPDT_method == CPDT_METHOD_MIT ) {
				info[4] = CPDT_getKKTNumNegEvals( c );
			}
			
			/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
			 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
			 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
			
			// free memory allocated above (inside this conditional only)
			
			if( freecns     ) { free( cns     ); }
			if( freegrad    ) { free( objGrad ); }
			if( freejac     ) { free( Jdata   
									 ); }
			switch( HessType ) {
					
				case KTR_SOSC_EXACT_HESSIAN:
					if( freehessian ) { free( Hdata ); }
					break;
					
				case KTR_SOSC_HESSIAN_MULTIPLY:
					break;
					
				default:
					free( data.newx );
					free( data.newobjGrad );
					free( data.newJdata );
					break;
					
			}
			
			// release CPDT context
			CPDT_free_context( &c );
			
			/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
			 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
			 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
			
		}
		
	}
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	if( actBnds != NULL ) { free( actBnds ); }
	if( actCons != NULL ) { free( actCons ); }
	
	if( Arows != NULL ) { free( Arows ); }
	if( Acols != NULL ) { free( Acols ); }
	if( Adata != NULL ) { free( Adata ); }
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	return flag;
	
}
/*
 *  ktrsosc.h
 *  
 *
 *  Created by W. Ross Morrow on 10/12/11.
 *  Copyright 2011 Iowa State University. All rights reserved.
 *
 */

#ifndef _KTRSOSC_H
#define _KTRSOSC_H

#include <knitro.h>

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * KNITRO version of the sparse, Hessian-free Implicit Cholesky check
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
int ktr_sosc(int		N,			// number of variables
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
			   KTR_callback* ktr_objcons, // KTR_callback style function that can evaluate constraint values
			   KTR_callback* ktr_gradjac, // KTR_callback style function that can evaluate constraint Jacobian values
			   KTR_callback* ktr_Hessian, // KTR_callback style function that can evaluate Hessian matrix (can be NULL)
			   KTR_callback* ktr_HessMult,// KTR_callback style function that can evaluate Hessian-vector multiplies (can be NULL)
			   void*	userParams, // user parameters that should be passed to *any* KNITRO callback functions
			   int		CPDT_method,		// test type
			   int*		info);		// detailed information about SOSC check (can be NULL)

#endif
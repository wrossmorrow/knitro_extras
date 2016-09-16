/*
 *  ktrextras.h
 *  
 *
 *  Created by W. Ross Morrow on 12/1/11.
 *  Copyright 2011 Iowa State University. All rights reserved.
 *
 */

#ifndef _KTR_EXTRAS_H
#define _KTR_EXTRAS_H

// #include <cpdt.h>
#include <knitro.h>

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Check first derivatives for a KNITRO callback routine
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void ktr_evalga_check(int			N, // number of variables
					  int			M, // number of constraints
					  KTR_callback  func_call, // callback to evaluate objective and constraint values
					  KTR_callback  grad_call, // callback routine to evaluate objective gradient and constraint Jacobian
					  int			Jnnz, // (constant) constraint Jacobian number of nonzeros
					  int*			Jrows, // (constant) constraint Jacobian structure - row indices
					  int*			Jcols, // (constant) constraint Jacobian structure - column indices
					  double*		x,		// point at which we want to check first derivatives
					  double*		xLoBnds,  // lower bounds on variables
					  double*		xUpBnds,  // upper bounds on variables
					  void*			userParams, // user parameters
					  double*		info); 

// "long" version: steps and finite differences computed using
// extended precision arithmetic (long doubles)
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
						double*			info); 

// Hessian check routine
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
						double*		info);		// information (can be NULL)



#endif
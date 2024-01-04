#ifndef _EIGEN_RIDGE_HPP_

#define _EIGEN_RIDGE_HPP_

/*
 * L2-regularized (ridge) linear regression without intercept in c++11 as Eigen3 template function
 * Uses singular value decomposition, works with both tall and wide design matrices
 * Written by Carlos Guerreiro carlos@perceptiveconstructs.com
 * This is free and unencumbered software released into the public domain.
 */

#include <Eigen/Dense>

template<typename M, typename V, typename P>
M ridge(const M& A, const V& y,  P alpha) {
  const auto& svd = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto& s = svd.singularValues();
  const auto r = s.rows();
  const auto& D = s.cwiseQuotient((s.array().square() + alpha).matrix()).asDiagonal();
  return svd.matrixV().leftCols(r) * D * svd.matrixU().transpose().topRows(r) * y;
}

#endif
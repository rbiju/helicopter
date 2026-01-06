#pragma once
#include <Eigen/Dense>
#include <Eigen/SVD>

typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> MatrixX3fRow;

class KabschSolver {
public:
    static std::pair<Eigen::Matrix3f, Eigen::Vector3f> kabsch(const MatrixX3fRow& q, const MatrixX3fRow& p) {
        Eigen::Vector3f q_mean = q.colwise().mean();
        Eigen::Vector3f p_mean = p.colwise().mean();

        const MatrixX3fRow q_c = q.rowwise() - q_mean.transpose();
        const MatrixX3fRow p_c = p.rowwise() - p_mean.transpose();

        const Eigen::Matrix3f covar = p_c.transpose() * q_c;

        const Eigen::JacobiSVD svd(covar, Eigen::ComputeFullU | Eigen::ComputeFullV);

        const Eigen::Matrix3f& U = svd.matrixU();
        const Eigen::Matrix3f& V = svd.matrixV();

        Eigen::Matrix3f R = V * U.transpose();

        if (R.determinant() < 0) {
            Eigen::Matrix3f V_corr = V;
            V_corr.col(2) *= -1;
            R = V_corr * U.transpose();
        }

        Eigen::Vector3f t = q_mean - R * p_mean;

        return {R, t};
    }

    static MatrixX3fRow apply(const MatrixX3fRow& points, const Eigen::Matrix3f& R, const Eigen::Vector3f& t) {
        MatrixX3fRow points_transformed = (points * R.transpose()).rowwise() + t.transpose();
        return points_transformed;
    }
};
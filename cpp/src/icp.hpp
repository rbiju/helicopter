#pragma once
#include <vector>
#include <Eigen/Dense>

#include "kabsch.hpp"

typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> MatrixX3fRow;

class ICP {
public:
    int max_iter;
    float max_error_sq;
    float etol;

    ICP(const int max_iter, const float distance_threshold, const float etol)
        : max_iter(max_iter), etol(etol) {
        max_error_sq = distance_threshold * distance_threshold;
    }

    [[nodiscard]] std::pair<std::vector<int>, std::vector<int>> get_correspondences(
        const MatrixX3fRow& ref_points,
        const MatrixX3fRow& sample_points) const {

        std::vector<int> ref_idxs;
        std::vector<int> sample_idxs;

        for (int i = 0; i < sample_points.rows(); i++) {
            float min_dist = std::numeric_limits<float>::max();
            int closest_idx = -1;

            for (int j = 0; j < ref_points.rows(); j++) {
                if (const float dist = (ref_points.row(j) - sample_points.row(i)).squaredNorm(); dist < min_dist) {
                    min_dist = dist;
                    closest_idx = j;
                }
            }

            if (closest_idx != -1 && min_dist <= max_error_sq) {
                sample_idxs.push_back(i);
                ref_idxs.push_back(closest_idx);
            }
        }
        return {ref_idxs, sample_idxs};
    }

    [[nodiscard]] std::pair<Eigen::Quaternionf, Eigen::Vector3f> iterate(
        const MatrixX3fRow& ref_points,
        const MatrixX3fRow& sample_points) const {

        float error = std::numeric_limits<float>::max();
        int iter_count = 0;

        Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
        Eigen::Vector3f translation = Eigen::Vector3f::Zero();

        MatrixX3fRow current_ref = ref_points;

        while (error > etol && iter_count < max_iter) {
            const auto [ref_idxs, sample_idxs] = get_correspondences(current_ref, sample_points);

            if (ref_idxs.size() < 3) break;

            MatrixX3fRow ref_subset(ref_idxs.size(), 3);
            MatrixX3fRow sample_subset(sample_idxs.size(), 3);

            for (size_t i = 0; i < ref_idxs.size(); ++i) {
                ref_subset.row(static_cast<int>(i)) = current_ref.row(ref_idxs[i]);
                sample_subset.row(static_cast<int>(i)) = sample_points.row(sample_idxs[i]);
            }

            auto [R, t] = KabschSolver::kabsch(sample_subset, ref_subset);

            rotation = R * rotation;
            translation = R * translation + t;

            current_ref = KabschSolver::apply(ref_points, rotation, translation);

            float current_sum_error = 0;
            for(int i = 0; i < static_cast<int>(ref_subset.rows()); ++i) {
                current_sum_error += (sample_subset.row(i) - (R * ref_subset.row(i).transpose() + t).transpose()).norm();
            }
            error = current_sum_error / static_cast<float>(ref_subset.rows());

            iter_count++;
        }

        return {Eigen::Quaternionf(rotation), translation};
    }
};
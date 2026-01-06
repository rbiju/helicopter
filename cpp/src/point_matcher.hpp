#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>

#include "nanoflann.hpp"
#include "kd_tree.hpp"
#include "kabsch.hpp"

// We define a Row-Major matrix type to match NumPy's default memory layout.
// This prevents memory corruption and avoids expensive data reordering when
// calling from Python.
typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> MatrixX3fRow;

struct PointTriplet {
    int i, j, k;
};

class PointMatcher {
public:
    // Ensure member variables match the layout passed by pybind11
    MatrixX3fRow reference_points;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> reference_distance_matrix;

    explicit PointMatcher(const MatrixX3fRow& ref_points) : reference_points(ref_points) {
        reference_distance_matrix = get_distance_matrix(ref_points, ref_points);
    }

    virtual ~PointMatcher() = default;

    // Optimized vectorized distance matrix calculation
    static Eigen::MatrixXf get_distance_matrix(const MatrixX3fRow& p1, const MatrixX3fRow& p2) {
        // ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a.b)
        const Eigen::VectorXf p1_sq = p1.rowwise().squaredNorm();
        Eigen::VectorXf p2_sq = p2.rowwise().squaredNorm();

        const Eigen::MatrixXf dists = p1_sq.replicate(1, p2.rows()) +
                                p2_sq.transpose().replicate(p1.rows(), 1) -
                                2.0f * (p1 * p2.transpose());

        // Clamp to 0 to avoid sqrt of tiny negatives due to float precision
        return dists.cwiseMax(0.0f).cwiseSqrt();
    }

    virtual std::pair<Eigen::Quaternionf, Eigen::Vector3f> get_alignment(const MatrixX3fRow& sample_points) = 0;
};

class TrianglePointMatcher final : public PointMatcher {
    int n_matches;
    MatrixX3fRow ref_triangle_data;
    std::vector<PointTriplet> ref_lookup; // Replaced map for safety and speed
    KDTreeWrapper kd_tree;

    struct Candidate {
        float dist_sq;
        int ref_idx;
        int s_i, s_j, s_k;

        // Smallest distance error first
        bool operator<(const Candidate& other) const {
            return dist_sq < other.dist_sq;
        }
    };

public:
    TrianglePointMatcher(const MatrixX3fRow& ref_points, const int n)
        : PointMatcher(ref_points), n_matches(n) {

        const int num_pts = static_cast<int>(reference_points.rows());
        const size_t total_combos = static_cast<size_t>(num_pts) * num_pts * num_pts;

        ref_triangle_data.resize(static_cast<int>(total_combos), 3);
        ref_lookup.reserve(total_combos);

        int idx = 0;
        for (int i = 0; i < num_pts; ++i) {
            for (int j = 0; j < num_pts; ++j) {
                for (int k = 0; k < num_pts; ++k) {
                    ref_triangle_data.row(idx) <<
                        reference_distance_matrix(i, j),
                        reference_distance_matrix(j, k),
                        reference_distance_matrix(i, k);

                    ref_lookup.push_back({i, j, k});
                    idx++;
                }
            }
        }
        kd_tree.build(ref_triangle_data);
    }

    std::pair<Eigen::Quaternionf, Eigen::Vector3f> get_alignment(const MatrixX3fRow& sample_points) override {
        const int S = static_cast<int>(sample_points.rows());
        Eigen::MatrixXf sample_dm = get_distance_matrix(sample_points, sample_points);

        std::vector<Candidate> all_candidates;
        // Logic check: only proceed if we have at least 3 points
        if (S < 3) return {Eigen::Quaternionf::Identity(), Eigen::Vector3f::Zero()};

        // Safety cap to prevent massive memory spikes in Python if sample cloud is large
        const size_t expected_triangles = static_cast<size_t>(S) * (S - 1) * (S - 2) / 6;
        all_candidates.reserve(std::min<size_t>(expected_triangles, 500000));

        for (int i = 0; i < S; ++i) {
            for (int j = i + 1; j < S; ++j) {
                for (int k = j + 1; k < S; ++k) {
                    Eigen::Vector3f sample_tri(sample_dm(i, j), sample_dm(j, k), sample_dm(i, k));

                    if (std::vector<size_t> match = kd_tree.query_top_n(sample_tri, 1); !match.empty()) {
                        // Bounds checking against ref_lookup prevents most segfaults
                        if (const int r_idx = static_cast<int>(match[0]); r_idx >= 0 && r_idx < static_cast<int>(ref_lookup.size())) {
                            const float d_sq = (ref_triangle_data.row(r_idx).transpose() - sample_tri).squaredNorm();
                            all_candidates.push_back({d_sq, r_idx, i, j, k});
                        }
                    }
                }
            }
        }

        std::sort(all_candidates.begin(), all_candidates.end());

        float best_error = std::numeric_limits<float>::max();
        Eigen::Matrix3f best_R = Eigen::Matrix3f::Identity();
        Eigen::Vector3f best_t = Eigen::Vector3f::Zero();

        const int limit = std::min(static_cast<int>(all_candidates.size()), n_matches);
        for (int i = 0; i < limit; ++i) {
            const auto& cand = all_candidates[i];
            const auto&[t_i, t_j, t_k] = ref_lookup[cand.ref_idx];

            Eigen::Matrix3f p_ref, p_sam;
            p_ref.row(0) = reference_points.row(t_i);
            p_ref.row(1) = reference_points.row(t_j);
            p_ref.row(2) = reference_points.row(t_k);

            p_sam.row(0) = sample_points.row(cand.s_i);
            p_sam.row(1) = sample_points.row(cand.s_j);
            p_sam.row(2) = sample_points.row(cand.s_k);

            // Assuming KabschSolver::kabsch handles Eigen::Matrix3f correctly
            auto [R, t] = KabschSolver::kabsch(p_sam, p_ref);

            // Calculate RMSE/Alignment error
            MatrixX3fRow transformed_ref = (reference_points * R.transpose()).rowwise() + t.transpose();
            Eigen::MatrixXf dists = get_distance_matrix(transformed_ref, sample_points);

            float current_error = 0;
            for(int r=0; r < dists.rows(); ++r) {
                current_error += dists.row(r).minCoeff();
            }

            if (current_error < best_error) {
                best_error = current_error;
                best_R = R;
                best_t = t;
            }
        }

        return {Eigen::Quaternionf(best_R), best_t};
    }
};
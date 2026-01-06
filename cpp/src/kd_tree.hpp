#pragma once
#include <vector>
#include <memory>

#include "nanoflann.hpp"

class KDTreeWrapper {
    using MyKDTree = nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf, 3, nanoflann::metric_L2_Simple>;
    std::unique_ptr<MyKDTree> index;

public:
    void build(const Eigen::MatrixXf& data) {
        index = std::make_unique<MyKDTree>(3, std::cref(data), 2);
        index->index_->buildIndex();
    }

    [[nodiscard]] std::vector<size_t> query_top_n(const Eigen::Vector3f& query_pt, const int n) const {
        std::vector<size_t> ret_indexes(n);
        std::vector<float> out_dists_sqr(n);
        nanoflann::KNNResultSet<float> resultSet(n);

        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
        index->index_->findNeighbors(resultSet, query_pt.data(), nanoflann::SearchParameters(10));

        return ret_indexes;
    }
};
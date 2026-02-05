#pragma once
#ifndef DBSCAN_H
#define DBSCAN_H

#include <set>
#include <vector>
#include <cmath>
#include "./AickTensorrt.h"

#define UNCLASSIFIED -1
#define CORE_POINT 1
#define BORDER_POINT 2
#define NOISE -2
#define SUCCESS 0
#define FAILURE -3

using namespace std;

class DBSCAN {
public:
    DBSCAN(int minPts, float eps, const vector<PointZX>& points) {
        m_minPoints = minPts;
        m_epsilon = eps;
        m_points = points;
        m_pointSize = points.size();
    }
    ~DBSCAN() {}

    int run();
    vector<int> calculateCluster(PointZX point);
    int expandCluster(PointZX point, int clusterID);
    inline double calculateDistance(const PointZX& pointCore, const PointZX& pointTarget) {
        return pow(pointCore.x - pointTarget.x, 2) + pow(pointCore.y - pointTarget.y, 2);
    }

    int getTotalPointSize() { return m_pointSize; }
    int getMinimumClusterSize() { return m_minPoints; }
    float getEpsilonSize() { return m_epsilon; }

public:
    vector<PointZX> m_points;

private:
    unsigned int m_pointSize;
    unsigned int m_minPoints;
    float m_epsilon;
};

#endif
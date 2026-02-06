#include "dbscan.h"
#include <iostream>


int DBSCAN::run()
{
    int clusterID = 1;
    for (auto iter = m_points.begin(); iter != m_points.end(); ++iter)
    {
        if (iter->clusterID == UNCLASSIFIED)
        {
            if (expandCluster(*iter, clusterID) != FAILURE)
            {
                clusterID += 1;
            }
        }
    }
    return 0;
}

int DBSCAN::expandCluster(PointZX point, int clusterID)
{
    vector<int> clusterSeeds = calculateCluster(point);

    if (clusterSeeds.size() < m_minPoints)
    {
        for (auto& p : m_points) {
            if (p.x == point.x && p.y == point.y) { 
                p.clusterID = NOISE;
                break;
            }
        }
        return FAILURE;
    }
    else
    {
        for (int seedIdx : clusterSeeds) {
            m_points[seedIdx].clusterID = clusterID;
        }

        // 扩展邻域 
        size_t i = 0;
        while (i < clusterSeeds.size()) {
            int currentSeedIdx = clusterSeeds[i];
            vector<int> clusterNeighbors = calculateCluster(m_points[currentSeedIdx]);

            if (clusterNeighbors.size() >= m_minPoints) {
                for (int neighborIdx : clusterNeighbors) {
                    if (m_points[neighborIdx].clusterID == UNCLASSIFIED || m_points[neighborIdx].clusterID == NOISE) {
                        if (m_points[neighborIdx].clusterID == UNCLASSIFIED) {
                            clusterSeeds.push_back(neighborIdx); 
                        }
                        m_points[neighborIdx].clusterID = clusterID;
                    }
                }
            }
            i++;
        }
    }
    return SUCCESS;
}


vector<int> DBSCAN::calculateCluster(PointZX point)
{
    vector<int> clusterIndex;
    float eps_sq = m_epsilon * m_epsilon; 

    for (size_t i = 0; i < m_points.size(); ++i)
    {
        // 直接计算平方距离
        double dist_sq = pow(point.x - m_points[i].x, 2) + pow(point.y - m_points[i].y, 2);
        if (dist_sq <= eps_sq)
        {
            clusterIndex.push_back(i);
        }
    }
    return clusterIndex;
}

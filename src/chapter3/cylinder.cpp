#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <opencv2/core/eigen.hpp>

#include"sophus/se3.hpp"
#include "g2o/config.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/icp/types_icp.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>

#include "g2o/types/sba/types_six_dof_expmap.h"

#include "cyRepresentation.h"

void createLandmarks(std::vector<Eigen::Vector3d> &points)
{
    float scale = 5;
    const double k = 0.5;
    double r=10;

    points.clear();
    
    std::mt19937 gen{12345};
    //从均匀分布中生成随机的浮点数
    std::uniform_real_distribution<float> theta(-3.1415926535, 3.1415926535);
    std::uniform_real_distribution<double> length{-15.0, 15.0};
    std::uniform_real_distribution<double> obstacle{0, 1};
    std::normal_distribution<double> noise{0.0, 0.5};

    Eigen::Vector3d world(0,0,1);
    Eigen::Vector3d cyAxis(1,1,1);
    //得到cy->world的旋转矩阵
    Eigen::Matrix3d R_w_cy(Eigen::Quaterniond::FromTwoVectors(world, cyAxis));
    Sophus::SO3d SO3_w_cy(R_w_cy);
    Eigen::Vector3d so3_w_cy = SO3_w_cy.log();
    std::cout<<"so3实际值:"<<so3_w_cy<<std::endl;
    //平移分量
    Eigen::Vector3d cyPos(4,4,4);

    for (int i = 0; i < 200; i ++)
    {
        //生成圆柱坐标系的pt,圆柱的z轴指向轴线,x轴指向世界坐标系原点
        Eigen::Vector3d pt;
        float theta1=theta(gen);
        pt[0] = r*cos(theta1);
        pt[1] = r*sin(theta1);
        pt[2] = length(gen);
        //        pt[2]=1;
        //pt转为世界坐标系下
        pt=R_w_cy*pt+cyPos;
        points.push_back(pt);
    }

    for (int j=0; j< 0; j++) {
        Eigen::Vector3d pt;
        float theta1=0.3*theta(gen);
        pt[0] = obstacle(gen)*r*cos(theta1);
        pt[1] = obstacle(gen)*r*sin(theta1);
        pt[2] = length(gen);

        pt=R_w_cy*pt+cyPos;
        points.push_back(pt);

    }
}

std::vector<Eigen::Vector3d> addLandmarksNoise(std::vector<Eigen::Vector3d> points)
{
    size_t n=points.size();
    std::vector<Eigen::Vector3d> noisyPoints;

    //gen里面 random_device?
    std::mt19937 gen{12345};
    std::normal_distribution<double> noise{0.0, 0.025};

    for (int i = 0; i < n; i ++) {
        Eigen::Vector3d pt;
        pt[0] = points[i][0] *(1+noise(gen)) ;
        pt[1] = points[i][1]  *(1+noise(gen)) ;
        pt[2] = points[i][2]  *(1+noise(gen)) ;
        noisyPoints.push_back(pt);
//        std::cout<<pt.transpose()<<std::endl;
    }
    return noisyPoints;
}



void createCameraPose(std::vector<Eigen::Matrix4d> &v_Twc, std::vector<Eigen::Matrix4d> &v_noisyTwc)
{
    // 每创建一个相机窗口，两个vector中压入一个只有准确旋转的，再压入带噪声的T和不带噪声的T?
    v_Twc.clear();
    v_noisyTwc.clear();

    Eigen::Vector3d world(0,0,1);
    Eigen::Vector3d cyAxis(1,1,1);
    Eigen::Matrix3d R_w_c(Eigen::Quaterniond::FromTwoVectors(world, cyAxis));
    Eigen::Vector3d cPos(20,20,20);

    Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
    //block(i, j, p, q) p, q表示block大小，i, j 表示从哪个元素开始
    Twc.block(0, 0, 3, 3) = R_w_c;
    v_Twc.push_back(Twc);
    v_noisyTwc.push_back(Twc);

    Twc.block(0, 0, 3, 3) = R_w_c;
    Twc.block(0, 3, 3, 1) = cPos;
    v_Twc.push_back(Twc);

    cyAxis<<0.95,1,0.95;
    R_w_c=Eigen::Quaterniond::FromTwoVectors(world, cyAxis);
    cPos<<21,23,19;

    Twc.block(0, 0, 3, 3) = R_w_c;
    Twc.block(0, 3, 3, 1) = cPos;
    v_noisyTwc.push_back(Twc);
}

void detectFeatures(const Eigen::Matrix4d &Twc, const Eigen::Matrix3d &K,
                    const std::vector<Eigen::Vector3d> &landmarks, std::vector<Eigen::Vector2i> &features, bool add_noise = true)
{
    std::mt19937 gen{12345};
    const float pixel_sigma = 1.0;
    std::normal_distribution<> d{0.0, pixel_sigma};

    Eigen::Matrix3d Rwc = Twc.block(0, 0, 3, 3);
    Eigen::Vector3d twc = Twc.block(0, 3, 3, 1);
    Eigen::Matrix3d Rcw = Rwc.transpose();
    Eigen::Vector3d tcw = -Rcw * twc;

    features.clear();
    for (size_t l = 0; l < landmarks.size(); ++l)
    {
        Eigen::Vector3d wP = landmarks[l];
        Eigen::Vector3d cP = Rcw * wP + tcw;

        if(cP[2] < 0) continue;

        float noise_u = add_noise ? std::round(d(gen)) : 0.0f;
        float noise_v = add_noise ? std::round(d(gen)) : 0.0f;

        Eigen::Vector3d ft = K * cP;
        int u = ft[0]/ft[2] + 0.5 + noise_u;
        int v = ft[1]/ft[2] + 0.5 + noise_v;
        Eigen::Vector2i obs(u, v);
        features.push_back(obs);
//        std::cout << l << " " << obs.transpose() << std::endl;
    }
}

int main()
{
    std::vector<Eigen::Vector3d> landmarks;
    std::vector<Eigen::Matrix4d> v_Twc;
    std::vector<Eigen::Matrix4d> v_noisyTwc;

    createLandmarks(landmarks);
    //noisyLandmarks和landmarks是分开的
    std::vector<Eigen::Vector3d> noisyLandmarks=addLandmarksNoise(landmarks);
    createCameraPose(v_Twc,v_noisyTwc);
    const size_t pose_num = v_Twc.size();   //应该有两个

    cv::Mat cv_K = (cv::Mat_<double>(3, 3) << 480, 0, 320, 0, 480, 240, 0, 0, 1);
    Eigen::Matrix3d K;
    cv::cv2eigen(cv_K, K);

    std::vector<Eigen::Vector2i> features_curr;
    // Setup optimizer
    g2o::SparseOptimizer optimizer;

    //动态求解块
    typedef g2o::BlockSolverX BlockSolverType;
    // 使用dense cholesky分解法
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
    // 梯度下降方法，可以从GN, LM, DogLeg 中选
    //创建总求解器
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    optimizer.setVerbose(true);       // 打开调试输出
    optimizer.setAlgorithm(solver);

    double focal_length= 480.;
    Eigen::Vector2d principal_point(320., 240.);
    //inline g2o::CameraParameters(number_t focal_length, const g2o::Vector2 &principle_point, number_t baseline)
    g2o::CameraParameters* camera = new g2o::CameraParameters( focal_length, principal_point, 0 );
    camera->setId(0);
    optimizer.addParameter( camera );
//
    Eigen::Vector2d obs;

    // 往图中增加顶点
    CylinderFittingVertex *v = new CylinderFittingVertex();
    //动态vector?
    CylinderIntrinsics abc;
    abc.rotation = Sophus::SO3d::exp(Eigen::Vector3d(0.1,0.1,23.0));
    abc.qx = -2.0;
    abc.r = 1.0;
    v->setEstimate(abc);
    v->setId(0);
//    v->setFixed(true);
    optimizer.addVertex(v);

    //    //map points
    for (size_t j = 0; j < noisyLandmarks.size(); j++) {
        g2o::VertexPointXYZ *vPoint = new g2o::VertexPointXYZ();
//        std::cout << noisyLandmarks[j].transpose() << std::endl;
        vPoint->setEstimate(noisyLandmarks[j]);
        vPoint->setId(j + 3);
//        vPoint->setFixed(true);
//        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
    }

    std::vector<CylinderFittingEdge *>  vcy;

    // 往图中增加边
    for (size_t i = 0; i < noisyLandmarks.size(); i++) {
        CylinderFittingEdge *edge2 = new CylinderFittingEdge();
        edge2->setId(noisyLandmarks.size()*2+i);
        edge2->setVertex(0, optimizer.vertices()[i+3]);             // 设置连接的顶点
        edge2->setVertex(1, optimizer.vertices()[0]);
        edge2->setInformation(Eigen::Matrix<double, 1, 1>::Identity()); // 信息矩阵：协方差矩阵之逆
        optimizer.addEdge(edge2);
        vcy.push_back(edge2);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(50);
    std::cout<<v->estimate().rotation.log()<<std::endl;
    std::cout<< v->estimate().qx<<' '<< v->estimate().r<<std::endl;

    return 0;
}


    #include <iostream>
    #include <cmath>
    #include <ceres/ceres.h>

    #include "sophus/se3.hpp"

    #include <Eigen/Core>
    #include <Eigen/Dense>
    #include </usr/include/eigen3/unsupported/Eigen/MatrixFunctions>
int main() {
    Eigen::Matrix3d A;
    A << 1, 0, 0, 0, 1, 0, 0, 0, 0;
    Eigen::Vector3d world(0,0,1);
    Eigen::Vector3d cyAxis(1,1,1);
    Eigen::Matrix3d Ryw(Eigen::Quaterniond::FromTwoVectors(world, cyAxis));
    // 旋转矩阵对应的李群
    Sophus::SO3d SO3_Ryw(Ryw);
    // 平移
    Eigen::Vector3d qx(1,0,0);
    // 数据点
    Eigen::Vector3d pw(1,2,3);
    double y = (A * (Ryw * pw + qx)).transpose() * (A * (Ryw * pw + qx));
    Eigen::Vector3d Rp = Ryw * pw;
    double u = Rp.transpose() * A.transpose() * A * Rp;
    u += qx.transpose() * A.transpose() * A * Rp;
        //todo ? * 2.0
    u +=  (Rp.transpose() * A.transpose() * A * qx)[0] * 2.0;
    u += qx.transpose() * A.transpose() * A * qx;
    std::cout<<"y = :"<<y<<std::endl;
    std::cout<<"u = :"<<u<<std::endl;
    Eigen::Matrix3d Rp_hat;
    Rp_hat << 0, -Rp(2), Rp(1), Rp(2), 0, -Rp(0), -Rp(1), Rp(0), 0;
    // 半径
    double r = 10.0;
    // 微小扰动
    double h = 0.0000001;
    Eigen::Vector3d so3_Rh_1;
    so3_Rh_1 << h, 0, 0;
    Eigen::Vector3d so3_Rh_2;
    so3_Rh_2 << 0, h, 0;
    Eigen::Vector3d so3_Rh_3;
    so3_Rh_3 << 0, 0, h;
    // 对旋转矩阵左乘一个微小扰动后的旋转矩阵
    // 用Sophus对三个维度分别加扰动
    // Sophus::SO3d SO3_updated_1 = Sophus::SO3d::exp(so3_Rh_1) * SO3_Ryw;
    // Eigen::Matrix3d Ryw_h_1 = SO3_updated_1.matrix();

    // Sophus::SO3d SO3_updated_2 = Sophus::SO3d::exp(so3_Rh_2) * SO3_Ryw;
    // Eigen::Matrix3d Ryw_h_2 = SO3_updated_2.matrix();

    // Sophus::SO3d SO3_updated_3 = Sophus::SO3d::exp(so3_Rh_3) * SO3_Ryw;
    // Eigen::Matrix3d Ryw_h_3 = SO3_updated_3.matrix();

    // 数学方式对三个维度分别加扰动
    // 微小扰动向量也就是李代数的反对称矩阵
    
    Eigen::Matrix3d Ryw_h_1 = Sophus::SO3d::hat(so3_Rh_1).exp() * Ryw;
    Eigen::Matrix3d Ryw_h_2 = Sophus::SO3d::hat(so3_Rh_2).exp() * Ryw;
    Eigen::Matrix3d Ryw_h_3 = Sophus::SO3d::hat(so3_Rh_3).exp() * Ryw;

    // 对旋转求的导数
    Eigen::Vector3d Rotation_def = (1./2 * pow(y, -1./2)) * (2.0 * (A * (Ryw * pw) + qx).transpose()) * (-Rp_hat);
    // 对平移求的导数
    double translation_def = (1./2 * pow(y, -1./2)) * (2.0 * ((Ryw * pw).transpose() + qx.transpose())[0]);
    // 对半径的求导结果不用验证了
    // 对旋转的自动求导
    double e = sqrt((A * (Ryw * pw + qx)).transpose() * (A * (Ryw * pw + qx))) - r; 
    double e_Rh_1 = sqrt((A * (Ryw_h_1 * pw + qx)).transpose() * (A * (Ryw_h_1 * pw + qx))) - r;
    double e_Rh_2 = sqrt((A * (Ryw_h_2 * pw + qx)).transpose() * (A * (Ryw_h_2 * pw + qx))) - r;
    double e_Rh_3 = sqrt((A * (Ryw_h_3 * pw + qx)).transpose() * (A * (Ryw_h_3 * pw + qx))) - r;
    double def1 = (e_Rh_1 - e) / h;
    double def2 = (e_Rh_2 - e) / h;
    double def3 = (e_Rh_3 - e) / h;

    Eigen::Vector3d Rotation_autodef;
    Rotation_autodef[0] = def1;
    Rotation_autodef[1] = def2;
    Rotation_autodef[2] = def3;
    // 对平移的自动求导
    Eigen::Vector3d qx_h(1 + h, 0, 0);
    double e_qh = sqrt((A * (Ryw * pw + qx_h)).transpose() * (A * (Ryw * pw + qx_h))) - r;
    double translation_autodef = (e_qh - e) / h;
    std::cout << "旋转的导数求导 e'(R) = " << Rotation_def.transpose() << std::endl;
    std::cout << "旋转的自动求导 e'(R) = " << Rotation_autodef.transpose() << std::endl;
    std::cout << "平移的导数求导 e'(qx) = " << translation_def << std::endl;
    std::cout << "平移的自动求导 e'(qx) = " << translation_autodef << std::endl;
    return 0;
}

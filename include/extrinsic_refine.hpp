#ifndef EXTRINSIC_REFINE_HPP
#define EXTRINSIC_REFINE_HPP

#include <glog/logging.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>

#include <Eigen/StdVector>
#include <cmath>
#include <fstream>
#include <mutex>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <unordered_map>

#include "BA/mypcl.hpp"
#include "common.h"

class EXTRIN_OPTIMIZER {
 public:
  int pose_size, ref_size, jacob_len;
  vector_quad poses, refQs, refQsTmp;
  vector_vec3d ts, refTs, refTsTmp;
  std::vector<vector_vec3d*> baseOriginPts;
  std::vector<std::vector<int>*> baseWinNums;
  std::vector<std::vector<vector_vec3d*>> refOriginPts;
  std::vector<std::vector<std::vector<int>*>> refWinNums;

  EXTRIN_OPTIMIZER(size_t win_sz, size_t r_sz)
      : pose_size(win_sz), ref_size(r_sz) {
    jacob_len = ref_size * 6;
    poses.resize(pose_size);
    ts.resize(pose_size);
    refQs.resize(ref_size);
    refQsTmp.resize(ref_size);
    refTs.resize(ref_size);
    refTsTmp.resize(ref_size);
    refOriginPts.resize(ref_size);
    refWinNums.resize(ref_size);
  };

  ~EXTRIN_OPTIMIZER() {
    for (uint i = 0; i < baseOriginPts.size(); i++) {
      delete (baseOriginPts[i]);
      delete (baseWinNums[i]);
    }
    baseOriginPts.clear();
    baseWinNums.clear();

    for (uint i = 0; i < refOriginPts.size(); i++) {
      for (uint j = 0; j < refOriginPts[i].size(); j++) {
        delete (refOriginPts[i][j]);
        delete (refWinNums[i][j]);
      }
      refOriginPts[i].clear();
      refWinNums[i].clear();
    }
    refOriginPts.clear();
    refWinNums.clear();
  }

  void get_center(vector_vec3d& originPc, int cur_frame, vector_vec3d& originPt,
                  std::vector<int>& winNum, int filterNum) {
    size_t pt_size = originPc.size();
    if (pt_size <= (size_t)filterNum) {
      for (size_t i = 0; i < pt_size; i++) {
        originPt.push_back(originPc[i]);
        winNum.push_back(cur_frame);
      }
      return;
    }

    Eigen::Vector3d center;
    double part = 1.0 * pt_size / filterNum;

    for (int i = 0; i < filterNum; i++) {
      size_t np = part * i;
      size_t nn = part * (i + 1);
      center.setZero();
      for (size_t j = np; j < nn; j++) center += originPc[j];

      center = center / (nn - np);
      originPt.push_back(center);
      winNum.push_back(cur_frame);
    }
  }

  void push_voxel(std::vector<vector_vec3d*>& baseOriginPc,
                  std::vector<std::vector<vector_vec3d*>>& refOriginPc) {
    int baseLidarPc = 0;
    for (int i = 0; i < pose_size; i++) {
      if (!baseOriginPc[i]->empty()) {
        baseLidarPc++;
      }
    }

    int refLidarPc = 0;
    for (int j = 0; j < ref_size; j++) {
      for (int i = 0; i < pose_size; i++) {
        if (!refOriginPc[j][i]->empty()) {
          refLidarPc++;
        }
      }
    }

    if (refLidarPc + baseLidarPc <= 1) {
      return;
    }

    constexpr int filterNum = 4;
    int baseLidarPt = 0;
    int refLidarPt = 0;

    vector_vec3d* baseOriginPt = new vector_vec3d();
    std::vector<int>* baseWinNum = new std::vector<int>();
    baseWinNum->reserve(filterNum * pose_size);
    baseOriginPt->reserve(filterNum * pose_size);

    for (int i = 0; i < pose_size; i++) {
      if (!baseOriginPc[i]->empty()) {
        get_center(*baseOriginPc[i], i, *baseOriginPt, *baseWinNum, filterNum);
      }
    }

    baseLidarPt += baseOriginPt->size();
    baseOriginPts.emplace_back(baseOriginPt);  // Note they might be empty
    baseWinNums.emplace_back(baseWinNum);
    assert(baseOriginPt->size() == baseWinNum->size());

    for (int j = 0; j < ref_size; j++) {
      vector_vec3d* refOriginPt = new vector_vec3d();
      std::vector<int>* refWinNum = new std::vector<int>();
      refWinNum->reserve(filterNum * pose_size);
      refOriginPt->reserve(filterNum * pose_size);

      for (int i = 0; i < pose_size; i++) {
        if (!refOriginPc[j][i]->empty()) {
          get_center(*refOriginPc[j][i], i, *refOriginPt, *refWinNum,
                     filterNum);
        }
      }

      refLidarPt += refOriginPt->size();

      refOriginPts[j].emplace_back(refOriginPt);
      refWinNums[j].emplace_back(refWinNum);
    }

    for (int j = 0; j < ref_size; j++) {
      assert(refOriginPts[j].size() == baseOriginPts.size());
      assert(refWinNums[j].size() == baseWinNums.size());
    }
  }

  void optimize() {
    Eigen::MatrixXd D(jacob_len, jacob_len), Hess(jacob_len, jacob_len);
    Eigen::VectorXd JacT(jacob_len), dxi(jacob_len);

    Eigen::MatrixXd Hess2(jacob_len, jacob_len);
    Eigen::VectorXd JacT2(jacob_len);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;

    cv::Mat matA(jacob_len, jacob_len, CV_64F, cv::Scalar::all(0));
    cv::Mat matB(jacob_len, 1, CV_64F, cv::Scalar::all(0));
    cv::Mat matX(jacob_len, 1, CV_64F, cv::Scalar::all(0));

    double u = 0.01, v = 2;

    for (int loop = 0; loop < 20; loop++) {
      if (is_calc_hess) {
        divide_thread(poses, ts, refQs, refTs, Hess, JacT, residual1);
      }

      D = Hess.diagonal().asDiagonal();
      Hess2 = Hess + u * D;

      for (int j = 0; j < jacob_len; j++) {
        // matB.at<double>(j, 0) = -JacT(j, 0);
        for (int f = 0; f < jacob_len; f++) {
          matA.at<double>(j, f) = Hess2(j, f);
        }
      }
      cv::solve(matA, matB, matX, cv::DECOMP_QR);

      for (int j = 0; j < jacob_len; j++) {
        dxi(j, 0) = matX.at<double>(j, 0);
      }

      for (int i = 0; i < ref_size; i++) {
        Eigen::Quaterniond q_tmp(exp(dxi.block<3, 1>(6 * i, 0)) * refQs[i]);
        Eigen::Vector3d t_tmp(dxi.block<3, 1>(6 * i + 3, 0) + refTs[i]);
        assign_qt(refQsTmp[i], refTsTmp[i], q_tmp, t_tmp);
      }

      double q1 = 0.5 * (dxi.transpose() * (u * D * dxi - JacT))[0];
      evaluate_only_residual(poses, ts, refQsTmp, refTsTmp, residual2);

      q = residual1 - residual2;

      assert(!std::isnan(residual1));
      assert(!std::isnan(residual2));

      if (q > 0) {
        LOG(INFO) << "........... residual decreased !";
        for (int i = 0; i < ref_size; i++) {
          assign_qt(refQs[i], refTs[i], refQsTmp[i], refTsTmp[i]);
        }
        q = q / q1;
        v = 2;
        q = 1 - pow(2 * q - 1, 3);
        u *= (q < 1.0 / 3 ? 1.0 / 3 : q);
        is_calc_hess = true;
      } else {
        LOG(INFO) << "........... residual not decreased !";
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;
      }

      if (fabs(residual1 - residual2) < 1e-9) {
        for (int i = 0; i < ref_size; i++) {
          assign_qt(refQs[i], refTs[i], refQsTmp[i], refTsTmp[i]);
        }
        break;
      }
    }
  }

  void divide_thread(vector_quad& poses, vector_vec3d& ts, vector_quad& refQs,
                     vector_vec3d& refTs, Eigen::MatrixXd& Hess,
                     Eigen::VectorXd& JacT, double& residual) {
    Hess.setZero();
    JacT.setZero();
    residual = 0;

    const int total_pose_num = refQs.size();

    std::vector<Eigen::MatrixXd> hessians(total_pose_num, Hess);
    std::vector<Eigen::VectorXd> jacobians(total_pose_num, JacT);
    std::vector<double> resis(total_pose_num, 0);

    uint gps_size = baseOriginPts.size();
    if (gps_size < total_pose_num) {
      calculate_HJ(poses, ts, refQs, refTs, 0, gps_size, Hess, JacT, residual);
      return;
    }

    std::vector<std::thread*> mthreads(total_pose_num);

    const double part = 1.0 * gps_size / total_pose_num;
    for (int i = 0; i < total_pose_num; i++) {
      const int np = part * i;
      const int nn = part * (i + 1);

      mthreads[i] = new std::thread(
          &EXTRIN_OPTIMIZER::calculate_HJ, this, std::ref(poses), std::ref(ts),
          std::ref(refQs), std::ref(refTs), np, nn, std::ref(hessians[i]),
          std::ref(jacobians[i]), std::ref(resis[i]));
    }

    for (int i = 0; i < total_pose_num; i++) {
      mthreads[i]->join();
      Hess += hessians[i];
      JacT += jacobians[i];
      residual += resis[i];
      delete mthreads[i];
    }
  }

  // (NOTE) BALM paper: https://arxiv.org/pdf/2010.08215.pdf.
  //        MLCC paper: https://arxiv.org/pdf/2109.06550.pdf.
  void calculate_HJ(vector_quad& poses, vector_vec3d& ts, vector_quad& refQs,
                    vector_vec3d& refTs, int head, int end,
                    Eigen::MatrixXd& Hess, Eigen::VectorXd& JacT,
                    double& residual) {
    Hess.setZero();
    JacT.setZero();
    residual = 0;
    Eigen::MatrixXd _hess(Hess);
    Eigen::MatrixXd _jact(JacT);

    for (int i = head; i < end; i++) {
      Eigen::Vector3d vec_tran;
      Eigen::Vector3d center(Eigen::Vector3d::Zero());
      Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());

      // Each ref LiDAR will occupy an element of the vector.
      std::vector<vector_vec3d> pt_trans(ref_size);
      std::vector<std::vector<Eigen::Matrix3d>> point_xis(ref_size);
      int point_num = 0;

      for (int k = 0; k < ref_size; ++k) {
        const vector_vec3d& ref_origin_pts = *refOriginPts[k][i];
        const std::vector<int>& ref_win_num = *refWinNums[k][i];
        point_xis[k].resize(ref_origin_pts.size());
        pt_trans[k].resize(ref_origin_pts.size());

        for (size_t j = 0; j < ref_origin_pts.size(); ++j) {
          // in base LiDAR's origin frame.
          vec_tran = refQs[k] * ref_origin_pts[j];
          point_xis[k][j] = -wedge(vec_tran);
          pt_trans[k][j] = poses[ref_win_num[j]] * (vec_tran + refTs[k]) +
                           ts[ref_win_num[j]];
          center += pt_trans[k][j];
          covMat += pt_trans[k][j] * pt_trans[k][j].transpose();
        }
        point_num += ref_origin_pts.size();
      }

      const vector_vec3d& baseorigin_pts = *baseOriginPts[i];
      const std::vector<int>& basewin_num = *baseWinNums[i];
      const size_t basepts_size = baseorigin_pts.size();

      for (size_t j = 0; j < basepts_size; j++) {
        vec_tran =
            poses[basewin_num[j]] * baseorigin_pts[j] + ts[basewin_num[j]];
        center += vec_tran;
        covMat += vec_tran * vec_tran.transpose();
      }

      const int N = point_num + basepts_size;
      covMat = covMat - center * center.transpose() / N;
      covMat = covMat / N;
      center = center / N;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
      const Eigen::Vector3d eigen_value = saes.eigenvalues();

      const Eigen::Matrix3d U = saes.eigenvectors();
      Eigen::Vector3d u[3];
      for (int j = 0; j < 3; ++j) {
        u[j] = U.block<3, 1>(0, j);
      }

      const Eigen::Matrix3d ukukT = u[0] * u[0].transpose();
      Eigen::Vector3d vec_Jt;

      for (int k = 0; k < ref_size; ++k) {
        const auto& ref_win_num = *refWinNums[k][i];
        assert(ref_win_num.size() == pt_trans[k].size());

        for (size_t j = 0; j < pt_trans[k].size(); ++j) {
          pt_trans[k][j] -= center;
          // df/d(pt_trans[k][j])
          vec_Jt = 2.0 / N * ukukT * pt_trans[k][j];

          const auto R = poses[ref_win_num[j]].toRotationMatrix();
          Eigen::MatrixXd D(3, 6);
          D.block<3, 3>(0, 0) = R * point_xis[k][j];
          D.block<3, 3>(0, 3) = R;
          _jact.block<6, 1>(6 * k, 0) += D.transpose() * vec_Jt;
        }
      }

      Eigen::Matrix3d Hessian33;
      Eigen::Matrix3d F;
      std::vector<Eigen::Matrix3d> F_(3);
      for (size_t j = 0; j < 3; ++j) {
        if (j == 0) {
          F_[j].setZero();
          continue;
        }
        Hessian33 = u[j] * u[0].transpose();
        F_[j] = 1.0 / N / (eigen_value[0] - eigen_value[j]) *
                (Hessian33 + Hessian33.transpose());
      }

      size_t rownum, colnum;

      for (int k1 = 0; k1 < ref_size; ++k1) {
        const auto& ref_win_num_1 = *refWinNums[k1][i];
        colnum = 6 * k1;

        for (int j1 = 0; j1 < pt_trans[k1].size(); ++j1) {
          // The column part.
          for (int f = 0; f < 3; ++f) {
            F.block<1, 3>(f, 0) = pt_trans[k1][j1].transpose() * F_[f];
          }
          F = U * F;

          Eigen::MatrixXd D1(3, 6);
          // point_xis already have a minus sign.
          const auto R1 = poses[ref_win_num_1[j1]].toRotationMatrix();
          // D_{j1, k1} according to MLCC paper's Equation (26).
          D1.block<3, 3>(0, 0) = R1 * point_xis[k1][j1];
          D1.block<3, 3>(0, 3) = R1;

          // The row part.
          for (int k2 = 0; k2 < ref_size; ++k2) {
            const auto& ref_win_num_2 = *refWinNums[k2][i];
            rownum = 6 * k2;

            for (int j2 = 0; j2 < pt_trans[k2].size(); ++j2) {
              Hessian33 = u[0] * (pt_trans[k2][j2]).transpose() * F +
                          u[0].dot(pt_trans[k2][j2]) * F;
              if (k1 == k2 && j1 == j2) {
                // The same point.
                Hessian33 += static_cast<double>(N - 1) / N * ukukT;
              } else {
                Hessian33 -= 1.0 / N * ukukT;
              }
              Hessian33 *= 2.0 / N;

              Eigen::MatrixXd D2(3, 6);
              // point_xis already have a minus sign.
              const auto R2 = poses[ref_win_num_2[j2]].toRotationMatrix();
              // D_{j2, k2} according to MLCC paper's Equation (26).
              D2.block<3, 3>(0, 0) = R2 * point_xis[k2][j2];
              D2.block<3, 3>(0, 3) = R2;

              // H(k2, k1) block = D2(j2, k2)^T * H33 * D1(j1, k1).
              _hess.block<3, 3>(rownum, colnum) +=
                  D2.block<3, 3>(0, 0).transpose() * Hessian33 *
                  D1.block<3, 3>(0, 0);
              _hess.block<3, 3>(rownum + 3, colnum) +=
                  D2.block<3, 3>(0, 3).transpose() * Hessian33 *
                  D1.block<3, 3>(0, 0);
              _hess.block<3, 3>(rownum, colnum + 3) +=
                  D2.block<3, 3>(0, 0).transpose() * Hessian33 *
                  D1.block<3, 3>(0, 3);
              _hess.block<3, 3>(rownum + 3, colnum + 3) +=
                  D2.block<3, 3>(0, 3).transpose() * Hessian33 *
                  D1.block<3, 3>(0, 3);
            }
          }
        }
      }

      residual += eigen_value[0];
      Hess += _hess;
      JacT += _jact;
      _hess.setZero();
      _jact.setZero();
    }
  }

  void evaluate_only_residual(const vector_quad& poses_,
                              const vector_vec3d& ts_,
                              const vector_quad& refposes_,
                              const vector_vec3d& refts_, double& residual) {
    residual = 0;
    size_t voxel_size = baseOriginPts.size();
    Eigen::Vector3d pt_trans, new_center;
    Eigen::Matrix3d new_A;

    for (size_t i = 0; i < voxel_size; i++) {
      new_center.setZero();
      new_A.setZero();
      for (size_t j = 0; j < baseWinNums[i]->size(); j++) {
        pt_trans = poses_[(*baseWinNums[i])[j]] * (*baseOriginPts[i])[j] +
                   ts_[(*baseWinNums[i])[j]];
        new_A += pt_trans * pt_trans.transpose();
        new_center += pt_trans;
      }

      size_t refpts_size = 0;
      for (int k = 0; k < ref_size; k++) {
        for (size_t j = 0; j < refWinNums[k][i]->size(); j++) {
          pt_trans = refposes_[k] * (*refOriginPts[k][i])[j] + refts_[k];
          pt_trans = poses_[(*refWinNums[k][i])[j]] * pt_trans +
                     ts_[(*refWinNums[k][i])[j]];
          new_A += pt_trans * pt_trans.transpose();
          new_center += pt_trans;
        }
        refpts_size += refOriginPts[k][i]->size();
      }
      size_t pt_size = baseOriginPts[i]->size() + refpts_size;
      new_center /= pt_size;
      new_A /= pt_size;
      new_A -= new_center * new_center.transpose();
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(new_A);
      Eigen::Vector3d eigen_value = saes.eigenvalues();
      residual += eigen_value(0);
    }
  }
};

class OCTO_TREE {
 public:
  std::vector<vector_vec3d*> baseOriginPc;
  std::vector<vector_vec3d*> baseTransPc;
  // store the point cloud collected at pose i
  std::vector<std::vector<vector_vec3d*>> refOriginPc;
  std::vector<std::vector<vector_vec3d*>> refTransPc;
  OCTO_TREE* leaves[8];

  int win_size, ref_size, points_size, layer;
  OT_STATE octo_state;

  double voxel_center[3];
  double quater_length, eigen_ratio;
  Eigen::Vector3d value_vector;

  OCTO_TREE(int window_size, double ref_lidar_size, double eigen_limit)
      : win_size(window_size),
        ref_size(ref_lidar_size),
        eigen_ratio(eigen_limit) {
    octo_state = UNKNOWN;
    layer = 0;

    for (int i = 0; i < 8; i++) {
      leaves[i] = nullptr;
    }

    for (int i = 0; i < win_size; i++) {
      baseOriginPc.emplace_back(new vector_vec3d());
      baseTransPc.emplace_back(new vector_vec3d());
    }

    for (int j = 0; j < ref_size; j++) {
      std::vector<vector_vec3d*> refOriginPc_;
      std::vector<vector_vec3d*> refTransPc_;
      for (int i = 0; i < win_size; i++) {
        refOriginPc_.emplace_back(new vector_vec3d());
        refTransPc_.emplace_back(new vector_vec3d());
      }
      refOriginPc.emplace_back(refOriginPc_);
      refTransPc.emplace_back(refTransPc_);
    }
  }

  ~OCTO_TREE() {
    for (int i = 0; i < win_size; i++) {
      delete (baseOriginPc[i]);
      delete (baseTransPc[i]);
    }
    baseOriginPc.clear();
    baseTransPc.clear();
    for (int i = 0; i < ref_size; i++) {
      for (int j = 0; j < win_size; j++) {
        delete refOriginPc[i][j];
        delete refTransPc[i][j];
      }
      refOriginPc[i].clear();
      refTransPc[i].clear();
    }
    refOriginPc.clear();
    refTransPc.clear();

    for (int i = 0; i < 8; i++) {
      if (leaves[i] != nullptr) {
        delete leaves[i];
      }
    }
  }

  void recut() {
    if (octo_state == UNKNOWN) {
      points_size = 0;
      for (int i = 0; i < win_size; i++) {
        points_size += baseOriginPc[i]->size();
        for (int j = 0; j < ref_size; j++) {
          points_size += refOriginPc[j][i]->size();
        }
      }

      if (points_size < MIN_PS) {
        octo_state = MID_NODE;
        return;
      }

      if (judge_eigen()) {
        octo_state = PLANE;
        return;
      } else {
        if (layer == LAYER_LIMIT) {
          octo_state = MID_NODE;
          return;
        }

        for (int i = 0; i < win_size; i++) {
          uint pt_size = baseTransPc[i]->size();

          for (size_t j = 0; j < pt_size; j++) {
            int xyz[3] = {0, 0, 0};
            for (size_t k = 0; k < 3; k++) {
              if ((*baseTransPc[i])[j][k] > voxel_center[k]) {
                xyz[k] = 1;
              }
            }

            int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

            if (leaves[leafnum] == nullptr) {
              leaves[leafnum] = new OCTO_TREE(win_size, ref_size, eigen_ratio);
              leaves[leafnum]->voxel_center[0] =
                  voxel_center[0] + (2 * xyz[0] - 1) * quater_length;
              leaves[leafnum]->voxel_center[1] =
                  voxel_center[1] + (2 * xyz[1] - 1) * quater_length;
              leaves[leafnum]->voxel_center[2] =
                  voxel_center[2] + (2 * xyz[2] - 1) * quater_length;
              leaves[leafnum]->quater_length = quater_length / 2;
              leaves[leafnum]->layer = layer + 1;
            }
            leaves[leafnum]->baseOriginPc[i]->emplace_back(
                (*baseOriginPc[i])[j]);
            leaves[leafnum]->baseTransPc[i]->emplace_back((*baseTransPc[i])[j]);
          }

          for (int k = 0; k < ref_size; k++) {
            pt_size = refTransPc[k][i]->size();
            for (size_t j = 0; j < pt_size; j++) {
              int xyz[3] = {0, 0, 0};
              for (size_t a = 0; a < 3; a++)
                if ((*refTransPc[k][i])[j][a] > voxel_center[a]) xyz[a] = 1;
              int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
              if (leaves[leafnum] == nullptr) {
                leaves[leafnum] =
                    new OCTO_TREE(win_size, ref_size, eigen_ratio);
                leaves[leafnum]->voxel_center[0] =
                    voxel_center[0] + (2 * xyz[0] - 1) * quater_length;
                leaves[leafnum]->voxel_center[1] =
                    voxel_center[1] + (2 * xyz[1] - 1) * quater_length;
                leaves[leafnum]->voxel_center[2] =
                    voxel_center[2] + (2 * xyz[2] - 1) * quater_length;
                leaves[leafnum]->quater_length = quater_length / 2;
                leaves[leafnum]->layer = layer + 1;
              }
              leaves[leafnum]->refOriginPc[k][i]->emplace_back(
                  (*refOriginPc[k][i])[j]);
              leaves[leafnum]->refTransPc[k][i]->emplace_back(
                  (*refTransPc[k][i])[j]);
            }
          }
        }
      }
    }

    for (size_t i = 0; i < 8; i++)
      if (leaves[i] != nullptr) leaves[i]->recut();
  }

  bool judge_eigen() {
    Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
    Eigen::Vector3d center(0, 0, 0);

    uint pt_size;
    for (int i = 0; i < win_size; i++) {
      pt_size = baseTransPc[i]->size();
      for (size_t j = 0; j < pt_size; j++) {
        covMat += (*baseTransPc[i])[j] * (*baseTransPc[i])[j].transpose();
        center += (*baseTransPc[i])[j];
      }
      for (int k = 0; k < ref_size; k++) {
        pt_size = refTransPc[k][i]->size();
        for (size_t j = 0; j < pt_size; j++) {
          covMat += (*refTransPc[k][i])[j] * (*refTransPc[k][i])[j].transpose();
          center += (*refTransPc[k][i])[j];
        }
      }
    }
    center /= points_size;
    covMat = covMat / points_size - center * center.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
    value_vector = saes.eigenvalues();
    if (eigen_ratio < saes.eigenvalues()[2] / saes.eigenvalues()[0])
      return true;
    return false;
  }

  void feed_pt(EXTRIN_OPTIMIZER& lm_opt) {
    if (octo_state == PLANE) {
      lm_opt.push_voxel(baseOriginPc, refOriginPc);
    } else {
      for (int i = 0; i < 8; i++) {
        if (leaves[i] != nullptr) leaves[i]->feed_pt(lm_opt);
      }
    }
  }
};

#endif
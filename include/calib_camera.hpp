#ifndef CALIB_CAMERA_HPP
#define CALIB_CAMERA_HPP

#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Header.h>
#include <stdio.h>
#include <time.h>

#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <unordered_map>

#include "BA/ba.hpp"
#include "BA/mypcl.hpp"
#include "BA/tools.hpp"
#include "common.h"

class LiDAR {
 public:
  int lidar_number;
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> lidar_clouds;
};

class Camera {
 public:
  float fx_, fy_, cx_, cy_, k1_, k2_, p1_, p2_, k3_, k4_, s_;
  int width_, height_;
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  cv::Mat init_ext_;
  Eigen::Matrix3d ext_R;  // 初始旋转矩阵
  Eigen::Vector3d ext_t;  // 初始平移向量
  std::vector<cv::Mat> rgb_imgs;
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> rgb_edge_clouds;

  void update_Rt(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    ext_R << R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1), R(1, 2), R(2, 0),
        R(2, 1), R(2, 2);
    ext_t << t(0), t(1), t(2);
  }
};

class Calibration {
 public:
  ros::NodeHandle _nh;

  // ROS
  ros::Publisher pub_plane =
      _nh.advertise<sensor_msgs::PointCloud2>("/voxel_plane", 100);
  ros::Publisher pub_edge =
      _nh.advertise<sensor_msgs::PointCloud2>("/lidar_edge", 100);
  ros::Publisher pub_color =
      _nh.advertise<sensor_msgs::PointCloud2>("/color_cloud", 100);

  // Camera Settings
  std::vector<Camera> cams;

  // LiDAR Settings
  std::string lidar_path;
  int total_ext_number, base_number;
  std::vector<mypcl::pose> base_poses, extrinsics;
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>
      base_clouds;  // base LiDAR clouds local frame
  std::vector<LiDAR> ext_lidars;
  pcl::PointCloud<pcl::PointXYZI>::Ptr
      lidar_edge_clouds;  // 存储平面交接点云 global frame
  std::vector<int> lidar_edge_numbers;

  float voxel_size_;
  float eigen_ratio_;
  float down_sample_size_;
  float ransac_dis_threshold_;
  float plane_size_threshold_;
  float theta_min_;
  float theta_max_;
  float direction_theta_min_;
  float direction_theta_max_;
  float line_dis_threshold_;
  float min_line_dis_threshold_;
  float max_line_dis_threshold_;

  int edge_number_ = 0;
  int plane_max_size_ = 5;
  int rgb_canny_threshold_ = 20;
  int rgb_edge_minLen_ = 100;

  bool use_adaptive_voxel = true;

  Calibration(const std::vector<std::string>& CamCfgPaths,
              const std::string& LidarDataPath, const std::string& CalibCfgFile,
              bool use_ada_voxel) {
    use_adaptive_voxel = use_ada_voxel;
    loadCameraConfig(cams, CamCfgPaths);
    loadCalibConfig(CalibCfgFile, LidarDataPath);
    loadImgAndPointcloud();

    time_t t1 = clock();
    if (use_ada_voxel) {
      std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
      cut_voxel(surf_map, voxel_size_, eigen_ratio_);
      for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        iter->second->recut();
      estimate_edge(surf_map);
      ROS_INFO_STREAM("Adaptive voxel sucess!");
      time_t t2 = clock();
      LOG(INFO) << "adaptive time:" << (double)(t2 - t1) / (CLOCKS_PER_SEC)
                << "s";
    } else {
      std::unordered_map<VOXEL_LOC, Voxel*> voxel_map;
      initVoxel(voxel_size_, voxel_map);
      ROS_INFO_STREAM("Init voxel sucess!");
      LiDAREdgeExtraction(voxel_map, ransac_dis_threshold_,
                          plane_size_threshold_, lidar_edge_clouds);
      time_t t3 = clock();
      LOG(INFO) << "voxel time:" << (double)(t3 - t1) / (CLOCKS_PER_SEC);
    }
    LOG(INFO) << "lidar edge size:" << lidar_edge_clouds->size();

    for (size_t i = 0; i < cams.size(); i++)
      for (size_t j = 0; j < cams[i].rgb_imgs.size(); j++)
        if (!cams[i].rgb_imgs[j].data) {
          std::string msg = "Can not load image from " + lidar_path;
          ROS_ERROR_STREAM(msg.c_str());
          exit(-1);
        }
    ROS_INFO_STREAM("Load all data!");

    for (size_t i = 0; i < cams.size(); i++) {
      std::vector<cv::Mat> grey_imgs, rgb_edge_imgs;
      grey_imgs.resize(base_poses.size());
      for (size_t j = 0; j < cams[i].rgb_imgs.size(); j++)
        cv::cvtColor(cams[i].rgb_imgs[j], grey_imgs[j], cv::COLOR_BGR2GRAY);
      edgeDetector(rgb_canny_threshold_, rgb_edge_minLen_, grey_imgs,
                   rgb_edge_imgs, cams[i].rgb_edge_clouds);
    }
    ROS_INFO_STREAM("Initialization complete");
  }

  bool loadCameraConfig(std::vector<Camera>& cams,
                        const std::vector<std::string>& CamCfgPaths) {
    cams.resize(CamCfgPaths.size());
    for (size_t i = 0; i < CamCfgPaths.size(); i++) {
      cv::FileStorage fCamSet(CamCfgPaths[i], cv::FileStorage::READ);
      if (!fCamSet.isOpened()) {
        std::cerr << "Failed to open cams settings file at " << CamCfgPaths[i];
        exit(-1);
      }
      cams[i].width_ = fCamSet["Camera.width"];
      cams[i].height_ = fCamSet["Camera.height"];
      fCamSet["CameraMat"] >> cams[i].camera_matrix_;
      fCamSet["DistCoeffs"] >> cams[i].dist_coeffs_;
      cams[i].fx_ = cams[i].camera_matrix_.at<double>(0, 0);
      cams[i].s_ = cams[i].camera_matrix_.at<double>(0, 1);
      cams[i].cx_ = cams[i].camera_matrix_.at<double>(0, 2);
      cams[i].fy_ = cams[i].camera_matrix_.at<double>(1, 1);
      cams[i].cy_ = cams[i].camera_matrix_.at<double>(1, 2);
      cams[i].k1_ = cams[i].dist_coeffs_.at<double>(0, 0);
      cams[i].k2_ = cams[i].dist_coeffs_.at<double>(0, 1);
      cams[i].p1_ = cams[i].dist_coeffs_.at<double>(0, 2);
      cams[i].p2_ = cams[i].dist_coeffs_.at<double>(0, 3);
      cams[i].k3_ = cams[i].dist_coeffs_.at<double>(0, 4);
      fCamSet["ExtrinsicMat"] >> cams[i].init_ext_;
      cams[i].ext_R << cams[i].init_ext_.at<double>(0, 0),
          cams[i].init_ext_.at<double>(0, 1),
          cams[i].init_ext_.at<double>(0, 2),
          cams[i].init_ext_.at<double>(1, 0),
          cams[i].init_ext_.at<double>(1, 1),
          cams[i].init_ext_.at<double>(1, 2),
          cams[i].init_ext_.at<double>(2, 0),
          cams[i].init_ext_.at<double>(2, 1),
          cams[i].init_ext_.at<double>(2, 2);
      cams[i].ext_t << cams[i].init_ext_.at<double>(0, 3),
          cams[i].init_ext_.at<double>(1, 3),
          cams[i].init_ext_.at<double>(2, 3);
      ROS_INFO_STREAM("Camera " << i << " Configuration");
      LOG(INFO) << "Camera Matrix: " << cams[i].camera_matrix_;
      LOG(INFO) << "Distortion Coeffs: " << cams[i].dist_coeffs_;
      LOG(INFO) << "Extrinsic Params: " << cams[i].init_ext_;
    }
    return true;
  }

  bool loadCalibConfig(const std::string& config_file,
                       const std::string& lidar_data_path) {
    cv::FileStorage fSettings(config_file, cv::FileStorage::READ);
    if (!fSettings.isOpened()) {
      std::cerr << "Failed to open settings file at: " << config_file;
      exit(-1);
    }
    lidar_path = lidar_data_path;
    fSettings["ExtrinsicNumber"] >> total_ext_number;
    fSettings["BaseLiDARNumber"] >> base_number;

    LOG(INFO) << "extrinsic LiDAR number:" << total_ext_number;

    ext_lidars.resize(total_ext_number);
    fSettings["ExtLiDARNumber1"] >> ext_lidars[0].lidar_number;
    fSettings["ExtLiDARNumber2"] >> ext_lidars[1].lidar_number;
    if (total_ext_number > 2) {
      fSettings["ExtLiDARNumber3"] >> ext_lidars[2].lidar_number;
    }

    std::fstream file;
    std::string lidar_pose_file;
    std::string extrinsics_file;
    file.open(lidar_path + "pose.json");
    double tx, ty, tz, w, x, y, z;
    while (!file.eof()) {
      file >> tx >> ty >> tz >> w >> x >> y >> z;
      base_poses.emplace_back(mypcl::pose(Eigen::Quaterniond(w, x, y, z),
                                          Eigen::Vector3d(tx, ty, tz)));
    }
    file.close();

    file.open(lidar_path + "ref.json");
    while (!file.eof()) {
      file >> tx >> ty >> tz >> w >> x >> y >> z;
      extrinsics.emplace_back(mypcl::pose(Eigen::Quaterniond(w, x, y, z),
                                          Eigen::Vector3d(tx, ty, tz)));
    }
    file.close();

    ROS_INFO_STREAM("read " << base_poses.size() << " poses, "
                            << extrinsics.size() << " extrinsics");

    rgb_canny_threshold_ = fSettings["Canny.gray_threshold"];
    rgb_edge_minLen_ = fSettings["Canny.len_threshold"];
    voxel_size_ = fSettings["Voxel.size"];
    eigen_ratio_ = fSettings["Voxel.eigen_ratio"];
    plane_size_threshold_ = fSettings["Plane.min_points_size"];
    plane_max_size_ = fSettings["Plane.max_size"];
    ransac_dis_threshold_ = fSettings["Ransac.dis_threshold"];
    min_line_dis_threshold_ = fSettings["Edge.min_dis_threshold"];
    max_line_dis_threshold_ = fSettings["Edge.max_dis_threshold"];
    theta_min_ = fSettings["Plane.normal_theta_min"];
    theta_max_ = fSettings["Plane.normal_theta_max"];
    theta_min_ = cos(DEG2RAD(theta_min_));
    theta_max_ = cos(DEG2RAD(theta_max_));
    return true;
  }

  void loadImgAndPointcloud() {
    size_t pose_size = base_poses.size();
    base_clouds.resize(pose_size);
    for (size_t i = 0; i < pose_size; i++) {
      pcl::PointCloud<PointType>::Ptr pc_base(new pcl::PointCloud<PointType>);
      pcl::io::loadPCDFile(
          lidar_path + to_string(base_number) + "/" + to_string(i) + ".pcd",
          *pc_base);
      base_clouds[i] = pc_base;
    }

    for (int i = 0; i < total_ext_number; i++) {
      ext_lidars[i].lidar_clouds.resize(pose_size);
      for (size_t j = 0; j < pose_size; j++) {
        pcl::PointCloud<PointType>::Ptr pc_ref(new pcl::PointCloud<PointType>);
        pcl::io::loadPCDFile(lidar_path +
                                 to_string(ext_lidars[i].lidar_number) + "/" +
                                 to_string(j) + ".pcd",
                             *pc_ref);
        ext_lidars[i].lidar_clouds[j] = pc_ref;
      }
    }
    ROS_INFO_STREAM("Sucessfully load Point Cloud");

    for (size_t i = 0; i < cams.size(); i++)
      for (size_t j = 0; j < pose_size; j++) {
        cv::Mat img = cv::imread(
            lidar_path + "image/" + to_string(i) + "/" + to_string(j) + ".png",
            cv::IMREAD_COLOR);
        cams[i].rgb_imgs.emplace_back(img);
      }
    ROS_INFO_STREAM("Sucessfully load Image");
  }

  void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& feat_map,
                 double voxel_size, float eigen_ratio) {
    for (size_t i = 0; i < base_clouds.size(); i++) {
      float loc_xyz[3];
      for (pcl::PointXYZI& p_c : base_clouds[i]->points) {
        Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
        pvec_orig = base_poses[i].q * pvec_orig + base_poses[i].t;
        for (int j = 0; j < 3; j++) {
          loc_xyz[j] = pvec_orig[j] / voxel_size;
          if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
        }

        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                           (int64_t)loc_xyz[2]);
        auto iter = feat_map.find(position);
        if (iter != feat_map.end()) {
          iter->second->all_points.push_back(pvec_orig);
          iter->second->vec_orig.push_back(pvec_orig);
          iter->second->sig_orig.push(pvec_orig);
        } else {
          OCTO_TREE_ROOT* ot = new OCTO_TREE_ROOT(eigen_ratio);
          ot->all_points.push_back(pvec_orig);
          ot->vec_orig.push_back(pvec_orig);
          ot->sig_orig.push(pvec_orig);
          ot->voxel_center[0] = (0.5 + position.x) * voxel_size;
          ot->voxel_center[1] = (0.5 + position.y) * voxel_size;
          ot->voxel_center[2] = (0.5 + position.z) * voxel_size;
          ot->quater_length = voxel_size / 4.0;
          ot->layer = 0;
          feat_map[position] = ot;
        }
      }
    }
  }

  void estimate_edge(std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& surf_map) {
    ros::Rate loop(500);
    lidar_edge_clouds = pcl::PointCloud<pcl::PointXYZI>::Ptr(
        new pcl::PointCloud<pcl::PointXYZI>);
    for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++) {
      std::vector<Plane*> plane_list;
      std::vector<Plane*> merge_plane_list;
      iter->second->get_plane_list(plane_list);

      if (plane_list.size() > 1) {
        pcl::KdTreeFLANN<pcl::PointXYZI> kd_tree;
        pcl::PointCloud<pcl::PointXYZI> input_cloud;
        for (auto pv : iter->second->all_points) {
          pcl::PointXYZI p;
          p.x = pv(0);
          p.y = pv(1);
          p.z = pv(2);
          input_cloud.push_back(p);
        }
        kd_tree.setInputCloud(input_cloud.makeShared());
        mergePlane(plane_list, merge_plane_list);
        if (merge_plane_list.size() <= 1) continue;

        for (auto plane : merge_plane_list) {
          pcl::PointCloud<pcl::PointXYZRGB> color_cloud;
          std::vector<unsigned int> colors;
          colors.push_back(static_cast<unsigned int>(rand() % 255));
          colors.push_back(static_cast<unsigned int>(rand() % 255));
          colors.push_back(static_cast<unsigned int>(rand() % 255));
          for (auto pv : plane->plane_points) {
            pcl::PointXYZRGB pi;
            pi.x = pv[0];
            pi.y = pv[1];
            pi.z = pv[2];
            pi.r = colors[0];
            pi.g = colors[1];
            pi.b = colors[2];
            color_cloud.points.push_back(pi);
          }
          sensor_msgs::PointCloud2 dbg_msg;
          pcl::toROSMsg(color_cloud, dbg_msg);
          dbg_msg.header.frame_id = "camera_init";
          pub_plane.publish(dbg_msg);
          loop.sleep();
        }

        for (size_t p1_index = 0; p1_index < merge_plane_list.size() - 1;
             p1_index++)
          for (size_t p2_index = p1_index + 1;
               p2_index < merge_plane_list.size(); p2_index++) {
            std::vector<Eigen::Vector3d> line_point;
            projectLine(merge_plane_list[p1_index], merge_plane_list[p2_index],
                        line_point);

            if (line_point.size() == 0) break;

            pcl::PointCloud<pcl::PointXYZI> line_cloud;

            for (size_t j = 0; j < line_point.size(); j++) {
              pcl::PointXYZI p;
              p.x = line_point[j][0];
              p.y = line_point[j][1];
              p.z = line_point[j][2];

              int K = 5;
              // 创建两个向量，分别存放近邻的索引值、近邻的中心距
              std::vector<int> pointIdxNKNSearch(K);
              std::vector<float> pointNKNSquaredDistance(K);
              if (kd_tree.nearestKSearch(p, K, pointIdxNKNSearch,
                                         pointNKNSquaredDistance) == K) {
                Eigen::Vector3d tmp(
                    input_cloud.points[pointIdxNKNSearch[K - 1]].x,
                    input_cloud.points[pointIdxNKNSearch[K - 1]].y,
                    input_cloud.points[pointIdxNKNSearch[K - 1]].z);
                // if(pointNKNSquaredDistance[K-1] < 0.01)
                if ((tmp - line_point[j]).norm() < 0.05) {
                  line_cloud.points.push_back(p);
                  lidar_edge_clouds->points.push_back(p);
                }
              }
            }
            sensor_msgs::PointCloud2 dbg_msg;
            pcl::toROSMsg(line_cloud, dbg_msg);
            dbg_msg.header.frame_id = "camera_init";
            pub_edge.publish(dbg_msg);
            loop.sleep();
          }
      }
    }
  }

  void mergePlane(std::vector<Plane*>& origin_list,
                  std::vector<Plane*>& merge_list) {
    for (size_t i = 0; i < origin_list.size(); i++) origin_list[i]->id = 0;

    int current_id = 1;
    for (auto iter = origin_list.end() - 1; iter != origin_list.begin();
         iter--) {
      for (auto iter2 = origin_list.begin(); iter2 != iter; iter2++) {
        Eigen::Vector3d normal_diff = (*iter)->normal - (*iter2)->normal;
        Eigen::Vector3d normal_add = (*iter)->normal + (*iter2)->normal;
        double dis1 =
            fabs((*iter)->normal(0) * (*iter2)->center(0) +
                 (*iter)->normal(1) * (*iter2)->center(1) +
                 (*iter)->normal(2) * (*iter2)->center(2) + (*iter)->d);
        double dis2 =
            fabs((*iter2)->normal(0) * (*iter)->center(0) +
                 (*iter2)->normal(1) * (*iter)->center(1) +
                 (*iter2)->normal(2) * (*iter)->center(2) + (*iter2)->d);
        if (normal_diff.norm() < 0.2 || normal_add.norm() < 0.2)
          if (dis1 < 0.05 && dis2 < 0.05) {
            if ((*iter)->id == 0 && (*iter2)->id == 0) {
              (*iter)->id = current_id;
              (*iter2)->id = current_id;
              current_id++;
            } else if ((*iter)->id == 0 && (*iter2)->id != 0)
              (*iter)->id = (*iter2)->id;
            else if ((*iter)->id != 0 && (*iter2)->id == 0)
              (*iter2)->id = (*iter)->id;
          }
      }
    }
    std::vector<int> merge_flag;
    for (size_t i = 0; i < origin_list.size(); i++) {
      auto it =
          std::find(merge_flag.begin(), merge_flag.end(), origin_list[i]->id);
      if (it != merge_flag.end()) continue;

      if (origin_list[i]->id == 0) {
        merge_list.push_back(origin_list[i]);
        continue;
      }
      Plane* merge_plane = new Plane;
      (*merge_plane) = (*origin_list[i]);
      for (size_t j = 0; j < origin_list.size(); j++) {
        if (i == j) continue;
        if (origin_list[i]->id != 0)
          if (origin_list[j]->id == origin_list[i]->id)
            for (auto pv : origin_list[j]->plane_points)
              merge_plane->plane_points.push_back(pv);
      }
      merge_plane->covariance = Eigen::Matrix3d::Zero();
      merge_plane->center = Eigen::Vector3d::Zero();
      merge_plane->normal = Eigen::Vector3d::Zero();
      merge_plane->points_size = merge_plane->plane_points.size();
      merge_plane->radius = 0;
      for (auto pv : merge_plane->plane_points) {
        merge_plane->covariance += pv * pv.transpose();
        merge_plane->center += pv;
      }
      merge_plane->center = merge_plane->center / merge_plane->points_size;
      merge_plane->covariance =
          merge_plane->covariance / merge_plane->points_size -
          merge_plane->center * merge_plane->center.transpose();
      Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance);
      Eigen::Matrix3cd evecs = es.eigenvectors();
      Eigen::Vector3cd evals = es.eigenvalues();
      Eigen::Vector3d evalsReal;
      evalsReal = evals.real();
      Eigen::Matrix3f::Index evalsMin, evalsMax;
      evalsReal.rowwise().sum().minCoeff(&evalsMin);
      evalsReal.rowwise().sum().maxCoeff(&evalsMax);
      merge_plane->id = origin_list[i]->id;
      merge_plane->normal << evecs.real()(0, evalsMin),
          evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
      merge_plane->min_eigen_value = evalsReal(evalsMin);
      merge_plane->radius = sqrt(evalsReal(evalsMax));
      merge_plane->d = -(merge_plane->normal(0) * merge_plane->center(0) +
                         merge_plane->normal(1) * merge_plane->center(1) +
                         merge_plane->normal(2) * merge_plane->center(2));
      merge_plane->p_center.x = merge_plane->center(0);
      merge_plane->p_center.y = merge_plane->center(1);
      merge_plane->p_center.z = merge_plane->center(2);
      merge_plane->p_center.normal_x = merge_plane->normal(0);
      merge_plane->p_center.normal_y = merge_plane->normal(1);
      merge_plane->p_center.normal_z = merge_plane->normal(2);
      merge_plane->is_plane = true;
      merge_flag.push_back(merge_plane->id);
      merge_list.push_back(merge_plane);
    }
  }

  void initVoxel(const float voxel_size,
                 std::unordered_map<VOXEL_LOC, Voxel*>& voxel_map) {
    ROS_INFO_STREAM("Building Voxel");
    for (size_t a = 0; a < base_clouds.size(); a++) {
      for (size_t i = 0; i < base_clouds[a]->size(); i++) {
        const pcl::PointXYZI& p_t = base_clouds[a]->points[i];
        Eigen::Vector3d pt(p_t.x, p_t.y, p_t.z);
        pt = base_poses[a].q * pt + base_poses[a].t;
        pcl::PointXYZI p_c;
        p_c.x = pt(0);
        p_c.y = pt(1);
        p_c.z = pt(2);
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
          loc_xyz[j] = p_c.data[j] / voxel_size;
          if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
        }
        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                           (int64_t)loc_xyz[2]);
        auto iter = voxel_map.find(position);
        if (iter != voxel_map.end())
          voxel_map[position]->cloud->push_back(p_c);
        else {
          Voxel* voxel = new Voxel(voxel_size);
          voxel_map[position] = voxel;
          voxel_map[position]->voxel_origin[0] = position.x * voxel_size;
          voxel_map[position]->voxel_origin[1] = position.y * voxel_size;
          voxel_map[position]->voxel_origin[2] = position.z * voxel_size;
          voxel_map[position]->cloud->push_back(p_c);
        }
      }
      for (size_t b = 0; b < ext_lidars.size(); b++)
        for (size_t i = 0; i < ext_lidars[b].lidar_clouds[a]->size(); i++) {
          const pcl::PointXYZI& p_t = ext_lidars[b].lidar_clouds[a]->points[i];
          Eigen::Vector3d pt(p_t.x, p_t.y, p_t.z);
          pt = extrinsics[b].q * pt + extrinsics[b].t;
          pt = base_poses[a].q * pt + base_poses[a].t;
          pcl::PointXYZI p_c;
          p_c.x = pt(0);
          p_c.y = pt(1);
          p_c.z = pt(2);
          float loc_xyz[3];
          for (int j = 0; j < 3; j++) {
            loc_xyz[j] = p_c.data[j] / voxel_size;
            if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
          }
          VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                             (int64_t)loc_xyz[2]);
          auto iter = voxel_map.find(position);
          if (iter != voxel_map.end())
            voxel_map[position]->cloud->push_back(p_c);
          else {
            Voxel* voxel = new Voxel(voxel_size);
            voxel_map[position] = voxel;
            voxel_map[position]->voxel_origin[0] = position.x * voxel_size;
            voxel_map[position]->voxel_origin[1] = position.y * voxel_size;
            voxel_map[position]->voxel_origin[2] = position.z * voxel_size;
            voxel_map[position]->cloud->push_back(p_c);
          }
        }
      for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++)
        if (iter->second->cloud->size() > 20)
          downsample_voxel(*(iter->second->cloud), 0.03);
    }
  }

  void LiDAREdgeExtraction(
      const std::unordered_map<VOXEL_LOC, Voxel*>& voxel_map,
      const float ransac_dis_thre, const int plane_size_threshold,
      pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_edge_clouds_3d) {
    ROS_INFO_STREAM("Extracting Lidar Edge");
    ros::Rate loop(5000);
    lidar_edge_clouds_3d = pcl::PointCloud<pcl::PointXYZI>::Ptr(
        new pcl::PointCloud<pcl::PointXYZI>);
    for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
      if (iter->second->cloud->size() > 50) {
        std::vector<SinglePlane> plane_lists;
        // 创建一个体素滤波器
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filter(
            new pcl::PointCloud<pcl::PointXYZI>);
        pcl::copyPointCloud(*iter->second->cloud, *cloud_filter);
        //创建一个模型参数对象，用于记录结果
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        // inliers表示误差能容忍的点，记录点云序号
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        //创建一个分割器
        pcl::SACSegmentation<pcl::PointXYZI> seg;
        // Optional,设置结果平面展示的点是分割掉的点还是分割剩下的点
        seg.setOptimizeCoefficients(true);
        // Mandatory-设置目标几何形状
        seg.setModelType(pcl::SACMODEL_PLANE);
        //分割方法：随机采样法
        seg.setMethodType(pcl::SAC_RANSAC);
        //设置误差容忍范围，也就是阈值
        seg.setDistanceThreshold(ransac_dis_thre);
        pcl::PointCloud<pcl::PointXYZRGB> color_planner_cloud;
        int plane_index = 0;
        while (cloud_filter->points.size() > 10) {
          pcl::PointCloud<pcl::PointXYZI> planner_cloud;
          pcl::ExtractIndices<pcl::PointXYZI> extract;
          //输入点云
          seg.setInputCloud(cloud_filter);
          seg.setMaxIterations(500);
          //分割点云
          seg.segment(*inliers, *coefficients);
          if (inliers->indices.size() == 0) {
            ROS_INFO_STREAM(
                "Could not estimate a planner model for the given dataset");
            break;
          }
          extract.setIndices(inliers);
          extract.setInputCloud(cloud_filter);
          extract.filter(planner_cloud);

          if (planner_cloud.size() > plane_size_threshold) {
            pcl::PointCloud<pcl::PointXYZRGB> color_cloud;
            std::vector<unsigned int> colors;
            colors.push_back(static_cast<unsigned int>(rand() % 256));
            colors.push_back(static_cast<unsigned int>(rand() % 256));
            colors.push_back(static_cast<unsigned int>(rand() % 256));
            pcl::PointXYZ p_center(0, 0, 0);
            for (size_t i = 0; i < planner_cloud.points.size(); i++) {
              pcl::PointXYZRGB p;
              p.x = planner_cloud.points[i].x;
              p.y = planner_cloud.points[i].y;
              p.z = planner_cloud.points[i].z;
              p_center.x += p.x;
              p_center.y += p.y;
              p_center.z += p.z;
              p.r = colors[0];
              p.g = colors[1];
              p.b = colors[2];
              color_cloud.push_back(p);
              color_planner_cloud.push_back(p);
            }
            p_center.x = p_center.x / planner_cloud.size();
            p_center.y = p_center.y / planner_cloud.size();
            p_center.z = p_center.z / planner_cloud.size();
            SinglePlane single_plane;
            single_plane.cloud = planner_cloud;
            single_plane.p_center = p_center;
            single_plane.normal << coefficients->values[0],
                coefficients->values[1], coefficients->values[2];
            single_plane.index = plane_index;
            plane_lists.push_back(single_plane);
            plane_index++;
          }
          extract.setNegative(true);
          pcl::PointCloud<pcl::PointXYZI> cloud_f;
          extract.filter(cloud_f);
          *cloud_filter = cloud_f;
        }
        if (plane_lists.size() >= 1) {
          sensor_msgs::PointCloud2 dbg_msg;
          pcl::toROSMsg(color_planner_cloud, dbg_msg);
          dbg_msg.header.frame_id = "camera_init";
          pub_plane.publish(dbg_msg);
          loop.sleep();
        }
        std::vector<pcl::PointCloud<pcl::PointXYZI>> edge_cloud_lists;
        calcLine(plane_lists, voxel_size_, iter->second->voxel_origin,
                 edge_cloud_lists);
        if (edge_cloud_lists.size() > 0 && edge_cloud_lists.size() <= 5)
          for (size_t a = 0; a < edge_cloud_lists.size(); a++) {
            for (size_t i = 0; i < edge_cloud_lists[a].size(); i++) {
              pcl::PointXYZI p = edge_cloud_lists[a].points[i];
              lidar_edge_clouds->points.push_back(p);
              lidar_edge_numbers.push_back(edge_number_);
            }
            sensor_msgs::PointCloud2 dbg_msg;
            pcl::toROSMsg(edge_cloud_lists[a], dbg_msg);
            dbg_msg.header.frame_id = "camera_init";
            pub_edge.publish(dbg_msg);
            loop.sleep();
            edge_number_++;
          }
      }
    }
  }

  void projectLine(const Plane* plane1, const Plane* plane2,
                   std::vector<Eigen::Vector3d>& line_point) {
    float theta = plane1->normal.dot(plane2->normal);
    if (!(theta > theta_max_ && theta < theta_min_)) return;

    Eigen::Vector3d c1 = plane1->center;
    Eigen::Vector3d c2 = plane2->center;
    Eigen::Vector3d n1 = plane1->normal;
    Eigen::Vector3d n2 = plane2->normal;

    Eigen::Matrix3d A;
    Eigen::Vector3d d = n1.cross(n2).normalized();
    A.row(0) = n1.transpose();
    A.row(1) = d.transpose();
    A.row(2) = n2.transpose();
    Eigen::Vector3d b(n1.dot(c1), d.dot(c1), n2.dot(c2));
    Eigen::Vector3d O = A.colPivHouseholderQr().solve(b);

    double c1_to_line = (c1 - O).norm();
    double c2_to_line = ((c2 - O) - (c2 - O).dot(d) * d).norm();

    if (c1_to_line / c2_to_line > 8 || c2_to_line / c1_to_line > 8) return;

    if (plane1->points_size < plane2->points_size)
      for (auto pt : plane1->plane_points) {
        Eigen::Vector3d p = (pt - O).dot(d) * d + O;
        line_point.push_back(p);
      }
    else
      for (auto pt : plane2->plane_points) {
        Eigen::Vector3d p = (pt - O).dot(d) * d + O;
        line_point.push_back(p);
      }

    return;
  }

  void calcLine(
      const std::vector<SinglePlane>& plane_lists, const double voxel_size,
      const Eigen::Vector3d origin,
      std::vector<pcl::PointCloud<pcl::PointXYZI>>& edge_cloud_lists) {
    if (plane_lists.size() >= 2 && plane_lists.size() <= plane_max_size_) {
      pcl::PointCloud<pcl::PointXYZI> temp_line_cloud;
      for (size_t plane_idx1 = 0; plane_idx1 < plane_lists.size() - 1;
           plane_idx1++)
        for (size_t plane_idx2 = plane_idx1 + 1;
             plane_idx2 < plane_lists.size(); plane_idx2++) {
          float a1 = plane_lists[plane_idx1].normal[0];
          float b1 = plane_lists[plane_idx1].normal[1];
          float c1 = plane_lists[plane_idx1].normal[2];
          float x1 = plane_lists[plane_idx1].p_center.x;
          float y1 = plane_lists[plane_idx1].p_center.y;
          float z1 = plane_lists[plane_idx1].p_center.z;
          float a2 = plane_lists[plane_idx2].normal[0];
          float b2 = plane_lists[plane_idx2].normal[1];
          float c2 = plane_lists[plane_idx2].normal[2];
          float x2 = plane_lists[plane_idx2].p_center.x;
          float y2 = plane_lists[plane_idx2].p_center.y;
          float z2 = plane_lists[plane_idx2].p_center.z;
          float theta = a1 * a2 + b1 * b2 + c1 * c2;
          float point_dis_threshold = 0.00;
          if (theta > theta_max_ && theta < theta_min_) {
            if (plane_lists[plane_idx1].cloud.size() > 0 ||
                plane_lists[plane_idx2].cloud.size() > 0) {
              float matrix[4][5];
              matrix[1][1] = a1;
              matrix[1][2] = b1;
              matrix[1][3] = c1;
              matrix[1][4] = a1 * x1 + b1 * y1 + c1 * z1;
              matrix[2][1] = a2;
              matrix[2][2] = b2;
              matrix[2][3] = c2;
              matrix[2][4] = a2 * x2 + b2 * y2 + c2 * z2;

              std::vector<Eigen::Vector3d> points;
              Eigen::Vector3d point;
              matrix[3][1] = 1;
              matrix[3][2] = 0;
              matrix[3][3] = 0;
              matrix[3][4] = origin[0];
              calc<float>(matrix, point);
              if (point[0] >= origin[0] - point_dis_threshold &&
                  point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                  point[1] >= origin[1] - point_dis_threshold &&
                  point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                  point[2] >= origin[2] - point_dis_threshold &&
                  point[2] <= origin[2] + voxel_size + point_dis_threshold)
                points.push_back(point);

              matrix[3][1] = 0;
              matrix[3][2] = 1;
              matrix[3][3] = 0;
              matrix[3][4] = origin[1];
              calc<float>(matrix, point);
              if (point[0] >= origin[0] - point_dis_threshold &&
                  point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                  point[1] >= origin[1] - point_dis_threshold &&
                  point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                  point[2] >= origin[2] - point_dis_threshold &&
                  point[2] <= origin[2] + voxel_size + point_dis_threshold)
                points.push_back(point);

              matrix[3][1] = 0;
              matrix[3][2] = 0;
              matrix[3][3] = 1;
              matrix[3][4] = origin[2];
              calc<float>(matrix, point);
              if (point[0] >= origin[0] - point_dis_threshold &&
                  point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                  point[1] >= origin[1] - point_dis_threshold &&
                  point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                  point[2] >= origin[2] - point_dis_threshold &&
                  point[2] <= origin[2] + voxel_size + point_dis_threshold)
                points.push_back(point);

              matrix[3][1] = 1;
              matrix[3][2] = 0;
              matrix[3][3] = 0;
              matrix[3][4] = origin[0] + voxel_size;
              calc<float>(matrix, point);
              if (point[0] >= origin[0] - point_dis_threshold &&
                  point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                  point[1] >= origin[1] - point_dis_threshold &&
                  point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                  point[2] >= origin[2] - point_dis_threshold &&
                  point[2] <= origin[2] + voxel_size + point_dis_threshold)
                points.push_back(point);

              matrix[3][1] = 0;
              matrix[3][2] = 1;
              matrix[3][3] = 0;
              matrix[3][4] = origin[1] + voxel_size;
              calc<float>(matrix, point);
              if (point[0] >= origin[0] - point_dis_threshold &&
                  point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                  point[1] >= origin[1] - point_dis_threshold &&
                  point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                  point[2] >= origin[2] - point_dis_threshold &&
                  point[2] <= origin[2] + voxel_size + point_dis_threshold)
                points.push_back(point);

              matrix[3][1] = 0;
              matrix[3][2] = 0;
              matrix[3][3] = 1;
              matrix[3][4] = origin[2] + voxel_size;
              calc<float>(matrix, point);
              if (point[0] >= origin[0] - point_dis_threshold &&
                  point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                  point[1] >= origin[1] - point_dis_threshold &&
                  point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                  point[2] >= origin[2] - point_dis_threshold &&
                  point[2] <= origin[2] + voxel_size + point_dis_threshold)
                points.push_back(point);

              if (points.size() == 2) {
                pcl::PointCloud<pcl::PointXYZI> edge_clouds;
                pcl::PointXYZ p1(points[0][0], points[0][1], points[0][2]);
                pcl::PointXYZ p2(points[1][0], points[1][1], points[1][2]);
                float length = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) +
                                    pow(p1.z - p2.z, 2));
                // 指定近邻个数
                int K = 1;
                // 创建两个向量，分别存放近邻的索引值、近邻的中心距
                std::vector<int> pointIdxNKNSearch1(K);
                std::vector<float> pointNKNSquaredDistance1(K);
                std::vector<int> pointIdxNKNSearch2(K);
                std::vector<float> pointNKNSquaredDistance2(K);
                pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree1(
                    new pcl::search::KdTree<pcl::PointXYZI>());
                pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree2(
                    new pcl::search::KdTree<pcl::PointXYZI>());
                kdtree1->setInputCloud(
                    plane_lists[plane_idx1].cloud.makeShared());
                kdtree2->setInputCloud(
                    plane_lists[plane_idx2].cloud.makeShared());
                for (float inc = 0; inc <= length; inc += 0.01) {
                  pcl::PointXYZI p;
                  p.x = p1.x + (p2.x - p1.x) * inc / length;
                  p.y = p1.y + (p2.y - p1.y) * inc / length;
                  p.z = p1.z + (p2.z - p1.z) * inc / length;
                  p.intensity = 100;
                  if ((kdtree1->nearestKSearch(p, K, pointIdxNKNSearch1,
                                               pointNKNSquaredDistance1) > 0) &&
                      (kdtree2->nearestKSearch(p, K, pointIdxNKNSearch2,
                                               pointNKNSquaredDistance2) > 0)) {
                    float dis1 =
                        pow(p.x - plane_lists[plane_idx1]
                                      .cloud.points[pointIdxNKNSearch1[0]]
                                      .x,
                            2) +
                        pow(p.y - plane_lists[plane_idx1]
                                      .cloud.points[pointIdxNKNSearch1[0]]
                                      .y,
                            2) +
                        pow(p.z - plane_lists[plane_idx1]
                                      .cloud.points[pointIdxNKNSearch1[0]]
                                      .z,
                            2);
                    float dis2 =
                        pow(p.x - plane_lists[plane_idx2]
                                      .cloud.points[pointIdxNKNSearch2[0]]
                                      .x,
                            2) +
                        pow(p.y - plane_lists[plane_idx2]
                                      .cloud.points[pointIdxNKNSearch2[0]]
                                      .y,
                            2) +
                        pow(p.z - plane_lists[plane_idx2]
                                      .cloud.points[pointIdxNKNSearch2[0]]
                                      .z,
                            2);
                    if ((dis1 < min_line_dis_threshold_ *
                                    min_line_dis_threshold_ &&
                         dis2 < max_line_dis_threshold_ *
                                    max_line_dis_threshold_) ||
                        ((dis1 < max_line_dis_threshold_ *
                                     max_line_dis_threshold_ &&
                          dis2 < min_line_dis_threshold_ *
                                     min_line_dis_threshold_)))
                      edge_clouds.push_back(p);
                  }
                }
                if (edge_clouds.size() > 30)
                  edge_cloud_lists.push_back(edge_clouds);
              }
            }
          }
        }
    }
  }

  void edgeDetector(
      const int& canny_threshold, const int& edge_threshold,
      const std::vector<cv::Mat>& src_img, std::vector<cv::Mat>& edge_img,
      std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& edge_clouds) {
    int gaussian_size = 5;
    edge_clouds.resize(src_img.size());
    for (size_t i = 0; i < src_img.size(); i++)
      edge_clouds[i] = pcl::PointCloud<pcl::PointXYZ>::Ptr(
          new pcl::PointCloud<pcl::PointXYZ>);
    edge_img.resize(src_img.size());
    for (size_t a = 0; a < src_img.size(); a++) {
      cv::GaussianBlur(src_img[a], src_img[a],
                       cv::Size(gaussian_size, gaussian_size), 0, 0);
      cv::Mat canny_result =
          cv::Mat::zeros(src_img[a].rows, src_img[a].cols, CV_8UC1);
      cv::Canny(src_img[a], canny_result, canny_threshold, canny_threshold * 3,
                3, true);
      std::vector<std::vector<cv::Point>> contours;
      std::vector<cv::Vec4i> hierarchy;
      cv::findContours(canny_result, contours, hierarchy, cv::RETR_EXTERNAL,
                       cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
      edge_img[a] = cv::Mat::zeros(src_img[a].rows, src_img[a].cols, CV_8UC1);

      for (size_t i = 0; i < contours.size(); i++)
        if (contours[i].size() > edge_threshold) {
          cv::Mat debug_img =
              cv::Mat::zeros(src_img[a].rows, src_img[a].cols, CV_8UC1);
          for (size_t j = 0; j < contours[i].size(); j++) {
            pcl::PointXYZ p;
            p.x = contours[i][j].x;
            p.y = -contours[i][j].y;
            p.z = 0;
            edge_img[a].at<uchar>(-p.y, p.x) = 255;
          }
        }
      for (int x = 0; x < edge_img[a].cols; x++)
        for (int y = 0; y < edge_img[a].rows; y++)
          if (edge_img[a].at<uchar>(y, x) == 255) {
            pcl::PointXYZ p;
            p.x = x;
            p.y = -y;
            p.z = 0;
            edge_clouds[a]->points.push_back(p);
          }
      edge_clouds[a]->width = edge_clouds[a]->points.size();
      edge_clouds[a]->height = 1;
    }
  }

  void buildVPnp(
      const Camera& cam, const Vector6d& extrinsic_params,
      const int dis_threshold, const bool show_residual,
      const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>&
          cam_edge_clouds_2d,
      const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_edge_clouds_3d,
      std::vector<VPnPData>& pnp_list) {
    pnp_list.clear();
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << cam.fx_, cam.s_, cam.cx_,
                             0.0, cam.fy_, cam.cy_, 0.0, 0.0, 1.0);
    cv::Mat distortion_coeff =
        (cv::Mat_<double>(1, 5) << cam.k1_, cam.k2_, cam.p1_, cam.p2_, cam.k3_);
    Eigen::AngleAxisd rotation_vector3;
    rotation_vector3 =
        Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());
    Eigen::Quaterniond q_(rotation_vector3);
    for (size_t a = 0; a < base_poses.size(); a += 1)  // for each camera pose
    {
      std::vector<std::vector<std::vector<pcl::PointXYZI>>> img_pts_container;
      for (int y = 0; y < cam.height_; y++) {
        std::vector<std::vector<pcl::PointXYZI>> row_pts_container;
        for (int x = 0; x < cam.width_; x++) {
          std::vector<pcl::PointXYZI> col_pts_container;
          row_pts_container.push_back(col_pts_container);
        }
        img_pts_container.push_back(row_pts_container);
      }
      std::vector<cv::Point3f> pts_3d;
      std::vector<cv::Point2f> pts_2d;
      cv::Mat r_vec =
          (cv::Mat_<double>(3, 1) << rotation_vector3.angle() *
                                         rotation_vector3.axis().transpose()[0],
           rotation_vector3.angle() * rotation_vector3.axis().transpose()[1],
           rotation_vector3.angle() * rotation_vector3.axis().transpose()[2]);
      Eigen::Vector3d t_(extrinsic_params[3], extrinsic_params[4],
                         extrinsic_params[5]);
      cv::Mat t_vec = (cv::Mat_<double>(3, 1) << t_(0), t_(1), t_(2));

      for (size_t i = 0; i < lidar_edge_clouds_3d->size(); i++) {
        pcl::PointXYZI point_3d = lidar_edge_clouds_3d->points[i];
        Eigen::Vector3d pt1(point_3d.x, point_3d.y, point_3d.z);
        Eigen::Vector3d pt2(0, 0, 1);
        Eigen::Vector3d pt;
        pt = base_poses[a].q.inverse() * (pt1 - base_poses[a].t);
        if (cos_angle(q_ * pt + t_, pt2) > 0.8)  // FoV check
          pts_3d.emplace_back(cv::Point3f(pt(0), pt(1), pt(2)));
      }
      cv::projectPoints(pts_3d, r_vec, t_vec, camera_matrix, distortion_coeff,
                        pts_2d);

      pcl::PointCloud<pcl::PointXYZ>::Ptr line_edge_cloud_2d(
          new pcl::PointCloud<pcl::PointXYZ>);
      std::vector<int> line_edge_cloud_2d_number;
      for (size_t i = 0; i < pts_2d.size(); i++) {
        pcl::PointXYZ p;
        p.x = pts_2d[i].x;
        p.y = -pts_2d[i].y;
        p.z = 0;
        pcl::PointXYZI pi_3d;
        pi_3d.x = pts_3d[i].x;
        pi_3d.y = pts_3d[i].y;
        pi_3d.z = pts_3d[i].z;
        pi_3d.intensity = 1;
        if (p.x > 0 && p.x < cam.width_ && pts_2d[i].y > 0 &&
            pts_2d[i].y < cam.height_) {
          if (img_pts_container[pts_2d[i].y][pts_2d[i].x].size() == 0) {
            line_edge_cloud_2d->points.push_back(p);
            img_pts_container[pts_2d[i].y][pts_2d[i].x].push_back(pi_3d);
          } else
            img_pts_container[pts_2d[i].y][pts_2d[i].x].push_back(pi_3d);
        }
      }
      if (show_residual)
        if (a == 16) {
          cv::Mat residual_img = getConnectImg(
              cam, dis_threshold, cam_edge_clouds_2d[a], line_edge_cloud_2d);
          std::string img_name = std::to_string(a);
          cv::imshow(img_name, residual_img);
          cv::waitKey(10);
        }

      pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_cam(
          new pcl::search::KdTree<pcl::PointXYZ>());
      pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_lidar(
          new pcl::search::KdTree<pcl::PointXYZ>());
      pcl::PointCloud<pcl::PointXYZ>::Ptr search_cloud =
          pcl::PointCloud<pcl::PointXYZ>::Ptr(
              new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud_cam =
          pcl::PointCloud<pcl::PointXYZ>::Ptr(
              new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud_lidar =
          pcl::PointCloud<pcl::PointXYZ>::Ptr(
              new pcl::PointCloud<pcl::PointXYZ>);
      kdtree_cam->setInputCloud(cam_edge_clouds_2d[a]);
      kdtree_lidar->setInputCloud(line_edge_cloud_2d);
      tree_cloud_cam = cam_edge_clouds_2d[a];
      tree_cloud_lidar = line_edge_cloud_2d;
      search_cloud = line_edge_cloud_2d;

      int K = 5;  // 指定近邻个数
      // 创建两个向量，分别存放近邻的索引值、近邻的中心距
      std::vector<int> pointIdxNKNSearch(K);
      std::vector<float> pointNKNSquaredDistance(K);
      std::vector<int> pointIdxNKNSearchLidar(K);
      std::vector<float> pointNKNSquaredDistanceLidar(K);
      std::vector<cv::Point2d> lidar_2d_list;
      std::vector<cv::Point2d> img_2d_list;
      std::vector<Eigen::Vector2d> camera_direction_list;
      std::vector<Eigen::Vector2d> lidar_direction_list;
      std::vector<int> lidar_2d_number;
      for (size_t i = 0; i < search_cloud->points.size(); i++) {
        pcl::PointXYZ searchPoint = search_cloud->points[i];
        kdtree_lidar->nearestKSearch(searchPoint, K, pointIdxNKNSearchLidar,
                                     pointNKNSquaredDistanceLidar);
        if (kdtree_cam->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                                       pointNKNSquaredDistance) > 0) {
          bool dis_check = true;
          for (int j = 0; j < K; j++) {
            float distance =
                sqrt(pow(searchPoint.x -
                             tree_cloud_cam->points[pointIdxNKNSearch[j]].x,
                         2) +
                     pow(searchPoint.y -
                             tree_cloud_cam->points[pointIdxNKNSearch[j]].y,
                         2));
            if (distance > dis_threshold) dis_check = false;
          }
          if (dis_check) {
            cv::Point p_l_2d(search_cloud->points[i].x,
                             -search_cloud->points[i].y);
            cv::Point p_c_2d(tree_cloud_cam->points[pointIdxNKNSearch[0]].x,
                             -tree_cloud_cam->points[pointIdxNKNSearch[0]].y);
            Eigen::Vector2d direction_cam(0, 0);
            std::vector<Eigen::Vector2d> points_cam;
            for (size_t i = 0; i < pointIdxNKNSearch.size(); i++) {
              Eigen::Vector2d p(
                  tree_cloud_cam->points[pointIdxNKNSearch[i]].x,
                  -tree_cloud_cam->points[pointIdxNKNSearch[i]].y);
              points_cam.push_back(p);
            }
            calcDirection(points_cam, direction_cam);
            Eigen::Vector2d direction_lidar(0, 0);
            std::vector<Eigen::Vector2d> points_lidar;
            for (size_t i = 0; i < pointIdxNKNSearch.size(); i++) {
              Eigen::Vector2d p(
                  tree_cloud_lidar->points[pointIdxNKNSearchLidar[i]].x,
                  -tree_cloud_lidar->points[pointIdxNKNSearchLidar[i]].y);
              points_lidar.push_back(p);
            }
            calcDirection(points_lidar, direction_lidar);
            if (p_l_2d.x > 0 && p_l_2d.x < cam.width_ && p_l_2d.y > 0 &&
                p_l_2d.y < cam.height_) {
              lidar_2d_list.push_back(p_l_2d);
              img_2d_list.push_back(p_c_2d);
              camera_direction_list.push_back(direction_cam);
              lidar_direction_list.push_back(direction_lidar);
            }
          }
        }
      }
      for (size_t i = 0; i < lidar_2d_list.size(); i++) {
        int y = lidar_2d_list[i].y;
        int x = lidar_2d_list[i].x;
        int pixel_points_size = img_pts_container[y][x].size();
        if (pixel_points_size > 0) {
          VPnPData pnp;
          pnp.x = 0;
          pnp.y = 0;
          pnp.z = 0;
          pnp.u = img_2d_list[i].x;
          pnp.v = img_2d_list[i].y;
          for (int j = 0; j < pixel_points_size; j++) {
            pnp.x += img_pts_container[y][x][j].x;
            pnp.y += img_pts_container[y][x][j].y;
            pnp.z += img_pts_container[y][x][j].z;
          }
          pnp.x = pnp.x / pixel_points_size;
          pnp.y = pnp.y / pixel_points_size;
          pnp.z = pnp.z / pixel_points_size;
          pnp.direction = camera_direction_list[i];
          pnp.direction_lidar = lidar_direction_list[i];
          pnp.number = 0;
          float theta = pnp.direction.dot(pnp.direction_lidar);
          if (theta > direction_theta_min_ || theta < direction_theta_max_)
            pnp_list.push_back(pnp);
        }
      }
    }
  }

  cv::Mat getConnectImg(
      const Camera& cam, const int dis_threshold,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& rgb_edge_cloud,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& depth_edge_cloud) {
    cv::Mat connect_img = cv::Mat::zeros(cam.height_, cam.width_, CV_8UC3);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_cam(
        new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr search_cloud =
        pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud_cam =
        pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    kdtree_cam->setInputCloud(rgb_edge_cloud);
    tree_cloud_cam = rgb_edge_cloud;
    for (size_t i = 0; i < depth_edge_cloud->points.size(); i++) {
      cv::Point2d p2(depth_edge_cloud->points[i].x,
                     -depth_edge_cloud->points[i].y);
      if (p2.x > 0 && p2.x < cam.width_ && p2.y > 0 && p2.y < cam.height_) {
        pcl::PointXYZ p = depth_edge_cloud->points[i];
        search_cloud->points.push_back(p);
      }
    }

    int line_count = 0;
    // 指定近邻个数
    int K = 1;
    // 创建两个向量，分别存放近邻的索引值、近邻的中心距
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    for (size_t i = 0; i < search_cloud->points.size(); i++) {
      pcl::PointXYZ searchPoint = search_cloud->points[i];
      if (kdtree_cam->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                                     pointNKNSquaredDistance) > 0) {
        for (int j = 0; j < K; j++) {
          float distance =
              sqrt(pow(searchPoint.x -
                           tree_cloud_cam->points[pointIdxNKNSearch[j]].x,
                       2) +
                   pow(searchPoint.y -
                           tree_cloud_cam->points[pointIdxNKNSearch[j]].y,
                       2));
          if (distance < dis_threshold) {
            cv::Scalar color = cv::Scalar(0, 255, 0);
            line_count++;
            if ((line_count % 3) == 0) {
              cv::line(
                  connect_img,
                  cv::Point(search_cloud->points[i].x,
                            -search_cloud->points[i].y),
                  cv::Point(tree_cloud_cam->points[pointIdxNKNSearch[j]].x,
                            -tree_cloud_cam->points[pointIdxNKNSearch[j]].y),
                  color, 1);
            }
          }
        }
      }
    }
    for (size_t i = 0; i < rgb_edge_cloud->size(); i++) {
      connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y,
                                rgb_edge_cloud->points[i].x)[0] = 255;
      connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y,
                                rgb_edge_cloud->points[i].x)[1] = 0;
      connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y,
                                rgb_edge_cloud->points[i].x)[2] = 0;
    }
    for (size_t i = 0; i < search_cloud->size(); i++) {
      connect_img.at<cv::Vec3b>(-search_cloud->points[i].y,
                                search_cloud->points[i].x)[0] = 0;
      connect_img.at<cv::Vec3b>(-search_cloud->points[i].y,
                                search_cloud->points[i].x)[1] = 0;
      connect_img.at<cv::Vec3b>(-search_cloud->points[i].y,
                                search_cloud->points[i].x)[2] = 255;
    }
    return connect_img;
  }

  void calcDirection(const std::vector<Eigen::Vector2d>& points,
                     Eigen::Vector2d& direction) {
    Eigen::Vector2d mean_point(0, 0);
    for (size_t i = 0; i < points.size(); i++) {
      mean_point(0) += points[i](0);
      mean_point(1) += points[i](1);
    }
    mean_point(0) = mean_point(0) / points.size();
    mean_point(1) = mean_point(1) / points.size();
    Eigen::Matrix2d S;
    S << 0, 0, 0, 0;
    for (size_t i = 0; i < points.size(); i++) {
      Eigen::Matrix2d s =
          (points[i] - mean_point) * (points[i] - mean_point).transpose();
      S += s;
    }
    Eigen::EigenSolver<Eigen::Matrix<double, 2, 2>> es(S);
    Eigen::MatrixXcd evecs = es.eigenvectors();
    Eigen::MatrixXcd evals = es.eigenvalues();
    Eigen::MatrixXd evalsReal;
    evalsReal = evals.real();
    Eigen::MatrixXf::Index evalsMax;
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);  //得到最大特征值的位置
    direction << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax);
  }

  void colorCloud(
      const Vector6d& extrinsic_params, const int density, const Camera& cam,
      const std::vector<cv::Mat>& rgb_imgs,
      const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& lidar_clouds) {
    Eigen::AngleAxisd rotation_vector3;
    rotation_vector3 =
        Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());
    Eigen::Quaterniond q_(rotation_vector3);
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << cam.fx_, cam.s_, cam.cx_,
                             0.0, cam.fy_, cam.cy_, 0.0, 0.0, 1.0);
    cv::Mat distortion_coeff =
        (cv::Mat_<double>(1, 5) << cam.k1_, cam.k2_, cam.p1_, cam.p2_, cam.k3_);
    cv::Mat r_vec =
        (cv::Mat_<double>(3, 1) << rotation_vector3.angle() *
                                       rotation_vector3.axis().transpose()[0],
         rotation_vector3.angle() * rotation_vector3.axis().transpose()[1],
         rotation_vector3.angle() * rotation_vector3.axis().transpose()[2]);
    cv::Mat t_vec = (cv::Mat_<double>(3, 1) << extrinsic_params[3],
                     extrinsic_params[4], extrinsic_params[5]);
    Eigen::Vector3d t_(extrinsic_params[3], extrinsic_params[4],
                       extrinsic_params[5]);

    for (size_t a = 0; a < rgb_imgs.size(); a++)  // camera poses
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud(
          new pcl::PointCloud<pcl::PointXYZRGB>);
      std::vector<cv::Point3f> pts_3d;
      for (size_t b = 0; b < base_poses.size(); b++) {
        for (size_t i = 0; i < lidar_clouds[b]->size(); i += density) {
          pcl::PointXYZI point = lidar_clouds[b]->points[i];
          Eigen::Vector3d pt1(point.x, point.y, point.z);
          pt1 = base_poses[b].q * pt1 + base_poses[b].t;
          pt1 = base_poses[a].q.inverse() * (pt1 - base_poses[a].t);
          Eigen::Vector3d pt2(0, 0, 1);
          if (cos_angle(q_ * pt1 + t_, pt2) > 0.8)  // FoV check
          {
            float depth =
                sqrt(pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2));
            if (depth > 2.5 && depth < 50)
              pts_3d.emplace_back(cv::Point3f(pt1(0), pt1(1), pt1(2)));
          }
        }
      }
      std::vector<cv::Point2f> pts_2d;
      cv::projectPoints(pts_3d, r_vec, t_vec, camera_matrix, distortion_coeff,
                        pts_2d);
      int image_rows = rgb_imgs[a].rows;
      int image_cols = rgb_imgs[a].cols;

      for (size_t i = 0; i < pts_2d.size(); i++) {
        if (pts_2d[i].x > 1 && pts_2d[i].x < image_cols - 1 &&
            pts_2d[i].y > 1 && pts_2d[i].y < image_rows - 1) {
          cv::Scalar color = rgb_imgs[a].at<cv::Vec3b>(pts_2d[i]);
          if (color[0] == 0 && color[1] == 0 && color[2] == 0) continue;
          if (pts_3d[i].x > 100) continue;
          Eigen::Vector3d pt(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z);
          pt = base_poses[a].q * pt + base_poses[a].t;
          pcl::PointXYZRGB p;
          p.x = pt(0);
          p.y = pt(1);
          p.z = pt(2);
          p.b = color[0];
          p.g = color[1];
          p.r = color[2];
          color_cloud->points.push_back(p);
        }
      }
      color_cloud->width = color_cloud->points.size();
      color_cloud->height = 1;
      sensor_msgs::PointCloud2 pub_cloud;
      pcl::toROSMsg(*color_cloud, pub_cloud);
      pub_cloud.header.frame_id = "camera_init";
      pub_color.publish(pub_cloud);
    }
  }

  void projection(const Vector6d& extrinsic_params, const Camera& cam,
                  const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_cloud,
                  cv::Mat& projection_img) {
    std::vector<cv::Point3f> pts_3d;
    std::vector<float> intensity_list;
    Eigen::AngleAxisd rotation_vector3;
    rotation_vector3 =
        Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());
    for (size_t i = 0; i < lidar_cloud->size(); i++) {
      pcl::PointXYZI point_3d = lidar_cloud->points[i];
      pts_3d.emplace_back(cv::Point3f(point_3d.x, point_3d.y, point_3d.z));
      intensity_list.emplace_back(lidar_cloud->points[i].intensity);
    }
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << cam.fx_, cam.s_, cam.cx_,
                             0.0, cam.fy_, cam.cy_, 0.0, 0.0, 1.0);
    cv::Mat distortion_coeff =
        (cv::Mat_<double>(1, 5) << cam.k1_, cam.k2_, cam.p1_, cam.p2_, cam.k3_);
    cv::Mat r_vec =
        (cv::Mat_<double>(3, 1) << rotation_vector3.angle() *
                                       rotation_vector3.axis().transpose()[0],
         rotation_vector3.angle() * rotation_vector3.axis().transpose()[1],
         rotation_vector3.angle() * rotation_vector3.axis().transpose()[2]);
    cv::Mat t_vec = (cv::Mat_<double>(3, 1) << extrinsic_params[3],
                     extrinsic_params[4], extrinsic_params[5]);
    // project 3d-points into image view
    std::vector<cv::Point2f> pts_2d;
    cv::projectPoints(pts_3d, r_vec, t_vec, camera_matrix, distortion_coeff,
                      pts_2d);
    cv::Mat image_project = cv::Mat::zeros(cam.height_, cam.width_, CV_16UC1);
    cv::Mat rgb_image_project =
        cv::Mat::zeros(cam.height_, cam.width_, CV_8UC3);
    for (size_t i = 0; i < pts_2d.size(); ++i) {
      cv::Point2f point_2d = pts_2d[i];
      if (point_2d.x <= 0 || point_2d.x >= cam.width_ || point_2d.y <= 0 ||
          point_2d.y >= cam.height_)
        continue;
      else {
        // test depth and intensity both
        float depth = sqrt(pow(pts_3d[i].x, 2) + pow(pts_3d[i].y, 2) +
                           pow(pts_3d[i].z, 2));
        if (depth >= 40) depth = 40;
        float grey = depth / 40 * 65535;
        image_project.at<ushort>(point_2d.y, point_2d.x) = grey;
      }
    }
    cv::Mat grey_image_projection;
    cv::cvtColor(rgb_image_project, grey_image_projection, cv::COLOR_BGR2GRAY);

    image_project.convertTo(image_project, CV_8UC1, 1 / 256.0);
    projection_img = image_project.clone();
  }

  cv::Mat getProjectionImg(const Vector6d& extrinsic_params,
                           const int camera_id, const int image_number) {
    cv::Mat depth_projection_img;
    Camera cam = cams[camera_id];
    pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_cloud(
        new pcl::PointCloud<PointType>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_cloud_(
        new pcl::PointCloud<PointType>);
    for (size_t a = 0; a < base_clouds.size(); a++) {
      pcl::PointCloud<pcl::PointXYZI>::Ptr tmp(new pcl::PointCloud<PointType>);
      mypcl::transform_pointcloud(*base_clouds[a], *tmp, base_poses[a].t,
                                  base_poses[a].q);
      lidar_cloud_ = mypcl::append_cloud(lidar_cloud_, *tmp);
    }
    Eigen::AngleAxisd rotation_vector3;
    rotation_vector3 =
        Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());
    Eigen::Quaterniond q_(rotation_vector3);
    Eigen::Vector3d t_(extrinsic_params[3], extrinsic_params[4],
                       extrinsic_params[5]);
    int cnt = 0;
    lidar_cloud->points.resize(5e6);
    for (size_t i = 0; i < lidar_cloud_->points.size(); i++) {
      pcl::PointXYZI point_3d = lidar_cloud_->points[i];
      Eigen::Vector3d pt1(point_3d.x, point_3d.y, point_3d.z);
      Eigen::Vector3d pt2(0, 0, 1);
      pt1 = base_poses[image_number].q.inverse() *
            (pt1 - base_poses[image_number].t);
      if (cos_angle(q_ * pt1 + t_, pt2) > 0.8) {
        lidar_cloud->points[cnt].x = pt1(0);
        lidar_cloud->points[cnt].y = pt1(1);
        lidar_cloud->points[cnt].z = pt1(2);
        lidar_cloud->points[cnt].intensity = lidar_cloud_->points[i].intensity;
        cnt++;
      }
    }
    LOG(INFO) << "lidar cloud size:" << lidar_cloud->size();
    lidar_cloud->points.resize(cnt);
    // downsample_voxel(*lidar_cloud, 0.03);
    projection(extrinsic_params, cam, lidar_cloud, depth_projection_img);
    cv::Mat map_img = cv::Mat::zeros(cam.height_, cam.width_, CV_8UC3);
    for (int x = 0; x < map_img.cols; x++) {
      for (int y = 0; y < map_img.rows; y++) {
        uint8_t r, g, b;
        float norm = depth_projection_img.at<uchar>(y, x) / 256.0;
        mapJet(norm, 0, 1, r, g, b);
        map_img.at<cv::Vec3b>(y, x)[0] = b;
        map_img.at<cv::Vec3b>(y, x)[1] = g;
        map_img.at<cv::Vec3b>(y, x)[2] = r;
      }
    }
    cv::Mat merge_img = 0.8 * map_img + 0.8 * cam.rgb_imgs[image_number];
    return merge_img;
  }
};

#endif
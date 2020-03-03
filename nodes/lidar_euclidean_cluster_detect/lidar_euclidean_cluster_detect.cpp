#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <numeric>
#include <iterator>

#include <ros/ros.h>
#include <pcl/common/impl/angles.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/conditional_removal.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/don.h>
#include <pcl/features/fpfh_omp.h>

#include <pcl/kdtree/kdtree.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include <pcl/common/common.h>

#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>

#include <pcl/segmentation/extract_clusters.h>

#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayLayout.h>
#include <std_msgs/MultiArrayDimension.h>

#include "autoware_msgs/Centroids.h"
#include "autoware_msgs/CloudCluster.h"
#include "autoware_msgs/CloudClusterArray.h"
#include "autoware_msgs/DetectedObject.h"
#include "autoware_msgs/DetectedObjectArray.h"

#include <vector_map/vector_map.h>

#include <tf/tf.h>

#include <yaml-cpp/yaml.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/version.hpp>
#include "munkres/munkres.h"
#include "munkres/adapters/boostmatrixadapter.h"
#include "spline.h"

#if (CV_MAJOR_VERSION == 3)

#include "gencolors.cpp"

#else

#include <opencv2/contrib/contrib.hpp>
#include <autoware_msgs/DetectedObjectArray.h>

#endif

#include "cluster.h"

#define GPU_CLUSTERING
#ifdef GPU_CLUSTERING

#include "gpu_euclidean_clustering.h"

#endif

#define __APP_NAME__ "euclidean_clustering"

using namespace cv;

ros::Publisher _pub_cluster_cloud;
ros::Publisher _pub_ground_cloud;
ros::Publisher _centroid_pub;

ros::Publisher _pub_clusters_message;
ros::Publisher _pub_leitbakes_cloud;
ros::Publisher _pub_points_lanes_cloud;

ros::Publisher _pub_detected_objects;

std_msgs::Header _pcl_header;

std::string _output_frame;


static double _radius_outlier_removal;
static int _neighbours_outlier_removal;
static bool _velodyne_transform_available;
static bool _downsample_cloud;
static bool _pose_estimation;
static double _leaf_size;
static int _cluster_size_min;
static int _cluster_size_max;
static const double _initial_quat_w = 1.0;
static double _ransac_height;
static double _ransac_angle;
static bool _remove_ground;  // only ground
static double _min_cluster_height;
static bool _using_sensor_cloud;
static bool _use_diffnormals;


static double _clip_min_height;
static double _clip_max_height;

static bool _keep_lanes;
static double _keep_lane_left_distance;
static double _keep_lane_right_distance;

static double _max_boundingbox_side;
static double _remove_points_upto;
static double _cluster_merge_threshold;
static double _clustering_distance;

static bool _use_gpu;
static std::chrono::system_clock::time_point _start, _end;

std::vector<std::vector<geometry_msgs::Point>> _way_area_points;
std::vector<cv::Scalar> _colors;
pcl::PointCloud<pcl::PointXYZ> _sensor_cloud;
visualization_msgs::Marker _visualization_marker;

static bool _use_multiple_thres;
std::vector<double> _clustering_distances;
std::vector<double> _clustering_ranges;

tf::StampedTransform *_transform;
tf::StampedTransform *_velodyne_output_transform;
tf::TransformListener *_transform_listener;
tf::TransformListener *_vectormap_transform_listener;

//image
static bool _simulation = true;
static bool _intensity_filter;
static bool _first_frame_batches = true;
static int _visual_clear_flag = false;
static int _birdview_scale = 50;
const int _img_buffer_size = 8;
static int _frame_count = 0;
static int _birdview_width = 1000;
static int _birdview_height = 1000;
static cv::Mat _birdview_buffer_8UC3[_img_buffer_size];
static cv::Mat _track_mat_8UC1(_birdview_height, _birdview_width, CV_8UC1, cv::Scalar(0));
static cv::Mat _filtered_mat_8UC1 = Mat::zeros(_track_mat_8UC1.size(), CV_8UC1);
static cv::Mat _coeffient_linefitting_mat_64F;
static cv::Mat _visual_linefitting_mat_8UC3 = Mat::zeros(1500, 1000, CV_8UC3);

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

float angleBetween(const Point &v1, const Point &v2)
{
    float len1 = sqrt(v1.x * v1.x + v1.y * v1.y);
    float len2 = sqrt(v2.x * v2.x + v2.y * v2.y);
    if (len1 != 0 && len2 != 0)
    {
      float dot = v1.x * v2.x + v1.y * v2.y;

      float a = dot / (len1 * len2);

      if (a >= 1.0)
          return 0.0;
      else if (a <= -1.0)
          return CV_PI;
      else
          return acos(a); // 0..PI
    }
    else
    {
      return 0.0;
    }

}

double euclidean_distance(cv::Point& p1, cv::Point& p2)
{
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

bool less_by_x(const geometry_msgs::Point& lhs, const geometry_msgs::Point& rhs)
{
  return lhs.x < rhs.x;
}

bool less_by_y(const geometry_msgs::Point& lhs, const geometry_msgs::Point& rhs)
{
  return lhs.y < rhs.y;
}

tf::StampedTransform findTransform(const std::string &in_target_frame, const std::string &in_source_frame)
{
  tf::StampedTransform transform;

  try
  {
    // What time should we use?
    _vectormap_transform_listener->lookupTransform(in_target_frame, in_source_frame, ros::Time(0), transform);
  }
  catch (tf::TransformException ex)
  {
    ROS_ERROR("%s", ex.what());
    return transform;
  }

  return transform;
}

geometry_msgs::Point transformPoint(const geometry_msgs::Point& point, const tf::Transform& tf)
{
  tf::Point tf_point;
  tf::pointMsgToTF(point, tf_point);

  tf_point = tf * tf_point;

  geometry_msgs::Point ros_point;
  tf::pointTFToMsg(tf_point, ros_point);

  return ros_point;
}

template <class T>
cv::Point coodinateTransformationFromPclToMat(T in_point)
{
  if (_pcl_header.frame_id == "/os1_lidar")
  { 
    return cv::Point(int(in_point.y*_birdview_scale+_birdview_width/2), int(in_point.x*_birdview_scale+_birdview_height));
  }
  else if (_pcl_header.frame_id == "velodyne")
  {
    return cv::Point(int(in_point.y*_birdview_scale+_birdview_width/2), int(-in_point.x*_birdview_scale+_birdview_height));
  }
}

// static bool lineIntersection(const cv::Point2f &a1, const cv::Point2f &b1, const cv::Point2f &a2,
//                              const cv::Point2f &b2, cv::Point2f &intersection) {
//     double A1 = b1.y - a1.y;
//     double B1 = a1.x - b1.x;
//     double C1 = (a1.x * A1) + (a1.y * B1);

//     double A2 = b2.y - a2.y;
//     double B2 = a2.x - b2.x;
//     double C2 = (a2.x * A2) + (a2.y * B2);

//     double det = (A1 * B2) - (A2 * B1);

//     if (!almostEqual(det, 0)) {
//         intersection.x = static_cast<float>(((C1 * B2) - (C2 * B1)) / (det));
//         intersection.y = static_cast<float>(((C2 * A1) - (C1 * A2)) / (det));

//         return true;
//     }

//     return false;
// }

void polyfit(std::vector<cv::Point>& in_points, int n)
{
  int _birdview_width = 1000;
  int _birdview_height = 1000;
  int _birdview_scale = 50;
	int size = in_points.size();
	int x_num = n + 1;
	cv::Mat mat_u(size, x_num, CV_64F);
	cv::Mat mat_y(size, 1, CV_64F);
 
	for (int i = 0; i < mat_u.rows; ++i)
		for (int j = 0; j < mat_u.cols; ++j)
		{
      // double y_transformed = in_points[i].x * _birdview_scale + _birdview_height;
			mat_u.at<double>(i, j) = pow(in_points[i].y, j);
		}
 
	for (int i = 0; i < mat_y.rows; ++i)
	{
    // double x_transformed = in_points[i].y * _birdview_scale + _birdview_width/2;
		mat_y.at<double>(i, 0) = in_points[i].x;
	}

  std::string mat_u_type = type2str(mat_u.type());
  std::string mat_y_type = type2str(mat_y.type());
  // ROS_INFO("Matrix: %s %dx%d \n", mat_u_type.c_str(), mat_u.cols, mat_u.rows );
  // ROS_INFO("Matrix: %s %dx%d \n", mat_y_type.c_str(), mat_y.cols, mat_y.rows );
	// _coeffient_linefitting_mat_64F = cv::Mat(x_num, 1, CV_64F);
	_coeffient_linefitting_mat_64F = (mat_u.t()*mat_u).inv()*mat_u.t()*mat_y; //in the case of one point input, (mat_u.t()*mat_u) is singular, inv() generates all zero

}

void filterCentroids(autoware_msgs::Centroids &in_centroids, std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Point>& centroids_filtered)
{
	for (auto centroid:in_centroids.points)
	{
		cv::Point ipt(coodinateTransformationFromPclToMat<geometry_msgs::Point>(centroid));
    cv::circle(_filtered_mat_8UC1, ipt, 3, Scalar(255), 2, CV_AA);
    for(int i = 0; i< contours.size(); i++)
    {
      if (pointPolygonTest(contours[i], ipt, false) == 1) //1 inside, 0 edge, -1 outside, notice that -1 is also true
      {
        centroids_filtered.push_back(ipt);
		    cv::circle(_filtered_mat_8UC1, ipt, 3, Scalar(255), 2, CV_AA);
      }
    }
	}
  // char path [1000];
  // static int count = 0;
  // sprintf (path, "/home/autoware/shared_dir/debugfolder/%d.jpg", count);
  // cv::imwrite(path, _filtered_mat_8UC1);
  // count++;
}

void centroidsToMat(autoware_msgs::Centroids &in_centroids)
{

  cv::Mat birdview_mat_8UC3(_birdview_width, _birdview_height, CV_8UC3, cv::Scalar::all(0));
	for (auto centroid:in_centroids.points)
	{
		cv::Point ipt(coodinateTransformationFromPclToMat<geometry_msgs::Point>(centroid));
		cv::circle(birdview_mat_8UC3, ipt, 30, Scalar(255, 255, 255), CV_FILLED, CV_AA);
	}
  _birdview_buffer_8UC3[_frame_count] = birdview_mat_8UC3;
}

void visualizeFitting(autoware_msgs::Centroids &in_centroids, int n)
{
  int _birdview_width = 1000;
  int _birdview_height = 1000;
  int _birdview_scale = 50;
  int num_y_polyline = 1000;

  // if (!_visual_clear_flag)
  // {
  //   _visual_linefitting_mat_8UC3 = _visual_linefitting_mat_8UC3&(cv::Scalar(0,0,0));
  // }

  // cv::Mat temp_mat;
  // cv::cvtColor(_filtered_mat_8UC1, temp_mat, COLOR_GRAY2BGR);
  // cv::bitwise_or(_visual_linefitting_mat_8UC3, temp_mat, _visual_linefitting_mat_8UC3);

    // cv::Mat temp_mat;
    // cv::cvtColor(_filtered_mat_8UC1, temp_mat, COLOR_GRAY2BGR);
    // cv::bitwise_or(_visual_linefitting_mat_8UC3, temp_mat, _visual_linefitting_mat_8UC3);
    // if (!_coeffient_linefitting_mat_64F.empty())
    // {
    //   _filtered_mat_8UC1.copyTo(_visual_linefitting_mat_8UC3);
    //   for (int i = 0; i < num_y_polyline; ++i)
    //   { 
    //     cv::Point2d ipt;
    //     ipt.y = i;
    //     float x = 0;
    //     for (int j = 0; j < n + 1; ++j)
    //     {
    //       x += _coeffient_linefitting_mat_64F.at<double>(j, 0)*pow(i,j);
    //     }
    //     ipt.x = x;
    //     circle(_visual_linefitting_mat_8UC3, ipt, 1, Scalar(255), CV_FILLED, CV_AA);
    //   }
    // }
  cv::Mat resized_mat; 
  cv::resize(_visual_linefitting_mat_8UC3, resized_mat, cv::Size(_visual_linefitting_mat_8UC3.cols * 0.5,_visual_linefitting_mat_8UC3.rows * 0.5));
  cv::imshow("origin", _birdview_buffer_8UC3[_frame_count]);
  cv::imshow("_filtered_mat_8UC1", _filtered_mat_8UC1);
  cv::imshow("_visual_linefitting_mat_8UC3", resized_mat);
  cv::imshow("_track_mat_8UC1", _track_mat_8UC1);
}

void outlierRemoval(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr)
{
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> rorfilter (false); // Initializing with true will allow us to extract the removed indices
    rorfilter.setInputCloud (in_cloud_ptr);
    ros::param::get("/lidar_euclidean_cluster_detect/radius_outlier_removal", _radius_outlier_removal);
    ros::param::get("/lidar_euclidean_cluster_detect/neighbours_outlier_removal", _neighbours_outlier_removal);
    rorfilter.setRadiusSearch (_radius_outlier_removal);
    rorfilter.setMinNeighborsInRadius (_neighbours_outlier_removal);
    rorfilter.setNegative (true);
    rorfilter.filter (*out_cloud_ptr);
    // The resulting cloud_out contains all points of cloud_in that have 4 or less neighbors within the 0.1 search radius
    //indices_rem = rorfilter.getRemovedIndices ();
    // The indices_rem array indexes all points of cloud_in that have 5 or more neighbors within the 0.1 search radius
}

void intensityFilter(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr)
{
  int intensity_threshold = (_pcl_header.frame_id == "velodyne") ? 50 : 300;
  for (unsigned int i = 0; i < in_cloud_ptr->points.size(); i++)
  {
      if (in_cloud_ptr->points[i].intensity > intensity_threshold)
      {   
          pcl::PointXYZ current_point;
          current_point.x = in_cloud_ptr->points[i].x;
          current_point.y = in_cloud_ptr->points[i].y;
          current_point.z = in_cloud_ptr->points[i].z;
          out_cloud_ptr->points.push_back(current_point);
      }
  }    
}

//filter those centroids which last for short frames (and are not stable)
void findAndGetFilteredContours(std::vector<std::vector<cv::Point>>& contours_filtered)
{
  std::vector<std::vector<cv::Point>> contours;
  std::vector<std::vector<cv::Point>> centroids_filtered_second;
  std::vector<cv::Vec4i> hierarchy; 
  std::vector<cv::Point> centroids_filtered;
  std::vector<cv::Point> centroids_left_lane; 
  std::vector<cv::Point> centroids_right_lane; 

  cv::findContours(_track_mat_8UC1, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
  _filtered_mat_8UC1 = _filtered_mat_8UC1&(cv::Scalar(0));

  for(int i = 0; i< contours.size(); i++)
  {
    if (contourArea(contours[i]) >= 9000.0)
    { 
      cv::drawContours(_filtered_mat_8UC1, contours, i, cv::Scalar(255), 2, 8);   
      contours_filtered.push_back(contours[i]);
    }
  }
}

Matrix<double> dataAssociation(int num_centroids_last_frame, int num_centroids_this_frame, 
std::vector<cv::Point>& centroids_last_frame, std::vector<cv::Point>& centroids_this_frame)
{
  Matrix<double> dist_Mat(num_centroids_last_frame, num_centroids_this_frame);
  for (int row = 0 ; row < num_centroids_last_frame ; row++)
    for (int col = 0 ; col < num_centroids_this_frame ; col++) 
    {
      dist_Mat(row, col) = euclidean_distance(centroids_last_frame[row], centroids_this_frame[col]);
    }
  Munkres<double> matcher;
  matcher.solve(dist_Mat);
  // for (int row = 0 ; row < num_centroids_last_frame ; row++)
  // {
  //   for (int col = 0 ; col < num_centroids_this_frame ; col++) 
  //   {
  //     std::cout.width(2);
  //     std::cout << dist_Mat(row, col) << ",";
  //   }
  //   std::cout << std::endl;
  // }
  return dist_Mat;
}

cv::Point getDisplacementVector(cv::Point& p1,cv::Point& p2)
{
  cv::Point vec_displacement;
  vec_displacement.x = p2.x - p1.x;
  vec_displacement.y = p2.y - p1.y;
  return vec_displacement;
}

cv::Point calculateAveragePoint(std::vector<cv::Point>& in_points)
{
  float sum_x = 0;
  float sum_y = 0;
  int in_points_size = in_points.size();
  cv::Point average_point;

  for (auto point:in_points)
  {
    sum_x += point.x;
    sum_y += point.y;
  }

  average_point.x = sum_x / in_points_size;
  average_point.y = sum_y / in_points_size;

  return average_point;
}

void predictTrajectory(cv::Point* velocity_vectors, cv::Point* predicted_trajectory, const unsigned int velocity_vectors_buffer_size)
{
  tk::spline s;
  std::vector<double> control_points_x(velocity_vectors_buffer_size + 1), control_points_y(velocity_vectors_buffer_size + 1);
  control_points_y[velocity_vectors_buffer_size] = 0;
  control_points_x[velocity_vectors_buffer_size] = 0; 
  cv::Point past_trajectory_offset;

  for (int i = 1; i < (velocity_vectors_buffer_size + 1); i++)
  {
    control_points_y[velocity_vectors_buffer_size - i] = control_points_y[velocity_vectors_buffer_size - i + 1] + velocity_vectors[i - 1].y;
    control_points_x[velocity_vectors_buffer_size - i] = control_points_x[velocity_vectors_buffer_size - i + 1] + velocity_vectors[i - 1].x;
  }

  past_trajectory_offset.y = 1000 - control_points_y[0];
  past_trajectory_offset.x = 500 - control_points_x[0];

  for (int i = 0; i < (velocity_vectors_buffer_size + 1); i++)
  {
    control_points_y[i] = control_points_y[i] + past_trajectory_offset.y;
    control_points_x[i] = control_points_x[i] + past_trajectory_offset.x;
  }

  s.set_boundary(tk::spline::second_deriv, 0.0,
                  tk::spline::first_deriv, -2.0, false);
  s.set_points(control_points_y, control_points_x);

  for (int y = control_points_y[velocity_vectors_buffer_size]; y > control_points_y[0]; y--)
  {
    cv::Point spline_point(int(s(y)), y);
    cv::circle(_visual_linefitting_mat_8UC3, spline_point, 1, Scalar(100, 50, 30), 1, CV_AA);
  }

  for (int y = 1000; y >= 500; y--)
  {
    cv::Point spline_point(int(s(y)), y);
    cv::circle(_visual_linefitting_mat_8UC3, spline_point, 1, Scalar(30, 255, 30), 1, CV_AA);
  }  
  // for(int i=-50; i<250; i++) {
  //   double x=0.01*i;
  //   printf("%f %f %f %f %f\n", x, s(x),
  //           s.deriv(1,x), s.deriv(2,x), s.deriv(3,x));
  //   // checking analytic derivatives and finite differences are close
  //   assert(fabs(s.deriv(1,x)-deriv1(s,x)) < 1e-8);
  //   assert(fabs(s.deriv(2,x)-deriv2(s,x)) < 1e-8);
  // }
}

void seperateAndFittingLanes(int num_centroids_last_frame, int num_centroids_this_frame, 
std::vector<cv::Point>& centroids_filtered_last_frame, std::vector<cv::Point>& centroids_filtered_this_frame, 
Matrix<double>& association_Mat)
{
  static float control_points[_img_buffer_size];
  static cv::Point vec_average_last;
  static bool first_vec_average_flag = true;
  _visual_linefitting_mat_8UC3 = _visual_linefitting_mat_8UC3&(cv::Scalar(0));
  std::vector<cv::Point> vecs_displacement;
  std::vector<cv::Point> points_left;
  std::vector<cv::Point> points_left_last;
  std::vector<cv::Point> points_right;
  std::vector<cv::Point> points_right_last;
  cv::Point vec_average;
  cv::Point vec_sensor_orientation;
  cv::Point point_sensor_position(int(_birdview_width / 2), int(_birdview_height));
  //trajectory prediction
  static bool first_velocity_vectors_frame_batch = true;
  static unsigned int velocity_vectors_frame_count = 0;
  const unsigned int velocity_vectors_buffer_size = 5;
  static cv::Point velocity_vectors_buffer[velocity_vectors_buffer_size];
  cv::Point predicted_trajectory[velocity_vectors_buffer_size];

  for (int row = 0 ; row < num_centroids_last_frame ; row++)
  {
    for (int col = 0 ; col < num_centroids_this_frame ; col++) 
    {
      if (association_Mat(row, col) == 0)
      {
        if (euclidean_distance(centroids_filtered_last_frame[row], centroids_filtered_this_frame[col]) < 100)
        {
          cv::line(_visual_linefitting_mat_8UC3, centroids_filtered_last_frame[row], centroids_filtered_this_frame[col], cv::Scalar(0, 0, 255));
          vecs_displacement.push_back(getDisplacementVector(centroids_filtered_this_frame[col], centroids_filtered_last_frame[row]));
        }
      }
    }
  }

  if (!vecs_displacement.empty())
  {
    if (first_vec_average_flag) //what if at the first frame we have wrongly matched?
    {
      vec_average_last = calculateAveragePoint(vecs_displacement);
      // vec_average_last = cv::Point(0, -50);
      first_vec_average_flag = false;
    }
    else
    {
      vec_average = calculateAveragePoint(vecs_displacement); //sometimes vecs_displacement is empty(no matching)
      float angle_varied = angleBetween(vec_average_last, vec_average);
      if (angle_varied >= (CV_PI / 12)) //15 degrees difference between frames should not be larget than 15 degrees
      {
        vec_average = vec_average_last;
      }
      vec_average_last = vec_average;
      if (vec_average.y != 0) //prevent spline fitting from crashing
      {
        velocity_vectors_buffer[velocity_vectors_frame_count] = vec_average;
        velocity_vectors_frame_count++;
      }
      if (!first_velocity_vectors_frame_batch)
      {
        predictTrajectory(velocity_vectors_buffer, predicted_trajectory, velocity_vectors_buffer_size);
      }
    }

    if (velocity_vectors_frame_count == velocity_vectors_buffer_size)
    {
      first_velocity_vectors_frame_batch = false;
      velocity_vectors_frame_count = 0;
    }    

    vec_sensor_orientation.x = 1000 * vec_average.x + point_sensor_position.x;
    vec_sensor_orientation.y = 1000 * vec_average.y + point_sensor_position.y;
    for (auto centroid:centroids_filtered_this_frame)
    {
      if ((centroid.x * vec_average.y - vec_average.x * centroid.y + vec_average.x * point_sensor_position.y - point_sensor_position.x * vec_average.y) > 0)
      {
        points_left.push_back(centroid);
        cv::circle(_visual_linefitting_mat_8UC3, centroid, 15, Scalar(0, 255, 0), 2, CV_AA);
      }
      else
      {
        points_right.push_back(centroid);
        cv::circle(_visual_linefitting_mat_8UC3, centroid, 15, Scalar(255, 0, 0), 2, CV_AA);
      }
    }
    
    for (auto centroid:centroids_filtered_last_frame)
    {
      if ((centroid.x * vec_average.y - vec_average.x * centroid.y + vec_average.x * point_sensor_position.y - point_sensor_position.x * vec_average.y) > 0)
      {
        points_left_last.push_back(centroid);
        cv::circle(_visual_linefitting_mat_8UC3, centroid, 15, Scalar(0, 100, 0), 2, CV_AA);
      }
      else
      {
        points_right_last.push_back(centroid);
        cv::circle(_visual_linefitting_mat_8UC3, centroid, 15, Scalar(100, 0, 0), 2, CV_AA);
      }
    }

    if (!points_left.empty())
    {
      sort(points_left.begin(), points_left.end(), 
        [](const Point & a, const Point & b) -> bool
      { 
        return a.y > b.y; 
      });   //descending order
      for(std::vector<Point>::iterator it = points_left.begin(); it != points_left.end()-1; ++it)
      {
        cv::line(_visual_linefitting_mat_8UC3, *it, *(next(it)), cv::Scalar(0, 255, 0));
      }
    }

    if (!points_right.empty())
    {
      sort(points_right.begin(), points_right.end(), 
        [](const Point & a, const Point & b) -> bool
      { 
          return a.y > b.y; 
      });   //descending order
      for(std::vector<Point>::iterator it_right = points_right.begin(); it_right != points_right.end()-1; ++it_right)
      {
        cv::line(_visual_linefitting_mat_8UC3, *it_right, *(next(it_right)), cv::Scalar(255, 0, 0));
      }
    }

    // lineIntersection();
    cv::line(_visual_linefitting_mat_8UC3, point_sensor_position, vec_sensor_orientation, cv::Scalar(255, 255, 255));
    char path [1000];
    static int count = 0;
    sprintf (path, "/home/autoware/shared_dir/debugfolder/%d.jpg", count);
    cv::imwrite(path, _visual_linefitting_mat_8UC3);
    count++;
  }
}

void findLane(autoware_msgs::Centroids &in_centroids, int n)
{
  //remember that in_centroids may be invalid, check the validity
  static unsigned int state; //0-not yet entered 1-confirming entering 2-entered 3-confirming leaving
  static bool first_find_lane = true;
  static bool _construction_site_flag = false;
  static bool _construction_site_flag_last = false;
  static std::vector<cv::Point> centroids_filtered_last_frame;
  static int num_centroids_filtered_last_frame;
  Matrix<double> association_Mat;
  std::vector<std::vector<cv::Point>> contours_filtered_last_frame;
  std::vector<std::vector<cv::Point>> contours_filtered_this_frame;
  std::vector<cv::Point> centroids_filtered_this_frame;
  int num_centroids_filtered_this_frame;

  if (!first_find_lane)
  {
    _track_mat_8UC1 = _track_mat_8UC1 & (cv::Scalar(0));
    for (int i = 0; i < _img_buffer_size; i++)
    {
      cv::Mat birdview_gray;
      cv::cvtColor(_birdview_buffer_8UC3[i], birdview_gray, cv::COLOR_BGR2GRAY);
      cv::bitwise_or(_track_mat_8UC1, birdview_gray, _track_mat_8UC1);
    }

    findAndGetFilteredContours(contours_filtered_this_frame);
    centroids_filtered_this_frame.clear();
    filterCentroids(in_centroids, contours_filtered_this_frame, centroids_filtered_this_frame);
    num_centroids_filtered_this_frame = centroids_filtered_this_frame.size();    
    // _visual_clear_flag = false;

    if (state == 0)
    {
      if (num_centroids_filtered_this_frame > 1 && num_centroids_filtered_last_frame > 1)      
      {
        state++;
      }
    }
    else if (state == 1)
    {
      if (num_centroids_filtered_this_frame == 0 || num_centroids_filtered_last_frame == 0)
      {
        state--;
      }
      else if (num_centroids_filtered_this_frame > 0 && num_centroids_filtered_last_frame > 0)      
      {
        state++;
      }      
    }
    else if (state == 2)
    {
      if (num_centroids_filtered_this_frame > 1 && num_centroids_filtered_last_frame > 1)
      {
        ROS_INFO("yes");
        association_Mat = dataAssociation(num_centroids_filtered_last_frame, num_centroids_filtered_this_frame,
        centroids_filtered_last_frame, centroids_filtered_this_frame);
        seperateAndFittingLanes(num_centroids_filtered_last_frame, num_centroids_filtered_this_frame, 
        centroids_filtered_last_frame, centroids_filtered_this_frame, association_Mat);
      }   
      else if (num_centroids_filtered_this_frame == 0 || num_centroids_filtered_last_frame == 0)   
      {
        state++;
      }
    }
    else if (state == 3)
    {
      if (num_centroids_filtered_this_frame > 0 && num_centroids_filtered_last_frame > 0)
      {
        state--;
      }
      else if (num_centroids_filtered_this_frame == 0 || num_centroids_filtered_last_frame == 0)   
      {
        state = 0;
      }     
    }
    ROS_INFO("%d", state);
    centroids_filtered_last_frame = centroids_filtered_this_frame;
    num_centroids_filtered_last_frame = num_centroids_filtered_this_frame;
  }
  else
  {
    findAndGetFilteredContours(contours_filtered_last_frame);
    filterCentroids(in_centroids, contours_filtered_last_frame, centroids_filtered_last_frame);
    num_centroids_filtered_last_frame = contours_filtered_last_frame.size();
    first_find_lane = false;
  }
}

void publishDetectedObjects(const autoware_msgs::CloudClusterArray &in_clusters)
{
  autoware_msgs::DetectedObjectArray detected_objects;
  detected_objects.header = in_clusters.header;

  for (size_t i = 0; i < in_clusters.clusters.size(); i++)
  {
    autoware_msgs::DetectedObject detected_object;
    detected_object.header = in_clusters.header;
    detected_object.label = "unknown";
    detected_object.score = 1.;
    detected_object.space_frame = in_clusters.header.frame_id;
    detected_object.pose = in_clusters.clusters[i].bounding_box.pose;
    detected_object.dimensions = in_clusters.clusters[i].dimensions;
    detected_object.pointcloud = in_clusters.clusters[i].cloud;
    detected_object.convex_hull = in_clusters.clusters[i].convex_hull;
    detected_object.valid = true;

    detected_objects.objects.push_back(detected_object);
  }
  _pub_detected_objects.publish(detected_objects);
}

void publishCloudClusters(const ros::Publisher *in_publisher, const autoware_msgs::CloudClusterArray &in_clusters,
                          const std::string &in_target_frame, const std_msgs::Header &in_header)
{
  if (in_target_frame != in_header.frame_id)
  {
    autoware_msgs::CloudClusterArray clusters_transformed;
    clusters_transformed.header = in_header;
    clusters_transformed.header.frame_id = in_target_frame;
    for (auto i = in_clusters.clusters.begin(); i != in_clusters.clusters.end(); i++)
    {
      autoware_msgs::CloudCluster cluster_transformed;
      cluster_transformed.header = in_header;
      try
      {
        _transform_listener->lookupTransform(in_target_frame, _pcl_header.frame_id, ros::Time(),
                                             *_transform);
        pcl_ros::transformPointCloud(in_target_frame, *_transform, i->cloud, cluster_transformed.cloud);
        _transform_listener->transformPoint(in_target_frame, ros::Time(), i->min_point, in_header.frame_id,
                                            cluster_transformed.min_point);
        _transform_listener->transformPoint(in_target_frame, ros::Time(), i->max_point, in_header.frame_id,
                                            cluster_transformed.max_point);
        _transform_listener->transformPoint(in_target_frame, ros::Time(), i->avg_point, in_header.frame_id,
                                            cluster_transformed.avg_point);
        _transform_listener->transformPoint(in_target_frame, ros::Time(), i->centroid_point, in_header.frame_id,
                                            cluster_transformed.centroid_point);

        cluster_transformed.dimensions = i->dimensions;
        cluster_transformed.eigen_values = i->eigen_values;
        cluster_transformed.eigen_vectors = i->eigen_vectors;
        
        cluster_transformed.convex_hull = i->convex_hull;
        cluster_transformed.bounding_box.pose.position = i->bounding_box.pose.position;
        if(_pose_estimation)
        {
          cluster_transformed.bounding_box.pose.orientation = i->bounding_box.pose.orientation;
        }
        else
        {
          cluster_transformed.bounding_box.pose.orientation.w = _initial_quat_w;
        }
        clusters_transformed.clusters.push_back(cluster_transformed);
      }
      catch (tf::TransformException &ex)
      {
        ROS_ERROR("publishCloudClusters: %s", ex.what());
      }
    }
    in_publisher->publish(clusters_transformed);
    
    (clusters_transformed);
  } else
  {
    in_publisher->publish(in_clusters);
    publishDetectedObjects(in_clusters);
  }
}

void publishCentroids(const ros::Publisher *in_publisher, const autoware_msgs::Centroids &in_centroids,
                      const std::string &in_target_frame, const std_msgs::Header &in_header)
{
  if (in_target_frame != in_header.frame_id)
  {
    autoware_msgs::Centroids centroids_transformed;
    centroids_transformed.header = in_header;
    centroids_transformed.header.frame_id = in_target_frame;
    for (auto i = centroids_transformed.points.begin(); i != centroids_transformed.points.end(); i++)
    {
      geometry_msgs::PointStamped centroid_in, centroid_out;
      centroid_in.header = in_header;
      centroid_in.point = *i;
      try
      {
        _transform_listener->transformPoint(in_target_frame, ros::Time(), centroid_in, in_header.frame_id,
                                            centroid_out);

        centroids_transformed.points.push_back(centroid_out.point);
      }
      catch (tf::TransformException &ex)
      {
        ROS_ERROR("publishCentroids: %s", ex.what());
      }
    }
    in_publisher->publish(centroids_transformed);
  } else
  {
    in_publisher->publish(in_centroids);
  }
}

void publishCloud(const ros::Publisher *in_publisher, const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_to_publish_ptr)
{
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
  cloud_msg.header = _pcl_header;
  in_publisher->publish(cloud_msg);
}

void publishColorCloud(const ros::Publisher *in_publisher,
                       const pcl::PointCloud<pcl::PointXYZRGB>::Ptr in_cloud_to_publish_ptr)
{
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
  cloud_msg.header = _pcl_header;
  in_publisher->publish(cloud_msg);
}

void keepLanePoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr, float in_left_lane_threshold = 1.5,
                    float in_right_lane_threshold = 1.5)
{
  pcl::PointIndices::Ptr far_indices(new pcl::PointIndices);
  for (unsigned int i = 0; i < in_cloud_ptr->points.size(); i++)
  {
    pcl::PointXYZ current_point;
    current_point.x = in_cloud_ptr->points[i].x;
    current_point.y = in_cloud_ptr->points[i].y;
    current_point.z = in_cloud_ptr->points[i].z;

    if (current_point.y > (in_left_lane_threshold) || current_point.y < -1.0 * in_right_lane_threshold)
    {
      far_indices->indices.push_back(i);
    }
  }
  out_cloud_ptr->points.clear();
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(in_cloud_ptr);
  extract.setIndices(far_indices);
  extract.setNegative(true);  // true removes the indices, false leaves only the indices
  extract.filter(*out_cloud_ptr);
}

#ifdef GPU_CLUSTERING

std::vector<ClusterPtr> clusterAndColorGpu(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr,
                                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud_ptr,
                                           autoware_msgs::Centroids &in_out_centroids,
                                           double in_max_cluster_distance = 0.5)
{
  std::vector<ClusterPtr> clusters;

  // Convert input point cloud to vectors of x, y, and z

  int size = in_cloud_ptr->points.size();

  if (size == 0)
    return clusters;

  float *tmp_x, *tmp_y, *tmp_z;

  tmp_x = (float *) malloc(sizeof(float) * size);
  tmp_y = (float *) malloc(sizeof(float) * size);
  tmp_z = (float *) malloc(sizeof(float) * size);

  for (int i = 0; i < size; i++)
  {
    pcl::PointXYZ tmp_point = in_cloud_ptr->at(i);

    tmp_x[i] = tmp_point.x;
    tmp_y[i] = tmp_point.y;
    tmp_z[i] = tmp_point.z;
  }

  GpuEuclideanCluster gecl_cluster;
  gecl_cluster.setInputPoints(tmp_x, tmp_y, tmp_z, size);
  gecl_cluster.setThreshold(in_max_cluster_distance);
  gecl_cluster.setMinClusterPts(_cluster_size_min);
  gecl_cluster.setMaxClusterPts(_cluster_size_max);
  gecl_cluster.extractClusters();
  std::vector<GpuEuclideanCluster::GClusterIndex> cluster_indices = gecl_cluster.getOutput();
  unsigned int k = 0;
  for (auto it = cluster_indices.begin(); it != cluster_indices.end(); it++)
  {
    ClusterPtr cluster(new Cluster());
    cluster->SetCloud(in_cloud_ptr, it->points_in_cluster, _pcl_header, k, (int) _colors[k].val[0],
                      (int) _colors[k].val[1], (int) _colors[k].val[2], "", _pose_estimation);
    ros::param::get("/lidar_euclidean_cluster_detect/min_cluster_height",_min_cluster_height);
    if (!_simulation)
    {
      if (cluster->GetHeight() >= _min_cluster_height && cluster->GetWidth() <= 1 && cluster->GetLength() <= 0.8)
      {
        clusters.push_back(cluster);
      }
    }
    else
    {
      clusters.push_back(cluster);
    }
    k++;
  }
  free(tmp_x);
  free(tmp_y);
  free(tmp_z);
  return clusters;
}

#endif

std::vector<ClusterPtr> clusterAndColor(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr,
                                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud_ptr,
                                        autoware_msgs::Centroids &in_out_centroids,
                                        double in_max_cluster_distance = 0.5)
{
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

  // create 2d pc
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2d(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(*in_cloud_ptr, *cloud_2d);
  // make it flat
  for (size_t i = 0; i < cloud_2d->points.size(); i++)
  {
    cloud_2d->points[i].z = 0;
  }

  if (cloud_2d->points.size() > 0)
    tree->setInputCloud(cloud_2d);

  std::vector<pcl::PointIndices> cluster_indices;

  // perform clustering on 2d cloud
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(in_max_cluster_distance);  //
  ec.setMinClusterSize(_cluster_size_min);
  ec.setMaxClusterSize(_cluster_size_max);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud_2d);
  ec.extract(cluster_indices);
  // use indices on 3d cloud

  /////////////////////////////////
  //---	3. Color clustered points
  /////////////////////////////////
  unsigned int k = 0;
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr final_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);

  std::vector<ClusterPtr> clusters;
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);//coord + color
  // cluster
  for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
  {
    ClusterPtr cluster(new Cluster());
    cluster->SetCloud(in_cloud_ptr, it->indices, _pcl_header, k, (int) _colors[k].val[0],
                      (int) _colors[k].val[1],
                      (int) _colors[k].val[2], "", _pose_estimation);
    clusters.push_back(cluster);

    k++;
  }
  // std::cout << "Clusters: " << k << std::endl;
  return clusters;
}

void checkClusterMerge(size_t in_cluster_id, std::vector<ClusterPtr> &in_clusters,
                       std::vector<bool> &in_out_visited_clusters, std::vector<size_t> &out_merge_indices,
                       double in_merge_threshold)
{
  // std::cout << "checkClusterMerge" << std::endl;
  pcl::PointXYZ point_a = in_clusters[in_cluster_id]->GetCentroid();
  for (size_t i = 0; i < in_clusters.size(); i++)
  {
    if (i != in_cluster_id && !in_out_visited_clusters[i])
    {
      pcl::PointXYZ point_b = in_clusters[i]->GetCentroid();
      double distance = sqrt(pow(point_b.x - point_a.x, 2) + pow(point_b.y - point_a.y, 2));
      if (distance <= in_merge_threshold)
      {
        in_out_visited_clusters[i] = true;
        out_merge_indices.push_back(i);
        // std::cout << "Merging " << in_cluster_id << " with " << i << " dist:" << distance << std::endl;
        checkClusterMerge(i, in_clusters, in_out_visited_clusters, out_merge_indices, in_merge_threshold);
      }
    }
  }
}

void mergeClusters(const std::vector<ClusterPtr> &in_clusters, std::vector<ClusterPtr> &out_clusters,
                   std::vector<size_t> in_merge_indices, const size_t &current_index,
                   std::vector<bool> &in_out_merged_clusters)
{
  // std::cout << "mergeClusters:" << in_merge_indices.size() << std::endl;
  pcl::PointCloud<pcl::PointXYZRGB> sum_cloud;
  pcl::PointCloud<pcl::PointXYZ> mono_cloud;
  ClusterPtr merged_cluster(new Cluster());
  for (size_t i = 0; i < in_merge_indices.size(); i++)
  {
    sum_cloud += *(in_clusters[in_merge_indices[i]]->GetCloud());
    in_out_merged_clusters[in_merge_indices[i]] = true;
  }
  std::vector<int> indices(sum_cloud.points.size(), 0);
  for (size_t i = 0; i < sum_cloud.points.size(); i++)
  {
    indices[i] = i;
  }

  if (sum_cloud.points.size() > 0)
  {
    pcl::copyPointCloud(sum_cloud, mono_cloud);
    merged_cluster->SetCloud(mono_cloud.makeShared(), indices, _pcl_header, current_index,
                             (int) _colors[current_index].val[0], (int) _colors[current_index].val[1],
                             (int) _colors[current_index].val[2], "", _pose_estimation);
    out_clusters.push_back(merged_cluster);
  }
}

void checkAllForMerge(std::vector<ClusterPtr> &in_clusters, std::vector<ClusterPtr> &out_clusters,
                      float in_merge_threshold)
{
  // std::cout << "checkAllForMerge" << std::endl;
  std::vector<bool> visited_clusters(in_clusters.size(), false);
  std::vector<bool> merged_clusters(in_clusters.size(), false);
  size_t current_index = 0;
  for (size_t i = 0; i < in_clusters.size(); i++)
  {
    if (!visited_clusters[i])
    {
      visited_clusters[i] = true;
      std::vector<size_t> merge_indices;
      checkClusterMerge(i, in_clusters, visited_clusters, merge_indices, in_merge_threshold);
      mergeClusters(in_clusters, out_clusters, merge_indices, current_index++, merged_clusters);
    }
  }
  for (size_t i = 0; i < in_clusters.size(); i++)
  {
    // check for clusters not merged, add them to the output
    if (!merged_clusters[i])
    {
      out_clusters.push_back(in_clusters[i]);
    }
  }

  // ClusterPtr cluster(new Cluster());
}

void segmentByDistance(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr,
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud_ptr,
                       autoware_msgs::Centroids &in_out_centroids, autoware_msgs::CloudClusterArray &in_out_clusters)
{
  // cluster the pointcloud according to the distance of the points using different thresholds (not only one for the
  // entire pc)
  // in this way, the points farther in the pc will also be clustered

  // 0 => 0-15m d=0.5
  // 1 => 15-30 d=1
  // 2 => 30-45 d=1.6
  // 3 => 45-60 d=2.1
  // 4 => >60   d=2.6
  std::vector<ClusterPtr> all_clusters;

  if (!_use_multiple_thres)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);

    for (unsigned int i = 0; i < in_cloud_ptr->points.size(); i++)
    {
      pcl::PointXYZ current_point;
      current_point.x = in_cloud_ptr->points[i].x;
      current_point.y = in_cloud_ptr->points[i].y;
      current_point.z = in_cloud_ptr->points[i].z;
      // if (current_point.x != 0 && current_point.y != 0 && current_point.z != 0)
      // {
      //   ROS_INFO("x=%f",current_point.x);
      //   ROS_INFO("y=%f",current_point.y);
      //   ROS_INFO("z=%f",current_point.z);
      // }
      cloud_ptr->points.push_back(current_point);
    }
    
#ifdef GPU_CLUSTERING
    if (_use_gpu)
    {
      all_clusters = clusterAndColorGpu(cloud_ptr, out_cloud_ptr, in_out_centroids,
                                        _clustering_distance);
    } else
    {
      all_clusters =
        clusterAndColor(cloud_ptr, out_cloud_ptr, in_out_centroids, _clustering_distance);
    }
#else
    all_clusters =
        clusterAndColor(cloud_ptr, out_cloud_ptr, in_out_centroids, _clustering_distance);
#endif
  } else
  { 
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_segments_array(5);
    for (unsigned int i = 0; i < cloud_segments_array.size(); i++)
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      cloud_segments_array[i] = tmp_cloud;
    }

    for (unsigned int i = 0; i < in_cloud_ptr->points.size(); i++)
    {
      pcl::PointXYZ current_point;
      current_point.x = in_cloud_ptr->points[i].x;
      current_point.y = in_cloud_ptr->points[i].y;
      current_point.z = in_cloud_ptr->points[i].z;

      float origin_distance = sqrt(pow(current_point.x, 2) + pow(current_point.y, 2));

      if (origin_distance < _clustering_ranges[0])
      {
        cloud_segments_array[0]->points.push_back(current_point);
      }
      else if (origin_distance < _clustering_ranges[1])
      {
        cloud_segments_array[1]->points.push_back(current_point);

      }else if (origin_distance < _clustering_ranges[2])
      {
        cloud_segments_array[2]->points.push_back(current_point);

      }else if (origin_distance < _clustering_ranges[3])
      {
        cloud_segments_array[3]->points.push_back(current_point);

      }else
      {
        cloud_segments_array[4]->points.push_back(current_point);
      }
    }

    std::vector<ClusterPtr> local_clusters;
    for (unsigned int i = 0; i < cloud_segments_array.size(); i++)
    {
      ROS_INFO("1");
#ifdef GPU_CLUSTERING
      if (_use_gpu)
      {
        local_clusters = clusterAndColorGpu(cloud_segments_array[i], out_cloud_ptr,
                                            in_out_centroids, _clustering_distances[i]);
      } else
      {
        local_clusters = clusterAndColor(cloud_segments_array[i], out_cloud_ptr,
                                         in_out_centroids, _clustering_distances[i]);
      }
#else
      local_clusters = clusterAndColor(
          cloud_segments_array[i], out_cloud_ptr, in_out_centroids, _clustering_distances[i]);
#endif
      all_clusters.insert(all_clusters.end(), local_clusters.begin(), local_clusters.end());
    }
  }
  
  // Clusters can be merged or checked in here
  //....
  // check for mergable clusters
  std::vector<ClusterPtr> mid_clusters;
  std::vector<ClusterPtr> final_clusters;

  if (all_clusters.size() > 0)
    checkAllForMerge(all_clusters, mid_clusters, _cluster_merge_threshold);
  else
    mid_clusters = all_clusters;

  if (mid_clusters.size() > 0)
    checkAllForMerge(mid_clusters, final_clusters, _cluster_merge_threshold);
  else
    final_clusters = mid_clusters;

    // Get final PointCloud to be published
    for (unsigned int i = 0; i < final_clusters.size(); i++)
    {
      *out_cloud_ptr = *out_cloud_ptr + *(final_clusters[i]->GetCloud());

      jsk_recognition_msgs::BoundingBox bounding_box = final_clusters[i]->GetBoundingBox();
      geometry_msgs::PolygonStamped polygon = final_clusters[i]->GetPolygon();
      jsk_rviz_plugins::Pictogram pictogram_cluster;
      pictogram_cluster.header = _pcl_header;

      // PICTO
      pictogram_cluster.mode = pictogram_cluster.STRING_MODE;
      pictogram_cluster.pose.position.x = final_clusters[i]->GetMaxPoint().x;
      pictogram_cluster.pose.position.y = final_clusters[i]->GetMaxPoint().y;
      pictogram_cluster.pose.position.z = final_clusters[i]->GetMaxPoint().z;
      tf::Quaternion quat(0.0, -0.7, 0.0, 0.7);
      tf::quaternionTFToMsg(quat, pictogram_cluster.pose.orientation);
      pictogram_cluster.size = 4;
      std_msgs::ColorRGBA color;
      color.a = 1;
      color.r = 1;
      color.g = 1;
      color.b = 1;
      pictogram_cluster.color = color;
      pictogram_cluster.character = std::to_string(i);
      // PICTO

      // pcl::PointXYZ min_point = final_clusters[i]->GetMinPoint();
      // pcl::PointXYZ max_point = final_clusters[i]->GetMaxPoint();
      pcl::PointXYZ center_point = final_clusters[i]->GetCentroid();
      geometry_msgs::Point centroid;
      centroid.x = center_point.x;
      centroid.y = center_point.y;
      centroid.z = center_point.z;
      bounding_box.header = _pcl_header;
      polygon.header = _pcl_header;

      if (final_clusters[i]->IsValid())
      {

        in_out_centroids.points.push_back(centroid);

        autoware_msgs::CloudCluster cloud_cluster;
        final_clusters[i]->ToROSMessage(_pcl_header, cloud_cluster);
        in_out_clusters.clusters.push_back(cloud_cluster);
      }
    }
}

void removeFloor(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr out_nofloor_cloud_ptr,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr out_onlyfloor_cloud_ptr, float in_max_height = 2,
                 float in_floor_max_angle = 20)
{
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(100);
  seg.setAxis(Eigen::Vector3f(0, 0, 1));
  // seg.setEpsAngle(pcl::deg2rad(in_floor_max_angle)); //largest angle inside which plane can shake

  seg.setDistanceThreshold(in_max_height);  // floor distance
  seg.setOptimizeCoefficients(true);
  seg.setInputCloud(in_cloud_ptr);
  seg.segment(*inliers, *coefficients);

  if (inliers->indices.size() == 0)
  {
    std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
  }

  // REMOVE THE FLOOR FROM THE CLOUD
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(in_cloud_ptr);
  extract.setIndices(inliers);
  extract.setNegative(true);  // true removes the indices, false leaves only the indices
  extract.filter(*out_nofloor_cloud_ptr);
  // EXTRACT THE FLOOR FROM THE CLOUD
  extract.setNegative(false);  // true removes the indices, false leaves only the indices
  extract.filter(*out_onlyfloor_cloud_ptr);
}

void downsampleCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr, float in_leaf_size = 0.2)
{
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(in_cloud_ptr);
  sor.setLeafSize((float) in_leaf_size, (float) in_leaf_size, (float) in_leaf_size);
  sor.filter(*out_cloud_ptr);
}

void clipCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr,
               pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr, float in_min_height = -1.3, float in_max_height = 0.5)
{
  
  out_cloud_ptr->points.clear();
  for (unsigned int i = 0; i < in_cloud_ptr->points.size(); i++)
  {
    if (in_cloud_ptr->points[i].z >= in_min_height && in_cloud_ptr->points[i].z <= in_max_height)
    {
      out_cloud_ptr->points.push_back(in_cloud_ptr->points[i]);
    }
  }
}

void differenceNormalsSegmentation(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr,
                                   pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr)
{
  float small_scale = 0.5;
  float large_scale = 2.0;
  float angle_threshold = 0.5;
  pcl::search::Search<pcl::PointXYZ>::Ptr tree;
  if (in_cloud_ptr->isOrganized())
  {
    tree.reset(new pcl::search::OrganizedNeighbor<pcl::PointXYZ>());
  } else
  {
    tree.reset(new pcl::search::KdTree<pcl::PointXYZ>(false));
  }

  // Set the input pointcloud for the search tree
  tree->setInputCloud(in_cloud_ptr);

  pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::PointNormal> normal_estimation;
  // pcl::gpu::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> normal_estimation;
  normal_estimation.setInputCloud(in_cloud_ptr);
  normal_estimation.setSearchMethod(tree);

  normal_estimation.setViewPoint(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::max());

  pcl::PointCloud<pcl::PointNormal>::Ptr normals_small_scale(new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointNormal>::Ptr normals_large_scale(new pcl::PointCloud<pcl::PointNormal>);

  normal_estimation.setRadiusSearch(small_scale);
  normal_estimation.compute(*normals_small_scale);

  normal_estimation.setRadiusSearch(large_scale);
  normal_estimation.compute(*normals_large_scale);

  pcl::PointCloud<pcl::PointNormal>::Ptr diffnormals_cloud(new pcl::PointCloud<pcl::PointNormal>);
  pcl::copyPointCloud<pcl::PointXYZ, pcl::PointNormal>(*in_cloud_ptr, *diffnormals_cloud);

  // Create DoN operator
  pcl::DifferenceOfNormalsEstimation<pcl::PointXYZ, pcl::PointNormal, pcl::PointNormal> diffnormals_estimator;
  diffnormals_estimator.setInputCloud(in_cloud_ptr);
  diffnormals_estimator.setNormalScaleLarge(normals_large_scale);
  diffnormals_estimator.setNormalScaleSmall(normals_small_scale);

  diffnormals_estimator.initCompute();

  diffnormals_estimator.computeFeature(*diffnormals_cloud);

  pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond(new pcl::ConditionOr<pcl::PointNormal>());
  range_cond->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr(
    new pcl::FieldComparison<pcl::PointNormal>("curvature", pcl::ComparisonOps::GT, angle_threshold)));
  // Build the filter
  pcl::ConditionalRemoval<pcl::PointNormal> cond_removal;
  cond_removal.setCondition(range_cond);
  cond_removal.setInputCloud(diffnormals_cloud);

  pcl::PointCloud<pcl::PointNormal>::Ptr diffnormals_cloud_filtered(new pcl::PointCloud<pcl::PointNormal>);

  // Apply filter
  cond_removal.filter(*diffnormals_cloud_filtered);

  pcl::copyPointCloud<pcl::PointNormal, pcl::PointXYZ>(*diffnormals_cloud, *out_cloud_ptr);
}

void removePointsUpTo(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr,
                      pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr, const double in_distance)
{
  out_cloud_ptr->points.clear();
  for (unsigned int i = 0; i < in_cloud_ptr->points.size(); i++)
  {
    float origin_distance = sqrt(pow(in_cloud_ptr->points[i].x, 2) + pow(in_cloud_ptr->points[i].y, 2));
    if (origin_distance > in_distance)
    {
      out_cloud_ptr->points.push_back(in_cloud_ptr->points[i]);
    }
  }
}

void velodyne_callback(const sensor_msgs::PointCloud2ConstPtr& in_sensor_cloud)
{

  //_start = std::chrono::system_clock::now();
  if (!_using_sensor_cloud)
  {
    _using_sensor_cloud = true;

    pcl::PointCloud<pcl::PointXYZ>::Ptr removed_points_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_removed_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr intensity_filtered_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr inlanes_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr nofloor_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr onlyfloor_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr diffnormals_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr clipped_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_clustered_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);


    autoware_msgs::Centroids centroids;
    autoware_msgs::CloudClusterArray cloud_clusters;

    
    _pcl_header = in_sensor_cloud->header;
    _output_frame = _pcl_header.frame_id;

    if (_intensity_filter)
    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr current_sensor_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
      pcl::fromROSMsg(*in_sensor_cloud, *current_sensor_cloud_ptr);
      intensityFilter(current_sensor_cloud_ptr, intensity_filtered_cloud_ptr);
    }
    else
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr current_sensor_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::fromROSMsg(*in_sensor_cloud, *current_sensor_cloud_ptr);
      *intensity_filtered_cloud_ptr = *current_sensor_cloud_ptr;
    }
    
    if (_remove_points_upto > 0.0)
    {
      removePointsUpTo(intensity_filtered_cloud_ptr, removed_points_cloud_ptr, _remove_points_upto); //remove points near
    }
    else
    {
      removed_points_cloud_ptr = intensity_filtered_cloud_ptr;
    }

    if (_downsample_cloud)
      downsampleCloud(removed_points_cloud_ptr, downsampled_cloud_ptr, _leaf_size);
    else
      downsampled_cloud_ptr = removed_points_cloud_ptr;

    outlierRemoval(downsampled_cloud_ptr, outlier_removed_cloud_ptr);    
    
    clipCloud(outlier_removed_cloud_ptr, clipped_cloud_ptr, _clip_min_height, _clip_max_height); //clip according to z axis

    if (_keep_lanes)
      keepLanePoints(clipped_cloud_ptr, inlanes_cloud_ptr, _keep_lane_left_distance, _keep_lane_right_distance); //clip according to y axis
    else
      inlanes_cloud_ptr = clipped_cloud_ptr;
    ros::param::get("/lidar_euclidean_cluster_detect/ransac_height",_ransac_height);
    ros::param::get("/lidar_euclidean_cluster_detect/ransac_angle",_ransac_angle);
    if (_remove_ground)
    {
      removeFloor(inlanes_cloud_ptr, nofloor_cloud_ptr, onlyfloor_cloud_ptr, _ransac_height,_ransac_angle);
      publishCloud(&_pub_ground_cloud, onlyfloor_cloud_ptr);
    }
    else
    {
      nofloor_cloud_ptr = inlanes_cloud_ptr;
    }

    publishCloud(&_pub_points_lanes_cloud, nofloor_cloud_ptr);

    if (_use_diffnormals)
      differenceNormalsSegmentation(nofloor_cloud_ptr, diffnormals_cloud_ptr);
    else
      diffnormals_cloud_ptr = nofloor_cloud_ptr;

    
    segmentByDistance(diffnormals_cloud_ptr, colored_clustered_cloud_ptr, centroids,
                      cloud_clusters);
    publishColorCloud(&_pub_cluster_cloud, colored_clustered_cloud_ptr); //colored_clustered_cloud_ptr and cloud_clusters difference?

    centroids.header = _pcl_header;

    publishCentroids(&_centroid_pub, centroids, _output_frame, _pcl_header);

    cloud_clusters.header = _pcl_header;

    publishCloudClusters(&_pub_clusters_message, cloud_clusters, _output_frame, _pcl_header); //autoware_msgs::CloudClusterArray

    centroidsToMat(centroids);
    if (!_first_frame_batches)
    {
      findLane(centroids, 3);
      visualizeFitting(centroids, 3);
    }
    cv::waitKey(1);
    _using_sensor_cloud = false;
  }
  _frame_count++;
  if (_frame_count == _img_buffer_size)
  {
    _first_frame_batches = false;
    _frame_count = 0;
  }
}
int main(int argc, char **argv)
{
  // Initialize ROS
  ros::init(argc, argv, "euclidean_cluster");

  ros::NodeHandle h;
  ros::NodeHandle private_nh("~");

  tf::StampedTransform transform;
  tf::TransformListener listener;
  tf::TransformListener vectormap_tf_listener;

  _vectormap_transform_listener = &vectormap_tf_listener;
  _transform = &transform;
  _transform_listener = &listener;

#if (CV_MAJOR_VERSION == 3)
  generateColors(_colors, 255);
#else
  cv::generateColors(_colors, 255);
#endif

  _pub_cluster_cloud = h.advertise<sensor_msgs::PointCloud2>("/points_cluster", 1);
  _pub_ground_cloud = h.advertise<sensor_msgs::PointCloud2>("/points_ground", 1);
  _centroid_pub = h.advertise<autoware_msgs::Centroids>("/cluster_centroids", 1);

  _pub_points_lanes_cloud = h.advertise<sensor_msgs::PointCloud2>("/points_lanes", 1);
  _pub_clusters_message = h.advertise<autoware_msgs::CloudClusterArray>("/detection/lidar_detector/cloud_clusters", 1);
  _pub_detected_objects = h.advertise<autoware_msgs::DetectedObjectArray>("/detection/lidar_detector/objects", 1);
  std::cout << _birdview_buffer_8UC3 << std::endl;
  std::fill_n(_birdview_buffer_8UC3, _img_buffer_size, Mat(_birdview_height, _birdview_width, CV_8UC1, cv::Scalar(0)));
  std::string points_topic, gridmap_topic;

  _using_sensor_cloud = false;

  if (private_nh.getParam("points_node", points_topic))
  {
    ROS_INFO("euclidean_cluster > Setting points node to %s", points_topic.c_str());
  }
  else
  {
    ROS_INFO("euclidean_cluster > No points node received, defaulting to points_raw, you can use "
               "_points_node:=YOUR_TOPIC");
    points_topic = "/points_raw";
  }

  _use_diffnormals = false;
  if (private_nh.getParam("use_diffnormals", _use_diffnormals))
  {
    if (_use_diffnormals)
      ROS_INFO("Euclidean Clustering: Applying difference of normals on clustering pipeline");
    else
      ROS_INFO("Euclidean Clustering: Difference of Normals will not be used.");
  }

 /* Initialize tuning parameter */
  private_nh.param("simulation", _simulation, true);
  ROS_INFO("[%s] simulation: %d", __APP_NAME__, _simulation);
  private_nh.param("intensity_filter", _intensity_filter, false);
  ROS_INFO("[%s] intensity_filter: %d", __APP_NAME__, _intensity_filter);
  private_nh.param("min_cluster_height", _min_cluster_height, 0.5);
  ROS_INFO("[%s] min_cluster_height: %f", __APP_NAME__, _min_cluster_height);
  private_nh.param("ransac_height", _ransac_height, 1.0);
  ROS_INFO("[%s] ransac_height: %f", __APP_NAME__, _ransac_height);
  private_nh.param("ransac_angle", _ransac_angle, 20.0);
  ROS_INFO("[%s] ransac_angle: %f", __APP_NAME__, _ransac_angle);  
  private_nh.param("downsample_cloud", _downsample_cloud, true);
  ROS_INFO("[%s] downsample_cloud: %d", __APP_NAME__, _downsample_cloud);
  private_nh.param("remove_ground", _remove_ground, true);
  ROS_INFO("[%s] remove_ground: %d", __APP_NAME__, _remove_ground);
  private_nh.param("leaf_size", _leaf_size, 0.1);
  ROS_INFO("[%s] leaf_size: %f", __APP_NAME__, _leaf_size);
  if (!_simulation)
  {
    private_nh.param("cluster_size_min", _cluster_size_min, 20);
    ROS_INFO("[%s] cluster_size_min %d", __APP_NAME__, _cluster_size_min);
  }
  else
  {
    private_nh.param("cluster_size_min", _cluster_size_min, 0);
    ROS_INFO("[%s] cluster_size_min %d", __APP_NAME__, _cluster_size_min);    
  }
  private_nh.param("cluster_size_max", _cluster_size_max, 100000);
  ROS_INFO("[%s] cluster_size_max: %d", __APP_NAME__, _cluster_size_max);
  private_nh.param("pose_estimation", _pose_estimation, false);
  ROS_INFO("[%s] pose_estimation: %d", __APP_NAME__, _pose_estimation);
  private_nh.param("clip_min_height", _clip_min_height, -5.0);
  ROS_INFO("[%s] clip_min_height: %f", __APP_NAME__, _clip_min_height);
  private_nh.param("clip_max_height", _clip_max_height, 5.0);
  ROS_INFO("[%s] clip_max_height: %f", __APP_NAME__, _clip_max_height);
  private_nh.param("keep_lanes", _keep_lanes, false);
  ROS_INFO("[%s] keep_lanes: %d", __APP_NAME__, _keep_lanes);
  private_nh.param("keep_lane_left_distance", _keep_lane_left_distance, 5.0);
  ROS_INFO("[%s] keep_lane_left_distance: %f", __APP_NAME__, _keep_lane_left_distance);
  private_nh.param("keep_lane_right_distance", _keep_lane_right_distance, 5.0);
  ROS_INFO("[%s] keep_lane_right_distance: %f", __APP_NAME__, _keep_lane_right_distance);
  private_nh.param("max_boundingbox_side", _max_boundingbox_side, 10.0);
  ROS_INFO("[%s] max_boundingbox_side: %f", __APP_NAME__, _max_boundingbox_side);
  private_nh.param("cluster_merge_threshold", _cluster_merge_threshold, 1.5);
  ROS_INFO("[%s] cluster_merge_threshold: %f", __APP_NAME__, _cluster_merge_threshold);
  private_nh.param<std::string>("output_frame", _output_frame, "os1_lidar");
  ROS_INFO("[%s] output_frame: %s", __APP_NAME__, _output_frame.c_str());

  private_nh.param("remove_points_upto", _remove_points_upto, 0.0);
  ROS_INFO("[%s] remove_points_upto: %f", __APP_NAME__, _remove_points_upto);

  private_nh.param("clustering_distance", _clustering_distance, 0.75);
  ROS_INFO("[%s] clustering_distance: %f", __APP_NAME__, _clustering_distance);

  private_nh.param("use_gpu", _use_gpu, true);
  ROS_INFO("[%s] use_gpu: %d", __APP_NAME__, _use_gpu);

  private_nh.param("use_multiple_thres", _use_multiple_thres, false);
  ROS_INFO("[%s] use_multiple_thres: %d", __APP_NAME__, _use_multiple_thres);

  std::string str_distances;
  std::string str_ranges;
  private_nh.param("clustering_distances", str_distances, std::string("[0.5,1.1,1.6,2.1,2.6]"));
  ROS_INFO("[%s] clustering_distances: %s", __APP_NAME__, str_distances.c_str());
  private_nh.param("clustering_ranges", str_ranges, std::string("[15,30,45,60]"));
    ROS_INFO("[%s] clustering_ranges: %s", __APP_NAME__, str_ranges.c_str());

  if (_use_multiple_thres)
  {
    YAML::Node distances = YAML::Load(str_distances);
    YAML::Node ranges = YAML::Load(str_ranges);
    size_t distances_size = distances.size();
    size_t ranges_size = ranges.size();
    if (distances_size == 0 || ranges_size == 0)
    {
      ROS_ERROR("Invalid size of clustering_ranges or/and clustering_distance. \
    The size of clustering distance and clustering_ranges should not be 0");
      ros::shutdown();
    }
    if ((distances_size - ranges_size) != 1)
    {
      ROS_ERROR("Invalid size of clustering_ranges or/and clustering_distance. \
    Expecting that (distances_size - ranges_size) == 1 ");
      ros::shutdown();
    }
    for (size_t i_distance = 0; i_distance < distances_size; i_distance++)
    {
      _clustering_distances.push_back(distances[i_distance].as<double>());
    }
    for (size_t i_range = 0; i_range < ranges_size; i_range++)
    {
      _clustering_ranges.push_back(ranges[i_range].as<double>());
    }
  }

  _velodyne_transform_available = false;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = h.subscribe(points_topic, 1, velodyne_callback);

  // Spin
  ros::spin();
}

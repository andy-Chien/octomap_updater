/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Author: Jon Binney, Ioan Sucan */

#include <cmath>
#include <time.h>
#include <thread>
#include <moveit/occupancy_map_monitor/occupancy_map_monitor.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Vector3.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <XmlRpcException.h>
#include <pcl/filters/voxel_grid.h>
#include "octomap_updater/pointcloud_octomap_updater/pointcloud_octomap_updater.h"
#include "octomap_updater/pointcloud_octomap_updater/mesh_sampling.hpp"

#include <memory>

namespace occupancy_map_monitor
{
PointCloudOctomapUpdaterFast::PointCloudOctomapUpdaterFast()
  : OccupancyMapUpdater("PointCloudUpdater")
  , private_nh_("~")
  , scale_(1.0)
  , padding_(0.0)
  , max_range_(std::numeric_limits<double>::infinity())
  , point_subsample_(1)
  , max_update_rate_(0)
  , point_cloud_subscriber_(nullptr)
  , point_cloud_filter_(nullptr)
{
}

PointCloudOctomapUpdaterFast::~PointCloudOctomapUpdaterFast()
{
  stopHelper();
}

bool PointCloudOctomapUpdaterFast::setParams(XmlRpc::XmlRpcValue& params)
{
  try
  {
    if (!params.hasMember("point_cloud_topic"))
      return false;
    point_cloud_topic_ = static_cast<const std::string&>(params["point_cloud_topic"]);

    readXmlParam(params, "max_range", &max_range_);
    readXmlParam(params, "padding_offset", &padding_);
    readXmlParam(params, "padding_scale", &scale_);
    readXmlParam(params, "point_subsample", &point_subsample_);
    if (params.hasMember("max_update_rate"))
      readXmlParam(params, "max_update_rate", &max_update_rate_);
    if (params.hasMember("filtered_cloud_topic"))
      filtered_cloud_topic_ = static_cast<const std::string&>(params["filtered_cloud_topic"]);
  }
  catch (XmlRpc::XmlRpcException& ex)
  {
    ROS_ERROR("[Octomap Updater]: XmlRpc Exception: %s", ex.getMessage().c_str());
    return false;
  }

  return true;
}

bool PointCloudOctomapUpdaterFast::initialize()
{
  accept_mesh_ = true;
  tf_buffer_.reset(new tf2_ros::Buffer());
  tf_listener_.reset(new tf2_ros::TransformListener(*tf_buffer_, root_nh_));
  shape_mask_.reset(new point_containment_filter::ShapeMask());
  shape_mask_->setTransformCallback(boost::bind(&PointCloudOctomapUpdaterFast::getShapeTransform, this, _1, _2));
  if (!filtered_cloud_topic_.empty())
    filtered_cloud_publisher_ = private_nh_.advertise<sensor_msgs::PointCloud2>(filtered_cloud_topic_, 10, false);
  gvlInitialize();
  return true;
}

void PointCloudOctomapUpdaterFast::gvlInitialize()
{
  shape_mask_mpc = new gpu_voxels::MetaPointCloud();
  received_cloud = new gpu_voxels::PointCloud();
  std::vector<Vector3f> empty_cloud;
  addCloudToMPC(empty_cloud);
  std::unique_lock<std::mutex> lock(mpc_mutex);
  voxel_side_length = VOXEL_SIDE_LENGTH;
  map_dimensions =  Vector3ui(max_range_ * 2 / voxel_side_length, max_range_ * 2 / voxel_side_length, max_range_ / voxel_side_length);
  init_transform = Matrix4f(1, 0, 0, (float)map_dimensions.x * voxel_side_length / 2,
                            0, 1, 0, (float)map_dimensions.y * voxel_side_length / 2,
                            0, 0, 1, 0,
                            0, 0, 0, 1);
  init_transform.invertMatrix(inv_init_transform);

  /*
   * First, we generate an API class, which defines the
   * volume of our space and the resolution.
   * Be careful here! The size is limited by the memory
   * of your GPU. Even if an empty Octree is small, a
   * Voxelmap will always require the full memory.
   */

  icl_core::logging::initialize();

  gvl = GpuVoxels::getInstance();
  // ==> 200 Voxels, each one is 1 cm in size so the map represents 20x20x20 centimeter
  gvl->initialize(map_dimensions.x, map_dimensions.y, map_dimensions.z, voxel_side_length); 

  // Add a map:
  gvl->addMap(MT_BITVECTOR_VOXELLIST, "myHandVoxellist");
  gvl->addMap(MT_BITVECTOR_VOXELLIST, "maskVoxelList");
  gvl->addMap(MT_COUNTING_VOXELLIST, "pointCloudVoxelList");
  maskVoxelList = dynamic_pointer_cast<BitVectorVoxelList>(gvl->getMap("maskVoxelList"));
  pointCloudVoxelList = dynamic_pointer_cast<CountingVoxelList>(gvl->getMap("pointCloudVoxelList"));
} 

void PointCloudOctomapUpdaterFast::start()
{
  if (point_cloud_subscriber_)
    return;
  /* subscribe to point cloud topic using tf filter*/
  point_cloud_subscriber_ = new message_filters::Subscriber<sensor_msgs::PointCloud2>(root_nh_, point_cloud_topic_, 1);
  if (tf_listener_ && tf_buffer_ && !monitor_->getMapFrame().empty())
  {
    point_cloud_filter_ = new tf2_ros::MessageFilter<sensor_msgs::PointCloud2>(*point_cloud_subscriber_, *tf_buffer_,
                                                                               monitor_->getMapFrame(), 1, root_nh_);
    point_cloud_filter_->registerCallback(boost::bind(&PointCloudOctomapUpdaterFast::cloudMsgCallback, this, _1));
    ROS_INFO("[Octomap Updater]: Listening to '%s' using message filter with target frame '%s'", point_cloud_topic_.c_str(),
             point_cloud_filter_->getTargetFramesString().c_str());
  }
  else
  {
    point_cloud_subscriber_->registerCallback(boost::bind(&PointCloudOctomapUpdaterFast::cloudMsgCallback, this, _1));
    ROS_INFO("[Octomap Updater]: Listening to '%s'", point_cloud_topic_.c_str());
  }
}

void PointCloudOctomapUpdaterFast::stopHelper()
{
  delete point_cloud_filter_;
  delete point_cloud_subscriber_;
}

void PointCloudOctomapUpdaterFast::stop()
{
  stopHelper();
  point_cloud_filter_ = nullptr;
  point_cloud_subscriber_ = nullptr;
}

ShapeHandle PointCloudOctomapUpdaterFast::excludeShape(const shapes::ShapeConstPtr& shape)
{
  ShapeHandle shape_handle = checkShapeExist(shape);
  if(shape_handle != 0)
  {
    return shape_handle;
  }

  bool add_shape_success = true;
  switch (shape->type)
  {
    case shapes::BOX:
    {
      accept_mesh_ = false;
      const double *size = static_cast<const shapes::Box*>(shape.get())->size;
      Vector3f center_min = Vector3f(-1*scale_*size[0]/2 - padding_, -1*scale_*size[1]/2 - padding_, -1*scale_*size[2]/2 - padding_);
      Vector3f center_max = Vector3f(scale_*size[0]/2 + padding_, scale_*size[1]/2 + padding_, scale_*size[2]/2 + padding_);
      shape_handle = addCloudToMPC(gpu_voxels::geometry_generation::createBoxOfPoints(center_min, center_max, voxel_side_length));
      ROS_DEBUG("[Octomap Updater]: Created new shape of BOX, handle is %i", shape_handle);
      break;
    }
    case shapes::SPHERE:
    {
      double sphere_radius = static_cast<const shapes::Sphere*>(shape.get())->radius;
      Vector3f sphere_center = Vector3f(0, 0, 0);
      shape_handle = addCloudToMPC(gpu_voxels::geometry_generation::createSphereOfPoints(sphere_center, sphere_radius, voxel_side_length));
      break;
    }
    case shapes::CYLINDER:
    {
      double cylinder_radius = static_cast<const shapes::Cylinder*>(shape.get())->radius;
      double cylinder_length = static_cast<const shapes::Cylinder*>(shape.get())->length;
      Vector3f cylinder_center = Vector3f(0, 0, 0);
      shape_handle = addCloudToMPC(gpu_voxels::geometry_generation::createCylinderOfPoints(cylinder_center, cylinder_radius, cylinder_length, voxel_side_length));
      break;
    }
    case shapes::MESH:
    {
      if(!accept_mesh_)
      {
        ROS_DEBUG("[Octomap Updater]: Ignore MESH input");
        shape_handle = 0;
        add_shape_success = false;
        break;
      }
      const shapes::Mesh* mesh = static_cast<const shapes::Mesh*>(shape.get());
      Eigen::Vector3d scale_indx(1, 1, 1);
       Eigen::Vector3d padding_indx;
      if (mesh->vertex_count > 1)
      {
        double mx = std::numeric_limits<double>::max();
        Eigen::Vector3d min(mx, mx, mx);
        Eigen::Vector3d max(-mx, -mx, -mx);
        unsigned int cnt = mesh->vertex_count * 3;
        for (unsigned int i = 0; i < cnt; i += 3)
        {
          Eigen::Vector3d v(mesh->vertices[i + 0], mesh->vertices[i + 1], mesh->vertices[i + 2]);
          min = min.cwiseMin(v);
          max = max.cwiseMax(v);
        }
        scale_indx = 1.732 * (max - min).normalized(); //1.732 = sqrt(3)
      }
      padding_indx = scale_indx;
      pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filled_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
      pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mesh_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
      pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr down_sample_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

      float scale_x, scale_y, scale_z, padding_x, padding_y, padding_z;
      for(float padding=padding_; padding>-0.01; padding-=0.01)
      {
        for(float scale=scale_; scale>0.1; scale-=0.03)
        {
          if(padding > 0.01 && scale < 0.9)
            continue;
          scale_x = (scale > 1) ? 1 + (scale - 1) / scale_indx[0] : scale;
          scale_y = (scale > 1) ? 1 + (scale - 1) / scale_indx[1] : scale;
          scale_z = (scale > 1) ? 1 + (scale - 1) / scale_indx[2] : scale;
          padding_x = (padding > 0) ? padding / scale_indx[0] : padding;
          padding_y = (padding > 0) ? padding / scale_indx[1] : padding;
          padding_z = (padding > 0) ? padding / scale_indx[2] : padding;
          shapes::Shape* shape_for_sample = shape->clone();
          static_cast<shapes::Mesh*>(shape_for_sample)->scaleAndPadd(scale_x, scale_y, scale_z, padding_x, padding_y, padding_z);
          // shape_for_sample->scaleAndPadd(scale, padding);
          samplePointFromMesh(shape_for_sample, mesh_cloud, POINTS_PER_MESH * scale);
          *filled_cloud += *mesh_cloud;
        }
      }
      pcl::VoxelGrid<pcl::PointXYZRGBNormal> vg;
      vg.setInputCloud(filled_cloud);
      vg.setLeafSize(voxel_side_length / 2, voxel_side_length / 2, voxel_side_length / 2);
      vg.filter(*down_sample_cloud);
      std::vector<Vector3f> model_coordinates;
      for(auto cloud_it=down_sample_cloud->begin(); cloud_it!=down_sample_cloud->end(); ++cloud_it)
        model_coordinates.push_back(Vector3f(cloud_it->x, cloud_it->y, cloud_it->z));
      shape_handle = addCloudToMPC(model_coordinates);
      ROS_INFO("[Octomap Updater]: Created new shape of mesh, handle is %i", shape_handle);
      break;
    }
    default:
    {
      ROS_ERROR("[Octomap Updater]: Creating body from shape: Unknown shape type");
      shape_handle = 0;
      add_shape_success = false;
    }
  }

  if(add_shape_success)
  {
    std::unique_lock<std::mutex> data_lock(data_mutex);
    contain_shape_.insert(std::pair<ShapeHandle, shapes::Shape*>(shape_handle, shape->clone()));
    // shapes_transform_.insert(std::pair<ShapeHandle, Eigen::Isometry3d>(shape_handle, Eigen::Isometry3d(Eigen::Isometry3d::Identity())));
    tmp_shapes_transform_.insert(std::pair<ShapeHandle, Eigen::Isometry3d>(shape_handle, Eigen::Isometry3d(Eigen::Isometry3d::Identity())));
  }
  return shape_handle;
}

ShapeHandle PointCloudOctomapUpdaterFast::checkShapeExist(const shapes::ShapeConstPtr& shape)
{
  std::unique_lock<std::mutex> data_lock(data_mutex);
  ShapeHandle shape_handle = 0;
  if(shape->type != shapes::MESH)
  {
    for(auto it=forget_list_.begin(); it!=forget_list_.end(); it++)
    {
      if(contain_shape_[*it]->type == shape->type)
      {
        switch (shape->type)
        {
          case shapes::BOX:
          {
            const double *size_1 = static_cast<const shapes::Box*>(shape.get())->size;
            const double *size_2 = static_cast<const shapes::Box*>(contain_shape_[*it])->size;
            double dis = fabs(size_1[0] - size_2[0]) + fabs(size_1[1] - size_2[1]) + fabs(size_1[2] - size_2[2]);
            if(dis < 0.01)
            {
              shape_handle = *it;
              forget_list_.erase(it++);
            }
            break;
          }
          case shapes::SPHERE:
          {
            double sphere_radius = static_cast<const shapes::Sphere*>(shape.get())->radius;
            if(fabs(sphere_radius - static_cast<const shapes::Sphere*>(contain_shape_[*it])->radius) < 0.005)
            {
              shape_handle = *it;
              forget_list_.erase(it++);
            }
            break;
          }
          case shapes::CYLINDER:
          {
            double cylinder_radius = static_cast<const shapes::Cylinder*>(shape.get())->radius;
            double cylinder_length = static_cast<const shapes::Cylinder*>(shape.get())->length;
            double cylinder_radius_1 = static_cast<const shapes::Cylinder*>(contain_shape_[*it])->radius;
            double cylinder_length_1 = static_cast<const shapes::Cylinder*>(contain_shape_[*it])->length;
            if((fabs(cylinder_radius - cylinder_radius_1) + fabs(cylinder_length - cylinder_length_1)) < 0.01)
            {
              shape_handle = *it;
              forget_list_.erase(it++);
            }
            break;
          }
          default:
          {
            ROS_WARN("[Octomap Updater]: Shape not exist, create a new one");
          }
        }
        if(shape_handle != 0)
          break;
      }
    }
  }
  return shape_handle;
}
void PointCloudOctomapUpdaterFast::samplePointFromMesh(const shapes::Shape* shape, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr target_cloud, int sample_num)
{
  pcl::PolygonMesh polygon_mesh;
  size_t size = static_cast<const shapes::Mesh*>(shape)->triangle_count;
  polygon_mesh.polygons.resize(size);
  pcl::PointCloud<pcl::PointXYZ> vertex_cloud;

  for(size_t i = 0; i < static_cast<const shapes::Mesh*>(shape)->vertex_count; i++)
  {
    pcl::PointXYZ p;
    p.x = static_cast<const shapes::Mesh*>(shape)->vertices[3 * i];
    p.y = static_cast<const shapes::Mesh*>(shape)->vertices[3 * i + 1];
    p.z = static_cast<const shapes::Mesh*>(shape)->vertices[3 * i + 2];
    vertex_cloud.push_back(p);
  }
  pcl::toPCLPointCloud2(vertex_cloud, polygon_mesh.cloud);

  for (size_t i = 0; i < polygon_mesh.polygons.size(); i++)
    for (int j = 0; j < 3; j++)
      polygon_mesh.polygons[i].vertices.push_back(static_cast<const shapes::Mesh*>(shape)->triangles[3*i + j]);

  vtkSmartPointer<vtkPolyData> vtk_poly_data = vtkSmartPointer<vtkPolyData>::New ();
  pcl::io::mesh2vtk (polygon_mesh, vtk_poly_data);
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mesh_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  uniform_sampling(vtk_poly_data, sample_num, false, false, *mesh_cloud);
  pcl::VoxelGrid<pcl::PointXYZRGBNormal> vg;
  vg.setInputCloud(mesh_cloud);
  vg.setLeafSize(voxel_side_length / 2, voxel_side_length / 2, voxel_side_length / 2);
  vg.filter(*target_cloud);
}

uint16_t PointCloudOctomapUpdaterFast::addCloudToMPC(const std::vector<Vector3f> &cloud)
{
  std::unique_lock<std::mutex> mpc_lock(mpc_mutex);
  std::unique_lock<std::mutex> data_lock(data_mutex);
  uint16_t cloud_id = 0;
  shape_mask_mpc->syncToHost();
  if(empty_handle.empty())
  {
    shape_mask_mpc->addCloud(cloud);
    cloud_id = shape_mask_mpc->getNumberOfPointclouds() - 1;
  }
  else
  {
    cloud_id = empty_handle.front();
    shape_mask_mpc->updatePointCloud(cloud_id, cloud);
    empty_handle.pop();
  }
  shape_mask_mpc->syncToDevice();
  shape_mask_mpc->transformSelfSubCloud(cloud_id, &init_transform);
  shape_mask_mpc->syncToHost();
  return cloud_id;
}

void PointCloudOctomapUpdaterFast::removeShape(ShapeHandle handle)
{
  ROS_DEBUG("[Octomap Updater]: Remove old shape, handle is %i", handle);
  if(contain_shape_.find(handle) != contain_shape_.end())
    contain_shape_.erase(handle);
  // shapes_transform_.erase(handle);
  if(tmp_shapes_transform_.find(handle) != tmp_shapes_transform_.end())
    tmp_shapes_transform_.erase(handle);
  std::vector<Vector3f> empty_cloud;

  std::unique_lock<std::mutex> mpc_lock(mpc_mutex);
  shape_mask_mpc->syncToHost();
  shape_mask_mpc->updatePointCloud(handle, empty_cloud);
  shape_mask_mpc->syncToDevice();
  empty_handle.push(handle);
}

void PointCloudOctomapUpdaterFast::forgetShape(ShapeHandle handle)
{
  
  ROS_DEBUG("[Octomap Updater]: Forget shape, handle is %i", handle);
  std::unique_lock<std::mutex> data_lock(data_mutex);
  if(contain_shape_.find(handle) != contain_shape_.end())
    forget_list_.insert(handle);
}

bool PointCloudOctomapUpdaterFast::updatePointCloud(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg, tf2::Stamped<tf2::Transform> &map_h_sensor)
{
  std::vector<Vector3f> point_data;
  point_data.reserve(cloud_msg->width * cloud_msg->height);
  try
  {
    for (unsigned int row = 0; row < cloud_msg->height; row += point_subsample_)
    {
      unsigned int row_c = row * cloud_msg->width;
      sensor_msgs::PointCloud2ConstIterator<float> pt_iter(*cloud_msg, "x");
      // set iterator to point at start of the current row
      pt_iter += row_c;
      for (unsigned int col = 0; col < cloud_msg->width; col += point_subsample_, pt_iter += point_subsample_)
      {
        /* check for NaN */
        if (!std::isnan(pt_iter[0]) && !std::isnan(pt_iter[1]) && !std::isnan(pt_iter[2]))
        {
          float length = sqrt(pt_iter[0]*pt_iter[0] + pt_iter[1]*pt_iter[1] + pt_iter[2]*pt_iter[2]);
          if(length < max_range_)
            point_data.push_back(Vector3f(pt_iter[0], pt_iter[1], pt_iter[2]));
        }
      }
    }
  }
  catch (...)
  {
    ROS_ERROR("[Octomap Updater]: Point Cloud Update Failed!!!");
    return false;
  }
  if(point_data.size() == 0)
    return false;

  received_cloud->update(point_data);
  received_cloud->transformSelf(&init_transform);
  pointCloudVoxelList->clearMap();
  pointCloudVoxelList->insertPointCloud(*received_cloud, eBVM_OCCUPIED);
  return true;
}

void PointCloudOctomapUpdaterFast::updateShapeMask()
{
  ROS_DEBUG("[Octomap Updater]: Into updateShapeMask");
  std::unique_lock<std::mutex> data_lock(data_mutex);
  for(auto it = contain_shape_.begin(); it != contain_shape_.end(); ++it)
  {
    // data_mutex.lock();
    if(forget_list_.find(it->first) != forget_list_.end())
    {
      // data_mutex.unlock();
      continue;
    }
    // data_mutex.unlock();

    Eigen::Isometry3d tmp, now;
    if(!getShapeTransform(it->first, now))
    {
      now = tmp_shapes_transform_[it->first];
      ROS_WARN("[Octomap Updater]: Get Shape Transform Failed!!!");
    }
    tmp = now * tmp_shapes_transform_[it->first].inverse();
    Matrix4f transformation(tmp.matrix().coeff(0,0), tmp.matrix().coeff(0,1), tmp.matrix().coeff(0,2), tmp.matrix().coeff(0,3),
                            tmp.matrix().coeff(1,0), tmp.matrix().coeff(1,1), tmp.matrix().coeff(1,2), tmp.matrix().coeff(1,3),
                            tmp.matrix().coeff(2,0), tmp.matrix().coeff(2,1), tmp.matrix().coeff(2,2), tmp.matrix().coeff(2,3),
                            tmp.matrix().coeff(3,0), tmp.matrix().coeff(3,1), tmp.matrix().coeff(3,2), tmp.matrix().coeff(3,3));
    Matrix4f transformation_1 = init_transform * transformation * inv_init_transform;

    std::unique_lock<std::mutex> mpc_lock(mpc_mutex);
    shape_mask_mpc->transformSelfSubCloud(it->first, &transformation_1);
    tmp_shapes_transform_[it->first] = now;
  }
  std::unique_lock<std::mutex> mpc_lock(mpc_mutex);
  maskVoxelList->clearMap();
  maskVoxelList->insertMetaPointCloud(*shape_mask_mpc, eBVM_SWEPT_VOLUME_START);
}

bool PointCloudOctomapUpdaterFast::getShapeTransform(ShapeHandle shape_handle, Eigen::Isometry3d& transform) const
{
  ShapeTransformCache::const_iterator it = transform_cache_.find(shape_handle);
  if (it != transform_cache_.end())
  {
    transform = it->second;
  }
  return it != transform_cache_.end();
}

void PointCloudOctomapUpdaterFast::updateMask(const sensor_msgs::PointCloud2& cloud, const Eigen::Vector3d& sensor_origin,
                                          std::vector<int>& mask)
{
}

void PointCloudOctomapUpdaterFast::emptyOctomap()
{
  try
  {
    octomap::KeySet occupied_cells;
    tf2::Vector3 point_tf;
    point_tf = map_h_sensor * tf2::Vector3(0, 0, 0);
    tree_->lockRead();
    occupied_cells.insert(tree_->coordToKey(point_tf.getX(), point_tf.getY(), point_tf.getZ()));
    tree_->unlockRead();
    tree_->lockWrite();
    tree_->clear();
    tree_->updateNode(*occupied_cells.begin(), true);
    tree_->unlockWrite();
    tree_->triggerUpdateCallback();
  }
  catch (...)
  {
    ROS_ERROR("[Octomap Updater]: Internal error while updating empty octree");
  }
}

void PointCloudOctomapUpdaterFast::cloudMsgCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
{
  double time_diff = (ros::Time::now() - cloud_msg->header.stamp).toSec();
  if(time_diff > 0.5)
  {
    ROS_WARN("[Octomap Updater]: Received point cloud message is too old, time differemce is %lf s", time_diff);
    return;
  }
  ROS_DEBUG("[Octomap Updater]: Received a new point cloud message");
  ros::WallTime start = ros::WallTime::now();
  ros::WallTime mid;
  ros::WallTime mid1;
  ros::WallTime mid2;
  ros::WallTime mid3;
  ros::WallTime mid4;
  ros::WallTime mid5;
  ros::WallTime mid6;

  if (max_update_rate_ > 0)
  {
    // ensure we are not updating the octomap representation too often
    if (ros::Time::now() - last_update_time_ <= ros::Duration(1.0 / max_update_rate_))
      return;
    last_update_time_ = ros::Time::now();
  }



  if (!updateTransformCache(cloud_msg->header.frame_id, cloud_msg->header.stamp))
  {
    ROS_WARN("[Octomap Updater]: updateTransformCache Failed");
    return;
  }

  if (monitor_->getMapFrame().empty())
    monitor_->setMapFrame(cloud_msg->header.frame_id);
  /* get transform for cloud into map frame */
  
  if (monitor_->getMapFrame() == cloud_msg->header.frame_id)
    map_h_sensor.setIdentity();
  else
  {
    if (tf_buffer_)
    {
      try
      {
        tf2::fromMsg(tf_buffer_->lookupTransform(monitor_->getMapFrame(), cloud_msg->header.frame_id,
                                                 cloud_msg->header.stamp),
                     map_h_sensor);
      }
      catch (tf2::TransformException& ex)
      {
        ROS_WARN("[Octomap Updater]: Lookup Transform Failed");
        return;
      }
    }
    else
      return;
  }
  /* compute sensor origin in map frame */
  const tf2::Vector3& sensor_origin_tf = map_h_sensor.getOrigin();
  octomap::point3d sensor_origin(sensor_origin_tf.getX(), sensor_origin_tf.getY(), sensor_origin_tf.getZ());
  Eigen::Vector3d sensor_origin_eigen(sensor_origin_tf.getX(), sensor_origin_tf.getY(), sensor_origin_tf.getZ());


  if(!updatePointCloud(cloud_msg, map_h_sensor))
  {
    emptyOctomap();
    return;
  }

  octomap::KeySet free_cells, occupied_cells, model_cells, clip_cells;
  std::unique_ptr<sensor_msgs::PointCloud2> filtered_cloud;

  // We only use these iterators if we are creating a filtered_cloud for
  // publishing. We cannot default construct these, so we use unique_ptr's
  // to defer construction
  std::unique_ptr<sensor_msgs::PointCloud2Iterator<float> > iter_filtered_x;
  std::unique_ptr<sensor_msgs::PointCloud2Iterator<float> > iter_filtered_y;
  std::unique_ptr<sensor_msgs::PointCloud2Iterator<float> > iter_filtered_z;

  if (!filtered_cloud_topic_.empty())
  {
    filtered_cloud.reset(new sensor_msgs::PointCloud2());
    filtered_cloud->header = cloud_msg->header;
    sensor_msgs::PointCloud2Modifier pcd_modifier(*filtered_cloud);
    pcd_modifier.setPointCloud2FieldsByString(1, "xyz");
    pcd_modifier.resize(cloud_msg->width * cloud_msg->height);

    // we have created a filtered_out, so we can create the iterators now
    iter_filtered_x.reset(new sensor_msgs::PointCloud2Iterator<float>(*filtered_cloud, "x"));
    iter_filtered_y.reset(new sensor_msgs::PointCloud2Iterator<float>(*filtered_cloud, "y"));
    iter_filtered_z.reset(new sensor_msgs::PointCloud2Iterator<float>(*filtered_cloud, "z"));
  }
  size_t filtered_cloud_size = 0;

  tree_->lockWrite();
  tree_->clear();
  tree_->unlockWrite();

  {
    std::unique_lock<std::mutex> data_lock(data_mutex);
    for(auto it=forget_list_.begin(); it!=forget_list_.end();)
    {
      removeShape(*it);
      forget_list_.erase(it++);
    }
  }
  
  /* mask out points on the robot */
  mid = ros::WallTime::now();
  updateShapeMask();
  std::vector<Vector3ui> filtered_point;
  pointCloudVoxelList->as<voxellist::CountingVoxelList>()->subtractFromCountingVoxelList(
        maskVoxelList->as<voxellist::BitVectorVoxelList>(), Vector3i());
  pointCloudVoxelList->copyCoordsToHost(filtered_point);
  mid1 = ros::WallTime::now();
  
  if(filtered_point.size() == 0)
  {
    emptyOctomap();
    return;
  }

  tree_->lockRead();
  
  try
  {
    /* do ray tracing to find which cells this point cloud indicates should be free, and which it indicates
     * should be occupied */
    Vector3f point;
    tf2::Vector3 point_tf;
    for(auto iter_filtered = filtered_point.begin(); iter_filtered!=filtered_point.end(); ++iter_filtered)
    {
      point = inv_init_transform * Vector3f((float)iter_filtered->x * voxel_side_length ,
                                  (float)iter_filtered->y * voxel_side_length , (float)iter_filtered->z * voxel_side_length);
      point_tf = map_h_sensor * tf2::Vector3(point.x, point.y, point.z);
      occupied_cells.insert(tree_->coordToKey(point_tf.getX(), point_tf.getY(), point_tf.getZ()));
      if (filtered_cloud)
      {
        **iter_filtered_x = point.x;
        **iter_filtered_y = point.y;
        **iter_filtered_z = point.z;
        ++filtered_cloud_size;
        ++*iter_filtered_x;
        ++*iter_filtered_y;
        ++*iter_filtered_z;
      }
    }

    /* compute the free cells along each ray that ends at an occupied cell */
    // for (octomap::KeySet::iterator it = occupied_cells.begin(), end = occupied_cells.end(); it != end; ++it)
    //   if (tree_->computeRayKeys(sensor_origin, tree_->keyToCoord(*it), key_ray_))
    //     free_cells.insert(key_ray_.begin(), key_ray_.end());

    // /* compute the free cells along each ray that ends at a model cell */
    // for (octomap::KeySet::iterator it = model_cells.begin(), end = model_cells.end(); it != end; ++it)
    //   if (tree_->computeRayKeys(sensor_origin, tree_->keyToCoord(*it), key_ray_))
    //     free_cells.insert(key_ray_.begin(), key_ray_.end());

    // /* compute the free cells along each ray that ends at a clipped cell */
    // for (octomap::KeySet::iterator it = clip_cells.begin(), end = clip_cells.end(); it != end; ++it)
    //   if (tree_->computeRayKeys(sensor_origin, tree_->keyToCoord(*it), key_ray_))
    //     free_cells.insert(key_ray_.begin(), key_ray_.end());
  }
  catch (...)
  {
    tree_->unlockRead();
    return;
  }
  tree_->unlockRead();

  // /* cells that overlap with the model are not occupied */
  // for (octomap::KeySet::iterator it = model_cells.begin(), end = model_cells.end(); it != end; ++it)
  //   occupied_cells.erase(*it);

  /* occupied cells are not free */
  // for (octomap::KeySet::iterator it = occupied_cells.begin(), end = occupied_cells.end(); it != end; ++it)
  //   free_cells.erase(*it);

  tree_->lockWrite();

  try
  {
    /* mark free cells only if not seen occupied in this cloud */
    // for (octomap::KeySet::iterator it = free_cells.begin(), end = free_cells.end(); it != end; ++it)
    //   tree_->updateNode(*it, false);

    /* now mark all occupied cells */
    for (octomap::KeySet::iterator it = occupied_cells.begin(), end = occupied_cells.end(); it != end; ++it)
      tree_->updateNode(*it, true);

    // // set the logodds to the minimum for the cells that are part of the model
    // const float lg = tree_->getClampingThresMinLog() - tree_->getClampingThresMaxLog();
    // for (octomap::KeySet::iterator it = model_cells.begin(), end = model_cells.end(); it != end; ++it)
    //   tree_->updateNode(*it, lg);
  }
  catch (...)
  {
    ROS_ERROR("[Octomap Updater]: Internal error while updating octree");
  }
  tree_->unlockWrite();
  tree_->triggerUpdateCallback();
  mid2 = ros::WallTime::now();
  if (filtered_cloud)
  {
    sensor_msgs::PointCloud2Modifier pcd_modifier(*filtered_cloud);
    pcd_modifier.resize(filtered_cloud_size);
    filtered_cloud_publisher_.publish(*filtered_cloud);
  }
  if((ros::WallTime::now() - start).toSec() * 1000.0 > 100)
    ROS_WARN("[Octomap Updater]: Slow octomap update, spend %lf ms", (ros::WallTime::now() - start).toSec() * 1000.0);
  ROS_DEBUG("[Octomap Updater]: Processed point cloud in %lf ms and %lf ms and %lf ms and %lf ms\n", (mid - start).toSec() * 1000.0,  (mid1 - mid).toSec() * 1000.0, (mid2 - mid1).toSec() * 1000.0, (ros::WallTime::now() - start).toSec() * 1000.0);
  // gvl->visualizeMap("maskVoxelList");
  // gvl->visualizeMap("pointCloudVoxelList");
}
}  // namespace occupancy_map_monitor

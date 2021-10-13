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
#include <moveit/occupancy_map_monitor/occupancy_map_monitor.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Transform.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <XmlRpcException.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
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
    ROS_ERROR("XmlRpc Exception: %s", ex.getMessage().c_str());
    return false;
  }

  return true;
}

bool PointCloudOctomapUpdaterFast::initialize()
{
  next_handle_ = 0;
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
  Vector3ui map_dim(256, 256, 256);
  map_dimensions =  map_dim;
  voxel_side_length = 0.01f; // 1 cm voxel size

  /*
   * First, we generate an API class, which defines the
   * volume of our space and the resolution.
   * Be careful here! The size is limited by the memory
   * of your GPU. Even if an empty Octree is small, a
   * Voxelmap will always require the full memory.
   */

  icl_core::logging::initialize();
  const Vector3f camera_offsets(1.2, 0.7, -2.7);
  float roll = icl_core::config::paramOptDefault<float>("roll", 0.0f) * 3.141592f / 180.0f;
  float pitch = icl_core::config::paramOptDefault<float>("pitch", 0.0f) * 3.141592f / 180.0f;
  float yaw = icl_core::config::paramOptDefault<float>("yaw", 0.0f) * 3.141592f / 180.0f;
  tf = Matrix4f::createFromRotationAndTranslation(Matrix3f::createFromRPY(roll, pitch, yaw), camera_offsets);

  gvl = GpuVoxels::getInstance();
  gvl->initialize(map_dimensions.x, map_dimensions.y, map_dimensions.z, voxel_side_length); // ==> 200 Voxels, each one is 1 cm in size so the map represents 20x20x20 centimeter

  //Vis Helper
  gvl->addPrimitives(primitive_array::ePRIM_SPHERE, "measurementPoints");

  // Add a map:
  gvl->addMap(MT_BITVECTOR_VOXELLIST, "myHandVoxellist");
  gvl->addMap(MT_BITVECTOR_VOXELLIST, "maskVoxelList");
  gvl->addMap(MT_COUNTING_VOXELLIST, "pointCloudVoxelList");
  maskVoxelList = dynamic_pointer_cast<BitVectorVoxelList>(gvl->getMap("maskVoxelList"));
  pointCloudVoxelList = dynamic_pointer_cast<CountingVoxelList>(gvl->getMap("pointCloudVoxelList"));

  // And a robot, generated from a ROS URDF file:
  root_nh_.getParam("/robot_description_voxels/urdf_path", urdf_path_);
  gvl->addRobot("myUrdfRobot", urdf_path_, false);

  // update the robot joints:
  gvl->setRobotConfiguration("myUrdfRobot", myRobotJointValues);
  // insert the robot into the map:
  gvl->insertRobotIntoMap("myUrdfRobot", "myHandVoxellist", eBVM_OCCUPIED);
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
    ROS_INFO("Listening to '%s' using message filter with target frame '%s'", point_cloud_topic_.c_str(),
             point_cloud_filter_->getTargetFramesString().c_str());
  }
  else
  {
    point_cloud_subscriber_->registerCallback(boost::bind(&PointCloudOctomapUpdaterFast::cloudMsgCallback, this, _1));
    ROS_INFO("Listening to '%s'", point_cloud_topic_.c_str());
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
  ShapeHandle h = 0;
  if (shape_mask_)
  {
    // h = shape_mask_->addShape(shape, scale_, padding_);
    h = next_handle_;
    next_handle_ += 1;
    contain_shape_.insert(std::pair<ShapeHandle, shapes::Shape*>(h, shape->clone()));
    shapes_transform_.insert(std::pair<ShapeHandle, Eigen::Isometry3d>(h, Eigen::Isometry3d(Eigen::Isometry3d::Identity())));
    tmp_shapes_transform_.insert(std::pair<ShapeHandle, Eigen::Isometry3d>(h, Eigen::Isometry3d(Eigen::Isometry3d::Identity())));

    float side_length;
    gvl->getVoxelSideLength(side_length);
    float delta = side_length / 1.0f;
    for(auto it = contain_shape_.begin(); it != contain_shape_.end(); ++it)
    {
      if(it->second->type == shapes::BOX)
      { 
        ROS_DEBUG("Into updateShapeMask shapes::BOX");
        const double *size = static_cast<const shapes::Box*>(it->second)->size;
        Vector3f center_min = Vector3f(-1*size[0]/2, -1*size[1]/2, -1*size[2]/2);
        Vector3f center_max = Vector3f(size[0]/2, size[1]/2, size[2]/2);
        std::vector<Vector3f> box_coordinates = geometry_generation::createBoxOfPoints(center_min, center_max, delta);
        cloud_vector.push_back(box_coordinates);
        // maskVoxelList->insertCoordinateList(box_coordinates, eBVM_OCCUPIED);
        ROS_DEBUG("Into updateShapeMask shapes::BOX end");
      }
      else if(it->second->type == shapes::SPHERE)
      {
        ROS_DEBUG("Into updateShapeMask shapes::SPHERE");
        double sphere_radius = static_cast<const shapes::Sphere*>(it->second)->radius;
        Vector3f sphere_center = Vector3f(0, 0, 0);
        std::vector<Vector3f> sphere_coordinates = geometry_generation::createSphereOfPoints(sphere_center, sphere_radius, delta);
        cloud_vector.push_back(sphere_coordinates);
        // maskVoxelList->insertCoordinateList(sphere_coordinates, eBVM_OCCUPIED);
        ROS_DEBUG("Into updateShapeMask shapes::SPHERE end");
      }
      else if(it->second->type == shapes::CYLINDER)
      {
        ROS_DEBUG("Into updateShapeMask shapes::CYLINDER");
        double cylinder_radius = static_cast<const shapes::Cylinder*>(it->second)->radius;
        double cylinder_length = static_cast<const shapes::Cylinder*>(it->second)->length;
        Vector3f cylinder_center = Vector3f(0, 0, 0);
        std::vector<Vector3f> cylinder_coordinates = geometry_generation::createCylinderOfPoints(cylinder_center, cylinder_radius, cylinder_length, delta);
        cloud_vector.push_back(cylinder_coordinates);
        // maskVoxelList->insertCoordinateList(cylinder_coordinates, eBVM_OCCUPIED);
        ROS_DEBUG("Into updateShapeMask shapes::CYLINDER end");
      }
      else if(it->second->type == shapes::MESH)
      {
        ROS_ERROR("Into updateShapeMask shapes::MESH");
        pcl::PolygonMesh polygon_mesh;

        // sensor_msgs::PointCloud2Modifier pcd_modifier(polygon_mesh.cloud);

        // size_t size = sizeof(static_cast<const shapes::Mesh*>(it->second)->vertices) / sizeof(double) / 3;
        
        // pcd_modifier.resize(size);

        // std::cout << "polys: " << sizeof(static_cast<const shapes::Mesh*>(it->second)->triangles)/sizeof(unsigned int) << " vertices: " << pcd_modifier.size() << "\n";

        // sensor_msgs::PointCloud2ConstIterator<float> pt_iter(polygon_mesh.cloud, "x");
        
        // for(size_t i = 0; i < size ; i++, ++pt_iter){
        //   pt_iter[0] = static_cast<const shapes::Mesh*>(it->second)->vertices[3*i];
        //   pt_iter[1] = static_cast<const shapes::Mesh*>(it->second)->vertices[3*i + 1];
        //   pt_iter[2] = static_cast<const shapes::Mesh*>(it->second)->vertices[3*i + 2];
        // }
        std::cout<<static_cast<const shapes::Mesh*>(it->second)->triangle_count<<", "<<static_cast<const shapes::Mesh*>(it->second)->vertex_count<<std::endl;
        size_t size = static_cast<const shapes::Mesh*>(it->second)->triangle_count;
        ROS_ERROR("Into updateShapeMask shapes::MESH 1");
        polygon_mesh.polygons.resize(size);
        std::cout<<"polygon_mesh.polygons.size = "<<polygon_mesh.polygons.size()<<std::endl;
        size = static_cast<const shapes::Mesh*>(it->second)->vertex_count;
        polygon_mesh.cloud.data.resize(size);
        polygon_mesh.cloud.height = 1;
        polygon_mesh.cloud.width = size;
        std::cout<<"polygon_mesh.cloud.data.size() = "<<polygon_mesh.cloud.data.size()<<std::endl;
        for(size_t i = 0; i < polygon_mesh.cloud.data.size(); i++)
        {
          // for (int j = 0; j < 3; j++)
          // {
          // std::cout<<i<<", "<<j<<std::endl;
          polygon_mesh.cloud.data[i] = static_cast<const shapes::Mesh*>(it->second)->vertices[i];
          // }
        }
        for (size_t i = 0; i < polygon_mesh.polygons.size(); i++)
        {
          for (int j = 0; j < 3; j++)
          {
            // polygon_mesh.polygons[i].vertices.push_back(static_cast<const shapes::Mesh*>(it->second)->vertices
            //                                            [static_cast<const shapes::Mesh*>(it->second)->triangles[3*i + j]]);
            polygon_mesh.polygons[i].vertices.push_back(static_cast<const shapes::Mesh*>(it->second)->triangles[3*i + j]);
            std::cout<<static_cast<const shapes::Mesh*>(it->second)->triangles[3*i + j]<<std::endl;
          }
        }
        ROS_ERROR("Into updateShapeMask shapes::MESH 3");
        vtkSmartPointer<vtkPolyData> vtk_poly_data = vtkSmartPointer<vtkPolyData>::New ();
        pcl::io::mesh2vtk (polygon_mesh, vtk_poly_data);
        ROS_ERROR("Into updateShapeMask shapes::MESH 4");
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        uniform_sampling(vtk_poly_data, 1000, false, false, *model_cloud);
        ROS_ERROR("Into updateShapeMask shapes::MESH end");
        std::cout<<"cloud_out->width = "<<model_cloud->width<<std::endl;
        std::vector<Vector3f> model_coordinates;
        for(auto it=model_cloud->begin(); it!=model_cloud->end(); ++it)
        {
          model_coordinates.push_back(Vector3f(it->x, it->y, it->z));
        }
        cloud_vector.push_back(model_coordinates);
      }
      else
      {
        ROS_DEBUG("Into updateShapeMask shapes::UNKNOWN");
        contain_shape_.erase(it);
        ROS_DEBUG("Creating body from shape: Unknown shape type");
      }
    }
    mpc = new MetaPointCloud(cloud_vector);
  }
  return h;
}

void PointCloudOctomapUpdaterFast::updateShapeMask()
{
  ROS_DEBUG("Into updateShapeMask");
  maskVoxelList->clearMap();
  //the upper one is always the on generated by coordinates
  //box, by coordinates
  // std::vector<std::vector<Vector3f>> cloud_vector;
  // float side_length;
  // gvl->getVoxelSideLength(side_length);
  // float delta = side_length / 1.0f;
  // for(auto it = contain_shape_.begin(); it != contain_shape_.end(); ++it)
  // {
  //   if(it->second->type == shapes::BOX)
  //   { 
  //     ROS_DEBUG("Into updateShapeMask shapes::BOX");
  //     const double *size = static_cast<const shapes::Box*>(it->second)->size;
  //     Vector3f center_min = Vector3f(-1*size[0]/2, -1*size[1]/2, -1*size[2]/2);
  //     Vector3f center_max = Vector3f(size[0]/2, size[1]/2, size[2]/2);
  //     std::vector<Vector3f> box_coordinates = geometry_generation::createBoxOfPoints(center_min, center_max, delta);
  //     cloud_vector.push_back(box_coordinates);
  //     // maskVoxelList->insertCoordinateList(box_coordinates, eBVM_OCCUPIED);
  //     ROS_DEBUG("Into updateShapeMask shapes::BOX end");
  //   }
  //   else if(it->second->type == shapes::SPHERE)
  //   {
  //     ROS_DEBUG("Into updateShapeMask shapes::SPHERE");
  //     double sphere_radius = static_cast<const shapes::Sphere*>(it->second)->radius;
  //     Vector3f sphere_center = Vector3f(0, 0, 0);
  //     std::vector<Vector3f> sphere_coordinates = geometry_generation::createSphereOfPoints(sphere_center, sphere_radius, delta);
  //     cloud_vector.push_back(sphere_coordinates);
  //     // maskVoxelList->insertCoordinateList(sphere_coordinates, eBVM_OCCUPIED);
  //     ROS_DEBUG("Into updateShapeMask shapes::SPHERE end");
  //   }
  //   else if(it->second->type == shapes::CYLINDER)
  //   {
  //     ROS_DEBUG("Into updateShapeMask shapes::CYLINDER");
  //     double cylinder_radius = static_cast<const shapes::Cylinder*>(it->second)->radius;
  //     double cylinder_length = static_cast<const shapes::Cylinder*>(it->second)->length;
  //     Vector3f cylinder_center = Vector3f(0, 0, 0);
  //     std::vector<Vector3f> cylinder_coordinates = geometry_generation::createCylinderOfPoints(cylinder_center, cylinder_radius, cylinder_length, delta);
  //     cloud_vector.push_back(cylinder_coordinates);
  //     // maskVoxelList->insertCoordinateList(cylinder_coordinates, eBVM_OCCUPIED);
  //     ROS_DEBUG("Into updateShapeMask shapes::CYLINDER end");
  //   }
  //   else if(it->second->type == shapes::MESH)
  //   {
  //     ROS_ERROR("Into updateShapeMask shapes::MESH");
  //     pcl::PolygonMesh polygon_mesh;

  //     // sensor_msgs::PointCloud2Modifier pcd_modifier(polygon_mesh.cloud);

  //     // size_t size = sizeof(static_cast<const shapes::Mesh*>(it->second)->vertices) / sizeof(double) / 3;
      
  //     // pcd_modifier.resize(size);

  //     // std::cout << "polys: " << sizeof(static_cast<const shapes::Mesh*>(it->second)->triangles)/sizeof(unsigned int) << " vertices: " << pcd_modifier.size() << "\n";

  //     // sensor_msgs::PointCloud2ConstIterator<float> pt_iter(polygon_mesh.cloud, "x");
      
  //     // for(size_t i = 0; i < size ; i++, ++pt_iter){
  //     //   pt_iter[0] = static_cast<const shapes::Mesh*>(it->second)->vertices[3*i];
  //     //   pt_iter[1] = static_cast<const shapes::Mesh*>(it->second)->vertices[3*i + 1];
  //     //   pt_iter[2] = static_cast<const shapes::Mesh*>(it->second)->vertices[3*i + 2];
  //     // }
  //     std::cout<<static_cast<const shapes::Mesh*>(it->second)->triangle_count<<", "<<static_cast<const shapes::Mesh*>(it->second)->vertex_count<<std::endl;
  //     size_t size = static_cast<const shapes::Mesh*>(it->second)->triangle_count;
  //     ROS_ERROR("Into updateShapeMask shapes::MESH 1");
  //     polygon_mesh.polygons.resize(size);
  //     std::cout<<"polygon_mesh.polygons.size = "<<polygon_mesh.polygons.size()<<std::endl;
  //     size = static_cast<const shapes::Mesh*>(it->second)->vertex_count;
  //     polygon_mesh.cloud.data.resize(size);
  //     polygon_mesh.cloud.height = 1;
  //     polygon_mesh.cloud.width = size;
  //     std::cout<<"polygon_mesh.cloud.data.size() = "<<polygon_mesh.cloud.data.size()<<std::endl;
  //     for(size_t i = 0; i < polygon_mesh.cloud.data.size(); i++)
  //     {
  //       // for (int j = 0; j < 3; j++)
  //       // {
  //       // std::cout<<i<<", "<<j<<std::endl;
  //       polygon_mesh.cloud.data[i] = static_cast<const shapes::Mesh*>(it->second)->vertices[i];
  //       // }
  //     }
  //     for (size_t i = 0; i < polygon_mesh.polygons.size(); i++)
  //     {
  //       for (int j = 0; j < 3; j++)
  //       {
  //         // polygon_mesh.polygons[i].vertices.push_back(static_cast<const shapes::Mesh*>(it->second)->vertices
  //         //                                            [static_cast<const shapes::Mesh*>(it->second)->triangles[3*i + j]]);
  //         polygon_mesh.polygons[i].vertices.push_back(static_cast<const shapes::Mesh*>(it->second)->triangles[3*i + j]);
  //         std::cout<<static_cast<const shapes::Mesh*>(it->second)->triangles[3*i + j]<<std::endl;
  //       }
  //     }
  //     ROS_ERROR("Into updateShapeMask shapes::MESH 3");
  //     vtkSmartPointer<vtkPolyData> vtk_poly_data = vtkSmartPointer<vtkPolyData>::New ();
  //     pcl::io::mesh2vtk (polygon_mesh, vtk_poly_data);
  //     ROS_ERROR("Into updateShapeMask shapes::MESH 4");
  //     pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  //     uniform_sampling(vtk_poly_data, 1000, false, false, *model_cloud);
  //     ROS_ERROR("Into updateShapeMask shapes::MESH end");
  //     std::cout<<"cloud_out->width = "<<model_cloud->width<<std::endl;
  //     std::vector<Vector3f> model_coordinates;
  //     for(auto it=model_cloud->begin(); it!=model_cloud->end(); ++it)
  //     {
  //       model_coordinates.push_back(Vector3f(it->x, it->y, it->z));
  //     }
  //     cloud_vector.push_back(model_coordinates);
  //   }
  //   else
  //   {
  //     ROS_DEBUG("Into updateShapeMask shapes::UNKNOWN");
  //     contain_shape_.erase(it);
  //     ROS_DEBUG("Creating body from shape: Unknown shape type");
  //   }
  // }
  // mpc = new MetaPointCloud(cloud_vector);
  // MetaPointCloud mpc(cloud_vector);
  
  int i = 0;
  for(auto it = contain_shape_.begin(); it != contain_shape_.end(); ++it)
  {
    Eigen::Isometry3d tmp;
    getShapeTransform(it->first, tmp);
    Matrix4f transformation(tmp.matrix().coeff(0,0), tmp.matrix().coeff(0,1), tmp.matrix().coeff(0,2), tmp.matrix().coeff(0,3),
                            tmp.matrix().coeff(1,0), tmp.matrix().coeff(1,1), tmp.matrix().coeff(1,2), tmp.matrix().coeff(1,3),
                            tmp.matrix().coeff(2,0), tmp.matrix().coeff(2,1), tmp.matrix().coeff(2,2), tmp.matrix().coeff(2,3),
                            tmp.matrix().coeff(3,0), tmp.matrix().coeff(3,1), tmp.matrix().coeff(3,2), tmp.matrix().coeff(3,3));
    mpc->transformSelfSubCloud(i, &transformation);
    i++;
  }
  maskVoxelList->insertMetaPointCloud(*mpc, eBVM_SWEPT_VOLUME_START);
}

void PointCloudOctomapUpdaterFast::forgetShape(ShapeHandle handle)
{
  if (shape_mask_)
    // shape_mask_->removeShape(handle);
    contain_shape_.erase(handle);
}

bool PointCloudOctomapUpdaterFast::getShapeTransform(ShapeHandle h, Eigen::Isometry3d& transform) const
{
  ShapeTransformCache::const_iterator it = transform_cache_.find(h);
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

void PointCloudOctomapUpdaterFast::cloudMsgCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
{
  ROS_DEBUG("Received a new point cloud message");
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

  if (monitor_->getMapFrame().empty())
    monitor_->setMapFrame(cloud_msg->header.frame_id);

  /* get transform for cloud into map frame */
  tf2::Stamped<tf2::Transform> map_h_sensor;
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
        ROS_ERROR_STREAM("Transform error of sensor data: " << ex.what() << "; quitting callback");
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

  if (!updateTransformCache(cloud_msg->header.frame_id, cloud_msg->header.stamp))
    return;

  /* mask out points on the robot */
  shape_mask_->maskContainment(*cloud_msg, sensor_origin_eigen, 0.0, max_range_, mask_);
  updateMask(*cloud_msg, sensor_origin_eigen, mask_);
  updateShapeMask();
  octomap::KeySet free_cells, occupied_cells, model_cells, clip_cells;
  std::unique_ptr<sensor_msgs::PointCloud2> filtered_cloud;
  std::unique_ptr<sensor_msgs::PointCloud2> filtered_cloud_;

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


  tree_->lockRead();

  try
  {
    /* do ray tracing to find which cells this point cloud indicates should be free, and which it indicates
     * should be occupied */
    for (unsigned int row = 0; row < cloud_msg->height; row += point_subsample_)
    {
      unsigned int row_c = row * cloud_msg->width;
      sensor_msgs::PointCloud2ConstIterator<float> pt_iter(*cloud_msg, "x");
      // set iterator to point at start of the current row
      pt_iter += row_c;

      for (unsigned int col = 0; col < cloud_msg->width; col += point_subsample_, pt_iter += point_subsample_)
      {
        // if (mask_[row_c + col] == point_containment_filter::ShapeMask::CLIP)
        //  continue;

        /* check for NaN */
        if (!std::isnan(pt_iter[0]) && !std::isnan(pt_iter[1]) && !std::isnan(pt_iter[2]))
        {
          /* transform to map frame */
          tf2::Vector3 point_tf = map_h_sensor * tf2::Vector3(pt_iter[0], pt_iter[1], pt_iter[2]);

          /* occupied cell at ray endpoint if ray is shorter than max range and this point
             isn't on a part of the robot*/
          if (mask_[row_c + col] == point_containment_filter::ShapeMask::INSIDE)
            model_cells.insert(tree_->coordToKey(point_tf.getX(), point_tf.getY(), point_tf.getZ()));
          else if (mask_[row_c + col] == point_containment_filter::ShapeMask::CLIP)
            clip_cells.insert(tree_->coordToKey(point_tf.getX(), point_tf.getY(), point_tf.getZ()));
          else
          {
            occupied_cells.insert(tree_->coordToKey(point_tf.getX(), point_tf.getY(), point_tf.getZ()));
            // build list of valid points if we want to publish them
            if (filtered_cloud)
            {
              **iter_filtered_x = pt_iter[0];
              **iter_filtered_y = pt_iter[1];
              **iter_filtered_z = pt_iter[2];
              ++filtered_cloud_size;
              ++*iter_filtered_x;
              ++*iter_filtered_y;
              ++*iter_filtered_z;
            }
          }
        }
      }
    }

    /* compute the free cells along each ray that ends at an occupied cell */
    for (octomap::KeySet::iterator it = occupied_cells.begin(), end = occupied_cells.end(); it != end; ++it)
      if (tree_->computeRayKeys(sensor_origin, tree_->keyToCoord(*it), key_ray_))
        free_cells.insert(key_ray_.begin(), key_ray_.end());

    /* compute the free cells along each ray that ends at a model cell */
    for (octomap::KeySet::iterator it = model_cells.begin(), end = model_cells.end(); it != end; ++it)
      if (tree_->computeRayKeys(sensor_origin, tree_->keyToCoord(*it), key_ray_))
        free_cells.insert(key_ray_.begin(), key_ray_.end());

    /* compute the free cells along each ray that ends at a clipped cell */
    for (octomap::KeySet::iterator it = clip_cells.begin(), end = clip_cells.end(); it != end; ++it)
      if (tree_->computeRayKeys(sensor_origin, tree_->keyToCoord(*it), key_ray_))
        free_cells.insert(key_ray_.begin(), key_ray_.end());
  }
  catch (...)
  {
    tree_->unlockRead();
    return;
  }

  tree_->unlockRead();

  /* cells that overlap with the model are not occupied */
  for (octomap::KeySet::iterator it = model_cells.begin(), end = model_cells.end(); it != end; ++it)
    occupied_cells.erase(*it);

  /* occupied cells are not free */
  for (octomap::KeySet::iterator it = occupied_cells.begin(), end = occupied_cells.end(); it != end; ++it)
    free_cells.erase(*it);

  tree_->lockWrite();

  try
  {
    /* mark free cells only if not seen occupied in this cloud */
    for (octomap::KeySet::iterator it = free_cells.begin(), end = free_cells.end(); it != end; ++it)
      tree_->updateNode(*it, false);

    /* now mark all occupied cells */
    for (octomap::KeySet::iterator it = occupied_cells.begin(), end = occupied_cells.end(); it != end; ++it)
      tree_->updateNode(*it, true);

    // set the logodds to the minimum for the cells that are part of the model
    const float lg = tree_->getClampingThresMinLog() - tree_->getClampingThresMaxLog();
    for (octomap::KeySet::iterator it = model_cells.begin(), end = model_cells.end(); it != end; ++it)
      tree_->updateNode(*it, lg);
  }
  catch (...)
  {
    ROS_ERROR("Internal error while updating octree");
  }
  tree_->unlockWrite();
  ROS_DEBUG("Processed point cloud in %lf ms and %lf ms and %lf ms and %lf ms", (mid5 - mid).toSec() * 1000.0,  (mid3 - mid5).toSec() * 1000.0, (mid4 - mid3).toSec() * 1000.0, (ros::WallTime::now() - mid4).toSec() * 1000.0);
  tree_->triggerUpdateCallback();

  if (filtered_cloud)
  {
    sensor_msgs::PointCloud2Modifier pcd_modifier(*filtered_cloud);
    pcd_modifier.resize(filtered_cloud_size);
    filtered_cloud_publisher_.publish(*filtered_cloud);
  }
}
// void PointCloudOctomapUpdaterFast::robotStateCallback(const robot_state::RobotState& state)
// {
//   // 0.09 ~ 1.2 ms  avg 0.1 ms
//   gvl->clearMap("myHandVoxellist");
//   for (int i=0; i<state.getVariableCount(); i++)
//   {
//     myRobotJointValues[state.getVariableNames()[i]] = state.getVariablePositions()[i];
//   }
//   // update the robot joints:  spend 0.15 ~ 2 ms avg 0.16 ms
//   gvl->setRobotConfiguration("myUrdfRobot", myRobotJointValues);
//   // insert the robot into the map: spend 0.02 ~ 0.04 ms
//   gvl->insertRobotIntoMap("myUrdfRobot", "myHandVoxellist", eBVM_OCCUPIED);
//   gvl->visualizeMap("myHandVoxellist");
// }
}  // namespace occupancy_map_monitor

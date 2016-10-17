#ifndef KINECT_OBJECT_H
#define KINECT_OBJECT_H

#include <vector>
#include <opencv2\opencv.hpp>
#include <pcl\visualization\pcl_visualizer.h>

using namespace std;
using namespace cv;
using namespace pcl;

typedef struct KinectObject{

	PointXYZRGB centroid;
	vector<double> colorHistogram;

	//bool operator==(const FloorObject& obj) const{ return (box.center == obj.box.center && box.size == obj.box.size); }


} KinectObject;

#endif
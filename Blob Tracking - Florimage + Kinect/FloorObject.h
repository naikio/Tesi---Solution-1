#ifndef FLOOR_OBJECT_H
#define FLOOR_OBJECT_H

#include <vector>
#include <opencv2\opencv.hpp>
#include <pcl\visualization\pcl_visualizer.h>

using namespace std;
using namespace cv;
using namespace pcl;

enum State { alpha, stable, hidden, dead };

typedef struct FloorObject{

	vector<Point> contour;
	RotatedRect box;
	Point2f averagePosition;
	Scalar color;
	int visualCounter = 0;
	State state = alpha;
	PointXYZRGB centroid;
	Mat redHistogram = Mat(Size(1, 256), CV_32FC1, Scalar(0));
	Mat blueHistogram = Mat(Size(1, 256), CV_32FC1, Scalar(0));
	Mat greenHistogram = Mat(Size(1, 256), CV_32FC1, Scalar(0));

	bool operator==(const FloorObject& obj) const{ return (box.center == obj.box.center && box.size == obj.box.size); }

	bool isDead(const FloorObject& obj) const{ return (obj.state == dead); }

} FloorObject;

#endif
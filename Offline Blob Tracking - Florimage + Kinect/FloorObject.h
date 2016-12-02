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
	vector<Point2i> positions;
	Scalar color;
	int visualCounter = 1;
	State state = alpha;
	PointXYZRGB centroid;
	vector<float> bgrHistogram;
	double confidence;

	bool operator==(const FloorObject& obj) const{ return (box.center == obj.box.center && box.size == obj.box.size); }

	bool isDead(const FloorObject& obj) const{ return (obj.state == dead); }

} FloorObject;

#endif
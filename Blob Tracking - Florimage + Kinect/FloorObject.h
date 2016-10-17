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
	vector<int> redHistogram = vector<int>(256, 0);
	vector<int> blueHistogram = vector<int>(256, 0);
	vector<int> greenHistogram = vector<int>(256, 0);

	bool operator==(const FloorObject& obj) const{ return (box.center == obj.box.center && box.size == obj.box.size); }

	bool isDead(const FloorObject& obj) const{ return (obj.state == dead); }

} FloorObject;

#endif
#ifndef FLOOR_OBJECT_H
#define FLOOR_OBJECT_H

#include <vector>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

enum State { alpha, stable, hidden, dead };

typedef struct FloorObject{

	vector<Point> contour;
	RotatedRect box;
	Point2f averagePosition;
	Scalar color;
	int visualCounter = 0;
	State state = alpha;

	bool operator==(const FloorObject& obj) const{ return (box.center == obj.box.center && box.size == obj.box.size); }

	bool isDead(const FloorObject& obj) const{ return (obj.state == dead); }

} FloorObject;

#endif
#include "FloorObject.h"

#ifndef BLOB_DISTANCE_H
#define BLOB_DISTANCE_H

typedef struct BlobDistance{
	double distance;
	FloorObject previous;
	FloorObject current;

	//Operator <  
	//used for sorting
	bool operator<(const BlobDistance& bDist) const{ return distance < bDist.distance; }

	bool has_same_previous_blob(FloorObject previous_blob) const { return previous_blob == previous; }

} BlobDistance;

#endif
#include "stdafx.h"

#define NOMINMAX
#define _WINSOCK_DEPRECATED_NO_WARNINGS

#define BGD_SUBTRACTION
//#define CROP_BOX_FILTERING
//#define PLANAR_PROJECTION false
//#define PCL_VISUALIZER
//#define DRAW_TILES
//#define GET_PLANE_COEFFICIENTS  //used to search for floor plane coefficients (useful to retrieve plane coefficients the 1st time)

#define BEST_MATCH_FIRST 1
#define HUNGARIAN_MIN_COST 2

#define ALPHA_PARAM 9
#define DEATH_PARAM 8
#define T_0 0
#define T_MIN 0.0
#define LAMBDA_UP 0.75 //must be a value between 0 and 1
#define LAMBDA_DOWN 0.95 //must be a value between 0 and 1

#define BLOB_RADIUS 0.4

#include <Windows.h>
//OPENCV 2.4.10
#include <opencv2\opencv.hpp>
//Kinect framework 
#include <Kinect.h>
#include "acquisitionkinect2.h"
//PCL inclusions
#include <pcl\visualization\cloud_viewer.h>
#include <pcl\visualization\pcl_visualizer.h>
#include <pcl/filters/crop_box.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/project_inliers.h>

#pragma comment(lib,"ws2_32.lib") //Winsock Library
#include <boost/thread.hpp>
#include <boost/chrono.hpp>

//Custom STRUCTs
#include "FloorObject.h"
#include "BlobDistance.h"

//Custom Algorithms
#include "bipartite-mincost.h"
#include "alphanum.hpp"

int SHARPNESS = 100;
// Coordinates of the top left corner of the floor (x,y,z of Kinect's camera space)
Eigen::Vector3f originPointInCameraSpace = Eigen::Vector3f(1.42858958, 0.241279319, 4.67700005);
// Floor coordinates and dimensions
float topLeftX = 0;
float topLeftY = 0;
float bottomRightX = 3.00;
float bottomRightY = 4.20;
float floorWidth = bottomRightX - topLeftX;
float floorHeight = bottomRightY - topLeftY;

using namespace cv;
using namespace std;
using namespace pcl;

string exec(const char* cmd) {
	char buffer[128];
	std::string result = "";
	std::shared_ptr<FILE> pipe(_popen(cmd, "r"), _pclose);
	if (!pipe) throw std::runtime_error("popen() failed!");
	while (!feof(pipe.get())) {
		if (fgets(buffer, 128, pipe.get()) != NULL)
			result += buffer;
	}
	return result;
}

void contrastStretching(Mat& mat){
	//Contrast stretching - ugly
	int min = 0;
	int max = 0;
	for (int r = 0; r < mat.rows; ++r) {
		for (int c = 0; c < mat.cols; ++c) {
			if (mat.at<uchar>(r, c) > max)
				max = mat.at<uchar>(r, c);
		}
	}
	//cout << "Max = " << max << endl;

	for (int r = 0; r < mat.rows; ++r) {
		for (int c = 0; c < mat.cols; ++c) {
			//increase pixel for each pointcloud's point projected on (c,r)
			mat.at<uchar>(r, c) = (mat.at<uchar>(r, c) - min) * 255 / std::max(1, (max - min));
		}
	}
}

int AcquireDepthAndRGBFrames(AcquisitionKinect2& acq, IColorFrameReader* colorReader, IDepthFrameReader* depthReader, ICoordinateMapper* coordinateMapper, Mat& colorMat, vector<RGBQUAD>& colorBuffer, Mat& depthMat, vector<UINT16>& depthBuffer){

	// Acquire Latest Color Frame
	IColorFrame* pColorFrame = nullptr;
	HRESULT hResult = S_OK;
	hResult = colorReader->AcquireLatestFrame(&pColorFrame);
	if (SUCCEEDED(hResult)){
		// Retrieved Color Data
		hResult = pColorFrame->CopyConvertedFrameDataToArray(colorBuffer.size() * sizeof(RGBQUAD), reinterpret_cast<BYTE*>(&colorBuffer[0]), ColorImageFormat::ColorImageFormat_Bgra);
		if (FAILED(hResult)){
			std::cerr << "Error : IColorFrame::CopyConvertedFrameDataToArray()" << std::endl;
		}
	}
	else{ return -1; }
	if (SUCCEEDED(hResult))
	{
		// conversion to OPENCV Mat
		Mat ris2 = Mat(Size(colorMat.size()), CV_8UC4, &colorBuffer[0], Mat::AUTO_STEP).clone();
		cvtColor(ris2, ris2, CV_BGRA2BGR);
		colorMat = ris2.clone();
		//imshow("", colorMat);
		//waitKey(30);
	}
	SafeRelease(pColorFrame);

	// Acquire Latest Depth Frame
	IDepthFrame* pDepthFrame = nullptr;
	hResult = depthReader->AcquireLatestFrame(&pDepthFrame);
	if (SUCCEEDED(hResult)){
		// Retrieved Depth Data
		hResult = pDepthFrame->CopyFrameDataToArray(depthBuffer.size(), &depthBuffer[0]);
		if (FAILED(hResult)){
			std::cerr << "Error : IDepthFrame::CopyFrameDataToArray()" << std::endl;
		}
	}
	else{ return -1; }
	if (SUCCEEDED(hResult))
	{
		// conversion to OPENCV Mat
		UINT16* depthArray = new UINT16[depthMat.cols*depthMat.rows];
		copy(depthBuffer.begin(), depthBuffer.end(), depthArray);

		Mat ris2 = Mat(Size(depthMat.size()), CV_16UC1, depthArray, Mat::AUTO_STEP).clone();
		Mat3b risDef(depthMat.rows, depthMat.cols);

		int x, y;
		for (y = 0; y<depthMat.rows; y++)
			for (x = 0; x<depthMat.cols; x++){
				risDef(y, x)[0] = (ris2.at<UINT16>(y, x) - 500) / 8;
				risDef(y, x)[1] = (ris2.at<UINT16>(y, x) - 500) / 8;
				risDef(y, x)[2] = (ris2.at<UINT16>(y, x) - 500) / 8;
			}

		depthMat = ris2.clone();
		//imshow("", depthMat);
		//waitKey(30);
		delete[] depthArray;
	}
	SafeRelease(pDepthFrame);
	return 0;
}

Point3d GetBodySpaceCoordinatesFromJoints(Joint joints[JointType_Count]){

	Point3d tmpDepth;
	tmpDepth.x = joints[JointType_Neck].Position.X;
	tmpDepth.y = joints[JointType_Neck].Position.Y;
	tmpDepth.z = joints[JointType_Neck].Position.Z;

	return tmpDepth;
}

Mat CustomBGDSubtraction(BackgroundSubtractorMOG2& pMOG2, Mat& depthMat, vector<UINT16> &foregroundDepthBuffer){
	// BGD SUBTRACTION
	Mat fgMaskMOG2; //fg mask generated by MOG2 method

	pMOG2(depthMat, fgMaskMOG2);

	// Morphology: OPEN
	int morph_size = 2;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(4 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_OPEN, element);

	Mat foregroundDepthMat; //where we store the final result
	depthMat.copyTo(foregroundDepthMat, fgMaskMOG2); //filtering the original depthMat through bgdSUB mask to obtain final result

	//we have to revert to a vector to use it in a pointcloud
	foregroundDepthBuffer.clear(); //clear content (original depth image (see initialization))
	foregroundDepthBuffer.assign((UINT16*)(foregroundDepthMat.datastart), (UINT16*)(foregroundDepthMat.dataend));
	//(debug)
	//imshow("Original Depth", depthMat);
	//imshow("foreground Depth Mat", foregroundDepthMat);
	//waitKey(30);

	return foregroundDepthMat;
}

void FillPointCloudWithDepthAndColorPoints(PointCloud<PointXYZRGB>::Ptr pointCloud, ICoordinateMapper* coordinateMapper, int depthHeight, int depthWidth, vector<UINT16>& foregroundDepthBuffer, int colorHeight, int colorWidth, vector<RGBQUAD>& colorBuffer){
	for (int y = 0; y < depthHeight; y++){
		for (int x = 0; x < depthWidth; x++){
			PointXYZRGB point;

			DepthSpacePoint depthSpacePoint = { static_cast<float>(x), static_cast<float>(y) };
			UINT16 depth = foregroundDepthBuffer[y * depthWidth + x];
			//if depth is 0, this points belong to the background: we dont need them
			if (depth == 0)
				continue;

			// Coordinate Mapping Depth to Color Space, and Setting PointCloud RGB
			ColorSpacePoint colorSpacePoint = { 0.0f, 0.0f };
			coordinateMapper->MapDepthPointToColorSpace(depthSpacePoint, depth, &colorSpacePoint);
			int colorX = static_cast<int>(std::floor(colorSpacePoint.X + 0.5f));
			int colorY = static_cast<int>(std::floor(colorSpacePoint.Y + 0.5f));
			if ((0 <= colorX) && (colorX < colorWidth) && (0 <= colorY) && (colorY < colorHeight)){
				RGBQUAD color = colorBuffer[colorY * colorWidth + colorX];
				point.b = color.rgbBlue;
				point.g = color.rgbGreen;
				point.r = color.rgbRed;
			}

			// Coordinate Mapping Depth to Camera Space, and Setting PointCloud XYZ
			CameraSpacePoint cameraSpacePoint = { 0.0f, 0.0f, 0.0f };
			coordinateMapper->MapDepthPointToCameraSpace(depthSpacePoint, depth, &cameraSpacePoint);
			if ((0 <= colorX) && (colorX < colorWidth) && (0 <= colorY) && (colorY < colorHeight)){
				point.x = cameraSpacePoint.X;
				point.y = cameraSpacePoint.Y;
				point.z = cameraSpacePoint.Z;
			}
			pointCloud->push_back(point);
		}
	}
}

void FillPointCloudWithJointPoints(PointCloud<PointXYZRGB>::Ptr pointCloud, Point3d bodySpacePoints[BODY_COUNT]){
	for (int i = 0; i < BODY_COUNT; i++){
		if (!(bodySpacePoints[i].x == 0 && bodySpacePoints[i].y == 0 && bodySpacePoints[i].z == 0)){
			PointXYZRGB point;

			point.x = bodySpacePoints[i].x;
			point.y = bodySpacePoints[i].y;
			point.z = bodySpacePoints[i].z;
			point.r = 255;
			point.g = 255;
			point.b = 0;

			pointCloud->push_back(point);
		}

	}
}

ModelCoefficients::Ptr DetectPlaneAndGetCoefficients(PointCloud<PointXYZRGB>::Ptr pointCloud){
	ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	PointIndices::Ptr inliers(new pcl::PointIndices);
	// Create the segmentation object
	SACSegmentation<PointXYZRGB> seg;
	// Optional
	seg.setOptimizeCoefficients(true);
	// Mandatory
	seg.setModelType(SACMODEL_PLANE);
	seg.setMethodType(SAC_RANSAC);
	seg.setDistanceThreshold(0.01);
	seg.setProbability(0.9999);
	seg.setMaxIterations(300);

	seg.setInputCloud(pointCloud);
	seg.segment(*inliers, *coefficients);

	cout << "* * * New Iteration * * *" << endl;
	cout << "Model coefficients: " << coefficients->values[0] << " "
		<< coefficients->values[1] << " "
		<< coefficients->values[2] << " "
		<< coefficients->values[3] << endl;

	return coefficients;
}

PointXYZRGB ComputePointCloudCentroid(PointCloud<PointXYZRGB>::Ptr pointCloud){
	Eigen::Vector4f centroid;
	Eigen::Matrix3f covariance_matrix;

	// Extract the eigenvalues and eigenvectors
	Eigen::Vector3f eigen_values;
	Eigen::Matrix3f eigen_vectors;

	compute3DCentroid(*pointCloud, centroid);
	//computeCovarianceMatrix(*pointCloud, centroid, covariance_matrix);
	//eigen33(covariance_matrix, eigen_vectors, eigen_values);
	PointXYZRGB pointCentroid;
	pointCentroid.x = centroid[0];
	pointCentroid.y = centroid[1];
	pointCentroid.z = centroid[2];

	return pointCentroid;
}

void PointCloudXYPlaneToMat(PointCloud<PointXYZRGB>::Ptr pointCloud, Mat& floorProjection, float topLeftX, float topLeftY, float bottomRightX, float bottomRightY, int depthHeight, int depthWidth) {
	for (PointCloud<PointXYZRGB>::iterator it = pointCloud->points.begin(); it < pointCloud->points.end(); it++){
		if ((it->x > topLeftX) && (it->y > topLeftY) && (it->x < bottomRightX) && (it->y < bottomRightY) && ((it->y * SHARPNESS)<depthHeight) && ((it->x * SHARPNESS)<depthWidth)){
			//starting from 0 (black) we increment the value for each point projected on a given floor point (same(x,y))
			// *X multiplication is needed because unit coordinates are meters (*100 = cm, for example) 
			floorProjection.at<unsigned short int>(static_cast<int>((it->y - topLeftY) * SHARPNESS), static_cast<int>((it->x - topLeftX) * SHARPNESS))++;
		}
	}
}

void FindFloorBlobsContours(Mat& floor, int binaryThreshold, vector<FloorObject>& floorBlobs, PointCloud<PointXYZRGB>::Ptr pointCloud, float topLeftX, float topLeftY){
	Mat binaryFloor = floor >= binaryThreshold;
	//dilate(floorProjection, floorProjection, Mat(), Point(1, 2), 1);
	vector<vector<Point> > contours;
	findContours(binaryFloor, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	for (size_t i = 0; i < contours.size(); i++){

		size_t count = contours[i].size();
		if (count < 6)
			continue;

		Mat pointsf;
		Mat(contours[i]).convertTo(pointsf, CV_32F);
		RotatedRect box = fitEllipse(pointsf);
		if ((box.size.height > 0.25*SHARPNESS) || (box.size.width > 0.25*SHARPNESS)){
			FloorObject blob_temp;
			blob_temp.box = box;


			PointCloud<PointXYZRGB>::Ptr pCloud_temp(new PointCloud<PointXYZRGB>());
			Mat redHistogram = Mat(Size(1, 256), CV_32FC1, Scalar(0));
			Mat blueHistogram = Mat(Size(1, 256), CV_32FC1, Scalar(0));
			Mat greenHistogram = Mat(Size(1, 256), CV_32FC1, Scalar(0));
			for (PointCloud<PointXYZRGB>::iterator it = pointCloud->points.begin(); it < pointCloud->points.end(); it++){
				Point2f p;
				p.x = (it->x - topLeftX)*SHARPNESS;
				p.y = (it->y - topLeftY)*SHARPNESS;

				Point2f c = box.center;

				double sqrdRadius = (BLOB_RADIUS*SHARPNESS)*(BLOB_RADIUS*SHARPNESS);

				//if (pointPolygonTest(contours[i], p, false) > 0){    // Point Inside CONTOUR
				if ((pow((p.x - c.x), 2) + pow((p.y - c.y), 2)) < (sqrdRadius)){ //Point inside CIRCLE
					//Point is inside contour
					pCloud_temp->push_back(*it);

					redHistogram.at<float>(it->r, 0)++;
					blueHistogram.at<float>(it->b, 0)++;
					greenHistogram.at<float>(it->g, 0)++;
				}
			}
			for (int i = 0; i < 256; i++){
				blob_temp.bgrHistogram.push_back(blueHistogram.at<float>(i));
				blob_temp.bgrHistogram.push_back(greenHistogram.at<float>(i));
				blob_temp.bgrHistogram.push_back(redHistogram.at<float>(i));
			}
			// Normalize all values between 0 and 1
			normalize(blob_temp.bgrHistogram, blob_temp.bgrHistogram, 1.0, 0.0, NORM_MINMAX, -1, Mat());
			if (pCloud_temp->size()!=0)
				blob_temp.centroid = ComputePointCloudCentroid(pCloud_temp);
			blob_temp.contour = contours[i];
			uchar b = rand() % 255;
			uchar g = rand() % 255;
			uchar r = rand() % 255;
			blob_temp.color = Scalar(b, g, r);
			//temp.positions.push_back(box.center);
			blob_temp.averagePosition = box.center;
			blob_temp.visualCounter = 1;
			blob_temp.confidence = T_0;
			floorBlobs.push_back(blob_temp);
		}
	}
}

double SquaredDistance(Point2d a, Point2d b){
	double dx = a.x - b.x;
	double dy = a.y - b.y;

	return dx*dx + dy*dy;
}

void ComputeBlobDistances(vector<BlobDistance>& squaredDistances, vector<FloorObject>& modelBlobs, vector<FloorObject>& currentBlobs){
	for (int cur_i = 0; cur_i < currentBlobs.size(); cur_i++){
		for (int prev_i = 0; prev_i < modelBlobs.size(); prev_i++){
			double distance = SquaredDistance(currentBlobs[cur_i].box.center, modelBlobs[prev_i].averagePosition);
			BlobDistance temp;
			temp.distance = distance;
			temp.current = currentBlobs[cur_i];
			temp.previous = modelBlobs[prev_i];
			squaredDistances.push_back(temp);
		}
	}
	sort(squaredDistances.begin(), squaredDistances.end());
}

double HistogramDistance(FloorObject& blob1, FloorObject& blob2){

	if (blob1.bgrHistogram.empty() || blob2.bgrHistogram.empty())
		return 1.0; //Maximum distance if using CV_COMP_BHATTACARYYA
	else
		return compareHist(blob1.bgrHistogram, blob2.bgrHistogram, CV_COMP_BHATTACHARYYA);
}

void ComputeCostMatrix(vector<vector<double>>& costMatrix, vector<FloorObject>& modelBlobs, vector<FloorObject>& currentBlobs){

	//erase costMatrix to ensure we start from scratch
	vector<vector<double>>().swap(costMatrix);

	for (int cur_i = 0; cur_i < currentBlobs.size(); cur_i++){
		vector<double> tempRow;
		for (int prev_i = 0; prev_i < modelBlobs.size(); prev_i++){

			double distance = SquaredDistance(currentBlobs[cur_i].box.center, modelBlobs[prev_i].averagePosition);
			double rgbDistance = HistogramDistance(currentBlobs[cur_i], modelBlobs[prev_i]);

			double totalDistance = SquaredDistance(Point2d(distance, 0), Point2d(rgbDistance, 0));
			tempRow.push_back(totalDistance/modelBlobs[prev_i].confidence);
		}

		costMatrix.push_back(tempRow);
		vector<double>().swap(tempRow);
	}
}

void IdentifyFloorBlobs(vector<BlobDistance>& distances, vector<FloorObject>& previousBlobs, vector<FloorObject>& currentBlobs, vector<FloorObject>& trackedBlobs){
	vector <FloorObject> alreadyDetected;

	for (vector<FloorObject>::iterator i = currentBlobs.begin(); i < currentBlobs.end(); ++i){

		bool newBlob = true;

		for (vector<BlobDistance>::iterator j = distances.begin(); j < distances.end(); j++){

			if (j->current == *i){
				if (!(find(alreadyDetected.begin(), alreadyDetected.end(), j->previous) != alreadyDetected.end())){
					FloorObject temp;
					temp.box = j->current.box;
					temp.contour = j->current.contour;
					temp.color = j->previous.color;
					trackedBlobs.push_back(temp);
					alreadyDetected.push_back(j->previous);
					newBlob = false;
					break;
				}
			}
		}
		if (newBlob){
			trackedBlobs.push_back(*i);
		}
	}

	if (distances.empty()){
		trackedBlobs.swap(previousBlobs);
	}

	vector<FloorObject>().swap(alreadyDetected);
}

void UpdateModelBestMatchFirst(vector<FloorObject>& model, vector<FloorObject>& currentBlobs){

	vector<FloorObject> currentDetected;
	vector<FloorObject> modelDetected;

	vector<BlobDistance> distances;
	ComputeBlobDistances(distances, model, currentBlobs);

	for (vector<BlobDistance>::iterator i = distances.begin(); i < distances.end(); i++){

		bool cur_found = find(currentDetected.begin(), currentDetected.end(), i->current) != currentDetected.end();
		bool model_found = find(modelDetected.begin(), modelDetected.end(), i->previous) != modelDetected.end();
		bool both_not_found = !cur_found && !model_found;


		if (both_not_found){
			vector<FloorObject>::iterator matching;
			matching = find_if(model.begin(), model.end(), [&i](const FloorObject& obj) {return obj.averagePosition == i->previous.averagePosition; });
			//Update correspondent element inside the model
			matching->box = i->current.box;
			matching->contour = i->current.contour;
			matching->visualCounter = 0;
			//mark this element as already detected
			currentDetected.push_back(i->current);
			modelDetected.push_back(i->previous);
		}

	}

	for (vector<FloorObject>::iterator i = currentBlobs.begin(); i < currentBlobs.end(); i++){
		bool cur_found = find(currentDetected.begin(), currentDetected.end(), *i) != currentDetected.end();

		if (!cur_found)
			model.push_back(*i);

	}

	vector<BlobDistance>().swap(distances);
	vector<FloorObject>().swap(currentDetected);
	vector<FloorObject>().swap(modelDetected);

}

void UpdateBlobColorDistribution(FloorObject& currentBlob, FloorObject& modelBlob){

	for (int i = 0; i < modelBlob.bgrHistogram.size(); i++){
		modelBlob.bgrHistogram[i] = modelBlob.bgrHistogram[i] + currentBlob.bgrHistogram[i];
		modelBlob.bgrHistogram[i] /= 2;
	}

}

void UpdateAveragePositionOfModelBlob(FloorObject& modelBlob){

	/*
	//update average position:
	Point2f p = modelBlob.averagePosition;
	Point2f new_p = modelBlob.box.center;
	float deltaX = (new_p.x - p.x) / 2;
	float deltaY = (new_p.y - p.y) / 2;
	modelBlob.averagePosition = Point2f(((p.x + (2 * new_p.x)) / 3), (p.y + (2 * new_p.y)) / 3);
	//modelBlob.averagePosition.x -= deltaX;
	//modelBlob.averagePosition.y -= deltaY;
	*/

	//NEW AVERAGE POSITION
	int maxPositions = min((int)modelBlob.positions.size(), 3);
	Point2f temp(0,0);

	vector<Point2i>::reverse_iterator i = modelBlob.positions.rbegin();
	float deltaX = (i->x - (i+1)->x) / 2;
	float deltaY = (i->y - (i+1)->y) / 2;
	for (int j = 0; j < maxPositions; j++){
		temp.x += (i + j)->x;
		temp.y += (i + j)->y;
	}

	temp.x /= maxPositions;
	temp.y /= maxPositions;

	temp.x = i->x + deltaX;
	temp.y = i->y + deltaY;

	modelBlob.averagePosition = temp;

}

void UpdateConfidenceScore(FloorObject& modelBlob){
	
	if (modelBlob.state == stable || modelBlob.state == hidden){
		if (modelBlob.visualCounter <= 0){
			// Decrease confidence
			modelBlob.confidence = max(modelBlob.confidence*(LAMBDA_DOWN), T_MIN);
		}
		else {
			// Increase confidence
			modelBlob.confidence = max(modelBlob.confidence*LAMBDA_UP + (1 - LAMBDA_UP), T_MIN);
		}
		//modelBlob.confidence = max(modelBlob.confidence*LAMBDA + zero_or_one*(1 - LAMBDA), T_MIN);
	}
}

void AssociateCurrentBlobWithModel(FloorObject& currentBlob, FloorObject& modelBlob){

	modelBlob.box = currentBlob.box;
	modelBlob.contour = currentBlob.contour;
	modelBlob.positions.push_back(currentBlob.box.center);
	(modelBlob.visualCounter < 0) ? modelBlob.visualCounter = 1 : modelBlob.visualCounter++;

	//FEATURES UPDATE:
	//update color distribution
	UpdateBlobColorDistribution(currentBlob, modelBlob);
	//update positions
	UpdateAveragePositionOfModelBlob(modelBlob);
}

void UpdateModelGlobalMinCost(vector<FloorObject>& modelBlobs, vector<FloorObject>& currentBlobs, vector<vector<double>>& costMatrix){

	int prova = currentBlobs.size();

	ComputeCostMatrix(costMatrix, modelBlobs, currentBlobs);

	if (currentBlobs.size() != modelBlobs.size()){
		//CostMatrix is not squared: must be filled with rows or columns of zeroes
		if (currentBlobs.size() > modelBlobs.size()){
			//CASE:
			// 2	5	1
			// 9	1	3
			// 2	4	4
			// 1	1	7
			// 1	3	2
			//must add columns (there will be new blobs)
			for (int i = 0; i < costMatrix.size(); i++){
				for (int j = modelBlobs.size(); j < currentBlobs.size(); j++)
					costMatrix[i].push_back(DBL_MAX);
			}
		}
		else if (modelBlobs.size() > currentBlobs.size()){
			//CASE:
			// 3	7	9	1	3
			// 2	6	1	1	5
			// 8	8	2	4	6
			//must add rows (there will be hidden blobs)
			for (int i = currentBlobs.size(); i < modelBlobs.size(); i++){
				costMatrix.push_back(vector<double>(modelBlobs.size(), DBL_MAX));
			}
		}
	}

	vector<int> currentMatchings;
	vector<int> modelMatchings;

	MinCostMatching(costMatrix, currentMatchings, modelMatchings);

	for (int i = 0; i < currentBlobs.size(); i++){
		
		if (costMatrix[i][currentMatchings[i]] == DBL_MAX){
			//new Blob -> state: ALPHA
			modelBlobs.push_back(currentBlobs[i]);
		}
		else {
			// Matching found
			AssociateCurrentBlobWithModel(currentBlobs[i], modelBlobs[currentMatchings[i]]);
		}
	}

	if (currentBlobs.empty())
		for (int i = 0; i < modelBlobs.size(); i++)
			// Hide all blobs (set visual counter to ZERO)
			(modelBlobs[i].visualCounter <= 0) ? (modelBlobs[i].visualCounter--) : (modelBlobs[i].visualCounter=0);
	else {
		for (int i = 0; i < modelBlobs.size(); i++){

			if (costMatrix[modelMatchings[i]][i] == DBL_MAX){
				// State: HIDDEN
				if (!(modelBlobs[i].state == alpha && modelBlobs[i].visualCounter == 1))
					(modelBlobs[i].visualCounter <= 0) ? (modelBlobs[i].visualCounter--) : (modelBlobs[i].visualCounter = 0);
			}
		}
	}

	for (int kk = 0; kk< currentMatchings.size(); kk++)
		cout << "Current Matchings: " << currentMatchings[kk] << endl;

	for (int kk = 0; kk < modelMatchings.size(); kk++)
		cout << "Model Matchings: " << modelMatchings[kk] << endl;

	vector<int>().swap(currentMatchings);
	vector<int>().swap(modelMatchings);
}

//void UpdateModelPositions(vector<FloorObject>& modelBlobs){
//
//	for (vector<FloorObject>::iterator i = modelBlobs.begin(); i < modelBlobs.end(); i++){
//		//update average position:
//		Point2f p = i->averagePosition;
//		Point2f new_p = i->box.center;
//		float deltaX = (new_p.x - p.x) / 2;
//		float deltaY = (new_p.y - p.y) / 2;
//		i->averagePosition = Point2f(((p.x + (2 * new_p.x)) / 3), (p.y + (2 * new_p.y)) / 3);
//		i->averagePosition.x += deltaX;
//		i->averagePosition.y += deltaY;
//	}
//}

void RemoveDeadElements(vector<FloorObject>& modelBlobs){
	modelBlobs.erase(remove_if(modelBlobs.begin(), modelBlobs.end(), [](const FloorObject& obj) {return obj.state == dead; }), modelBlobs.end());
}

void UpdateModelState(vector<FloorObject>& modelBlobs, vector<FloorObject>& deadBlobs){
	
	for (vector<FloorObject>::iterator i = modelBlobs.begin(); i < modelBlobs.end(); i++){


		if ((i->visualCounter == 0 && i->state == alpha) || (i->visualCounter <= -DEATH_PARAM)){
			i->state = dead;
			deadBlobs.push_back(*i);
		}
		else if ((i->visualCounter > 0 && (i->state == hidden || i->state == stable)) || (i->visualCounter >= ALPHA_PARAM))
			i->state = stable;
		else if (i->visualCounter <= 0 && i->visualCounter > -DEATH_PARAM)
			i->state = hidden;

		//update confidence score
		UpdateConfidenceScore(*i);
	}

	// delete elements whose state is: DEAD
	RemoveDeadElements(modelBlobs);

}

int _tmain(int argc, _TCHAR* argv[])
{
	//Custom Kinect Acquisition
	AcquisitionKinect2 acq = AcquisitionKinect2();
	FrameSet kinectFrame; //Acquisition frame: contains both depth and rgb images
	// Kinect v2 Depth image's resolution is 512x424 pixels
	int depthWidth = 512;
	int depthHeight = 424;
	// Kinect v2 Color image's resolution is FullHD (1920x1080)
	int colorWidth = 1920;
	int colorHeight = 1080;

	// OPENCV stuff
	BackgroundSubtractorMOG2 pMOG2; //MOG2 Background subtractor + some parameters
	pMOG2.set("varThresholdGen", 625.0);
	pMOG2.set("backgroundRatio", 0.4);
	Mat colorMat(colorHeight, colorWidth, CV_8UC4);
	Mat depthMat(depthHeight, depthWidth, CV_16UC1);	//color and depth images from Kinect
	Mat floor;

	// folder management
	vector<String> fnames_depth, fnames_rgb, fnames_floor, fnames_bodies;
	String folder = "C:/Users/Niccol�/Documents/Visual Studio 2013/Projects/Tesi_nb/Util - Record Video -- Floor+Kinect/videos";
	int vidNumber, wait_key;
	cout << "Vid number: ";
	cin >> vidNumber;
	cout << "Wait key value: ";
	cin >> wait_key;

	folder = folder + "/" + to_string(vidNumber) + "/";
	cout << folder << endl;

	//Output file for measurements
	fstream reportFile;
	reportFile.open("report_vid_" + to_string(vidNumber) + "_.txt", ios::out);
	
	//RANDOM initializer (for tracking IDs)
	srand(time(NULL));

	// Create PointCloud (plus its ColorHandler) and add it to the Visualizer
	PointCloud<PointXYZRGB>::Ptr pointCloud(new PointCloud<PointXYZRGB>());
	PointCloud<PointXYZRGB>::Ptr jointsPointCloud(new PointCloud<PointXYZRGB>());

	vector<FloorObject> currentBlobs; //container of objects detected on the floor
	vector<FloorObject> modelBlobs;
	vector<FloorObject> deadBlobs;

#ifdef PCL_VISUALIZER //Visualizer init
	//Init PCL Visualizer
	boost::shared_ptr<visualization::PCLVisualizer> PCLviewer(new visualization::PCLVisualizer("3D Viewer"));
	PCLviewer->setBackgroundColor(0, 0, 0);
	PCLviewer->addCoordinateSystem(1);
	PCLviewer->initCameraParameters();

	//Create PointCloud's ColorHandler and add everything to the Visualizer
	visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb(pointCloud);
	PCLviewer->addPointCloud<PointXYZRGB>(pointCloud, rgb, "Kinect Depth Cloud");
	PCLviewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Kinect Depth Cloud");
#endif

	// Look for filenames of all images
	glob(folder + "/RGB", fnames_rgb);
	glob(folder + "/D", fnames_depth);
	glob(folder + "/Floor", fnames_floor);
	glob(folder + "/Bodies", fnames_bodies);

	//we need natural sorting for filenames
	sort(fnames_rgb.begin(), fnames_rgb.end(), doj::alphanum_less<String>());
	sort(fnames_depth.begin(), fnames_depth.end(), doj::alphanum_less<String>());
	sort(fnames_floor.begin(), fnames_floor.end(), doj::alphanum_less<String>());
	sort(fnames_bodies.begin(), fnames_bodies.end(), doj::alphanum_less<String>());
	int n_frames = 0;

	// Main loop: kinect image processing
	while (!GetAsyncKeyState(VK_ESCAPE) && (n_frames < fnames_rgb.size())){

#ifdef DRAW_TILES
		for (int r = 0; r < floor.rows; r++){
			for (int c = 0; c < floor.cols; c++){
				if ((r % (floor.rows / 5)) == 0 || (c % ((floor.cols / 7))) == 0){
					floor(r, c) = 255;
				}
			}
		}
#endif
		floor = imread(fnames_floor[n_frames], 0);
		// Morphology: OPEN
		imshow("Floor", floor);

		/////////////////////////////
		////////// KINECT ACQUISITION
		/////////////////////////////

		ICoordinateMapper* coordinateMapper = acq.GetCoordinateMapper(); //init coordinate mapper to map color on depth data
		//DEPTH
		depthMat = imread(fnames_depth[n_frames], CV_LOAD_IMAGE_ANYDEPTH);
		Mat temp = depthMat.clone();
		colorMat = imread(fnames_rgb[n_frames], 1);
		cvtColor(colorMat, colorMat, CV_BGR2BGRA);
		//COLOR
		vector<RGBQUAD> colorBuffer(colorWidth * colorHeight);
		colorBuffer.assign((RGBQUAD*)(colorMat.datastart), (RGBQUAD*)(colorMat.dataend));
		vector<UINT16> foregroundDepthBuffer(depthWidth*depthHeight); //right now it's the original depth buffer, will become foreground's depth buffer after bgd_subtraction 
		foregroundDepthBuffer.assign((UINT16*)(depthMat.datastart), (UINT16*)(depthMat.dataend));
		//BODY
		ifstream bodyStream;
		bodyStream.open(fnames_bodies[n_frames], ios::in);
		
		Point3d bodySpacePoints[BODY_COUNT] = { 0 };
		CameraSpacePoint p;
		Joint joints[JointType_Count];
		
		for (int j = 0; j < BODY_COUNT; j++){
			bool isTracked = false;
			for (int i = 0; i < 25; i++){
				int enum_tmp;
				if (bodyStream >> p.X >> p.Y >> p.Z >> enum_tmp){
					//bodyStream.read((char*)&j_track_state, sizeof(enum TrackingState));
					joints[i].Position = p;
					joints[i].TrackingState = (TrackingState)enum_tmp;
					joints[i].JointType = (JointType)i;
					isTracked = true;
				}
			}
			if (isTracked)
				bodySpacePoints[j] = GetBodySpaceCoordinatesFromJoints(joints);
		}

		bodyStream.close();



		/////////////////////////////
		//////////// BGD SUBTRACTION
		/////////////////////////////
		//Update the background model
#ifdef BGD_SUBTRACTION
		// BGD SUBTRACTION
		Mat foregroundDepthMat = CustomBGDSubtraction(pMOG2, depthMat, foregroundDepthBuffer);
#endif

		/////////////////////////////
		//////////// CREATE POINTCLOUD
		/////////////////////////////
		// Set Point Cloud Parameters
		pointCloud->width = static_cast<uint32_t>(depthWidth);
		pointCloud->height = static_cast<uint32_t>(depthHeight);
		pointCloud->is_dense = false;
		// Fill point cloud with points
		FillPointCloudWithDepthAndColorPoints(pointCloud, coordinateMapper, depthHeight, depthWidth, foregroundDepthBuffer, colorHeight, colorWidth, colorBuffer);

		/////////////////////////////
		//////////// GROUND TRUTH POINT CLOUD
		/////////////////////////////
		jointsPointCloud->width = static_cast<uint32_t>(depthWidth);
		jointsPointCloud->height = static_cast<uint32_t>(depthHeight);
		jointsPointCloud->is_dense = false;
		FillPointCloudWithJointPoints(jointsPointCloud, bodySpacePoints);

		/////////////////////////////
		////////////   CROP BOX F
		/////////////////////////////
		// Crop box: min and max points are corners of the parallelepiped that will be used as a filter
#ifdef CROP_BOX_FILTERING
		Eigen::Vector4f minPoint;
		minPoint[0] = -1.8;  // define minimum point x (R)
		minPoint[1] = -1.20;  // define minimum point y (G)
		minPoint[2] = 0;  // define minimum point z (B)
		Eigen::Vector4f maxPoint;
		maxPoint[0] = 3.65;  // define max point x 
		maxPoint[1] = 1.56;  // define max point y 
		maxPoint[2] = 2.96;  // define max point z 
		CropBox<PointXYZRGB> cropFilter;
		PointCloud<PointXYZRGB>::Ptr pointCloud2(new PointCloud<PointXYZRGB>());
		*pointCloud2 = *pointCloud;
		cropFilter.setInputCloud(pointCloud2);
		cropFilter.setMin(minPoint);
		cropFilter.setMax(maxPoint);
		cropFilter.filter(*pointCloud);
		pointCloud2->clear();
#endif

		///////////////////////////
		/////////// PLANE DETECTION
		///////////////////////////

		ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
#ifdef GET_PLANE_COEFFICIENTS
		coefficients = DetectPlaneAndGetCoefficients(pointCloud);

#else 
		//If variable is false, we set the coefficients manually and use them for the projection
		coefficients->values.resize(4);
		coefficients->values[0] = 0.0382139;
		coefficients->values[1] = -0.91068;
		coefficients->values[2] = 0.411342;
		coefficients->values[3] = -1.76073;
#endif

		////////////////////////////
		///////  PLANAR PROJECTION
		////////////////////////////
#ifdef PCL_VISUALIZER
		PCLviewer->removeShape("Projection plane");
#endif

#ifdef PLANAR_PROJECTION
		//PCLviewer->addPlane(*coefficients, 0, 0, 1.7, "Projection plane"); //show projection plane
		ProjectInliers<pcl::PointXYZRGB> proj;
		proj.setModelType(pcl::SACMODEL_PLANE);
		proj.setInputCloud(pointCloud);
		proj.setModelCoefficients(coefficients);
		proj.filter(*pointCloud);
#endif

		////////////////////////////
		////////// EIGEN TRANSFORM
		////////////////////////////
		// Rotation to move origin on the floor and align XY plane to it
		Eigen::Affine3f transformation;
		getTransformationFromTwoUnitVectorsAndOrigin(Eigen::Vector3f(0, -coefficients->values[3] / coefficients->values[1], 0), // Y direction (intersection with XZ plane)
			Eigen::Vector3f(-coefficients->values[0], coefficients->values[1], coefficients->values[2]), // Z direction (normal vector to the floor plane)
			originPointInCameraSpace, // Origin
			transformation
			);
		transformPointCloud(*pointCloud, *pointCloud, transformation);
		transformPointCloud(*jointsPointCloud, *jointsPointCloud, transformation);


		////////////////////////////
		////////// POINT CLOUD CENTROID
		////////////////////////////

		PointXYZRGB centroid;
		if (pointCloud->size() != 0)
		{
			centroid = ComputePointCloudCentroid(pointCloud);

		}

		///////////////////////////////////////
		///////////		POINTCLOUD TO MAT
		//////////////////////////////////////

		//At this point, we fill the Mat to revert to a grayscale image of the floor projection
		// offsets: translation from the Origin point
		Mat floorProjection(static_cast<int>(floorHeight * SHARPNESS), static_cast<int>(floorWidth*SHARPNESS), CV_16UC1, Scalar(0, 0, 0));

		PointCloudXYPlaneToMat(pointCloud, floorProjection, topLeftX, topLeftY, bottomRightX, bottomRightY, depthHeight, depthWidth);

		//////////////////////////////////
		/////////  BLOB DETECTION - FLOOR
		//////////////////////////////////
		FindFloorBlobsContours(floor, 20, currentBlobs, pointCloud, topLeftX, topLeftY);
		if (modelBlobs.empty()){
			modelBlobs.swap(currentBlobs);
		}
		else {
			//updatemodel, by confronting new Blobs with old ones

			/////////////////
			// BEST MATCH FIRST
			//UpdateModelBestMatchFirst(modelBlobs, currentBlobs);
			/////////////////

			///////// HUNGARIAN
			vector<vector<double>> costMatrix;
			UpdateModelGlobalMinCost(modelBlobs, currentBlobs, costMatrix);
			vector<vector<double>>().swap(costMatrix);
			UpdateModelState(modelBlobs, deadBlobs);
		}

		// write measures to file
		reportFile << n_frames << '\t';
		reportFile << "GT" << '\t';
		for (PointCloud<PointXYZRGB>::iterator it = jointsPointCloud->points.begin(); it < jointsPointCloud->points.end(); it++){
			reportFile << it->x << '\t';
			reportFile << it->y << '\t';
			double CentroidJointDistance = sqrt(SquaredDistance(Point2d(centroid.x, centroid.y), Point2d(it->x, it->y)));
			reportFile << "DISTANCE" << '\t';
			reportFile << CentroidJointDistance << '\t';

		}
		reportFile << "MODEL" << '\t';
		
		for (int kk = 0; kk < modelBlobs.size(); kk++){
			reportFile << modelBlobs[kk].state << '\t';
			reportFile << modelBlobs[kk].box.center.x << '\t';
			reportFile << modelBlobs[kk].box.center.y << '\t';
			reportFile << modelBlobs[kk].color << '\t';
		}

		reportFile << "CENTROID" << '\t';
		reportFile << centroid.x << '\t';
		reportFile << centroid.y << '\t';	

		reportFile << '\n';

		cout << n_frames << endl;
		cout << "Model blobs: " << modelBlobs.size() << endl;
		cout << "Current blobs: " << currentBlobs.size() << endl;
		cout << "Dead blobs: " << deadBlobs.size() << endl;
		cout << "----------------------------" << endl;
		for (int kk = 0; kk < modelBlobs.size(); kk++){
			cout << "Model blob[" << kk << "]: " << endl;
			cout << "State: " << modelBlobs[kk].state << endl;
			cout << "Average pos: " << modelBlobs[kk].averagePosition << endl;
			cout << "Color: " << modelBlobs[kk].color << endl;
			cout << "Counter: " << modelBlobs[kk].visualCounter << endl;
			cout << "Confidence Score: " << modelBlobs[kk].confidence << endl;
		}
		cout << "-----------" << endl;
		/*
		for (int kk = 0; kk < currentBlobs.size(); kk++){
			cout << "Current blob[" << kk << "]: " << endl;
			cout << "State: " << currentBlobs[kk].state << endl;
			cout << "Average pos: " << currentBlobs[kk].averagePosition << endl;
			cout << "Color: " << currentBlobs[kk].color << endl;
		}
		*/
		cout << endl << endl;

		//reconversion to Uchar for better visualization
		floorProjection.convertTo(floorProjection, CV_8UC1);
		contrastStretching(floorProjection);

		//////////////////////////
		/////  Put infos together
		//////////////////////////
		Mat3b mergedChannels(floorProjection.rows, floorProjection.cols, Vec3b(0, 0, 0));

		for (int r = 0; r < mergedChannels.rows; r++){
			for (int c = 0; c < mergedChannels.cols; c++){
				// Red channel : floor projection
				// Green channel: florimage data
				mergedChannels(r, c)[2] = (char)floorProjection.at<uchar>(r, c);
				mergedChannels(r, c)[1] = (char)floor.at<uchar>(r, c);
			}
		}

		for (vector<FloorObject>::iterator it = modelBlobs.begin(); it < modelBlobs.end(); it++){

			if (it->state == stable){
				Point2f vertices[4];
				it->box.points(vertices);
				for (int a = 0; a < 4; a++)
				{
					vertices[a].x = vertices[a].x / 2;
					vertices[a].y = vertices[a].y / 2;
				}
				ellipse(mergedChannels, it->box, it->color, 1, CV_AA);
				circle(mergedChannels, it->averagePosition, BLOB_RADIUS*SHARPNESS, it->color, 1);
				line(mergedChannels, (vertices[0] + vertices[1]), (vertices[2] + vertices[3]), it->color);
				line(mergedChannels, (vertices[1] + vertices[2]), (vertices[3] + vertices[0]), it->color);
			}
		}

		for (PointCloud<PointXYZRGB>::iterator it = jointsPointCloud->points.begin(); it < jointsPointCloud->points.end(); it++){
			if ((it->x>0) && (it->y>0)){
				circle(mergedChannels, Point(((it->x - topLeftX) * SHARPNESS), ((it->y - topLeftY) * SHARPNESS)), BLOB_RADIUS*SHARPNESS / 8, Scalar(0, 255, 255), 3);
			}
		}

		circle(mergedChannels, Point(((centroid.x) * SHARPNESS), ((centroid.y) * SHARPNESS)), BLOB_RADIUS*SHARPNESS / 8, Scalar(255, 255, 0), 3);

		//Show results
		resize(mergedChannels, mergedChannels, Size(900, 1260), CV_INTER_CUBIC);
		namedWindow("Merged Channels", 1);
		createTrackbar("Sharpness", "Merged Channels", &SHARPNESS, 100);
		imshow("Projection", floorProjection);
		imshow("Merged Channels", mergedChannels);
		imwrite("mergedChannels/mergedChannel" + to_string(n_frames) + ".png", mergedChannels);
		waitKey(wait_key);

#ifdef PCL_VISUALIZER
		// PCL Visualizer
		PCLviewer->updatePointCloud<PointXYZRGB>(pointCloud, rgb, "Kinect Depth Cloud");
		PCLviewer->spinOnce();
		boost::this_thread::sleep(boost::posix_time::microseconds(10000));
#endif

		// Free memory
		vector<FloorObject>().swap(currentBlobs);
		vector<RGBQUAD>().swap(colorBuffer);
		vector<UINT16>().swap(foregroundDepthBuffer);
		pointCloud->clear();
		jointsPointCloud->clear();
		n_frames++;
	}

	deadBlobs.erase(remove_if(deadBlobs.begin(), deadBlobs.end(), [](const FloorObject& obj) {return obj.positions.size() <= 1; }), deadBlobs.end());

	Mat trackletViewer(300, 420, CV_8UC3, Scalar(0, 0, 0));
	for (int i = 0; i < modelBlobs.size(); i++){
		for (int j = 0; j < modelBlobs[i].positions.size() - 1; j++){
			line(trackletViewer, modelBlobs[i].positions[j], modelBlobs[i].positions[j + 1], modelBlobs[i].color, 1);
		}
	}

	/*
	SHOW DEAD BLOBS
	for (int i = 0; i < deadBlobs.size(); i++){
	for (int j = 0; j < deadBlobs[i].positions.size() - 1; j++){
	line(trackletViewer, deadBlobs[i].positions[j], deadBlobs[i].positions[j + 1], deadBlobs[i].color, 1);
	}
	}
	*/

	imshow("tracklet", trackletViewer);
	waitKey();
	reportFile.close();

	// End Processing
	return 0;
}
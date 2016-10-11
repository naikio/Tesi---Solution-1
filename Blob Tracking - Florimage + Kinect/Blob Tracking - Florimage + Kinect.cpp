#include "stdafx.h"

#define NOMINMAX
#define _WINSOCK_DEPRECATED_NO_WARNINGS

#define BGD_SUBTRACTION
#define CROP_BOX_FILTERING
//#define GET_PLANE_COEFFICIENTS  //used to search for floor plane coefficients (useful to retrieve plane coefficients the 1st time)
#define PLANAR_PROJECTION false
//#define PCL_VISUALIZER
//#define DRAW_TILES

#define BEST_MATCH_FIRST 1
#define HUNGARIAN_MIN_COST 2

#define ALPHA_PARAM 8
#define DEATH_PARAM 12

int SHARPNESS = 100;
//OPENCV 2.4.10
#include <opencv2\opencv.hpp>

#include <Windows.h>
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

#include<winsock2.h>
#pragma comment(lib,"ws2_32.lib") //Winsock Library
#include <boost/thread.hpp>
#include <boost/chrono.hpp>

//Custom STRUCTs
#include "FloorObject.h"
#include "BlobDistance.h"

//Custom Algorithms
#include "bipartite-mincost.h"

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

int ConnectToSocket(SOCKET socket, struct sockaddr_in server){
	if (connect(socket, (struct sockaddr *)&server, sizeof(server)) < 0)
	{
		puts("connect error");
		return 1;
	}
	puts("Connected");
	return 0;
}

int SendToSocket(SOCKET socket, char* message){
	if (send(socket, message, strlen(message), 0) < 0)
	{
		puts("Send failed");
		return 2;
	}
	puts("Data Sent\n");
	return 0;
}

void ReceiveFromSocket(SOCKET socket, char* server_reply, int &recv_size){
	if ((recv_size = recv(socket, server_reply, 2000, 0)) == SOCKET_ERROR)
	{
		puts("recv failed");
	}

	puts("Reply received\n");
	//Add a NULL terminating character to make it a proper string before printing
	server_reply[recv_size] = '\0';
	puts(server_reply);
}

int CreateMatFromSocketReply(Mat1b& floor, char* server_reply, int &consecutiveWrongFrames){
	// 1st byte: 0xFD
	// 2nd, 3rd: w, h of the image
	// 
	// data: w*h bytes, one for each pixels
	//
	// checksum: 0x0 for now
	// final byte: 0xFF

	int k = 0; //used to iterate over data
	// 1: 0xFD
	if (server_reply[k++] != (char)0xFD){
		cerr << "Error: wrong header" << endl;
		consecutiveWrongFrames++;
		if (consecutiveWrongFrames > 10){
			cerr << "Too many wrong frames" << endl;
			return 1;
		}
	}

	// 2,3: w, h
	int floorViewerWidth = (int)server_reply[k++];
	int floorViewerHeight = (int)server_reply[k++];
	Mat1b tempFloor(floorViewerHeight, floorViewerWidth); //floor image has 32x24 px

	// image data
	for (int r = 0; r < tempFloor.rows; r++){
		for (int c = 0; c < tempFloor.cols; c++){
			tempFloor(r, c) = (uchar)server_reply[k++];
		}
	}

	// last byte
	if (server_reply[k++] != (char)0x0){
		cerr << "Error: wrong checksum" << endl;
		consecutiveWrongFrames++;
		if (consecutiveWrongFrames > 10){
			cerr << "Too many wrong frames" << endl;
			return 1;
		}
	}
	if (server_reply[k] == (char)0xFF){
		consecutiveWrongFrames = 0;
		cout << "Successful data verification" << endl;
		floor = tempFloor.clone();
		return 0;
	}

	consecutiveWrongFrames++;
	if (consecutiveWrongFrames > 10){
		cerr << "Too many wrong frames" << endl;
		return 1;
	}

	return 1;
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
	computeCovarianceMatrix(*pointCloud, centroid, covariance_matrix);
	eigen33(covariance_matrix, eigen_vectors, eigen_values);
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

void FindFloorBlobsContours(Mat& floor, int binaryThreshold, vector<FloorObject>& floorBlobs){
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
			FloorObject temp;
			temp.box = box;
			temp.contour = contours[i];
			uchar b = rand() % 255;
			uchar g = rand() % 255;
			uchar r = rand() % 255;
			temp.color = Scalar(b, g, r);
			//temp.positions.push_back(box.center);
			temp.averagePosition = box.center;
			temp.visualCounter = 0;
			floorBlobs.push_back(temp);
		}
	}
}

double SquaredDistance(Point2f a, Point2f b){
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

void ComputeCostMatrix(vector<vector<double>>& costMatrix, vector<FloorObject>& modelBlobs, vector<FloorObject>& currentBlobs){

	//erase costMatrix to ensure we start from scratch
	vector<vector<double>>().swap(costMatrix);

	for (int cur_i = 0; cur_i < currentBlobs.size(); cur_i++){
		vector<double> tempRow;
		for (int prev_i = 0; prev_i < modelBlobs.size(); prev_i++){
			double distance = SquaredDistance(currentBlobs[cur_i].box.center, modelBlobs[prev_i].averagePosition);
			tempRow.push_back(distance);
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
			matching = find_if(model.begin(), model.end(), [&i](FloorObject obj) {return obj.averagePosition == i->previous.averagePosition; });
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

void UpdateModelGlobalMinCost(vector<FloorObject>& modelBlobs, vector<FloorObject>& currentBlobs, vector<vector<double>>& costMatrix){

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
					costMatrix[i].push_back(0);
			}
		}
		else if (modelBlobs.size() > currentBlobs.size()){
			//CASE:
			// 3	7	9	1	3
			// 2	6	1	1	5
			// 8	8	2	4	6
			//must add rows (there will be hidden blobs)
			for (int i = currentBlobs.size(); i < modelBlobs.size(); i++){
				costMatrix.push_back(vector<double>(modelBlobs.size(), 0));
			}
		}
	}

	vector<int> currentMatchings;
	vector<int> modelMatchings;

	MinCostMatching(costMatrix, currentMatchings, modelMatchings);

	for (int i = 0; i < currentBlobs.size(); i++){
		
		if (costMatrix[i][currentMatchings[i]] == 0){
			//new Blob -> state: ALPHA
			currentBlobs[i].visualCounter++;
			modelBlobs.push_back(currentBlobs[i]);
		}
		else {
			modelBlobs[currentMatchings[i]].box = currentBlobs[i].box;
			modelBlobs[currentMatchings[i]].contour = currentBlobs[i].contour;
			modelBlobs[currentMatchings[i]].visualCounter++;
		}
	}

	if (currentBlobs.empty())
		for (int i = 0; i < modelBlobs.size(); i++)
			// Hide all blobs (set visual counter to ZERO)
			(modelBlobs[i].visualCounter <= 0) ? (modelBlobs[i].visualCounter--) : (modelBlobs[i].visualCounter=0);

	for (int i = 0; i < modelBlobs.size(); i++){

		if (costMatrix[modelMatchings[i]][i] == 0){
			// State: HIDDEN
			if (modelBlobs[i].visualCounter <= 0)
				modelBlobs[i].visualCounter--;
			else
				modelBlobs[i].visualCounter = 0;
			
		}
	}

	vector<int>().swap(currentMatchings);
	vector<int>().swap(modelMatchings);
}

void UpdateModelPositions(vector<FloorObject>& modelBlobs){

	for (vector<FloorObject>::iterator i = modelBlobs.begin(); i < modelBlobs.end(); i++){
		//update average position:
		Point2f p = i->averagePosition;
		Point2f new_p = i->box.center;
		float deltaX = (new_p.x - p.x) / 2;
		float deltaY = (new_p.y - p.y) / 2;
		i->averagePosition = Point2f(((p.x + (2 * new_p.x)) / 3), (p.y + (2 * new_p.y)) / 3);
		i->averagePosition.x += deltaX;
		i->averagePosition.y += deltaY;
	}
}

void UpdateModelState(vector<FloorObject>& modelBlobs){
	
	for (vector<FloorObject>::iterator i = modelBlobs.begin(); i < modelBlobs.end(); i++){

		if (i->visualCounter > 0 && (i->state == hidden || i->state == stable))
			i->state = stable;
		else if (i->visualCounter >= ALPHA_PARAM)
			i->state = stable;
		else if (i->visualCounter <= 0 && i->visualCounter > -DEATH_PARAM)
			i->state = hidden;
		else if (i->visualCounter <= -DEATH_PARAM)
			i->state = dead;
	}

	modelBlobs.erase(remove_if(modelBlobs.begin(), modelBlobs.end(), [](FloorObject obj) {return obj.state == dead; }), modelBlobs.end());

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
	Mat colorMat(colorHeight, colorWidth, CV_8UC4), depthMat(depthHeight, depthWidth, CV_8UC4);	//color and depth images from Kinect

	//Create PointCloud (plus its ColorHandler) and add it to the Visualizer
	PointCloud<PointXYZRGB>::Ptr pointCloud(new PointCloud<PointXYZRGB>());

	vector<FloorObject> currentBlobs; //container of objects detected on the floor
	vector<FloorObject> modelBlobs;

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

	//Socket opening and connection
	WSADATA wsa;
	SOCKET s;
	struct sockaddr_in server;
	char *message, server_reply[2000];
	int recv_size = NULL;
	int consecutiveWrongFrames = 0;

	cout << "\nInitialising Winsock...";
	if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
	{
		cout << "Failed. Error Code : %d", WSAGetLastError();
		return 1;
	}

	cout << "Initialised.\n";

	//Create a socket
	if ((s = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET)
	{
		cout << "Could not create socket : %d", WSAGetLastError();
	}
	cout << "Socket created.\n";
	//Socket parameters
	server.sin_addr.s_addr = inet_addr("127.0.0.1");
	server.sin_family = AF_INET;
	server.sin_port = htons(4444);

	//Connect to remote server
	ConnectToSocket(s, server);

	// Main loop: kinect image processing
	while (!GetAsyncKeyState(VK_ESCAPE)){
		//Socket request and floor image retrieval
		message = "f"; //Request: "f" asks for a frame
		SendToSocket(s, message);
		//Receive a reply from the server
		ReceiveFromSocket(s, server_reply, recv_size);
		//Check received data
		Mat1b floor;
		if (CreateMatFromSocketReply(floor, server_reply, consecutiveWrongFrames) != 0){
			//Frame not validated - continue to next iteration
			continue;
		}

		//Process Floor image: resize, draw tiles, show
		resize(floor, floor, Size(static_cast<int>(4.2*SHARPNESS), static_cast<int>(3.0*SHARPNESS)), 0, 0, cv::INTER_CUBIC);
#ifdef DRAW_TILES
		for (int r = 0; r < floor.rows; r++){
			for (int c = 0; c < floor.cols; c++){
				if ((r % (floor.rows / 5)) == 0 || (c % ((floor.cols / 7))) == 0){
					floor(r, c) = 255;
				}
			}
		}
#endif
		imshow("Floor", floor);

		/////////////////////////////
		////////// KINECT ACQUISITION
		/////////////////////////////
		//Buffers will be used as containers for raw data
		vector<RGBQUAD> colorBuffer(colorWidth * colorHeight);
		vector<UINT16> depthBuffer(depthWidth * depthHeight);
		vector<UINT16> foregroundDepthBuffer(depthBuffer); //BGD Subtraction result

		ICoordinateMapper* coordinateMapper = acq.GetCoordinateMapper(); //init coordinate mapper to map color on depth data
		IColorFrameReader* colorReader = acq.GetColorReader();
		IDepthFrameReader* depthReader = acq.GetDepthReader();
		if (AcquireDepthAndRGBFrames(acq, colorReader, depthReader, coordinateMapper, colorMat, colorBuffer, depthMat, depthBuffer) != 0){
			continue;
		}

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
		coefficients->values[0] = -0.00424754;
		coefficients->values[1] = -0.981478;
		coefficients->values[2] = 0.19153;
		coefficients->values[3] = -1.11181;
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
			Eigen::Vector3f(0.893489282, -0.618893445, 2.42300010), // Origin
			transformation
			);
		transformPointCloud(*pointCloud, *pointCloud, transformation);

		///////////////////////////
		///// Centroid Calculation
		///////////////////////////
		PointXYZRGB sphereCenter = ComputePointCloudCentroid(pointCloud);

#ifdef PCL_VISUALIZER
		//PCLviewer->removeShape("Centroid Sphere");
		//PCLviewer->addSphere(sphereCenter, 0.2, "Centroid Sphere");
#endif
		///////////////////////////////////////
		///////////		POINTCLOUD TO MAT
		//////////////////////////////////////

		//At this point, we fill the Mat to revert to a grayscale image of the floor projection
		// offsets: translation from the Origin point
		float topLeftX = -0.55;
		float topLeftY = -0.75;
		float bottomRightX = topLeftX + 4.20; //floor is 4.20 x 3.00 m
		float bottomRightY = topLeftY + 3.00;
		Mat floorProjection(static_cast<int>(3.0 * SHARPNESS), static_cast<int>(4.2*SHARPNESS), CV_16UC1, Scalar(0, 0, 0));

		PointCloudXYPlaneToMat(pointCloud, floorProjection, topLeftX, topLeftY, bottomRightX, bottomRightY, depthHeight, depthWidth);

		//////////////////////////////
		/////  BLOB DETECTION - FLOOR
		/////////////////////////////
		FindFloorBlobsContours(floor, 20, currentBlobs);
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
			UpdateModelPositions(modelBlobs);
			UpdateModelState(modelBlobs);
		}

		//reconversion to Uchar for better visualization
		floorProjection.convertTo(floorProjection, CV_8UC1);
		contrastStretching(floorProjection);

		//////////////////////////
		///// Put infos together
		/////////////////////////
		Mat3b mergedChannels(floorProjection.rows, floorProjection.cols, Vec3b(0, 0, 0));

		for (int r = 0; r < mergedChannels.rows; r++){
			for (int c = 0; c < mergedChannels.cols; c++){
				// Red channel : floor projection
				// Green channel: florimage data
				mergedChannels(r, c)[2] = (char)floorProjection.at<uchar>(r, c);
				mergedChannels(r, c)[1] = (char)floor.at<uchar>(r, c);
			}
		}

		circle(mergedChannels, Point((sphereCenter.x - topLeftX)*SHARPNESS, (sphereCenter.y - topLeftY)*SHARPNESS), 6, Scalar(255, 0, 0));
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
				line(mergedChannels, (vertices[0] + vertices[1]), (vertices[2] + vertices[3]), it->color);
				line(mergedChannels, (vertices[1] + vertices[2]), (vertices[3] + vertices[0]), it->color);
			}
		}


		//Show results
		resize(mergedChannels, mergedChannels, Size(1200, 900), CV_INTER_CUBIC);
		namedWindow("Merged Channels", 1);
		createTrackbar("Sharpness", "Merged Channels", &SHARPNESS, 100);
		imshow("Projection", floorProjection);
		imshow("Merged Channels", mergedChannels);
		waitKey(100);

#ifdef PCL_VISUALIZER
		// PCL Visualizer
		PCLviewer->updatePointCloud<PointXYZRGB>(pointCloud, rgb, "Kinect Depth Cloud");
		PCLviewer->spinOnce();
		boost::this_thread::sleep(boost::posix_time::microseconds(10000));
#endif

		// Free memory
		vector<FloorObject>().swap(currentBlobs);
		vector<RGBQUAD>().swap(colorBuffer);
		vector<UINT16>().swap(depthBuffer);
		vector<UINT16>().swap(foregroundDepthBuffer);
		pointCloud->clear();
	}
	// End Processing
	return 0;
}
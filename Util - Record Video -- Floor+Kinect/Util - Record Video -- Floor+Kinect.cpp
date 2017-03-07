#include "stdafx.h"

#define NOMINMAX
#define _WINSOCK_DEPRECATED_NO_WARNINGS

#define BEST_MATCH_FIRST 1
#define HUNGARIAN_MIN_COST 2

#define ALPHA_PARAM 35
#define DEATH_PARAM 25
#define T_0 0.5
#define LAMBDA 0.75 //must be a value between 0 and 1

#define BLOB_RADIUS 0.4

int SHARPNESS = 100;

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

		// Qui si rimanda indietro risDef perchè sì (nel MAIN invece ris2)
		depthMat = ris2.clone();
		//imshow("", depthMat);
		//waitKey(30);
		delete[] depthArray;
	}
	SafeRelease(pDepthFrame);
	return 0;
}

int AcquireBodyFrameAndJoints(AcquisitionKinect2& acq, IBodyFrameReader* bodyReader, ICoordinateMapper* coordinateMapper, int num, int k){
	// BODY reader
	IBodyFrame* bodyFrame = NULL;
	HRESULT hResult = S_OK;
	hResult = bodyReader->AcquireLatestFrame(&bodyFrame);

	IBody* ppBodies[BODY_COUNT] = { 0 };
	if (SUCCEEDED(hResult))
	{
		// Saves bodies information into ppBodies var
		hResult = bodyFrame->GetAndRefreshBodyData(_countof(ppBodies), ppBodies);

		SafeRelease(bodyFrame);
	}

	fstream bodyStream;
	bodyStream.open("videos/" + to_string(num) + "/Bodies/ppBodies" + to_string(k) + ".bin", ios::out);

	for (int i = 0; i < BODY_COUNT; i++){
		IBody* pBody = ppBodies[i];

		if (pBody)
		{
			BOOLEAN bTracked = false;
			pBody->get_IsTracked(&bTracked);
			UINT64 id;
			pBody->get_TrackingId(&id);

			Joint joints[JointType_Count];

			if (bTracked){
				pBody->GetJoints(_countof(joints), joints);
				for (int i = 0; i < 25; i++){
					bodyStream << joints[i].Position.X << "\t";
					bodyStream << joints[i].Position.Y << "\t";
					bodyStream << joints[i].Position.Z << "\t";
					bodyStream << joints[i].TrackingState << "\t" << "\n";
				}
				bodyStream << "\n";
			}
		}
	}
	bodyStream.close();
	return 0;
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
	Mat colorMat(colorHeight, colorWidth, CV_8UC4), depthMat(depthHeight, depthWidth, CV_8UC4);	//color and depth images from Kinect

	// OPENCV RECORDING STUFF
	int fourcc = CV_FOURCC('M', 'J', 'P', 'G');
	String recPath = "C:/Users/Niccolò/Documents/Visual Studio 2013/Projects/Tesi_nb/Util - Record Video -- Floor+Kinect\videos";
	//VideoWriter KinectRGBVideo("videos/kinect_RGB.avi", fourcc, 30, Size(colorWidth, colorHeight), true);
	//VideoWriter KinectDVideo("videos/kinect_D.avi", fourcc, 30, Size(depthWidth, depthHeight), true);
	//VideoWriter FloorVideo("videos/floor.avi", fourcc, 30, Size(420, 300), 0);

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
	int k = 0;
	int num;
	cout << "Vid number: ";
	cin >> num;
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
		resize(floor, floor, Size(static_cast<int>(4.2*SHARPNESS), static_cast<int>(3.0*SHARPNESS)), 0, 0, cv::INTER_CUBIC);

		//Need to rotate Image 90° CCW
		transpose(floor, floor);
		flip(floor, floor, 0);

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
		IBodyFrameReader* bodyReader = acq.GetBodyReader();
		Joint* ppBodiesJoints[BODY_COUNT];
		for (int i = 0; i < BODY_COUNT; i++){
			ppBodiesJoints[i] = new Joint[JointType_Count];
		}
		if (AcquireDepthAndRGBFrames(acq, colorReader, depthReader, coordinateMapper, colorMat, colorBuffer, depthMat, depthBuffer) != 0){
			continue;
		}

		if (AcquireBodyFrameAndJoints(acq, bodyReader, coordinateMapper, num, k) != 0){
			continue;
		}

		String end = to_string(k) + ".png";
#ifdef _DEBUG
		end = "_debug_" + end;
#endif
		imwrite("videos/" + to_string(num) + "/RGB/rgb" + end, colorMat);
		imwrite("videos/" + to_string(num) + "/D/depth" + end, depthMat);
		imwrite("videos/" + to_string(num) + "/Floor/floor" + end, floor);

		//KinectRGBVideo << colorMat;
		//KinectDVideo << depthMat;
		//FloorVideo << floor;

		//Show results
		//imshow("RGB", colorMat);
		//imshow("Depth", depthMat);
		//imshow("Floor", floor);
		//waitKey(1);
		k++;
	}
	// End Processing
	return 0;
}
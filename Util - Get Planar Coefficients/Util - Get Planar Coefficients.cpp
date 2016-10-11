#include "stdafx.h"

#define NOMINMAX
#define _WINSOCK_DEPRECATED_NO_WARNINGS

#define BGD_SUBTRACTION false
#define CROP_BOX_FILTERING true
#define GET_PLANE_COEFFICIENTS true //used to search for floor plane coefficients (useful to retrieve plane coefficients the 1st time)
#define PCL_VISUALIZER

#include <opencv2\opencv.hpp>

#include <Windows.h>

#include <Kinect.h>
#include "acquisitionkinect2.h"

#include <pcl\visualization\cloud_viewer.h>
#include <pcl\visualization\pcl_visualizer.h>
#include <pcl/filters/crop_box.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/project_inliers.h>

#include <boost/thread.hpp>
#include <boost/chrono.hpp>

using namespace cv;
using namespace std;
using namespace pcl;


int _tmain(int argc, _TCHAR* argv[])
{
	// OPENCV stuff
	BackgroundSubtractorMOG2 pMOG2; //MOG2 Background subtractor
	Mat fgMaskMOG2; //fg mask generated by MOG2 method
	Mat colorMat, depthMat;	//color and depth images from Kinect

	//Custom Kinect Acquisition
	AcquisitionKinect2 acq = AcquisitionKinect2();
	FrameSet kinectFrame; //Acquisition frame: contains both depth and rgb images

	//Create PointCloud (plus its ColorHandler) and add it to the Visualizer
	PointCloud<PointXYZRGB>::Ptr pointCloud(new PointCloud<PointXYZRGB>());

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

	
	// Main loop: kinect image processing
	while (!GetAsyncKeyState(VK_ESCAPE)){
		
		//Get coordinate mapper to map colors into depth image, and readers to read data from RGB and Depth cameras
		ICoordinateMapper* coordinateMapper = acq.GetCoordinateMapper();
		IColorFrameReader* colorReader = acq.GetColorReader();
		IDepthFrameReader* depthReader = acq.GetDepthReader();

		// Kinect v2 Depth image's resolution is 512x424 pixels
		int depthWidth = 512;
		int depthHeight = 424;
		// To Reserve Depth Frame Buffer
		std::vector<UINT16> depthBuffer(depthWidth * depthHeight);
		// Kinect v2 Color image's resolution is FullHD (1920x1080)
		int colorWidth = 1920;
		int colorHeight = 1080;
		// To Reserve Color Frame Buffer
		std::vector<RGBQUAD> colorBuffer(colorWidth * colorHeight);
		//Buffers contain raw data from the 2 images

		/////////////////////////////
		////////// KINECT ACQUISITION
		/////////////////////////////

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
		else{ continue; }
		if (SUCCEEDED(hResult))
		{
			// conversion to OPENCV Mat
			Mat ris2 = Mat(colorHeight, colorWidth, CV_8UC4, &colorBuffer[0], Mat::AUTO_STEP).clone();
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
		else{ continue; }
		if (SUCCEEDED(hResult))
		{
			// conversion to OPENCV Mat
			UINT16* depthArray = new UINT16[depthHeight*depthWidth];
			copy(depthBuffer.begin(), depthBuffer.end(), depthArray);

			Mat ris2 = Mat(depthHeight, depthWidth, CV_16UC1, depthArray, Mat::AUTO_STEP).clone();
			Mat3b risDef(depthHeight, depthWidth);

			int x, y;
			for (y = 0; y<depthHeight; y++)
				for (x = 0; x<depthWidth; x++){
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

		/////////////////////////////
		////////////// IMG PROCESSING
		/////////////////////////////

		// BGD SUBTRACTION
		// Lets change some MOG2 parameters
		pMOG2.set("varThresholdGen", 625.0);
		pMOG2.set("backgroundRatio", 0.4);

		//store depthImage filtered after BGD Subtraction
		// initialized with depthBuffer, so that if BGD_SUBTRACTION is set to False, it contains original depth image
		std::vector<UINT16> foregroundDepthBuffer(depthBuffer);

		//Update the background model
		if (BGD_SUBTRACTION){
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
		}

		// Set Point Cloud Parameters
		pointCloud->width = static_cast<uint32_t>(depthWidth);
		pointCloud->height = static_cast<uint32_t>(depthHeight);
		pointCloud->is_dense = false;
		// Fill point cloud with points
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


		// Free memory
		vector<UINT16>().swap(depthBuffer);
		vector<UINT16>().swap(foregroundDepthBuffer);

		// Crop box: min and max points are corners of the parallelepiped that will be used as a filter
		Eigen::Vector4f minPoint;
		minPoint[0] = -1.8;  // define minimum point x (R)
		minPoint[1] = -1.20;  // define minimum point y (G)
		minPoint[2] = 0;  // define minimum point z (B)
		Eigen::Vector4f maxPoint;
		maxPoint[0] = 3.65;  // define max point x 
		maxPoint[1] = 1.56;  // define max point y 
		maxPoint[2] = 3.95;  // define max point z 

		if (CROP_BOX_FILTERING){
			CropBox<PointXYZRGB> cropFilter;
			PointCloud<PointXYZRGB>::Ptr pointCloud2(new PointCloud<PointXYZRGB>());
			*pointCloud2 = *pointCloud;
			cropFilter.setInputCloud(pointCloud2);
			cropFilter.setMin(minPoint);
			cropFilter.setMax(maxPoint);
			cropFilter.filter(*pointCloud);
			pointCloud2->clear();
		}

		///////////////////////////
		/////////// Plane Detection
		///////////////////////////

		//GET_PLANE_COEFFICIENTS variable is use to get coefficients of the floor plane (printed out to the command line)
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		if (GET_PLANE_COEFFICIENTS){
			pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
			// Create the segmentation object
			SACSegmentation<PointXYZRGB> seg;
			// Optional
			seg.setOptimizeCoefficients(true);
			// Mandatory
			seg.setModelType(SACMODEL_PLANE);
			seg.setMethodType(SAC_RANSAC);
			//seg.setDistanceThreshold(0.01);
			//seg.setProbability(0.9999);
			//seg.setMaxIterations(300);
			coefficients->values.resize(4);
			seg.setInputCloud(pointCloud);
			seg.segment(*inliers, *coefficients);

			cout << "* * * New Iteration * * *" << endl;
			cout << "Model coefficients: " << coefficients->values[0] << " "
				<< coefficients->values[1] << " "
				<< coefficients->values[2] << " "
				<< coefficients->values[3] << endl;
		}
		else{ //If variable is false, we set the coefficients manually and use them for the projection
			coefficients->values.resize(4);
			coefficients->values[0] = -0.00424754;
			coefficients->values[1] = -0.981478;
			coefficients->values[2] = 0.19153;
			coefficients->values[3] = -1.11181;
		}

		// Planar Projection
#ifdef PCL_VISUALIZER
		PCLviewer->removeShape("Projection plane");
#endif
		PCLviewer->addPlane(*coefficients, 0, 0, 1.7, "Projection plane"); //show projection plane

#ifdef PCL_VISUALIZER
		// PCL Visualizer
		PCLviewer->updatePointCloud<PointXYZRGB>(pointCloud, rgb, "Kinect Depth Cloud");
		PCLviewer->spinOnce();
		boost::this_thread::sleep(boost::posix_time::microseconds(10000));
#endif

		pointCloud->clear();
	}
	// End Processing
	return 0;
}
//
//  main.cpp
//  Testing
//
//  Created by hukama on 10/20/16.
//  Copyright © 2016 hukama. All rights reserved.
//

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
using namespace cv;
using namespace cv::ml;
using namespace std;

int main(int argc, char** argv)
{
    // Data for visual representation
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);
    //
    //
    vector<String> filenames;
    String folder = "/Users/hukama/Documents/OpenCVProject/Car/SVMGalaxy/Testing/Testing/TrainingData/";
    glob(folder, filenames);
    //
    int labels[6] ;//{0, 0, 0, 0};
    float trainingData[6][1] ;//= { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
    //
    for(size_t i = 0; i < filenames.size(); ++i)
    {
        if (filenames[i].find("galaxies_elliptical") != string::npos) {
            //.. found.
            Mat src = imread(filenames[i]);
            if(!src.data)
                cerr << "Problem loading image!!!" << endl;
            Mat gray,edge;
            cvtColor( src, gray, CV_BGR2GRAY );
            Canny( gray, edge, 50, 150, 3);
            //imshow(filenames[i], edge);
            double TotalNumberOfPixels = edge.rows * edge.cols;
            double ZeroPixels = countNonZero(edge)/TotalNumberOfPixels;
            //cout<<"galaxies_elliptical"<<i<<"="<<ZeroPixels<<endl;
            labels[i]=1;
            trainingData[i][0]=ZeroPixels;
            /* do whatever you want with your images here */
        }
        if (filenames[i].find("galaxies_spiral") != string::npos) {
            //.. found.
            Mat src = imread(filenames[i]);
            if(!src.data)
                cerr << "Problem loading image!!!" << endl;
            Mat gray,edge;
            cvtColor( src, gray, CV_BGR2GRAY );
            Canny( gray, edge, 50, 150, 3);
            //imshow(filenames[i], edge);
            double TotalNumberOfPixels = edge.rows * edge.cols;
            double ZeroPixels = countNonZero(edge)/TotalNumberOfPixels;
            //cout<<"galaxies_spiral"<<i<<"="<<ZeroPixels<<endl;
            labels[i]=-1;
            trainingData[i][0]=ZeroPixels;
            /* do whatever you want with your images here */
        }

    }
    //
    
    Mat trainingDataMat(6, 1, CV_32FC1, trainingData);
    Mat labelsMat(6, 1, CV_32SC1, labels);
    // Train the SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
    //svm->predict({505,5});
    ///Users/hukama/Documents/OpenCVProject/Car/SVMGalaxy/Testing/Testing/test1.jpg
   
    ostringstream alamat;
    string name ;
    if(argc>1){
        alamat <<"/Users/hukama/Documents/OpenCVProject/Car/SVMGalaxy/Testing/Testing/"<<argv[1];
        name=argv[1];
    }
    else{
        alamat <<"/Users/hukama/Documents/OpenCVProject/Car/SVMGalaxy/Testing/Testing/test3.jpeg";
        name="test";
    }
     //cout<<alamat.str()<<endl;
    Mat src1 = imread(alamat.str());
    if(!src1.data){
        cerr << "Problem loading image!!!" << endl;
        return 0;
    }
    Mat gray1,edge1;
    cvtColor( src1, gray1, CV_BGR2GRAY );
    Canny( gray1, edge1, 50, 150, 3);
    //imshow(filenames[i], edge);
    double TotalNumberOfPixels = edge1.rows * edge1.cols;
    double ZeroPixels1 = countNonZero(edge1)/TotalNumberOfPixels;
    //
    float testData[1][1];// = { {ZeroPixels1} };
    testData[0][0]=ZeroPixels1;
    Mat testDataMat(1, 1, CV_32FC1, testData);
    int response = (int)svm->predict( testDataMat );
    //std::cout<<"test="<<response<<std::endl;
    if(response==-1){
        std::cout<<name<<" is Spiral Galaxy"<<endl;
    }
    if(response==1){
        std::cout<<name<<" is Eliptical Galaxy"<<endl;
    }
    /*
    // Show the decision regions given by the SVM
    Vec3b green(0,255,0), blue (255,0,0);
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1,2) << j,i);
            float response = svm->predict(sampleMat);
            if (response == 1)
                image.at<Vec3b>(i,j)  = green;
            else if (response == -1)
                image.at<Vec3b>(i,j)  = blue;
        }
    // Show the training data
    int thickness = -1;
    int lineType = 8;
    circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType );
    circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType );
    circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType );
    circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType );
    // Show support vectors
    thickness = 2;
    lineType  = 8;
    Mat sv = svm->getUncompressedSupportVectors();
    for (int i = 0; i < sv.rows; ++i)
    {
        const float* v = sv.ptr<float>(i);
        circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
    }
    //imwrite("result.png", image);        // save the image
    //imshow("SVM Simple Example", image); // show it to the user
     */
    //waitKey(0);
    
    return 0;
}

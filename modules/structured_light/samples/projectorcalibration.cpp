/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "opencv2/opencv.hpp"
#include <opencv/highgui.h>
#include <vector>
#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

static const char* keys =
{
    "{@camCalibPath | | Path of camera calibration file}"
    "{@projSettingsPath | | Path of projector settings}"
    "{@patternPath | | Path to checkerboard pattern}"
    "{@outputName | | Base name for the calibration data}"
};

static void help()
{
    cout << endl;
}

enum calibrationPattern{ CHESSBOARD, CIRCLES_GRID, ASYMETRIC_CIRCLES_GRID };

struct Settings
{
    Settings();
    int patternType;
    Size patternSize;
    Size subpixelSize;
    Size imageSize;
    float squareSize;
    int nbrOfFrames;
};

void loadSettings( String path, Settings &sttngs );

void createObjectPoints( vector<Point3f> &patternCorners, Size patternSize, float squareSize,
                        int patternType );

void createProjectorObjectPoints( vector<Point2f> &patternCorners, Size patternSize, float squareSize,
                        int patternType );

float calibrate( vector< vector<Point3f> > objPoints, vector< vector<Point2f> > imgPoints,
               Mat &cameraMatrix, Mat &distCoeffs, vector<Mat> &r, vector<Mat> &t, Size imgSize );

void fromCamToWorld( Mat cameraMatrix, vector<Mat> rV, vector<Mat> tV,
                    vector< vector<Point2f> > imgPoints, vector< vector<Point3f> > &worldPoints );

int main( int argc, char **argv )
{
    VideoCapture cap(CV_CAP_PVAPI);
    Mat frame;
    Mat gray;

    int nbrOfValidFrames = 0;

    vector< vector<Point2f> > imagePointsCam, imagePointsProj, PointsInProj;
    vector< vector<Point3f> > objectPointsCam, worldPointsProj;
    vector<Point3f> tempCam;
    vector<Point2f> tempProj;

    vector<Mat> rVecs, tVecs, projectorRVecs, projectorTVecs;
    Mat cameraMatrix, distCoeffs, projectorMatrix, projectorDistCoeffs;
    Mat pattern;

    Settings camSettings, projSettings;

    CommandLineParser parser(argc, argv, keys);

    String camSettingsPath = parser.get<String>(0);
    String projSettingsPath = parser.get<String>(1);
    String patternPath = parser.get<String>(2);

    if( camSettingsPath.empty() || projSettingsPath.empty() || patternPath.empty() ){
        help();
        return -1;
    }

    pattern = imread(patternPath);

    loadSettings(camSettingsPath, camSettings);
    loadSettings(projSettingsPath, projSettings);

    projSettings.imageSize = Size(pattern.rows, pattern.cols);

    createObjectPoints(tempCam, camSettings.patternSize,
                       camSettings.squareSize, camSettings.patternType);
    createProjectorObjectPoints(tempProj, projSettings.patternSize,
                                projSettings.squareSize, projSettings.patternType);

    if(!cap.isOpened())
    {
        cout << "Camera could not be opened" << endl;
        return -1;
    }
    cap.set(CAP_PROP_PVAPI_PIXELFORMAT, CAP_PVAPI_PIXELFORMAT_MONO8);

    namedWindow("pattern", WINDOW_NORMAL);
    setWindowProperty("pattern", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

    namedWindow("camera view", WINDOW_NORMAL);

    imshow("pattern", pattern);
    cout << "Press any key when ready" << endl;
    waitKey(0);

    while( nbrOfValidFrames < camSettings.nbrOfFrames )
    {
        cap >> frame;
        if( frame.data )
        {
            if( camSettings.imageSize.height == 0 || camSettings.imageSize.width == 0 )
            {
                camSettings.imageSize = Size(frame.rows, frame.cols);
            }

            bool foundProj, foundCam;

            vector<Point2f> projPointBuf;
            vector<Point2f> camPointBuf;

            imshow("camera view", gray);
            if( camSettings.patternType == CHESSBOARD && projSettings.patternType == CHESSBOARD )
            {
                int calibFlags = CALIB_CB_ADAPTIVE_THRESH;

                foundCam = findChessboardCorners(frame, camSettings.patternSize,
                                                 camPointBuf, calibFlags);

                foundProj = findChessboardCorners(frame, projSettings.patternSize,
                                                  projPointBuf, calibFlags);

                if( foundCam && foundProj )
                {
                    cout << "found pattern" << endl;
                    Mat projCorners, camCorners;
                    cornerSubPix(gray, camPointBuf, camSettings.subpixelSize, Size(-1, -1),
                            TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

                    cornerSubPix(gray, projPointBuf, projSettings.subpixelSize, Size(-1, -1),
                            TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

                    drawChessboardCorners(frame, camSettings.patternSize, camPointBuf, foundCam);
                    drawChessboardCorners(frame, projSettings.patternSize, projPointBuf, foundProj);

                    imshow("camera view", gray);
                    char c = (char)waitKey(0);
                    if( c == 10 )
                    {
                        cout << "saving pattern #" << nbrOfValidFrames << " for calibration" << endl;
                        ostringstream name;
                        name << nbrOfValidFrames;
                        imwrite("capture" + name.str() + ".png", frame);
                        nbrOfValidFrames += 1;

                        imagePointsCam.push_back(camPointBuf);
                        imagePointsProj.push_back(projPointBuf);
                        objectPointsCam.push_back(tempCam);
                        PointsInProj.push_back(tempProj);
                    }
                    else if( c == 32 )
                    {
                       cout << "capture discarded" << endl;
                    }
                    else if( c == 27 )
                    {
                        cout << "closing program" << endl;
                        return -1;
                    }
                }
                else
                {
                    cout << "no pattern found, move board and press any key" << endl;
                    imshow("camera view", gray);
                    waitKey(0);
                }
            }
        }
    }

    float rms = calibrate(objectPointsCam, imagePointsCam, cameraMatrix, distCoeffs,
                          rVecs, tVecs, camSettings.imageSize);
    cout << "rms = " << rms << endl;
    cout << "camera matrix = \n" << cameraMatrix << endl;
    cout << "dist coeffs = \n" << distCoeffs << endl;

    fromCamToWorld(cameraMatrix, rVecs, tVecs, imagePointsProj, worldPointsProj);

    rms = calibrate(worldPointsProj, PointsInProj, projectorMatrix, projectorDistCoeffs,
                    projectorRVecs, projectorTVecs, projSettings.imageSize);

    return 0;
}

Settings::Settings(){
    patternType = CHESSBOARD;
    patternSize = Size(13, 9);
    subpixelSize = Size(11, 11);
    squareSize = 50;
    nbrOfFrames = 25;
}

void loadSettings( String path, Settings &sttngs )
{
    FileStorage fsInput(path, FileStorage::READ);

    fsInput["PatternWidth"] >> sttngs.patternSize.width;
    fsInput["PatternHeight"] >> sttngs.patternSize.height;
    fsInput["SubPixelWidth"] >> sttngs.subpixelSize.width;
    fsInput["SubPixelHeight"] >> sttngs.subpixelSize.height;
    fsInput["SquareSize"] >> sttngs.squareSize;
    fsInput["NbrOfFrames"] >> sttngs.nbrOfFrames;
    fsInput["PatternType"] >> sttngs.patternType;
    fsInput.release();
}

float calibrate( vector< vector<Point3f> > objPoints, vector< vector<Point2f> > imgPoints,
               Mat &cameraMatrix, Mat &distCoeffs, vector<Mat> &r, vector<Mat> &t, Size imgSize )
{
    int calibFlags = 0;

    float rms = calibrateCamera(objPoints, imgPoints, imgSize, cameraMatrix,
                                distCoeffs, r, t, calibFlags,
                                TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

    return rms;
}

void createObjectPoints( vector<Point3f> &patternCorners, Size patternSize, float squareSize,
                         int patternType )
{
    switch( patternType )
    {
        case CHESSBOARD:
        case CIRCLES_GRID:
            for( int i = 0; i < patternSize.height; ++i )
            {
                for( int j = 0; j < patternSize.width; ++j )
                {
                    patternCorners.push_back(Point3f(float(i*squareSize), float(j*squareSize), 0));
                }
            }
            break;
        case ASYMETRIC_CIRCLES_GRID:
            break;
    }
}

void createProjectorObjectPoints( vector<Point2f> &patternCorners, Size patternSize, float squareSize,
                        int patternType )
{
    switch( patternType )
    {
        case CHESSBOARD:
        case CIRCLES_GRID:
            for( int i = 1; i <= patternSize.height; ++i )
            {
                for( int j = 1; j <= patternSize.width; ++j )
                {
                    patternCorners.push_back(Point2f(float(j*squareSize), float(i*squareSize)));
                }
            }
            break;
        case ASYMETRIC_CIRCLES_GRID:
            break;
    }
}

void fromCamToWorld( Mat cameraMatrix, vector<Mat> rV, vector<Mat> tV,
                    vector< vector<Point2f> > imgPoints, vector< vector<Point3f> > &worldPoints )
{
    int s = (int) rV.size();
    Mat invK64, invK;
    invK64 = cameraMatrix.inv();
    invK64.convertTo(invK, CV_32F);

    for(int i = 0; i < s; ++i)
    {
        Mat r, t, rMat;
        rV[i].convertTo(r, CV_32F);
        tV[i].convertTo(t, CV_32F);

        Rodrigues(r, rMat);
        Mat transPlaneToCam = rMat.inv()*t;

        vector<Point3f> wpTemp;
        int s2 = (int) imgPoints[i].size();
        for(int j = 0; j < s2; ++j){
            Mat coords(3, 1, CV_32F);
            coords.at<float>(0, 0) = imgPoints[i][j].x;
            coords.at<float>(1, 0) = imgPoints[i][j].y;
            coords.at<float>(2, 0) = 1.0f;

            Mat worldPtCam = invK*coords;
            Mat worldPtPlane = rMat.inv()*worldPtCam;

            float scale = transPlaneToCam.at<float>(2)/worldPtPlane.at<float>(2);
            Mat worldPtPlaneReproject = scale*worldPtPlane - transPlaneToCam;

            Point3f pt;
            pt.x = worldPtPlaneReproject.at<float>(0);
            pt.y = worldPtPlaneReproject.at<float>(1);
            pt.z = 0;
            wpTemp.push_back(pt);
        }
        worldPoints.push_back(wpTemp);
    }
}
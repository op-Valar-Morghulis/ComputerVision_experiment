#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


void alpha_blend(const char* path1, const char* path2) {
    //@todo:提取图1的alpha通道然后显示；
    //@todo:新图=图1*alpha+图2*（1-alpha），显示；

    Mat alphapre = imread(path1,IMREAD_UNCHANGED);
    Mat foreground = imread(path1);
    Mat background = imread(path2);
    resize(foreground, foreground, Size(background.cols, background.rows));
    resize(alphapre, alphapre, Size(background.cols, background.rows));
    assert(alphapre.channels() == 4);

    vector<Mat> mask_channels;
    split(alphapre, mask_channels);
    Mat alphamask = mask_channels[3];

    namedWindow("Alphamask");
    imshow("Alphamask",alphamask);
    waitKey(3000);

    //都转换成这个防止乘法溢出
    alphamask.convertTo(alphamask,CV_32FC1,1.0/255);//@todo:这里是不是没有必要
    foreground.convertTo(foreground,CV_32FC3,1.0/255);
    background.convertTo(background,CV_32FC3,1.0/255);


    Mat outImage = Mat::zeros(foreground.size(),foreground.type());

    //分离每个通道分别相乘
    vector<Mat> foreground_channels;
    split(foreground,foreground_channels);
    for(int i=0;i<3;i++){
        multiply(alphamask,foreground_channels[i],foreground_channels[i]);
    }


    vector<Mat> background_channels;
    split(background,background_channels);
    for(int i=0;i<3;i++){
        multiply(Scalar::all(1)-alphamask,background_channels[i],background_channels[i]);
    }


    merge(foreground_channels,foreground);
    merge(background_channels,background);

    add(foreground,background,outImage);

    namedWindow("Alphablended");
    imshow("Alphablended",outImage);
    waitKey(0);

    return;
}
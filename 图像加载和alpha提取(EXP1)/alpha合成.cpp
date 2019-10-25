//
// Created by 陈佳睿 on 2019/9/18.
//

#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


//alpha合成的逻辑
//新图=图1*alpha+图2*（1-alpha）
int main()
{
    Mat img = imread("/Users/chenjiarui/Desktop/a.png", IMREAD_UNCHANGED);
    //imshow("img",img);
    //cout<<img.rows<<" "<<img.cols<<endl;
    Mat back_img = imread("/Users/chenjiarui/Desktop/b.png", IMREAD_UNCHANGED);
    Mat mat(img.rows, img.cols, CV_8UC4);//可以创建8位无符号的四通道透明色的RGBA图像
    //cout<<mat.rows<<" "<<mat.cols<<endl;
    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            double temp = img.at<Vec4b>(i, j)[3] / 255.00;
            mat.at<Vec4b>(i, j)[0] = (1 - temp)*back_img.at<Vec4b>(i, j)[0] + temp * img.at<Vec4b>(i, j)[0];
            mat.at<Vec4b>(i, j)[1] = (1 - temp)*back_img.at<Vec4b>(i, j)[1] + temp * img.at<Vec4b>(i, j)[1];
            mat.at<Vec4b>(i, j)[2] = (1 - temp)*back_img.at<Vec4b>(i, j)[2] + temp * img.at<Vec4b>(i, j)[2];
            mat.at<Vec4b>(i, j)[3] = (1 - temp)*back_img.at<Vec4b>(i, j)[3] + temp * img.at<Vec4b>(i, j)[3];
        }
    }
    imshow("alpha混合", mat);
    waitKey();
    return 0;
}

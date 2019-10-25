#include <opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include"opencv2/imgproc/imgproc.hpp"
#include<math.h>
#include <iostream>

using namespace std;
using namespace cv;



int main()
{
    Mat foreground,background;
    foreground = imread("/Users/chenjiarui/Desktop/bgs-data/13.jpg");
    background = imread("/Users/chenjiarui/Desktop/bgs-data/13_bg.jpg");
    if(!foreground.data||!background.data)
    {
        cout<<"读取照片失败"<<endl;
        return -1;
    }
    if(foreground.rows!=background.rows||foreground.cols!=background.cols)
        cout<<"前后景尺寸不一致"<<endl;
    Mat target = Mat::zeros(foreground.size(),foreground.type());
    namedWindow("target_pic",WINDOW_AUTOSIZE);



    double sum = 0.0;

    for (int y = 0; y < foreground.rows; y++)
    {
        for (int x = 0; x < foreground.cols; x++)
        {
            sum = 0.0;
            for (int c = 0; c < 3; c++)
            {
                sum += pow((foreground.at<Vec3b>(y, x)[c] - background.at<Vec3b>(y, x)[c]), 2);
            }
            sum = sqrt(sum);
            if (sum >= 120) {
                for (int c = 0; c < 3; c++)
                {
                    target.at<Vec3b>(y, x)[c] = saturate_cast < uchar>(255);
                }
            }
            else {
                for (int c = 0; c < 3; c++)
                {
                    target.at<Vec3b>(y, x)[c] = saturate_cast < uchar>(0);
                }
            }
        }
    }

//    for(int y=0;y<foreground.rows;y++) {
//        for (int x = 0; x < foreground.cols; x++) {
//            double sum = 0.0;
//            for (int c = 0; c < 3; c++) {
//                sum += pow(foreground.ptr<Vec3b>(y)[x][c] - background.ptr<Vec3b>(y)[x][c], 2);
//            }
//            if (sum >= 30)
//                for (int c = 0; c < 3; c++) {
//                    target.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(0);
//                }
//            else {
//                for (int c = 0; c < 3; c++) {
//                    target.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(255);
//                }
//
//            }
//        }
//    }
    imshow("target_pic", target);
    waitKey(0);
    return 0;

}

#include <opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include"opencv2/imgproc/imgproc.hpp"
#include<math.h>
#include <iostream>
using namespace std;
using namespace cv;

const string wndName1 = "raw_picture";
const string wndName2 = "Contrast_adjust";

const string trackName = "track";
int ContrastValue; //对比度值
int BrightValue;  //亮度值

Mat src;     //原始图片
Mat dst;     //目标图片

// 设置的调整参数
//void cvTrackbarCallback(int, void*)
//{
//    //三个for循环，执行运算 dst(i,j) =a*src(i,j) + b
//    for(int y = 0; y < src.rows; y++) {
//        for(int x = 0; x < src.cols; x++) {
//            for(int c = 0; c < 3; c++) {
//                float  t = (255/(src.at<Vec3b>(y, x)[c] - 127.5))*ContrastValue*0.1;
//                dst.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(src.at<Vec3b>(y, x)[c] * ((1.00 / (1.00 + exp(-t))) + 0.3) + BrightValue - 100);
//            }
//        }
//    }
//    //显示图像
//    imshow(wndName2, dst);
//}

void cvTrackbarCallback(int, void*){
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            for(int c=0;c<=3;c++)
            {
                float  t = (((src.ptr<Vec3b>(i)[j][c]-127.5) / 255.00)*ContrastValue*0.1);
                dst.ptr<Vec3b>(i)[j][c]= saturate_cast<uchar>(src.ptr<Vec3b>(i)[j][c]*((1.00 / (1.00 + exp(-t))) + 0.3) + BrightValue - 100);
            }
        }

    }
    //显示图像
    imshow(wndName1, src);
    imshow(wndName2, dst);

}

void TrackAndAdjustImage()
{
    src = imread("/Users/chenjiarui/Desktop/a.png"); //读取图片1
    if (!src.data)
    {
        cout << "图片读取失败！" << endl;
        return;
    }
    dst= Mat::zeros(src.size(), src.type()); //按照图片1的尺寸和类型初始化图片2


    ContrastValue = 20; //设置对比度初始值
    BrightValue = 100;   //设置亮度初始值

//    namedWindow(wndName1, 1); //创建两个窗口用来对比图片
    namedWindow(wndName2, 1);

    //创建滑块函数：名称+窗口名称+关联数据+最大值+回调函数接口
    createTrackbar("Contrast", wndName2, &ContrastValue, 200, cvTrackbarCallback);
    createTrackbar("Bright", wndName2, &BrightValue, 200, cvTrackbarCallback);
    cvTrackbarCallback(ContrastValue, 0);
    cvTrackbarCallback(BrightValue,0);
}

int main(int argc, char *argv[])
{
    TrackAndAdjustImage();
    waitKey(0);
    return 0;
}

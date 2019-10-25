#include<opencv2/core/saturate.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include<iostream>
#include<cmath>
#include <ctime>

using namespace std;
using namespace cv;

//滤波窗口（卷积核）大小取为[6*sigma-1] 取整
//取整是为了让卷积核窗口取得奇整数值
void Gaussian(const Mat &src, Mat &dst, double sigma)
{
    CV_Assert(src.channels() || src.channels() == 3); // 只处理单通道或者三通道图像
    int ksize = (int)((6*sigma-1)/2)*2+1;
    // 根据窗口大小和sigma生成高斯滤波器模板
    // 申请一个二维数组，存放生成的高斯模板矩阵
    double **templateMatrix = new double*[ksize];
    for (int i = 0; i < ksize; i++)
        templateMatrix[i] = new double[ksize];
    int origin = ksize / 2; // 以模板的中心为原点
    double x2, y2;
    double sum = 0;
    for (int i = 0; i < ksize; i++)
    {
        x2 = pow(i - origin, 2);
        for (int j = 0; j < ksize; j++)
        {
            y2 = pow(j - origin, 2);
            // 高斯函数前的常数可以不用计算，会在归一化的过程中给消去
            double g = exp(-(x2 + y2) / (2 * sigma * sigma));
            sum += g;
            templateMatrix[i][j] = g;
        }
    }
    for (int i = 0; i < ksize; i++)
    {
        for (int j = 0; j < ksize; j++)
        {
            templateMatrix[i][j] /= sum;
//            cout << templateMatrix[i][j] << " ";
        }
//        cout << endl;
    }
    // 将模板应用到图像中
    int border = ksize / 2;
    // 对称法，BORDER_REFLECT_101 用边缘像素点进行padding
    copyMakeBorder(src, dst, border, border, border, border, BorderTypes::BORDER_REFLECT_101);
    int channels = dst.channels();
    int rows = dst.rows - border;
    int cols = dst.cols - border;
    //对于原图进行遍历 卷积
    for (int i = border; i < rows; i++)
    {
        for (int j = border; j < cols; j++)
        {
            double sum[3] = { 0 };
            for (int a = -border; a <= border; a++)
            {
                for (int b = -border; b <= border; b++)
                {
                    if (channels == 1)
                    {
                        sum[0] += templateMatrix[border + a][border + b] * dst.at<uchar>(i + a, j + b);
                    }
                    else if (channels == 3)
                    {
                        Vec3b rgb = dst.at<Vec3b>(i + a, j + b);
                        auto k = templateMatrix[border + a][border + b];
                        sum[0] += k * rgb[0];
                        sum[1] += k * rgb[1];
                        sum[2] += k * rgb[2];
                    }
                }
            }
            for (int k = 0; k < channels; k++)
            {
                if (sum[k] < 0)
                    sum[k] = 0;
                else if (sum[k] > 255)
                    sum[k] = 255;
            }
            if (channels == 1)
                dst.at<uchar>(i, j) = static_cast<uchar>(sum[0]);
            else if (channels == 3)
            {
                Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
                dst.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    // 释放模板数组
    for (int i = 0; i < ksize; i++)
        delete[] templateMatrix[i];
    delete[] templateMatrix;
}

void separateGaussianFilter(const Mat &src, Mat &dst, double sigma)
{
    int ksize = (int)((6*sigma-1)/2)*2+1;
    CV_Assert(src.channels() || src.channels() == 3); // 只处理单通道或者三通道图像
    double *matrix = new double[ksize];
    double sum = 0;
    int origin = ksize / 2;
    for (int i = 0; i < ksize; i++)
    {
        // 高斯函数前的常数可以不用计算，会在归一化的过程中给消去
        double g = exp(-(i - origin) * (i - origin) / (2 * sigma * sigma));
        sum += g;
        matrix[i] = g;
    }
    // 归一化
    for (int i = 0; i < ksize; i++)
        matrix[i] /= sum;
    // 将模板应用到图像中
    int border = ksize / 2;
    copyMakeBorder(src, dst, border, border, border, border, BorderTypes::BORDER_REFLECT_101);
    int channels = dst.channels();
    int rows = dst.rows - border;
    int cols = dst.cols - border;
    // 水平方向
    for (int i = border; i < rows; i++)
    {
        for (int j = border; j < cols; j++)
        {
            double sum[3] = { 0 };
            for (int k = -border; k <= border; k++)
            {
                if (channels == 1)
                {
                    sum[0] += matrix[border + k] * dst.at<uchar>(i, j + k); // 行不变，列变化；先做水平方向的卷积
                }
                else if (channels == 3)
                {
                    Vec3b rgb = dst.at<Vec3b>(i, j + k);
                    sum[0] += matrix[border + k] * rgb[0];
                    sum[1] += matrix[border + k] * rgb[1];
                    sum[2] += matrix[border + k] * rgb[2];
                }
            }
            for (int k = 0; k < channels; k++)
            {
                if (sum[k] < 0)
                    sum[k] = 0;
                else if (sum[k] > 255)
                    sum[k] = 255;
            }
            if (channels == 1)
                dst.at<uchar>(i, j) = static_cast<uchar>(sum[0]);
            else if (channels == 3)
            {
                Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
                dst.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    // 竖直方向
    for (int i = border; i < rows; i++)
    {
        for (int j = border; j < cols; j++)
        {
            double sum[3] = { 0 };
            for (int k = -border; k <= border; k++)
            {
                if (channels == 1)
                {
                    sum[0] += matrix[border + k] * dst.at<uchar>(i + k, j); // 列不变，行变化；竖直方向的卷积
                }
                else if (channels == 3)
                {
                    Vec3b rgb = dst.at<Vec3b>(i + k, j);
                    sum[0] += matrix[border + k] * rgb[0];
                    sum[1] += matrix[border + k] * rgb[1];
                    sum[2] += matrix[border + k] * rgb[2];
                }
            }
            for (int k = 0; k < channels; k++)
            {
                if (sum[k] < 0)
                    sum[k] = 0;
                else if (sum[k] > 255)
                    sum[k] = 255;
            }
            if (channels == 1)
                dst.at<uchar>(i, j) = static_cast<uchar>(sum[0]);
            else if (channels == 3)
            {
                Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
                dst.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    delete[] matrix;

}
void Fast_integral(cv::Mat& input, cv::Mat& output){
    int nr = input.rows;
    int nc = input.cols;
    int sum_r =0;
    output = Mat::zeros(nr+1,nc+1,CV_64F);
    for(int i=1;i<output.rows;++i){
        for(int j=1,sum_r=0;j<output.cols;++j){
            sum_r = input.at<uchar>(i-1 , j-1) + sum_r;
            output.at<double>(i, j) = output.at<double>(i-1, j)+sum_r;
        }
    }
}


//采用积分图进行加速，实现与滤波窗口大小无关的效率
//滤波窗口的大小为2w+1
void MeanFilter(const Mat &input, Mat &output, int window_size)
{

    int Image_Height = (window_size-1)/2;
    int Image_Width = (window_size-1)/2;
    Mat src;
    copyMakeBorder(input,src,Image_Height,Image_Height,Image_Width,Image_Width,BORDER_REFLECT_101);
    //src 为输入图像padding后的结果
    //dst 为最后得到的结果图像
    Mat dst = Mat::zeros(input.size(),input.type());

    Mat integral; //图积分的结果
    Fast_integral(src,integral);

    double mean = 0;
    for(int i=Image_Height+1;i<input.rows+Image_Height+1;++i){
        for(int j=Image_Width+1;j<input.cols+Image_Width+1;++j){
            double top_left = integral.at<uchar>(i-Image_Height-1,j-Image_Width-1);
            double  top_right = integral.at<uchar>(i-Image_Height-1,j+Image_Width);
            double bottom_left = integral.at<uchar>(i+Image_Height,j-Image_Width-1);
            double bottom_right = integral.at<uchar>(i+Image_Height,j-Image_Width);
            mean = (bottom_right+top_left-top_right-bottom_left)/pow(window_size,2);
            if (mean < 0)
                mean = 0;
            else if (mean>255)
                mean = 255;
            dst.at<uchar>(i-Image_Height-1,j-Image_Width-1) = static_cast<uchar>(mean);
        }

    }
}
//
//void Fast_MeanFilter(cv::Mat& src, cv::Mat& dst, cv::Size wsize){
//
//    //图像边界扩充
//    if(wsize.height%2==0||wsize.width%2==0){
//        cout<<"输入的窗口大小为偶数"<<endl;
//        exit(1);
//    }
//    int hh = (wsize.height - 1) / 2;
//    int hw = (wsize.width - 1) / 2;
//    cv::Mat Newsrc;
//    cv::copyMakeBorder(src, Newsrc, hh, hh, hw, hw, cv::BORDER_REFLECT_101);//以边缘为轴，对称
//    dst = cv::Mat::zeros(src.size(), src.type());
//
//    //计算积分图
//    cv::Mat inte;
//    Fast_integral(Newsrc, inte);
//    int channels = src.channels();
//    //均值滤波
//    double mean = 0;
//    if(channels == 1){
//        cout << "单通道图片均值滤波..." << endl;
//        for (int i = hh+1; i < src.rows + hh + 1;++i){  //积分图图像比原图（边界扩充后的）多一行和一列
//            for (int j = hw+1; j < src.cols + hw + 1; ++j){
//                double top_left = inte.at<double>(i - hh - 1, j - hw-1);
//                double top_right = inte.at<double>(i-hh-1,j+hw);
//                double buttom_left = inte.at<double>(i + hh, j - hw- 1);
//                double buttom_right = inte.at<double>(i+hh,j+hw);
//                mean = (buttom_right - top_right - buttom_left + top_left) / wsize.area();
//
//                //一定要进行判断和数据类型转换
//                if (mean < 0)
//                    mean = 0;
//                else if (mean>255)
//                    mean = 255;
//                dst.at<uchar>(i - hh - 1, j - hw - 1) = static_cast<uchar> (mean);
//            }
//        }
//    }
//    else if (channels == 3){
//        cout << "三通道图片均值滤波..." << endl;
//        double top_left[3],top_right[3],buttom_left[3],buttom_right[3];
//        double mean[3];
//        for (int i = hh+1; i < src.rows + hh + 1;++i){  //积分图图像比原图（边界扩充后的）多一行和一列
//            for (int j = hw+1; j < src.cols + hw + 1; ++j){
//                for(int c=0;c<3;c++){
//                    top_left[c]  = inte.at<Vec3b>(i - hh - 1, j - hw-1)[c];
//                    top_right[c] = inte.at<Vec3b>(i-hh-1,j+hw)[c];
//                    buttom_left[c] = inte.at<Vec3b>(i + hh, j - hw- 1)[c];
//                    buttom_right[c] = inte.at<Vec3b>(i+hh,j+hw)[c];
//                    mean[c] = (buttom_right[c] - top_right[c] - buttom_left[c] + top_left[c]) / wsize.area();
//                    if (mean[c] < 0)
//                        mean[c] = 0;
//                    else if (mean[c]>255)
//                        mean[c] = 255;
//                    Vec3b rgb = { static_cast<uchar>(mean[0]), static_cast<uchar>(mean[1]), static_cast<uchar>(mean[2]) };
//                    dst.at<Vec3b>(i - hh - 1, j - hw - 1) = rgb;
//                }
//            }
//        }
//    }
//}

void Fast_MeanFilter(cv::Mat& src, cv::Mat& dst, cv::Size wsize){

    //图像边界扩充
    if(wsize.height%2==0||wsize.width%2==0){
        cout<<"输入的窗口大小为偶数"<<endl;
        exit(1);
    }
    int hh = (wsize.height - 1) / 2;
    int hw = (wsize.width - 1) / 2;
    cv::Mat Newsrc;
    cv::copyMakeBorder(src, Newsrc, hh, hh, hw, hw, cv::BORDER_REFLECT_101);//以边缘为轴，对称
    dst = cv::Mat::zeros(src.size(), src.type());

    //计算积分图
    cv::Mat inte;
    Fast_integral(Newsrc, inte);
    int channels = src.channels();
    //均值滤波
    double mean = 0;

        cout << "单通道图片均值滤波..." << endl;
        for (int i = hh+1; i < src.rows + hh + 1;++i){  //积分图图像比原图（边界扩充后的）多一行和一列
            for (int j = hw+1; j < src.cols + hw + 1; ++j){
                double top_left = inte.at<double>(i - hh - 1, j - hw-1);
                double top_right = inte.at<double>(i-hh-1,j+hw);
                double buttom_left = inte.at<double>(i + hh, j - hw- 1);
                double buttom_right = inte.at<double>(i+hh,j+hw);
                mean = (buttom_right - top_right - buttom_left + top_left) / wsize.area();

                //一定要进行判断和数据类型转换
                if (mean < 0)
                    mean = 0;
                else if (mean>255)
                    mean = 255;
                dst.at<uchar>(i - hh - 1, j - hw - 1) = static_cast<uchar> (mean);
            }
        }
}


int main(){
    Mat src,dst;
    clock_t startTime,endTime;
    int sigma = 8;
    src = imread("/Users/chenjiarui/Desktop/timg.jpeg",1);
    cvtColor(src,src,COLOR_BGR2GRAY);

//    dst = Mat::zeros(src.rows, src.cols, src.type());
//    namedWindow("source", WINDOW_AUTOSIZE);
//    imshow("source", src);
//    cout << src.cols << ' ' << src.rows << ' ' << endl;
//    namedWindow("padding", WINDOW_AUTOSIZE);
//    startTime = clock();
//    Gaussian(src,dst,sigma);
//    endTime = clock();
//    cout << "The run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
//    imshow("padding",dst);

//#########################################################

//    namedWindow("seperate_padding", WINDOW_AUTOSIZE);
//    startTime = clock();
//    separateGaussianFilter(src,dst,sigma);
//    endTime = clock();
//    cout << "The run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
//    imshow("seperate_padding",dst);

//##############################################################


    Mat meanf;
    namedWindow("meanFilter",WINDOW_AUTOSIZE);
    Fast_MeanFilter(src,meanf,cv::Size(9,9));
    imshow("meanFilter",meanf);

    waitKey(0);
    return 0;
}


// 函数模版
// 参数解释 window是存储生成的系数 ksize存储卷积核的大小  sigma是标准差
// 归一化  => 使用window矩阵左上角第一个元素的倒数
void generateGaussianTemplate_1(double window[][11],int result[][11],int ksize,double sigma)
{
    static const double pi = 3.1415926;
    int center = ksize/2;
    double x2,y2;
    for(int i=0;i<ksize;i++){
        x2 = pow(i-center,2);
        for(int j=0;j<ksize;j++){
            y2 = pow(j-center,2);
            double G = exp(-(x2+y2)/(2*sigma*sigma));
            G /=2*pi*sigma;
            window[i][j]=G;
        }
    }
    double k = 1/window[0][0];
    for(int i=0;i<ksize;i++)
    {
        for(int j=0;j<ksize;j++)
        {
            // 此时得到的矩阵仍然是一个带小数的高斯滤波器矩阵
            window[i][j]*=k;
            // result为最终得到的整数型的高斯滤波器模版
            result[i][j]=(int)window[i][j];
        }
    }
}

// 归一化  => 每一个系数除以所有系数之和
void generateGaussianTemplate_2(double window[][11], int ksize, double sigma)
{
    static const double pi = 3.1415926;
    int center = ksize / 2; // 模板的中心位置，也就是坐标的原点
    double x2, y2;
    double sum = 0;
    for (int i = 0; i < ksize; i++)
    {
        x2 = pow(i - center, 2);
        for (int j = 0; j < ksize; j++)
        {
            y2 = pow(j - center, 2);
            double g = exp(-(x2 + y2) / (2 * sigma * sigma));
            g /= 2 * pi * sigma;
            sum += g;
            window[i][j] = g;
        }
    }
    for (int i = 0; i < ksize; i++)
    {
        for (int j = 0; j < ksize; j++)
        {
            window[i][j] /= sum;
        }
    }
}
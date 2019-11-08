#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include"opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

//    /Users/chenjiarui/Desktop/demo.avi
//    /Users/chenjiarui/Downloads/hush.mp4
const char* videoPath = "/Users/chenjiarui/Downloads/hush.mp4";
bool leftbuttom_downflag = false;
Point startPoint; //矩形框起点
Point endPoint; //矩形框终点
Mat image;
Mat imageCopy;
Mat objectImage;
int finish = 0; //用来标记是否框选取完成


//H、S通道
int channels[] = { 0, 1 };
int histSize[] = { 30, 32 };
float HRanges[] = { 0, 180 };
float SRanges[] = { 0, 256 };
const float *ranges[] = { HRanges, SRanges };


void drawHist(Mat rectImage);
void getObject();
void Mouse_areaSelection(int event, int x, int y, int flags, void *ustc);
void tracing(Mat srcHist);
double compHist(const MatND srcHist, Mat compareImage);


//画出直方图
void drawHist(const Mat rectImage)
{
    //图片数量nimages
    int nimages = 1;
    //通道数量,习惯用数组来表示
    int channels[3] = { 0,1,2 };
    //输出直方图
    cv::Mat outputHist_red, outputHist_green, outputHist_blue;
    //维数
    int dims = 1;
    //存放每个维度直方图尺寸（bin数量）的数组histSize
    int histSize[3] = { 256,256,256 };
    //每一维数值的取值范围ranges
    float hranges[2] = { 0, 255 };
    //值范围的指针
    const float *ranges[3] = { hranges,hranges,hranges };
    //是否均匀
    bool uni = true;
    //是否累积
    bool accum = false;

    //计算图像的直方图(红色通道部分)
    cv::calcHist(&rectImage, nimages, &channels[0], cv::Mat(), outputHist_red, dims, &histSize[0], &ranges[0], uni, accum);
    //计算图像的直方图(绿色通道部分)
    cv::calcHist(&rectImage, nimages, &channels[1], cv::Mat(), outputHist_green, dims, &histSize[1], &ranges[1], uni, accum);
    //计算图像的直方图(蓝色通道部分)
    cv::calcHist(&rectImage, nimages, &channels[2], cv::Mat(), outputHist_blue, dims, &histSize[2], &ranges[2], uni, accum);

    //遍历每个箱子(bin)检验，这里的是在控制台输出的。
    //for (int i = 0; i < 256; i++)
    //std::cout << "bin/value:" << i << "=" << outputHist_red.at<float>(i) << std::endl;

    //画出直方图
    int scale = 1;
    //直方图的图片,因为尺寸是一样大的,所以就以histSize[0]来表示全部了.
    cv::Mat histPic(histSize[0], histSize[0] * scale * 3, CV_8UC3, cv::Scalar(0, 0, 0));
    //找到最大值和最小值,索引从0到2分别是红,绿,蓝
    double maxValue[3] = { 0, 0, 0 };
    double minValue[3] = { 0, 0, 0 };
    cv::minMaxLoc(outputHist_red, &minValue[0], &maxValue[0], NULL, NULL);
    cv::minMaxLoc(outputHist_green, &minValue[1], &maxValue[1], NULL, NULL);
    cv::minMaxLoc(outputHist_blue, &minValue[2], &maxValue[2], NULL, NULL);
    //测试
    std::cout << minValue[0] << " " << minValue[1] << " " << minValue[2] << std::endl;
    std::cout << maxValue[0] << " " << maxValue[1] << " " << maxValue[2] << std::endl;

    //纵坐标缩放比例
    double rate_red = (histSize[0] / maxValue[0])*0.9;
    double rate_green = (histSize[0] / maxValue[1])*0.9;
    double rate_blue = (histSize[0] / maxValue[2])*0.9;

    for (int i = 0; i < histSize[0]; i++)
    {
        float value_red = outputHist_red.at<float>(i);
        float value_green = outputHist_green.at<float>(i);
        float value_blue = outputHist_blue.at<float>(i);
        //分别画出直线
        cv::line(histPic, cv::Point(i*scale, histSize[0]), cv::Point(i*scale, histSize[0] - value_red * rate_red), cv::Scalar(0, 0, 255));
        cv::line(histPic, cv::Point((i + 256)*scale, histSize[0]), cv::Point((i + 256)*scale, histSize[0] - value_green * rate_green), cv::Scalar(0, 255, 0));
        cv::line(histPic, cv::Point((i + 512)*scale, histSize[0]), cv::Point((i + 512)*scale, histSize[0] - value_blue * rate_blue), cv::Scalar(255, 0, 0));
    }
    cv::imshow("histgram", histPic);
}

void getObject(){
//    namedWindow("video",WINDOW_AUTOSIZE);
//    实例化了一个对象video
    VideoCapture cap(videoPath);
    if(!cap.isOpened()){
        cout<<"video not open.err"<<endl;
        return ;
    }
    //对于视频进行的操作
    double fps = cap.get(CAP_PROP_FPS);
    double pauseTime = 1000 / fps; //两帧画面中间间隔
    namedWindow("interImage");
    // 通过回调函数得到我所要框选的区域图像 objectImage
    setMouseCallback("interImage",Mouse_areaSelection);
    while(true){
        // 鼠标左键未按下 播放视频
        if(!leftbuttom_downflag)
        {
            cap>>image;
        }
        //图像为空或Esc键按下退出播放
        if (!image.data || waitKey(pauseTime) == 27)
        {
            break;
        }
        //两种情况下不在原始视频图像上刷新矩形
        //1. 起点等于终点
        //2. 左键按下且未抬起
        if(startPoint!= endPoint && !leftbuttom_downflag)
        {
            rectangle(image,startPoint,endPoint,Scalar(255,0,0),2);
        }
        imshow("interImage",image);
        if(finish==1)
        {
            destroyWindow("interImage");
            break;
        }
    }
    cap.release();
}

void tracing(const Mat srcHist)
{
    int width = abs(endPoint.x-startPoint.x);
    int height = abs(endPoint.y-startPoint.y);
    cout<<width<<" "<<height<<endl;

//    int X1 = startPoint.x - 2*width;
//    int X2 = startPoint.x + 2*width;
//    int Y1 = startPoint.y - 2*height;
//    int Y2 = startPoint.y + 2*height;
    int X1 = startPoint.x - width;
    int X2 = startPoint.x + width;
    int Y1 = startPoint.y - height;
    int Y2 = startPoint.y + height;
    //越界检查
    if (X1 < 0)
        X1 = 0;
    if (Y1 < 0)
        Y1 = 0;

    Point preStart;
    Point preEnd;

    Point get1(0, 0);
    Point get2(0, 0);


    VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        cout << "video not open.error!" << std::endl;
        return;
    }
    double fps = cap.get(CAP_PROP_FPS); //获取视频帧率
    double pauseTime = 1000 / fps; //两幅画面中间间隔

    // 参数解释
    // CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
    //CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
    int w = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    cv::Size S(w, h);
    namedWindow("targetTracing");

    VideoWriter write;
    write.open("/Users/chenjiarui/Downloads/hush.mp4",-1,fps,S,true);
    while(true)
    {
        cap>>image;
        if(!image.data||waitKey(pauseTime)==27)
        {
            break;
        }
        //使用直方图对于关注物体区域的图像特征进行统计的参数
        double comnum = 1.0;

        //初始化搜索区域
        for(int Cy = Y1;Cy <= Y2;Cy+=10)
        {
            for(preStart.x=X1,preStart.y=Cy;preStart.x<=X2;preStart.x+=10)
            {
                // 因为进行了边界控制，所以可能会出现向右移动时超界的情况
                if((preStart.x+width)<image.cols)
                    preEnd.x = preStart.x + width;
                else
                    preEnd.x = image.cols-1;
                if((preStart.y+height)<image.rows)
                    preEnd.y = preStart.y+height;
                else
                    preEnd.y = image.rows-1;


                //边界控制程序
                Mat compareImg;
                compareImg = image(Rect(preStart,preEnd));
                double c = compHist(srcHist,compareImg);
                if(comnum >c){
                    get1= preStart;
                    get2 = preEnd;
                    comnum = c;
                }
            }
        }
        if (comnum < 0.15) {
            X1 = get1.x - width;
            X2 = get1.x + width;
            Y1 = get1.y - height;
            Y2 = get1.y + height;

            if (X1 < 0)
                X1 = 0;
            if (Y1 < 0)
                Y1 = 0;
        }
        if(comnum<0.5)
            rectangle(image, get1, get2, Scalar(0, 0, 255), 2);
        //写入一帧
        write.write(image);
        imshow("targetTracing", image);
    }
    cap.release();
    write.release();
}

//鼠标回调函数逻辑实现
void Mouse_areaSelection(int event, int x, int y, int flags, void *ustc)
{
    // 如果用户按下鼠标左键的逻辑
    if(event == EVENT_LBUTTONDOWN){
        // 设置 bool 标记
        leftbuttom_downflag = true;
        //设置矩形框的开始的坐标（左上角）
        startPoint = Point(x,y);
//        endPoint = startPoint;
    }
    // 如果鼠标移动 并且鼠标左键还是按下的（即鼠标拖动中）
    if(event == EVENT_MOUSEMOVE && leftbuttom_downflag){
        imageCopy = image.clone();
        endPoint = Point(x,y);
        if(startPoint != endPoint){
            //在复制的图像上进行矩形绘制做标记
            rectangle(imageCopy,startPoint,endPoint,Scalar(255,0,0),2);
        }
        imshow("interImage",imageCopy);
    }
    if(event == EVENT_LBUTTONUP)
    {
        leftbuttom_downflag  = false;
        // 获取目标图像
        objectImage = image(Rect(startPoint, endPoint));
        finish = 1;
    }
}

double compHist(const MatND srcHist,Mat compareImage)
{
    //在比较直方图时，最佳操作是在HSV空间中操作，所以需要将BGR空间转换为HSV空间
    Mat compareHsvImage;
    cvtColor(compareImage, compareHsvImage, COLOR_BGR2Lab);
    //采用H-S直方图进行处理
    //首先得配置直方图的参数
    Mat  compHist;
    //进行原图直方图的计算

    //对需要比较的图进行直方图的计算
    calcHist(&compareHsvImage, 1, channels, Mat(), compHist, 2, histSize, ranges, true, false);
    //注意：这里需要对两个直方图进行归一化操作
    normalize(compHist, compHist, 0, 1, NORM_MINMAX);
    //对得到的直方图对比
    double g_dCompareRecult = compareHist(srcHist, compHist, 3);//3表示采用巴氏距离进行两个直方图的比较
    return g_dCompareRecult;
}
int main(){
    getObject();
    imshow("Target object",objectImage);
    cout << startPoint << endPoint << endl;
    drawHist(objectImage);
    Mat srcHsvImage;
    cvtColor(objectImage, srcHsvImage, COLOR_BGR2Lab);
    Mat srcHist;
    //进行原图直方图的计算
    calcHist(&srcHsvImage, 1, channels, Mat(), srcHist, 2, histSize, ranges, true, false);
    //归一化
    normalize(srcHist, srcHist, 0, 1, NORM_MINMAX);
    tracing(srcHist);
    waitKey(0);
    return 0;
}
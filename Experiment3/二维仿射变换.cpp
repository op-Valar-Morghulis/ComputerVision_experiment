#include<opencv2/core/saturate.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include<iostream>
#include<cmath>
using namespace std;
using namespace cv;
//仿射变换
void WarpAffine(const Mat &src,Mat &dst,double rot_mat[][3]){
    for(int x=0;x<dst.rows;x++){
        for(int y=0;y<dst.cols;y++){
            //计算在原图中的坐标  默认二维仿射变换使用2*3矩阵rot_mat
            double x_raw = x * rot_mat[0][0] + y * rot_mat[0][1] + rot_mat[0][2];
            double y_raw = x * rot_mat[1][0] + y * rot_mat[1][1] + rot_mat[1][2];
            //边缘处理

            if(x_raw<0||y_raw<0||x_raw>=src.rows||y_raw>=src.cols){
                //暗色填充
                for (int c = 0; c < 3; c++) {
                    dst.ptr<Vec3b>(x)[y][c] = saturate_cast<uchar>(0);
                }
            }
            else{
                for(int c=0;c<3;c++){
                    int Q11_x = (int)x_raw;
                    int Q11_y = (int)y_raw;
                    //对于边界处理  最外界的边缘点不存在四个点围绕 所以直接填充
                    if(Q11_x==(src.rows-1)||Q11_y==(src.cols-1)||Q11_x==0||Q11_y==0){
                        dst.ptr<Vec3b>(x)[y][c]=saturate_cast<uchar>(src.ptr<Vec3b>(Q11_x)[Q11_y][c]);
                    }
                    //双线性插值进行重采样
                    else{
                        int Q11 = src.ptr<Vec3b>(Q11_x)[Q11_y][c];
                        int Q12 = src.ptr<Vec3b>(Q11_x)[Q11_y+1][c];
                        int Q21 = src.ptr<Vec3b>(Q11_x+1)[Q11_y][c];
                        int Q22 = src.ptr<Vec3b>(Q11_x+1)[Q11_y+1][c];

                        double dx = x_raw-(double)Q11_x;
                        double dy = y_raw-(double)Q11_y;
                        double R1 = Q11+dx*(Q21-Q11);
                        double R2 = Q12+dx*(Q22-Q12);
                        dst.ptr<Vec3b>(x)[y][c]=saturate_cast<uchar>(R1+dy*(R2-R1));
                    }

                }
            }
        }
    }


}

void WarpAffine_self(const Mat &src,Mat &dst,Mat rot_mat){
    for(int x=0;x<dst.rows;x++){
        for(int y=0;y<dst.cols;y++){
            double *data0 = rot_mat.ptr<double>(0);
            double *data1 = rot_mat.ptr<double>(1);
            //计算在原图中的坐标  默认二维仿射变换使用2*3矩阵rot_mat
            double x_raw = x * data0[0] + y * data0[1] + data0[2];
            double y_raw = x * data1[0] + y * data1[1] + data1[2];
            //边缘处理
            if(x_raw<0||y_raw<0||x_raw>=src.rows||y_raw>=src.cols){
                //暗色填充
                for (int c = 0; c < 3; c++) {
                    dst.ptr<Vec3b>(x)[y][c] = saturate_cast<uchar>(0);
                }
            }
            else{
                for(int c=0;c<3;c++){
                    int Q11_x = (int)x_raw;
                    int Q11_y = (int)y_raw;
                    //对于边界处理  最外界的边缘点不存在四个点围绕 所以直接填充
                    if(Q11_x==(src.rows-1)||Q11_y==(src.cols-1)||Q11_x==0||Q11_y==0){
                        dst.ptr<Vec3b>(x)[y][c]=saturate_cast<uchar>(src.ptr<Vec3b>(Q11_x)[Q11_y][c]);
                    }
                        //双线性插值进行重采样
                    else{
                        int Q11 = src.ptr<Vec3b>(Q11_x)[Q11_y][c];
                        int Q12 = src.ptr<Vec3b>(Q11_x)[Q11_y+1][c];
                        int Q21 = src.ptr<Vec3b>(Q11_x+1)[Q11_y][c];
                        int Q22 = src.ptr<Vec3b>(Q11_x+1)[Q11_y+1][c];

                        double dx = x_raw-(double)Q11_x;
                        double dy = y_raw-(double)Q11_y;
                        double R1 = Q11+dx*(Q21-Q11);
                        double R2 = Q12+dx*(Q22-Q12);
                        dst.ptr<Vec3b>(x)[y][c]=saturate_cast<uchar>(R1+dy*(R2-R1));
                    }

                }
            }
        }
    }


}

//仿射变化矩阵求逆
void getRotationMatrix(double rot_mat[2][3],int x,int y,double degree){
    //计算旋转角的弧度值
    double Angle = degree * CV_PI / 180;
    double s = sin(Angle);
    double c = cos(Angle);
    double t1 = (-x) * c + y * s + x;
    double t2 = (-x) * s - y * c + y;
    rot_mat[0][0]=rot_mat[1][1] = c;
    rot_mat[0][1] = s;
    rot_mat[1][0] = -s;
    rot_mat[0][2] = -s * t2 - c * t1;
    rot_mat[1][2] = -c * t2 + s * t1;
}

// 参数： 原图像  旋转中心 （x坐标 y坐标） 旋转角度
Mat Rotate(const Mat &src,int x, int y, double degree){
    double rot_mat[2][3] = {0};
    // 处理输入异常
    if (x < 0 || y < 0 || x >= src.rows || y >= src.cols) {
        cout << "please input valid rotating center!" << endl;
        exit(1);
    }
    Mat Rot_Image = Mat::zeros(src.rows, src.cols, src.type());
    getRotationMatrix(rot_mat,x,y,degree);
    WarpAffine(src, Rot_Image,rot_mat);
    return Rot_Image;
}

Mat ImageDeformation(const Mat &src) {
    Mat imgAffine = Mat::zeros(src.rows, src.cols, src.type());
    int row = imgAffine.rows, col = imgAffine.cols;
    for (int x = 0; x < row; x++) {
        for (int y = 0; y < col; y++) {

            double X = x / ((row - 1) / 2.0) - 1.0;
            double Y = y / ((col - 1) / 2.0) - 1.0;
            double r = sqrt(X * X + Y * Y);

            if (r >= 1) {
                imgAffine.at<Vec3b>(x, y)[0] = saturate_cast<uchar>(src.at<Vec3b>(x, y)[0]);
                imgAffine.at<Vec3b>(x, y)[1] = saturate_cast<uchar>(src.at<Vec3b>(x, y)[1]);
                imgAffine.at<Vec3b>(x, y)[2] = saturate_cast<uchar>(src.at<Vec3b>(x, y)[2]);
            }
            else {

                double theta = 1.0 + X * X + Y * Y - 2.0*sqrt(X * X + Y * Y);//修改不用（1-r)*(1-r)
                double x_ = cos(theta)*X - sin(theta)*Y;
                double y_ = sin(theta)*X + cos(theta)*Y;

                x_ = (x_ + 1.0)*((row - 1) / 2.0);
                y_ = (y_ + 1.0)*((col - 1) / 2.0);


                if (x_ < 0 || y_ < 0||x_>=src.rows||y_>=src.cols) {
                    for (int c = 0; c < 3; c++) {
                        imgAffine.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(0);
                    }
                }
                else {
                    //左上角坐标（X1，Y1)
                    //计算双线性插值
                    int X1 = (int)x_;
                    int Y1 = (int)y_;

                    for (int c = 0; c < 3; c++) {
                        if (X1 == (src.rows - 1) || Y1 == (src.cols - 1)) {
                            imgAffine.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(src.at<Vec3b>(X1, Y1)[c]);
                        }
                        else {
                            //四个顶点像素值
                            //注意访问越界
                            int aa = src.at<Vec3b>(X1, Y1)[c];
                            int bb = src.at<Vec3b>(X1, Y1 + 1)[c];
                            int cc = src.at<Vec3b>(X1 + 1, Y1)[c];
                            int dd = src.at<Vec3b>(X1 + 1, Y1 + 1)[c];

                            double dx = x_ - (double)X1;
                            double dy = y_ - (double)Y1;
                            double h1 = aa + dx * (bb - aa);
                            double h2 = cc + dx * (dd - cc);
                            imgAffine.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(h1 + dy * (h2 - h1));
                        }
                    }
                }
            }
        }
    }
    return imgAffine;
}

int main(){
    Mat srcImage,Rot_Image,Deformation_Image,Affine_Image;
    //默认存在一个仿射变换矩阵
    double rot_mat[3][3],inverse_mat[3][3];
    rot_mat[2][0]=rot_mat[2][1]=0;
    rot_mat[2][2]=1;
    //加载图像
//    /Users/chenjiarui/Desktop/orl_faces/s1/1.pgm
//    /Users/chenjiarui/Desktop/input.png
    srcImage = imread("/Users/chenjiarui/Desktop/input.png",1);
    
    if(srcImage.empty())
    {
        cout << "图像加载失败!" << endl;
        return -1;
    }
    else
        cout << "图像加载成功!" << endl;
//    cout<<"选择是否进行rotate旋转变换，否则请自行输入仿射变换矩阵"<<endl;
    cout<<"输入2，进行图像的变形"<<endl;
    cout<<"进行Rotate变换请输入1，否则输入0，自行输入仿射变换矩阵"<<endl;
    int choice = 0;
    cin>>choice;
    if(choice==1){
        int x=0, y=0;
        cout<<"请输入旋转中心坐标"<<"x=";
        cin>>x;
        cout<<"y=";
        cin>>y;
        cout<<"请输入旋转角度=";
        double degree=45.0;
        cin>>degree;
        namedWindow("Rotate",WINDOW_AUTOSIZE);
        Rot_Image = Rotate(srcImage,x,y,degree);
        imshow("Rotate",Rot_Image);
        waitKey(0);
    }
    else if(choice==0){
        cout << "请输入一个2*3的仿射变换矩阵" << endl;
        cin>>rot_mat[0][0]>>rot_mat[0][1]>>rot_mat[0][2]>>rot_mat[1][0]>>rot_mat[1][1]>>rot_mat[1][2];
        Mat input,inverse_mat;
        input.create(3,3,CV_64F);
        inverse_mat.create(3,3,CV_64F);
        input = (Mat_<double>(3,3)<<rot_mat[0][0],rot_mat[0][1],rot_mat[0][2],rot_mat[1][0],rot_mat[1][1],rot_mat[1][2],rot_mat[2][0],rot_mat[2][1],rot_mat[2][2]);
        invert(input,inverse_mat,DECOMP_LU);
        Mat Affine_Image = Mat::zeros(srcImage.rows,srcImage.cols,srcImage.type());
        namedWindow("Affine",WINDOW_AUTOSIZE);
        WarpAffine_self(srcImage, Affine_Image,inverse_mat);
        imshow("Affine",Affine_Image);
        waitKey(0);
    }
    // 进行图像变形操作
    else if(choice ==2)
    {
        namedWindow("Deformation",WINDOW_AUTOSIZE);
        Deformation_Image = ImageDeformation(srcImage);
        imshow("Deformation",Deformation_Image);
        waitKey(0);
    }
    return 0;
}

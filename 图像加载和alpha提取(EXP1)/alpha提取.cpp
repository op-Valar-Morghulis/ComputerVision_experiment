#include<iostream>
#include <vector>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;



int main()
{
      Mat img_1 = imread("/Users/chenjiarui/Desktop/a.png",IMREAD_UNCHANGED);   //读取原图
      vector<Mat> merged;
      split(img_1,merged);
      imshow("alpha",merged.at(3));
      waitKey(0);
      return 0;
}



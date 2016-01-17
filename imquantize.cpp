/* To do:
 * Documentation
 * Values
 * */
#include<vector>
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int imquantize(InputArray _I,OutputArray _dst,InputArray _levels)
{
    Mat I=_I.getMat();
    Mat levels=_levels.getMat();
         
    _dst.create(I.size(),CV_8U);
    Mat dst=_dst.getMat();
    
    int n;
    if(levels.rows==1)
        n=levels.cols;
    else if(levels.cols==1)
        n=levels.rows;
    else
    {
        cerr<<"Levels should be either row or column matrix or a vector.";
        return -1;
    } 

    dst=Mat::zeros(I.size(),CV_8U);
    Mat term=Mat::zeros(I.size(),CV_8U);
    for(int i=0;i<n;i++)
    {
        compare(I,levels.at<uint>(i),term,CMP_GT);
        term/=255;
        dst+=term;
    }

    return 0;
}

int imquantize(InputArray _I,OutputArray _dst,InputArray _levels,InputArray _values);

int main()
{
    Mat I=imread("im0.png",0);
    Mat levels=Mat::ones(1,2,CV_8U);
    //vector<int> levels;
    levels.at<uint>(0)=78;
    levels.at<uint>(1)=143;

    Mat ans;
    imquantize(I,ans,levels);
    ans=ans*255/2;

    
    imshow("Quantized",ans);
    waitKey(0);
    destroyWindow("Quantized");
}

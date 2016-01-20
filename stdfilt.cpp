/* stdfilt - Local standard deviation of an image.
 * Written by Varun Bhatt
 *
 * See comments before stdfilt function for details about the function.
 *
 * -------------------------------------------
 * Example:
 * 
 * Mat I=imread("im0.png");
 * Mat ans;
 * stdfilt(I,ans);
 * imshow("Original image",I);
 * imshow("Local standard deviation",ans);
 * -------------------------------------------
 *
 * References:
 * 1. http://in.mathworks.com/help/images/ref/stdfilt.html - MATLAB stdfilt documentation
 * 2. Source code of stdfilt in MATLAB (stdfilt.m)
 * 3. http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#filter2d - filter2D documentaion
 * */

#include<iostream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

//Algorithm for stdfilt. Input validation is done in stdfilt function.
//See comments before stdfilt function for details. 
void algstdfilt(InputArray _I, OutputArray _dst, InputArray _h=Mat::ones(3,3,CV_64FC1))
{
    Mat I=_I.getMat();
    Mat h=_h.getMat();
    _dst.create(I.size(),I.type());
    Mat dst=_dst.getMat();
    
    int n=sum(h).val[0];
    int n1=n-1;

    //If n is 1, then matrix with all zeros is returned. Otherwise standard deviation is calculated.
    //Standard deviation is calculated using the theoritical definition.
    //Correlation in filter2D is used to perform calculations.
    if(n!=1)
    {
        Mat conv1,conv2;
        filter2D(I.mul(I),conv1,-1,h/n1,Point(-1,-1),0,BORDER_REFLECT);
        filter2D(I,conv2,-1,h,Point(-1,-1),0,BORDER_REFLECT);
        conv2=conv2.mul(conv2)/(n*n1);
        sqrt(max((conv1-conv2),0),dst);
    }
    else
        dst=Mat::zeros(I.size(),I.type());
}

//Main stdfilt function. It returns an image "dst" of same size and type as the input image "I"
//where each output pixel contains the standard deviation value of its neighborhood.
//
//The neighborhood "h" can be specified in the form of a matrix of zeros and ones where ones
//denote neighbors. The size of h should be odd in all dimensions. 
//Default is a 3-by-3 neighborhood.
//
//For pixels in the border of I, symmetric padding is used (padding is reflection of 
//border pixels)
//
//Inf and NaN are not handled by stdfilt.
//
//Example usage shown in the main function.

//I - Input Image
//dst - Output Image
//h - Matrix specifying the neighborhood. Default is a 3x3 neighborhood. 
//Returns 0 on successful completion, -1 when there are errors.
int stdfilt(InputArray _I, OutputArray _dst, InputArray _h=Mat::ones(3,3,CV_64FC1))
{ 
    Mat I=_I.getMat();
    Mat h=_h.getMat();

    //h should be a single channel matrix.
    if(h.channels()!=1)
    {
        cerr<<"Invalid neighborhood type\n";
        return -1;
    }

    //h should contain only 0 or 1.
    for(int i=0;i<h.rows;i++)
        for(int j=0;j<h.cols;j++)
            if(h.at<double>(i,j)!=0 && h.at<double>(i,j)!=1)
            {
                cerr<<"Invalid neighborhood value\n";
                return -1;
            }

    //h's size must be odd in all dimensions.
    for(int i=0;i<h.dims;i++)
        if(h.size[i]%2==0)
        {
            cerr<<"Invalid neighborhood size\n";
            return -1;
        }

    //Convert h, I to double format.
    h.convertTo(h,CV_64FC1);
    if(I.type()!=CV_64F && I.type()!=CV_32F)
    {
        I.convertTo(I,CV_64F);
        I/=255;
    }

    algstdfilt(I,_dst,h);
    return 0;
}

int main()
{
    //Read an image
    Mat I=imread("im0.png");

    //Apply stdfilt
    Mat ans;
    stdfilt(I,ans);   

    //Show output
    imshow("Original image",I);
    imshow("Local standard deviation",ans);
    waitKey(0);
    destroyWindow("Original image");
    destroyWindow("Local standard deviation");
}

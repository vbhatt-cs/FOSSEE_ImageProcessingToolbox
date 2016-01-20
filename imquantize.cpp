/* imquantize - Quantize image based on levels
 * Written by Varun Bhatt
 *
 * imquantize takes an input image, levels based on which quantization takes place
 * and optional values to set to each part. See comments before the imquantize function
 * for details.
 *
 * ----------------------------------
 * Example:
 *
 * Mat I=imread("im0.png",0);
 *
 * Mat levels(2,1,CV_8U);
 * levels.at<uchar>(0)=78;
 * levels.at<uchar>(1)=143;
 *
 * Mat values(3,1,CV_8U);
 * values.at<uchar>(0)=4;
 * values.at<uchar>(1)=9;
 * values.at<uchar>(2)=25;
 * 
 * Mat ans1,ans2;
 * imquantize(I,ans1,levels);
 * imquantize(I,ans2,levels,values);
 * ----------------------------------
 *
 * References:
 * 1. http://in.mathworks.com/help/images/ref/imquantize.html - MATLAB imquantize documentation
 * 2. imquantize source code in MATLAB (imquantize.m)
 * */

#include<vector>
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

//The algorithm used by imquantize function. Input validation is done in the imquantize function.
//See comments before imquantize for details.
int algquantize(InputArray _I,OutputArray _dst,InputArray _levels,InputArray _values)
{
    Mat I=_I.getMat();
    Mat levels=_levels.getMat();
    Mat values=_values.getMat();

    int channelsI=I.channels();
    int valuesType=values.type();

    //Convert everything temporarily to double for proper comparison.
    I.convertTo(I,CV_64FC(channelsI));
    levels.convertTo(levels,CV_64FC1);
    values.convertTo(values,CV_64FC1);

    _dst.create(I.size(),values.type());
    Mat dst=_dst.getMat();
 
    int n=levels.cols;

    dst.setTo(values.at<double>(0));
    Mat term=Mat::zeros(I.size(),I.type());
    for(int i=0;i<n;i++)
    {
        compare(I,levels.at<double>(i),term,CMP_GT);
        
        //Make sure term is 1 for true and 0 for false.
        double min,max;
        minMaxLoc(term,&min,&max);
        term/=max;

        //result of compare is of the same type as "I" but "dst" should be of the same type as "values".
        term.convertTo(term,values.type());

        //the term is added in such a way that at every higher level, the previous value is removed 
        //and the new value is added. 
        term*=(values.at<double>(i+1)-values.at<double>(i));
        dst+=term;
    }

    //Convert "dst" back to the original type of "values"
    dst.convertTo(dst,valuesType);
    return 0;
}

//imquantize uses the levels specified in "levels" to quantize the input image "I" and put
//the output into "dst". The values in the output are assigned as
//if A(i)<=levels(0) then dst(i)=0
//if levels(m-1)<A(i)<=levels(m) then dst(i)=m
//if A(i)>levels(n-1) then dst(i)=n
//
//Levels need to be in increasing order. If it is not, then it is sorted before applying the algorithm.
//
//_I - Input image
//_dst - Output. Type is int.
//_levels - Levels based on which quantization should happen.
int imquantize(InputArray _I,OutputArray _dst,InputArray _levels)
{
    Mat I=_I.getMat();
    Mat levels=_levels.getMat();
    
    if(I.depth()!=levels.depth())
    {
        cerr<<"Levels and input image should be of the same type.";
        return -1;
    }

    int n;
    if(levels.rows==1)
    {
        n=levels.cols;
        cv::sort(levels,levels,CV_SORT_ASCENDING|CV_SORT_EVERY_ROW);
    }

    else if(levels.cols==1)
    {
        n=levels.rows;
        cv::sort(levels,levels,CV_SORT_ASCENDING|CV_SORT_EVERY_COLUMN);
        levels=levels.t();
    }

    else
    {
        cerr<<"Levels should be either row or column matrix or a vector.";
        return -1;
    } 

    Mat values(1,n+1,CV_32S);
    for(int i=0;i<=n;i++)
        values.at<int>(i)=i;

    values.convertTo(values,levels.type());
    return algquantize(I,_dst,levels,values);
}

//imquantize uses the levels specified in "levels" to quantize the input image "I" and put
//the output into "dst". The values in the output are assigned as
//if A(i)<=levels(0) then dst(i)=values(0)
//if levels(m-1)<A(i)<=levels(m) then dst(i)=values(m)
//if A(i)>levels(n-1) then dst(i)=values(n)
//
//Levels need to be in increasing order. If it is not, then it is sorted before applying the algorithm.
//Size of "values" should be one more than size of levels.
//
//_I - Input image
//_dst - Output. Type is same as that of "values"
//_levels - Levels based on which quantization should happen.
//_values - Values used to populate the output.
int imquantize(InputArray _I,OutputArray _dst,InputArray _levels,InputArray _values)
{
    Mat I=_I.getMat();
    Mat levels=_levels.getMat();
    Mat values=_values.getMat();
         
    if(I.type()!=levels.type())
    {
        cerr<<"Levels and input image should be of the same type.";
        return -1;
    }

    int n,n1;
    if(levels.rows==1)
    {
        n=levels.cols;
        cv::sort(levels,levels,CV_SORT_ASCENDING|CV_SORT_EVERY_ROW);
    }

    else if(levels.cols==1)
    {
        n=levels.rows;
        cv::sort(levels,levels,CV_SORT_ASCENDING|CV_SORT_EVERY_COLUMN);
        levels=levels.t();
    }

    else
    {
        cerr<<"Levels should be either row or column matrix or a vector.";
        return -1;
    }
    
    if(values.rows==1)
        n1=values.cols;
    else if(values.cols==1)
    {
        n1=values.rows;
        values=values.t();
    }

    else
    {
        cerr<<"Values should be either row or column matrix or a vector.";
        return -1;
    } 

    if(n1!=(n+1))
    {
        cerr<<"Values and levels size mismatch.";
        return -1;
    }
    
    return algquantize(I,_dst,levels,values); 
}

int main()
{
    //Read the image
    Mat I=imread("im0.png",0);

    //Specify levels
    Mat levels(2,1,CV_8U);
    levels.at<uchar>(0)=78;
    levels.at<uchar>(1)=143;

    //Specify values
    Mat values(3,1,CV_8U);
    values.at<uchar>(0)=4;
    values.at<uchar>(1)=9;
    values.at<uchar>(2)=25;

    Mat ans1,ans2;
    double min1,max1,min2,max2;

    //Default values used
    imquantize(I,ans1,levels);
    minMaxLoc(ans1,&min1,&max1);
    ans1-=min1;
    ans1.convertTo(ans1,ans1.type(),1/(max1-min1));

    //Custom values used
    imquantize(I,ans2,levels,values); 
    minMaxLoc(ans2,&min2,&max2);
    ans2-=min2;
    ans2.convertTo(ans2,ans2.type(),1/(max2-min2));

    //Display images
    imshow("Quantized",ans1);
    imshow("Quantized with custom values",ans2);
    waitKey(0);
    destroyWindow("Quantized");
    destroyWindow("Quantized with custom values");
}

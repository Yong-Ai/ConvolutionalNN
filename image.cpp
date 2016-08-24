#include <stdio.h>
#include "stdafx.h"
#include "image.h"
#include <cv.h>
#include <highgui.h>


void Make_first(IplImage *src_image, Img *image)
{

	int i, j;
	image->height = src_image->height;
	image->width = src_image->width;
	uchar *data;
	data = (uchar*)src_image->imageData;

	for(i = 0; i < image->height; i++)
	{
		for(j = 0; j < image->width; j++)
		{
			image->data[i][j] = data[i * src_image->width + j] - 127.5;
			image->data[i][j] /= 127.5;
		}
	}

}
void Test_showimage(Img image, int index)
{
	int i, j;
	IplImage *view;
	view = cvCreateImage(cvSize(image.width, image.height), IPL_DEPTH_8U, 1);

	for(i = 0; i < image.height; i++)
	{
		for(j = 0; j < image.width; j++)
		{
			cvSetReal2D(view, i, j, image.data[i][j]*127.5+127.5);
		}
	}

	cvShowImage("view", view);
	char file_name[30];
	sprintf(file_name, "%d.jpg", index);
	//cvSaveImage(file_name, view);

	cvWaitKey();
	cvDestroyAllWindows();
	cvReleaseImage(&view);
}


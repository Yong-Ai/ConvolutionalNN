#include <cv.h>


typedef struct Weight{
	double **filter;
	double beta;
	int height, width;
	
}Weight;

typedef struct Img{
	double **data;
	double **delta;
	int height, width;
	double bias;
	double bias_delta;
	double **fn;
}Img;

void Make_first(IplImage *src_image, Img *image);
void Test_showimage(Img image, int index);
void Test_showimage_sig(Img image, int index);






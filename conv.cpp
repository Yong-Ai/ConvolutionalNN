#include "stdafx.h"
#include "image.h"
#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define TARGET_SIZE 60000	//타겟갯수는 50개
#define CARTEGORY 10	//카테고리는 5개
#define NUM_EPOCH 200	//epoch


char input[40] = ".\\data\\1.bmp";
char output[40] = ".\\result\\3.txt";
char parameters[40] = ".\\parameter.txt";
char target_file[40] = ".target.txt";

FILE *fp_log;

Img **arch;
int layers[7] = {1,8,8,24,24,100,10};
//int filter_layers[6] = { 12, 8, 96, 24, 2400, 500};
int filter_layers[6] = { 16, 8, 192, 24, 2400, 500};
int filter_size[6] = {5,4,6,3,6,1};
Weight ***weights;
int target[TARGET_SIZE][CARTEGORY];
int resultTarget[TARGET_SIZE];
double learning_gain = 0.02;

void Img_alloc(Img* arch);
void Convolution(int i, int j);
void Subsampling(int i, int j);
void MLP(int i, int j);
void Cal_delta(Img *arch_before, Img *arch_after, Weight *weight, int mode);
void Update_weight(Img *arch_before, Img *arch_after, Weight *weight, int mode);
void init_delta();
void Save_Weight();
void Save_Weight(int epoch, double energy);
void NewSaveWeight(int epoch, double energy);

int main()
{
	IplImage *src;		//처음 이미지를 저장할 IplImage
	FILE *fp_target;
	int temp_target;
	int count_for_filter_layer = 0;
	int epoch = NUM_EPOCH;
	double energy = 0;

	double temp = -1;
	int Label = -1;

	fp_log = fopen("log.txt", "w");
	srand((unsigned)time(NULL));
	fp_target = fopen("target.txt", "r");
	for(int i = 0; i < TARGET_SIZE; i++)
	{
		fscanf(fp_target, "%d", &temp_target);
		for(int j = 0; j < CARTEGORY; j++)
		{
			resultTarget[i] = temp_target;
			if(j == temp_target)
			{
				target[i][j] = 1;
			}else
			{
				target[i][j] = -1;
			}
		}
	}//타겟 만들기


	src = cvLoadImage(input, 0);
	//이미지 정의를 위한 데이터 불러오기


	arch = (Img**)calloc(7, sizeof(Img*));	//구조생성 7개의 가로배열
	for(int i = 0; i < 7; i++)		
	{
		arch[i] = (Img*)calloc(layers[i], sizeof(Img));		//7개의 가로배열에 세로배열 생성
	}

	for(int i = 0; i < layers[0]; i++)
	{
		arch[0][i].height = src->height;
		arch[0][i].width = src->width;
		Img_alloc(&arch[0][i]);//동적할당 머신이 있다. 반환은 없고 arch의 주소를 가져감.
		//동적할당머신 제작, 얘는 입력된 높이와 넓이값을 가지고 자기 자신을 동적할당함
	}

	for(int i = 1; i < 7; i++)
	{
		if(i % 2 == 0)
		{
			for(int j = 0; j < layers[i]; j++)
			{
				arch[i][j].height = arch[i-1][0].height / filter_size[count_for_filter_layer];
				arch[i][j].width = arch[i-1][0].width / filter_size[count_for_filter_layer];
				Img_alloc(&arch[i][j]);
			}
			count_for_filter_layer++;
		}
		else
		{
			for(int j = 0; j < layers[i]; j++)
			{
				arch[i][j].height = arch[i-1][0].height-filter_size[count_for_filter_layer]+1;
				arch[i][j].width = arch[i-1][0].width-filter_size[count_for_filter_layer]+1;
				Img_alloc(&arch[i][j]);
			}
			count_for_filter_layer++;
		}
	}//Lenet구조 동적할당.



	weights = (Weight***)calloc(6, sizeof(Weight**));//0번쨰가 첫 레이어
	for(int i = 0; i < 5; i++)
	{
		if(i % 2 == 0)
		{
			weights[i] = (Weight**)calloc(layers[i+1], sizeof(Weight*));			
			for(int j = 0; j < layers[i+1]; j++)
			{
				weights[i][j] = (Weight*)calloc(layers[i], sizeof(Weight));
			}
		}else
		{
			weights[i] = (Weight**)calloc(layers[i+1], sizeof(Weight*));
			for(int j = 0; j < layers[i+1]; j++)
			{
				weights[i][j] = (Weight*)calloc(1, sizeof(Weight));
			}
		}
	}//C레이어와 S레이어
	
	weights[5] = (Weight**)calloc(layers[6], sizeof(Weight*));
	for(int i = 0; i < layers[6]; i++)
	{
		weights[5][i] = (Weight*)calloc(layers[5], sizeof(Weight));		
	}
	//MLP레이어
	//웨이트 구조 생성
	for(int i = 0; i < 5; i++)
	{
		if(i % 2 == 0)
		{
			for(int j = 0; j < layers[i+1]; j++)
			{
				for(int k = 0; k < layers[i]; k++)
				{
					weights[i][j][k].filter = (double**)calloc(filter_size[i], sizeof(double*));
					for(int l = 0; l < filter_size[i]; l++)
					{
						weights[i][j][k].filter[l] = (double*)calloc(filter_size[i], sizeof(double));
						for(int m = 0; m < filter_size[i]; m++)
						{
							weights[i][j][k].filter[l][m] =  ((rand() % 201) * 0.01) -1;
							weights[i][j][k].height = filter_size[i];
							weights[i][j][k].width = filter_size[i];
						}
					}
				}
			}

		}else
		{
			for(int j = 0; j < layers[i+1]; j++)
			{
				weights[i][j][0].beta = ((rand() % 201) * 0.01) -1;
				weights[i][j][0].height = filter_size[i];
				weights[i][j][0].width = filter_size[i];
			}
		}
	}

	for(int i = 0; i < layers[6]; i++)
	{
		for(int j = 0; j < layers[5]; j++)
		{
			weights[5][i][j].beta =  ((rand() % 201) * 0.01) -1;
			weights[5][i][j].height = 1;
			weights[5][i][j].width = 1;
		}
		
	}


	//웨이트 내부 구조 생성 끝

//	printf("d");

 	int file_index = 1;
	int output_index = 0;


	//여기서부터 학습
	while(epoch--)//이포크 
	{
		energy = 0;
		file_index = 1;
		for(int it = 0; it < TARGET_SIZE; it++)//이터레이션
		{
		
			init_delta();

			for(int i = 0; i < 7; i++)
			{
				if(i == 0)
				{
					for(int j = 0; j < layers[i]; j++)
					{
						sprintf(input, ".\\data\\%d.bmp", file_index++);
						src = cvLoadImage(input, 0);
						Make_first(src,&arch[0][j]);
					}
				}else if(i % 2 == 1)
				{
					for(int j = 0; j < layers[i]; j++)
					{
						Convolution(i,j);
					}
				}else if(i == 6)
				{
					for(int j = 0; j < layers[i]; j++)
					{
						MLP(i, j);
					}
					
				}else if(i % 2 == 0)
				{
					for(int j = 0; j < layers[i]; j++)
					{
						Subsampling(i,j);

					}
				}
			}

			//마지막 6 레이어 델타 구하기
			for(int i = 0; i < layers[6]; i++)
			{
				arch[6][i].delta[0][0] = arch[6][i].fn[0][0] * (-arch[6][i].data[0][0] + target[it][i]);
				energy += (((target[it][i] - arch[6][i].data[0][0]) * (target[it][i] - arch[6][i].data[0][0])) / 2.0)/ (double)CARTEGORY;
			}
			for(int i = 0; i < layers[5]; i++)//100
			{
				for(int j = 0; j < layers[6]; j++)//5
				{
					arch[5][i].delta[0][0] += arch[6][j].delta[0][0] * arch[5][i].fn[0][0] * weights[5][j][i].beta;
				}
			}
			
		//	//웨이트 업데이트
			for(int i = 0; i < layers[6]; i++)	//5
			{
				for(int j = 0; j < layers[5]; j++)	//100
				{
					weights[5][i][j].beta += arch[6][i].delta[0][0] * arch[5][j].data[0][0] * learning_gain;
				}
			}

			for(int i = 0; i < layers[4]; i++)//24	델타를 구해보자
			{
				for(int j = 0; j < layers[5]; j++)//100
				{
					Cal_delta(&arch[4][i], &arch[5][j], &weights[4][j][i], 1);
				}
			}

			for(int i = 0; i < layers[3]; i++)//24	델타를 구해보자
			{
				Cal_delta(&arch[3][i], &arch[4][i], &weights[3][i][0], 2);
			}


			for(int i = 0; i < layers[2]; i++)//8	델타를 구해보자
			{
				for(int j = 0; j < layers[3]; j++)//24
				{
					Cal_delta(&arch[2][i], &arch[3][j], &weights[2][j][i], 1);
				}
			}

			for(int i = 0; i < layers[1]; i++)//24	델타를 구해보자
			{
				Cal_delta(&arch[1][i], &arch[2][i], &weights[1][i][0], 2);
			}

			for(int i = 0; i < layers[0]; i++)//2
			{
				for(int j = 0; j < layers[1]; j++)//8
				{
					Update_weight(&arch[0][i], &arch[1][j], &weights[0][j][i], 1);
				}
			}

			for(int i = 0; i < layers[1]; i++)//8
			{
				Update_weight(&arch[1][i], &arch[2][i], &weights[1][i][0], 2);
			}
			
			for(int i = 0; i < layers[2]; i++)//8
			{
				for(int j = 0; j < layers[3]; j++)//24
				{
					Update_weight(&arch[2][i], &arch[3][j], &weights[2][j][i], 1);
				}
			}

			for(int i = 0; i < layers[3]; i++)//24
			{
				Update_weight(&arch[3][i], &arch[4][i], &weights[3][i][0], 2);
			}

			for(int i = 0; i < layers[4]; i++)//24
			{
				for(int j = 0; j < layers[5]; j++)//100
				{
					Update_weight(&arch[4][i], &arch[5][j], &weights[4][j][i], 1);
				}
			}
			cvReleaseImage(&src);			
		}
		
		energy /= TARGET_SIZE;
		printf("Error rate = %lf\n", energy);
		printf("%d epoch 완료\n", NUM_EPOCH -epoch);
		Save_Weight(NUM_EPOCH -epoch, energy);
		NewSaveWeight(NUM_EPOCH -epoch, energy);
	}


	Save_Weight();
	int trueCount = 0;
	float tempCount;
	
	for(int it = 0; it< TARGET_SIZE; it++)
	{
		for(int i = 0; i < 7; i++)
		{
			if(i == 0)
			{
				for(int j = 0; j < layers[i]; j++)
				{
					sprintf(input, ".\\data\\%d.bmp", it+1);
					src = cvLoadImage(input, 0);
					Make_first(src,&arch[0][j]);
						
				}
			}else if(i % 2 == 1)
			{
				for(int j = 0; j < layers[i]; j++)
				{
					Convolution(i,j);						
				}
			}else if(i == 6)
			{
				temp = -1;
				Label = -1;
				for(int j = 0; j < layers[i]; j++)
				{
					MLP(i, j);
					//printf("output = %lf\n", arch[6][j].data[0][0]);
					if ( temp < arch[6][j].data[0][0])
					{
						temp = arch[6][j].data[0][0];
						Label = j;
					}
				}
				//printf("result = %d\n", Label);
				if( Label == resultTarget[file_index])
				{
					//printf("True\n");
					trueCount++;
				}
				//else
					//printf("false\n");
					
			}else if(i % 2 == 0)
			{
				for(int j = 0; j < layers[i]; j++)
				{
					Subsampling(i,j);						
				}
			}
		}

		
	}
	tempCount = (float)trueCount;
	tempCount /= TARGET_SIZE;
	printf(" True Count Number = %d\n", trueCount);
	printf(" Percentage = %f\n", tempCount*100);


	//filter 해제 아직 안함
	//동적할당 해제
	for(int j = 0; j < layers[0]; j++)
	{
		cvReleaseImage(&src);
	}

	for(int i = 0; i < 7; i++)
	{
		for(int j = 0; j < layers[i]; j++)
		{
			for(int k = 0; k < arch[i][j].height; k++)
			{
				free(arch[i][j].data[k]);
			}
			free(arch[i][j].data);
		}
		free(arch[i]);
	}
	free(arch);
	return 0;
}

void init_delta()
{
	for(int i = 0; i < 6; i++)
	{
		for(int j = 0; j < layers[i]; j++)
		{
			for(int k = 0; k < arch[i][j].height; k++)
			{
				for(int l = 0; l < arch[i][j].width; l++)
				{
					arch[i][j].delta[k][l] = 0;
				}
			}
		}
	}
}
void MLP(int i, int j)
{
	arch[i][j].data[0][0] = arch[i][j].bias;
	for(int k = 0; k < layers[i-1]; k++)//100
	{
		arch[i][j].data[0][0] += arch[i-1][k].data[0][0] * weights[i-1][j][k].beta;
		//fprintf(fp_log, "arch[%d][%d].data[0][0] += arch[%d][%d].data[0][0] * weights[%d][%d][%d].beta\n", i, j, i-1, k, i-1, j, k);
	}
	arch[i][j].data[0][0] = tanh(arch[i][j].data[0][0]);
//	arch[i][j].fn[0][0] = arch[i][j].data[0][0] * (1-arch[i][j].data[0][0]);
		arch[i][j].fn[0][0] = 1- tanh(arch[i][j].data[0][0])*tanh(arch[i][j].data[0][0]);
}


void Update_weight(Img *arch_before, Img *arch_after, Weight *weight, int mode)
{
	double delta_weight_sum = 0;

	if(mode == 1)
	{
		for(int i = 0; i < weight->height; i++)
		{
			for(int j = 0; j < weight->width; j++)
			{
				delta_weight_sum = 0;
				for(int k = 0; k < arch_after->height; k++)
				{
					for(int l = 0; l < arch_after->width; l++)
					{
						delta_weight_sum += arch_before->data[i+k][j+l] * arch_after->delta[k][l];
					}
				}

				weight->filter[i][j] += delta_weight_sum * learning_gain;
			}
		}
	}else if(mode == 2)
	{
		delta_weight_sum = 0;
		for(int i = 0; i < arch_before->height; i+= weight->height)
		{
			for(int j = 0; j < arch_before->width; j+=weight->width)
			{
				for(int k = 0; k < weight->height; k++)
				{
					for(int l = 0; l < weight->width; l++)
					{
						delta_weight_sum += arch_before->data[i+k][j+l] * arch_after->delta[i/weight->height][j/weight->width];
					}
				}
			}
		}
		delta_weight_sum /= (arch_before->height * arch_before->width);
		weight->beta += delta_weight_sum * learning_gain;

	}

}

void Convolution(int i, int j)
{
	int margin = filter_size[i-1];


	for(int l = 0; l < arch[i-1][0].height - margin + 1; l++)
	{
		for(int m = 0; m < arch[i-1][0].width - margin + 1; m++)
		{
			arch[i][j].data[l][m] = arch[i][j].bias;
			for(int k = 0; k < layers[i-1]; k++)
			{
				for(int n = 0; n < margin; n++)
				{
					for(int o = 0; o < margin; o++)
					{
						arch[i][j].data[l][m] += arch[i-1][k].data[l+n][m+o] * weights[i-1][j][k].filter[n][o];
					}
				}
			//	fprintf(fp_log, "arch[%d][%d].data[%d][%d] = arch[%d][%d] * weights[%d][%d][%d]\n", i, j, l, m , i-1, k, i-1, j, k);
			}
			
			arch[i][j].data[l][m] = tanh(arch[i][j].data[l][m]);
			arch[i][j].fn[l][m] = 1- tanh(arch[i][j].data[l][m])*tanh(arch[i][j].data[l][m]);
		}
	}
}

void Cal_delta(Img *arch_before, Img *arch_after, Weight *weight, int mode)
{
	//mode가 1이면 conv, 2이면 sampling, 3이면 MLP 델타구하기
	if(mode == 1)
	{
		for(int ii = 0; ii < arch_after->height; ii++)
		{
			for(int jj = 0; jj < arch_after->width; jj++)
			{
				for(int kk = 0; kk < weight->height; kk++)
				{
					for(int ll = 0; ll< weight->width; ll++)
					{
						arch_before->delta[ii+kk][jj+ll] += arch_before->fn[ii+kk][jj+ll] * weight->filter[kk][ll] * arch_after->delta[ii][jj];
					}
				}
			}
		}
	}else if(mode ==2)
	{
		for(int i = 0; i < arch_before->height; i+=weight->height)
		{
			for(int j = 0; j < arch_before->width; j+=weight->width)
			{
				for(int k = 0; k < weight->height; k++)
				{
					for(int l = 0; l < weight->width; l++)
					{
 						arch_before->delta[i+k][j+l] += arch_before->fn[i+k][j+l] * weight->beta * arch_after->delta[i/weight->height][j/weight->height];
					}
				}
			}
		}
		
	}
}

void Subsampling(int i, int j)
{
	int sub_size = filter_size[i-1];
	double sum = 0;
	int num_input = sub_size * sub_size;

	for(int l = 0; l < arch[i-1][j].height; l+=sub_size)
	{
		for(int m = 0; m < arch[i-1][j].width; m+=sub_size)
		{
			sum = 0;
			for(int n = 0; n < sub_size; n++)
			{
				for(int o = 0; o < sub_size; o++)
				{
					sum += arch[i-1][j].data[l+n][m+o];
				}
			}
			
			arch[i][j].data[l/sub_size][m/sub_size] = (sum / (double)num_input) + arch[i][j].bias;
			arch[i][j].data[l/sub_size][m/sub_size] *= weights[i-1][j][0].beta;     //filters[i-1][j].beta;
			//fprintf(fp_log, "arch[%d][%d].data[%d][%d] *= weights[%d][%d][%d].beta\n", i, j, l/sub_size, m/sub_size, i-1, j, 0);
			arch[i][j].data[l/sub_size][m/sub_size] = tanh(arch[i][j].data[l/sub_size][m/sub_size]);
			
			arch[i][j].fn[l/sub_size][m/sub_size] = 1- tanh(arch[i][j].data[l/sub_size][m/sub_size])*tanh(arch[i][j].data[l/sub_size][m/sub_size]);
			
		}

	}

}

void Img_alloc(Img* archs)
{
	archs->data = (double**)calloc(archs->height, sizeof(double*));
	archs->delta = (double**)calloc(archs->height, sizeof(double*));
	archs->fn = (double**)calloc(archs->height, sizeof(double*));
	for(int i = 0; i < archs->height; i++)
	{
		archs->data[i] = (double*)calloc(archs->width, sizeof(double));
		archs->delta[i] = (double*)calloc(archs->width, sizeof(double));
		archs->fn[i] = (double*)calloc(archs->width, sizeof(double));
	}	
	archs->bias = ((rand() % 201) * 0.01) -1;


}

void NewSaveWeight(int epoch, double energy)
{
	FILE *fp_save;
	char Name[50];

	sprintf(Name, "result\\weight-%d-%lf.txt", epoch, energy);
	fp_save = fopen(Name, "w");
	for(int i =0; i<5; i++)
	{
		if(i % 2 == 0)
		{
			for(int j = 0; j < layers[i+1]; j++)
			{
				fprintf(fp_save, "%lf\t", arch[i+1][j].bias);
				for(int k = 0; k < layers[i]; k++)
				{
					for(int l = 0; l < filter_size[i]; l++)
					{
						for(int m = 0; m < filter_size[i]; m++)
						{
							fprintf(fp_save, "%lf\t", weights[i][j][k].filter[l][m]);
						}						
					}										
				}
				fprintf(fp_save,"\n");
			}

		}else
		{
			for(int j = 0; j < layers[i+1]; j++)
			{
				//weights[i][j][0].beta = ((rand() % 201) * 0.01) -1;				
				fprintf(fp_save, "%lf\t", weights[i][j][0].beta );
			}
			fprintf(fp_save, "\n");
		}
	}

	for(int i = 0; i < layers[6]; i++)
	{
		for(int j = 0; j < layers[5]; j++)
		{
			//weights[5][i][j].beta =  ((rand() % 201) * 0.01) -1;
			fprintf(fp_save, "%lf\t", weights[5][i][j].beta);
		}		
	}

	fclose(fp_save);
}
void Save_Weight(int epoch, double energy)
{
	FILE *fp_save;
	char Name[50];

	sprintf(Name, "weight\\weight-%d-%lf.txt", epoch, energy);
	
	fp_save = fopen(Name, "w");

	for(int i = 0; i < 6; i++)//0, 1
	{
		for(int j = 0; j < layers[i+1]; j++)//3, 2
		{
			for(int k = 0; k < layers[i]; k++)//1, 3
			{
				fprintf(fp_save, "%lf ", arch[i+1][j].bias);
				if(i == 6)
				{
					fprintf(fp_save, "%lf ", weights[i][j][k].beta);
				}else if(i % 2 == 0)
				{
					for(int l = 0; l < weights[i][j][k].height; l++)
					{
						for(int m = 0; m < weights[i][j][k].width; m++)
						{
							fprintf(fp_save, "%lf ", weights[i][j][k].filter[l][m]);
						}
						fprintf(fp_save, "\n");
					}
				}else if(i % 2 == 1)
				{
					fprintf(fp_save, "%lf ", weights[i][j][k].beta);
				}

			}
		}
	}

	
	fclose(fp_save);

	

}
void Save_Weight()
{
	FILE *fp_save;
	fp_save = fopen("save.txt", "w");

	for(int i = 0; i < 6; i++)//0, 1
	{
		for(int j = 0; j < layers[i+1]; j++)//3, 2
		{
			for(int k = 0; k < layers[i]; k++)//1, 3
			{
				fprintf(fp_save, "%lf ", arch[i+1][j].bias);
				if(i == 6)
				{
					fprintf(fp_save, "%lf ", weights[i][j][k].beta);
				}else if(i % 2 == 0)
				{
					for(int l = 0; l < weights[i][j][k].height; l++)
					{
						for(int m = 0; m < weights[i][j][k].width; m++)
						{
							fprintf(fp_save, "%lf ", weights[i][j][k].filter[l][m]);
						}
						fprintf(fp_save, "\n");
					}
				}else if(i % 2 == 1)
				{
					fprintf(fp_save, "%lf ", weights[i][j][k].beta);
				}

			}
		}
	}



	fclose(fp_save);
}
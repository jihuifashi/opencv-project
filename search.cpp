 
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include <assert.h>
 
#include <thread>
#include<queue>
 
 
using namespace cv;
using namespace std;
 
#define PARTICLE_COUNT (100) //粒子数目
#define MATCH_THRESHOLD  (0.7) //HSV直方图对比巴氏距离阈值
 
class Particle
{
public:
	int orix, oriy;         //原始粒子坐标
	int x, y;               //当前粒子的坐标
	double scale;           //当前粒子窗口的尺寸
	int prex, prey;         //上一帧粒子的坐标
	double prescale;        //上一帧粒子窗口的尺寸
	Rect rectRoi;              //当前粒子矩形窗口
	double weight;          //当前粒子权值
	int T;
};
 
class ParticleTrack {
private:
	bool trackPaused = true;
	bool foundMatches = false; //本次或上次检测,是否匹配到目标
 
	//****有关粒子窗口变化用到的相关变量,用于更新当前粒子的位置****/
	int A1 = 2;
	int A2 = -1;
	int B0 = 1;
	double sigmax = 1.0;
	double sigmay = 0.5;
	double sigmas = 0.001;
 
	int times = 0;//计算次数
 
	//直方图
	MatND targetHist;
	Rect targetRectRoi;
 
 
	vector<Particle> newParticle;//定义一个新的粒子数组
	vector<Particle> particles;  // 粒子参数
 
private:
	//对粒子权重值的降序排列
	static bool compareFunc(Particle p1, Particle p2)
	{
		return p1.weight > p2.weight;
	}
 
	//计算指定image的直方图
	static MatND calculateHistogram(Mat imgRoi) {
		Mat hsvImage;
 
		// 【1】将图像由BGR色彩空间转换到 HSV色彩空间
		cvtColor(imgRoi, hsvImage, COLOR_BGR2HSV);
 
		//【2】初始化计算直方图需要的实参
		// 对hue通道使用30个bin,对saturatoin通道使用32个bin
		int h_bins = 50; int s_bins = 60;
		int histSize[] = { h_bins, s_bins };
		// hue的取值范围从0到256, saturation取值范围从0到180
		float h_ranges[] = { 0, 256 };
		float s_ranges[] = { 0, 180 };
		const float* ranges[] = { h_ranges, s_ranges };
		// 使用第0和第1通道
		int channels[] = { 0, 1 };
 
		// 【3】创建储存直方图的 MatND 类的实例:
		MatND baseHist;
 
		// 【4】计算基准图像，的HSV直方图:
		calcHist(&hsvImage, 1, channels, Mat(), baseHist, 2, histSize, ranges, true, false);
		normalize(baseHist, baseHist, 0, 1, NORM_MINMAX, -1, Mat());
 
		//获取巴氏距离方法.值越小,越相似
		//double base_base = compareHist(baseHist, baseHist, CV_COMP_BHATTACHARYYA);//获取巴氏距离
 
		return baseHist;
	}
 
public:
	void pause() {
		trackPaused = true;
	}
 
	void resume() {
		trackPaused = false;
	}
 
	bool isPaused() {
		return trackPaused;
	}
 
	void setTarget(Mat imgRoi, Rect &rectRoi)
	{
		times = 0;
 
		targetRectRoi = rectRoi;
		targetHist = calculateHistogram(imgRoi);
 
		particles.clear();
		// 目标粒子初始化
		for (int k = 0; k < PARTICLE_COUNT; k++)               //对于每个粒子
		{
			Particle particle;
			particle.x = cvRound(rectRoi.x + 0.5*rectRoi.width);//当前粒子的x坐标
			particle.y = cvRound(rectRoi.y + 0.5*rectRoi.height);//当前粒子的y坐标
 
			//粒子的原始坐标为选定矩形框(即目标)的中心
			particle.orix = particle.x;
			particle.oriy = particle.y;
			//当前粒子窗口的尺寸
			particle.scale = 1;//初始化为1，然后后面粒子到搜索的时候才通过计算更新
 
			//更新上一帧粒子的坐标
			particle.prex = particle.x;
			particle.prey = particle.y;
			//上一帧粒子窗口的尺寸
			particle.prescale = 1;
 
			//当前粒子矩形窗口
			particle.rectRoi = rectRoi;
 
			//当前粒子权值
			particle.weight = 0;//权重初始为0
 
			particle.T = 0;
 
			particles.push_back(particle);
		}
	}
 
	Mat doParticleTracking(Mat imgFrame)
	{
		int xpre, ypre;
		double prescale, scale;
		int x, y;
		double sum = 0.0;
 
		RNG rng;                        //随机数产生器
 
		bool lastFoundMatches = foundMatches;
		foundMatches = false;
		//动态更新每个粒子的位置,并根据和目标的距离,计算粒子的权重W
		for (int k = 0; k < PARTICLE_COUNT; k++)
		{
			if (lastFoundMatches) { //上次至少一个检测到目标
				//当前粒子的坐标
				xpre = particles.at(k).x;
				ypre = particles.at(k).y;
 
				//当前粒子窗口的尺寸
				prescale = particles.at(k).scale;
 
				/*更新跟踪矩形框中心，即粒子中心*///使用二阶动态回归来自动更新粒子状态
				x = cvRound(A1*(particles.at(k).x - particles.at(k).orix) + A2 * (particles.at(k).prex - particles.at(k).orix) +
					B0 * rng.gaussian(sigmax) + particles.at(k).orix);
				particles.at(k).x = max(0, min(x, imgFrame.cols - 1));
 
				y = cvRound(A1*(particles.at(k).y - particles.at(k).oriy) + A2 * (particles.at(k).prey - particles.at(k).oriy) +
					B0 * rng.gaussian(sigmay) + particles.at(k).oriy);
				particles.at(k).y = max(0, min(y, imgFrame.rows - 1));
 
				scale = A1 * (particles.at(k).scale - 1) + A2 * (particles.at(k).prescale - 1) + B0 * (rng.gaussian(sigmas)) + 1.0;
				particles.at(k).scale = max(1.0, min(scale, 3.0));//此处参数设置存疑
 
				particles.at(k).prex = xpre;
				particles.at(k).prey = ypre;
				particles.at(k).prescale = prescale;
			}
			else {
				particles.at(k).x = rand() % imgFrame.cols;//当前粒子的x坐标  随机
				particles.at(k).y = rand() % imgFrame.rows;//当前粒子的y坐标  随机
 
				//粒子的原始坐标为选定矩形框(即目标)的中心
				particles.at(k).orix = particles.at(k).x;
				particles.at(k).oriy = particles.at(k).y;
				//当前粒子窗口的尺寸
				particles.at(k).scale = 1;//初始化为1，然后后面粒子到搜索的时候才通过计算更新
 
				//更新上一帧粒子的坐标
				particles.at(k).prex = particles.at(k).x;
				particles.at(k).prey = particles.at(k).y;
				//上一帧粒子窗口的尺寸
				particles.at(k).prescale = 1;
 
				//当前粒子矩形窗口
				particles.at(k).rectRoi.x = particles.at(k).x;
				particles.at(k).rectRoi.y = particles.at(k).y;
				particles.at(k).rectRoi.width = targetRectRoi.width;
				particles.at(k).rectRoi.height = targetRectRoi.height;
 
				//当前粒子权值
				particles.at(k).weight = 0;//权重初始为0
 
				particles.at(k).T = 0;
			}
 
			/*计算更新得到矩形框数据*/
			particles.at(k).rectRoi.x = max(0, min(cvRound(particles.at(k).x - 0.5*particles.at(k).scale*particles.at(k).rectRoi.width), imgFrame.cols));
			particles.at(k).rectRoi.y = max(0, min(cvRound(particles.at(k).y - 0.5*particles.at(k).scale*particles.at(k).rectRoi.height), imgFrame.rows));
			particles.at(k).rectRoi.width = min(cvRound(particles.at(k).rectRoi.width), imgFrame.cols - particles.at(k).rectRoi.x);
			particles.at(k).rectRoi.height = min(cvRound(particles.at(k).rectRoi.height), imgFrame.rows - particles.at(k).rectRoi.y);
 
			MatND particleHist = calculateHistogram(Mat(imgFrame, particles.at(k).rectRoi));
 
			double weight = compareHist(targetHist, particleHist, CV_COMP_BHATTACHARYYA);//获取巴氏距离,值越小,约相似
			if (weight <= MATCH_THRESHOLD)
				foundMatches = true;
 
			particles.at(k).weight = 1 - weight; //这个必须有!
			/*粒子权重累加*/
			sum += particles.at(k).weight;
		}
 
		Rect rectDot;
		rectDot.width = 5;
		rectDot.height = 5;
		if (!foundMatches) { //本次没有匹配到(均>巴氏距离阈值),画出随机粒子,直接返回
			for (int k = 0; k < PARTICLE_COUNT; k++) {
				//画出随机粒子
				rectDot.x = particles.at(k).rectRoi.x + particles.at(k).rectRoi.width / 2;
				rectDot.y = particles.at(k).rectRoi.y + particles.at(k).rectRoi.height / 2;
				rectangle(imgFrame, rectDot, Scalar(0, 255, 0), 1, 8, 0);//显示跟踪结果，框出
			}
 
			return imgFrame;
		}
 
		// 赋值每个粒子权重
		int countT = 0;
		int countParticle = 0;
 
		//归一化权值
		for (int k = 0; k < PARTICLE_COUNT; k++)
		{
			particles.at(k).weight /= sum;
			particles.at(k).T = cvRound(particles.at(k).weight*PARTICLE_COUNT);
			if (particles.at(k).T > 0) {
				countT += particles.at(k).T;
				countParticle++;
			}
 
			//前面都已经识别完毕,标注出画粒子
			rectDot.x = particles.at(k).rectRoi.x + particles.at(k).rectRoi.width / 2;
			rectDot.y = particles.at(k).rectRoi.y + particles.at(k).rectRoi.height / 2;
			rectangle(imgFrame, rectDot, Scalar(0, 255, 0), 1, 8, 0);//显示跟踪结果，框出
		}
 
		// 对归一化后权重排序
		sort(particles.begin(), particles.end(), ParticleTrack::compareFunc);
 
		//取前1/4个粒子作为跟踪结果,计算其最大权重目标的期望位置
		Rect rectTracking;              //初始化一个Rect作为跟踪的临时
		double weight_temp = 0.0;
		for (int k = 0; k < PARTICLE_COUNT / 4; k++)
		{
			weight_temp += particles.at(k).weight;
		}
		for (int k = 0; k < PARTICLE_COUNT / 4; k++)
		{
			particles.at(k).weight /= weight_temp;
		}
 
		//更新检测框,显示匹配最佳的第一个(红色)和匹配优秀的前5个(绿色,包含第一个已红色)
		for (int k = 0; k < 5; k++)
		{
			rectTracking.x = particles.at(k).rectRoi.x;
			rectTracking.y = particles.at(k).rectRoi.y;
			rectTracking.width = particles.at(k).rectRoi.width;
			rectTracking.height = particles.at(k).rectRoi.height;
 
			if (k == 0)
				rectangle(imgFrame, rectTracking, Scalar(0, 0, 255), 1, 8, 0);//显示跟踪结果，红框
			else
				rectangle(imgFrame, rectTracking, Scalar(0, 255, 0), 1, 8, 0);//显示跟踪结果，红框
		}
 
		
		bool flag = false;          // 完成粒子赋值,结束标志
		newParticle.clear();
		//重采样，根据粒子权重 决定重采样粒子多少
		for (int k = 0; k < PARTICLE_COUNT; k++)
		{
			if (flag)            // 完成粒子赋值，跳出循环
			{
				break;
			}
			if (particles.at(k).T > 0)
			{
				int generateCount = (int)((particles.at(k).T*1.0 / countParticle)*PARTICLE_COUNT);
				for (int i = 0; i < generateCount; i++)           // 权重越大，该点赋值数越多
				{
					newParticle.push_back(particles.at(k));
					if (newParticle.size() >= PARTICLE_COUNT)
					{
						flag = true;
						break;
					}
				}
			}
		}
 
		//没有采样满PARTICLE_COUNT个,复制并填充最大权值的信息
		while (newParticle.size() < PARTICLE_COUNT)
		{
			newParticle.push_back(particles.at(0));//复制相关性最大的权值的样本填满剩余粒子空间
		}
 
		//更新粒子
		particles.clear();
		// 将粒子点替换为更新后的粒子点
		for (int k = 0; k < PARTICLE_COUNT; k++)
		{
			//重新计算粒子检测框尺寸,  避免萎缩过小
			if (newParticle.at(k).rectRoi.width < targetRectRoi.width)
				newParticle.at(k).rectRoi.width = imgFrame.cols - newParticle.at(k).x;
			if (newParticle.at(k).rectRoi.height < targetRectRoi.height)
				newParticle.at(k).rectRoi.height = imgFrame.rows - newParticle.at(k).y;
			particles.push_back(newParticle.at(k));
		}
 
		cout << times++ << endl;            // 输出检测帧数
		return imgFrame;
	}
};
 
bool leftButtonDownFlag = false;  //左键单击后的标志位
bool leftButtonUpFlag = false;    //左键单击后松开的标志位
Point Point_s;                  //目标矩形框选取起点
Point Point_move;             //鼠标按下移动过程中,矩形框移动的当前点
Point Point_e;                  //目标矩形框选取终点,鼠标左键弹起来的终点
bool  selectTargetCompleted = false;
bool  saveImage = false;
int   saveImageNameIndex = 0;
 
void onMouse(int event, int x, int y, int flags, void *param)
{
	if (event == CV_EVENT_RBUTTONDOWN)
	{
		saveImage = true;
	}
 
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		selectTargetCompleted = false;
		leftButtonDownFlag = true; //标志位
		leftButtonUpFlag = false;
		Point_move = Point(x, y);  //设置左键按下点的矩形起点
		Point_s = Point_move;
	}
	else if (event == CV_EVENT_MOUSEMOVE && leftButtonDownFlag)
	{
		Point_move = Point(x, y);
	}
	else if (event == CV_EVENT_LBUTTONUP && leftButtonDownFlag)
	{
		leftButtonDownFlag = false;
		Point_move = Point(x, y);
		Point_e = Point_move;
		leftButtonUpFlag = true;
		selectTargetCompleted = true;
	}
}
 
 
queue<Mat> imageFrameQueue;
queue<Mat> resultFrameQueue;
bool threadWorking = true;
 
//粒子跟踪线程
void particleTrackThread(ParticleTrack *pParticleTrack) {
 
	if (pParticleTrack == NULL)
		return;
	while (threadWorking) {
		if (!pParticleTrack->isPaused() && selectTargetCompleted && imageFrameQueue.size() >= 1) {
			Mat imageFrame = imageFrameQueue.front();
			imageFrameQueue.pop();
 
			blur(imageFrame, imageFrame, Size(2, 2));//先对原图进行均值滤波处理
 
			imageFrame = pParticleTrack->doParticleTracking(imageFrame);
 
			resultFrameQueue.push(imageFrame); //将i压入队列的尾部
		}
 
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}
 
//这里是入口函数,若果独立运行,请修改成main()
//int doParticleFilter_main(/*int argc, char **argv*/)
int main(int argc, char **argv)
{
	Mat frame,imgROI;
	Rect rectROI;
 
	//打开摄像头或者特定视频
	VideoCapture videoCapture;
	videoCapture.open(0); //打开摄像头
	//videoCapture.open("saiche.mp4");//打开视频文件
	//读入视频是否为空
	if (!videoCapture.isOpened())
	{
		return -1;
	}
	namedWindow("原始视频", 1);
	namedWindow("目标跟踪视频", 1);
	namedWindow("选中的ROI区域", 1);
	setMouseCallback("原始视频", onMouse, 0);//鼠标回调函数，响应鼠标以选择跟踪区域
 
 
	ParticleTrack particleTrack_1;
 
	threadWorking = true;
	std::thread workingThread(particleTrackThread, &particleTrack_1);//启动粒子跟踪线程
 
	while (1)
	{
		videoCapture >> frame;
		if (frame.empty())
		{
			threadWorking = false;
			workingThread.join();
			return -1;
		}
 
		if (leftButtonDownFlag) // 绘制截取目标窗口
		{
			particleTrack_1.pause();
 
			rectROI.x = Point_s.x;
			rectROI.y = Point_s.y;
			rectROI.width = Point_move.x - Point_s.x;
			rectROI.height = Point_move.y - Point_s.y;
			rectangle(frame, rectROI, Scalar(0, 255, 0), 3, 8, 0);
 
			cout << "rectROI:  "<<rectROI << endl;
		}
 
		if (selectTargetCompleted && leftButtonUpFlag)
		{
			leftButtonUpFlag = false;
			rectROI.x = Point_s.x;
			rectROI.y = Point_s.y;
			rectROI.width = Point_e.x - Point_s.x;
			rectROI.height = Point_e.y - Point_s.y;
			imgROI = Mat(frame, rectROI);   //目标图像
			imgROI = imgROI.clone();//复制一份出来
			imshow("选中的ROI区域", imgROI);
 
			particleTrack_1.setTarget(imgROI, rectROI);
			particleTrack_1.resume();
		}
 
		imshow("原始视频", frame);
 
		if (selectTargetCompleted)
		{
			if (imageFrameQueue.size() > 1)
				imageFrameQueue.pop();
			imageFrameQueue.push(frame.clone()); //将新image压入队列的尾部
		}
 
		if (resultFrameQueue.size() >= 1) {
			Mat resultFrame = resultFrameQueue.front();
			resultFrameQueue.pop();
			
			imshow("目标跟踪视频", resultFrame);
 
			if (saveImage) {
				saveImage = false;
 
				string name = "image_" + to_string(saveImageNameIndex)+ ".png";
				saveImageNameIndex++;
				imwrite(name, resultFrame);
			}
		}
 
		if (cv::waitKey(50) >= 0)
			break;
	}
 
	threadWorking = false;
	workingThread.join();
 
	return 0;
}

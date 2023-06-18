#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
using namespace cv;
using namespace std;

#define pi 3.1415926
double yc(double x, double p1, double p2, double p3, double p4);
double statcom(double q, int i, int j, int b);
double studt(double t, int n);
int fitsin(vector<double>tt, vector<double>yy, double &amp, double &omega, double &phase, double &offset);

#define Npoint 352
#define Npar 4
int main()
{

    ifstream filex("xdata.txt");
    ifstream filey("ydata.txt");
    string comma[Npoint];
    double xx[Npoint];
    double zz[Npoint];
    for (int i = 0; i < Npoint; i++)
    {
        filex >> comma[i] >> xx[i];
        filey >> comma[i] >> zz[i];
    }

    vector<double>tt;
    vector<double>yy;
    for (int i = 0; i < Npoint; i++)
    {
        tt.push_back(xx[i]);
        yy.push_back(zz[i]);
    }

    double amp = 0.75; // 初始值
    double omega = 1.88;
    double phase = 1.8;
    double offset = 1.2;

    int flag = fitsin(tt, yy, amp, omega, phase, offset);
    cout <<flag<<endl;
    cout<<"amp: "<<amp<<"\tomega: "<<omega<<"\tphase: "<<phase<<"\toffset: "<<offset<<endl;
}

int fitsin(vector<double>tt, vector<double>yy, double &amp, double &omega, double &phase, double &offset)
{
    if(tt.size() != yy.size()) 
        return 0;
    int flag = -1;
    double dgfre = tt.size()-4; //degree of freedom
    double par[4] = {amp, omega, phase, offset};
    double eval[4]; //评估参数
    double lasteval3 = 0;
    double cost; //方差
    double lastcost = 0;
    for (int epoch = 0;; epoch++)
    {
        double rms;//root mean square
        Mat arr = Mat::zeros(4, 5, CV_64FC1);
        cost = 0;
        for (int iter = 0; iter <tt.size(); iter++)
        {
            double derivative[5];
            derivative[0] = (yc(tt[iter], par[0] + par[0] / 1e3, par[1], par[2], par[3]) - yc(tt[iter], par[0], par[1], par[2], par[3])) / (par[0] / 1e3);
            derivative[1] = (yc(tt[iter], par[0], par[1] + par[1] / 1e3, par[2], par[3]) - yc(tt[iter], par[0], par[1], par[2], par[3])) / (par[1] / 1e3);
            derivative[2] = (yc(tt[iter], par[0], par[1], par[2] + par[2] / 1e3, par[3]) - yc(tt[iter], par[0], par[1], par[2], par[3])) / (par[2] / 1e3);
            derivative[3] = (yc(tt[iter], par[0], par[1], par[2], par[3] + par[3] / 1e3) - yc(tt[iter], par[0], par[1], par[2], par[3])) / (par[3] / 1e3);
            derivative[4] = yy[iter] - yc(tt[iter], par[0], par[1], par[2], par[3]);
            cost += pow(derivative[4],2);

            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 5; j++)
                    arr.at<double>(i, j) += derivative[j] * derivative[i];
        }
        for (int i = 0; i < 4; i++) //手解方程
        {
            double save = arr.at<double>(i, i);
            arr.at<double>(i, i) = 1;
            for (int k = 0; k < 5; k++)
                arr.at<double>(i, k) /= save;
            for (int j = 0; j < 4; j++)
            {
                if (i != j)
                {
                    save = arr.at<double>(j, i);
                    arr.at<double>(j, i) = 0;
                    for (int k = 0; k < 5; k++)
                        arr.at<double>(j, k) -= arr.at<double>(i, k) * save;
                }
            }
        }
        rms = sqrt(cost / dgfre);
        for (int i = 0; i < 4; i++)
        {
            par[i] += arr.at<double>(i, 4);
            eval[i] = studt(par[i]/(rms * sqrt(arr.at<double>(i, i))), dgfre);
        }

        //写终止条件
        if(flag > 0) 
        {
            amp = par[0];
            omega = par[1];
            phase = par[2];
            offset = par[3];
            return flag; //判断完后必须再迭代一次
        }
        if(eval[0]<1e-8 && eval[1]<1e-8 && eval[2]<1e-8 && eval[3]<1e-8)
            flag = 2; //解的很好
        else if(eval[0]<1e-8 && eval[1]<1e-8 && eval[3]<1e-8 && abs(lasteval3-eval[2])<1e-3)
            flag = 1; //解的还行
        else 
            flag = -1; //继续迭代
        if(epoch>0 && (cost > tt.size()*0.2 || abs(cost-lastcost)>tt.size()*0.2))
            return 0; //根本就不收敛
        if(epoch > 7)
            return 0;
        lastcost = cost;
        lasteval3 = eval[2];
    }
}

double yc(double x, double p1, double p2, double p3, double p4)
{
    return p1 * sin(p2 * x + p3) + p4;
}

double statcom(double q, int i, int j, int b)
{
    double zz = 1;
    double z =  zz;
    int k = i;
    while(k <= j)
    {
        zz = zz * q * k / (k-b);
        z += zz;
        k += 2;
    }
    return z;
}

double studt(double t, int n)
{
    t = abs(t);
    double w = t / sqrt(n);
    double th = atan(w);
    double sth = sin(th);
    double cth = cos(th);
    if(n / 2 ==1)
        return 1-(th + sth*cth*statcom(cth*cth,2,n-3,-1))/(pi/2);
    else
        return 1-sth*statcom(cth*cth,1,n-3,-1);
}
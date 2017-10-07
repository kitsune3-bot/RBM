#ifdef _DEBUG
#define CV_EXT "d.lib"
#else
#define CV_EXT ".lib"
#endif
#pragma comment(lib, "opencv_core320" CV_EXT)
#pragma comment(lib, "opencv_highgui320" CV_EXT)
#pragma comment(lib, "opencv_imgcodecs320" CV_EXT)
#pragma comment(lib, "opencv_imgproc320" CV_EXT)

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include "Eigen/Core"
#include "ConditionalGRBM.h"
#include "ConditionalGRBMTrainer.h"
#include "ConditionalGRBMSampler.h"

std::vector<double> make_colordata(const std::string & fname);

std::vector<double> make_colordata(const std::string & fname) {
    cv::Mat_<cv::Vec3b> img = cv::imread(fname, 1);
    int img_size = img.cols * img.rows;

    std::vector<double> data(3 * img_size);

    for (int i = 0; i < img.cols; i++) {
        for (int j = 0; j < img.rows; j++) {
			data[img_size * 0 + j * img.cols + i] = static_cast<double>(img.at<cv::Vec3b>(j, i)[0]);// / 255.0;
			data[img_size * 1 + j * img.cols + i] = static_cast<double>(img.at<cv::Vec3b>(j, i)[1]);// / 255.0;
			data[img_size * 2 + j * img.cols + i] = static_cast<double>(img.at<cv::Vec3b>(j, i)[2]);// / 255.0;
        }
    }

    return data;
}

std::vector<double> make_monodata(const std::string & fname) {
    cv::Mat_<cv::Vec3b> img = cv::imread(std::string(fname), 1);
    int img_size = img.cols * img.rows;

    std::vector<double> data(img_size);

    for (int i = 0; i < img.cols; i++) {
        for (int j = 0; j < img.rows; j++) {
            double pixel = 0.0;

			pixel += static_cast<double>(img.at<cv::Vec3b>(j, i)[0]);// / 255.0;
			pixel += static_cast<double>(img.at<cv::Vec3b>(j, i)[1]);// / 255.0;
			pixel += static_cast<double>(img.at<cv::Vec3b>(j, i)[2]);// / 255.0;
            data[j * img.cols + i] = pixel / 3.0;
        }
    }

    return data;
}


int main(void) {
    int im_num = 2;
    std::vector<std::vector<double>> dataset(im_num);
    for (int i = 0; i < im_num; i++) {
        dataset[i] = make_colordata(std::string("./image/color/") + std::to_string(i) + std::string(".png"));
    }
    
    std::vector<std::vector<double>> cond_dataset(im_num);
    for (int i = 0; i < im_num; i++) {
        cond_dataset[i] = make_monodata(std::string("./image/mono/") + std::to_string(i) + std::string(".png"));
    }

    int v_size = dataset[0].size();
    int x_size = cond_dataset[0].size();
    ConditionalGRBM cgrbm(v_size, 10, x_size);
    ConditionalGRBMTrainer cgrbm_train(cgrbm);
    cgrbm_train.batchSize = 1;
    cgrbm_train.epoch = 1000;
    cgrbm_train.cdk = 1;
    cgrbm_train.learningRate = 0.01;
    // FIXME: 学習にバグがあるらしい
    auto w1(cgrbm.params.hvW);
    cgrbm_train.train(cgrbm, dataset, cond_dataset);
    auto w2(cgrbm.params.hvW);
    std::cout << (w1 - w2).sum() << std::endl;
    {
        Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(cond_dataset[0].data(), cond_dataset[0].size());
        cgrbm.nodes.x = vect;
        std::cout << vect.sum() << std::endl;

        Eigen::VectorXd col_vect = Eigen::Map<Eigen::VectorXd>(dataset[0].data(), dataset[0].size());
        cgrbm.nodes.v = col_vect;
       // cgrbm.nodes.v.setConstant(0.0);
        std::cout << col_vect.sum() << std::endl;

        // 画像の期待値計算
        // FIXME: 期待値にしたほうが良いのでは…
        for (int i = 0; i < 1; i++) {
            ConditionalGRBMSampler sampler;
            sampler.updateByBlockedGibbsSamplingHidden(cgrbm);
            sampler.updateByBlockedGibbsSamplingVisible(cgrbm);
        }

        //std::cout << cgrbm.nodes.v << std::endl;

        //画像をいじる
        cv::Mat_<cv::Vec3b> img = cv::imread(std::string("./image/color/0.png"));
        int img_size = img.cols * img.rows;

        for (int i = 0; i < img.cols; i++) {
            for (int j = 0; j < img.rows; j++) {
                img.at<cv::Vec3b>(j, i)[0] = 255;
                img.at<cv::Vec3b>(j, i)[1] = 0;
                img.at<cv::Vec3b>(j, i)[2] = 0;
            }
        }


        for (int i = 0; i < img.cols; i++) {
            for (int j = 0; j < img.rows; j++) {
				img.at<cv::Vec3b>(j, i)[0] = cgrbm.nodes.v(img_size * 0 + j * img.cols + i);// *255.0;
				img.at<cv::Vec3b>(j, i)[1] = cgrbm.nodes.v(img_size * 1 + j * img.cols + i);// *255.0;
				img.at<cv::Vec3b>(j, i)[2] = cgrbm.nodes.v(img_size * 2 + j * img.cols + i);// *255.0;
            }
        }

        cv::imwrite("out1.png", img);
        cv::namedWindow("img", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
        cv::imshow("img", img);
    }

    {
        Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(cond_dataset[1].data(), cond_dataset[1].size());
        cgrbm.nodes.x = vect;
        std::cout << vect.sum() << std::endl;

        Eigen::VectorXd col_vect = Eigen::Map<Eigen::VectorXd>(dataset[1].data(), dataset[1].size());
        cgrbm.nodes.v = col_vect;
       // cgrbm.nodes.v.setConstant(0.0);

        // 画像の期待値計算
        // FIXME: 期待値にしたほうが良いのでは…
        for (int i = 0; i < 1; i++) {
            ConditionalGRBMSampler sampler;
            sampler.updateByBlockedGibbsSamplingHidden(cgrbm);
            sampler.updateByBlockedGibbsSamplingVisible(cgrbm);
        }

        //std::cout << cgrbm.nodes.v << std::endl;

        //画像をいじる
        cv::Mat_<cv::Vec3b> img = cv::imread(std::string("./image/color/1.png"));
        int img_size = img.cols * img.rows;

        for (int i = 0; i < img.cols; i++) {
            for (int j = 0; j < img.rows; j++) {
                img.at<cv::Vec3b>(j, i)[0] = 255;
                img.at<cv::Vec3b>(j, i)[1] = 0;
                img.at<cv::Vec3b>(j, i)[2] = 0;
            }
        }


        for (int i = 0; i < img.cols; i++) {
            for (int j = 0; j < img.rows; j++) {
				img.at<cv::Vec3b>(j, i)[0] = cgrbm.nodes.v(img_size * 0 + j * img.cols + i);// *255.0;
				img.at<cv::Vec3b>(j, i)[1] = cgrbm.nodes.v(img_size * 1 + j * img.cols + i);// *255.0;
				img.at<cv::Vec3b>(j, i)[2] = cgrbm.nodes.v(img_size * 2 + j * img.cols + i);// *255.0;
            }
        }

        cv::imwrite("out2.png", img);

        cv::namedWindow("img2", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
        cv::imshow("img2", img);
    }



    cv::waitKey(0);
}
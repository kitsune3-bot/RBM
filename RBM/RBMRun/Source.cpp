#include <iostream>
#include "RBM.h"
#include "RBMTrainer.h"
#include "GeneralizedRBM.h"
#include "GeneralizedRBMTrainer.h"
#include "GBRBM.h"
#include "GBRBMTrainer.h"
#include "GeneralizedGRBM.h"
#include "GeneralizedGRBMTrainer.h"
#include "ConditionalGRBM.h"

int main(int argc, char **argv) {
    std::vector<std::vector<double>> dataset(10);
    dataset[0] = std::vector<double>({ 1,1,1,1,1,1,1,0,1,1 });
    dataset[1] = std::vector<double>({ 0,0,1,1,1,1,1,1,0,1 });
    dataset[2] = std::vector<double>({ 1,1,0,1,1,1,0,1,1,0 });
    dataset[3] = std::vector<double>({ 0,1,1,0,1,1,1,1,0,1 });
    dataset[4] = std::vector<double>({ 1,1,1,1,0,1,1,0,1,1 });
    dataset[5] = std::vector<double>({ 0,0,1,0,1,1,1,1,1,0 });
    dataset[6] = std::vector<double>({ 1,1,1,0,1,1,0,1,1,1 });
    dataset[7] = std::vector<double>({ 1,0,1,1,1,1,1,0,0,1 });
    dataset[8] = std::vector<double>({ 1,1,1,1,1,1,1,1,1,1 });
    dataset[9] = std::vector<double>({ 0,1,0,1,0,1,0,0,1,1 });

    /*
    RBM rbm(10, 10);
    RBMTrainer rbm_t(rbm);
    
    rbm_t.epoch = 300;
    rbm_t.cdk = 3;
    rbm_t.batchSize = 5;
    rbm_t.learningRate = 0.001;
    std::cout << rbm.params.w << std::endl;
    rbm_t.train(rbm, dataset);
    std::cout << rbm.params.w << std::endl;
    */
    
    /*
    GeneralizedRBM grbm(10, 10);
    grbm.setHiddenDiveSize(5);
    grbm.setHiddenMax(2);
    grbm.setHiddenMin(-2);
    auto set = grbm.splitHiddenSet();
    GeneralizedRBMTrainer grbm_t(grbm);
    grbm_t.epoch = 300;
    grbm_t.cdk = 3;
    grbm_t.batchSize = 5;
    grbm_t.learningRate = 0.001;
    std::cout << grbm.mu(1) << std::endl;

    std::cout << grbm.params.w << std::endl;
    grbm_t.train(grbm, dataset);
    std::cout << grbm.params.w << std::endl;
    */

    /*
    GBRBM gbrbm(10, 10);
    GBRBMTrainer gbrbm_t(gbrbm);
    gbrbm_t.epoch = 300;
    gbrbm_t.cdk = 3;
    gbrbm_t.batchSize = 5;
    gbrbm_t.learningRate = 0.001;
    std::cout << gbrbm.mu(1) << std::endl;

    std::cout << gbrbm.params.w << std::endl;
    gbrbm_t.train(gbrbm, dataset);
    std::cout << gbrbm.params.w << std::endl;
    */

    /*
    GeneralizedGRBM gen_grbm(10, 10);
    gen_grbm.setHiddenDiveSize(1);
    gen_grbm.setHiddenMax(2);
    gen_grbm.setHiddenMin(-2);
    auto set = gen_grbm.splitHiddenSet();
    GeneralizedGRBMTrainer gen_grbm_t(gen_grbm);
    gen_grbm_t.epoch = 10;
    gen_grbm_t.cdk = 3;
    gen_grbm_t.batchSize = 5;
    gen_grbm_t.learningRate = 0.1;
    std::cout << gen_grbm.mu(1) << std::endl;

    std::cout << gen_grbm.params.w << std::endl;
    gen_grbm_t.train(gen_grbm, dataset);
    std::cout << gen_grbm.params.w << std::endl;
    */

    ConditionalGRBM cond_grbm(10, 3, 10);

    return 0;
}
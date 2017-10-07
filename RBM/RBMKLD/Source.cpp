#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "rbmutil.h"
#include "GeneralizedRBM.h"
#include "GeneralizedRBMTrainer.h"
#include "GeneralizedSparseRBM.h"
#include "GeneralizedSparseRBMTrainer.h"
#include "SQLiteCpp/SQLiteCpp.h"
#include "sqlite3.h"

typedef struct {
	int vSize = 5;
	int hSize = 10;
	int appendH = 0;
	int datasize = 25;
	int epoch = 1000;
	int cdk = 1;
	int batchsize = datasize;
	double learningRate = 0.1;
	double momentumRate = 0.9;
	int divSize = 1;
	bool realFlag = false;
} OPTION;


/**
// SQLITE3 データベース仕様(仮)
// +-------------+------+-------------+
// |              result              |
// +-------------+------+------+------+
// | uid(UNIQUE) | name | json | time |
// +-------------+------+------+------+
//
//   uid INTEGER PRIMARY KEY AUTOINCREMENT
//   name TEXT
//   json BLOB
//   time TEXT
//
**/

int exist_flag = 0;

void make_table(SQLite::Database & db) {
	try {
		db.exec("BEGIN TRANSACTION");
	}
	catch (std::exception& e)
	{
		std::cout << "exception: " << e.what() << std::endl;
	}

	char *err_msg = NULL;
	if (db.tableExists("result")) {
		std::string query = "DROP TABLE result;";
		db.exec(query);
	}

	std::string query = "CREATE TABLE result (uid INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, json BLOB, time TEXT);";

	int ret = db.exec(query);

	try {
		db.exec("COMMIT");
	}
	catch (std::exception& e)
	{
		std::cout << "exception: " << e.what() << std::endl;
	}

}



void write_to_db(SQLite::Database & db, std::string & name, std::string & json) {
	try {
		std::stringstream ss_query;
		//	ss_query << "INSERT INTO result (name, json, time) VALUES("  << name << ", " << json << ", )";
		ss_query << "INSERT INTO result(name, json, time) VALUES(?, ?, datetime(CURRENT_TIMESTAMP,'localtime'));";
		std::string query = ss_query.str();

		SQLite::Statement state(db, query);
		state.bind(1, name);
		state.bind(2, json);
		state.exec();
		//if (ret != sqlite_ok) {
		//	throw;
		//}
	}
	catch (std::exception& e)
	{
		std::cout << "exception: " << e.what() << std::endl;
	}

}

// パラメータファイル書き込み(SQLITEに)
template <class RBM>
void write_params(SQLite::Database & db, RBM & rbm, std::string fname) {
	write_to_db(db, fname, rbm.params.serialize());
}

// 学習情報書き込み(SQLITEに)
template <class RBM, class Trainer>
void write_train_info(SQLite::Database & db, RBM & rbm, Trainer & trainer, std::string fname) {
	auto js = trainer.trainInfoJson(rbm);
	write_to_db(db, fname, js);
}

// (汎化|訓練)誤差情報書き込み(SQLITEに)
template <class RBMGEN, class RBMTRAIN, class Trainer, class Dataset>
void write_error_info(SQLite::Database & db, RBMGEN & rbm_gen, RBMTRAIN & rbm_train, Trainer & trainer, Dataset & dataset, std::string fname) {

	auto js = nlohmann::json();
	js["kld"] = rbmutil::kld(rbm_gen, rbm_train, std::vector<int>{0, 1});
	js["logLikelihood"] = trainer.logLikeliHood(rbm_train, dataset);

	write_to_db(db, fname, js.dump());
}

// 実行ルーチン
template<class RBM, class DATASET>
void run(SQLite::Database & db, OPTION & option, int try_count, RBM & rbm_gen, DATASET & dataset) {
	auto rbm_exact = GeneralizedRBM(option.vSize, option.hSize + option.appendH);
	rbm_exact.params.initParamsRandom(-0.01, 0.01);
	rbm_exact.setHiddenDiveSize(option.divSize);
	rbm_exact.setHiddenMin(-1.0);
	rbm_exact.setHiddenMax(1.0);
	rbm_exact.setRealHiddenValue(option.realFlag);

	auto rbm_trainer_exact = GeneralizedRBMTrainer(rbm_exact);
	rbm_trainer_exact.epoch = option.epoch;
	rbm_trainer_exact.cdk = option.cdk;
	rbm_trainer_exact.batchSize = option.batchsize;
	rbm_trainer_exact.learningRate = option.learningRate;
	rbm_trainer_exact.momentumRate = option.momentumRate;

	auto rbm_cd = GeneralizedRBM(option.vSize, option.hSize + option.appendH);
	rbm_cd.params.initParamsRandom(-0.01, 0.01);
	rbm_cd.setHiddenDiveSize(option.divSize);
	rbm_cd.setHiddenMin(-1.0);
	rbm_cd.setHiddenMax(1.0);
	rbm_cd.setRealHiddenValue(option.realFlag);

	auto rbm_trainer_cd = GeneralizedRBMTrainer(rbm_cd);
	rbm_trainer_cd.epoch = option.epoch;
	rbm_trainer_cd.cdk = option.cdk;
	rbm_trainer_cd.batchSize = option.batchsize;
	rbm_trainer_cd.learningRate = option.learningRate;
	rbm_trainer_cd.momentumRate = option.momentumRate;

	//std::cout << "[Exact]" << std::endl;
	//rbmutil::print_params(rbm_exact);
	//std::cout << "KLD: " << rbmutil::kld(rbm_gen, rbm_exact, std::vector<int>{0, 1}) << std::endl;
	//std::cout << "Logliklihood: " << rbm_trainer_exact.logLikeliHood(rbm_exact, dataset) << std::endl;
	//std::cout << std::endl;

	//std::cout << "[Contrastive Divergence]" << std::endl;
	//rbmutil::print_params(rbm_cd);
	//std::cout << "KLD: " << rbmutil::kld(rbm_gen, rbm_cd, std::vector<int>{0, 1}) << std::endl;
	//std::cout << "Logliklihood: " << rbm_trainer_cd.logLikeliHood(rbm_cd, dataset) << std::endl;

	for (int epoch_count = 0; epoch_count < option.epoch; epoch_count++) {
		// Exact
		std::string rbm_div = option.realFlag ? "c" : std::to_string(option.divSize);

		rbm_trainer_exact.trainOnceExact(rbm_exact, dataset);
		std::stringstream ss_exact_fname;
		ss_exact_fname << try_count << "_exact" << "_epoch" << epoch_count << "_div" << rbm_div << ".train.json";
		write_train_info(db, rbm_exact, rbm_trainer_exact, ss_exact_fname.str());

		std::stringstream ss_exact_error_fname;
		ss_exact_error_fname << try_count << "_error_exact" << "_epoch" << epoch_count << "_div" << rbm_div << ".error.json";
		write_error_info(db, rbm_gen, rbm_exact, rbm_trainer_exact, dataset, ss_exact_error_fname.str());




		// Contrastive Divergence
		rbm_trainer_cd.trainOnceCD(rbm_cd, dataset);
		std::stringstream ss_cd_fname;
		ss_cd_fname << try_count << "_cd" << "_epoch" << epoch_count << "_div" << rbm_div << ".train.json";
		write_train_info(db, rbm_cd, rbm_trainer_cd, ss_cd_fname.str());

		std::stringstream ss_cd_error_fname;
		ss_cd_error_fname << try_count << "_error_cd" << "_epoch" << epoch_count << "_div" << rbm_div << ".error.json";
		write_error_info(db, rbm_gen, rbm_cd, rbm_trainer_cd, dataset, ss_cd_error_fname.str());
	}
}

// 実行ルーチン
template<class SRBM, class DATASET>
void run_sparse(SQLite::Database & db, OPTION & option, int try_count, SRBM & rbm_gen, DATASET & dataset) {
	auto rbm_exact = GeneralizedSparseRBM(option.vSize, option.hSize + option.appendH);
	rbm_exact.params.initParamsRandom(-0.01, 0.01);
	rbm_exact.setHiddenDiveSize(option.divSize);
	rbm_exact.setHiddenMin(-1.0);
	rbm_exact.setHiddenMax(1.0);
	rbm_exact.setRealHiddenValue(option.realFlag);

	auto rbm_trainer_exact = GeneralizedSparseRBMTrainer(rbm_exact);
	rbm_trainer_exact.epoch = option.epoch;
	rbm_trainer_exact.cdk = option.cdk;
	rbm_trainer_exact.batchSize = option.batchsize;
	rbm_trainer_exact.learningRate = option.learningRate;
	rbm_trainer_exact.momentumRate = option.momentumRate;

	auto rbm_cd = GeneralizedSparseRBM(option.vSize, option.hSize + option.appendH);
	rbm_cd.params.initParamsRandom(-0.01, 0.01);
	rbm_cd.setHiddenDiveSize(option.divSize);
	rbm_cd.setHiddenMin(-1.0);
	rbm_cd.setHiddenMax(1.0);
	rbm_cd.setRealHiddenValue(option.realFlag);

	auto rbm_trainer_cd = GeneralizedSparseRBMTrainer(rbm_cd);
	rbm_trainer_cd.epoch = option.epoch;
	rbm_trainer_cd.cdk = option.cdk;
	rbm_trainer_cd.batchSize = option.batchsize;
	rbm_trainer_cd.learningRate = option.learningRate;
	rbm_trainer_cd.momentumRate = option.momentumRate;

	//std::cout << "[Exact]" << std::endl;
	//rbmutil::print_params(rbm_exact);
	//std::cout << "KLD: " << rbmutil::kld(rbm_gen, rbm_exact, std::vector<int>{0, 1}) << std::endl;
	//std::cout << "Logliklihood: " << rbm_trainer_exact.logLikeliHood(rbm_exact, dataset) << std::endl;
	//std::cout << std::endl;

	//std::cout << "[Contrastive Divergence]" << std::endl;
	//rbmutil::print_params(rbm_cd);
	//std::cout << "KLD: " << rbmutil::kld(rbm_gen, rbm_cd, std::vector<int>{0, 1}) << std::endl;
	//std::cout << "Logliklihood: " << rbm_trainer_cd.logLikeliHood(rbm_cd, dataset) << std::endl;

	for (int epoch_count = 0; epoch_count < option.epoch; epoch_count++) {
		// Exact
		std::string rbm_div = option.realFlag ? "c" : std::to_string(option.divSize);

		rbm_trainer_exact.trainOnceExact(rbm_exact, dataset);
		std::stringstream ss_exact_fname;
		ss_exact_fname << try_count << "_exact_sparse" << "_epoch" << epoch_count << "_div" << rbm_div << ".train.json";
		write_train_info(db, rbm_exact, rbm_trainer_exact, ss_exact_fname.str());

		std::stringstream ss_exact_error_fname;
		ss_exact_error_fname << try_count << "_error_exact_sparse" << "_epoch" << epoch_count << "_div" << rbm_div << ".error.json";
		write_error_info(db, rbm_gen, rbm_exact, rbm_trainer_exact, dataset, ss_exact_error_fname.str());




		// Contrastive Divergence
		rbm_trainer_cd.trainOnceCD(rbm_cd, dataset);
		std::stringstream ss_cd_fname;
		ss_cd_fname << try_count << "_cd" << "_epoch_sparse" << epoch_count << "_div" << rbm_div << ".train.json";
		write_train_info(db, rbm_cd, rbm_trainer_cd, ss_cd_fname.str());

		std::stringstream ss_cd_error_fname;
		ss_cd_error_fname << try_count << "_error_cd_sparse" << "_epoch" << epoch_count << "_div" << rbm_div << ".error.json";
		write_error_info(db, rbm_gen, rbm_cd, rbm_trainer_cd, dataset, ss_cd_error_fname.str());
	}
}


//
// 生成モデルと学習モデルとのカルバックライブラー情報量を比較
//
int main(void) {

	std::cout << "append h:";
	int append_h;
	std::cin >> append_h;

	OPTION option;
	option.vSize = 5;
	option.hSize = 10;
	option.appendH = append_h;
	option.datasize = 25;
	option.epoch = 1000;
	option.cdk = 1;
	option.batchsize = option.datasize;
	option.learningRate = 0.1;
	option.momentumRate = 0.9;
	option.divSize = 1;
	option.realFlag = false;
	int try_num = 100;


	SQLite::Database db("./output/sqlite3.db", SQLite::OPEN_CREATE | SQLite::OPEN_READWRITE);

	make_table(db);

	for (int try_count = 0; try_count < try_num; try_count++) {
		try {
			db.exec("BEGIN");
		}
		catch (std::exception& e)
		{
			std::cout << "exception: " << e.what() << std::endl;
		}

		auto rbm_gen = GeneralizedRBM(option.vSize, option.hSize);
		rbm_gen.params.initParamsRandom(-2, 2);

		std::vector<std::vector<double>> dataset;
		for (int i = 0; i < option.datasize; i++) {
			dataset.push_back(rbmutil::data_gen<GeneralizedRBM, std::vector<double> >(rbm_gen, option.vSize));
		}

		//std::cout << "[Generative Model]" << std::endl;
		//rbmutil::print_params(rbm_gen);
		std::stringstream ss_gen_fname;
		ss_gen_fname << try_count << "_gen.rbm.json";
		write_params(db, rbm_gen, ss_gen_fname.str());

		// try rbm 2, 3, 4, 5, cont
		option.realFlag = false;
		option.divSize = 1;
		run(db, option, try_count, rbm_gen, dataset);

		option.realFlag = false;
		option.divSize = 2;
		run(db, option, try_count, rbm_gen, dataset);

		option.realFlag = false;
		option.divSize = 3;
		run(db, option, try_count, rbm_gen, dataset);

		option.realFlag = false;
		option.divSize = 4;
		run(db, option, try_count, rbm_gen, dataset);

		option.realFlag = true;
		run(db, option, try_count, rbm_gen, dataset);

		// SparseRBM
		option.realFlag = false;
		option.divSize = 2;
		run_sparse(db, option, try_count, rbm_gen, dataset);

		option.realFlag = false;
		option.divSize = 3;
		run_sparse(db, option, try_count, rbm_gen, dataset);

		option.realFlag = false;
		option.divSize = 4;
		run_sparse(db, option, try_count, rbm_gen, dataset);

		try{
    		db.exec("COMMIT");
			std::cout << "h10 + " << option.appendH << ", cimmit: " << try_count << std::endl;
		}
		catch (std::exception& e)
		{
			std::cout << "exception: " << e.what() << std::endl;
		}
	}


	return 0;
}
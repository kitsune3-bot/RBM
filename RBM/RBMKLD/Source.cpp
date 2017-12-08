#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "rbmutil.h"
#include "RBMCore.h"
#include "Trainer.h"
#include "Sampler.h"
#include "SQLiteCpp/SQLiteCpp.h"
#include "sqlite3.h"
#include <omp.h>

typedef struct {
	int vSize = 5;
	int hSize = 10;
	int appendH = 0;
	int datasize = 100;
	int epoch = 1000;
	int cdk = 1;
	int batchsize = datasize;
	double learningRate = 0.1;
	double momentumRate = 0.9;
	int divSize = 1;
	bool realFlag = false;
	int runFlag = 0; // 1: exact, 2: cd, 3:exact & sparse
} OPTION;

typedef struct {
	double kld;
	double loglikelihood;
	int data_size;
	int  v_size;
	int  h_size;
	std::string rbm_type;
	int div_size;
	std::string train_type;
	int epoch;
	int sparse;
	int try_count;
} RESULT;

/**
// SQLITE3 データベース仕様(仮)
// +-------------+------+-------------+
// |             datafile               |
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

/**
// SQLITE3 データベース仕様(仮)
// +----------------------------------+
// |              result              |
// +----------------------------------+
//
//   uid INTEGER PRIMARY KEY AUTOINCREMENT
//   kld REAL
//   loglikelihood REAL
//   v_size INTEGER
//   h_size INTEGER
//   rbm_type TEXT (d or c)
//   div_size INTEGER
//   train_type (cd or exact)
//   epoch INTEGER
//   sparse INTEGER (has sparse: 1)
//   try_count INTEGER
**/


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

	if (db.tableExists("datafile")) {
		std::string query = "DROP TABLE datafile;";
		db.exec(query);
	}

	// make train_info table
	std::string query = "CREATE TABLE datafile (uid INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, json BLOB, time TEXT);";
	db.exec(query);

	// make result table
	//   uid INTEGER PRIMARY KEY AUTOINCREMENT
	//   kld REAL
	//   loglikelihood REAL
	//   data_size INTEGER
	//   v_size INTEGER
	//   h_size INTEGER
	//   rbm_type TEXT (d or c)
	//   div_size INTEGER
	//   train_type (cd or exact)
	//   epoch INTEGER
	//   sparse INTEGER (has sparse: 1)
	//   try_count INTEGER
	std::stringstream ss_query;
	ss_query << "CREATE TABLE result ("
		<< "uid INTEGER PRIMARY KEY AUTOINCREMENT"
		<< ", kld REAL"
		<< ", loglikelihood REAL"
		<< ", data_size INTEGER"
		<< ", v_size INTEGER"
		<< ", h_size INTEGER"
		<< ", rbm_type TEXT"
		<< ", div_size INTEGER"
		<< ", train_type"
		<< ", epoch INTEGER"
		<< ", sparse INTEGER"
		<< ", try_count INTEGER)";
		db.exec(ss_query.str());

	try {
		db.exec("COMMIT");
	}
	catch (std::exception& e)
	{
		std::cout << "exception: " << e.what() << std::endl;
	}

}



void write_to_db_data_table(SQLite::Database & db, std::string & name, std::string & json) {
	try {
		std::stringstream ss_query;
		ss_query << "INSERT INTO datafile(name, json, time) VALUES(?, ?, datetime(CURRENT_TIMESTAMP,'localtime'));";
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

void write_to_db_result_table(SQLite::Database & db, RESULT & result) {
	try {
		//   kld REAL
		//   loglikelihood REAL
		//   data_size
		//   v_size INTEGER
		//   h_size INTEGER
		//   rbm_type TEXT (d or c)
		//   div_size INTEGER
		//   train_type (cd or exact)
		//   epoch INTEGER
		//   sparse (has sparse: 1)
		//   try_count INGERGER
		std::stringstream ss_query;
		//	ss_query << "INSERT INTO result (name, json, time) VALUES("  << name << ", " << json << ", )";
		ss_query << "INSERT INTO result(kld, loglikelihood, data_size, v_size, h_size, rbm_type, div_size, train_type, epoch, sparse, try_count) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";
		std::string query = ss_query.str();

		SQLite::Statement state(db, query);
		state.bind(1, result.kld);
		state.bind(2, result.loglikelihood);
		state.bind(3, result.data_size);
		state.bind(4, result.v_size);
		state.bind(5, result.h_size);
		state.bind(6, result.rbm_type);
		state.bind(7, result.div_size);
		state.bind(8, result.train_type);
		state.bind(9, result.epoch);
		state.bind(10, result.sparse);
		state.bind(11, result.try_count);
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
	write_to_db_data_table(db, fname, rbm.params.serialize());
}

// 学習情報書き込み(SQLITEに)
template <class RBM, class Trainer>
void write_train_info(SQLite::Database & db, RBM & rbm, Trainer & trainer, std::string fname) {
	auto js = trainer.trainInfoJson(rbm);
	write_to_db_data_table(db, fname, js);
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
template<class RBM_G, class RBM_T, class DATASET>
void run(SQLite::Database & db, OPTION & option, int try_count, RBM_G & rbm_gen, RBM_T & rbm_train, DATASET & dataset) {
	if (!(option.runFlag & 1)) return;


	auto rbm_exact = rbm_train;
	rbm_exact.params.initParamsXavier();
	rbm_exact.setHiddenDiveSize(option.divSize);
	rbm_exact.setHiddenMin(-1.0);
	rbm_exact.setHiddenMax(1.0);
	rbm_exact.setRealHiddenValue(option.realFlag);

	auto rbm_trainer_exact = Trainer<GeneralizedRBM, OptimizerType::AdaMax>(rbm_exact);
	rbm_trainer_exact.epoch = option.epoch;
	rbm_trainer_exact.cdk = option.cdk;
	rbm_trainer_exact.batchSize = option.batchsize;
	rbm_trainer_exact.learningRate = option.learningRate;

	auto rbm_cd = rbm_train;
	rbm_cd.params.initParamsRandom(-0.01, 0.01);
	rbm_cd.setHiddenDiveSize(option.divSize);
	rbm_cd.setHiddenMin(-1.0);
	rbm_cd.setHiddenMax(1.0);
	rbm_cd.setRealHiddenValue(option.realFlag);

	auto rbm_trainer_cd = Trainer<GeneralizedRBM, OptimizerType::AdaMax>(rbm_cd);
	rbm_trainer_cd.epoch = option.epoch;
	rbm_trainer_cd.cdk = option.cdk;
	rbm_trainer_cd.batchSize = option.batchsize;
	rbm_trainer_cd.learningRate = option.learningRate;

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
		RESULT result;

		// Exact
		std::string rbm_div = option.realFlag ? "c" : std::to_string(option.divSize);

		rbm_trainer_exact.trainOnceExact(rbm_exact, dataset);
		std::stringstream ss_exact_fname;
		ss_exact_fname << try_count << "_exact" << "_epoch" << epoch_count << "_div" << rbm_div << ".train.json";
		//write_train_info(db, rbm_exact, rbm_trainer_exact, ss_exact_fname.str());

		std::stringstream ss_exact_error_fname;
		ss_exact_error_fname << try_count << "_error_exact" << "_epoch" << epoch_count << "_div" << rbm_div << ".error.json";
	
		result.kld = rbmutil::kld(rbm_gen, rbm_exact, std::vector<int>{0, 1});
		result.loglikelihood = rbm_trainer_exact.logLikeliHood(rbm_exact, dataset);
		result.data_size = dataset.size();
		result.v_size = rbm_exact.getVisibleSize();
		result.h_size = rbm_exact.getHiddenSize();
		result.rbm_type = rbm_exact.isRealHiddenValue() ? "c" : "d";
		result.div_size = rbm_exact.getHiddenDivSize();
		result.train_type = "exact";
		result.epoch = epoch_count;
		result.sparse = 0;
		result.try_count = try_count;
		write_to_db_result_table(db, result);

		// Contrastive Divergence
		rbm_trainer_cd.trainOnceCD(rbm_cd, dataset);
		std::stringstream ss_cd_fname;
		ss_cd_fname << try_count << "_cd" << "_epoch" << epoch_count << "_div" << rbm_div << ".train.json";
		//write_train_info(db, rbm_cd, rbm_trainer_cd, ss_cd_fname.str());

		std::stringstream ss_cd_error_fname;
		ss_cd_error_fname << try_count << "_error_cd" << "_epoch" << epoch_count << "_div" << rbm_div << ".error.json";

		result.kld = rbmutil::kld(rbm_gen, rbm_cd, std::vector<int>{0, 1});
		result.loglikelihood = rbm_trainer_cd.logLikeliHood(rbm_cd, dataset);
		result.data_size = dataset.size();
		result.v_size = rbm_cd.getVisibleSize();
		result.h_size = rbm_cd.getHiddenSize();
		result.rbm_type = rbm_cd.isRealHiddenValue() ? "c" : "d";
		result.div_size = rbm_cd.getHiddenDivSize();
		result.train_type = "cd";
		result.epoch = epoch_count;
		result.sparse = 0;
		result.try_count = try_count;
		write_to_db_result_table(db, result);
	}
}

// 実行ルーチン
template<class RBM_G, class RBM_T, class DATASET>
void run_sparse(SQLite::Database & db, OPTION & option, int try_count, RBM_G & rbm_gen, RBM_T & rbm_train, DATASET & dataset) {
	if (!(option.runFlag & 2)) return;

	auto rbm_exact = rbm_train;
	rbm_exact.params.sparse.setConstant(4.0);
	rbm_exact.params.initParamsXavier();
	rbm_exact.setHiddenDiveSize(option.divSize);
	rbm_exact.setHiddenMin(-1.0);
	rbm_exact.setHiddenMax(1.0);
	rbm_exact.setRealHiddenValue(option.realFlag);

	auto rbm_trainer_exact = Trainer<GeneralizedSparseRBM, OptimizerType::AdaMax>(rbm_exact);
	rbm_trainer_exact.epoch = option.epoch;
	rbm_trainer_exact.cdk = option.cdk;
	rbm_trainer_exact.batchSize = option.batchsize;
	rbm_trainer_exact.learningRate = option.learningRate;

	auto rbm_cd = rbm_train;
	rbm_exact.params.sparse.setRandom() *= 0.5;
	rbm_cd.params.initParamsRandom(-0.01, 0.01);
	rbm_cd.setHiddenDiveSize(option.divSize);
	rbm_cd.setHiddenMin(-1.0);
	rbm_cd.setHiddenMax(1.0);
	rbm_cd.setRealHiddenValue(option.realFlag);

	auto rbm_trainer_cd = Trainer<GeneralizedSparseRBM, OptimizerType::AdaMax>(rbm_cd);
	rbm_trainer_cd.epoch = option.epoch;
	rbm_trainer_cd.cdk = option.cdk;
	rbm_trainer_cd.batchSize = option.batchsize;
	rbm_trainer_cd.learningRate = option.learningRate;

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
		RESULT result;

		// Exact
		std::string rbm_div = option.realFlag ? "c" : std::to_string(option.divSize);

		rbm_trainer_exact.trainOnceExact(rbm_exact, dataset);
		std::stringstream ss_exact_fname;
		ss_exact_fname << try_count << "_exact_sparse" << "_epoch" << epoch_count << "_div" << rbm_div << ".train.json";
		//write_train_info(db, rbm_exact, rbm_trainer_exact, ss_exact_fname.str());

		result.kld = rbmutil::kld(rbm_gen, rbm_exact, std::vector<int>{0, 1});
		result.loglikelihood = rbm_trainer_exact.logLikeliHood(rbm_exact, dataset);
		result.data_size = dataset.size();
		result.v_size = rbm_exact.getVisibleSize();
		result.h_size = rbm_exact.getHiddenSize();
		result.rbm_type = rbm_exact.isRealHiddenValue() ? "c" : "d";
		result.div_size = rbm_exact.getHiddenDivSize();
		result.train_type = "exact";
		result.epoch = epoch_count;
		result.sparse = 1;
		result.try_count = try_count;
		write_to_db_result_table(db, result);

		// Contrastive Divergence
		rbm_trainer_cd.trainOnceCD(rbm_cd, dataset);
		std::stringstream ss_cd_fname;
		ss_cd_fname << try_count << "_cd_sparse" << "_epoch" << epoch_count << "_div" << rbm_div << ".train.json";
		//write_train_info(db, rbm_cd, rbm_trainer_cd, ss_cd_fname.str());

		result.kld = rbmutil::kld(rbm_gen, rbm_cd, std::vector<int>{0, 1});
		result.loglikelihood = rbm_trainer_cd.logLikeliHood(rbm_cd, dataset);
		result.data_size = dataset.size();
		result.v_size = rbm_cd.getVisibleSize();
		result.h_size = rbm_cd.getHiddenSize();
		result.rbm_type = rbm_cd.isRealHiddenValue() ? "c" : "d";
		result.div_size = rbm_cd.getHiddenDivSize();
		result.train_type = "cd";
		result.epoch = epoch_count;
		result.sparse = 1;
		result.try_count = try_count;
		write_to_db_result_table(db, result);
	}
}

//
// 生成モデルと学習モデルとのカルバックライブラー情報量を比較
//
int main(void) {

	std::cout << "append h:";
	int append_h;
	std::cin >> append_h;
	int datasize;

	std::cout << "datasize:";
	std::cin >> datasize;

	int epoch;
	std::cout << "epoch:";
	std::cin >> epoch;

	int run_flag;
	std::cout << "run flag(1: exact, 2: cd, 3: exact & cd):";
	std::cin >> run_flag;

	OPTION option;
	option.vSize = 8;
	option.hSize = 5;
	option.appendH = append_h;
	option.datasize = datasize;
	option.epoch = epoch;
	option.cdk = 1;
	option.batchsize = option.datasize;
	option.learningRate = 0.1;
	option.divSize = 1;
	option.realFlag = false;
	option.runFlag = run_flag;
	int try_num = 1000;


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
		rbm_gen.params.initParamsXavier();

		std::vector<std::vector<double>> dataset;
		for (int i = 0; i < option.datasize; i++) {
			dataset.push_back(rbmutil::data_gen<GeneralizedRBM, std::vector<double> >(rbm_gen, option.vSize));
		}

		//std::cout << "[Generative Model]" << std::endl;
		//rbmutil::print_params(rbm_gen);
		std::stringstream ss_gen_fname;
		ss_gen_fname << try_count << "_gen.rbm.json";
		//write_params(db, rbm_gen, ss_gen_fname.str());

		auto rbm_train = GeneralizedRBM(option.vSize, option.hSize + option.appendH);

		// try rbm 2, 3, 4, 5, cont
		option.realFlag = false;
		option.divSize = 1;
		run(db, option, try_count, rbm_gen, rbm_train, dataset);

		option.realFlag = false;
		option.divSize = 2;
		run(db, option, try_count, rbm_gen, rbm_train, dataset);

		option.realFlag = false;
		option.divSize = 3;
		run(db, option, try_count, rbm_gen, rbm_train, dataset);

		option.realFlag = false;
		option.divSize = 4;
		run(db, option, try_count, rbm_gen, rbm_train, dataset);

		option.realFlag = true;
		run(db, option, try_count, rbm_gen, rbm_train, dataset);

		// SparseRBM
		auto rbm_train_sparse = GeneralizedSparseRBM(option.vSize, option.hSize + option.appendH);

		option.realFlag = false;
		option.divSize = 2;
		run_sparse(db, option, try_count, rbm_gen, rbm_train_sparse, dataset);

		option.realFlag = false;
		option.divSize = 3;
		run_sparse(db, option, try_count, rbm_gen, rbm_train_sparse, dataset);

		option.realFlag = false;
		option.divSize = 4;
		run_sparse(db, option, try_count, rbm_gen, rbm_train_sparse, dataset);

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
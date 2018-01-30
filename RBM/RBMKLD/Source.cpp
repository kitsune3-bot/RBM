#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <boost/program_options.hpp>
#include "rbmutil.h"
#include "RBMCore.h"
#include "Trainer.h"
#include "Sampler.h"
#include "SQLiteCpp/SQLiteCpp.h"
#include "sqlite3.h"


#include "mysql_driver.h"
#include "mysql_connection.h"
#include "mysql_error.h"
#include "cppconn/Statement.h"
#include "cppconn/prepared_statement.h"
#include "cppconn/ResultSet.h"

typedef struct {
	int vSize = 5;
	int genHsize = 0;
	int trainHsize = 0;
	int datasize = 100;
	int epoch = 1000;
	int cdk = 1;
	int batchsize = datasize;
	double learningRate = 0.1;
	double momentumRate = 0.9;
	int divSize = 1;
	bool realFlag = false;
	int trainFlag = 0; // 1: exact, 2: cd, 3:exact & cd
	int rbmFlag = 0;   // 1: normal, 2: sparse, 3:normal & sparse
	int seed = 0;
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
	int seed = 0;
} RESULT;

typedef struct {
	bool isSetValue = false;
	std::string valuesQuery;
} QUERY;


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


void make_table(sql::Connection * con);
QUERY make_sql_result_insert();
QUERY & make_sql_result_values(QUERY & query, RESULT & result);
void write_to_db_data_table(sql::Connection * con, std::string & name, std::string & json);
void write_to_db_result_table(sql::Connection *con, RESULT & result, OPTION & option);
void write_to_db_result_table(sql::Connection *con, std::string & query);

// パラメータファイル書き込み(SQLITEに)
template <class RBM>
void write_params(sql::Connection * con, RBM & rbm, std::string fname);

// 学習情報書き込み(SQLITEに)
template <class RBM, class Trainer>
void write_train_info(sql::Connection * con, RBM & rbm, Trainer & trainer, std::string fname);

// (汎化|訓練)誤差情報書き込み(SQLITEに)
template <class RBMGEN, class RBMTRAIN, class Trainer, class Dataset>
void write_error_info(sql::Connection * con, RBMGEN & rbm_gen, RBMTRAIN & rbm_train, Trainer & trainer, Dataset & dataset, std::string fname);

// 実行ルーチン
template<class RBM_G, class RBM_T, class DATASET>
void run(sql::Connection * con, OPTION & option, int try_count, RBM_G & rbm_gen, RBM_T & rbm_train, DATASET & dataset);

template<class RBM_G, class RBM_T, class DATASET>
void run_sparse(sql::Connection * con, OPTION & option, int try_count, RBM_G & rbm_gen, RBM_T & rbm_train, DATASET & dataset);

// コマンドラインオプションの設定
// 対話するかしないかも。
OPTION get_option(int argc, char** argv);


void make_table(sql::Connection *con) {
	auto stmt = con->createStatement();

	try {
		stmt->execute("BEGIN TRANSACTION");
	}
	catch (std::exception& e)
	{
		std::cout << "exception: " << e.what() << std::endl;
	}

	char *err_msg = NULL;
	//if (stmt->tableExists("result")) {
	//	std::string query = "DROP TABLE result;";
	//	stmt->execute(query);
	//}

	//if (stmt->tableExists("datafile")) {
	//	std::string query = "DROP TABLE datafile;";
	//	stmt->execute(query);
	//}

	// make train_info table
	std::string query = "CREATE TABLE datafile (uid INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, json BLOB, time TEXT);";
	stmt->execute(query);

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
	stmt->execute(ss_query.str());

	try {
		stmt->execute("COMMIT");
	}
	catch (std::exception& e)
	{
		std::cout << "exception: " << e.what() << std::endl;
	}

	delete stmt;

}


QUERY make_sql_result_insert() {
	QUERY query;
	query.valuesQuery = "INSERT INTO result(kld, loglikelihood, data_size, v_size, h_size, rbm_type, div_size, train_type, epoch, sparse, try_count, seed_num) VALUES";
	query.isSetValue = false;

	return query;
}
QUERY & make_sql_result_values(QUERY & query, RESULT & result) {

	std::string values_query = "";
	if (query.isSetValue) {
		values_query = ", ";
	}

	values_query += "(";
	values_query += std::to_string(result.kld);
	values_query += ", " + std::to_string(result.loglikelihood);
	values_query += ", " + std::to_string(result.data_size);
	values_query += ", " + std::to_string(result.v_size);
	values_query += ", " + std::to_string(result.h_size);
	values_query += ", " + std::string("\"") + (result.rbm_type) + std::string("\"");
	values_query += ", " + std::to_string(result.div_size);
	values_query += ", " + std::string("\"") + (result.train_type) + std::string("\"");
	values_query += ", " + std::to_string(result.epoch);
	values_query += ", " + std::to_string(result.sparse);
	values_query += ", " + std::to_string(result.try_count);
	values_query += ", " + std::to_string(result.seed);
	values_query += ")";
	query.valuesQuery += values_query;
	query.isSetValue = true;

	return query;
}



void write_to_db_data_table(sql::Connection *con, std::string & name, std::string & json) {

	try {
		std::stringstream ss_query;
		ss_query << "INSERT INTO datafile(name, json, time) VALUES(?, ?, datetime(CURRENT_TIMESTAMP,'localtime'));";
		std::string query = ss_query.str();

		auto *prep_stmt = con->prepareStatement(query);
		prep_stmt->setString(1, name);
		prep_stmt->setString(2, json);
		prep_stmt->execute();
		//if (ret != sqlite_ok) {
		//	throw;
		//}

		delete prep_stmt;
	}
	catch (std::exception& e)
	{
		std::cout << "exception: " << e.what() << std::endl;
	}
}

void write_to_db_result_table(sql::Connection *con, RESULT & result, OPTION & option) {
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
		ss_query << "INSERT INTO result(kld, loglikelihood, data_size, v_size, h_size, rbm_type, div_size, train_type, epoch, sparse, try_count, seed_num) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";
		std::string query = ss_query.str();

		// TODO: ほんとにtry_count??? option読まなくていいの???
		auto prep_stmt = con->prepareStatement(query);
		prep_stmt->setDouble(1, result.kld);
		prep_stmt->setDouble(2, result.loglikelihood);
		prep_stmt->setInt(3, result.data_size);
		prep_stmt->setInt(4, result.v_size);
		prep_stmt->setInt(5, result.h_size);
		prep_stmt->setString(6, result.rbm_type);
		prep_stmt->setInt(7, result.div_size);
		prep_stmt->setString(8, result.train_type);
		prep_stmt->setInt(9, result.epoch);
		prep_stmt->setInt(10, result.sparse);
		prep_stmt->setInt(11, result.try_count);
		prep_stmt->setInt(12, option.seed);
		prep_stmt->execute();
		//if (ret != sqlite_ok) {
		//	throw;
		//}

		delete prep_stmt;
	}
	catch (std::exception& e)
	{
		std::cout << "exception: " << e.what() << std::endl;
	}

}

void write_to_db_result_table(sql::Connection *con, std::string & query) {
	try {
		auto stmt = con->createStatement();
		stmt->execute(query);
	}
	catch (std::exception& e)
	{
		std::cout << "exception: " << e.what() << std::endl;
	}

}



// パラメータファイル書き込み(SQLITEに)
template <class RBM>
void write_params(sql::Connection *con, RBM & rbm, std::string fname) {
	write_to_db_data_table(db, fname, rbm.params.serialize());
}

// 学習情報書き込み(SQLITEに)
template <class RBM, class Trainer>
void write_train_info(sql::Connection *con, RBM & rbm, Trainer & trainer, std::string fname) {
	auto js = trainer.trainInfoJson(rbm);
	write_to_db_data_table(db, fname, js);
}

// (汎化|訓練)誤差情報書き込み(SQLITEに)
template <class RBMGEN, class RBMTRAIN, class Trainer, class Dataset>
void write_error_info(sql::Connection *con, RBMGEN & rbm_gen, RBMTRAIN & rbm_train, Trainer & trainer, Dataset & dataset, std::string fname) {
	auto js = nlohmann::json();
	js["kld"] = rbmutil::kld(rbm_gen, rbm_train, std::vector<int>{0, 1});
	js["logLikelihood"] = trainer.logLikeliHood(rbm_train, dataset);

	write_to_db(db, fname, js.dump());
}

// 実行ルーチン
template<class RBM_G, class RBM_T, class DATASET>
void run(sql::Connection *con, OPTION & option, int try_count, RBM_G & rbm_gen, RBM_T & rbm_train, DATASET & dataset) {
	if (!(option.rbmFlag == 0)) return;

	std::mt19937 random_device(option.seed);
	auto stmt = con->createStatement();
	stmt->execute("START TRANSACTION");
	QUERY query = make_sql_result_insert();

	auto rbm_exact = rbm_train;
	rbm_exact.params.initParamsXavier(option.seed);
	rbm_exact.setHiddenDivSize(option.divSize);
	rbm_exact.setHiddenMin(-1.0);
	rbm_exact.setHiddenMax(1.0);
	rbm_exact.setRealHiddenValue(option.realFlag);

	auto rbm_trainer_exact = Trainer<GeneralizedRBM, OptimizerType::AdaMax>(rbm_exact);
	rbm_trainer_exact.epoch = option.epoch;
	rbm_trainer_exact.cdk = option.cdk;
	rbm_trainer_exact.batchSize = option.batchsize;
	rbm_trainer_exact.learningRate = option.learningRate;
	rbm_trainer_exact.randDevice = random_device;

	auto rbm_cd = rbm_train;
	rbm_cd.params.initParamsXavier(option.seed);
	rbm_cd.setHiddenDivSize(option.divSize);
	rbm_cd.setHiddenMin(-1.0);
	rbm_cd.setHiddenMax(1.0);
	rbm_cd.setRealHiddenValue(option.realFlag);

	auto rbm_trainer_cd = Trainer<GeneralizedRBM, OptimizerType::AdaMax>(rbm_cd);
	rbm_trainer_cd.epoch = option.epoch;
	rbm_trainer_cd.cdk = option.cdk;
	rbm_trainer_cd.batchSize = option.batchsize;
	rbm_trainer_cd.learningRate = option.learningRate;
	rbm_trainer_cd.randDevice = random_device;


	for (int epoch_count = 0; epoch_count < option.epoch; epoch_count++) {
		RESULT result;
		result.seed = option.seed;

		std::string rbm_div = option.realFlag ? "c" : std::to_string(option.divSize);




		// Exact
		if (option.trainFlag == 0) {
			rbm_trainer_exact.trainOnceExact(rbm_exact, dataset);
			std::stringstream ss_exact_fname;
			ss_exact_fname << try_count << "_exact" << "_epoch" << epoch_count << "_div" << rbm_div << ".train.json";
			//write_train_info(db, rbm_exact, rbm_trainer_exact, ss_exact_fname.str());

			std::stringstream ss_exact_error_fname;
			ss_exact_error_fname << try_count << "_error_exact" << "_epoch" << epoch_count << "_div" << rbm_div << ".error.json";

			result.kld = rbmutil::kld(rbm_gen, rbm_exact, std::vector<double>{-1.0, 1.0});
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
			//write_to_db_result_table(con, result, option);
			make_sql_result_values(query, result);
		}

		// Contrastive Divergence
		if (option.trainFlag == 1) {
			rbm_trainer_cd.trainOnceCD(rbm_cd, dataset);
			std::stringstream ss_cd_fname;
			ss_cd_fname << try_count << "_cd" << "_epoch" << epoch_count << "_div" << rbm_div << ".train.json";
			//write_train_info(db, rbm_cd, rbm_trainer_cd, ss_cd_fname.str());

			std::stringstream ss_cd_error_fname;
			ss_cd_error_fname << try_count << "_error_cd" << "_epoch" << epoch_count << "_div" << rbm_div << ".error.json";

			result.kld = rbmutil::kld(rbm_gen, rbm_cd, std::vector<double>{-1.0, 1.0});
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
			//			write_to_db_result_table(con, result, option);
			make_sql_result_values(query, result);
		}
	}
	write_to_db_result_table(con, query.valuesQuery);
	stmt->execute("COMMIT;");
	delete stmt;
}

// 実行ルーチン
template<class RBM_G, class RBM_T, class DATASET>
void run_sparse(sql::Connection *con, OPTION & option, int try_count, RBM_G & rbm_gen, RBM_T & rbm_train, DATASET & dataset) {
	if (!(option.rbmFlag == 1)) return;
	std::mt19937 random_device(option.seed);
	auto stmt = con->createStatement();
	stmt->execute("START TRANSACTION");
	QUERY query = make_sql_result_insert();


	auto rbm_exact = rbm_train;
	//	rbm_exact.params.sparse.setConstant(4.0);
	rbm_exact.params.initParamsXavier(option.seed);
	rbm_exact.setHiddenMin(-1.0);
	rbm_exact.setHiddenMax(1.0);
	rbm_exact.setHiddenDivSize(option.divSize);
	rbm_exact.setRealHiddenValue(option.realFlag);

	auto rbm_trainer_exact = Trainer<GeneralizedSparseRBM, OptimizerType::AdaMax>(rbm_exact);
	rbm_trainer_exact.epoch = option.epoch;
	rbm_trainer_exact.cdk = option.cdk;
	rbm_trainer_exact.batchSize = option.batchsize;
	rbm_trainer_exact.learningRate = option.learningRate;
	rbm_trainer_exact.randDevice = random_device;


	auto rbm_cd = rbm_train;
	//	rbm_exact.params.sparse.setRandom() *= 0.5;
	rbm_cd.params.initParamsXavier(option.seed);
	rbm_cd.setHiddenMin(-1.0);
	rbm_cd.setHiddenMax(1.0);
	rbm_cd.setHiddenDivSize(option.divSize);
	rbm_cd.setRealHiddenValue(option.realFlag);

	auto rbm_trainer_cd = Trainer<GeneralizedSparseRBM, OptimizerType::AdaMax>(rbm_cd);
	rbm_trainer_cd.epoch = option.epoch;
	rbm_trainer_cd.cdk = option.cdk;
	rbm_trainer_cd.batchSize = option.batchsize;
	rbm_trainer_cd.learningRate = option.learningRate;
	rbm_trainer_cd.randDevice = random_device;



	for (int epoch_count = 0; epoch_count < option.epoch; epoch_count++) {
		RESULT result;
		result.seed = option.seed;

		std::string rbm_div = option.realFlag ? "c" : std::to_string(option.divSize);

		// Exact
		if (option.trainFlag == 0) {
			rbm_trainer_exact.trainOnceExact(rbm_exact, dataset);
			std::stringstream ss_exact_fname;
			ss_exact_fname << try_count << "_exact_sparse" << "_epoch" << epoch_count << "_div" << rbm_div << ".train.json";
			//write_train_info(db, rbm_exact, rbm_trainer_exact, ss_exact_fname.str());

			result.kld = rbmutil::kld(rbm_gen, rbm_exact, std::vector<double>{-1.0, 1.0});
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
			//			write_to_db_result_table(con, result, option);
			make_sql_result_values(query, result);
		}

		// Contrastive Divergence
		if (option.trainFlag == 1) {
			rbm_trainer_cd.trainOnceCD(rbm_cd, dataset);
			std::stringstream ss_cd_fname;
			ss_cd_fname << try_count << "_cd_sparse" << "_epoch" << epoch_count << "_div" << rbm_div << ".train.json";
			//write_train_info(db, rbm_cd, rbm_trainer_cd, ss_cd_fname.str());

			result.kld = rbmutil::kld(rbm_gen, rbm_cd, std::vector<double>{-1.0, 1.0});
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
			//write_to_db_result_table(con, result, option);
			make_sql_result_values(query, result);
		}
	}
	write_to_db_result_table(con, query.valuesQuery);
	stmt->execute("COMMIT;");
	delete stmt;
}


OPTION get_option(int argc, char** argv) {
	namespace po = boost::program_options;
	po::options_description opt("オプション");
	opt.add_options()
		("help,h", "ヘルプを表示")
		("vsize", po::value<int>()->default_value(4), "visible node size")
		("gen_hsize", po::value<int>()->default_value(4), "hidden node size(gen rbm)")
		("train_hsize", po::value<int>()->default_value(4), "hidden node size(train rbm)")
		("datasize", po::value<int>()->default_value(100), "gen datasize")
		("epoch", po::value<int>()->default_value(1000), "update count")
		("cdk", po::value<int>()->default_value(1), "cdk")
		//("batchsize", po::value<int>(), "batchsize")
		("learning_rate", po::value<double>()->default_value(0.1), "")
		("momentum_rate", po::value<double>()->default_value(0.9), "")
		("divsize", po::value<int>()->default_value(2), "0: real, others: digit")
		("train_mode", po::value<int>(), "0: Exact, 1: CD")
		("rbmtype", po::value<int>(), "0: RBM, 1: SRBM")
		("seed", po::value<int>()->default_value(0), "seed_value");


	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, opt), vm);
	po::notify(vm);


	if (vm.count("help")) {
		std::cout << opt << std::endl;
		exit(0);
	}

	OPTION option;
	try {
		option.vSize = vm["vsize"].as<int>();
		option.genHsize = vm["gen_hsize"].as<int>();
		option.trainHsize = vm["train_hsize"].as<int>();
		option.datasize = vm["datasize"].as<int>();
		option.epoch = vm["epoch"].as<int>();
		option.cdk = vm["cdk"].as<int>();
		option.batchsize = option.datasize;
		option.learningRate = vm["learning_rate"].as<double>();
		option.divSize = vm["divsize"].as<int>();
		option.realFlag = false;
		option.trainFlag = vm["train_mode"].as<int>();
		option.rbmFlag = vm["rbmtype"].as<int>();
		option.seed = vm["seed"].as<int>();

		if (option.divSize < 1) option.realFlag = true;
	}
	catch (std::exception& e)
	{
		std::cout << "exception: " << e.what() << std::endl;
		std::cout << opt << std::endl;
		exit(-1l);
	}

	return option;
}

//
// 生成モデルと学習モデルとのカルバックライブラー情報量を比較
//
int main(int argc, char** argv) {
	OPTION option = get_option(argc, argv);
	int try_num = 1;


	//SQLite::Database db("./output/sqlite3.db", SQLite::OPEN_CREATE | SQLite::OPEN_READWRITE);

	//make_table(db);


	auto mysql_host = std::string(std::getenv("KLDMYSQL_HOST"));
	auto mysql_user = std::string(std::getenv("KLDMYSQL_USER"));
	auto mysql_passwd = std::string(std::getenv("KLDMYSQL_PASSWD"));
	std::cout << mysql_user << "@" << mysql_host << std::endl;

	sql::mysql::MySQL_Driver *driver;
	sql::Connection *con;

	driver = sql::mysql::get_mysql_driver_instance();
	sql::ConnectOptionsMap connection_properties;

	connection_properties["hostName"] = mysql_host;
	connection_properties["userName"] = mysql_user;
	connection_properties["password"] = mysql_passwd;
	connection_properties["port"] = 3306;
	connection_properties["OPT_RECONNECT"] = true;
	connection_properties["CLIENT_COMPRESS"] = true;
	connection_properties["OPT_CONNECT_TIMEOUT"] = 60;

	auto url = std::string("tcp://") + mysql_host + std::string(":3306");
	try {
		con = driver->connect(connection_properties);
	}
	catch (std::exception & e) {
		std::cout << "MySQL Connection Error..." << std::endl;
		exit(-1);
	}

	auto stmt = con->createStatement();
	stmt->execute("USE kld");


	for (int try_count = 0; try_count < try_num; try_count++) {

		int seed_rbm_gen = option.seed + 12345;
		auto rbm_gen = GeneralizedRBM(option.vSize, option.genHsize);
		rbm_gen.setHiddenMin(-1.0);
		rbm_gen.setHiddenMax(1.0);
		rbm_gen.setHiddenDivSize(1);
		rbm_gen.params.initParamsXavier(seed_rbm_gen);

		std::vector<std::vector<double>> dataset;
		for (int i = 0; i < option.datasize; i++) {
			dataset.push_back(rbmutil::data_gen<GeneralizedRBM, std::vector<double> >(rbm_gen, option.vSize, seed_rbm_gen));
		}

		//std::cout << "[Generative Model]" << std::endl;
		//rbmutil::print_params(rbm_gen);
		std::stringstream ss_gen_fname;
		ss_gen_fname << try_count << "_gen.rbm.json";
		//write_params(db, rbm_gen, ss_gen_fname.str());

		auto rbm_train = GeneralizedRBM(option.vSize, option.trainHsize);

		// try rbm 2, 3, 4, 5, cont
		run(con, option, try_count, rbm_gen, rbm_train, dataset);


		// SparseRBM
		auto rbm_train_sparse = GeneralizedSparseRBM(option.vSize, option.trainHsize);
		run_sparse(con, option, try_count, rbm_gen, rbm_train_sparse, dataset);

		try {

			//db.exec("COMMIT");
			std::cout << "h" << option.genHsize << "(gen) / h" << option.trainHsize << "(train), cimmit: " << try_count << std::endl;
		}
		catch (std::exception& e)
		{
			std::cout << "exception: " << e.what() << std::endl;
		}
	}

	con->close();

	return 0;
}
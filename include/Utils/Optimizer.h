// main.cpp

#include <string>
#include <iostream>
#include <fstream>
#include "openGA.hpp"
#include "include/ModelAnalyze/ModelAnalyzer.h"

using std::string;
using std::cout;
using std::endl;

struct SplitSolution
{
	int breakpoint1;
	int breakpoint2;
	int breakpoint3;

	string to_string() const;
};


struct SplitVariance
{
	// This is where the results of simulation
	// is stored but not yet finalized.
	double objective1;
};

typedef EA::Genetic<SplitSolution,SplitVariance> GA_Type;
typedef EA::GenerationType<SplitSolution,SplitVariance> Generation_Type;

/// @brief 
/// @param p 
/// @param rnd01 
void init_genes(SplitSolution& p,const std::function<double(void)> &rnd01, ModelAnalyzer &analyzer);

/// @brief 
/// @param p 
/// @param c 
/// @return 
bool eval_solution(const SplitSolution& p, SplitVariance &c, ModelAnalyzer &analyzer);

/// @brief 
/// @param X_base 
/// @param rnd01 
/// @param shrink_scale 
/// @return 
SplitSolution mutate(const SplitSolution& X_base, const std::function<double(void)> &rnd01, double shrink_scale, ModelAnalyzer &analyzer);

/// @brief 
/// @param X1 
/// @param X2 
/// @param rnd01 
/// @return 
SplitSolution crossover(
	const SplitSolution& X1, const SplitSolution& X2, const std::function<double(void)> &rnd01);

/// @brief 
/// @param X 
/// @return 
double calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X);

static std::ofstream output_file;

/// @brief 
/// @param generation_number 
/// @param last_generation 
/// @param best_genes 
void SO_report_generation(int generation_number, const EA::GenerationType<SplitSolution,SplitVariance> &last_generation, const SplitSolution& best_genes);

/// @brief 
/// @param analyzer 
void optimize(ModelAnalyzer analyzer);
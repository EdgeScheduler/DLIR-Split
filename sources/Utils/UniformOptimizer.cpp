#include <string>
#include <iostream>
#include <fstream>
#include "openGA.hpp"
#include "Utils/UniformOptimizer.h"
#include "ModelAnalyze/ModelAnalyzer.h"
#include "Benchmark/evaluate_models.h"

namespace UniformOptimizer
{
	using std::cout;
	using std::endl;
	using std::string;

	typedef EA::Genetic<SplitSolution, SplitVariance> GA_Type;
	typedef EA::GenerationType<SplitSolution, SplitVariance> Generation_Type;

	string SplitSolution::to_string() const
	{
		return string("{") + "breakpoint1:" + std::to_string(breakpoint1) + ", breakpoint2:" + std::to_string(breakpoint2) /* +", breakpoint3:"+std::to_string(breakpoint3)*/ + "}";
	}

	void init_genes(SplitSolution &p, const std::function<double(void)> &rnd01, ModelAnalyzer &analyzer)
	{
		int range = analyzer.size();
		// rnd01() gives a random number in 0~1
		do
		{
			p.breakpoint1 = 0.0 + range * rnd01();
		} while (p.breakpoint1 == 0 || p.breakpoint1 == range - 1);

		do
		{
			p.breakpoint2 = 0.0 + range * rnd01();
		} while (p.breakpoint2 == p.breakpoint1 || p.breakpoint2 == 0 || p.breakpoint2 == range - 1);

		// do
		// {
		//     p.breakpoint3=0.0+range*rnd01();
		// }while(p.breakpoint3 == p.breakpoint2 || p.breakpoint3 == p.breakpoint1 || p.breakpoint3 == 0 || p.breakpoint3 == range - 1);
	}

	bool eval_solution(const SplitSolution &p, SplitVariance &c, ModelAnalyzer &analyzer)
	{
		const int &breakpoint1 = p.breakpoint1;
		const int &breakpoint2 = p.breakpoint2;
		// const int& breakpoint3=p.breakpoint3;

		// analyzer.SplitAndStoreChilds({analyzer[breakpoint1], analyzer[breakpoint2], analyzer[breakpoint3]});
		analyzer.SplitAndStoreChilds({analyzer[breakpoint1], analyzer[breakpoint2]});
		c.objective1 = evam::EvalStdCurrentModelSplit(analyzer.getName());
		// evam::EvalStdCurrentModelSplit(analyzer.getName(), analyzer.getName());
		return true; // solution is accepted
	}

	SplitSolution mutate(const SplitSolution &X_base, const std::function<double(void)> &rnd01, double shrink_scale, ModelAnalyzer &analyzer)
	{
		SplitSolution X_new;
		const double mu = 0.2 * shrink_scale; // mutation radius (adjustable)
		bool in_range;
		int range = analyzer.size();
		do
		{
			in_range = true;
			X_new = X_base;
			X_new.breakpoint1 += mu * (rnd01() - rnd01());
			in_range = in_range && (X_new.breakpoint1 >= 0.0 && X_new.breakpoint1 < range);
			X_new.breakpoint2 += mu * (rnd01() - rnd01());
			in_range = in_range && (X_new.breakpoint2 >= 0.0 && X_new.breakpoint2 < range);
			// X_new.breakpoint3+=mu*(rnd01()-rnd01());
			// in_range=in_range&&(X_new.breakpoint3>=0.0 && X_new.breakpoint3<range);
		} while (!in_range);
		return X_new;
	}

	SplitSolution crossover(const SplitSolution &X1, const SplitSolution &X2, const std::function<double(void)> &rnd01)
	{
		SplitSolution X_new;
		double r;
		r = rnd01();
		X_new.breakpoint1 = r * X1.breakpoint1 + (1.0 - r) * X2.breakpoint1;
		r = rnd01();
		X_new.breakpoint2 = r * X1.breakpoint2 + (1.0 - r) * X2.breakpoint2;
		// r=rnd01();
		// X_new.breakpoint3=r*X1.breakpoint3+(1.0-r)*X2.breakpoint3;
		return X_new;
	}

	double calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X)
	{
		// finalize the cost
		double final_cost = 0.0;
		final_cost += X.middle_costs.objective1;
		return final_cost;
	}

	void SO_report_generation(int generation_number, const EA::GenerationType<SplitSolution, SplitVariance> &last_generation, const SplitSolution &best_genes)
	{
		cout
			<< "Generation [" << generation_number << "], "
			<< "Best=" << last_generation.best_total_cost << ", "
			<< "Average=" << last_generation.average_cost << ", "
			<< "Best genes=(" << best_genes.to_string() << ")"
			<< ", "
			<< "Exe_time=" << last_generation.exe_time
			<< endl;

		output_file
			<< generation_number << "\t"
			<< last_generation.average_cost << "\t"
			<< last_generation.best_total_cost << "\t"
			<< best_genes.to_string() << "\n";
	}

	void optimize(ModelAnalyzer& analyzer)
	{
		output_file.open("results.txt");
		output_file << "step"
					<< "\t"
					<< "cost_avg"
					<< "\t"
					<< "cost_best"
					<< "\t"
					<< "solution_best"
					<< "\n";

		EA::Chronometer timer;
		timer.tic();

		GA_Type ga_obj = GA_Type(analyzer);
		// ga_obj.analyzer=analyzer;
		ga_obj.problem_mode = EA::GA_MODE::SOGA;
		ga_obj.multi_threading = false;
		ga_obj.idle_delay_us = 10; // switch between threads quickly
		ga_obj.dynamic_threading = true;
		ga_obj.verbose = false;
		ga_obj.population = 200;
		ga_obj.generation_max = 1000;
		ga_obj.calculate_SO_total_fitness = calculate_SO_total_fitness;
		ga_obj.init_genes = init_genes;
		ga_obj.eval_solution = eval_solution;
		ga_obj.mutate = mutate;
		ga_obj.crossover = crossover;
		ga_obj.SO_report_generation = SO_report_generation;
		ga_obj.crossover_fraction = 0.7;
		ga_obj.mutation_rate = 0.2;
		ga_obj.best_stall_max = 10;
		ga_obj.elite_count = 10;
		ga_obj.solve();

		cout << "The problem is optimized in " << timer.toc() << " seconds." << endl;

		output_file.close();
	}
}
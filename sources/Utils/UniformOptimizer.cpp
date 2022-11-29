#include <string>
#include <iostream>
#include <fstream>
#include "openGA.hpp"
#include "Utils/UniformOptimizer.h"
#include "ModelAnalyze/ModelAnalyzer.h"
#include "Benchmark/evaluate_models.h"
#include <algorithm>

namespace UniformOptimizer
{
	std::string GPU_Tag;
	int split_num;

	using std::cout;
	using std::endl;
	using std::string;

	typedef EA::Genetic<SplitSolution, SplitVariance> GA_Type;
	typedef EA::GenerationType<SplitSolution, SplitVariance> Generation_Type;


	SplitSolution::SplitSolution()
	{
		breakpoints = std::vector<int>(split_num);
	}

	string SplitSolution::to_string() const
	{
		string result = string("{ ");

		for (int i = 0; i < breakpoints.size(); i++)
		{
			result += "breakpoint" + std::to_string(i + 1) + ": " + std::to_string(breakpoints[i]);
			if (i < breakpoints.size() - 1)
			{
				result += "; ";
			}
		}
		result += "}";

		return result;
		// return string("{") + "breakpoint1:" + std::to_string(breakpoint1) + ", breakpoint2:" + std::to_string(breakpoint2) /* +", breakpoint3:"+std::to_string(breakpoint3)*/ + "}";
	}

	std::string SplitVariance::to_string() const
	{
		std::string result="{";

		if (costs.size() > 0)
		{
			for (auto cost : costs)
			{
				result += std::to_string(cost) + " ";
			}
			result += "}";
		}

		return result;
		// return string("{") + "breakpoint1:" + std::to_string(breakpoint1) + ", breakpoint2:" + std::to_string(breakpoint2) /* +", breakpoint3:"+std::to_string(breakpoint3)*/ + "}";
	}

	void init_genes(SplitSolution &p, const std::function<double(void)> &rnd01, ModelAnalyzer &analyzer)
	{
		int range = analyzer.size();
		int size = p.breakpoints.size();
		// rnd01() gives a random number in 0~1
		do
		{
			p.breakpoints[0] = 0.0 + range * rnd01();
			// std::cout<<"init_gene1";
		} while (p.breakpoints[0] == 0 || p.breakpoints[0] > range - size + 0);
		// std::cout<<std::endl;

		for (int i = 1; i <= size - 1; i++)
		{
			do
			{
				p.breakpoints[i] = 0.0 + range * rnd01();
				// std::cout<<"init_gene2";
			} while (p.breakpoints[i] <= p.breakpoints[i - 1] || p.breakpoints[i] > range - size + i);
			// std::cout<<std::endl;
		}

		// do
		// {
		//     p.breakpoint3=0.0+range*rnd01();
		// }while(p.breakpoint3 == p.breakpoint2 || p.breakpoint3 == p.breakpoint1 || p.breakpoint3 == 0 || p.breakpoint3 == range - 1);
	}

	bool eval_solution(const SplitSolution &p, SplitVariance &c, ModelAnalyzer &analyzer)
	{
		// const int &breakpoint1 = p.breakpoint1;
		// const int &breakpoint2 = p.breakpoint2;
		// const int& breakpoint3=p.breakpoint3;

		std::vector<GraphNode> splits;
		for (auto &point : p.breakpoints)
		{
			splits.push_back(analyzer[point]);
		}
		// analyzer.SplitAndStoreChilds({analyzer[breakpoint1], analyzer[breakpoint2], analyzer[breakpoint3]});
		// analyzer.SplitAndStoreChilds({analyzer[breakpoint1], analyzer[breakpoint2]});

		c.model_name=analyzer.getName();
		c.objective1 = analyzer.SplitAndEvaluateChilds(c.costs, splits, GPU_Tag);

		// std::cout<<"cost ==> ";
		// for(auto cost : c.costs)
		// {
		// 	std::cout<<cost <<" ";
		// }

		// std::cout<<std::endl;

		// analyzer.SplitAndStoreChilds(splits);
		// c.objective1 = evam::EvalStdCurrentModelSplit(analyzer.getName());

		// evam::EvalStdCurrentModelSplit(analyzer.getName(), analyzer.getName());
		return true; // solution is accepted
	}

	SplitSolution mutate(const SplitSolution &X_base, const std::function<double(void)> &rnd01, double shrink_scale, ModelAnalyzer &analyzer)
	{
		SplitSolution X_new;
		const double mu = 0.2 * shrink_scale; // mutation radius (adjustable)
		bool in_range;
		int range = analyzer.size();
		int size = split_num;
		do
		{	
			in_range = true;
			X_new = X_base;
			X_new.breakpoints[0] += mu * (rnd01() - rnd01());
			in_range = in_range && (X_new.breakpoints[0] > 0 && X_new.breakpoints[0] < range);
			// in_range = in_range && (X_new.breakpoints[0] > 0 && X_new.breakpoints[0] < range - size + 1);
			if(size > 1)
			{
				for (int i = 1; i <= size - 1; i++)
				{
					X_new.breakpoints[i] += mu * (rnd01() - rnd01());
					in_range = in_range && (X_new.breakpoints[i] > 0 && X_new.breakpoints[i] < range);
					for(int j = 0; j < i; j++)
					{
						in_range = in_range && (X_new.breakpoints[i] != X_new.breakpoints[j]);
					}

					// X_new.breakpoints[i] += mu * (rnd01() - rnd01());
					// in_range = in_range && (X_new.breakpoints[i] > X_new.breakpoints[i - 1] && X_new.breakpoints[i] < range - size + i + 1);

					std::cout << "mutate2";
				}
			}
		} while (!in_range);
		// std::cout<<std::endl;
		
		// std::cout<<"end";

		// X_new.breakpoint3+=mu*(rnd01()-rnd01());
		// in_range=in_range&&(X_new.breakpoint3>=0.0 && X_new.breakpoint3<range);
		return X_new;
	}

	SplitSolution crossover(const SplitSolution &X1, const SplitSolution &X2, const std::function<double(void)> &rnd01)
	{
		SplitSolution X_new;
		double r;
		int size = split_num;
		for (int i = 0; i <= size; i++)
		{
			r = rnd01();
			X_new.breakpoints[i] = r * X1.breakpoints[i] + (1.0 - r) * X2.breakpoints[i];
		}
		// r = rnd01();
		// X_new.breakpoint1 = r * X1.breakpoint1 + (1.0 - r) * X2.breakpoint1;
		// r = rnd01();
		// X_new.breakpoint2 = r * X1.breakpoint2 + (1.0 - r) * X2.breakpoint2;
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
		const SplitVariance &best = last_generation.chromosomes[last_generation.best_chromosome_index].middle_costs;

		static float raw_cost = [=]()
		{
			return evam::TimeEvaluateChildModels_impl(best.model_name,-1, GPU_Tag);
		}();

		float total=0.0f;
		for(auto& cost: best.costs)
		{
			total+=cost;
		}

		cout
			<< "Generation [" << generation_number << "], "
			<< "Best std=" << last_generation.best_total_cost << ", "
			<< "Average std=" << last_generation.average_cost << ", "
			<< "Best split=" << best_genes.to_string() << ", "
			<< "Exe_time=" << last_generation.exe_time
			<< endl
			<< "> Best:"<< endl
			<< "bench =|> " << best.to_string() << endl
			<< "total=" << total << "ms" << endl
			<< "raw=" <<raw_cost << "ms" << endl
			<< "overhead=" << (total-raw_cost)/raw_cost <<endl
			<< "std=" << best.objective1 << endl
			<< endl;

		output_file
			<< generation_number << ", "
			<< best.model_name << ", "
			<< last_generation.average_cost << ", "
			<< last_generation.best_total_cost << ", "
			<< best_genes.to_string() << ", "
			<< best.to_string() << ", "
			<< (total-raw_cost)/raw_cost << ", "
			<< total << endl;
	}

	void optimize(ModelAnalyzer &analyzer, int num, std::string Tag, bool enable_muti_thread, bool early_exit, int generation, int population, double tol_stall_best, int best_stall_max)
	{
		split_num = num;
		GPU_Tag = Tag;
		// std::cout<<num<<std::endl;
		// std::cout<<split_num<<std::endl;
		output_file.open(RootPathManager::GetRunRootFold() / "results.csv");
		output_file << "step" << ", "
					<< "model-name" << ", "
					<< "cost-avg" << ", "
					<< "cost-best" << ", "
					<< "solution-best" << ", "
					<< "runtime-bench (ms)" << ", "
					<< "overhead" << ", "
					<< "total cost (ms)" << endl;

		EA::Chronometer timer;
		timer.tic();

		GA_Type ga_obj = GA_Type(analyzer);
		ga_obj.split_num = num;
		// ga_obj.analyzer=analyzer;
		ga_obj.problem_mode = EA::GA_MODE::SOGA;
		ga_obj.multi_threading = enable_muti_thread;
		ga_obj.idle_delay_us = 10; // switch between threads quickly
		ga_obj.dynamic_threading = true;
		ga_obj.verbose = true;
		ga_obj.N_threads=10;
		ga_obj.population = population;
		ga_obj.generation_max = generation;
		ga_obj.calculate_SO_total_fitness = calculate_SO_total_fitness;
		ga_obj.init_genes = init_genes;
		ga_obj.eval_solution = eval_solution;
		ga_obj.mutate = mutate;
		ga_obj.crossover = crossover;
		ga_obj.SO_report_generation = SO_report_generation;
		ga_obj.crossover_fraction = 0.7;
		ga_obj.mutation_rate = 0.2;
		ga_obj.best_stall_max = best_stall_max;
		ga_obj.elite_count = std::max(1, int(population/4));
		ga_obj.tol_stall_best = tol_stall_best;
		ga_obj.early_exit = early_exit;
		ga_obj.solve();

		cout << "The problem is optimized in " << timer.toc() << " seconds." << endl;

		output_file.close();
	}
}
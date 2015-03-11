/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Contributors:
 *  - Andrea Battistini (andreabattistini@hotmail.com)
 *  - Chiara Pierucci (chiarapierucci14@gmail.com)
 */
#include "learning/linear/coordinate_ascent.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cfloat>
#include <cmath>
#include <chrono>

#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>

namespace quickrank {
namespace learning {
namespace linear {

void preCompute(Feature* training_dataset, unsigned int num_docs,
                unsigned int num_fx, Score* PreSum, double* weights,
                Score* MyTrainingScore, unsigned int i) {

#pragma omp parallel for
  for (unsigned int j = 0; j < num_docs; j++) {
    PreSum[j] = 0;
    MyTrainingScore[j] = 0;
    // compute feature*weight for all the feature different from i
    for (unsigned int k = 0; k < num_fx; k++) {
      MyTrainingScore[j] += weights[k] * training_dataset[j * num_fx + k];
    }
    PreSum[j] = MyTrainingScore[j]
        - (weights[i] * training_dataset[j * num_fx + i]);
  }

}

const std::string CoordinateAscent::NAME_ = "COORDASC";

CoordinateAscent::CoordinateAscent(const boost::property_tree::ptree &info_ptree,
                   const boost::property_tree::ptree &model_ptree){
                   
	num_points_=0;
	window_size_=0.0;
	reduction_factor_=0.0;
	max_iterations_=0;
	max_failed_vali_=0;
	
	//read (training) info  
	num_points_=info_ptree.get<unsigned int>("num-points");
	window_size_=info_ptree.get<double>("window-size");
	reduction_factor_=info_ptree.get<double>("reduction-factor");
	max_iterations_=info_ptree.get<unsigned int>("max-iterations");
	max_failed_vali_ = info_ptree.get<unsigned int>("max-failed-vali"); 
	
	unsigned int max_feature=0;
	BOOST_FOREACH(const boost::property_tree::ptree::value_type &couple, model_ptree){
		
		if (couple.first =="couple"){
			unsigned int feature=couple.second.get<unsigned int>("feature");
			if(feature>max_feature){
				max_feature=feature;
			}
		}
	} 
	
	best_weights_ = new double[max_feature];
	best_weights_size_=max_feature;
	for (unsigned int i=0; i<max_feature; i++){
		best_weights_[i]=0.0;
	}
	
	BOOST_FOREACH(const boost::property_tree::ptree::value_type &couple, model_ptree){
		
		if (couple.first =="couple"){
			int feature=couple.second.get<int>("feature");
			double weight=couple.second.get<double>("weight");
			best_weights_[feature-1]=weight; 
			//std::cout<<feature<<" "<<weight<<std::endl;
		}
	}                
}

CoordinateAscent::~CoordinateAscent() {
  if (best_weights_ != NULL) {
    delete[] best_weights_;
  }
}

std::ostream& CoordinateAscent::put(std::ostream& os) const {
  os 
		<< "# Ranker: " << name() << std::endl
		<< "# number of points = " << num_points_ << std::endl
		<< "# window size = " << window_size_ << std::endl
		<< "# reduction factor = " << reduction_factor_ << std::endl
		<< "# number of max iterations = " << max_iterations_ << std::endl
		<< "# number of fails on validation before exit = " << max_failed_vali_ << std::endl;

	
  return os;
}

void CoordinateAscent::preprocess_dataset(
    std::shared_ptr<data::Dataset> dataset) const {
  if (dataset->format() != data::Dataset::HORIZ)
    dataset->transpose();

}

void CoordinateAscent::learn(
    std::shared_ptr<quickrank::data::Dataset> training_dataset,
    std::shared_ptr<quickrank::data::Dataset> validation_dataset,
    std::shared_ptr<quickrank::metric::ir::Metric> scorer,
    unsigned int partial_save, const std::string output_basename) {

  auto begin = std::chrono::steady_clock::now();
  double window_size=window_size_; //preserve original value of the window

  // Do some initialization
  preprocess_dataset(training_dataset);
  if (validation_dataset)
    preprocess_dataset(validation_dataset);

  std::cout << "# Training:" << std::endl;
  std::cout << std::fixed << std::setprecision(4);
	std::cout << "# --------------------------" << std::endl;
	std::cout << "# iter. training validation" << std::endl;
  std::cout << "# --------------------------" << std::endl;

  // initialize weights and best_weights a 1/n

  double* weights = new double[training_dataset->num_features()];
  best_weights_ = new double[training_dataset->num_features()];
  best_weights_size_=training_dataset->num_features();
  for (unsigned int i = 0; i < training_dataset->num_features(); i++) {
    weights[i] = 1.0 / training_dataset->num_features();
    best_weights_[i] = weights[i];
  }

  // array of points in the window to be used to compute NDCG 
  double* points = new double[num_points_ + 1];
  MetricScore* MyNDCGs = new MetricScore[num_points_ + 1];
  MetricScore Bestmetric_on_validation = 0;
  Score* PreSum = new Score[training_dataset->num_instances()];
  Score* MyTrainingScore = new Score[training_dataset->num_instances()
      * (num_points_ + 1)];
      
  Score* MyValidationScore = NULL;
  if (validation_dataset){
  	MyValidationScore = new Score[validation_dataset->num_instances()];
	}
	// counter of sequential iterations without improvement on validation
	unsigned int count_failed_vali=0;
  // loop for max_iterations_
  for (unsigned int b = 0; b < max_iterations_; b++) {

    double step = 2 * window_size / num_points_;  // step to select points in the window
    for (unsigned int i = 0; i < training_dataset->num_features(); i++) {
      double lower_bound = weights[i] - window_size;  // lower and upper bounds of the window
      double upper_bound = weights[i] + window_size;
      // compute feature*weight for all the feature different from i

      preCompute(training_dataset->at(0, 0), training_dataset->num_instances(),
                 training_dataset->num_features(), PreSum, weights,
                 MyTrainingScore, i);
      MetricScore MyBestNDCG = scorer->evaluate_dataset(training_dataset,
                                                        MyTrainingScore);
      bool dirty = false;		// flag to remind if weights were changed or not  
      unsigned int effective_len = 0; // len of array of only positive points 

      while (lower_bound <= upper_bound) {
        if (lower_bound >= 0) {
          points[effective_len] = lower_bound;
          effective_len++;
        }
        lower_bound += step;
      }
#pragma omp parallel for default(none) shared(effective_len,training_dataset,PreSum,MyNDCGs,scorer,i,points,std::cout,MyTrainingScore) 
      for (unsigned int p = 0; p < effective_len; p++) {
      //loop to add partial scores to the total score of the feature i
        for (unsigned int j = 0; j < training_dataset->num_instances(); j++) {
          MyTrainingScore[j + (training_dataset->num_instances() * p)] =
              points[p] * training_dataset->at(j, i)[0] + PreSum[j];
        }
				// each thread computes NDCG on some points of the window
        // scorer gets a part of array MyTrainingScore for the thread p-th
        // Operator & is used to obtain the first position of the sub-array
        MyNDCGs[p] = scorer->evaluate_dataset(
            training_dataset,
            &MyTrainingScore[training_dataset->num_instances() * p]);
      }
      // End parallel
      
      // Find the best NDCG (not in parallel)
      for (unsigned int p = 0; p < effective_len; p++) {
        if (MyBestNDCG < MyNDCGs[p]) {
          MyBestNDCG = MyNDCGs[p];
          weights[i] = points[p];
          dirty = true;
        }
      }

      if (dirty == true) {  // normalize if needed
        double normalized_sum = 0;
        for (unsigned int h = 0; h < training_dataset->num_features(); h++) {
          normalized_sum += weights[h];
        }
        for (unsigned int h = 0; h < training_dataset->num_features(); h++) {
          weights[h] /= normalized_sum;
        }
      }
      
    }// end for i
    
    
    for (unsigned int j = 0; j < training_dataset->num_instances(); j++) {
       //compute scores of training documents 
    	MyTrainingScore[j] = 0;
      for (unsigned int k = 0; k < training_dataset->num_features(); k++) {
      	MyTrainingScore[j] += weights[k] * training_dataset->at(j, k)[0];
      }
    }
    
     // compute NDCG using best_weights
    MetricScore metric_on_training = scorer->evaluate_dataset(training_dataset,
                                                              MyTrainingScore);
                                                              
    std::cout <<std::setw(7)<<b+1<<std::setw(9)<<metric_on_training;                                                          
   
   // check if there is validation_dataset 
   if(validation_dataset){  
      for (unsigned int j = 0; j < validation_dataset->num_instances(); j++) {
       //compute scores of validation documents
        MyValidationScore[j] = 0;
        for (unsigned int k = 0; k < validation_dataset->num_features(); k++) {
          MyValidationScore[j] += weights[k] * validation_dataset->at(j, k)[0];
        }
      }
			MetricScore metric_on_validation = scorer->evaluate_dataset(
        validation_dataset, MyValidationScore);	
			
			std::cout <<std::setw(9)<<metric_on_validation;	
    	if (metric_on_validation > Bestmetric_on_validation) {
    		count_failed_vali=0;//reset to zero when validation improves
      	Bestmetric_on_validation = metric_on_validation;
      	for (unsigned int h = 0; h < training_dataset->num_features(); h++) {
        	best_weights_[h] = weights[h];
      	}
				std::cout <<" *";	
    	}
    	else{
    		count_failed_vali++;
    		if(count_failed_vali>=max_failed_vali_){
					std::cout <<std::endl;
					break;
				}	
			}
		}
		
		std::cout <<std::endl;	 
    window_size *= reduction_factor_;
  }
  //end iterations
  
  //if there is no validation dataset get the weights of training as best_weights 
  if(validation_dataset==NULL){
  	for (unsigned int i=0; i<best_weights_size_;i++){
  		best_weights_[i]=weights[i];
  	}
	}
	
	if (MyValidationScore!=NULL){
		delete[] MyValidationScore;
		}
  
  delete[] MyTrainingScore;
  delete[] weights;
  delete[] PreSum;
  delete[] points;
  delete[] MyNDCGs;

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = std::chrono::duration_cast<
      std::chrono::duration<double>>(end - begin);
	std::cout << std::endl;
  std::cout << "# \t Training time: " << std::setprecision(2)<< elapsed.count() << " seconds" <<std::endl;

}

void CoordinateAscent::score_dataset(std::shared_ptr<data::Dataset> dataset,
                                     Score* scores) const {
  preprocess_dataset(dataset);

  for (unsigned int q = 0; q < dataset->num_queries(); q++) {
    std::shared_ptr<data::QueryResults> r = dataset->getQueryResults(q);
    score_query_results(r, scores, dataset->num_features());
    scores += r->num_results();
  }
}

// assumes vertical dataset
// offset to next feature of the same instance
void CoordinateAscent::score_query_results(
    std::shared_ptr<data::QueryResults> results, Score* scores,
    unsigned int offset) const {
  const quickrank::Feature* d = results->features();
  for (unsigned int i = 0; i < results->num_results(); i++) {
    scores[i] = score_document(d, offset);
    d += offset;
  }
}

// assumes vertical dataset
Score CoordinateAscent::score_document(const quickrank::Feature* d,
                                       const unsigned int offset) const {
  Score score = 0;
  for (unsigned int k = 0; k < offset; k++) {
    score += best_weights_[k] * d[k];
  }
  return score;

}

std::ofstream& CoordinateAscent::save_model_to_file(std::ofstream& os) const {
  // write ranker description
	os <<"\t<info>" <<std::endl;
  os <<"\t\t<type>" <<name() <<"</type>"<<std::endl;
	os <<"\t\t<num-points>" <<num_points_ <<"</num-points>"<<std::endl;
  os <<"\t\t<window-size>" <<window_size_ <<"</window-size>"<<std::endl;
  os <<"\t\t<reduction-factor>" <<reduction_factor_ <<"</reduction-factor>"<<std::endl;
  os <<"\t\t<max-iterations>" <<max_iterations_ <<"</max-iterations>"<<std::endl;
  os <<"\t\t<max-failed-vali>" <<max_failed_vali_ <<"</max-failed-vali>"<<std::endl;
	os <<"\t</info>" <<std::endl;

	os <<"\t<ensemble>" <<std::endl;
	auto old_precision = os.precision();
	os.setf(std::ios::floatfield, std::ios::fixed);
	for (unsigned int i=0; i<best_weights_size_; i++){
	//10 seems to be a "good" precision
		os << std::setprecision(10);
		os <<"\t\t<couple>" <<std::endl;
		os <<"\t\t\t<feature>" <<i+1<<"</feature>"<<std::endl;
		os <<"\t\t\t<weight>" <<best_weights_[i]<<"</weight>"<<std::endl;
		os <<"\t\t</couple>" <<std::endl;
	}
	os <<"\t</ensemble>" <<std::endl;
  os << std::setprecision(old_precision);
  // save xml model
  // TODO: Save model to file
  return os;
}

}  // namespace linear
}  // namespace learning
}  // namespace quickrank

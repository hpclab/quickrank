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
                unsigned int num_fx, Score* PreSum, double* pesi,
                Score* MyTrainingScore, unsigned int i) {

#pragma omp parallel for
  for (unsigned int j = 0; j < num_docs; j++) {
    PreSum[j] = 0;
    MyTrainingScore[j] = 0;
    //loop per assegnare lo score ai documenti per tutte le feature diverse da i
    for (unsigned int k = 0; k < num_fx; k++) {
      MyTrainingScore[j] += pesi[k] * training_dataset[j * num_fx + k];
    }
    PreSum[j] = MyTrainingScore[j]
        - (pesi[i] * training_dataset[j * num_fx + i]);
  }

}

const std::string CoordinateAscent::NAME_ = "COORDASC";

CoordinateAscent::CoordinateAscent(const boost::property_tree::ptree &info_ptree,
                   const boost::property_tree::ptree &model_ptree){
  std::cout <<"Entrati!"<<std::endl;
                   
	num_points_=0;
	window_size_=0.0;
	reduction_factor_=0.0;
	max_iterations_=0;
	
	//read (training) info  
	num_points_=info_ptree.get<unsigned int>("num-points");
	window_size_=info_ptree.get<double>("window-size");
	reduction_factor_=info_ptree.get<double>("reduction-factor");
	max_iterations_=info_ptree.get<unsigned int>("max-iterations");
	
	unsigned int max_num_feature=0;
	BOOST_FOREACH(const boost::property_tree::ptree::value_type &couple, model_ptree){
		
		if (couple.first =="couple"){
			unsigned int num_feature=couple.second.get<unsigned int>("num-feature");
			if(num_feature>max_num_feature){
				max_num_feature=num_feature;
			}
		}
	} 
	
	best_weights_ = new double[max_num_feature];
	best_weights_size_=max_num_feature;
	for (unsigned int i=0; i<max_num_feature; i++){
		best_weights_[i]=0.0;
	}
	
	BOOST_FOREACH(const boost::property_tree::ptree::value_type &couple, model_ptree){
		
		if (couple.first =="couple"){
			int num_feature=couple.second.get<int>("num-feature");
			double weight=couple.second.get<double>("weight");
			best_weights_[num_feature-1]=weight; 
			//std::cout<<num_feature<<" "<<weight<<std::endl;
		}
	}
		
		
	std::cout <<"Usciti!"<<std::endl;
                   
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
		<< "# number of max iterations = " << max_iterations_ << std::endl;

	
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

  // Do some initialization
  preprocess_dataset(training_dataset);
  if (validation_dataset)
    preprocess_dataset(validation_dataset);

  std::cout << "# Training:" << std::endl;
  std::cout << std::fixed << std::setprecision(4);
	std::cout << "# --------------------------" << std::endl;
	std::cout << "# iter. training validation" << std::endl;
  std::cout << "# --------------------------" << std::endl;

  // inizializzazione pesi a 1/N e definizione costanti

  double* pesi = new double[training_dataset->num_features()];
  best_weights_ = new double[training_dataset->num_features()];
  best_weights_size_=training_dataset->num_features();
  for (unsigned int i = 0; i < training_dataset->num_features(); i++) {
    pesi[i] = 1.0 / training_dataset->num_features();
    best_weights_[i] = pesi[i];
  }

  // vettore contenente i punti in cui valutare NDCG per i vari thread
  double* punti = new double[num_points_ + 1];
  MetricScore* MyNDCGs = new MetricScore[num_points_ + 1];
  MetricScore Bestmetric_on_validation = 0;
  Score* PreSum = new Score[training_dataset->num_instances()];
  Score* MyTrainingScore = new Score[training_dataset->num_instances()
      * (num_points_ + 1)];
  Score* MyValidationScore = new Score[validation_dataset->num_instances()];

  //calcola lo score per ogni documento del training set come combinazione lineare dei pesi.Faccio il ciclo per tutti i documenti del dataset
  for (unsigned int b = 0; b < max_iterations_; b++) {

    double passo = 2 * window_size_ / num_points_;  // passo con cui mi muovo nell'intervallo
    for (unsigned int i = 0; i < training_dataset->num_features(); i++) {
      double estremoMin = pesi[i] - window_size_;  //estremi finestra di ricerca
      double estremoMax = pesi[i] + window_size_;
      // inizializzo l'NDCG con il valore ottenuto con i pesi iniziali

      preCompute(training_dataset->at(0, 0), training_dataset->num_instances(),
                 training_dataset->num_features(), PreSum, pesi,
                 MyTrainingScore, i);
      MetricScore MyBestNDCG = scorer->evaluate_dataset(training_dataset,
                                                        MyTrainingScore);
      bool dirty = false;		// pesi non alterati
      unsigned int lungReale = 0;
      //riempiamo con i valori di estremoMin positivi per poi darlo ai thread per valutare NDCG
      while (estremoMin <= estremoMax) {
        if (estremoMin >= 0) {
          punti[lungReale] = estremoMin;
          lungReale++;
        }
        estremoMin += passo;
      }
#pragma omp parallel for default(none) shared(lungReale,training_dataset,PreSum,MyNDCGs,scorer,i,punti,std::cout,MyTrainingScore) 
      for (unsigned int p = 0; p < lungReale; p++) {
        for (unsigned int j = 0; j < training_dataset->num_instances(); j++) {
          //loop per sommare gli score parziali a quello relativo alla feature di cui sto cercando il peso migliore
          MyTrainingScore[j + (training_dataset->num_instances() * p)] =
              punti[p] * training_dataset->at(j, i)[0] + PreSum[j];
        }

        //ogni thread prende un tot di punti (p) e su essi calcola l'NDCG
        // passo allo scorer la parte del vettore MyTrainingScore che spetta al thread p. Notare che uso & perche voglio l'indirizzo della prima posizione di quel sottovettore
        MyNDCGs[p] = scorer->evaluate_dataset(
            training_dataset,
            &MyTrainingScore[training_dataset->num_instances() * p]);
      }
      // Cerco il punto che mi ha dato il miglior NDCG per la feature i
      for (unsigned int p = 0; p < lungReale; p++) {
        if (MyBestNDCG < MyNDCGs[p]) {
          MyBestNDCG = MyNDCGs[p];
          pesi[i] = punti[p];
          dirty = true;
        }
      }

      if (dirty == true) {  //se ho cambiato il peso della feature devo rinormalizzare
        double sommaNorm = 0;
        for (unsigned int h = 0; h < training_dataset->num_features(); h++) {
          sommaNorm += pesi[h];
        }
        for (unsigned int h = 0; h < training_dataset->num_features(); h++) {
          pesi[h] /= sommaNorm;
        }
      }
    }
      for (unsigned int j = 0; j < training_dataset->num_instances(); j++) {
        //loop per assegnare lo score ai documenti in base al pesi per train
        MyTrainingScore[j] = 0;
        for (unsigned int k = 0; k < training_dataset->num_features(); k++) {
          MyTrainingScore[j] += pesi[k] * training_dataset->at(j, k)[0];
        }
      }
     
      for (unsigned int j = 0; j < validation_dataset->num_instances(); j++) {
        //loop per assegnare lo score ai documenti in base al pesi per validation
        MyValidationScore[j] = 0;
        for (unsigned int k = 0; k < validation_dataset->num_features(); k++) {
          MyValidationScore[j] += pesi[k] * validation_dataset->at(j, k)[0];
        }
      }
    
    //NDCG calcolato con i pesi best trovati sul train e validation
    MetricScore metric_on_training = scorer->evaluate_dataset(training_dataset,
                                                              MyTrainingScore);

   
    std::cout <<std::setw(7)<<b+1<<std::setw(9)<<metric_on_training;
    if(validation_dataset){
			 MetricScore metric_on_validation = scorer->evaluate_dataset(
        validation_dataset, MyValidationScore);	
			
			std::cout <<std::setw(9)<<metric_on_validation;	
    	if (metric_on_validation > Bestmetric_on_validation) {
      		Bestmetric_on_validation = metric_on_validation;
      		for (unsigned int h = 0; h < training_dataset->num_features(); h++) {
        		best_weights_[h] = pesi[h];
      		}
				std::cout <<" *";	
    	}
    	else{
				std::cout <<std::endl;
				break;	
			}
		}
		std::cout <<std::endl;	 
    window_size_ *= reduction_factor_;
  }

  delete[] MyTrainingScore;
  delete[] MyValidationScore;
  delete[] pesi;
  delete[] PreSum;
  delete[] punti;
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
	os <<"\t</info>" <<std::endl;

	os <<"\t<ensemble>" <<std::endl;
	auto old_precision = os.precision();
	os.setf(std::ios::floatfield, std::ios::fixed);
	for (unsigned int i=0; i<best_weights_size_; i++){
	//10 seems to be a "good" precision
		os << std::setprecision(10);
		os <<"\t\t<couple>" <<std::endl;
		os <<"\t\t\t<num-feature>" <<i+1<<"</num-feature>"<<std::endl;
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

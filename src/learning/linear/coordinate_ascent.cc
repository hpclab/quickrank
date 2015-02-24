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
 *  - Andrea Battistini (email...)
 *  - Chiara Pierucci (email...)
 */
#include "learning/linear/coordinate_ascent.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cfloat>
#include <cmath>
#include <chrono>

namespace quickrank {
namespace learning {
namespace linear {
//flag openMP
bool go_parallel1 = true;
bool go_parallelP = true;

void preCompute(Feature* training_dataset, unsigned int num_docs,
                unsigned int num_fx, Score* PreSum, double* pesi,
                Score* MyTrainingScore, unsigned int i) {

#pragma omp parallel for if(go_parallel1)
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

CoordinateAscent::~CoordinateAscent() {
  if (pesiBest != NULL) {
    delete[] pesiBest;
  }
}

std::ostream& CoordinateAscent::put(std::ostream& os) const {
  os << "# Ranker: " << name() << std::endl;
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
  double tempoNDCG = 0;

  // Do some initialization
  preprocess_dataset(training_dataset);
  if (validation_dataset)
    preprocess_dataset(validation_dataset);

  std::cout << "# Training..." << std::endl;
  std::cout << std::fixed << std::setprecision(4);

  // inizializzazione pesi a 1/N e definizione costanti

  double* pesi = new double[training_dataset->num_features()];
  pesiBest = new double[training_dataset->num_features()];
  for (unsigned int i = 0; i < training_dataset->num_features(); i++) {
    pesi[i] = 1.0 / training_dataset->num_features();
    pesiBest[i] = pesi[i];
  }

  // vettore contenente i punti in cui valutare NDCG per i vari thread
  double* punti = new double[num_points_ + 1];
  MetricScore* MyNDCGs = new MetricScore[num_points_ + 1];
  MetricScore Bestmetric_on_validation = 0;
  Score* PreSum = new Score[training_dataset->num_instances()];
  Score* MyTrainingScore = new Score[training_dataset->num_instances()
      * (num_points_ + 1)];
  Score* MyValidationScore = new Score[validation_dataset->num_instances()];

  std::cout << "punti : " << num_points_ << std::endl;
  std::cout << "finestra : " << window_size_ << std::endl;
  std::cout << "alpha: " << reduction_factor_ << std::endl;
  std::cout << "iterazioni : " << num_max_iterations_ << std::endl;
  //calcola lo score per ogni documento del training set come combinazione lineare dei pesi.Faccio il ciclo per tutti i documenti del dataset
  for (unsigned int b = 0; b < num_max_iterations_; b++) {

    double passo = 2 * window_size_ / num_points_;  // passo con cui mi muovo nell'intervallo
    std::cout << "w e': " << window_size_ << " all'iterazione b= " << b
        << std::endl;
    std::cout << "passo e': " << passo << " all'iterazione b= " << b
        << std::endl;
    for (unsigned int i = 0; i < training_dataset->num_features(); i++) {
      double estremoMin = pesi[i] - window_size_;  //estremi finestra di ricerca
      double estremoMax = pesi[i] + window_size_;
      // inizializzo l'NDCG con il valore ottenuto con i pesi iniziali
      //orologio primo NDCG

      preCompute(training_dataset->at(0, 0), training_dataset->num_instances(),
                 training_dataset->num_features(), PreSum, pesi,
                 MyTrainingScore, i);
      auto begin1 = std::chrono::steady_clock::now();
      MetricScore MyBestNDCG = scorer->evaluate_dataset(training_dataset,
                                                        MyTrainingScore);
      auto end1 = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed1 = std::chrono::duration_cast<
          std::chrono::duration<double>>(end1 - begin1);
      tempoNDCG += elapsed1.count();

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
#pragma omp parallel for if(go_parallelP) default(none) shared(lungReale,training_dataset,PreSum,MyNDCGs,scorer,i,punti,tempoNDCG,std::cout,MyTrainingScore) 
      for (unsigned int p = 0; p < lungReale; p++) {
        for (unsigned int j = 0; j < training_dataset->num_instances(); j++) {
          //loop per sommare gli score parziali a quello relativo alla feature di cui sto cercando il peso migliore
          MyTrainingScore[j + (training_dataset->num_instances() * p)] =
              punti[p] * training_dataset->at(j, i)[0] + PreSum[j];
        }

        //ogni thread prende un tot di punti (p) e su essi calcola l'NDCG
        auto begin2 = std::chrono::steady_clock::now();
        // passo allo scorer la parte del vettore MyTrainingScore che spetta al thread p. Notare che uso & perche voglio l'indirizzo della prima posizione di quel sottovettore
        MyNDCGs[p] = scorer->evaluate_dataset(
            training_dataset,
            &MyTrainingScore[training_dataset->num_instances() * p]);
        auto end2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed2 = std::chrono::duration_cast<
            std::chrono::duration<double>>(end2 - begin2);
        tempoNDCG += elapsed2.count();
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

    //parallelo su for in cascata
    //#pragma omp parallel if(go_parallel)
    {
      //#pragma omp for nowait
      for (unsigned int j = 0; j < training_dataset->num_instances(); j++) {
        //loop per assegnare lo score ai documenti in base al pesi per train
        MyTrainingScore[j] = 0;
        for (unsigned int k = 0; k < training_dataset->num_features(); k++) {
          MyTrainingScore[j] += pesi[k] * training_dataset->at(j, k)[0];
        }
      }
      //#pragma omp for nowait
      for (unsigned int j = 0; j < validation_dataset->num_instances(); j++) {
        //loop per assegnare lo score ai documenti in base al pesi per validation
        MyValidationScore[j] = 0;
        for (unsigned int k = 0; k < validation_dataset->num_features(); k++) {
          MyValidationScore[j] += pesi[k] * validation_dataset->at(j, k)[0];
        }
      }
    }
    //NDCG calcolato con i pesi best trovati sul train e validation
    auto begin3 = std::chrono::steady_clock::now();
    MetricScore metric_on_training = scorer->evaluate_dataset(training_dataset,
                                                              MyTrainingScore);

    MetricScore metric_on_validation = scorer->evaluate_dataset(
        validation_dataset, MyValidationScore);
    auto end3 = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed3 = std::chrono::duration_cast<
        std::chrono::duration<double>>(end3 - begin3);
    tempoNDCG += elapsed3.count();
    std::cout << "Il valore dell'NDCG sul train e': " << metric_on_training
        << " all'iterazione b= " << b << std::endl;
    std::cout << "Il valore dell'NDCG sul validation e': "
        << metric_on_validation << " all'iterazione b= " << b << std::endl;

    if (metric_on_validation > Bestmetric_on_validation) {
      Bestmetric_on_validation = metric_on_validation;
      for (unsigned int h = 0; h < training_dataset->num_features(); h++) {
        pesiBest[h] = pesi[h];
      }
    } else {
      //Stampo pesi best
      for (unsigned int k = 0; k < training_dataset->num_features(); k++) {
        std::cout << "il peso della feature best " << k << " e': "
            << pesiBest[k] << std::endl;
      }
      std::cout << "Il valore dell'NDCG sul validation best e': "
          << Bestmetric_on_validation << std::endl;
      break;
    }

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
  std::cout << "Tempo totale learning: " << elapsed.count() << std::endl;

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
    score += pesiBest[k] * d[k];
  }
  return score;

}

std::ofstream& CoordinateAscent::save_model_to_file(std::ofstream& os) const {
  // write ranker description
  os << *this;
  // save xml model
  // TODO: Save model to file
  return os;
}

}  // namespace linear
}  // namespace learning
}  // namespace quickrank

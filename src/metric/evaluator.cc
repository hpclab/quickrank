#include "metric/evaluator.h"
#include "io/svml.h"


namespace quickrank {
namespace metric {

Evaluator::Evaluator() {
}

Evaluator::~Evaluator() {
}

void Evaluator::evaluate( learning::LTR_Algorithm* algo,
                          ir::Metric* train_metric,
                          ir::Metric* test_metric,
                          const std::string trainingfilename,
                          const std::string validationfilename,
                          const std::string testfilename,
                          const std::string featurefilename,
                          const std::string outputfilename) {
  // create reader: assum svml as ltr format
  quickrank::io::Svml reader;

  std::shared_ptr<quickrank::data::Dataset> training_dataset;
  std::shared_ptr<quickrank::data::Dataset> validation_dataset;

  if(!trainingfilename.empty()) {
    std::cout << "# Reading training dataset ..." << std::endl;
    training_dataset = reader.read_horizontal(trainingfilename);
    std::cout << reader;
    algo->set_training_dataset(training_dataset);
  } else {
    std::cerr << "!!! Error while loading training dataset" << std::endl;
    exit(EXIT_FAILURE);
  }

  if(!validationfilename.empty()) {
    std::cout << "# Reading validation dataset ..." << std::endl;
    validation_dataset = reader.read_horizontal(validationfilename);
    std::cout << reader;
    algo->set_validation_dataset(validation_dataset);
  }

  if(!featurefilename.empty()) {
    /// \todo TODO: filter features while loading dataset
  }

  if(!outputfilename.empty())
    algo->set_outputfilename(outputfilename);


  algo->set_scorer(train_metric);

  // run the learning process
  algo->init();
  algo->learn();

  if(test_metric and !testfilename.empty()) {
    // pre-clean
    training_dataset.reset();
    validation_dataset.reset();

    std::cout << "# Reading test dataset ..." << std::endl;
    std::shared_ptr<quickrank::data::Dataset> test_dataset = reader.read_horizontal(testfilename);
    std::cout << reader;
    quickrank::Score* test_scores = new quickrank::Score[test_dataset->num_instances()];
    algo->score_dataset(*test_dataset, test_scores);
    quickrank::MetricScore test_score = test_metric->evaluate_dataset(*test_dataset, test_scores);
    std::cout << "# " << *test_metric
        << " on test data = " << test_score << std::endl;
    delete [] test_scores;
  }

  if (!outputfilename.empty()) {
    algo->write_outputtofile();
  }
}

} // namespace metric
} // namespace quickrank

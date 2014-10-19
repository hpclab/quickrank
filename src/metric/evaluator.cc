#include "metric/evaluator.h"
#include "io/svml.h"
namespace qr {
namespace metric {

evaluator::~evaluator() {
  // TODO: (by cla) Do not delete objected you didn't create. Move.
  delete r;
  delete training_scorer;
  delete test_scorer;
}
void evaluator::evaluate(const char *trainingfilename, const char *validationfilename, const char *testfilename, const char *featurefilename, const char *outputfilename) {
  quickrank::io::Svml reader;
  if(not is_empty(trainingfilename)) {
    printf("Reading Training dataset:\n");
    // TODO: (by cla) Where is the delete of this dpset?
    r->set_trainingset( reader.read_vertical(trainingfilename) );
  } else exit(6);
  if(not is_empty(validationfilename)) {
    // TODO: (by cla) Where is the delete of this dpset?
    printf("Reading validation dataset:\n");
    r->set_validationset( reader.read_vertical(validationfilename));

    std::shared_ptr<quickrank::data::Dataset> dataset = reader.read_horizontal(validationfilename);
    r->set_validation_dataset(dataset);
  }
  LTR_VerticalDataset *testset = NULL;
  if(test_scorer and not is_empty(testfilename)) {
    printf("Reading test dataset:\n");
    testset = reader.read_vertical(testfilename);
  }
  if(not is_empty(featurefilename)) {
    // init featureidxs from file
  }
  if(not is_empty(outputfilename))
    r->set_outputfilename(outputfilename);
  if(normalize) {
    //normalization
  }
  r->set_scorer(training_scorer);
  r->init();
  r->learn();
  if(testset) {
    printf("Testing:\n");
#ifdef SHOWTIMER
    double timer = omp_get_wtime();
#endif
    float score = r->compute_score(testset, test_scorer);
#ifdef SHOWTIMER
    timer = omp_get_wtime()-timer;
#endif
    std::cout << "\t" << *test_scorer
              << " on test data = " << score << std::endl;
#ifdef SHOWTIMER
    printf("\t\033[1melapsed time = %.3f seconds\033[0m\n", timer);
#endif
    printf("\tdone\n");
    delete testset;
    // check on horiz dataset
    std::unique_ptr<quickrank::data::Dataset> dataset = reader.read_horizontal(testfilename);
    qr::Score* test_scores = new qr::Score[dataset->num_instances()];
    r->score_dataset(*dataset, test_scores);
    std::cout << "check: " << test_scorer->evaluate_dataset(*dataset, test_scores) << std::endl;
    delete [] test_scores;
  }
}
void evaluator::write() {
  printf("Writing output:\n");
  r->write_outputtofile();
  printf("\tdone\n");
}

} // namespace metric
} // namespace qr

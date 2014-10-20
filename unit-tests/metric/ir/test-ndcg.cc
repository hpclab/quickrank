#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "metric/ir/ndcg.h"
#include <cmath>

BOOST_AUTO_TEST_CASE( ndcg_test )
{
  double labels [] = {3,2,1,0,0};
  ResultList results (5, &labels[0], 666);

  quickrank::metric::ir::Ndcg ndcg_metric(5);
  quickrank::MetricScore idcg;

  // NDCG@k computation with K > num results
  idcg = (pow(2,labels[0])-1) + (pow(2,labels[1])-1)/log2(3) + (pow(2,labels[2])-1)/2;
  BOOST_CHECK_EQUAL( ndcg_metric.evaluate_result_list(results),
      ( (pow(2,labels[0])-1) + (pow(2,labels[1])-1)/log2(3) + (pow(2,labels[2])-1)/2
      ) / idcg
  );

  // NDCG@k computation with K = 0
  ndcg_metric.set_cutoff(0);
  BOOST_CHECK_EQUAL( ndcg_metric.evaluate_result_list(results),
      ( (pow(2,labels[0])-1) + (pow(2,labels[1])-1)/log2(3) + (pow(2,labels[2])-1)/2
      ) / idcg
  );

  // NDCG@k computation with K = No cutoff
  ndcg_metric.set_cutoff(ndcg_metric.NO_CUTOFF);
  BOOST_CHECK_EQUAL( ndcg_metric.evaluate_result_list(results) ,
      ( (pow(2,labels[0])-1) + (pow(2,labels[1])-1)/log2(3) + (pow(2,labels[2])-1)/2
      ) /idcg
  );


  // NDCG@k computation with K < num results
  ndcg_metric.set_cutoff(2);
  idcg = (pow(2,labels[0])-1) + (pow(2,labels[1])-1)/log2(3);
  BOOST_CHECK_EQUAL( ndcg_metric.evaluate_result_list(results),
      ( (pow(2,labels[0])-1) + (pow(2,labels[1])-1)/log2(3)
      ) / idcg
  );

}

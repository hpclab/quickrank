#include <iostream>

#include "data/dataset.h"
#include "io/svml.h"

double ranker(float* v);


int main(int argc, char *argv[]) {
  std::cout << "# ## ========================== ## #" << std::endl
            << "# ##          QuickRank         ## #" << std::endl
            << "# ## -------------------------- ## #" << std::endl
            << "# ## developed by the HPC. Lab. ## #" << std::endl
            << "# ##  http://hpc.isti.cnr.it/   ## #" << std::endl
            << "# ##  quickrank@.isti.cnr.it    ## #" << std::endl
            << "# ## ========================== ## #" << std::endl;

  std::cout << "usage: quickscore <dataset>" << std::endl;
  char* data_file = argv[1];

  // read and check dataset
  quickrank::io::Svml reader;
  std::unique_ptr<quickrank::data::Dataset> dataset =
      reader.read_horizontal(data_file);




  return 0;

}


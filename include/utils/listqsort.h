/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Contributor:
 *   HPC. Laboratory - ISTI - CNR - http://hpc.isti.cnr.it/
 */
#ifndef QUICKRANK_UTILS_LISTQSORT_H_
#define QUICKRANK_UTILS_LISTQSORT_H_

#include <cstdlib>

/*! \file listqsort.hpp
 * \brief sort single-linked lists
 */

/*! sort a single-linked list
 *  @tparam T typename of the list-element
 *  @param begin points to the begin of the list
 *  @param end points to the end of the list
 */
template<typename T> void listqsort(T *&begin, T *end = NULL) {
  if (begin != end) {
    //split input list in greater/less than pivot (i.e., *begin)
    T *gtfront = NULL, *ltfront = NULL;
    T **gtback = &gtfront, **ltback = &ltfront;
    for (T *current = begin->next; current != end; current = current->next)
      if (*current > *begin) {
        *gtback = current;
        gtback = &current->next;
      } else {
        *ltback = current;
        ltback = &current->next;
      }
    //set end for lists
    *gtback = end, *ltback = begin;
    //recursive step
    listqsort(ltfront, *ltback), listqsort(gtfront, *gtback);
    //concat resulting lists
    begin->next = gtfront;
    begin = ltfront;
  }
}
#endif

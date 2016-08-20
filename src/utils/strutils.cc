/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * Unless explicitly acquired and licensed from Licensor under another
 * license, the contents of this file are subject to the Reciprocal Public
 * License ("RPL") Version 1.5, or subsequent versions as allowed by the RPL,
 * and You may not copy or use this file in either source code or executable
 * form, except in compliance with the terms and conditions of the RPL.
 *
 * All software distributed under the RPL is provided strictly on an "AS
 * IS" basis, WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED, AND
 * LICENSOR HEREBY DISCLAIMS ALL SUCH WARRANTIES, INCLUDING WITHOUT
 * LIMITATION, ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, QUIET ENJOYMENT, OR NON-INFRINGEMENT. See the RPL for specific
 * language governing rights and limitations under the RPL.
 *
 * Contributor:
 *   HPC. Laboratory - ISTI - CNR - http://hpc.isti.cnr.it/
 */
#include "utils/strutils.h"

#include <algorithm>
#include <iostream>

#include <iomanip>

/*! \file strutils.hpp
 * \brief some useful functions for strings
 */

/*! \fn read_token(char *&str, const char exitch='\0')
 *  \brief skip a sequence of spaces (see ISSPC macro) and return the token which is considered closed at the next space encountered (if \a exitch is the FIRST not-space of str the function returns immediately).
 */
char *read_token(char *&str, const char exitch) {
  while (ISSPC(*str) && *str != '\0')
    ++str;
  if (*str == exitch)
    return str;
  char *token = str;
  while (!ISSPC(*str) && *str != '\0' && *str != exitch)
    ++str;
  if (*str != '\0')
    *str++ = '\0';
  return token;
}

/*! \fn is_empty(const char *str)
 *  \brief return true if \a str pointer is NULL or is made of spaces (see ISSPC macro)
 */
bool is_empty(const char *str) {
  if (str)
    for (size_t i = 0; str[i] != '\0'; ++i)
      if (!ISSPC(str[i]))
        return false;
  return true;
}

/*! \fn atou(char *str, const char *sep)
 *  \brief skip a sequence of spaces (see ISSPC macro) and return the unsigned int after \a sep
 */
unsigned int atou(char *str, const char *sep) {
  while (ISSPC(*str) && *str != '\0')
    ++str;
  for (size_t i = 0; sep[i] != '\0' && *str != '\0'; ++i, ++str)
    if (*str != sep[i])
      exit(1);
  int x = atoi(str);
  if (x < 0)
    exit(3);
  return (unsigned int) x;
}

/*
 int cisrtcmp(const char *a, const char* b) {
 while(*a!='\0' && *b!='\0') if(tolower(*a)!=tolower(*b)) return 1; else ++a, ++b;
 return (*a!='\0' || *b!='\0') ? 1 : 0;
 }*/

std::string &trim(std::string &str) {
  while (std::isspace(str[0])) {
    str.erase(str.begin()); // erase it
  }

  // if the last character of the string is a whitespace or a tab
  while (std::isspace(str[str.length() - 1])) {
    str.erase(str.end() - 1); // erase it
  }

  return str;
}

void print_weights(std::vector<double> weights, std::string header) {
  return;
  std::cout << std::endl << "# " << header << std::endl;
  for (size_t i=0; i < weights.size(); ++i) {
    std::cout << std::setprecision(0) << i << ":"
              << std::setprecision(3) << weights[i] << " ";
  }
  std::cout << std::endl << std::endl;
}
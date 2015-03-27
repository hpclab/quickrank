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
#ifndef QUICKRANK_UTILS_STRUTILS_H_
#define QUICKRANK_UTILS_STRUTILS_H_

#include <cctype>

/*! \file strutils.hpp
 * \brief some useful functions for strings
 */

/*! \def ISSPC
 *  \brief return true if \a ch is a space-char
 */
#define ISSPC(ch) (ch==' ' || ch=='\t' || ch=='\n' || ch=='\v' || ch=='\f' || ch=='\r')

/*! \def ISEMPTY
 *  \brief return true if \a str is empty string
 */
#define ISEMPTY(str) (*(str)=='\0')

/*! \fn read_token(char *&str, const char exitch='\0')
 *  \brief skip a sequence of spaces (see ISSPC macro) and return the token which is considered closed at the next space encountered (if \a exitch is the FIRST not-space of str the function returns immediately).
 */
char *read_token(char *&str, const char exitch = '\0');

/*! \fn is_empty(const char *str)
 *  \brief return true if \a str pointer is NULL or is made of spaces (see ISSPC macro)
 */
bool is_empty(const char *str);

/*! \fn atou(char *str, const char *sep)
 *  \brief skip a sequence of spaces (see ISSPC macro) and return the unsigned int after \a sep
 */
unsigned int atou(char *str, const char *sep);

#endif

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
char *read_token(char *&str, const char exitch='\0');

/*! \fn is_empty(const char *str)
 *  \brief return true if \a str pointer is NULL or is made of spaces (see ISSPC macro)
 */
bool is_empty(const char *str);

/*! \fn atou(char *str, const char *sep)
 *  \brief skip a sequence of spaces (see ISSPC macro) and return the unsigned int after \a sep
 */
unsigned int atou(char *str, const char *sep);

#endif

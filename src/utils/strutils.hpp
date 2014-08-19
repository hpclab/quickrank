#ifndef __STRUTILS_HPP__
#define __STRUTILS_HPP__

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
inline char *read_token(char *&str, const char exitch='\0') {
	while(ISSPC(*str) && *str!='\0') ++str;
	if(*str==exitch) return str;
	char *token = str;
	while(!ISSPC(*str) && *str!='\0' && *str!=exitch) ++str;
	if(*str!='\0') *str++ = '\0';
	return token;
}

/*! \fn is_empty(const char *str)
 *  \brief return true if \a str pointer is NULL or is made of spaces (see ISSPC macro)
 */
inline bool is_empty(const char *str) {
	if(str)
		for(size_t i=0; str[i]!='\0'; ++i)
			if(!ISSPC(str[i])) return false;
	return true;
}

/*! \fn atou(char *str, const char *sep)
 *  \brief skip a sequence of spaces (see ISSPC macro) and return the unsigned int after \a sep
 */
unsigned int inline atou(char *str, const char *sep) {
	while(ISSPC(*str) && *str!='\0') ++str;
	for(size_t i=0; sep[i]!='\0' && *str!='\0'; ++i, ++str)
		if(*str!=sep[i]) exit(1);
	int x = atoi(str);
	if(x<0) exit(3);
	return (unsigned int) x;
}

int cisrtcmp(const char *a, const char* b) {
	while(*a!='\0' && *b!='\0') if(tolower(*a)!=tolower(*b)) return 1; else ++a, ++b;
	return (*a!='\0' || *b!='\0') ? 1 : 0;
}

#endif

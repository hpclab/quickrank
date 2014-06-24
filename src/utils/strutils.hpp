#ifndef __STRUTILS_HPP__
#define __STRUTILS_HPP__

#define ISSPC(ch) (ch==' ' || ch=='\t' || ch=='\n' || ch=='\v' || ch=='\f' || ch=='\r')
#define ISEMPTY(str) (*(str)=='\0')

//skip a sequence of spaces (ISSPC) and get the next token which is considered closed at the next space encountered (if exitch is the FIRST not-space of str the function returns immediately).
inline char *read_token(char *&str, const char exitch='\0') {
	while(ISSPC(*str) && *str!='\0') ++str;
	if(*str==exitch) return str;
	char *token = str;
	while(!ISSPC(*str) && *str!='\0') ++str;
	if(*str!='\0') *str++ = '\0';
	return token;
}

bool is_empty(const char *str) {
	if(str)
		for(size_t i=0; str[i]!='\0'; ++i)
			if(!ISSPC(str[i])) return false;
	return true;
}

#endif

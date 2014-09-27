#include "learning/tree/rtnode.h"

void RTNode::save_leaves(RTNode **&leaves, unsigned int &nleaves, unsigned int &capacity) {
  if(featureidx==uint_max) {
    if(nleaves==capacity) {
      capacity = 2*capacity+1;
      leaves = (RTNode**)realloc(leaves, sizeof(RTNode*)*capacity);
    }
    leaves[nleaves++] = this;
  } else {
    left->save_leaves(leaves, nleaves, capacity);
    right->save_leaves(leaves, nleaves, capacity);
  }
}

void RTNode::write_outputtofile(FILE *f, const int indentsize) {
  char* indent = new char [indentsize+1]; // char indent[indentsize+1];
  for(int i=0; i<indentsize; indent[i++]='\t');
  indent[indentsize] = '\0';
  if(featureid==uint_max)
    fprintf(f, "%s\t<output> %.15f </output>\n", indent, avglabel);
  else {
    fprintf(f, "%s\t<feature> %u </feature>\n", indent, featureid);
    fprintf(f, "%s\t<threshold> %.8f </threshold>\n", indent, threshold);
    fprintf(f, "%s\t<split pos=\"left\">\n", indent);
    left->write_outputtofile(f, indentsize+1);
    fprintf(f, "%s\t</split>\n", indent);
    fprintf(f, "%s\t<split pos=\"right\">\n", indent);
    right->write_outputtofile(f, indentsize+1);
    fprintf(f, "%s\t</split>\n", indent);
  }
  delete [] indent;
}

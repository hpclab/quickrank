#ifndef __TRIE_HPP__
#define __TRIE_HPP__

/*! \file trie.hpp
 * \brief Implementation of trie data structure
 */

#include <cstring>

/*! \def ALPHABET_OFFSET
 * \brief define which char is used as first one for the set of possible chars used as children for a node
 */
#define ALPHABET_OFFSET '0'
/*! \def ALPHABET_LENGTH
 * \brief read chars are mapped into integers belonging to [ALPHABET_OFFSET, ALPHABET_OFFSET+ALPHABET_LENGTH-1]
 */
#define ALPHABET_LENGTH 80

/*! \class trie
 *  \brief trie implementation
 *  @tparam T typename of leaves in the trie
 *  \remark a \a T(char*) constructor is required
 */
template<typename T> class trie {
 public:
  /** \brief default constructor
   */
  trie()
      : nleaves(0) {
    root = new trienode;
  }
  ~trie() {
    delete root;
  }
  /** \brief traverse the trie for the string \a str and allocate the corresponding path is not yet allocated
   * @return poiter to the leaf data
   */
  T *insert(const char *str) {
    trienode *node = set_path(str);
    if (!node->leaf)
      node->leaf = new T(str),  //NOTE T needs to have a constructor like T(char*)
      ++nleaves;
    return node->leaf;
  }
  /** \brief traverse the trie for the string \a str
   * @return poiter to the leaf data if it exists, otherwise NULL is returned
   */
  T *lookup(const char *str) const {
    trienode *node = get_path(str);
    return node ? node->leaf : NULL;
  }
  /** \brief create a new array made up of the leaves pointers
   */
  T **get_leaves() const {
    if (nleaves == 0)
      return NULL;
    else {
      unsigned int size = 0;
      T **leaves = new T*[nleaves];
      root->copy_toarray(leaves, size);
      return leaves;
    }
  }
  /** \brief return the number of leaves in the trie
   */
  unsigned int get_nleaves() const {
    return nleaves;
  }
  /** \brief return true if the number of leaves in the trie is zero
   */
  bool is_empty() {
    return nleaves == 0;
  }
 private:
  struct trienode {
    trienode()
        : leaf(NULL) {
      children = new trienode*[ALPHABET_LENGTH]();
    }
    ~trienode() {
      for (int i = 0; i < ALPHABET_LENGTH; ++i)
        delete children[i];
      delete[] children, delete leaf;
    }
    trienode *set_child(const char ch) {
      int i = (ch - ALPHABET_OFFSET) % ALPHABET_LENGTH;
      if (!children[i])
        children[i] = new trienode;
      return children[i];
    }
    trienode *get_child(const char ch) {
      return children[(ch - ALPHABET_OFFSET) % ALPHABET_LENGTH];
    }
    void copy_toarray(T** leaves, unsigned int &size) {
      for (int i = 0; i < ALPHABET_LENGTH; ++i)
        if (children[i])
          children[i]->copy_toarray(leaves, size);
      if (leaf)
        leaves[size++] = leaf;
    }
    trienode **children;
    T *leaf;
  };
  trienode *set_path(const char *str) const {
    trienode *node = root;
    while (*str != '\0')
      node = node->set_child(*str++);
    return node;
  }
  trienode *get_path(const char *str) const {
    trienode *node = root;
    while (node && *str != '\0')
      node = node->get_child(*str++);
    return node;
  }
  unsigned int nleaves;
  trienode *root;
};

#undef ALPHABET_OFFSET
#undef ALPHABET_LENGTH

#endif

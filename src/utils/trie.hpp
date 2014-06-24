#ifndef __TRIE_HPP__
#define __TRIE_HPP__

#include <cstring>

#define ALPHABET_OFFSET '0' //set which char is placed at the first position in child-array
#define ALPHABET_LENGTH 80 //read chars are mapped into integers belonging to [ALPHABET_OFFSET, ALPHABET_OFFSET+ALPHABET_LENGTH-1]

template <typename T> class trie {
	public:
		trie() : nleaves(0) { root = new trienode; }
		~trie() { delete root; }
		T *insert(const char *str) {
			trienode *node = set_path(str);
			if(!node->leaf)
				node->leaf = new T(str), //NOTE T needs to have a constructor like T(char*)
				++nleaves;
			return node->leaf;
		}
		T *lookup(const char *str) const {
			trienode *node = get_path(str);
			return node ? node->leaf : NULL;
		}
		T **get_leaves() const {
			if(nleaves==0)
				return NULL;
			else {
				unsigned int size = 0;
				T **leaves = new T*[nleaves];
				root->copy_toarray(leaves, size);
				return leaves;
			}
		}
		unsigned int get_nleaves() const {
			return nleaves;
		}
		bool is_empty() {
			return nleaves==0;
		}
	private:
		struct trienode {
			trienode() : leaf(NULL) { children = new trienode*[ALPHABET_LENGTH](); }
			~trienode() { for(int i=0; i<ALPHABET_LENGTH; ++i) delete children[i]; delete [] children, delete leaf; }
			trienode *set_child(const char ch) {
				int i = (ch-ALPHABET_OFFSET)%ALPHABET_LENGTH;
				if(!children[i])
					children[i] = new trienode;
				return children[i];
			}
			trienode *get_child(const char ch) {
				return children[(ch-ALPHABET_OFFSET)%ALPHABET_LENGTH];
			}
			void copy_toarray(T** leaves, unsigned int &size) {
				for(int i=0; i<ALPHABET_LENGTH; ++i)
					if(children[i])
						children[i]->copy_toarray(leaves, size);
				if(leaf)
					leaves[size++] = leaf;
			}
			trienode **children;
			T *leaf;
		};
		trienode *set_path(const char *str) const {
			trienode *node = root;
			while(*str!='\0') node = node->set_child(*str++);
			return node;
		}
		trienode *get_path(const char *str) const {
			trienode *node = root;
			while(node && *str!='\0') node = node->get_child(*str++);
			return node;
		}
		unsigned int nleaves;
		trienode *root;
};

#undef ALPHABET_OFFSET
#undef ALPHABET_LENGTH

#endif

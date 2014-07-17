#ifndef __MAXHEAP_HPP__
#define __MAXHEAP_HPP__

/*! \file maxheap.hpp
 * \brief Implementation of max-heap data structure
 */

#include <cstdlib>
#include <cmath>

/*! \class mahheap
 *  \brief max-heap implementation with key of type float
 */
template<typename val_t, typename key_t>class maxheap {
	public:
		/** \brief default constructor
		 * @param initsize set the initial size of the data structure if available
		 */
		maxheap(size_t initsize=0) {
			memsize = initsize>1 ? initsize : 1;
			arr = (item*)malloc(sizeof(item)*memsize);
			arrsize = 0,
			arr[0] = item(INFINITY);
		}
		~maxheap() {
			free(arr);
		}
		/** \brief return true if heap is empty
		 */
		bool is_notempty() const {
			return arrsize!=0;
		}
		/** \brief return numebr of items stored in the heap
		 */
		size_t get_size() const {
			return arrsize;
		}
		/** \brief push a new element in the heap and resize the data structure if it is full
		 * @param key ordering key of the new element
		 * @param val value of the new element
		 */
		void push(const key_t &key, const val_t &val) {
			if(++arrsize==memsize) {
				memsize = 2*memsize+1;
				arr = (item*)realloc(arr, sizeof(item)*memsize);
			}
			arr[arrsize] = item(key, val);
			size_t curr;
			for(curr=arrsize; key>arr[curr>>1].key; curr >>= 1)
				arr[curr] = arr[curr>>1];
			arr[curr] = item(key, val);
		}
		/** \brief remove the element on the top of the heap, i.e. the element with max key value
		 */
		void pop() {
			const item &last = arr[arrsize--];
			size_t child, curr;
			for(curr=1; curr<<1<=arrsize; curr=child) {
				child = curr<<1;
				if(child<arrsize && arr[child+1].key>arr[child].key) ++child;
				if(last.key<arr[child].key) arr[curr] = arr[child]; else break;
			}
			arr[curr] = last;
		}
		/** \brief ref to the top element
		 */
		val_t &top() const {
			return arr[1].val;
		}
	protected:
		struct item {
			item(key_t key) : key(key) {}
			item(key_t key, val_t val) : key(key), val(val) {}
			key_t key;
			val_t val;
		};
		item *arr;
		size_t arrsize, memsize;
};

#endif

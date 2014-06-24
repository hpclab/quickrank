#ifndef __MAXHEAP_HPP__
#define __MAXHEAP_HPP__

#include <cstdlib>
#include <cmath>

template<typename val_t>class maxheap {
	public:
		maxheap(size_t initsize=0) {
			memsize = initsize>1 ? initsize : 1;
			arr = (item*)malloc(sizeof(item)*memsize);
			arrsize = 0,
			arr[0] = item(INFINITY);
		}
		~maxheap() {
			free(arr);
		}
		bool is_notempty() const {
			return arrsize!=0;
		}
		size_t get_size() const {
			return arrsize;
		}
		void push(const float key, const val_t &val) {
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
		val_t &top() const {
			return arr[1].val;
		}
	protected:
		struct item {
			item(float key) : key(key) {}
			item(float key, val_t val) : key(key), val(val) {}
			float key;
			val_t val;
		};
		item *arr;
		size_t arrsize, memsize;
};

#endif

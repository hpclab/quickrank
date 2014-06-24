#ifndef __SYMMATRIX_HPP__
#define __SYMMATRIX_HPP__

#include <cstdlib>

#define sm2v(i,j,size) ((i)*(size)-((i)-1)*(i)/2+(j)-(i))

template <typename T> class symmatrix {
	public:
		symmatrix(size_t size) : size(size) {
			data = size>0 ? new T[size*(size+1)/2]() : NULL;
		}
		~symmatrix() {
			delete [] data;
		}
		T &at(const size_t i, const size_t j) {
			return data[i<j ? sm2v(i,j,size) : sm2v(j,i,size)];
		}
		T at(const size_t i, const size_t j) const {
			return data[i<j ? sm2v(i,j,size) : sm2v(j,i,size)];
		}
		T &at(const size_t i) {
			return data[i];
		}
		T at(const size_t i) const {
			return data[i];
		}
		T *vectat(const size_t i, const size_t j) {
			return &data[i<j ? sm2v(i,j,size) : sm2v(j,i,size)];
		}
	private:
		T *data;
		size_t size;
};

#undef sm2v

#endif

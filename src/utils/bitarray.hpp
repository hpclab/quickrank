#ifndef __BITARRAY_HPP__
#define __BITARRAY_HPP__

/*! \file bitarray.hpp
 * \brief Implementation of a bit array
 */

#include <cstdint>

/*! \def DIV32(i)
 * \brief integer division of \a i by 32 by means of the bit-wise SHIFT operator
 */
#define DIV32(i) ((i)>>5)
/*! \def MUL32(i)
 * \brief multiply \a i by 32 by means of the bit-wise SHIFT operator
 */
#define MUL32(i) ((i)<<5)
/*! \def MOD32(i)
 * \brief return \a i mod 32 by means of the bit-wise AND operator
 */
#define MOD32(i) ((i)&0x1F)

/*! return the number of set bit in a 32bits value
 *  @param n input value
 */
inline int32_t bitcounter(int32_t n) {
	#ifdef __GNUC__
		return __builtin_popcount(n);
	#else
		n = n-((n>>1)&0x55555555);
		n = (n&0x33333333)+((n >> 2)&0x33333333);
		return (((n+(n>>4))&0xF0F0F0F)*0x1010101)>>24;
	#endif
}

/*! \class bitarray
 *  \brief bit array implementation (1 bit per element)
 */
class bitarray {
	public:
		/** \brief default constructor
		 */
		bitarray() : data(NULL), datasize(0) {}
		~bitarray() {
			free(data);
		}
		/** \brief set the \a i-th bit (the data structure is reallocated to store the \a i-th bit if needed)
		 */
		void set_up(const unsigned int i) {
			if(i>=MUL32(datasize)) {
				unsigned int newdatasize = DIV32(2*i)+1;
				data = (int32_t*)realloc(data, sizeof(int32_t)*newdatasize);
				while(datasize<newdatasize) data[datasize++] = 0x00000000;
			}
			data[DIV32(i)] |= 1<<MOD32(i);
		}
		/** \brief return true the \a i-th bit is set (no check is made on the size of the array)
		 */
		bool is_up(const unsigned int i) const {
			return (data[DIV32(i)]>>MOD32(i))&1;
		}
		/** \brief return the number of set bit in the array
		 */
		unsigned int get_upcounter() {
			unsigned int count=0;
			for(unsigned int i=0; i<datasize; ++i)
				count += bitcounter(data[i]);
			return count;
		}
		/** \brief return an array of integers made up of the set bits positions
		 */
		unsigned int *get_uparray(const unsigned int n) {
			unsigned int *arr = new unsigned int[n], arrsize=0;
			for(unsigned int i=0; i<datasize && arrsize<n; ++i)
				for(unsigned int j=0; j<32 && arrsize<n; ++j)
					if((data[i]>>j)&1) arr[arrsize++] = MUL32(i)+j;
			return arr;
		}
		/** \brief compute bitwse OR of two bit arrays and store the result in the left operand
		 */
		bitarray& operator|= (const bitarray& other) {
			if(datasize<other.datasize) {
				data = (int32_t*)realloc(data, sizeof(int32_t)*other.datasize);
				for(unsigned int i=0; i<datasize; ++i)
					data[i] |= other.data[i];
				for(unsigned int i=datasize; i<other.datasize; ++i)
					data[i] = other.data[i];
				datasize = other.datasize;
			} else
				for(unsigned int i=0; i<other.datasize; ++i)
					data[i] |= other.data[i];
			return *this;
		}
	private:
		int32_t *data;
		unsigned int datasize;
};

#undef DIV32
#undef MUL32
#undef MOD32

#endif

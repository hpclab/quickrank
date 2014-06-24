#ifndef __BITARRAY_HPP__
#define __BITARRAY_HPP__

#include <cstdint>

#define DIV32(i) ((i)>>5)
#define MUL32(i) ((i)<<5)
#define MOD32(i) ((i)&0x1F)

inline int32_t bitcounter(int32_t n) {
	#ifdef __GNUC__
		return __builtin_popcount(n);
	#else
		n = n-((n>>1)&0x55555555);
		n = (n&0x33333333)+((n >> 2)&0x33333333);
		return (((n+(n>>4))&0xF0F0F0F)*0x1010101)>>24;
	#endif
}

class bitarray {
	public:
		bitarray() : data(NULL), datasize(0) {}
		~bitarray() {
			free(data);
		}
		void set_up(const unsigned int i) {
			if(i>=MUL32(datasize)) {
				unsigned int newdatasize = DIV32(2*i)+1;
				data = (int32_t*)realloc(data, sizeof(int32_t)*newdatasize);
				while(datasize<newdatasize) data[datasize++] = 0x00000000;
			}
			data[DIV32(i)] |= 1<<MOD32(i);
		}
		bool is_up(const unsigned int i) const {
			return (data[DIV32(i)]>>MOD32(i))&1;
		}
		unsigned int get_upcounter() {
			unsigned int count=0;
			for(unsigned int i=0; i<datasize; ++i)
				count += bitcounter(data[i]);
			return count;
		}
		unsigned int *get_uparray(const unsigned int n) {
			unsigned int *arr = new unsigned int[n], arrsize=0;
			for(unsigned int i=0; i<datasize && arrsize<n; ++i)
				for(unsigned int j=0; j<32 && arrsize<n; ++j)
					if((data[i]>>j)&1) arr[arrsize++] = MUL32(i)+j;
			return arr;
		}
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

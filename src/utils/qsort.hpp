#ifndef __QSORT_HPP__
#define __QSORT_HPP__

template <typename T> void inline swap(T &a, T &b) {
	T tmp = a;
	a = b;
	b = tmp;
}

template <typename T> void qsort(T *arr, const int size) {
	//descending order
	int stack[size];
	int top = 1;
	stack[0] = 0,
	stack[1] = size-1;
	while(top>=0) {
		int h = stack[top--];
		int l = stack[top--];
		T p = arr[h];
		int i = l-1;
		for(int j=l; j<h; ++j)
			if(p<arr[j]) swap(arr[++i], arr[j]);
		swap(arr[++i], arr[h]);
 		if(i-1>l) {
			stack[++top] = l;
			stack[++top] = i-1;
		}
		if(i+1<h) {
			stack[++top] = i+1;
			stack[++top] = h;
		}
	}
}

#endif

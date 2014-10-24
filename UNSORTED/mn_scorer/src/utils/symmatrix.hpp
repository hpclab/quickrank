#ifndef __SYMMATRIX_HPP__
#define __SYMMATRIX_HPP__

/*! \file symmatrix.hpp
 * \brief implement a symetric matrix of order n by using n(n+1)/2 elements
 */

#include <cstdlib>

/*! \def sm2v(i,j,size)
 * \brief map a 2D square matrix coorinate pair (\a i, \a j) into a 1D array coordinate; \a size denotes the size of the matrix.
 */
#define sm2v(i,j,size) ((i)*(size)-((i)-1)*(i)/2+(j)-(i))

/*! \class symmatrix
 *  \brief symmetric matrix implementation
 */
template<typename T> class SymMatrix {
 public:
  /** \brief default constructor: allocate an array of size \a size*(\a size+1)/2.
   * @param size order of the matrix.
   */
  SymMatrix(size_t size)
      : size(size) {
    data = size > 0 ? new T[size * (size + 1) / 2]() : NULL;
  }
  ~SymMatrix() {
    delete[] data;
  }
  /** \brief return the element at position ( \a i, \a j ) for left-hand operation.
   * @param i row in [0 .. \a size -1]
   * @param j column in [0 .. \a size -1]
   */
  T &at(const size_t i, const size_t j) {
    return data[i < j ? sm2v(i, j, size) : sm2v(j, i, size)];
  }
  /** \brief return the element at position ( \a i, \a j ) for right-hand operation.
   * @param i row in [0 .. \a size -1]
   * @param j column in [0 .. \a size -1]
   */
  T at(const size_t i, const size_t j) const {
    return data[i < j ? sm2v(i, j, size) : sm2v(j, i, size)];
  }
  /** \brief return the element at position \a i of the array representing the matrix for left-hand operation.
   * @param i index in [0 .. \a size -1]
   */
  T &at(const size_t i) {
    return data[i];
  }
  /** \brief return the element at position \a i of the array representing the matrix for right-hand operation.
   * @param i index in [0 .. \a size -1]
   */
  T at(const size_t i) const {
    return data[i];
  }
  /** \brief return the pointer to the element at position ( \a i, \a j ) of the array representing the matrix.
   * @param i row in [0 .. \a size -1]
   * @param j column in [0 .. \a size -1]
   */
  T *vectat(const size_t i, const size_t j) {
    return &data[i < j ? sm2v(i, j, size) : sm2v(j, i, size)];
  }
  /** \brief return matrix size.
   */
  size_t get_size() const {
    return size;
  }
 private:
  T *data;
  size_t size;
};

#undef sm2v

#endif

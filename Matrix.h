
#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cstddef>

class Matrix{
	size_t m_rows,m_cols;
	std::vector<float> m_mat;
	public:
	Matrix(size_t rows, size_t cols);
	Matrix(size_t rows, size_t cols,std::vector<float>& data);
	Matrix(size_t rows, size_t cols,float* data);
	Matrix(const Matrix& other);
	friend void swap(Matrix& first, Matrix& second);
	
	void print() const; 
	size_t getNumRows() const;
	size_t getNumCol() const;
	float getAt(size_t row, size_t col) const {return m_mat[row*m_cols + col];}
	void setAt(size_t row, size_t col, float val) {m_mat[row*m_cols + col] = val;}
	const std::vector<float>& getMatrix() const;

	Matrix transpose() const;
	Matrix operator*(float s) const;
	Matrix operator*(const Matrix& other) const;
	Matrix& operator=(Matrix other);
	Matrix operator+(const Matrix& other) const;
	Matrix operator-(const Matrix& other) const;
	Matrix hadamard(const Matrix& other) const;

	void map(float (*op)(float));

};

#endif /* MATRIX_H */
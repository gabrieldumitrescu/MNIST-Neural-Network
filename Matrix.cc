#include "Matrix.h"

Matrix::Matrix(size_t rows, size_t cols):
	m_rows(rows),m_cols(cols),m_mat(rows*cols,0.0f)
{
}

Matrix::Matrix(size_t rows, size_t cols, std::vector<float>& data):
	m_rows(rows),m_cols(cols),m_mat(data)
{
}


Matrix::Matrix(size_t rows, size_t cols, float* data):
	m_rows(rows),m_cols(cols),m_mat(data, data+rows*cols)
{
}

Matrix::Matrix(const Matrix& other):
	m_rows(other.getNumRows()), m_cols(other.getNumCol()),
	m_mat(other.getMatrix())
{
}

const std::vector<float>& Matrix::getMatrix() const
{
	return m_mat;
}

void swap(Matrix& first, Matrix& second)
{
	using std::swap;

	swap(first.m_rows,second.m_rows);
	swap(first.m_cols,second.m_cols);
	swap(first.m_mat,second.m_mat);
}


Matrix& Matrix::operator=(Matrix other)
{
	swap(*this,other);
	return *this;
}

Matrix Matrix::operator*(const Matrix& other) const
{
	size_t ncols=other.getNumCol();
	Matrix result(m_rows,ncols);
	for (size_t i = 0; i < m_rows; ++i)
	{
		for (size_t j = 0; j < ncols; ++j)
		{
			float val=0;
			for (size_t k = 0; k < m_cols; ++k)
			{
				val+=getAt(i,k) * other.getAt(k,j);
			}
			result.setAt(i,j,val);
		}
		
	}	
	return result;
}

Matrix Matrix::operator*(float s) const
{
	Matrix result(m_rows,m_cols);
	for (size_t i = 0; i < m_rows; ++i)
	{
		for (size_t j = 0; j < m_cols; ++j)
		{
			result.setAt(i,j,getAt(i,j) * s);
		}
		
	}	
	return result;
}


Matrix Matrix::operator+(const Matrix& other) const
{
	Matrix result(m_rows,m_cols);
	for (size_t i = 0; i < m_rows; ++i)
	{
		for (size_t j = 0; j < m_cols; ++j)
		{
			result.setAt(i,j,getAt(i,j) + other.getAt(i,j));
		}
	}	
	return result;
}


Matrix Matrix::operator-(const Matrix& other) const
{
	Matrix result(m_rows,m_cols);
	for (size_t i = 0; i < m_rows; ++i)
	{
		for (size_t j = 0; j < m_cols; ++j)
		{
			result.setAt(i,j,getAt(i,j) - other.getAt(i,j));
		}
	}	
	return result;
}

//both the curent and the other matrix must be column vectors
Matrix Matrix::hadamard(const Matrix& other) const
{
	Matrix result(m_rows,m_cols);
	for (size_t i = 0; i < m_rows; ++i)
	{
		result.setAt(i,0,getAt(i,0) * other.getAt(i,0));
	}
	return result;

}

// float Matrix::getAt(size_t row, size_t col) const
// {
// 	return m_mat[row*m_cols + col];
// }

// void Matrix::setAt(size_t row, size_t col, float val)
// {
// 	m_mat[row*m_cols + col] = val;
// }

Matrix Matrix::transpose() const
{
	Matrix result(m_cols,m_rows);
	for (size_t i = 0; i < m_rows; ++i)
	{
		for (size_t j = 0; j < m_cols; ++j)
		{
			float val = getAt(i,j);
			result.setAt(j,i,val);
		}
	}
	return result;

}

void Matrix::print() const
{
	for (size_t i = 0; i < m_rows; ++i)
	{
		for (size_t j = 0; j < m_cols; ++j)
		{
			printf("%.2f ", m_mat[i*m_cols + j]);
		}
		printf("\n");
	}
	puts("");
}

size_t Matrix::getNumRows() const
{
	return m_rows;
}

size_t Matrix::getNumCol() const
{
	return m_cols;
}


void Matrix::map(float (*op)(float))
{
	for(size_t i = 0; i<m_mat.size(); ++i)
		m_mat[i] = op(m_mat[i]);
}
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Returns j mod n. Note that this C/C++ provides
// already the % operator. This is equivalente to `modulo' for
// positive `j' but not for negative values. 
int modulo(int j,int n) {
  int r = j%n;
  if (r<0) r += n;
  return r;
}

void matpow2(MatrixXd &A,MatrixXd &Ak,int k) {
  // Brute force approach
  int m = A.rows();
  assert(A.cols()==m);
  Ak = MatrixXd::Identity(m,m);
  for (int j=0; j<k; j++) Ak *= A;
}

void matpow(MatrixXd &A,MatrixXd &Ak,int k) {
  int m = A.rows();
  assert(A.cols()==m);
  if (k==0) Ak = MatrixXd::Identity(m,m);
  else if (k==1) Ak = A;
  else {
    MatrixXd Z;
    int r = k%2;
    int k2 = k/2;
    matpow(A,Z,k2);
    Ak = Z*Z;
    if (r) Ak *= A;
  }
}

void apply_poly(VectorXd &coefs, 
                MatrixXd &X,
                MatrixXd &PX) {
  int m = X.rows();
  assert(X.cols()==m);
  PX = 0*X;
  MatrixXd Id = MatrixXd::Identity(m,m);
  int N = coefs.size();
  for (int k=N-1; k>=0; k--) 
    PX = PX*X + coefs(k)*Id;
}

void test1() {
  int m=10;
  MatrixXd A(m,m),Akl,Akbf;
  for (int j=0; j<m; j++) 
    for (int k=0; k<m; k++)
      A(j,k) = 1+(j-k)/10.0;
  cout << A << endl;
  int k=5;
  matpow(A,Akl,k);
  cout << "A^" << k << " (Akl=log2 algo): "
       << endl << Akl << endl;

  matpow2(A,Akbf,k);
  cout << "A^" << k << " (Akbf=brute force algo): "
       << endl << Akbf << endl;
  MatrixXd error;
  error = Akl-Akbf;
  cout << "||Akl-Akbf||: " << error.norm() << endl;
}

void test2() {
  // Computes the exponential of a matrix 
  // The coefficients are 1,1,1/2,1/6,1/24,...,1/n!
  int N=2;
  MatrixXd X(2,2),PX;
  X << 0, 1, 1 ,0;
  int M=50;
  VectorXd coefs(M);
  coefs[0] = 1;
  for (int j=1; j<M; j++) coefs(j) = coefs(j-1)/j;
  apply_poly(coefs,X,PX);
  cout << "X" << endl << X << endl << endl;
  cout << "PX (=exp(X)): " << endl << PX << endl;
}
  
#define MP(x)  cout << #x ":" << endl << x << endl

void test3() {
  int m=5;
  MatrixXd A(m,m), D, Dinv, X, PX, invA;
  VectorXd d;
  // Computes the inverse of A matrix
  // using the series for 1/(1-X)
  for (int j=0; j<m; j++) {
    A(j,j) = 2.1;
    A(j,modulo(j+1,m)) = -1.0;
    A(j,modulo(j-1,m)) = -1.0;
  }
  
  MP(A);
  // Makes: X = inv(D) * (D-A)
  X = A;
  for (int j=0; j<m; j++) {
    double ajj = A(j,j);
    for (int k=0; k<m; k++) X(j,k) = -A(j,k)/ajj;
    X(j,j) = 0.0;
  }
  MP(X);
  int M=300;
  VectorXd coefs(M);
  for (int j=0; j<M; j++) coefs(j) = 1.0;
  apply_poly(coefs,X,PX);
  // Makes: inv(A) = PX*inv(D)
  MP(PX);
  invA = PX;
  for (int j=0; j<m; j++) 
    for (int k=0; k<m; k++) invA(j,k) = PX(j,k)/A(k,k);
  MP(invA);
}

int main() {
  test1();
  test2();
  test3();
  return 0;
}

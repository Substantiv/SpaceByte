#include <cstddef>
#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

void useResize();
void useMap();

int main()
{
    useMap();
    return 0;
}

void useResize(){
    /*
    resize()和conservativeResize()使用
    * 如果resize()前后变化的唯独不一样,会调用析构函数
    * 要想使用conservativeResize()保留原来数据后进行扩维, 应该放到resize()之后,resize()会对原来矩阵进行析构
    */
    MatrixXd mat_resize(2, 3);
    mat_resize << 1,2,3,4,5,6;

    cout << "原始矩阵：" << endl;
    cout << mat_resize << endl;

    mat_resize.conservativeResize(3,7);

    cout << "使用conservativeResize()：" << endl;
    cout<<mat_resize <<endl;

    mat_resize.resize(3,7);

    cout << "使用resize()：" << endl;
    cout<<mat_resize <<endl;
}

void useMap(){
    typedef Matrix<float,1,Dynamic> MatrixType;
    typedef Map<MatrixType> MapType;
    typedef Map<const MatrixType> MapTypeConst;   // a read-only map
    const int n_dims = 5;
    
    MatrixType m1(n_dims), m2(n_dims);
    m1.setRandom();
    m2.setRandom();
    float *p = &m2(0);  // get the address storing the data for m2
    cout << *p << endl;
    MapType m2map(p,m2.size());   // m2map shares data with m2
    MapTypeConst m2mapconst(p,m2.size());  // a read-only accessor for m2
    cout << "m1: " << m1 << endl;
    cout << "m2: " << m2 << endl;
    cout << "Squared euclidean distance: " << (m1-m2).squaredNorm() << endl;
    cout << "Squared euclidean distance, using map: " << (m1-m2map).squaredNorm() << endl;
    m2map(3) = 7;   // this will change m2, since they share the same array
    cout << "Updated m2: " << m2 << endl;
    cout << "m2 coefficient 2, constant accessor: " << m2mapconst(2) << endl;
    /* m2mapconst(2) = 5; */   // this yields a compile-time error
}
#include <cstddef>
#include <iostream>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

int main()
{
    Vector3i test = Vector3i::Ones();
    cout<<test<<endl;

    return 0;
}

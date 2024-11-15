#include <iostream>
#include <vector>

// size_t demo
void size_tExample(){
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // use size_t to iteration over the vector
    for (size_t i = 0; i < vec.size(); i++)
    {
        std::cout << "Element: " << i;
    }

    // Dynamically allocate memory
    size_t num = 10;
    int *arr = new int[num];
}

int main(int argc, char const *argv[])
{

    size_tExample();

    return 0;
}

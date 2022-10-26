#include "../../include/Random/UniformRandom.h"

UniformRandom::UniformRandom(unsigned int seed): engin(seed),uniform_creator(0,1)
{

}

float UniformRandom::Random(float lambda)
{
    return uniform_creator(engin);
}

#include "ModelAnalyze/ModelAnalyzerIterator.h"

ModelAnalyzerIterator::ModelAnalyzerIterator(GraphNode *p)
{
    _ptr = p;
}

bool ModelAnalyzerIterator::operator!=(const ModelAnalyzerIterator &iter)
{
    return _ptr != iter._ptr;
}

bool ModelAnalyzerIterator::operator==(const ModelAnalyzerIterator &iter)
{
    return _ptr == iter._ptr;
}

ModelAnalyzerIterator &ModelAnalyzerIterator::operator++()
{
    _ptr++;
    return *this;
}

ModelAnalyzerIterator ModelAnalyzerIterator::operator++(int)
{
    ModelAnalyzerIterator tmp = *this;
    _ptr++;
    return tmp;
}

GraphNode &ModelAnalyzerIterator::operator*()
{
    return *_ptr;
}

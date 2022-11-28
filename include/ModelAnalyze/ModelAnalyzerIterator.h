#ifndef __MODELANALYZERITERATOR_H__
#define __MODELANALYZERITERATOR_H__

#include "GraphNode.h"

class ModelAnalyzerIterator
{
public:
    /// @brief
    /// @param p
    ModelAnalyzerIterator(GraphNode *p);

    /// @brief
    /// @param iter
    /// @return
    bool operator!=(const ModelAnalyzerIterator &iter);

    /// @brief
    /// @param iter
    /// @return
    bool operator==(const ModelAnalyzerIterator &iter);

    /// @brief
    /// @return
    ModelAnalyzerIterator &operator++();

    /// @brief
    /// @param
    /// @return
    ModelAnalyzerIterator operator++(int);

    GraphNode &operator*();

private:
    GraphNode *_ptr;
};

#endif // __MODELANALYZERITERATOR_H__a
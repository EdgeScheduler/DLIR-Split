#ifndef __SAFEQUEUE_H__
#define __SAFEQUEUE_H__

#include <deque>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

template <class T>
class SafeQueue
{
public:
    SafeQueue(int capacity = -1) : capacity(capacity), size(0)  {}
    ~SafeQueue() {}

    /// @brief returns a read/write reference to the data at the front-element of the qeque.
    /// @return
    T &front();

    /// @brief add an element to queue tail.
    /// @param x element to add-into
    void Push(T &x);

    /// @brief add an element to queue tail.
    /// @param x element to add-into
    void Emplace(T &&x);

    /// @brief pop the front element of the queue
    /// @return rvalue of the first-element
    T Pop();

    /// @brief is the queue empty
    /// @return
    bool Empty() const;
    /// @brief
    /// @return
    bool Full() const;

private:
    /// @brief current item count of queue.
    int size;

    /// @brief the capacity of queue, -1 means no limit.
    int capacity;

    /// @brief used to save items.
    std::deque<T> dequeDatas;

    /// @brief for lock.
    std::mutex mutex;
    /// @brief add restrictions on producers-threads
    std::condition_variable m_notFull;
    /// @brief add restrictions on consumer-threads
    std::condition_variable m_notEmpty;
};

template <class T>
void SafeQueue<T>::Push(T &x)
{
    std::unique_lock<std::mutex> lock(mutex);
    //有一个判断谓词,等价于while(!Pred){m_notFull.wait()}
    m_notFull.wait(lock, [this]() -> bool
                   { return capacity < 0 || size < capacity; });
    dequeDatas.push_back(x);
    size++;

    // relese lock
    lock.unlock();

    m_notEmpty.notify_all();
}

template <class T>
void SafeQueue<T>::Emplace(T &&x)
{
    std::unique_lock<std::mutex> lock(mutex);
    //有一个判断谓词,等价于while(!Pred){m_notFull.wait()}
    m_notFull.wait(lock, [this]() -> bool
                   { return capacity < 0 || size < capacity; });
    dequeDatas.emplace_back(std::move(x));
    size++;

    // relese lock
    lock.unlock();

    m_notEmpty.notify_all();
}

template <class T>
T SafeQueue<T>::Pop()
{
    std::unique_lock<std::mutex> lock(mutex);
    m_notEmpty.wait(lock, [this]() -> bool
                    { return size > 0; });
    T front = std::move(dequeDatas.front());
    dequeDatas.pop_front();
    size--;

    // release lock
    lock.unlock();

    m_notFull.notify_all();

    return std::move(front);
}

template <class T>
T &SafeQueue<T>::front()
{
    std::unique_lock<std::mutex> lock(mutex);
    m_notEmpty.wait(lock, [this]() -> bool
                    { return size > 0; });
    lock.unlock();

    return dequeDatas.front();
}

template <class T>
bool SafeQueue<T>::Empty() const
{
    return size<1;
}

template <class T>
bool SafeQueue<T>::Full() const
{
    // if capacity==-1, always return false.
    return size >= capacity;
}

#endif // __SAFEQUEUE_H__
// #ifndef __SAFEQUEUE_H__
// #define __SAFEQUEUE_H__

// #include <deque>
// #include <thread>
// #include <queue>
// #include <mutex>
// #include <condition_variable>

// template <class T>
// class SafeQueue
// {
//     SafeQueue(int maxSize) : capacity(maxSize), size(0){}
//     ~SafeQueue(){}

//     void Push(const T &x)
//     {
//         std::unique_lock<std::mutex> lock(mutex);
//         //有一个判断谓词,等价于while(!Pred){m_notFull.wait()}
//         m_notFull.wait(lock, std::bind(&CBoundedQueue::CanPut, this));
//         dequeDatas.push_back(x);
//         ++size;
//         if (size > capacity)
//         {
//             size = capacity;
//         }

//         //手动释放锁，减小锁的范围
//         lock.unlock();

//         // m_notEmpty.notify_one();
//         m_notEmpty.notify_all();
//     }

//     T Pop()
//     {
//         std::unique_lock<std::mutex> lock(mutex);
//         //有一个判断谓词,等价于while(!Pred){m_notEmpty.wait()}
//         m_notEmpty.wait(lock, std::bind(&CBoundedQueue::CanGet, this));
//         T front(dequeDatas.front());
//         dequeDatas.pop_front();
//         --size;
//         //手动释放锁，减小锁的范围
//         lock.unlock();

//         // m_notFull.notify_one();
//         m_notFull.notify_all();
//         return front;
//     }

//     bool Empty() const
//     {
//         std::lock_guard<std::mutex> lock;
//         return dequeDatas.empty();
//     }

//     bool Full() const
//     {
//         std::lock_guard<std::mutex> lock;
//         return size == capacity;
//     }

// private:
//     //判断谓词
//     bool CanPut() { return size < capacity; }
//     bool CanGet() { return size > 0; }

// private:

//     /// @brief current item count of queue.
//     int size;

//     /// @brief the capacity of queue, -1 means no limit.
//     int capacity;
    
//     /// @brief save items
//     std::deque<T> dequeDatas;

//     //互斥量，对队列进行同步保护
//     std::mutex mutex;
//     //用于限制生产者线程
//     std::condition_variable m_notFull;
//     //用于限制消费者线程
//     std::condition_variable m_notEmpty;
// };

// #endif // __SAFEQUEUE_H__
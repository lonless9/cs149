# CS149: 并行计算 - 第15讲：锁的实现、细粒度同步与无锁编程

本讲介绍并行程序中同步机制的底层实现原理，从硬件原子指令到高级同步抽象，并讨论细粒度锁定策略和无锁编程技术，这些是构建高性能并发系统的关键基础。

## 1. 并发系统中的问题状态

在讨论同步机制前，我们先明确并发系统可能遇到的几种关键问题状态。

### 1.1 死锁 (Deadlock)

**死锁**是并发系统中最严重的问题状态之一，表现为系统中有待完成的操作，但没有任何操作能够取得进展。

**定义**：系统中各个部分互相等待对方持有的资源，形成循环依赖，导致所有相关操作永久阻塞。

**死锁的必要条件（Coffman条件）**：
1. **互斥(Mutual Exclusion)**：资源一次只能被一个进程使用
2. **持有并等待(Hold and Wait)**：进程持有资源的同时等待获取其他资源
3. **非抢占(No Preemption)**：资源只能由持有者自愿释放，不能被强制剥夺
4. **循环等待(Circular Wait)**：存在一个等待进程的循环链，每个进程都在等待链中下一个进程持有的资源

**死锁示例**：
```
// 线程1
lock(A);
lock(B);
// 使用资源A和B
unlock(B);
unlock(A);

// 线程2 (同时执行)
lock(B);
lock(A);
// 使用资源B和A
unlock(A);
unlock(B);
```

如果线程1获得了锁A，线程2获得了锁B，然后两个线程都尝试获取对方持有的锁，就会发生死锁。

### 1.2 活锁 (Livelock)

**活锁**是一种系统在执行大量操作，但没有线程取得有意义进展的状态。

**定义**：线程不断地改变状态或重试操作，但整体系统状态没有向前推进，类似于两个人在走廊相遇时不断互相避让但始终无法通过的情景。

**活锁示例**：
```
// 资源争用解决方案
while (true) {
    if (tryLock(resource)) {
        // 使用资源
        unlock(resource);
        break;
    } else {
        // 如果失败，随机等待然后重试
        randomBackoff();
    }
}
```

如果多个线程执行这段代码，且它们的随机回退时间恰好同步，可能导致它们重复地同时尝试获取资源、失败、等待，形成活锁。

### 1.3 饥饿 (Starvation)

**饥饿**表示系统整体在取得进展，但某些进程/线程无法获得所需资源，因此无法前进。

**定义**：由于资源分配策略不公平，导致某些线程长时间（甚至无限期）无法获得所需资源的情况。

**饥饿示例**：
- 读者优先的读写锁中，如果读者持续到来，写者可能永远无法获得锁
- 优先级调度中，低优先级任务在高负载下可能永远得不到执行

与死锁和活锁不同，饥饿是一个公平性问题，通常不是永久状态（理论上饥饿的线程最终可能获得资源）。

## 2. 锁的实现机制

锁是最基本的并发控制机制，用于保护共享资源。下面我们探讨如何在硬件支持的基础上实现高效的锁。

### 2.1 硬件原子操作

现代处理器提供了多种原子操作指令，作为实现同步机制的基础：

#### 2.1.1 Test-and-Set (TAS)

```c
// 原子操作伪代码
bool TestAndSet(bool *target) {
    bool oldValue = *target;
    *target = true;
    return oldValue;
}
```

**特点**：原子地读取内存地址的值，并将其设为真。返回读取的旧值。

#### 2.1.2 Compare-and-Swap (CAS)

```c
// 原子操作伪代码
bool CompareAndSwap(int *addr, int expected, int newValue) {
    if (*addr == expected) {
        *addr = newValue;
        return true;  // 交换成功
    }
    return false;     // 交换失败
}
```

**特点**：原子地比较内存值与期望值，如果相等则将内存更新为新值并返回成功，否则返回失败。

#### 2.1.3 Load-Linked/Store-Conditional (LL/SC)

某些架构（如ARM、MIPS、PowerPC）提供的另一种原子操作对：
- **Load-Linked** (LL)：读取内存并"监视"该地址
- **Store-Conditional** (SC)：尝试写入内存，仅当自上次LL以来该地址未被修改时才成功

这两条指令结合使用，可以实现与CAS类似的功能，但没有ABA问题（见后文）。

### 2.2 基于原子操作的锁实现

#### 2.2.1 简单Test-and-Set锁

```c
// 锁的数据结构
typedef struct {
    bool locked;
} tas_lock_t;

// 初始化
void init_tas_lock(tas_lock_t *lock) {
    lock->locked = false;
}

// 获取锁
void acquire_tas_lock(tas_lock_t *lock) {
    while (TestAndSet(&lock->locked)) {
        // 自旋等待
    }
}

// 释放锁
void release_tas_lock(tas_lock_t *lock) {
    lock->locked = false;
}
```

**问题**：
- **高一致性流量**：多个处理器持续尝试对锁变量执行TAS（写操作），每次失败的TAS仍需获取缓存行的独占所有权，导致缓存行在等待线程之间频繁失效和传输
- **性能随处理器数量增加而急剧下降**：主要因互连网络上的竞争加剧数据传输延迟
- **缺乏公平性**：无法保证线程按请求顺序获得锁

#### 2.2.2 改进：Test-and-Test-and-Set (TTAS) 锁

```c
void acquire_ttas_lock(tas_lock_t *lock) {
    while (1) {
        // 先读取锁状态（不修改）
        while (lock->locked) {
            // 本地自旋，无总线流量
        }
        // 当观察到锁可能被释放时，才尝试获取
        if (!TestAndSet(&lock->locked)) {
            break; // 获取锁成功
        }
    }
}
```

**改进**：
- **显著减少互连流量**：等待线程在本地缓存中自旋读取（共享状态），不产生总线流量
- **只有锁释放时才引发一致性事件**：当锁被释放（值变为false）时，所有等待者才因缓存无效化而更新值，然后才尝试TAS

**特点**：
- 无竞争延迟略高（需要先读再TAS）
- 互连流量大幅减少
- 更可扩展
- 仍然不保证公平性

#### 2.2.3 更公平的设计：Ticket Lock

```c
typedef struct {
    unsigned int next_ticket;
    unsigned int now_serving;
} ticket_lock_t;

void init_ticket_lock(ticket_lock_t *lock) {
    lock->next_ticket = 0;
    lock->now_serving = 0;
}

void acquire_ticket_lock(ticket_lock_t *lock) {
    // 原子获取并增加票号
    unsigned int my_ticket = __sync_fetch_and_add(&lock->next_ticket, 1);
    
    // 等待轮到自己
    while (lock->now_serving != my_ticket) {
        // 自旋等待
    }
}

void release_ticket_lock(ticket_lock_t *lock) {
    // 增加服务号
    lock->now_serving++;
}
```

**特点**：
- **FIFO公平性**：线程按获取票号顺序获得锁
- **低一致性流量**：获取锁时只需一次原子增量操作；等待时只读now_serving；解锁时只需普通写操作
- **更可预测的性能**：线程按确定顺序获得锁，避免饥饿

### 2.3 更高级的原子操作支持

现代系统提供更丰富的原子操作API：

#### 2.3.1 C++11 atomic<T>类型

```cpp
#include <atomic>

std::atomic<int> counter(0);  // 原子整数

// 原子增加操作
int old_value = counter.fetch_add(1, std::memory_order_seq_cst);

// 原子比较交换
int expected = 5;
bool success = counter.compare_exchange_strong(expected, 10);
```

**特点**：
- 提供类型安全的原子操作模板
- 支持多种内存序语义选项
- 通过硬件原子指令或内部锁实现

#### 2.3.2 使用CAS实现其他原子操作

任何复杂的原子操作都可以通过CAS实现：

```c
// 原子最小值实现
int atomic_min(int *addr, int val) {
    int old;
    do {
        old = *addr;
        if (val >= old) return old; // 如果新值不小于当前值，无需更新
    } while (!CompareAndSwap(addr, old, val)); // 尝试更新
    return old;
}
```

## 3. 细粒度锁定策略

在设计并发数据结构时，锁定粒度（锁保护的数据范围大小）是一个关键考量。

### 3.1 粗粒度锁vs细粒度锁

**粗粒度锁**：用单一锁保护整个数据结构
- **优点**：实现简单，容易保证正确性
- **缺点**：并发度低，所有操作都被串行化

**细粒度锁**：数据结构的不同部分使用不同的锁
- **优点**：提高并行度，允许对数据结构不同部分的并发操作
- **缺点**：实现复杂，容易引入死锁，额外内存开销

### 3.2 案例研究：链表的锁定策略

考虑一个简单的排序链表，支持插入、删除和查找操作：

```c
typedef struct node {
    int value;
    struct node *next;
} node_t;

typedef struct {
    node_t *head;
} list_t;
```

#### 3.2.1 问题：无同步的并发访问

如果没有适当的同步，多线程并发访问会导致：
- **丢失修改**：两个线程同时插入节点，一个修改可能被另一个覆盖
- **指针错误**：一个线程正在遍历链表时，另一个线程删除了节点
- **数据竞争**：对同一节点的并发读写导致不确定行为

#### 3.2.2 方案1：全局锁

```c
typedef struct {
    node_t *head;
    pthread_mutex_t lock;
} list_t;

void list_insert(list_t *list, int value) {
    pthread_mutex_lock(&list->lock);
    
    // 插入操作...
    
    pthread_mutex_unlock(&list->lock);
}

bool list_contains(list_t *list, int value) {
    pthread_mutex_lock(&list->lock);
    
    // 查找操作...
    
    pthread_mutex_unlock(&list->lock);
    return found;
}
```

**特点**：
- 操作简单清晰
- 所有操作都被串行化，在高负载下性能较差

#### 3.2.3 方案2：手递手锁定(Hand-over-Hand Locking)

```c
typedef struct node {
    int value;
    struct node *next;
    pthread_mutex_t lock;
} node_t;

typedef struct {
    node_t *head;
    pthread_mutex_t lock;
} list_t;

bool list_contains(list_t *list, int value) {
    pthread_mutex_lock(&list->lock);
    node_t *curr = list->head;
    if (curr == NULL) {
        pthread_mutex_unlock(&list->lock);
        return false;
    }
    
    pthread_mutex_lock(&curr->lock);
    pthread_mutex_unlock(&list->lock);
    
    while (curr != NULL) {
        if (curr->value == value) {
            pthread_mutex_unlock(&curr->lock);
            return true;
        }
        
        node_t *next = curr->next;
        if (next == NULL) {
            pthread_mutex_unlock(&curr->lock);
            return false;
        }
        
        pthread_mutex_lock(&next->lock);
        pthread_mutex_unlock(&curr->lock);
        curr = next;
    }
    
    return false;  // 不会到达这里
}
```

**手递手锁定原则**：
1. 先锁定节点的前驱
2. 然后锁定当前节点
3. 安全后释放前驱节点的锁
4. 重复此过程，锁如同"手递手"传递

**特点**：
- 允许对链表不同部分的并发操作
- 实现复杂，需要仔细管理锁获取顺序
- 每次遍历都需要加锁/解锁操作，开销较大
- 每个节点需要额外空间存储锁

**示例场景**：线程T1在链表头部操作，线程T2在链表尾部操作，可以并发进行：

```
List: [10] -> [20] -> [30] -> [40]
T1: 在头部删除节点10  (锁定head和节点10)
T2: 在尾部插入节点50  (锁定节点40和新节点50)
```

这两个操作可以并发执行，而在全局锁方案中将被串行化。

## 4. 无锁编程

无锁编程是一种不使用互斥锁的并发编程方法，通过原子操作和精心设计的算法来保证线程安全。

### 4.1 无锁vs锁定

**阻塞算法**（使用锁）：
- 如果持有锁的线程被挂起、崩溃或变慢，会阻塞其他线程

**无锁算法**：
- 保证系统整体总能取得进展（至少有一个线程能完成操作）
- 不使用互斥锁，避免锁相关问题
- 仍可能导致单个线程饥饿

### 4.2 无锁数据结构案例

#### 4.2.1 无锁队列：单生产者-单消费者

**使用场景**：只有一个线程添加元素，只有一个线程移除元素

**有界队列实现（循环数组）**：
```c
typedef struct {
    int items[SIZE];
    int head;  // 只被消费者修改
    int tail;  // 只被生产者修改
} spsc_queue_t;

bool enqueue(spsc_queue_t *q, int item) {
    int next_tail = (q->tail + 1) % SIZE;
    if (next_tail == q->head) return false;  // 队列满
    
    q->items[q->tail] = item;
    q->tail = next_tail;  // 原子写
    return true;
}

bool dequeue(spsc_queue_t *q, int *item) {
    if (q->head == q->tail) return false;  // 队列空
    
    *item = q->items[q->head];
    q->head = (q->head + 1) % SIZE;  // 原子写
    return true;
}
```

**无锁原理**：
- 生产者只修改`tail`和相应元素
- 消费者只修改`head`
- 由于单生产者/消费者，不会有数据竞争，无需加锁
- 要求内存模型保证原子读写

#### 4.2.2 无锁栈

**基于CAS的无锁栈实现**：
```c
typedef struct node {
    int value;
    struct node *next;
} node_t;

typedef struct {
    node_t *top;
} stack_t;

void push(stack_t *stack, node_t *node) {
    node_t *old_top;
    do {
        old_top = stack->top;
        node->next = old_top;
    } while (!CompareAndSwap(&stack->top, old_top, node));
}

node_t* pop(stack_t *stack) {
    node_t *old_top;
    node_t *new_top;
    do {
        old_top = stack->top;
        if (old_top == NULL) return NULL;  // 栈空
        new_top = old_top->next;
    } while (!CompareAndSwap(&stack->top, old_top, new_top));
    
    return old_top;
}
```

**无锁工作原理**：
1. 读取当前栈顶
2. 准备更新（push时设置next指针，pop时找到新的top）
3. 使用CAS检查栈顶是否仍是原来的值
4. 如果CAS失败（栈顶被其他线程修改），则重试

### 4.3 无锁编程的挑战

#### 4.3.1 ABA问题

**问题描述**：

1. 线程T1读取共享变量A的值为a
2. T1被挂起
3. 线程T2将A的值从a修改为b，然后又修改回a
4. T1恢复执行，发现A的值仍为a，认为没有变化
5. T1执行CAS成功，但实际上对象状态已经改变

**示例场景**：
```
初始栈: [A] -> [B] -> [C]
1. 线程T1准备pop，读取top=A，next=B
2. T1挂起
3. 线程T2执行pop两次，删除A和B
4. T2分配新节点D并push: [D] -> [C]
5. T2重用节点A的内存，push: [A] -> [D] -> [C]
6. T1恢复，执行CAS(top, A, B)
7. CAS成功，栈变成: [B] -> ?
```

结果：节点D丢失，栈结构被破坏。

**解决方案**：
1. **版本计数器**：每次修改时递增计数器，CAS同时检查值和版本
```c
typedef struct {
    node_t *ptr;
    int count;
} tagged_ptr_t;
```

2. **内存管理**：避免重用最近释放的内存
3. **特殊原语**：如Load-Linked/Store-Conditional (LL/SC)

#### 4.3.2 内存管理问题

无锁算法中的内存管理十分复杂，主要问题：
- **悬空指针**：一个线程正在访问节点，同时另一个线程释放了它
- **内存泄漏**：难以确定何时安全释放内存

**解决方案**：
1. **危险指针(Hazard Pointers)**：线程标记它正在使用的指针，仅当指针不被标记时才释放
2. **引用计数**：跟踪每个对象的引用次数
3. **延迟回收**：将删除的节点放入回收队列，延迟释放
4. **纪元删除(Epoch-based Reclamation)**：按时间段管理内存释放

### 4.4 无锁编程适用场景

无锁算法不一定比基于锁的算法快，其主要优势在于：

**适用场景**：
- 需要高可靠性的系统：线程崩溃或被挂起不会影响其他线程
- 实时系统：避免优先级反转
- 高度竞争环境：避免锁争用开销

**不那么适用的场景**：
- 复杂数据结构：无锁实现难度显著增加
- 低竞争环境：锁的简单性和低开销可能更有优势
- 纯粹追求性能的场景：精心设计的锁可能更高效

## 5. 总结与实践指导

### 5.1 锁的选择与实现

- **低竞争**：简单的互斥锁通常足够
- **高竞争**：考虑使用排队锁(ticket lock)或MCS/CLH锁等扩展性好的锁
- **短临界区**：自旋锁可能更高效
- **长临界区**：应使用可休眠的互斥锁

### 5.2 锁定策略

- **粗粒度锁**：简单场景或低竞争环境
- **细粒度锁**：高并发、复杂数据结构
- **混合策略**：例如读写锁、意向锁
- **锁层次结构**：避免死锁

### 5.3 无锁编程指导

- 从简单的无锁模式开始（如单生产者-单消费者队列）
- 理解并处理ABA问题和内存管理挑战
- 考虑使用成熟的无锁库，而非自行实现
- 彻底测试各种并发场景
- 使用内存模型感知的工具进行验证

无论是使用细粒度锁还是无锁技术，都需要在简洁性、正确性和性能之间取得平衡。对于大多数应用，建议从简单的同步机制开始，只在需要时才引入更复杂的技术。 
# CS149: 并行计算 - 第17讲：事务内存基础

本讲介绍事务内存(Transactional Memory, TM)作为一种高级同步抽象，旨在简化并行程序设计，同时提供良好的性能和正确性保证。

## 1. 并发编程抽象的演化

并发编程抽象经历了从底层到高层的演化过程，以提高可用性和正确性：

### 1.1 硬件原子操作

**硬件提供的基本原子指令**:
- **Test-and-Set (TAS)**：原子读取并设置一个位
- **Fetch-and-Op**：原子读取并修改(加、减、与、或等)
- **Compare-and-Swap (CAS)**：原子比较并交换
- **Load-Linked/Store-Conditional (LL/SC)**：条件存储

这些原子操作是构建更高级同步结构的基础，但直接使用它们编程十分困难。

### 1.2 传统同步原语

**基于原子操作构建的同步机制**:
- **锁(Locks)**：互斥访问共享资源
- **屏障(Barriers)**：同步线程执行点
- **信号量(Semaphores)**：控制并发访问数量
- **条件变量(Condition Variables)**：线程间通信

这些机制虽然普遍使用，但在大型程序中容易引入错误(死锁、竞态条件等)。

### 1.3 无锁数据结构

**利用原子操作构建的特殊并发数据结构**:
- 无锁队列、栈、哈希表等
- 避免传统锁的一些问题，但实现复杂且难以组合使用

## 2. 锁的局限与事务内存的动机

### 2.1 锁的主要局限

传统锁机制存在多方面的局限性：

#### 2.1.1 并发度与性能权衡

锁的设计需要在以下方面做出权衡：
- **粗粒度锁(Coarse-grained locking)**：易于正确实现，但并发度低
- **细粒度锁(Fine-grained locking)**：并发度高，但实现复杂且容易出错
- **过多的锁开销**：即使在无竞争的情况下也需要获取和释放锁，增加了执行时间

```c
// 细粒度锁示例：HashMap
void insert(HashMap* map, int key, int value) {
    int bucket = hash(key) % NUM_BUCKETS;
    pthread_mutex_lock(&map->bucket_locks[bucket]);  // 只锁一个桶
    // 插入操作
    pthread_mutex_unlock(&map->bucket_locks[bucket]);
}
```

#### 2.1.2 失败原子性问题

当在锁定区域内发生异常或错误时，可能导致系统处于不一致状态：

```c
void transfer(Account* from, Account* to, int amount) {
    pthread_mutex_lock(&from->lock);
    pthread_mutex_lock(&to->lock);
    
    from->balance -= amount;  // 如果这里发生异常?
    // 系统崩溃，资金已扣除但未添加到目标账户
    to->balance += amount;
    
    pthread_mutex_unlock(&to->lock);
    pthread_mutex_unlock(&from->lock);
}
```

**问题**：锁不提供自动恢复机制，程序员需要编写复杂的异常处理和回滚逻辑。

#### 2.1.3 可组合性问题

当尝试组合使用多个锁保护的操作时，容易出现死锁或违反原子性：

```c
// 模块A
void withdraw(Account* acc, int amount) {
    pthread_mutex_lock(&acc->lock);
    acc->balance -= amount;
    pthread_mutex_unlock(&acc->lock);
}

// 模块B
void deposit(Account* acc, int amount) {
    pthread_mutex_lock(&acc->lock);
    acc->balance += amount;
    pthread_mutex_unlock(&acc->lock);
}

// 组合使用 - 可能导致不一致状态
void transfer(Account* from, Account* to, int amount) {
    withdraw(from, amount);  // 获取并释放from锁
    // 这里可能被中断，账户状态不一致
    deposit(to, amount);     // 获取并释放to锁
}
```

**问题**：难以不破坏封装的情况下安全组合使用现有同步代码。

### 2.2 事务内存的优势

事务内存旨在解决上述锁的局限性：

- **简化同步**：无需手动管理锁
- **失败原子性**：事务失败时自动回滚
- **提高并发度**：自动细粒度并发
- **可组合性**：事务可以安全嵌套和组合

## 3. 事务内存概念与抽象

### 3.1 事务内存基本概念

**事务内存**是一种编程抽象，允许程序员将一组读写操作标记为"事务"，系统保证这些操作原子地执行：

```c
// 使用事务内存抽象
void transfer(Account* from, Account* to, int amount) {
    atomic {
        from->balance -= amount;
        to->balance += amount;
    }
}
```

**核心特性**：
- **声明式(Declarative)**：程序员指定"做什么"(原子执行)，而非"如何做"(锁管理)
- **高级抽象**：隐藏底层实现机制，降低编程复杂性

### 3.2 事务内存的语义

事务内存从数据库事务借鉴了核心语义：

- **原子性(Atomicity)**：事务的所有内存写入要么全部生效(提交)，要么全部不生效(中止)
- **隔离性(Isolation)**：事务的中间状态对其他线程不可见，直到提交
- **可串行化(Serializability)**：并发事务的执行效果等同于它们按某种串行顺序执行

**与数据库事务的区别**：内存事务通常不需要持久性(Durability)特性，且操作粒度更小、频率更高。

### 3.3 `atomic` 块与锁的区别

虽然`atomic`块和锁都用于同步，但它们有本质区别：

| 特性 | `atomic` 块 | `lock`/`unlock` |
|------|------------|-----------------|
| 抽象级别 | 高级声明式 | 低级命令式 |
| 指定内容 | 做什么(原子性) | 怎么做(操作步骤) |
| 失败处理 | 自动回滚 | 需手动处理 |
| 组合能力 | 自然组合 | 容易死锁 |
| 嵌套性 | 支持嵌套 | 需特殊处理 |

**关系**：锁可以用来实现`atomic`块，但`atomic`的概念更广泛。

## 4. 事务内存的实现策略

事务内存系统实现需要考虑两个核心问题：数据版本管理和冲突检测。

### 4.1 数据版本管理策略

数据版本管理决定如何处理事务内部的写操作：

#### 4.1.1 积极版本管理(Eager Versioning)

**原理**：直接在内存中更新数据，同时在Undo Log中记录旧值
```
执行流程：
1. 写入前，将原值保存到私有Undo Log
2. 直接在内存原位置写入新值
3. 事务提交：简单丢弃Undo Log
4. 事务中止：使用Undo Log恢复所有修改
```

**特点**：
- **优势**：提交操作快速(已写入内存)
- **劣势**：中止操作昂贵(需回滚)，容错性差(系统崩溃可能留下部分更新)

#### 4.1.2 惰性版本管理(Lazy Versioning)

**原理**：写操作缓存在私有缓冲区，直到提交时才写入内存
```
执行流程：
1. 写入时，将新值保存到私有Write Buffer
2. 内存中保持原始值不变
3. 事务提交：将Write Buffer中所有修改写入内存
4. 事务中止：简单丢弃Write Buffer
```

**特点**：
- **优势**：中止操作快速(只丢弃缓冲区)，容错性好(崩溃不留部分更新)
- **劣势**：提交操作昂贵(需写回内存)

### 4.2 冲突检测策略

冲突检测决定何时以及如何发现并发事务之间的冲突：

#### 4.2.1 冲突类型

TM系统需要检测两种关键冲突类型：
- **读写冲突(RAW/WAR)**：一个事务读取的地址被另一个事务写入
- **写写冲突(WAW)**：两个事务写入同一个地址

为此，系统需要追踪每个事务的：
- **读集(Read Set)**：事务读取的所有内存位置
- **写集(Write Set)**：事务写入的所有内存位置

#### 4.2.2 悲观检测(Pessimistic Detection)

**原理**：在每次内存访问时立即检查冲突
```
执行流程：
1. 每次读/写操作时，立即检查与其他事务的冲突
2. 发现冲突时，根据竞争管理策略决定阻塞或中止
```

**特点**：
- **优势**：尽早发现冲突，减少浪费工作，可将中止转为等待
- **劣势**：检测开销大(每次访存都检查)，不保证前向进展

#### 4.2.3 乐观检测(Optimistic Detection)

**原理**：假设冲突罕见，仅在事务提交时检查冲突
```
执行流程：
1. 事务执行过程中不检查冲突，记录读写集
2. 事务尝试提交时，验证读集中的值未被其他事务修改
3. 如有冲突，中止本事务或冲突事务
```

**特点**：
- **优势**：减少检测开销，批量冲突处理，保证前向进展
- **劣势**：可能浪费大量工作（中止前已完成的计算）

### 4.3 事务内存实现空间

不同的TM系统采用不同的版本管理和冲突检测组合：

| 系统 | 版本管理 | 冲突检测 |
|------|---------|---------|
| TL2 | 惰性 | 乐观 |
| McRT-STM | 积极 | 悲观写+乐观读 |
| LogTM | 积极 | 悲观 |
| TCC | 惰性 | 乐观 |

## 5. 软件事务内存(STM)实现

### 5.1 STM工作原理

软件事务内存通过编译器和运行时库支持实现：

```
编译器转换：
atomic {          变成    tm_begin();
  x = y + 1;              int tmp = tm_read(&y);
  z = 2;                  tm_write(&x, tmp + 1);
}                         tm_write(&z, 2);
                          tm_commit();
```

**关键要素**：
1. **编译器插桩(Instrumentation)**：转换内存访问为TM运行时调用
2. **运行时库**：实现事务管理、版本控制和冲突检测

### 5.2 STM数据结构

STM系统通常需要两个关键数据结构：

#### 5.2.1 事务描述符(Transaction Descriptor)

每个线程维护一个事务描述符，包含：
- **读集/写集**：事务访问的内存位置
- **Undo Log/Write Buffer**：根据版本管理策略保存版本数据
- **事务状态**：活跃、提交、中止等
- **元数据**：时间戳、锁引用等

#### 5.2.2 事务记录(Transaction Record)

与共享数据关联，用于冲突检测：
- **所有者信息**：当前写入该数据的事务ID
- **版本号**：数据的版本标识
- **锁状态**：该数据是否被锁定及持有者

### 5.3 冲突检测粒度

STM可以在不同级别检测冲突：

- **对象级(Object-based)**：每个对象一个事务记录
  - 优点：开销低，适合面向对象语言
  - 缺点：可能导致伪冲突(两个事务修改对象的不同字段)
   
- **字/字段级(Word/Field-based)**：每个内存字或对象字段一个事务记录
  - 优点：更细粒度并发，减少伪冲突
  - 缺点：空间开销大，检测开销高
   
- **缓存行级(Cache-line based)**：每个缓存行一个事务记录
  - 优点：与硬件缓存一致性单元匹配
  - 缺点：可能导致伪冲突(缓存行中包含无关数据)

### 5.4 示例STM算法：McRT风格

以下是一种混合策略STM算法的简化描述：

```
// 策略：Eager Versioning + Optimistic Reads + Pessimistic Writes

// 读操作
Value tm_read(Address addr) {
    // 先检查自己的写集，如已写入则返回本地值
    if (in_write_set(addr))
        return write_set[addr].value;
    
    // 读取内存值
    Value val = *addr;
    
    // 验证版本未过期
    if (tx_record[addr].version > start_time)
        abort(); // 此地址在事务开始后已被修改
    
    // 记录到读集
    read_set.add(addr, tx_record[addr].version);
    
    return val;
}

// 写操作
void tm_write(Address addr, Value val) {
    // 验证数据一致性
    if (!tx_record[addr].is_locked_by_me() && 
        tx_record[addr].version > start_time)
        abort();
    
    // 获取写锁
    if (!tx_record[addr].try_lock(my_tx_id))
        abort();
    
    // 记录旧值到undo log
    if (!in_write_set(addr))
        undo_log.add(addr, *addr);
    
    // 原地写入新值
    *addr = val;
    
    // 记录到写集
    write_set.add(addr, val);
}

// 提交操作
void tm_commit() {
    // 验证读集(读的值未被修改)
    for (each_entry in read_set) {
        if (tx_record[entry.addr].version != entry.version)
            abort();
    }
    
    // 原子增加全局时间戳
    new_version = atomic_inc(global_timestamp);
    
    // 更新写集中的版本号并释放锁
    for (each_entry in write_set) {
        tx_record[entry.addr].version = new_version;
        tx_record[entry.addr].unlock();
    }
}

// 中止操作
void tm_abort() {
    // 使用undo log回滚写操作
    for (each_entry in undo_log) {
        *entry.addr = entry.old_value;
    }
    
    // 释放写锁
    for (each_addr in write_set) {
        tx_record[addr].unlock();
    }
    
    // 清理事务状态
    read_set.clear();
    write_set.clear();
    undo_log.clear();
}
```

## 6. 事务内存的优势总结

事务内存作为一种高级同步抽象，相比传统锁机制有以下优势：

### 6.1 易用性

- **声明式同步**：程序员只需声明"要做什么"(原子性)
- **减少错误**：避免手动锁管理中的常见错误(忘记解锁等)
- **简化推理**：更容易推理并发程序的行为

### 6.2 性能潜力

- **细粒度并发**：系统自动管理细粒度同步
- **读-读并发**：不同事务可并发读取同一数据
- **乐观执行**：假设冲突少，优化常见情况
- **自动进度保证**：避免死锁和饥饿问题

### 6.3 失败原子性

- **自动回滚**：事务失败时，系统自动撤销部分更改
- **故障恢复**：线程崩溃不会导致系统处于不一致状态
- **异常安全**：抛出异常时自动回滚，无需复杂的清理代码

### 6.4 可组合性

- **模块化**：事务块可以安全组合，无需了解实现细节
- **嵌套事务**：内部事务自然融入外部事务
- **简化库设计**：库开发者可以设计原子操作，客户端代码可以任意组合使用

## 7. 总结

事务内存为并行程序提供了一种高级同步抽象，通过声明式编程模型简化了并发程序的开发。

**关键概念**：
- 事务内存允许程序员声明性地指定原子执行区域(`atomic`块)
- 系统保证事务具有原子性、隔离性和可串行化性质
- 实现策略主要涉及数据版本管理(积极vs惰性)和冲突检测(悲观vs乐观)
- 软件事务内存通过编译器插桩和运行时库实现，硬件事务内存将在下一讲讨论

事务内存在简化编程模型的同时,通过自动管理并发细节和失败恢复，提供了更高效、更可靠的并行编程范式。 
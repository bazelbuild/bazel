// Copyright 2025 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.concurrent;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.concurrent.AbstractExecutorService;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/**
 * An {@link java.util.concurrent.ExecutorService} that creates a pool of threads to execute
 * submitted tasks in a work-stealing manner.
 *
 * <p>Similar to {@link java.util.concurrent.ForkJoinPool}, {@link WorkStealingThreadPoolExecutor}
 * submits tasks to threads' local task queue in order to reduce contention and implements
 * work-stealing to improve the throughput.
 *
 * <p>One difference to {@link java.util.concurrent.ForkJoinPool} is, {@link
 * WorkStealingThreadPoolExecutor} accepts a {@link ThreadFactory} which allows users to use {@code
 * VirtualThread} as worker threads.
 */
public final class WorkStealingThreadPoolExecutor extends AbstractExecutorService {
  private final ThreadFactory threadFactory;
  private final ConcurrentHashMap<Thread, Worker> workerMap;

  private final ImmutableList<Worker> workers;
  private final CountDownLatch deregisteredWorkersCountDown;
  private volatile boolean isShutdown;
  private final AtomicInteger remainingTasks = new AtomicInteger(0);
  private final ReentrantLock remainingTasksAvailableLock = new ReentrantLock();

  /** A condition object for the condition {@code remainingTasks.get() > 0}. */
  @GuardedBy("remainingTasksAvailableLock")
  private final Condition remainingTasksAvailableCondition =
      remainingTasksAvailableLock.newCondition();

  /** A task queue that is local to a worker thread. All methods must be thread-safe. */
  private static final class TaskQueue {
    private final ConcurrentLinkedDeque<Runnable> queue = new ConcurrentLinkedDeque<>();

    /** Add a task to this queue */
    public void add(Runnable runnable) {
      queue.addLast(runnable);
    }

    /**
     * Retrieves and removes a task from this deque for the owning worker, or returns {@code null}
     * if this queue is empty.
     */
    @Nullable
    public Runnable poll() {
      return queue.pollLast();
    }

    /**
     * Retrieves and removes a task from this deque for the non-owning worker, or returns {@code
     * null} if this queue is empty.
     */
    @Nullable
    public Runnable steal() {
      return queue.pollFirst();
    }
  }

  public WorkStealingThreadPoolExecutor(int parallelism, ThreadFactory threadFactory) {
    checkState(parallelism > 0);

    this.threadFactory = threadFactory;
    this.workerMap = new ConcurrentHashMap<>(parallelism);
    this.deregisteredWorkersCountDown = new CountDownLatch(parallelism);

    ArrayList<Thread> threads = new ArrayList<>(parallelism);
    ImmutableList.Builder<Worker> workers = ImmutableList.builderWithExpectedSize(parallelism);
    for (int i = 0; i < parallelism; i++) {
      var worker = new Worker(this);
      workers.add(worker);
      threads.add(threadFactory.newThread(worker));
    }
    this.workers = workers.build();

    for (var thread : threads) {
      thread.start();
    }
  }

  private static class Worker implements Runnable {
    private final WorkStealingThreadPoolExecutor pool;

    private final TaskQueue queue = new TaskQueue();

    private Worker(WorkStealingThreadPoolExecutor pool) {
      this.pool = pool;
    }

    @Override
    public void run() {
      Thread thread = Thread.currentThread();
      pool.registerWorker(thread, this);
      try {
        Runnable task;
        while ((task = pool.pollOrStealOrWaitTask(this)) != null) {
          try {
            task.run();
          } catch (Throwable e) {
            // If the task throws an exception, let the thread exit so it gets reported to the
            // uncaught exception handler, but create a fresh thread to fill its place.
            pool.assignWorkerToNewThread(thread, this);
            throw e;
          }
        }
      } finally {
        pool.deregisterWorker(thread);
      }
    }

    /** Add a task to the worker's local queue. */
    private void addTask(Runnable task) {
      queue.add(task);
      pool.remainingTasks.incrementAndGet();
    }

    /**
     * Retrieves and removes a task from the worker's own queue, or returns {@code null} if the
     * queue is empty.
     */
    @Nullable
    private Runnable poll() {
      Runnable task = queue.poll();
      if (task != null) {
        pool.remainingTasks.decrementAndGet();
      }
      return task;
    }

    /**
     * Retrieves and removes a task from the worker's own queue, or returns {@code null} if the
     * queue is empty.
     *
     * <p>Thread that doesn't own the worker should use {@link #steal} to retrive task. It retrives
     * task from another end of the queue, so that, when the system is under pressure, the tasks
     * won't be retained for a long time. Failed to do so will likely cause more major GCs.
     */
    @Nullable
    private Runnable steal() {
      Runnable task = queue.steal();
      if (task != null) {
        pool.remainingTasks.decrementAndGet();
      }
      return task;
    }

    /**
     * Retrieves and removes all tasks from the worker's own queue and adds them to {@code
     * remainingTasks}.
     */
    private void pollRemainingTasks(ImmutableList.Builder<Runnable> remainingTasks) {
      Runnable task;
      while ((task = queue.poll()) != null) {
        pool.remainingTasks.decrementAndGet();
        remainingTasks.add(task);
      }
    }
  }

  private void registerWorker(Thread thread, Worker worker) {
    workerMap.put(thread, worker);
  }

  private void assignWorkerToNewThread(Thread thread, Worker worker) {
    var prevWorker = workerMap.remove(thread);
    checkState(prevWorker == worker);
    threadFactory.newThread(worker).start();
  }

  private void deregisterWorker(Thread thread) {
    // If the worker is assigned to a different thread, do nothing.
    if (workerMap.remove(thread) != null) {
      deregisteredWorkersCountDown.countDown();
    }
  }

  /**
   * Retrieves and removes a task from a random worker in the pool, or returns {@code null} if no
   * task is available.
   */
  @Nullable
  private Runnable scan() {
    int numWorkers = workers.size();
    // Scan workers starting at a random location and with a fixed step. Use an arbitrary prime
    // number for the step so that it shares no common divisor with `numWorkers`, thus ensuring
    // that each of the `numWorkers` iterations visits a distinct worker.
    int i = ThreadLocalRandom.current().nextInt(numWorkers);
    int step = 31;
    for (int n = numWorkers; n > 0; n--) {
      Runnable task = workers.get(i).steal();
      if (task != null) {
        return task;
      }
      i = (i + step) % numWorkers;
    }
    return null;
  }

  /**
   * Retrieves and removes a task from {@code worker}'s queue. If no task is available, steals tasks
   * from other workers in the pool. If still no task is available, blocking await until the thread
   * is signaled.
   *
   * <p>Returns {@code null} if no more tasks in the pool and the executor {@link #isShutdown()}.
   */
  @Nullable
  private Runnable pollOrStealOrWaitTask(Worker worker) {
    while (true) {
      Runnable task = worker.poll();
      if (task == null) {
        task = scan();
      }
      if (task != null) {
        return task;
      }

      remainingTasksAvailableLock.lock();
      try {
        // Only wait if there is no pending tasks and more tasks are allowed to be submitted.
        // Otherwise, the thread may never be signaled.
        while (remainingTasks.get() == 0) {
          if (isShutdown) {
            return null;
          }

          try {
            remainingTasksAvailableCondition.await();
          } catch (InterruptedException e) {
            // Intentionally ignored
          }
        }
      } finally {
        remainingTasksAvailableLock.unlock();
      }
    }
  }

  /** Wake up one worker that is awaiting new task. */
  private void signalOneWorker() {
    remainingTasksAvailableLock.lock();
    try {
      remainingTasksAvailableCondition.signal();
    } finally {
      remainingTasksAvailableLock.unlock();
    }
  }

  /** Wake up all workers that are awaiting new task. */
  private void signalAllWorkers() {
    remainingTasksAvailableLock.lock();
    try {
      remainingTasksAvailableCondition.signalAll();
    } finally {
      remainingTasksAvailableLock.unlock();
    }
  }

  @Override
  public void shutdown() {
    isShutdown = true;
    signalAllWorkers();
  }

  @Override
  public ImmutableList<Runnable> shutdownNow() {
    isShutdown = true;

    ImmutableList.Builder<Runnable> remainingTasks = ImmutableList.builder();
    for (var worker : workers) {
      worker.pollRemainingTasks(remainingTasks);
    }
    for (var thread : workerMap.keySet()) {
      thread.interrupt();
    }
    return remainingTasks.build();
  }

  @Override
  public boolean isShutdown() {
    return isShutdown;
  }

  @Override
  public boolean isTerminated() {
    return deregisteredWorkersCountDown.getCount() == 0;
  }

  @Override
  public boolean awaitTermination(long timeout, TimeUnit unit) throws InterruptedException {
    return deregisteredWorkersCountDown.await(timeout, unit);
  }

  private Worker getRandomWorker() {
    return workers.get(ThreadLocalRandom.current().nextInt(workers.size()));
  }

  @Override
  public void execute(Runnable command) {
    if (isShutdown()) {
      throw new RejectedExecutionException();
    }

    var worker = workerMap.get(Thread.currentThread());
    if (worker == null) {
      worker = getRandomWorker();
    }
    worker.addTask(command);
    // Wake up an idle worker to give it a chance to steal the task.
    signalOneWorker();
  }
}

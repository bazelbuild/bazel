// Copyright 2015 The Bazel Authors. All rights reserved.
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

import java.util.AbstractQueue;
import java.util.Collection;
import java.util.Iterator;
import java.util.concurrent.BlockingDeque;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;

/** A {@link BlockingQueue} with LIFO (last-in-first-out) ordering. */
public class BlockingStack<E> extends AbstractQueue<E> implements BlockingQueue<E> {
  // We just restrict to only using the *First methods on the deque, turning it into a stack.
  private final BlockingDeque<E> deque;

  public BlockingStack() {
    this.deque = new LinkedBlockingDeque<>();
  }

  @Override
  public Iterator<E> iterator() {
    return deque.iterator();
  }

  @Override
  public int size() {
    return deque.size();
  }

  @Override
  public void put(E e) throws InterruptedException {
    deque.putFirst(e);
  }

  @Override
  public boolean offer(E e, long timeout, TimeUnit unit) throws InterruptedException {
    return deque.offerFirst(e, timeout, unit);
  }

  @Override
  public boolean offer(E e) {
    return deque.offerFirst(e);
  }

  @Override
  public E take() throws InterruptedException {
    return deque.takeFirst();
  }

  @Override
  public E poll(long timeout, TimeUnit unit) throws InterruptedException {
    return deque.pollFirst(timeout, unit);
  }

  @Override
  public E poll() {
    return deque.pollFirst();
  }

  @Override
  public int remainingCapacity() {
    return deque.remainingCapacity();
  }

  @Override
  public int drainTo(Collection<? super E> c) {
    return deque.drainTo(c);
  }

  @Override
  public int drainTo(Collection<? super E> c, int maxElements) {
    return deque.drainTo(c, maxElements);
  }

  @Override
  public E peek() {
    return deque.peekFirst();
  }
}

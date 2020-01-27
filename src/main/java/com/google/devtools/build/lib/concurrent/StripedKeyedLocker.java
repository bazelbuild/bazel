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

import com.google.common.util.concurrent.Striped;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;

/**
 * An implementation of {@link KeyedLocker} backed by a {@link Striped}.
 */
public class StripedKeyedLocker<K> implements KeyedLocker<K> {

  private final Striped<ReadWriteLock> locks;

  public StripedKeyedLocker(int stripes) {
    locks = Striped.readWriteLock(stripes);
  }

  private static AutoUnlocker lockAndMakeAutoUnlocker(
      final Lock lock, final Object keyForDebugging) {
    lock.lock();
    return new AutoUnlocker() {
      private final AtomicBoolean closeCalled = new AtomicBoolean(false);

      @Override
      public void close() {
        if (closeCalled.getAndSet(true)) {
          String msg =
              String.format(
                  "For key %s, 'close' can be called at most once per AutoUnlocker instance",
                  keyForDebugging);
          throw new IllegalUnlockException(msg);
        }
        lock.unlock();
      }
    };
  }

  @Override
  public AutoUnlocker writeLock(K key) {
    return lockAndMakeAutoUnlocker(locks.get(key).writeLock(), key);
  }

  @Override
  public AutoUnlocker readLock(K key) {
    return lockAndMakeAutoUnlocker(locks.get(key).readLock(), key);
  }
}

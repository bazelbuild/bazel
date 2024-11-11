// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.util.concurrent.AbstractFuture;
import com.google.common.util.concurrent.ListenableFuture;
import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

/** A tailored settable future with ownership tracking. */
abstract class SettableFutureWithOwnership<T> extends AbstractFuture<T> {
  /** Used to establish exactly-once ownership of this future with {@link #tryTakeOwnership}. */
  @SuppressWarnings({"UnusedVariable", "FieldCanBeFinal"}) // set with OWNED_HANDLE
  private boolean owned = false;

  private boolean isSet = false;

  /**
   * Returns true once.
   *
   * <p>When using {@link com.github.benmanes.caffeine.cache.Cache#get} with future values and a
   * mapping function, there's a need to determine which thread owns the future. This method
   * provides such a mechanism.
   *
   * <p>When this returns true, the caller must call either {@link #completeWith} or {@link
   * #failWith}.
   */
  boolean tryTakeOwnership() {
    return OWNED_HANDLE.compareAndSet(this, false, true);
  }

  void completeWith(ListenableFuture<T> future) {
    checkState(setFuture(future), "already set %s", this);
    isSet = true;
  }

  void failWith(IOException e) {
    checkState(setException(e));
    isSet = true;
  }

  /**
   * Verifies that the future was complete.
   *
   * <p>With settable futures, there's a risk of deadlock-like behavior if the future is not
   * complete. The owning thread should call this in a finally clause to fail-fast instead.
   */
  void verifyComplete() {
    if (!isSet) {
      checkState(
          setException(
              new IllegalStateException(
                  "future was unexpectedly unset, look for unchecked exceptions")));
    }
  }

  private static final VarHandle OWNED_HANDLE;

  static {
    try {
      OWNED_HANDLE =
          MethodHandles.lookup()
              .findVarHandle(SettableFutureWithOwnership.class, "owned", boolean.class);
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }
}

// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.starlark;

import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.events.Reportable;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/**
 * Captures state that persists across different invocations of {@link
 * com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction}, specifically {@link
 * StarlarkRepositoryFunction}.
 */
class RepoFetchingSkyKeyComputeState implements SkyKeyComputeState {

  /**
   * Common interface for a message that can be sent from the worker thread back to the host
   * Skyframe thread.
   */
  sealed interface Message
      permits Message.NewSkyframeDependency, Message.Done, Message.Error, Message.Event {

    /** Indicates that a new SkyFrame dependency is requested. */
    record NewSkyframeDependency<
            E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>(
        SkyKey key,
        @Nullable Class<E1> e1,
        @Nullable Class<E2> e2,
        @Nullable Class<E3> e3,
        @Nullable Class<E4> e4,
        SettableFuture<SkyValue> valueFuture)
        implements Message {}

    /** Indicates that the fetching is done and has yielded the given result. */
    record Done(RepositoryDirectoryValue.Builder result) implements Message {}

    /** Indicates that an exception occurred during the fetch. */
    record Error(RepositoryFunctionException ex) implements Message {}

    /** Indicates that a reportable event occurred during the fetch. */
    record Event(Reportable reportable) implements Message {}
  }

  private final BlockingQueue<Message> messages = new ArrayBlockingQueue<>(8);
  private Thread worker = null;
  private final AtomicReference<SettableFuture<Void>> messageFuture = new AtomicReference<>();

  boolean isWorkerRunning() {
    return worker != null;
  }

  void startWorker(Runnable r) {
    worker = Thread.startVirtualThread(r);
  }

  void postMessage(Message message) throws InterruptedException {
    messages.put(message);
    SettableFuture<Void> future = messageFuture.getAndSet(null);
    if (future != null) {
      future.set(null);
      if (future.isCancelled()) {
        throw new InterruptedException();
      }
    }
  }

  @Nullable
  Message receiveMessage() throws InterruptedException {
    return messages.poll(10, TimeUnit.MILLISECONDS);
  }

  void registerMessageListener(Environment env) {
    SettableFuture<Void> future = SettableFuture.create();
    env.dependOnFuture(future);
    Preconditions.checkState(messageFuture.getAndSet(future) == null);
  }

  @Override
  public void close() {
    if (worker != null) {
      worker.interrupt();
      worker = null;
    }
    {
      SettableFuture<Void> future = messageFuture.getAndSet(null);
      if (future != null) {
        future.cancel(true);
      }
    }
    for (Message message : messages) {
      if (message instanceof Message.NewSkyframeDependency<?, ?, ?, ?> newDep) {
        newDep.valueFuture.cancel(true);
      }
    }
    messages.clear();
  }
}

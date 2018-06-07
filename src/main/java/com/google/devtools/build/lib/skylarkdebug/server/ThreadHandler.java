// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkdebug.server;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos;
import com.google.devtools.build.lib.syntax.Debuggable;
import com.google.devtools.build.lib.syntax.Debuggable.ReadyToPause;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/** Handles all thread-related state and debugging tasks. */
final class ThreadHandler {

  private static class ThreadState {
    final long id;
    final String name;
    final Debuggable debuggable;
    /** Non-null if the thread is currently paused. */
    @Nullable volatile PausedThreadState pausedState;
    /** Determines when execution should next be paused. Non-null if currently stepping. */
    @Nullable volatile ReadyToPause readyToPause;

    ThreadState(
        long id,
        String name,
        Debuggable debuggable,
        @Nullable PausedThreadState pausedState,
        @Nullable ReadyToPause readyToPause) {
      this.id = id;
      this.name = name;
      this.debuggable = debuggable;
      this.pausedState = pausedState;
      this.readyToPause = readyToPause;
    }
  }

  /** Information about a paused thread. */
  private static class PausedThreadState {

    /** The {@link Location} where execution is currently paused. */
    final Location location;

    /** Used to block execution of threads */
    final Semaphore semaphore;

    PausedThreadState(Location location) {
      this.location = location;
      this.semaphore = new Semaphore(0);
    }
  }

  /**
   * If true, all threads will pause at the earliest possible opportunity. New threads will also be
   * immediately paused upon creation.
   *
   * <p>The debugger starts with all threads paused, until a StartDebuggingRequest is received.
   */
  private volatile boolean pausingAllThreads = true;

  /** A map from thread identifiers to their state info. */
  @GuardedBy("itself")
  private final Map<Long, ThreadState> threads = new HashMap<>();

  /** All location-based breakpoints (the only type of breakpoint currently supported). */
  private volatile ImmutableSet<SkylarkDebuggingProtos.Location> breakpoints = ImmutableSet.of();

  /**
   * Threads which are set to be paused in the next checked execution step.
   *
   * <p>Invariant: Every thread id in this set is also in {@link #threads}, provided that we are not
   * in a synchronized block on that map.
   */
  private final Set<Long> threadsToPause = ConcurrentHashMap.newKeySet();

  /** Registers a Skylark thread with the {@link ThreadHandler}. */
  void registerThread(long threadId, String threadName, Debuggable debuggable) {
    doRegisterThread(threadId, threadName, debuggable);
  }

  private ThreadState doRegisterThread(long threadId, String threadName, Debuggable debuggable) {
    ThreadState thread = new ThreadState(threadId, threadName, debuggable, null, null);
    synchronized (threads) {
      threads.put(threadId, thread);
    }
    return thread;
  }

  /** Mark all current and future threads paused. Will take effect asynchronously. */
  void pauseAllThreads() {
    synchronized (threads) {
      threadsToPause.addAll(threads.keySet());
    }
    pausingAllThreads = true;
  }

  /** Mark the given thread paused. Will take effect asynchronously. */
  void pauseThread(long threadId) throws DebugRequestException {
    synchronized (threads) {
      if (!threads.containsKey(threadId)) {
        throw new DebugRequestException("Unknown thread: " + threadId);
      }
      threadsToPause.add(threadId);
    }
  }

  /** Called when Skylark execution for this thread is complete. */
  void unregisterThread(long threadId) {
    synchronized (threads) {
      threads.remove(threadId);
      threadsToPause.remove(threadId);
    }
  }

  void setBreakpoints(ImmutableSet<SkylarkDebuggingProtos.Location> breakpoints) {
    // all breakpoints cover the entire line, so unset the column number.
    this.breakpoints =
        breakpoints
            .stream()
            .map(location -> location.toBuilder().clearColumnNumber().build())
            .collect(toImmutableSet());
  }

  /** Resumes all threads. */
  void resumeAllThreads() {
    threadsToPause.clear();
    pausingAllThreads = false;
    synchronized (threads) {
      for (ThreadState thread : threads.values()) {
        // continue-all doesn't support stepping.
        resumeThread(thread, SkylarkDebuggingProtos.Stepping.NONE);
      }
    }
  }

  /**
   * Unpauses the given thread if it is currently paused. Also unsets {@link #pausingAllThreads}.
   */
  void resumeThread(long threadId, SkylarkDebuggingProtos.Stepping stepping)
      throws DebugRequestException {
    synchronized (threads) {
      ThreadState thread = threads.get(threadId);
      if (thread == null) {
        throw new DebugRequestException(String.format("Thread %s is not running.", threadId));
      }
      if (thread.pausedState == null) {
        throw new DebugRequestException(String.format("Thread %s is not paused.", threadId));
      }
      resumeThread(thread, stepping);
    }
  }

  /**
   * Unpauses the given thread if it is currently paused. Also unsets {@link #pausingAllThreads}.
   */
  @GuardedBy("threads")
  private void resumeThread(ThreadState thread, SkylarkDebuggingProtos.Stepping stepping) {
    PausedThreadState pausedState = thread.pausedState;
    if (pausedState == null) {
      return;
    }
    // once the user has resumed any thread, don't continue pausing future threads
    pausingAllThreads = false;
    thread.readyToPause =
        thread.debuggable.stepControl(DebugEventHelper.convertSteppingEnum(stepping));
    pausedState.semaphore.release();
    thread.pausedState = null;
  }

  void pauseIfNecessary(Environment env, Location location, DebugServerTransport transport) {
    if (shouldPauseCurrentThread(env, location)) {
      pauseCurrentThread(env, location, transport);
    }
  }

  ImmutableList<SkylarkDebuggingProtos.Thread> listThreads() {
    ImmutableList.Builder<SkylarkDebuggingProtos.Thread> list = ImmutableList.builder();
    synchronized (threads) {
      for (ThreadState thread : threads.values()) {
        list.add(getThreadProto(thread));
      }
    }
    return list.build();
  }

  /** Handles a {@code ListFramesRequest} and returns its response. */
  ImmutableList<SkylarkDebuggingProtos.Frame> listFrames(long threadId)
      throws DebugRequestException {
    Debuggable debuggable;
    PausedThreadState pausedState;
    synchronized (threads) {
      ThreadState thread = threads.get(threadId);
      if (thread == null) {
        throw new DebugRequestException(String.format("Thread %s is not running.", threadId));
      }
      pausedState = thread.pausedState;
      if (pausedState == null) {
        throw new DebugRequestException(String.format("Thread %s is not paused.", threadId));
      }
      debuggable = thread.debuggable;
    }
    // no need to list frames within the synchronize block: threads can only be resumed in response
    // to a client request, and requests are handled serially
    return debuggable
        .listFrames(pausedState.location)
        .stream()
        .map(DebugEventHelper::getFrameProto)
        .collect(toImmutableList());
  }

  SkylarkDebuggingProtos.Value evaluate(long threadId, String expression)
      throws DebugRequestException {
    Debuggable debuggable;
    synchronized (threads) {
      ThreadState thread = threads.get(threadId);
      if (thread == null) {
        throw new DebugRequestException(String.format("Thread %s is not running.", threadId));
      }
      if (thread.pausedState == null) {
        throw new DebugRequestException(String.format("Thread %s is not paused.", threadId));
      }
      debuggable = thread.debuggable;
    }
    // no need to evaluate within the synchronize block: threads can only be resumed in response
    // to a client request, and requests are handled serially
    try {
      Object result = debuggable.evaluate(expression);
      return DebuggerSerialization.getValueProto("Evaluation result", result);
    } catch (EvalException | InterruptedException e) {
      throw new DebugRequestException(e.getMessage());
    }
  }

  /**
   * Pauses the current thread's execution, blocking until it's resumed via a
   * ContinueExecutionRequest.
   */
  private void pauseCurrentThread(
      Environment env, Location location, DebugServerTransport transport) {
    long threadId = Thread.currentThread().getId();

    SkylarkDebuggingProtos.Thread threadProto;
    PausedThreadState pausedState;
    synchronized (threads) {
      ThreadState thread = threads.get(threadId);
      if (thread == null) {
        // this skylark entry point didn't call DebugServer#runWithDebugging. Now that we've hit a
        // breakpoint, register it anyway.
        // TODO(bazel-team): once all skylark evaluation routes through
        // DebugServer#runWithDebugging, report an error here instead
        String fallbackThreadName = "Untracked thread: " + threadId;
        transport.postEvent(DebugEventHelper.threadStartedEvent(threadId, fallbackThreadName));
        thread = doRegisterThread(threadId, fallbackThreadName, env);
      }
      threadProto = getThreadProto(thread);
      pausedState = new PausedThreadState(location);
      thread.pausedState = pausedState;
    }

    transport.postEvent(DebugEventHelper.threadPausedEvent(threadProto));
    pausedState.semaphore.acquireUninterruptibly();
    transport.postEvent(DebugEventHelper.threadContinuedEvent(threadProto));
  }

  private boolean shouldPauseCurrentThread(Environment env, Location location) {
    long threadId = Thread.currentThread().getId();
    if (threadsToPause.remove(threadId) || pausingAllThreads) {
      return true;
    }
    if (hasBreakpointAtLocation(location)) {
      return true;
    }

    // TODO(bazel-team): if contention becomes a problem, consider changing 'threads' to a
    // concurrent map, and synchronizing on individual entries
    synchronized (threads) {
      ThreadState thread = threads.get(threadId);
      if (thread != null && thread.readyToPause != null && thread.readyToPause.test(env)) {
        return true;
      }
    }
    return false;
  }

  private boolean hasBreakpointAtLocation(Location location) {
    // breakpoints is volatile, so taking a local copy
    ImmutableSet<SkylarkDebuggingProtos.Location> breakpoints = this.breakpoints;
    if (breakpoints.isEmpty()) {
      return false;
    }
    SkylarkDebuggingProtos.Location locationProto = DebugEventHelper.getLocationProto(location);
    // column data ignored for breakpoints
    return locationProto != null
        && breakpoints.contains(locationProto.toBuilder().clearColumnNumber().build());
  }

  /** Returns a {@code Thread} proto builder with information about the given thread. */
  private static SkylarkDebuggingProtos.Thread getThreadProto(ThreadState thread) {
    SkylarkDebuggingProtos.Thread.Builder builder =
        SkylarkDebuggingProtos.Thread.newBuilder().setId(thread.id).setName(thread.name);

    PausedThreadState pausedState = thread.pausedState;
    if (pausedState != null) {
      builder
          .setIsPaused(true)
          .setLocation(DebugEventHelper.getLocationProto(pausedState.location));
    }
    return builder.build();
  }
}

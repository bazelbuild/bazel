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
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.PauseReason;
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

  /** The state of a thread that is paused. */
  private static class PausedThreadState {
    final long id;
    final String name;
    final Debuggable debuggable;
    /** The {@link Location} where execution is currently paused. */
    final Location location;
    /** Used to block execution of threads */
    final Semaphore semaphore;

    PausedThreadState(long id, String name, Debuggable debuggable, Location location) {
      this.id = id;
      this.name = name;
      this.debuggable = debuggable;
      this.location = location;
      this.semaphore = new Semaphore(0);
    }
  }

  /**
   * The state of a thread that is stepping, i.e. currently running but expected to stop at a
   * subsequent statement even without a breakpoint. This may include threads that have completed
   * running while stepping, since the ThreadHandler doesn't know when a thread terminates.
   */
  private static class SteppingThreadState {
    /** Determines when execution should next be paused. */
    final ReadyToPause readyToPause;

    SteppingThreadState(ReadyToPause readyToPause) {
      this.readyToPause = readyToPause;
    }
  }

  /**
   * If true, all threads will pause at the earliest possible opportunity. New threads will also be
   * immediately paused upon creation.
   *
   * <p>The debugger starts with all threads paused, until a StartDebuggingRequest is received.
   */
  private volatile boolean pausingAllThreads = true;

  /** A map from identifiers of paused threads to their state info. */
  @GuardedBy("this")
  private final Map<Long, PausedThreadState> pausedThreads = new HashMap<>();

  /** A map from identifiers of stepping threads to their state. */
  @GuardedBy("this")
  private final Map<Long, SteppingThreadState> steppingThreads = new HashMap<>();

  /** All location-based breakpoints (the only type of breakpoint currently supported). */
  private volatile ImmutableSet<SkylarkDebuggingProtos.Location> breakpoints = ImmutableSet.of();

  /**
   * True if the thread is currently performing a debugger-requested evaluation. If so, we don't
   * check for breakpoints during the evaluation.
   */
  private final ThreadLocal<Boolean> servicingEvalRequest = ThreadLocal.withInitial(() -> false);

  /**
   * Threads which are not paused now, but that are set to be paused in the next checked execution
   * step as the result of a PauseThreadRequest.
   *
   * <p>Invariant: Every thread id in this set is also in {@link #steppingThreads}, provided that we
   * are not in a synchronized block on the class instance.
   */
  private final Set<Long> threadsToPause = ConcurrentHashMap.newKeySet();

  /** Mark all current and future threads paused. Will take effect asynchronously. */
  void pauseAllThreads() {
    pausingAllThreads = true;
  }

  /** Mark the given thread paused. Will take effect asynchronously. */
  void pauseThread(long threadId) throws DebugRequestException {
    synchronized (this) {
      if (!steppingThreads.containsKey(threadId)) {
        String error =
            pausedThreads.containsKey(threadId)
                ? "Thread is already paused"
                : "Unknown thread: only threads which are currently stepping can be paused";
        throw new DebugRequestException(error);
      }
      threadsToPause.add(threadId);
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

  /**
   * Resumes all threads. Any currently stepping threads have their stepping behavior cleared, so
   * will run unconditionally.
   */
  void resumeAllThreads() {
    threadsToPause.clear();
    pausingAllThreads = false;
    synchronized (this) {
      for (PausedThreadState thread : pausedThreads.values()) {
        // continue-all doesn't support stepping.
        resumePausedThread(thread, SkylarkDebuggingProtos.Stepping.NONE);
      }
      steppingThreads.clear();
    }
  }

  /**
   * Unpauses the given thread if it is currently paused. Also unsets {@link #pausingAllThreads}. If
   * the thread is not paused, but currently stepping, it clears the stepping behavior so it will
   * run unconditionally.
   */
  void resumeThread(long threadId, SkylarkDebuggingProtos.Stepping stepping)
      throws DebugRequestException {
    // once the user has requested any thread be resumed, don't continue pausing future threads
    pausingAllThreads = false;
    synchronized (this) {
      threadsToPause.remove(threadId);
      if (steppingThreads.remove(threadId) != null) {
        return;
      }
      PausedThreadState thread = pausedThreads.get(threadId);
      if (thread == null) {
        throw new DebugRequestException(
            String.format("Unknown thread %s: cannot resume.", threadId));
      }
      resumePausedThread(thread, stepping);
    }
  }

  /** Unpauses a currently-paused thread. */
  @GuardedBy("this")
  private void resumePausedThread(
      PausedThreadState thread, SkylarkDebuggingProtos.Stepping stepping) {
    pausedThreads.remove(thread.id);
    ReadyToPause readyToPause =
        thread.debuggable.stepControl(DebugEventHelper.convertSteppingEnum(stepping));
    if (readyToPause != null) {
      steppingThreads.put(thread.id, new SteppingThreadState(readyToPause));
    }
    thread.semaphore.release();
  }

  void pauseIfNecessary(Environment env, Location location, DebugServerTransport transport) {
    if (servicingEvalRequest.get()) {
      return;
    }
    PauseReason pauseReason = shouldPauseCurrentThread(env, location);
    if (pauseReason == null) {
      return;
    }
    long threadId = Thread.currentThread().getId();
    threadsToPause.remove(threadId);
    synchronized (this) {
      steppingThreads.remove(threadId);
    }
    pauseCurrentThread(env, pauseReason, location, transport);
  }

  /** Handles a {@code ListFramesRequest} and returns its response. */
  ImmutableList<SkylarkDebuggingProtos.Frame> listFrames(long threadId)
      throws DebugRequestException {
    synchronized (this) {
      PausedThreadState thread = pausedThreads.get(threadId);
      if (thread == null) {
        throw new DebugRequestException(
            String.format("Thread %s is not paused or does not exist.", threadId));
      }
      return thread
          .debuggable
          .listFrames(thread.location)
          .stream()
          .map(DebugEventHelper::getFrameProto)
          .collect(toImmutableList());
    }
  }

  SkylarkDebuggingProtos.Value evaluate(long threadId, String expression)
      throws DebugRequestException {
    Debuggable debuggable;
    synchronized (this) {
      PausedThreadState thread = pausedThreads.get(threadId);
      if (thread == null) {
        throw new DebugRequestException(
            String.format("Thread %s is not paused or does not exist.", threadId));
      }
      debuggable = thread.debuggable;
    }
    // no need to evaluate within the synchronize block: for paused threads, debuggable is only
    // accessed in response to a client request, and requests are handled serially
    // TODO(bazel-team): support asynchronous replies, and use finer-grained locks
    try {
      servicingEvalRequest.set(true);
      Object result = debuggable.evaluate(expression);
      return DebuggerSerialization.getValueProto("Evaluation result", result);
    } catch (EvalException | InterruptedException e) {
      throw new DebugRequestException(e.getMessage());
    } finally {
      servicingEvalRequest.set(false);
    }
  }

  /**
   * Pauses the current thread's execution, blocking until it's resumed via a
   * ContinueExecutionRequest.
   */
  private void pauseCurrentThread(
      Environment env, PauseReason pauseReason, Location location, DebugServerTransport transport) {
    long threadId = Thread.currentThread().getId();

    PausedThreadState pausedState =
        new PausedThreadState(threadId, Thread.currentThread().getName(), env, location);
    synchronized (this) {
      pausedThreads.put(threadId, pausedState);
    }
    SkylarkDebuggingProtos.PausedThread threadProto =
        getPausedThreadProto(pausedState, pauseReason);
    transport.postEvent(DebugEventHelper.threadPausedEvent(threadProto));
    pausedState.semaphore.acquireUninterruptibly();
    transport.postEvent(DebugEventHelper.threadContinuedEvent(threadId));
  }

  @Nullable
  private PauseReason shouldPauseCurrentThread(Environment env, Location location) {
    long threadId = Thread.currentThread().getId();
    if (pausingAllThreads) {
      return PauseReason.ALL_THREADS_PAUSED;
    }
    if (threadsToPause.contains(threadId)) {
      return PauseReason.PAUSE_THREAD_REQUEST;
    }
    if (hasBreakpointAtLocation(location)) {
      return PauseReason.HIT_BREAKPOINT;
    }

    // TODO(bazel-team): if contention becomes a problem, consider changing 'threads' to a
    // concurrent map, and synchronizing on individual entries
    synchronized (this) {
      SteppingThreadState steppingState = steppingThreads.get(threadId);
      if (steppingState != null && steppingState.readyToPause.test(env)) {
        return PauseReason.STEPPING;
      }
    }
    return null;
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
  private static SkylarkDebuggingProtos.PausedThread getPausedThreadProto(
      PausedThreadState thread, PauseReason pauseReason) {
    return SkylarkDebuggingProtos.PausedThread.newBuilder()
        .setId(thread.id)
        .setName(thread.name)
        .setPauseReason(pauseReason)
        .setLocation(DebugEventHelper.getLocationProto(thread.location))
        .build();
  }
}

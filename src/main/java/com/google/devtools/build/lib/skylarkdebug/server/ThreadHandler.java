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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.Breakpoint;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.Error;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.PauseReason;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.Value;
import com.google.devtools.build.lib.syntax.Debuggable;
import com.google.devtools.build.lib.syntax.Debuggable.ReadyToPause;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import java.util.Collection;
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

    final ThreadObjectMap objectMap;

    PausedThreadState(long id, String name, Debuggable debuggable, Location location) {
      this.id = id;
      this.name = name;
      this.debuggable = debuggable;
      this.location = location;
      this.semaphore = new Semaphore(0);
      this.objectMap = new ThreadObjectMap();
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

  /** Whether threads are globally paused, and if so, why. */
  private enum DebuggerState {
    INITIALIZING, // no StartDebuggingRequest has yet been received; all threads are paused
    ALL_THREADS_PAUSED, // all threads are paused in response to a PauseThreadRequest with id=0
    RUNNING, // normal running: threads are not globally paused
  }

  /** The debugger starts with all threads paused, until a StartDebuggingRequest is received. */
  private volatile DebuggerState debuggerState = DebuggerState.INITIALIZING;

  /** A map from identifiers of paused threads to their state info. */
  @GuardedBy("this")
  private final Map<Long, PausedThreadState> pausedThreads = new HashMap<>();

  /** A map from identifiers of stepping threads to their state. */
  @GuardedBy("this")
  private final Map<Long, SteppingThreadState> steppingThreads = new HashMap<>();

  /** All location-based breakpoints (the only type of breakpoint currently supported). */
  private volatile ImmutableMap<SkylarkDebuggingProtos.Location, SkylarkDebuggingProtos.Breakpoint>
      breakpoints = ImmutableMap.of();

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
    debuggerState = DebuggerState.ALL_THREADS_PAUSED;
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

  void setBreakpoints(Collection<Breakpoint> breakpoints) {
    Map<SkylarkDebuggingProtos.Location, SkylarkDebuggingProtos.Breakpoint> map = new HashMap<>();
    for (SkylarkDebuggingProtos.Breakpoint breakpoint : breakpoints) {
      if (breakpoint.getConditionCase()
          != SkylarkDebuggingProtos.Breakpoint.ConditionCase.LOCATION) {
        continue;
      }
      // all breakpoints cover the entire line, so unset the column number
      SkylarkDebuggingProtos.Location location =
          breakpoint.getLocation().toBuilder().clearColumnNumber().build();
      map.put(location, breakpoint);
    }
    this.breakpoints = ImmutableMap.copyOf(map);
  }

  /**
   * Resumes all threads. Any currently stepping threads have their stepping behavior cleared, so
   * will run unconditionally.
   */
  void resumeAllThreads() {
    threadsToPause.clear();
    debuggerState = DebuggerState.RUNNING;
    synchronized (this) {
      for (PausedThreadState thread : ImmutableList.copyOf(pausedThreads.values())) {
        // continue-all doesn't support stepping.
        resumePausedThread(thread, SkylarkDebuggingProtos.Stepping.NONE);
      }
      steppingThreads.clear();
    }
  }

  /**
   * Unpauses the given thread if it is currently paused. Also sets {@link #debuggerState} to
   * RUNNING. If the thread is not paused, but currently stepping, it clears the stepping behavior
   * so it will run unconditionally.
   */
  void resumeThread(long threadId, SkylarkDebuggingProtos.Stepping stepping)
      throws DebugRequestException {
    // once the user has requested any thread be resumed, don't continue pausing future threads
    debuggerState = DebuggerState.RUNNING;
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
    PauseReason pauseReason;
    Error error = null;
    try {
      pauseReason = shouldPauseCurrentThread(env, location);
    } catch (ConditionalBreakpointException e) {
      pauseReason = PauseReason.CONDITIONAL_BREAKPOINT_ERROR;
      error = Error.newBuilder().setMessage(e.getMessage()).build();
    }
    if (pauseReason == null) {
      return;
    }
    long threadId = Thread.currentThread().getId();
    threadsToPause.remove(threadId);
    synchronized (this) {
      steppingThreads.remove(threadId);
    }
    pauseCurrentThread(env, location, transport, pauseReason, error);
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
          .map(frame -> DebugEventHelper.getFrameProto(thread.objectMap, frame))
          .collect(toImmutableList());
    }
  }

  ImmutableList<Value> getChildrenForValue(long threadId, long valueId)
      throws DebugRequestException {
    ThreadObjectMap objectMap;
    synchronized (this) {
      PausedThreadState thread = pausedThreads.get(threadId);
      if (thread == null) {
        throw new DebugRequestException(
            String.format("Thread %s is not paused or does not exist.", threadId));
      }
      objectMap = thread.objectMap;
    }
    Object value = objectMap.getValue(valueId);
    if (value == null) {
      throw new DebugRequestException("Couldn't retrieve children; object not found.");
    }
    return DebuggerSerialization.getChildren(objectMap, value);
  }

  SkylarkDebuggingProtos.Value evaluate(long threadId, String statement)
      throws DebugRequestException {
    Debuggable debuggable;
    ThreadObjectMap objectMap;
    synchronized (this) {
      PausedThreadState thread = pausedThreads.get(threadId);
      if (thread == null) {
        throw new DebugRequestException(
            String.format("Thread %s is not paused or does not exist.", threadId));
      }
      debuggable = thread.debuggable;
      objectMap = thread.objectMap;
    }
    // no need to evaluate within the synchronize block: for paused threads, the debuggable and
    // object map are only accessed in response to a client request, and requests are handled
    // serially
    // TODO(bazel-team): support asynchronous replies, and use finer-grained locks
    try {
      Object result = doEvaluate(debuggable, statement);
      return DebuggerSerialization.getValueProto(objectMap, "Evaluation result", result);
    } catch (EvalException | InterruptedException e) {
      throw new DebugRequestException(e.getMessage());
    }
  }

  /**
   * Evaluate the given expression in the environment defined by the provided {@link Debuggable}.
   *
   * <p>The caller is responsible for ensuring that the associated skylark thread isn't currently
   * running.
   */
  private Object doEvaluate(Debuggable debuggable, String expression)
      throws EvalException, InterruptedException {
    try {
      servicingEvalRequest.set(true);
      return debuggable.evaluate(expression);
    } finally {
      servicingEvalRequest.set(false);
    }
  }

  /**
   * Pauses the current thread's execution, blocking until it's resumed via a
   * ContinueExecutionRequest.
   */
  private void pauseCurrentThread(
      Environment env,
      Location location,
      DebugServerTransport transport,
      PauseReason pauseReason,
      @Nullable Error conditionalBreakpointError) {
    long threadId = Thread.currentThread().getId();

    PausedThreadState pausedState =
        new PausedThreadState(threadId, Thread.currentThread().getName(), env, location);
    synchronized (this) {
      pausedThreads.put(threadId, pausedState);
    }
    SkylarkDebuggingProtos.PausedThread threadProto =
        getPausedThreadProto(pausedState, pauseReason, conditionalBreakpointError);
    transport.postEvent(DebugEventHelper.threadPausedEvent(threadProto));
    pausedState.semaphore.acquireUninterruptibly();
    transport.postEvent(DebugEventHelper.threadContinuedEvent(threadId));
  }

  @Nullable
  private PauseReason shouldPauseCurrentThread(Environment env, Location location)
      throws ConditionalBreakpointException {
    long threadId = Thread.currentThread().getId();
    DebuggerState state = debuggerState;
    if (state == DebuggerState.ALL_THREADS_PAUSED) {
      return PauseReason.ALL_THREADS_PAUSED;
    }
    if (state == DebuggerState.INITIALIZING) {
      return PauseReason.INITIALIZING;
    }
    if (threadsToPause.contains(threadId)) {
      return PauseReason.PAUSE_THREAD_REQUEST;
    }
    if (hasBreakpointMatchedAtLocation(env, location)) {
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

  /**
   * Returns true if there's a breakpoint at the current location, with a satisfied condition if
   * relevant.
   */
  private boolean hasBreakpointMatchedAtLocation(Environment env, Location location)
      throws ConditionalBreakpointException {
    // breakpoints is volatile, so taking a local copy
    ImmutableMap<SkylarkDebuggingProtos.Location, SkylarkDebuggingProtos.Breakpoint> breakpoints =
        this.breakpoints;
    if (breakpoints.isEmpty()) {
      return false;
    }
    SkylarkDebuggingProtos.Location locationProto = DebugEventHelper.getLocationProto(location);
    if (locationProto == null) {
      return false;
    }
    locationProto = locationProto.toBuilder().clearColumnNumber().build();
    SkylarkDebuggingProtos.Breakpoint breakpoint = breakpoints.get(locationProto);
    if (breakpoint == null) {
      return false;
    }
    String condition = breakpoint.getExpression();
    if (condition.isEmpty()) {
      return true;
    }
    try {
      return EvalUtils.toBoolean(doEvaluate(env, condition));
    } catch (EvalException | InterruptedException e) {
      throw new ConditionalBreakpointException(e.getMessage());
    }
  }

  /** Returns a {@code Thread} proto builder with information about the given thread. */
  private static SkylarkDebuggingProtos.PausedThread getPausedThreadProto(
      PausedThreadState thread,
      PauseReason pauseReason,
      @Nullable Error conditionalBreakpointError) {
    SkylarkDebuggingProtos.PausedThread.Builder builder =
        SkylarkDebuggingProtos.PausedThread.newBuilder()
            .setId(thread.id)
            .setName(thread.name)
            .setPauseReason(pauseReason)
            .setLocation(DebugEventHelper.getLocationProto(thread.location));
    if (conditionalBreakpointError != null) {
      builder.setConditionalBreakpointError(conditionalBreakpointError);
    }
    return builder.build();
  }
}

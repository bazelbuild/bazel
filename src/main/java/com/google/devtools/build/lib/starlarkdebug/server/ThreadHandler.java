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

package com.google.devtools.build.lib.starlarkdebug.server;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.Breakpoint;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.Error;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.PauseReason;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.Value;
import com.google.devtools.build.lib.syntax.Debug;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.FileOptions;
import com.google.devtools.build.lib.syntax.Location;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.ParserInput;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.SyntaxError;
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
    final StarlarkThread thread;
    /** The {@link Location} where execution is currently paused. */
    final Location location;
    /** Used to block execution of threads */
    final Semaphore semaphore;

    final ThreadObjectMap objectMap;

    PausedThreadState(long id, String name, StarlarkThread thread, Location location) {
      this.id = id;
      this.name = name;
      this.thread = thread;
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
    final Debug.ReadyToPause readyToPause;

    SteppingThreadState(Debug.ReadyToPause readyToPause) {
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
  private volatile ImmutableMap<
          StarlarkDebuggingProtos.Location, StarlarkDebuggingProtos.Breakpoint>
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
    Map<StarlarkDebuggingProtos.Location, StarlarkDebuggingProtos.Breakpoint> map = new HashMap<>();
    for (StarlarkDebuggingProtos.Breakpoint breakpoint : breakpoints) {
      if (breakpoint.getConditionCase()
          != StarlarkDebuggingProtos.Breakpoint.ConditionCase.LOCATION) {
        continue;
      }
      // all breakpoints cover the entire line, so unset the column number
      StarlarkDebuggingProtos.Location location =
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
        resumePausedThread(thread, StarlarkDebuggingProtos.Stepping.NONE);
      }
      steppingThreads.clear();
    }
  }

  /**
   * Unpauses the given thread if it is currently paused. Also sets {@link #debuggerState} to
   * RUNNING. If the thread is not paused, but currently stepping, it clears the stepping behavior
   * so it will run unconditionally.
   */
  void resumeThread(long threadId, StarlarkDebuggingProtos.Stepping stepping)
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
      PausedThreadState thread, StarlarkDebuggingProtos.Stepping stepping) {
    pausedThreads.remove(thread.id);
    Debug.ReadyToPause readyToPause =
        Debug.stepControl(thread.thread, DebugEventHelper.convertSteppingEnum(stepping));
    if (readyToPause != null) {
      steppingThreads.put(thread.id, new SteppingThreadState(readyToPause));
    }
    thread.semaphore.release();
  }

  void pauseIfNecessary(StarlarkThread thread, Location location, DebugServerTransport transport) {
    if (servicingEvalRequest.get()) {
      return;
    }
    PauseReason pauseReason;
    Error error = null;
    try {
      pauseReason = shouldPauseCurrentThread(thread, location);
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
    pauseCurrentThread(thread, location, transport, pauseReason, error);
  }

  /** Handles a {@code ListFramesRequest} and returns its response. */
  ImmutableList<StarlarkDebuggingProtos.Frame> listFrames(long threadId)
      throws DebugRequestException {
    synchronized (this) {
      PausedThreadState thread = pausedThreads.get(threadId);
      if (thread == null) {
        throw new DebugRequestException(
            String.format("Thread %s is not paused or does not exist.", threadId));
      }
      return Debug.getCallStack(thread.thread).stream()
          .map(frame -> DebugEventHelper.getFrameProto(thread.objectMap, frame))
          .collect(toImmutableList())
          .reverse();
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

  StarlarkDebuggingProtos.Value evaluate(long threadId, String statement)
      throws DebugRequestException {
    StarlarkThread thread;
    ThreadObjectMap objectMap;
    synchronized (this) {
      PausedThreadState threadState = pausedThreads.get(threadId);
      if (threadState == null) {
        throw new DebugRequestException(
            String.format("Thread %s is not paused or does not exist.", threadId));
      }
      thread = threadState.thread;
      objectMap = threadState.objectMap;
    }
    // no need to evaluate within the synchronize block: for paused threads, the thread and
    // object map are only accessed in response to a client request, and requests are handled
    // serially
    // TODO(bazel-team): support asynchronous replies, and use finer-grained locks
    try {
      Object result = doEvaluate(thread, statement);
      return DebuggerSerialization.getValueProto(objectMap, "Evaluation result", result);
    } catch (SyntaxError.Exception | EvalException | InterruptedException e) {
      throw new DebugRequestException(e.getMessage());
    }
  }

  /**
   * Executes the Starlark statements code in the environment defined by the provided {@link
   * StarlarkThread}. If the last statement is an expression, doEvaluate returns its value,
   * otherwise it returns null.
   *
   * <p>The caller is responsible for ensuring that the associated Starlark thread isn't currently
   * running.
   */
  private Object doEvaluate(StarlarkThread thread, String content)
      throws SyntaxError.Exception, EvalException, InterruptedException {
    try {
      servicingEvalRequest.set(true);

      // TODO(adonovan): opt: don't parse and resolve the expression every time we hit a breakpoint
      // (!).
      ParserInput input = ParserInput.fromString(content, "<debug eval>");
      // TODO(adonovan): the module or call frame should be a parameter.
      Module module = Module.ofInnermostEnclosingStarlarkFunction(thread);
      return EvalUtils.exec(input, FileOptions.DEFAULT, module, thread);
    } finally {
      servicingEvalRequest.set(false);
    }
  }

  /**
   * Pauses the current thread's execution, blocking until it's resumed via a
   * ContinueExecutionRequest.
   */
  private void pauseCurrentThread(
      StarlarkThread thread,
      Location location,
      DebugServerTransport transport,
      PauseReason pauseReason,
      @Nullable Error conditionalBreakpointError) {
    long threadId = Thread.currentThread().getId();

    PausedThreadState pausedState =
        new PausedThreadState(threadId, Thread.currentThread().getName(), thread, location);
    synchronized (this) {
      pausedThreads.put(threadId, pausedState);
    }
    StarlarkDebuggingProtos.PausedThread threadProto =
        getPausedThreadProto(pausedState, pauseReason, conditionalBreakpointError);
    transport.postEvent(DebugEventHelper.threadPausedEvent(threadProto));
    pausedState.semaphore.acquireUninterruptibly();
    transport.postEvent(DebugEventHelper.threadContinuedEvent(threadId));
  }

  @Nullable
  private PauseReason shouldPauseCurrentThread(StarlarkThread thread, Location location)
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
    if (hasBreakpointMatchedAtLocation(thread, location)) {
      return PauseReason.HIT_BREAKPOINT;
    }

    // TODO(bazel-team): if contention becomes a problem, consider changing 'threads' to a
    // concurrent map, and synchronizing on individual entries
    synchronized (this) {
      SteppingThreadState steppingState = steppingThreads.get(threadId);
      if (steppingState != null && steppingState.readyToPause.test(thread)) {
        return PauseReason.STEPPING;
      }
    }
    return null;
  }

  /**
   * Returns true if there's a breakpoint at the current location, with a satisfied condition if
   * relevant.
   */
  private boolean hasBreakpointMatchedAtLocation(StarlarkThread thread, Location location)
      throws ConditionalBreakpointException {
    // breakpoints is volatile, so taking a local copy
    ImmutableMap<StarlarkDebuggingProtos.Location, StarlarkDebuggingProtos.Breakpoint> breakpoints =
        this.breakpoints;
    if (breakpoints.isEmpty()) {
      return false;
    }
    StarlarkDebuggingProtos.Location locationProto = DebugEventHelper.getLocationProto(location);
    if (locationProto == null) {
      return false;
    }
    locationProto = locationProto.toBuilder().clearColumnNumber().build();
    StarlarkDebuggingProtos.Breakpoint breakpoint = breakpoints.get(locationProto);
    if (breakpoint == null) {
      return false;
    }
    String condition = breakpoint.getExpression();
    if (condition.isEmpty()) {
      return true;
    }
    try {
      return Starlark.truth(doEvaluate(thread, condition));
    } catch (SyntaxError.Exception | EvalException | InterruptedException e) {
      throw new ConditionalBreakpointException(e.getMessage());
    }
  }

  /** Returns a {@code Thread} proto builder with information about the given thread. */
  private static StarlarkDebuggingProtos.PausedThread getPausedThreadProto(
      PausedThreadState thread,
      PauseReason pauseReason,
      @Nullable Error conditionalBreakpointError) {
    StarlarkDebuggingProtos.PausedThread.Builder builder =
        StarlarkDebuggingProtos.PausedThread.newBuilder()
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

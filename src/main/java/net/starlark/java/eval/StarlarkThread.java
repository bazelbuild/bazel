// Copyright 2014 The Bazel Authors. All rights reserved.
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

package net.starlark.java.eval;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;
import net.starlark.java.syntax.Location;

/**
 * An StarlarkThread represents a Starlark thread.
 *
 * <p>It holds the stack of active Starlark and built-in function calls. In addition, it may hold
 * per-thread application state (see {@link #setThreadLocal}) that passes through Starlark functions
 * but does not directly affect them, such as information about the BUILD file being loaded.
 *
 * <p>Every {@code StarlarkThread} has a {@link Mutability} field, and must be used within a
 * function that creates and closes this {@link Mutability} with the try-with-resource pattern. This
 * {@link Mutability} is also used when initializing mutable objects within that {@code
 * StarlarkThread}. When the {@code Mutability} is closed at the end of the computation, it freezes
 * the {@code StarlarkThread} along with all of those objects. This pattern enforces the discipline
 * that there should be no dangling mutable {@code StarlarkThread}, or concurrency between
 * interacting {@code StarlarkThread}s. It is a Starlark-level error to attempt to mutate a frozen
 * {@code StarlarkThread} or its objects, but it is a Java-level error to attempt to mutate an
 * unfrozen {@code StarlarkThread} or its objects from within a different {@code StarlarkThread}.
 */
public final class StarlarkThread {

  /** The mutability of values created by this thread. */
  private final Mutability mutability;

  // profiler state
  //
  // The profiler field (and savedThread) are set when we first observe during a
  // push (function call entry) that the profiler is active. They are unset
  // not in the corresponding pop, but when the last frame is popped, because
  // the profiler session might start in the middle of a call and/or run beyond
  // the lifetime of this thread.
  final AtomicInteger cpuTicks = new AtomicInteger();
  @Nullable private CpuProfiler profiler;
  StarlarkThread savedThread; // saved StarlarkThread, when profiling reentrant evaluation

  private final Map<Class<?>, Object> threadLocals = new HashMap<>();

  private boolean interruptible = true;

  long steps; // count of logical computation steps executed so far
  long stepLimit = Long.MAX_VALUE; // limit on logical computation steps

  /**
   * Returns the number of Starlark computation steps executed by this thread according to a
   * small-step semantics. (Today, that means exec, eval, and assign operations executed by the
   * tree-walking evaluator, but in future will mean byte code instructions; the two are not
   * commensurable.)
   */
  public long getExecutedSteps() {
    return steps;
  }

  /**
   * Sets the maximum number of Starlark computation steps that may be executed by this thread (see
   * {@link #getExecutedSteps}). When the step counter reaches or exceeds this value, execution
   * fails with an EvalException.
   */
  public void setMaxExecutionSteps(long steps) {
    this.stepLimit = steps;
  }

  /**
   * Disables polling of the {@link java.lang.Thread#interrupted} flag during Starlark evaluation.
   */
  // TODO(adonovan): expose a public API for this if we can establish a stronger semantics. (There
  // are other ways besides polling for evaluation to be interrupted, such as calling certain
  // built-in functions.)
  void ignoreThreadInterrupts() {
    interruptible = false;
  }

  void checkInterrupt() throws InterruptedException {
    if (interruptible && Thread.interrupted()) {
      throw new InterruptedException();
    }
  }

  /**
   * setThreadLocal saves {@code value} as a thread-local variable of this Starlark thread, keyed by
   * {@code key}, so that it can later be retrieved by {@code getThreadLocal(key)}.
   */
  public <T> void setThreadLocal(Class<T> key, T value) {
    threadLocals.put(key, value);
  }

  /**
   * getThreadLocal returns the value {@code v} supplied to the most recent {@code
   * setThreadLocal(key, v)} call, or null if there was no prior call.
   */
  public <T> T getThreadLocal(Class<T> key) {
    Object v = threadLocals.get(key);
    return v == null ? null : key.cast(v);
  }

  /** A Frame records information about an active function call. */
  static final class Frame implements Debug.Frame {
    final StarlarkThread thread;
    final StarlarkCallable fn; // the called function

    @Nullable
    final Debug.Debugger dbg = Debug.debugger.get(); // the debugger, if active for this frame

    int compcount = 0; // number of enclosing comprehensions

    Object result = Starlark.NONE; // the operand of a Starlark return statement

    // Current PC location. Initially fn.getLocation(); for Starlark functions,
    // it is updated at key points when it may be observed: calls, breakpoints, errors.
    private Location loc;

    // Indicates that setErrorLocation has been called already and the error
    // location (loc) should not be overrwritten.
    private boolean errorLocationSet;

    // The locals of this frame, if fn is a StarlarkFunction, otherwise empty.
    Map<String, Object> locals;

    @Nullable private Object profileSpan; // current span of walltime call profiler

    private Frame(StarlarkThread thread, StarlarkCallable fn) {
      this.thread = thread;
      this.fn = fn;
    }

    // Updates the PC location in this frame.
    void setLocation(Location loc) {
      this.loc = loc;
    }

    // Sets location only the first time it is called,
    // to ensure that the location of the innermost expression
    // is used for errors.
    // (Once we switch to a bytecode interpreter, we can afford
    // to update fr.pc before each fallible operation, but until then
    // we must materialize Locations only after the fact of failure.)
    // Sets errorLocationSet.
    void setErrorLocation(Location loc) {
      if (!errorLocationSet) {
        errorLocationSet = true;
        setLocation(loc);
      }
    }

    @Override
    public StarlarkCallable getFunction() {
      return fn;
    }

    @Override
    public Location getLocation() {
      return loc;
    }

    @Override
    public ImmutableMap<String, Object> getLocals() {
      return ImmutableMap.copyOf(this.locals);
    }

    @Override
    public String toString() {
      return fn.getName() + "@" + loc;
    }
  }

  /** The semantics options that affect how Starlark code is evaluated. */
  private final StarlarkSemantics semantics;

  /** Whether recursive calls are allowed (cached from semantics). */
  private final boolean allowRecursion;

  /** PrintHandler for Starlark print statements. */
  private PrintHandler printHandler = StarlarkThread::defaultPrintHandler;

  /** Loader for Starlark load statements. Null if loading is disallowed. */
  @Nullable private Loader loader = null;

  /** Stack of active function calls. */
  private final ArrayList<Frame> callstack = new ArrayList<>();

  /** A hook for notifications of assignments at top level. */
  PostAssignHook postAssignHook;

  /** Pushes a function onto the call stack. */
  void push(StarlarkCallable fn) {
    Frame fr = new Frame(this, fn);
    callstack.add(fr);

    // Notify debug tools of the thread's first push.
    if (callstack.size() == 1 && Debug.threadHook != null) {
      Debug.threadHook.onPushFirst(this);
    }

    if (fn instanceof StarlarkFunction) {
      StarlarkFunction sfn = (StarlarkFunction) fn;
      fr.locals = Maps.newLinkedHashMapWithExpectedSize(sfn.getParameterNames().size());
    } else {
      // built-in function
      fr.locals = ImmutableMap.of();
    }

    fr.loc = fn.getLocation();

    // Start wall-time call profile span.
    CallProfiler callProfiler = StarlarkThread.callProfiler;
    if (callProfiler != null) {
      fr.profileSpan = callProfiler.start(fn);
    }

    // Poll for newly installed CPU profiler.
    if (profiler == null) {
      this.profiler = CpuProfiler.get();
      if (profiler != null) {
        cpuTicks.set(0);
        // Associated current Java thread with this StarlarkThread.
        // (Save the previous association so we can restore it later.)
        this.savedThread = CpuProfiler.setStarlarkThread(this);
      }
    }
  }

  /** Pops a function off the call stack. */
  void pop() {
    int last = callstack.size() - 1;
    Frame fr = callstack.get(last);

    if (profiler != null) {
      int ticks = cpuTicks.getAndSet(0);
      if (ticks > 0) {
        profiler.addEvent(ticks, getDebugCallStack());
      }

      // If this is the final pop in this thread,
      // unregister it from the profiler.
      if (last == 0) {
        // Restore the previous association (in case of reentrant evaluation).
        CpuProfiler.setStarlarkThread(this.savedThread);
        this.savedThread = null;
        this.profiler = null;
      }
    }

    callstack.remove(last); // pop

    // End wall-time profile span.
    CallProfiler callProfiler = StarlarkThread.callProfiler;
    if (callProfiler != null && fr.profileSpan != null) {
      callProfiler.end(fr.profileSpan);
    }

    // Notify debug tools of the thread's last pop.
    if (last == 0 && Debug.threadHook != null) {
      Debug.threadHook.onPopLast(this);
    }
  }

  /** Returns the mutability for values created by this thread. */
  public Mutability mutability() {
    return mutability;
  }

  /**
   * A PrintHandler determines how a Starlark thread deals with print statements. It is invoked by
   * the built-in {@code print} function. Its default behavior is to write the message to standard
   * error, preceded by the location of the print statement, {@code thread.getCallerLocation()}.
   */
  @FunctionalInterface
  public interface PrintHandler {
    void print(StarlarkThread thread, String msg);
  }

  /** Returns the PrintHandler for Starlark print statements. */
  PrintHandler getPrintHandler() {
    return printHandler;
  }

  /** Sets the behavior of Starlark print statements executed by this thread. */
  public void setPrintHandler(PrintHandler h) {
    this.printHandler = Preconditions.checkNotNull(h);
  }

  private static void defaultPrintHandler(StarlarkThread thread, String msg) {
    System.err.println(thread.getCallerLocation() + ": " + msg);
  }

  /**
   * A Loader determines the behavior of load statements executed by this thread. It returns the
   * named module, or null if not found.
   */
  @FunctionalInterface
  public interface Loader {
    @Nullable
    Module load(String module);
  }

  /** Returns the loader for Starlark load statements. */
  Loader getLoader() {
    return loader;
  }

  /** Sets the behavior of Starlark load statements executed by this thread. */
  public void setLoader(Loader loader) {
    this.loader = Preconditions.checkNotNull(loader);
  }

  /** Reports whether {@code fn} has been recursively reentered within this thread. */
  boolean isRecursiveCall(StarlarkFunction fn) {
    // Find fn buried within stack. (The top of the stack is assumed to be fn.)
    for (int i = callstack.size() - 2; i >= 0; --i) {
      Frame fr = callstack.get(i);
      // TODO(adonovan): compare code, not closure values, otherwise
      // one can defeat this check by writing the Y combinator.
      if (fr.fn.equals(fn)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns the location of the program counter in the enclosing call frame. If called from within
   * a built-in function, this is the location of the call expression that called the built-in. It
   * returns BUILTIN if called with fewer than two frames (such as within a test).
   */
  public Location getCallerLocation() {
    return toplevel() ? Location.BUILTIN : frame(1).loc;
  }

  /**
   * Reports whether the call stack has less than two frames. Zero frames means an idle thread. One
   * frame means the function for the top-level statements of a file is active. More than that means
   * a function call is in progress.
   *
   * <p>Every use of this function is a hack to work around the lack of proper local vs global
   * identifier resolution at top level.
   */
  boolean toplevel() {
    return callstack.size() < 2;
  }

  // Returns the stack frame at the specified depth. 0 means top of stack, 1 is its caller, etc.
  Frame frame(int depth) {
    return callstack.get(callstack.size() - 1 - depth);
  }

  /**
   * Constructs a StarlarkThread.
   *
   * @param mu the (non-frozen) mutability of values created by this thread.
   * @param semantics the StarlarkSemantics for this thread.
   */
  public StarlarkThread(Mutability mu, StarlarkSemantics semantics) {
    Preconditions.checkArgument(!mu.isFrozen());
    this.mutability = mu;
    this.semantics = semantics;
    this.allowRecursion = semantics.getBool(StarlarkSemantics.ALLOW_RECURSION);
  }

  /**
   * Specifies a hook function to be run after each assignment at top level.
   *
   * <p>This is a short-term hack to allow us to consolidate all StarlarkFile execution in one place
   * even while BzlLoadFunction implements the old "export" behavior, in which rules, aspects and
   * providers are "exported" as soon as they are assigned, not at the end of file execution.
   */
  public void setPostAssignHook(PostAssignHook postAssignHook) {
    this.postAssignHook = postAssignHook;
  }

  /** A hook for notifications of assignments at top level. */
  @FunctionalInterface
  public interface PostAssignHook {
    void assign(String name, Object value);
  }

  public StarlarkSemantics getSemantics() {
    return semantics;
  }

  /** Reports whether this thread is allowed to make recursive calls. */
  public boolean isRecursionAllowed() {
    return allowRecursion;
  }

  // Implementation of Debug.getCallStack.
  // Intentionally obscured to steer most users to the simpler getCallStack.
  ImmutableList<Debug.Frame> getDebugCallStack() {
    return ImmutableList.<Debug.Frame>copyOf(callstack);
  }

  /** Returns the size of the callstack. This is needed for the debugger. */
  int getCallStackSize() {
    return callstack.size();
  }

  /**
   * A CallStackEntry describes the name and PC location of an active function call. See {@link
   * #getCallStack}.
   */
  @Immutable
  public static final class CallStackEntry {
    public final String name;
    public final Location location;

    public CallStackEntry(String name, Location location) {
      this.location = location;
      this.name = name;
    }

    @Override
    public String toString() {
      return name + "@" + location;
    }
  }

  /**
   * Returns information about this thread's current stack of active function calls, outermost call
   * first. For each function, it reports its name, and the location of its current program counter.
   * The result is immutable and does not reference interpreter data structures, so it may retained
   * indefinitely and safely shared with other threads.
   */
  public ImmutableList<CallStackEntry> getCallStack() {
    ImmutableList.Builder<CallStackEntry> stack = ImmutableList.builder();
    for (Frame fr : callstack) {
      stack.add(new CallStackEntry(fr.fn.getName(), fr.loc));
    }
    return stack.build();
  }

  @Override
  public int hashCode() {
    throw new UnsupportedOperationException(); // avoid nondeterminism
  }

  @Override
  public boolean equals(Object that) {
    throw new UnsupportedOperationException();
  }

  @Override
  public String toString() {
    return String.format("<StarlarkThread%s>", mutability);
  }

  /** CallProfiler records the start and end wall times of function calls. */
  public interface CallProfiler {
    Object start(StarlarkCallable fn);

    void end(Object span);
  }

  /** Installs a global hook that will be notified of function calls. */
  public static void setCallProfiler(@Nullable CallProfiler p) {
    callProfiler = p;
  }

  @Nullable private static CallProfiler callProfiler = null;
}

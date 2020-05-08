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

package com.google.devtools.build.lib.syntax;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import javax.annotation.Nullable;

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
 *
 * <p>One creates an StarlarkThread using the {@link #builder} function, before evaluating code in
 * it with {@link StarlarkFile#eval}, or with {@link StarlarkFile#exec} (where the AST was obtained
 * by passing a {@link Resolver} constructed from the StarlarkThread to {@link StarlarkFile#parse}.
 * When the computation is over, the frozen StarlarkThread can still be queried with {@link
 * #lookup}.
 */
public final class StarlarkThread {

  // The mutability of the StarlarkThread comes from its initial module.
  // TODO(adonovan): not every thread initializes a module.
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
    @Nullable final Debugger dbg = Debug.debugger.get(); // the debugger, if active for this frame
    int compcount = 0; // number of enclosing comprehensions

    Object result = Starlark.NONE; // the operand of a Starlark return statement

    // Current PC location. Initially fn.getLocation(); for Starlark functions,
    // it is updated at key points when it may be observed: calls, breakpoints.
    private Location loc;

    // The locals of this frame, if fn is a StarlarkFunction, otherwise empty.
    Map<String, Object> locals;

    @Nullable private SilentCloseable profileSpan; // current span of walltime profiler

    private Frame(StarlarkThread thread, StarlarkCallable fn) {
      this.thread = thread;
      this.fn = fn;
    }

    // Updates the PC location in this frame.
    void setLocation(Location loc) {
      this.loc = loc;
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

  // The module initialized by this Starlark thread.
  //
  // TODO(adonovan): eliminate. First we need to simplify the set-up sequence like so:
  //
  //    // Filter predeclaredEnv based on semantics,
  //    // create a mutability, and retain the semantics:
  //    Module module = new Module(semantics, predeclaredEnv);
  //
  //    // Create a thread that takes its semantics and mutability
  //    // (and only them) from the Module.
  //    StarlarkThread thread = StarlarkThread.toInitializeModule(module);
  //
  // Then clients that call thread.getGlobals() should use 'module' directly.
  private final Module module;

  /** The semantics options that affect how Starlark code is evaluated. */
  private final StarlarkSemantics semantics;

  /** PrintHandler for Starlark print statements. */
  private PrintHandler printHandler = StarlarkThread::defaultPrintHandler;

  /** Loaded modules, keyed by load string. */
  private final Map<String, Module> loadedModules;

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

    ProfilerTask taskKind;
    if (fn instanceof StarlarkFunction) {
      StarlarkFunction sfn = (StarlarkFunction) fn;
      fr.locals = Maps.newLinkedHashMapWithExpectedSize(sfn.getParameterNames().size());
      taskKind = ProfilerTask.STARLARK_USER_FN;
    } else {
      // built-in function
      fr.locals = ImmutableMap.of();
      taskKind = ProfilerTask.STARLARK_BUILTIN_FN;
    }

    fr.loc = fn.getLocation();

    // start wall-time profile span
    // TODO(adonovan): throw this away when we build a CPU profiler.
    if (Profiler.instance().isActive()) {
      fr.profileSpan = Profiler.instance().profile(taskKind, fn.getName());
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

    // end profile span
    if (fr.profileSpan != null) {
      fr.profileSpan.close();
    }

    // Notify debug tools of the thread's last pop.
    if (last == 0 && Debug.threadHook != null) {
      Debug.threadHook.onPopLast(this);
    }
  }

  public Mutability mutability() {
    return mutability;
  }

  /** Returns the module initialized by this StarlarkThread. */
  // TODO(adonovan): get rid of this. Logically, a thread doesn't have module, but every
  // Starlark source function does. If you want to know the module of the innermost
  // enclosing call from a function defined in Starlark source code, use
  // Module.ofInnermostEnclosingStarlarkFunction.
  public Module getGlobals() {
    return module;
  }

  /**
   * A PrintHandler determines how a Starlark thread deals with print statements. It is invoked by
   * the built-in {@code print} function. Its default behavior is to write the message to standard
   * error, preceded by the location of the print statement, {@code thread.getCallerLocation()}.
   */
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
   * Constructs a StarlarkThread. This is the main, most basic constructor.
   *
   * @param module the module initialized by this StarlarkThread
   * @param semantics the StarlarkSemantics for this thread.
   * @param loadedModules modules for each load statement in the file
   */
  private StarlarkThread(
      Module module, StarlarkSemantics semantics, Map<String, Module> loadedModules) {
    this.module = Preconditions.checkNotNull(module);
    this.mutability = module.mutability();
    Preconditions.checkArgument(!module.mutability().isFrozen());
    this.semantics = semantics;
    this.loadedModules = loadedModules;
  }

  /**
   * A Builder class for StarlarkThread.
   *
   * <p>The caller must explicitly set the semantics by calling either {@link #setSemantics} or
   * {@link #useDefaultSemantics}.
   */
  // TODO(adonovan): eliminate the builder:
  // - replace loadedModules by a callback, since there's no need to enumerate them now.
  // - decouple Module from thread.
  public static class Builder {
    private final Mutability mutability;
    @Nullable private Module parent;
    @Nullable private StarlarkSemantics semantics;
    private Map<String, Module> loadedModules = ImmutableMap.of();

    Builder(Mutability mutability) {
      this.mutability = mutability;
    }

    /**
     * Inherits global bindings from the given parent Frame.
     *
     * <p>TODO(laurentlb): this should be called setUniverse.
     */
    public Builder setGlobals(Module parent) {
      Preconditions.checkState(this.parent == null);
      this.parent = parent;
      return this;
    }

    public Builder setSemantics(StarlarkSemantics semantics) {
      this.semantics = semantics;
      return this;
    }

    public Builder useDefaultSemantics() {
      this.semantics = StarlarkSemantics.DEFAULT;
      return this;
    }

    /** Sets the modules to be provided to each load statement. */
    public Builder setLoadedModules(Map<String, Module> loadedModules) {
      this.loadedModules = loadedModules;
      return this;
    }

    /** Builds the StarlarkThread. */
    public StarlarkThread build() {
      Preconditions.checkArgument(!mutability.isFrozen());
      if (semantics == null) {
        throw new IllegalArgumentException("must call either setSemantics or useDefaultSemantics");
      }
      // Filter out restricted objects from the universe scope. This cannot be done in-place in
      // creation of the input global universe scope, because this environment's semantics may not
      // have been available during its creation. Thus, create a new universe scope for this
      // environment which is equivalent in every way except that restricted bindings are
      // filtered out.
      parent = Module.filterOutRestrictedBindings(mutability, parent, semantics);

      Module module = new Module(mutability, parent);
      return new StarlarkThread(module, semantics, loadedModules);
    }
  }

  public static Builder builder(Mutability mutability) {
    return new Builder(mutability);
  }

  /**
   * Specifies a hook function to be run after each assignment at top level.
   *
   * <p>This is a short-term hack to allow us to consolidate all StarlarkFile execution in one place
   * even while StarlarkImportLookupFunction implements the old "export" behavior, in which rules,
   * aspects and providers are "exported" as soon as they are assigned, not at the end of file
   * execution.
   */
  public void setPostAssignHook(PostAssignHook postAssignHook) {
    this.postAssignHook = postAssignHook;
  }

  /** A hook for notifications of assignments at top level. */
  public interface PostAssignHook {
    void assign(String name, Object value);
  }

  public StarlarkSemantics getSemantics() {
    return semantics;
  }

  /**
   * Returns a set of all names of variables that are accessible in this {@code StarlarkThread}, in
   * a deterministic order.
   */
  // TODO(adonovan): eliminate this once we do resolution.
  Set<String> getVariableNames() {
    LinkedHashSet<String> vars = new LinkedHashSet<>();
    if (!callstack.isEmpty()) {
      vars.addAll(frame(0).locals.keySet());
    }
    vars.addAll(module.getTransitiveBindings().keySet());
    return vars;
  }

  // Implementation of Debug.getCallStack.
  // Intentionally obscured to steer most users to the simpler getCallStack.
  ImmutableList<Debug.Frame> getDebugCallStack() {
    return ImmutableList.<Debug.Frame>copyOf(callstack);
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

  /**
   * Given a requested stepping behavior, returns a predicate over the context that tells the
   * debugger when to pause. (Debugger API)
   *
   * <p>The predicate will return true if we are at the next statement where execution should pause,
   * and it will return false if we are not yet at that statement. No guarantee is made about the
   * predicate's return value after we have reached the desired statement.
   *
   * <p>A null return value indicates that no further pausing should occur.
   */
  // TODO(adonovan): move to Debug.
  @Nullable
  public ReadyToPause stepControl(Stepping stepping) {
    final int depth = callstack.size();
    switch (stepping) {
      case NONE:
        return null;
      case INTO:
        // pause at the very next statement
        return thread -> true;
      case OVER:
        return thread -> thread.callstack.size() <= depth;
      case OUT:
        // if we're at the outermost frame, same as NONE
        return depth == 0 ? null : thread -> thread.callstack.size() < depth;
    }
    throw new IllegalArgumentException("Unsupported stepping type: " + stepping);
  }

  /** See stepControl (Debugger API) */
  // TODO(adonovan): move to Debug.
  public interface ReadyToPause extends Predicate<StarlarkThread> {}

  /**
   * Describes the stepping behavior that should occur when execution of a thread is continued.
   * (Debugger API)
   */
  // TODO(adonovan): move to Debug.
  public enum Stepping {
    /** Continue execution without stepping. */
    NONE,
    /**
     * If the thread is paused on a statement that contains a function call, step into that
     * function. Otherwise, this is the same as OVER.
     */
    INTO,
    /**
     * Step over the current statement and any functions that it may call, stopping at the next
     * statement in the same frame. If no more statements are available in the current frame, same
     * as OUT.
     */
    OVER,
    /**
     * Continue execution until the current frame has been exited and then pause. If we are
     * currently in the outer-most frame, same as NONE.
     */
    OUT,
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
    return String.format("<StarlarkThread%s>", mutability());
  }

  Module getModule(String module) {
    return loadedModules.get(module);
  }
}

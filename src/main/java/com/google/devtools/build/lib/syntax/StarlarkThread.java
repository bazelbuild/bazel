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

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;
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
 * interacting {@code StarlarkThread}s. It is a Skylark-level error to attempt to mutate a frozen
 * {@code StarlarkThread} or its objects, but it is a Java-level error to attempt to mutate an
 * unfrozen {@code StarlarkThread} or its objects from within a different {@code StarlarkThread}.
 *
 * <p>One creates an StarlarkThread using the {@link #builder} function, before evaluating code in
 * it with {@link StarlarkFile#eval}, or with {@link StarlarkFile#exec} (where the AST was obtained
 * by passing a {@link ValidationEnvironment} constructed from the StarlarkThread to {@link
 * StarlarkFile#parse}. When the computation is over, the frozen StarlarkThread can still be queried
 * with {@link #lookup}.
 */
public final class StarlarkThread {

  // The mutability of the StarlarkThread comes from its initial module.
  // TODO(adonovan): not every thread initializes a module.
  private final Mutability mutability;

  private final Map<Class<?>, Object> threadLocals = new HashMap<>();

  private boolean interruptible = true;

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

  /** A CallFrame records information about an active function call. */
  // TODO(adonovan): move CallFrame to top level as "Frame".
  static final class CallFrame implements Debug.Frame {
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

    // Note that the inherited design is off-by-one:
    // the following fields are logically facts about the _enclosing_ frame.
    // TODO(adonovan): fix that.

    private final Module savedModule; // the saved module of the parent (TODO(adonovan): eliminate)

    CallFrame(StarlarkThread thread, StarlarkCallable fn, Module savedModule) {
      this.thread = thread;
      this.fn = fn;
      this.savedModule = savedModule;
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

  /** An Extension to be imported with load() into a BUILD or .bzl file. */
  @Immutable
  // TODO(janakr,brandjon): Do Extensions actually have to start their own memoization? Or can we
  // have a node higher up in the hierarchy inject the mutability?
  // TODO(adonovan): identify Extension with Module, abolish hash code, and make loading lazy (a
  // callback not a map) so that clients don't need to preemptively scan the set of load statements.
  @AutoCodec
  public static final class Extension {

    private final ImmutableMap<String, Object> bindings;

    /**
     * Cached hash code for the transitive content of this {@code Extension} and its dependencies.
     *
     * <p>Note that "content" refers to the AST content, not the evaluated bindings.
     */
    private final String transitiveContentHashCode;

    /** Constructs with the given hash code and bindings. */
    @AutoCodec.Instantiator
    public Extension(ImmutableMap<String, Object> bindings, String transitiveContentHashCode) {
      this.bindings = bindings;
      this.transitiveContentHashCode = transitiveContentHashCode;
    }

    /**
     * Constructs using the bindings from the global definitions of the given {@link
     * StarlarkThread}, and that {@code StarlarkThread}'s transitive hash code.
     */
    public Extension(StarlarkThread thread) {
      // Legacy behavior: all symbols from the global Frame are exported (including symbols
      // introduced by load).
      this(
          ImmutableMap.copyOf(thread.globalFrame.getExportedBindings()),
          thread.getTransitiveContentHashCode());
    }

    private String getTransitiveContentHashCode() {
      return transitiveContentHashCode;
    }

    /** Retrieves all bindings, in a deterministic order. */
    public ImmutableMap<String, Object> getBindings() {
      return bindings;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof Extension)) {
        return false;
      }
      Extension other = (Extension) obj;
      return transitiveContentHashCode.equals(other.getTransitiveContentHashCode())
          && bindings.equals(other.getBindings());
    }

    private static boolean skylarkObjectsProbablyEqual(Object obj1, Object obj2) {
      // TODO(b/76154791): check this more carefully.
      return obj1.equals(obj2)
          || (obj1 instanceof StarlarkValue
              && obj2 instanceof StarlarkValue
              && Starlark.repr(obj1).equals(Starlark.repr(obj2)));
    }

    /**
     * Throws {@link IllegalStateException} if this {@link Extension} is not equal to {@code obj}.
     *
     * <p>The exception explains the reason for the inequality, including all unequal bindings.
     */
    public void checkStateEquals(Object obj) {
      if (this == obj) {
        return;
      }
      if (!(obj instanceof Extension)) {
        throw new IllegalStateException(
            String.format(
                "Expected an equal Extension, but got a %s instead of an Extension",
                obj == null ? "null" : obj.getClass().getName()));
      }
      Extension other = (Extension) obj;
      ImmutableMap<String, Object> otherBindings = other.getBindings();

      Set<String> names = bindings.keySet();
      Set<String> otherNames = otherBindings.keySet();
      if (!names.equals(otherNames)) {
        throw new IllegalStateException(
            String.format(
                "Expected Extensions to be equal, but they don't define the same bindings: "
                    + "in this one but not given one: [%s]; in given one but not this one: [%s]",
                Joiner.on(", ").join(Sets.difference(names, otherNames)),
                Joiner.on(", ").join(Sets.difference(otherNames, names))));
      }

      ArrayList<String> badEntries = new ArrayList<>();
      for (String name : names) {
        Object value = bindings.get(name);
        Object otherValue = otherBindings.get(name);
        if (value.equals(otherValue)) {
          continue;
        }
        if (value instanceof Depset) {
          if (otherValue instanceof Depset
              && ((Depset) value).toCollection().equals(((Depset) otherValue).toCollection())) {
            continue;
          }
        } else if (value instanceof Dict) {
          if (otherValue instanceof Dict) {
            @SuppressWarnings("unchecked")
            Dict<Object, Object> thisDict = (Dict<Object, Object>) value;
            @SuppressWarnings("unchecked")
            Dict<Object, Object> otherDict = (Dict<Object, Object>) otherValue;
            if (thisDict.size() == otherDict.size()
                && thisDict.keySet().equals(otherDict.keySet())) {
              boolean foundProblem = false;
              for (Object key : thisDict.keySet()) {
                if (!skylarkObjectsProbablyEqual(
                    Preconditions.checkNotNull(thisDict.get(key), key),
                    Preconditions.checkNotNull(otherDict.get(key), key))) {
                  foundProblem = true;
                }
              }
              if (!foundProblem) {
                continue;
              }
            }
          }
        } else if (skylarkObjectsProbablyEqual(value, otherValue)) {
          continue;
        }
        badEntries.add(
            String.format(
                "%s: this one has %s (class %s, %s), but given one has %s (class %s, %s)",
                name,
                Starlark.repr(value),
                value.getClass().getName(),
                value,
                Starlark.repr(otherValue),
                otherValue.getClass().getName(),
                otherValue));
      }
      if (!badEntries.isEmpty()) {
        throw new IllegalStateException(
            "Expected Extensions to be equal, but the following bindings are unequal: "
                + Joiner.on("; ").join(badEntries));
      }

      if (!transitiveContentHashCode.equals(other.getTransitiveContentHashCode())) {
        throw new IllegalStateException(
            String.format(
                "Expected Extensions to be equal, but transitive content hashes don't match:"
                    + " %s != %s",
                transitiveContentHashCode, other.getTransitiveContentHashCode()));
      }
    }

    @Override
    public int hashCode() {
      return Objects.hash(bindings, transitiveContentHashCode);
    }
  }

  // Global environment of the current topmost call frame,
  // or of the file about to be initialized if no calls are active.
  // TODO(adonovan): eliminate once we represent even toplevel statements
  // as a StarlarkFunction that closes over its Module.
  private Module globalFrame;

  /** The semantics options that affect how Skylark code is evaluated. */
  private final StarlarkSemantics semantics;

  /** PrintHandler for Starlark print statements. */
  private PrintHandler printHandler = StarlarkThread::defaultPrintHandler;

  /**
   * For each imported extension, a global Skylark frame from which to load() individual bindings.
   */
  private final Map<String, Extension> importedExtensions;

  /** Stack of active function calls. */
  private final ArrayList<CallFrame> callstack = new ArrayList<>();

  /** A hook for notifications of assignments at top level. */
  PostAssignHook postAssignHook;

  /** Pushes a function onto the call stack. */
  void push(StarlarkCallable fn) {
    CallFrame fr = new CallFrame(this, fn, this.globalFrame);
    callstack.add(fr);

    // Push the function onto the allocation tracker's stack.
    // TODO(adonovan): optimize it out of existence.
    if (Callstack.enabled) {
      Callstack.push(fn);
    }

    ProfilerTask taskKind;
    if (fn instanceof StarlarkFunction) {
      StarlarkFunction sfn = (StarlarkFunction) fn;
      fr.locals = Maps.newLinkedHashMapWithExpectedSize(sfn.getSignature().numParameters());
      this.globalFrame = sfn.getModule();
      taskKind = ProfilerTask.STARLARK_USER_FN;
    } else {
      // built-in function
      fr.locals = ImmutableMap.of();
      // this.globalFrame is left as is.
      // For built-ins, thread.globals() returns the module
      // of the file from which the built-in was called.
      // Really they have no business knowing about that.

      taskKind = ProfilerTask.STARLARK_BUILTIN_FN;
    }

    fr.loc = fn.getLocation();

    // start profile span
    // TODO(adonovan): throw this away when we build a CPU profiler.
    if (Profiler.instance().isActive()) {
      fr.profileSpan = Profiler.instance().profile(taskKind, fn.getName());
    }
  }

  /** Pops a function off the call stack. */
  void pop() {
    int last = callstack.size() - 1;
    CallFrame top = callstack.get(last);
    callstack.remove(last); // pop
    this.globalFrame = top.savedModule;

    // end profile span
    if (top.profileSpan != null) {
      top.profileSpan.close();
    }

    if (Callstack.enabled) {
      Callstack.pop();
    }
  }

  private final String transitiveHashCode;

  public Mutability mutability() {
    return mutability;
  }

  /** Returns the global variables for the StarlarkThread (not including dynamic bindings). */
  // TODO(adonovan): get rid of this. Logically, a thread doesn't have module, but every
  // Starlark source function does.
  public Module getGlobals() {
    return globalFrame;
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

  /** Returns a PrintHandler that sends DEBUG events to the provided EventHandler. */
  // TODO(adonovan): move to lib.events.Event when we reverse the dependency.
  // For now, clients call thread.setPrintHandler(StarlarkThread.makeDebugPrintHandler(h));
  public static PrintHandler makeDebugPrintHandler(EventHandler h) {
    return (thread, msg) -> h.handle(Event.debug(thread.getCallerLocation(), msg));
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
      CallFrame fr = callstack.get(i);
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
  CallFrame frame(int depth) {
    return callstack.get(callstack.size() - 1 - depth);
  }

  /**
   * Constructs a StarlarkThread. This is the main, most basic constructor.
   *
   * @param globalFrame a frame for the global StarlarkThread
   * @param eventHandler an EventHandler for warnings, errors, etc
   * @param importedExtensions Extensions from which to import bindings with load()
   * @param fileContentHashCode a hash for the source file being evaluated, if any
   */
  private StarlarkThread(
      Module globalFrame,
      StarlarkSemantics semantics,
      Map<String, Extension> importedExtensions,
      @Nullable String fileContentHashCode) {
    this.globalFrame = Preconditions.checkNotNull(globalFrame);
    this.mutability = globalFrame.mutability();
    Preconditions.checkArgument(!globalFrame.mutability().isFrozen());
    this.semantics = semantics;
    this.importedExtensions = importedExtensions;
    this.transitiveHashCode =
        computeTransitiveContentHashCode(fileContentHashCode, importedExtensions);
  }

  /**
   * A Builder class for StarlarkThread.
   *
   * <p>The caller must explicitly set the semantics by calling either {@link #setSemantics} or
   * {@link #useDefaultSemantics}.
   */
  // TODO(adonovan): eliminate the builder:
  // - replace importedExtensions by a callback
  // - eliminate fileContentHashCode
  // - decouple Module from thread.
  public static class Builder {
    private final Mutability mutability;
    @Nullable private Module parent;
    @Nullable private StarlarkSemantics semantics;
    @Nullable private Map<String, Extension> importedExtensions;
    @Nullable private String fileContentHashCode;

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
      this.semantics = StarlarkSemantics.DEFAULT_SEMANTICS;
      return this;
    }

    /** Declares imported extensions for load() statements. */
    public Builder setImportedExtensions(Map<String, Extension> importMap) {
      Preconditions.checkState(this.importedExtensions == null);
      this.importedExtensions = importMap;
      return this;
    }

    /** Declares content hash for the source file for this StarlarkThread. */
    public Builder setFileContentHashCode(String fileContentHashCode) {
      this.fileContentHashCode = fileContentHashCode;
      return this;
    }

    /** Builds the StarlarkThread. */
    public StarlarkThread build() {
      Preconditions.checkArgument(!mutability.isFrozen());
      if (semantics == null) {
        throw new IllegalArgumentException("must call either setSemantics or useDefaultSemantics");
      }
      if (parent != null) {
        Preconditions.checkArgument(parent.mutability().isFrozen(), "parent frame must be frozen");
        if (parent.universe != null) { // This code path doesn't happen in Bazel.

          // Flatten the frame, ensure all builtins are in the same frame.
          parent =
              new Module(
                  parent.mutability(),
                  null /* parent */,
                  parent.label,
                  parent.getTransitiveBindings(),
                  parent.restrictedBindings);
        }
      }

      // Filter out restricted objects from the universe scope. This cannot be done in-place in
      // creation of the input global universe scope, because this environment's semantics may not
      // have been available during its creation. Thus, create a new universe scope for this
      // environment which is equivalent in every way except that restricted bindings are
      // filtered out.
      parent = Module.filterOutRestrictedBindings(mutability, parent, semantics);

      Module globalFrame = new Module(mutability, parent);
      if (importedExtensions == null) {
        importedExtensions = ImmutableMap.of();
      }
      return new StarlarkThread(globalFrame, semantics, importedExtensions, fileContentHashCode);
    }
  }

  public static Builder builder(Mutability mutability) {
    return new Builder(mutability);
  }

  /**
   * Specifies a hook function to be run after each assignment at top level.
   *
   * <p>This is a short-term hack to allow us to consolidate all StarlarkFile execution in one place
   * even while SkylarkImportLookupFunction implements the old "export" behavior, in which rules,
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
    vars.addAll(globalFrame.getTransitiveBindings().keySet());
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
    for (CallFrame fr : callstack) {
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

  Extension getExtension(String module) {
    return importedExtensions.get(module);
  }

  /**
   * Computes a deterministic hash for the given base hash code and extension map (the map's order
   * does not matter).
   */
  private static String computeTransitiveContentHashCode(
      @Nullable String baseHashCode, Map<String, Extension> importedExtensions) {
    // Calculate a new hash from the hash of the loaded Extensions.
    Fingerprint fingerprint = new Fingerprint();
    if (baseHashCode != null) {
      fingerprint.addString(Preconditions.checkNotNull(baseHashCode));
    }
    TreeSet<String> importStrings = new TreeSet<>(importedExtensions.keySet());
    for (String importString : importStrings) {
      fingerprint.addString(importedExtensions.get(importString).getTransitiveContentHashCode());
    }
    return fingerprint.hexDigestAndReset();
  }

  /**
   * Returns a hash code calculated from the hash code of this StarlarkThread and the transitive
   * closure of other StarlarkThreads it loads.
   */
  public String getTransitiveContentHashCode() {
    return transitiveHashCode;
  }
}

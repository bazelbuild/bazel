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
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.Mutability.Freezable;
import com.google.devtools.build.lib.syntax.Mutability.MutabilityException;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.SpellChecker;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
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
 * <p>One creates an StarlarkThread using the {@link #builder} function, then populates it with
 * {@link #setup} and sometimes {@link #setupOverride}, before to evaluate code in it with {@link
 * BuildFileAST#eval}, or with {@link BuildFileAST#exec} (where the AST was obtained by passing a
 * {@link ValidationEnvironment} constructed from the StarlarkThread to {@link BuildFileAST#parse}.
 * When the computation is over, the frozen StarlarkThread can still be queried with {@link
 * #lookup}.
 */
// TODO(adonovan): further steps for StarlarkThread remediation:
// Its API should expose the following concepts, and no more:
// 1) "thread local variables": this holds per-thread application
//    state such as the current Label, or BUILD package, for all the
//    native.* built-ins.
//    This may include any thread-specific behaviour relevant to the
//    load and print statements.
// 2) a stack of call frames, each representing an active function call.
//    Only clients needing debugger-like powers of reflection should need
//    this, such as the debugger itself, and the ill-conceived
//    generator_name attribute. The API for call frames should not
//    expose an object of class CallFrame, because for efficiency we
//    will want to recycle objects in place rather than generate garbage
//    on every call.
//    So the API will look like getCallerLocation(depth),
//    not getCaller(depth).location, with one method per "public" CallFrame
//    attribute, such as location.
//    We must expose these basic CallFrame attributes, for stack traces and errors:
//    - function name
//    - PC location
//    Advanced clients such as the debugger, and the generator_name rule attribute, also need:
//    - the function value (Warning: careless clients can pin closures in memory)
//    - Object getLocalValue(Identifier parameter).
// 3) Debugging support (thread name, profiling counters, etc).
// And that is all. See go.starlark.net for the model.
//
// The Frame interface should be hidden from clients and then eliminated.
// The dynamic lookup mechanism should go away.
// The GlobalFrame class should be redesigned.
// The concept struggling to get out of it is a Module,
// which is created before file initialization and
// populated by execution of the top-level statements in a file;
// every UserDefinedFunction value should hold a reference to its Module.
// As best I can tell, all the skyframe serialization
// as it applies to LexicalFrames is redundant, as these are transient
// and should not exist after loading.
// We will remove the FuncallExpression parameter from StarlarkFunction.call.
// Clients should use getCallerLocation instead.
// The Continuation class should be deleted.
// Once the API is small and sound, we can start to represent all
// the lexical frames within a single function using just an array,
// indexed by a small integer computed during the validation pass.
public final class StarlarkThread implements Freezable {

  /**
   * A mapping of bindings, either mutable or immutable according to an associated {@link
   * Mutability}. The order of the bindings within a single {@link Frame} is deterministic but
   * unspecified.
   *
   * <p>Any non-frozen {@link Frame} must have the same {@link Mutability} as the current {@link
   * StarlarkThread}, to avoid interference from other evaluation contexts. For example, a {@link
   * StarlarkFunction} will close over the global frame of the {@link StarlarkThread} in which it
   * was defined. When the function is called from other {@link StarlarkThread}s (possibly
   * simultaneously), that global frame must already be frozen; a new local {@link Frame} is created
   * to represent the lexical scope of the function.
   *
   * <p>A {@link Frame} can have an associated "parent" {@link Frame}, which is used in {@link #get}
   * and {@link #getTransitiveBindings()}
   *
   * <p>TODO(laurentlb): "parent" should be named "universe" since it contains only the builtins.
   * The "get" method shouldn't look at the universe (so that "moduleLookup" works as expected)
   */
  private interface Frame extends Freezable {
    /**
     * Gets a binding from this {@link Frame} or one of its transitive parents.
     *
     * <p>In case of conflicts, the binding found in the {@link Frame} closest to the current one is
     * used; the remaining bindings are shadowed.
     *
     * @param varname the name of the variable whose value should be retrieved
     * @return the value bound to the variable, or null if no binding is found
     */
    @Nullable
    Object get(String varname);

    /**
     * Assigns or reassigns a binding in the current {@code Frame}.
     *
     * <p>If the binding has the same name as one in a transitive parent, the parent binding is
     * shadowed (i.e., the parent is unaffected).
     *
     * @param thread the {@link StarlarkThread} attempting the mutation
     * @param varname the name of the variable to be bound
     * @param value the value to bind to the variable
     */
    void put(StarlarkThread thread, String varname, Object value) throws MutabilityException;

    /**
     * TODO(laurentlb): Remove this method when possible. It should probably not be part of the
     * public interface.
     */
    void remove(StarlarkThread thread, String varname) throws MutabilityException;

    /**
     * Returns a map containing all bindings of this {@link Frame} and of its transitive parents,
     * taking into account shadowing precedence.
     *
     * <p>The bindings are returned in a deterministic order (for a given sequence of initial values
     * and updates).
     */
    Map<String, Object> getTransitiveBindings();
  }

  interface LexicalFrame extends Frame {
    static LexicalFrame create(Mutability mutability) {
      return mutability.isFrozen()
          ? ImmutableEmptyLexicalFrame.INSTANCE
          : new MutableLexicalFrame(mutability);
    }

    static LexicalFrame create(Mutability mutability, int numArgs) {
      Preconditions.checkState(!mutability.isFrozen());
      return new MutableLexicalFrame(mutability, /*initialCapacity=*/ numArgs);
    }
  }

  private static final class ImmutableEmptyLexicalFrame implements LexicalFrame {
    private static final ImmutableEmptyLexicalFrame INSTANCE = new ImmutableEmptyLexicalFrame();

    @Override
    public Mutability mutability() {
      return Mutability.IMMUTABLE;
    }

    @Nullable
    @Override
    public Object get(String varname) {
      return null;
    }

    @Override
    public void put(StarlarkThread thread, String varname, Object value)
        throws MutabilityException {
      Mutability.checkMutable(this, thread.mutability());
      throw new IllegalStateException();
    }

    @Override
    public void remove(StarlarkThread thread, String varname) throws MutabilityException {
      Mutability.checkMutable(this, thread.mutability());
      throw new IllegalStateException();
    }

    @Override
    public Map<String, Object> getTransitiveBindings() {
      return ImmutableMap.of();
    }

    @Override
    public String toString() {
      return "<ImmutableEmptyLexicalFrame>";
    }
  }

  private static final class MutableLexicalFrame implements LexicalFrame {
    private final Mutability mutability;
    /** Bindings are maintained in order of creation. */
    private final LinkedHashMap<String, Object> bindings;

    private MutableLexicalFrame(Mutability mutability, int initialCapacity) {
      this.mutability = mutability;
      this.bindings = Maps.newLinkedHashMapWithExpectedSize(initialCapacity);
    }

    private MutableLexicalFrame(Mutability mutability) {
      this.mutability = mutability;
      this.bindings = new LinkedHashMap<>();
    }

    @Override
    public Mutability mutability() {
      return mutability;
    }

    @Nullable
    @Override
    public Object get(String varname) {
      return bindings.get(varname);
    }

    @Override
    public void put(StarlarkThread thread, String varname, Object value)
        throws MutabilityException {
      Mutability.checkMutable(this, thread.mutability());
      bindings.put(varname, value);
    }

    @Override
    public void remove(StarlarkThread thread, String varname) throws MutabilityException {
      Mutability.checkMutable(this, thread.mutability());
      bindings.remove(varname);
    }

    @Override
    public Map<String, Object> getTransitiveBindings() {
      return bindings;
    }

    @Override
    public String toString() {
      return String.format("<MutableLexicalFrame%s>", mutability());
    }
  }

  /**
   * A {@link Frame} that represents the top-level definitions of a file. It contains the
   * module-scope variables and has a reference to the universe.
   *
   * <p>Bindings in a {@link GlobalFrame} may shadow those inherited from its universe.
   *
   * <p>A {@link GlobalFrame} can also be constructed in a two-phase process. To do this, call the
   * nullary constructor to create an uninitialized {@link GlobalFrame}, then call {@link
   * #initialize}. It is illegal to use any other method in-between these two calls, or to call
   * {@link #initialize} on an already initialized {@link GlobalFrame}.
   */
  public static final class GlobalFrame implements Frame {
    /**
     * Final, except that it may be initialized after instantiation. Null mutability indicates that
     * this Frame is uninitialized.
     */
    @Nullable private Mutability mutability;

    /** Final, except that it may be initialized after instantiation. */
    @Nullable private GlobalFrame universe;

    // The label (an optional piece of metatata) associated with the file.
    @Nullable private Object label;

    /** Bindings are maintained in order of creation. */
    private final LinkedHashMap<String, Object> bindings;

    /**
     * A list of bindings which *would* exist in this global frame under certain semantic
     * flags, but do not exist using the semantic flags used in this frame's creation.
     * This map should not be used for lookups; it should only be used to throw descriptive
     * error messages when a lookup of a restricted object is attempted.
     **/
    private final LinkedHashMap<String, FlagGuardedValue> restrictedBindings;

    /** Set of bindings that are exported (can be loaded from other modules). */
    private final HashSet<String> exportedBindings;

    /** Constructs an uninitialized instance; caller must call {@link #initialize} before use. */
    public GlobalFrame() {
      this.mutability = null;
      this.universe = null;
      this.label = null;
      this.bindings = new LinkedHashMap<>();
      this.restrictedBindings = new LinkedHashMap<>();
      this.exportedBindings = new HashSet<>();
    }

    GlobalFrame(
        Mutability mutability,
        @Nullable GlobalFrame universe,
        @Nullable Object label,
        @Nullable Map<String, Object> bindings,
        @Nullable Map<String, FlagGuardedValue> restrictedBindings) {
      Preconditions.checkState(universe == null || universe.universe == null);
      this.mutability = Preconditions.checkNotNull(mutability);
      this.universe = universe;
      if (label != null) {
        this.label = label;
      } else if (universe != null) {
        this.label = universe.label;
      } else {
        this.label = null;
      }
      this.bindings = new LinkedHashMap<>();
      if (bindings != null) {
        this.bindings.putAll(bindings);
      }
      this.restrictedBindings = new LinkedHashMap<>();
      if (restrictedBindings != null) {
        this.restrictedBindings.putAll(restrictedBindings);
      }
      if (universe != null) {
        this.restrictedBindings.putAll(universe.restrictedBindings);
      }
      this.exportedBindings = new HashSet<>();
    }

    public GlobalFrame(Mutability mutability) {
      this(mutability, null, null, null, null);
    }

    public GlobalFrame(Mutability mutability, @Nullable GlobalFrame universe) {
      this(mutability, universe, null, null, null);
    }

    public GlobalFrame(
        Mutability mutability, @Nullable GlobalFrame universe, @Nullable Object label) {
      this(mutability, universe, label, null, null);
    }

    /** Constructs a global frame for the given builtin bindings. */
    public static GlobalFrame createForBuiltins(Map<String, Object> bindings) {
      Mutability mutability = Mutability.create("<builtins>").freeze();
      return new GlobalFrame(mutability, null, null, bindings, null);
    }

    /**
     * Constructs a global frame based on the given parent frame, filtering out flag-restricted
     * global objects.
     */
    private static GlobalFrame filterOutRestrictedBindings(
        Mutability mutability, GlobalFrame parent, StarlarkSemantics semantics) {
      if (parent == null) {
        return new GlobalFrame(mutability);
      }
      Map<String, Object> filteredBindings = new LinkedHashMap<>();
      Map<String, FlagGuardedValue> restrictedBindings = new LinkedHashMap<>();

      for (Entry<String, Object> binding : parent.getTransitiveBindings().entrySet()) {
        if (binding.getValue() instanceof FlagGuardedValue) {
          FlagGuardedValue val = (FlagGuardedValue) binding.getValue();
          if (val.isObjectAccessibleUsingSemantics(semantics)) {
            filteredBindings.put(binding.getKey(), val.getObject(semantics));
          } else {
            restrictedBindings.put(binding.getKey(), val);
          }
        } else {
          filteredBindings.put(binding.getKey(), binding.getValue());
        }
      }

      restrictedBindings.putAll(parent.restrictedBindings);

      return new GlobalFrame(
          mutability,
          null /*parent */,
          parent.label,
          filteredBindings,
          restrictedBindings);
    }

    private void checkInitialized() {
      Preconditions.checkNotNull(mutability, "Attempted to use Frame before initializing it");
    }

    public void initialize(
        Mutability mutability,
        @Nullable GlobalFrame universe,
        @Nullable Object label,
        Map<String, Object> bindings) {
      Preconditions.checkState(
          universe == null || universe.universe == null); // no more than 1 universe
      Preconditions.checkState(
          this.mutability == null, "Attempted to initialize an already initialized Frame");
      this.mutability = Preconditions.checkNotNull(mutability);
      this.universe = universe;
      if (label != null) {
        this.label = label;
      } else if (universe != null) {
        this.label = universe.label;
      } else {
        this.label = null;
      }
      this.bindings.putAll(bindings);
    }

    /**
     * Returns a new {@link GlobalFrame} with the same fields, except that {@link #label} is set to
     * the given value. The label associated with each function (frame) on the stack is accessible
     * using {@link #getLabel}, and is included in the result of {@code str(fn)} where {@code fn} is
     * a StarlarkFunction.
     */
    public GlobalFrame withLabel(Object label) {
      checkInitialized();
      return new GlobalFrame(mutability, /*universe*/ null, label, bindings,
          /*restrictedBindings*/ null);
    }

    /** Returns the {@link Mutability} of this {@link GlobalFrame}. */
    @Override
    public Mutability mutability() {
      checkInitialized();
      return mutability;
    }

    /**
     * Returns the parent {@link GlobalFrame}, if it exists.
     *
     * <p>TODO(laurentlb): Should be called getUniverse.
     */
    @Nullable
    public GlobalFrame getParent() {
      checkInitialized();
      return universe;
    }

    /**
     * Returns the label (an optional piece of metadata) associated with this {@code GlobalFrame}.
     * (For Bazel LOADING threads, this is the build label of its BUILD or .bzl file.)
     */
    @Nullable
    public Object getLabel() {
      checkInitialized();
      return label;
    }

    /**
     * Returns a map of direct bindings of this {@link GlobalFrame}, ignoring universe.
     *
     * <p>The bindings are returned in a deterministic order (for a given sequence of initial values
     * and updates).
     *
     * <p>For efficiency an unmodifiable view is returned. Callers should assume that the view is
     * invalidated by any subsequent modification to the {@link GlobalFrame}'s bindings.
     */
    public Map<String, Object> getBindings() {
      checkInitialized();
      return Collections.unmodifiableMap(bindings);
    }

    /**
     * Returns a map of bindings that are exported (i.e. symbols declared using `=` and
     * `def`, but not `load`).
     */
    public Map<String, Object> getExportedBindings() {
      checkInitialized();
      ImmutableMap.Builder<String, Object> result = new ImmutableMap.Builder<>();
      for (Map.Entry<String, Object> entry : bindings.entrySet()) {
        if (exportedBindings.contains(entry.getKey())) {
          result.put(entry);
        }
      }
      return result.build();
    }

    @Override
    public Map<String, Object> getTransitiveBindings() {
      checkInitialized();
      // Can't use ImmutableMap.Builder because it doesn't allow duplicates.
      LinkedHashMap<String, Object> collectedBindings = new LinkedHashMap<>();
      if (universe != null) {
        collectedBindings.putAll(universe.getTransitiveBindings());
      }
      collectedBindings.putAll(getBindings());
      return collectedBindings;
    }

    public Object getDirectBindings(String varname) {
      checkInitialized();
      return bindings.get(varname);
    }

    @Override
    public Object get(String varname) {
      checkInitialized();
      Object val = bindings.get(varname);
      if (val != null) {
        return val;
      }
      if (universe != null) {
        return universe.get(varname);
      }
      return null;
    }

    @Override
    public void put(StarlarkThread thread, String varname, Object value)
        throws MutabilityException {
      checkInitialized();
      Mutability.checkMutable(this, thread.mutability());
      bindings.put(varname, value);
    }

    @Override
    public void remove(StarlarkThread thread, String varname) throws MutabilityException {
      checkInitialized();
      Mutability.checkMutable(this, thread.mutability());
      bindings.remove(varname);
    }

    @Override
    public String toString() {
      if (mutability == null) {
        return "<Uninitialized GlobalFrame>";
      } else {
        return String.format("<GlobalFrame%s>", mutability());
      }
    }
  }

  // The mutability of the StarlarkThread comes from its initial global frame.
  private final Mutability mutability;

  private final Map<Class<?>, Object> threadLocals = new HashMap<>();

  /**
   * setThreadLocal saves {@code value} as a thread-local variable of this Starlark thread, keyed by
   * {@code key}, so that it can later be retrieved by {@code getThreadLocal(key)}.
   */
  public <T> void setThreadLocal(Class<T> key, T value) {
    // The clazz parameter is redundant, but it makes the API clearer.
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

  /**
   * A Continuation contains data saved during a function call and restored when the function exits.
   */
  private static final class Continuation {
    /** The {@link BaseFunction} being evaluated that will return into this Continuation. */
    final BaseFunction function;

    /** The {@link FuncallExpression} to which this Continuation will return. */
    @Nullable final FuncallExpression caller;

    /** The next Continuation after this Continuation. */
    @Nullable final Continuation continuation;

    /** The lexical Frame of the caller. */
    final Frame lexicalFrame;

    /** The global Frame of the caller. */
    final GlobalFrame globalFrame;

    Continuation(
        @Nullable Continuation continuation,
        BaseFunction function,
        @Nullable FuncallExpression caller,
        Frame lexicalFrame,
        GlobalFrame globalFrame) {
      this.continuation = continuation;
      this.function = function;
      this.caller = caller;
      this.lexicalFrame = lexicalFrame;
      this.globalFrame = globalFrame;
    }
  }

  /** An Extension to be imported with load() into a BUILD or .bzl file. */
  @Immutable
  // TODO(janakr,brandjon): Do Extensions actually have to start their own memoization? Or can we
  // have a node higher up in the hierarchy inject the mutability?
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
          ImmutableMap.copyOf(
              thread.getSemantics().incompatibleNoTransitiveLoads()
                  ? thread.globalFrame.getExportedBindings()
                  : thread.globalFrame.getBindings()),
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
          || (obj1 instanceof SkylarkValue
              && obj2 instanceof SkylarkValue
              && Printer.repr(obj1).equals(Printer.repr(obj2)));
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
        if (value instanceof SkylarkNestedSet) {
          if (otherValue instanceof SkylarkNestedSet
              && ((SkylarkNestedSet) value)
                  .toCollection()
                  .equals(((SkylarkNestedSet) otherValue).toCollection())) {
            continue;
          }
        } else if (value instanceof SkylarkDict) {
          if (otherValue instanceof SkylarkDict) {
            @SuppressWarnings("unchecked")
            SkylarkDict<Object, Object> thisDict = (SkylarkDict<Object, Object>) value;
            @SuppressWarnings("unchecked")
            SkylarkDict<Object, Object> otherDict = (SkylarkDict<Object, Object>) otherValue;
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
                Printer.repr(value),
                value.getClass().getName(),
                value,
                Printer.repr(otherValue),
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

  /**
   * Static Frame for lexical variables that are always looked up in the current StarlarkThread or
   * for the definition StarlarkThread of the function currently being evaluated.
   */
  private Frame lexicalFrame;

  /**
   * Static Frame for global variables; either the current lexical Frame if evaluation is currently
   * happening at the global scope of a BUILD file, or the global Frame at the time of function
   * definition if evaluation is currently happening in the body of a function. Thus functions can
   * close over other functions defined in the same file.
   */
  private GlobalFrame globalFrame;

  /** The semantics options that affect how Skylark code is evaluated. */
  private final StarlarkSemantics semantics;

  /**
   * An EventHandler for errors and warnings. This is not used in the BUILD language, however it
   * might be used in Skylark code called from the BUILD language, so shouldn't be null.
   */
  private final EventHandler eventHandler;

  /**
   * For each imported extension, a global Skylark frame from which to load() individual bindings.
   */
  private final Map<String, Extension> importedExtensions;

  /**
   * When in a lexical (Skylark) frame, this lists the names of the functions in the call stack. We
   * currently use it to artificially disable recursion.
   */
  @Nullable private Continuation continuation;

  /**
   * Enters a scope by saving state to a new Continuation
   *
   * @param function the function whose scope to enter
   * @param lexical the lexical frame to use
   * @param caller the source AST node for the caller
   * @param globals the global Frame that this function closes over from its definition
   *     StarlarkThread
   */
  void enterScope(
      BaseFunction function,
      Frame lexical,
      @Nullable FuncallExpression caller,
      GlobalFrame globals) {
    continuation = new Continuation(continuation, function, caller, lexicalFrame, globalFrame);
    lexicalFrame = lexical;
    globalFrame = globals;
  }

  /** Exits a scope by restoring state from the current continuation */
  void exitScope() {
    Preconditions.checkNotNull(continuation);
    lexicalFrame = continuation.lexicalFrame;
    globalFrame = continuation.globalFrame;
    continuation = continuation.continuation;
  }

  private final String transitiveHashCode;

  /**
   * Is this a global StarlarkThread?
   *
   * @return true if the current code is being executed at the top-level, as opposed to inside the
   *     body of a function.
   */
  boolean isGlobal() {
    return lexicalFrame instanceof GlobalFrame;
  }

  @Override
  public Mutability mutability() {
    return mutability;
  }

  /** Returns the global variables for the StarlarkThread (not including dynamic bindings). */
  public GlobalFrame getGlobals() {
    return globalFrame;
  }

  /**
   * Returns an EventHandler for errors and warnings. The BUILD language doesn't use it directly,
   * but can call Skylark code that does use it.
   *
   * @return an EventHandler
   */
  // TODO(adonovan): turn this into a print handler and break dependency on EventHandler.
  // First, we must report scan/parse/validation errors using an exception containing events.
  public EventHandler getEventHandler() {
    return eventHandler;
  }

  /**
   * Returns if calling the supplied function would be a recursive call, or in other words if the
   * supplied function is already on the stack.
   */
  boolean isRecursiveCall(StarlarkFunction function) {
    for (Continuation k = continuation; k != null; k = k.continuation) {
      // TODO(adonovan): compare code, not closure values, otherwise
      // one can defeat this check by writing the Y combinator.
      if (k.function.equals(function)) {
        return true;
      }
    }
    return false;
  }

  /** Returns the current function call, if it exists. */
  @Nullable
  BaseFunction getCurrentFunction() {
    return continuation != null ? continuation.function : null;
  }

  /** Returns the FuncallExpression and the BaseFunction for the top-level call being evaluated. */
  // TODO(adonovan): replace this by an API for walking the call stack.
  public Pair<FuncallExpression, BaseFunction> getTopCall() {
    Continuation continuation = this.continuation;
    if (continuation == null) {
      return null;
    }
    while (continuation.continuation != null) {
      continuation = continuation.continuation;
    }
    return new Pair<>(continuation.caller, continuation.function);
  }

  /**
   * Constructs an StarlarkThread. This is the main, most basic constructor.
   *
   * @param globalFrame a frame for the global StarlarkThread
   * @param eventHandler an EventHandler for warnings, errors, etc
   * @param importedExtensions Extension-s from which to import bindings with load()
   * @param fileContentHashCode a hash for the source file being evaluated, if any
   */
  private StarlarkThread(
      GlobalFrame globalFrame,
      StarlarkSemantics semantics,
      EventHandler eventHandler,
      Map<String, Extension> importedExtensions,
      @Nullable String fileContentHashCode) {
    this.lexicalFrame = Preconditions.checkNotNull(globalFrame);
    this.globalFrame = Preconditions.checkNotNull(globalFrame);
    this.mutability = globalFrame.mutability();
    Preconditions.checkArgument(!globalFrame.mutability().isFrozen());
    this.semantics = semantics;
    this.eventHandler = eventHandler;
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
  public static class Builder {
    private final Mutability mutability;
    @Nullable private GlobalFrame parent;
    @Nullable private StarlarkSemantics semantics;
    @Nullable private EventHandler eventHandler;
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
    public Builder setGlobals(GlobalFrame parent) {
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

    /** Sets an EventHandler for errors and warnings. */
    public Builder setEventHandler(EventHandler eventHandler) {
      Preconditions.checkState(this.eventHandler == null);
      this.eventHandler = eventHandler;
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
              new GlobalFrame(
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
      parent = GlobalFrame.filterOutRestrictedBindings(mutability, parent, semantics);

      GlobalFrame globalFrame = new GlobalFrame(mutability, parent);
      if (importedExtensions == null) {
        importedExtensions = ImmutableMap.of();
      }
      return new StarlarkThread(
          globalFrame, semantics, eventHandler, importedExtensions, fileContentHashCode);
    }
  }

  public static Builder builder(Mutability mutability) {
    return new Builder(mutability);
  }

  /** Remove variable from local bindings. */
  void removeLocalBinding(String varname) {
    try {
      lexicalFrame.remove(this, varname);
    } catch (MutabilityException e) {
      throw new AssertionError(e);
    }
  }

  /** Modifies a binding in the current Frame. If it is the module Frame, also export it. */
  StarlarkThread updateAndExport(String varname, Object value) throws EvalException {
    update(varname, value);
    if (isGlobal()) {
      globalFrame.exportedBindings.add(varname);
    }
    return this;
  }

  /**
   * Modifies a binding in the current Frame of this StarlarkThread, as would an {@link
   * AssignmentStatement}. Does not try to modify an inherited binding. This will shadow any
   * inherited binding, which may be an error that you want to guard against before calling this
   * function.
   *
   * @param varname the name of the variable to be bound
   * @param value the value to bind to the variable
   * @return this StarlarkThread, in fluid style
   */
  public StarlarkThread update(String varname, Object value) throws EvalException {
    Preconditions.checkNotNull(value, "trying to assign null to '%s'", varname);
    try {
      lexicalFrame.put(this, varname, value);
    } catch (MutabilityException e) {
      // Note that since at this time we don't accept the global keyword, and don't have closures,
      // end users should never be able to mutate a frozen StarlarkThread, and a MutabilityException
      // is therefore a failed assertion for Bazel. However, it is possible to shadow a binding
      // imported from a parent StarlarkThread by updating the current StarlarkThread, which will
      // not trigger a MutabilityException.
      throw new AssertionError(
          Printer.format("Can't update %s to %r in frozen environment", varname, value), e);
    }
    return this;
  }

  /**
   * Initializes a binding in this StarlarkThread. It is an error if the variable is already bound.
   * This is not for end-users, and will throw an AssertionError in case of conflict.
   *
   * @param varname the name of the variable to be bound
   * @param value the value to bind to the variable
   * @return this StarlarkThread, in fluid style
   */
  public StarlarkThread setup(String varname, Object value) {
    if (lookup(varname) != null) {
      throw new AssertionError(String.format("variable '%s' already bound", varname));
    }
    return setupOverride(varname, value);
  }

  /**
   * Initializes a binding in this environment. Overrides any previous binding. This is not for
   * end-users, and will throw an AssertionError in case of conflict.
   *
   * @param varname the name of the variable to be bound
   * @param value the value to bind to the variable
   * @return this StarlarkThread, in fluid style
   */
  public StarlarkThread setupOverride(String varname, Object value) {
    try {
      return update(varname, value);
    } catch (EvalException ee) {
      throw new AssertionError(ee);
    }
  }

  /**
   * Returns the value of a variable defined in Local scope. Do not search in any parent scope. This
   * function should be used once the AST has been analysed and we know which variables are local.
   */
  Object localLookup(String varname) {
    return lexicalFrame.get(varname);
  }

  /**
   * Returns the value of a variable defined in the Module scope (e.g. global variables, functions).
   */
  public Object moduleLookup(String varname) {
    return globalFrame.getDirectBindings(varname);
  }

  /** Returns the value of a variable defined in the Universe scope (builtins). */
  public Object universeLookup(String varname) {
    // TODO(laurentlb): look only at globalFrame.universe.
    return globalFrame.get(varname);
  }

  /**
   * Returns the value from the environment whose name is "varname" if it exists, otherwise null.
   *
   * <p>TODO(laurentlb): Remove this method. Callers should know where the value is defined and use
   * the corresponding method (e.g. localLookup or moduleLookup).
   */
  Object lookup(String varname) {
    // Lexical frame takes precedence, then globals.
    Object lexicalValue = lexicalFrame.get(varname);
    if (lexicalValue != null) {
      return lexicalValue;
    }
    Object globalValue = globalFrame.get(varname);
    if (globalValue == null) {
      return null;
    }
    return globalValue;
  }

  /**
   * Returns a map containing all bindings that are technically <i>present</i> but are
   * <i>restricted</i> in the current frame with the current semantics. Such bindings should be
   * treated unresolvable; this method should be invoked to prepare error messaging for evaluation
   * environments where access of these restricted objects may have been attempted.
   */
  Map<String, FlagGuardedValue> getRestrictedBindings() {
    return globalFrame.restrictedBindings;
  }

  public StarlarkSemantics getSemantics() {
    return semantics;
  }

  void handleEvent(Event event) {
    eventHandler.handle(event);
  }

  /**
   * Returns a set of all names of variables that are accessible in this {@code StarlarkThread}, in
   * a deterministic order.
   */
  // TODO(adonovan): eliminate sole external call from docgen.
  public Set<String> getVariableNames() {
    LinkedHashSet<String> vars = new LinkedHashSet<>();
    vars.addAll(lexicalFrame.getTransitiveBindings().keySet());
    // No-op when globalFrame = lexicalFrame
    vars.addAll(globalFrame.getTransitiveBindings().keySet());
    return vars;
  }

  private static final class EvalEventHandler implements EventHandler {
    List<Event> messages = new ArrayList<>();

    @Override
    public void handle(Event event) {
      if (event.getKind() == EventKind.ERROR) {
        messages.add(event);
      }
    }
  }

  /** Evaluates a Skylark statement in this thread. (Debugger API) */
  // TODO(adonovan): push this up into the debugger once the eval API is finalized.
  public Object debugEval(ParserInput input) throws EvalException, InterruptedException {
    EvalEventHandler handler = new EvalEventHandler();
    Expression expr = Expression.parse(input, handler);
    if (!handler.messages.isEmpty()) {
      Event ev = handler.messages.get(0);
      throw new EvalException(ev.getLocation(), ev.getMessage());
    }
    return Eval.eval(this, expr);
  }

  /** Executes a Skylark file (sequence of statements) in this thread. (Debugger API) */
  // TODO(adonovan): push this up into the debugger once the exec API is finalized.
  public void debugExec(ParserInput input) throws EvalException, InterruptedException {
    EvalEventHandler handler = new EvalEventHandler();
    BuildFileAST file = BuildFileAST.parse(input, handler);
    if (!handler.messages.isEmpty()) {
      Event ev = handler.messages.get(0);
      throw new EvalException(ev.getLocation(), ev.getMessage());
    }
    for (Statement stmt : file.getStatements()) {
      if (stmt instanceof LoadStatement) {
        throw new EvalException(null, "cannot execute load statements in debugger");
      }
    }
    ValidationEnvironment.validateFile(file, this, /*isBuildFile=*/ false);
    Eval.execStatements(this, file.getStatements());
  }

  /**
   * Returns the stack frames corresponding of the context's current (paused) state. (Debugger API)
   *
   * <p>For all stack frames except the innermost, location information is retrieved from the
   * current context. The innermost frame's location must be supplied as {@code currentLocation} by
   * the caller.
   */
  public ImmutableList<DebugFrame> listFrames(Location currentLocation) {
    ImmutableList.Builder<DebugFrame> frameListBuilder = ImmutableList.builder();

    Continuation currentContinuation = continuation;
    Frame currentFrame = lexicalFrame;

    // if there's a continuation then the current frame is a lexical frame
    while (currentContinuation != null) {
      frameListBuilder.add(
          DebugFrame.builder()
              .setLexicalFrameBindings(ImmutableMap.copyOf(currentFrame.getTransitiveBindings()))
              .setGlobalBindings(ImmutableMap.copyOf(getGlobals().getTransitiveBindings()))
              .setFunctionName(currentContinuation.function.getFullName())
              .setLocation(currentLocation)
              .build());

      currentFrame = currentContinuation.lexicalFrame;
      currentLocation =
          currentContinuation.caller != null ? currentContinuation.caller.getLocation() : null;
      currentContinuation = currentContinuation.continuation;
    }

    frameListBuilder.add(
        DebugFrame.builder()
            .setGlobalBindings(ImmutableMap.copyOf(getGlobals().getTransitiveBindings()))
            .setFunctionName("<top level>")
            .setLocation(currentLocation)
            .build());

    return frameListBuilder.build();
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
  @Nullable
  public ReadyToPause stepControl(Stepping stepping) {
    final Continuation pausedContinuation = continuation;

    switch (stepping) {
      case NONE:
        return null;
      case INTO:
        // pause at the very next statement
        return thread -> true;
      case OVER:
        return thread -> isAt(thread, pausedContinuation) || isOutside(thread, pausedContinuation);
      case OUT:
        // if we're at the outer-most frame, same as NONE
        return pausedContinuation == null ? null : thread -> isOutside(thread, pausedContinuation);
    }
    throw new IllegalArgumentException("Unsupported stepping type: " + stepping);
  }

  /** See stepControl (Debugger API) */
  public interface ReadyToPause extends Predicate<StarlarkThread> {}

  /**
   * Describes the stepping behavior that should occur when execution of a thread is continued.
   * (Debugger API)
   */
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

  /** Returns true if {@code thread} is in a parent frame of {@code pausedContinuation}. */
  private static boolean isOutside(
      StarlarkThread thread, @Nullable Continuation pausedContinuation) {
    return pausedContinuation != null && thread.continuation == pausedContinuation.continuation;
  }

  /** Returns true if {@code thread} is at the same frame as {@code pausedContinuation. */
  private static boolean isAt(StarlarkThread thread, @Nullable Continuation pausedContinuation) {
    return thread.continuation == pausedContinuation;
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

  /**
   * An Exception thrown when an attempt is made to import a symbol from a file that was not
   * properly loaded.
   */
  // TODO(adonovan): replace with plain EvalException.
  static class LoadFailedException extends Exception {
    LoadFailedException(String importString) {
      super(
          String.format(
              "file '%s' was not correctly loaded. "
                  + "Make sure the 'load' statement appears in the global scope in your file",
              importString));
    }

    LoadFailedException(String importString, String symbolString, Iterable<String> allKeys) {
      super(
          String.format(
              "file '%s' does not contain symbol '%s'%s",
              importString, symbolString, SpellChecker.didYouMean(symbolString, allKeys)));
    }
  }

  void importSymbol(String importString, Identifier symbol, String nameInLoadedFile)
      throws LoadFailedException {
    Preconditions.checkState(isGlobal()); // loading is only allowed at global scope.

    if (!importedExtensions.containsKey(importString)) {
      throw new LoadFailedException(importString);
    }

    Extension ext = importedExtensions.get(importString);

    Map<String, Object> bindings = ext.getBindings();
    if (!bindings.containsKey(nameInLoadedFile)) {
      throw new LoadFailedException(importString, nameInLoadedFile, bindings.keySet());
    }

    Object value = bindings.get(nameInLoadedFile);

    try {
      update(symbol.getName(), value);
    } catch (EvalException e) {
      throw new LoadFailedException(importString);
    }
  }

  /**
   * Computes a deterministic hash for the given base hash code and extension map (the map's order
   * does not matter).
   */
  private static String computeTransitiveContentHashCode(
      @Nullable String baseHashCode, Map<String, Extension> importedExtensions) {
    // Calculate a new hash from the hash of the loaded Extension-s.
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

  /** A read-only {@link StarlarkThread.GlobalFrame} with False/True/None constants only. */
  @AutoCodec static final GlobalFrame CONSTANTS_ONLY = createConstantsGlobals();

  /**
   * A read-only {@link StarlarkThread.GlobalFrame} with initial globals as defined in
   * MethodLibrary.
   */
  @AutoCodec public static final GlobalFrame DEFAULT_GLOBALS = createDefaultGlobals();

  /** To be removed when all call-sites are updated. */
  public static final GlobalFrame SKYLARK = DEFAULT_GLOBALS;

  private static StarlarkThread.GlobalFrame createConstantsGlobals() {
    ImmutableMap.Builder<String, Object> builder = ImmutableMap.builder();
    Runtime.addConstantsToBuilder(builder);
    return GlobalFrame.createForBuiltins(builder.build());
  }

  private static StarlarkThread.GlobalFrame createDefaultGlobals() {
    ImmutableMap.Builder<String, Object> builder = ImmutableMap.builder();
    Runtime.addConstantsToBuilder(builder);
    MethodLibrary.addBindingsToBuilder(builder);
    return GlobalFrame.createForBuiltins(builder.build());
  }

  /** An exception thrown by {@link #FAIL_FAST_HANDLER}. */
  // TODO(bazel-team): Possibly extend RuntimeException instead of IllegalArgumentException.
  public static class FailFastException extends IllegalArgumentException {
    public FailFastException(String s) {
      super(s);
    }
  }

  /**
   * A handler that immediately throws {@link FailFastException} whenever an error or warning
   * occurs.
   *
   * <p>We do not reuse an existing unchecked exception type, because callers (e.g., test
   * assertions) need to be able to distinguish between organically occurring exceptions and
   * exceptions thrown by this handler.
   */
  public static final EventHandler FAIL_FAST_HANDLER =
      new EventHandler() {
        @Override
        public void handle(Event event) {
          if (EventKind.ERRORS_AND_WARNINGS.contains(event.getKind())) {
            throw new FailFastException(event.toString());
          }
        }
      };
}

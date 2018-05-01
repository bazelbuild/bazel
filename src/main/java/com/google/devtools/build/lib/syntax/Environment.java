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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
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
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;
import javax.annotation.Nullable;

/**
 * An {@code Environment} is the main entry point to evaluating Skylark code. It embodies all the
 * state that is required to evaluate such code, except for the current instruction pointer, which
 * is an {@link ASTNode} that is evaluated (for expressions) or executed (for statements) with
 * respect to this {@code Environment}.
 *
 * <p>{@link Continuation}-s are explicitly represented, but only partly, with another part being
 * implicit in a series of try-catch statements, to maintain the direct style. One notable trick is
 * how a {@link UserDefinedFunction} implements returning values as the function catching a {@link
 * ReturnStatement.ReturnException} thrown by a {@link ReturnStatement} in the body.
 *
 * <p>Every {@code Environment} has a {@link Mutability} field, and must be used within a function
 * that creates and closes this {@link Mutability} with the try-with-resource pattern. This {@link
 * Mutability} is also used when initializing mutable objects within that {@code Environment}. When
 * the {@code Mutability} is closed at the end of the computation, it freezes the {@code
 * Environment} along with all of those objects. This pattern enforces the discipline that there
 * should be no dangling mutable {@code Environment}, or concurrency between interacting {@code
 * Environment}s. It is a Skylark-level error to attempt to mutate a frozen {@code Environment} or
 * its objects, but it is a Java-level error to attempt to mutate an unfrozen {@code Environment} or
 * its objects from within a different {@code Environment}.
 *
 * <p>One creates an Environment using the {@link #builder} function, then populates it with {@link
 * #setup}, {@link #setupDynamic} and sometimes {@link #setupOverride}, before to evaluate code in
 * it with {@link BuildFileAST#eval}, or with {@link BuildFileAST#exec} (where the AST was obtained
 * by passing a {@link ValidationEnvironment} constructed from the Environment to {@link
 * BuildFileAST#parseBuildFile} or {@link BuildFileAST#parseSkylarkFile}). When the computation is
 * over, the frozen Environment can still be queried with {@link #lookup}.
 *
 * <p>Final fields of an Environment represent its dynamic state, i.e. state that remains the same
 * throughout a given evaluation context, and don't change with source code location, while mutable
 * fields embody its static state, that change with source code location. The seeming paradox is
 * that the words "dynamic" and "static" refer to the point of view of the source code, and here we
 * have a dual point of view.
 */
public final class Environment implements Freezable {

  /**
   * A phase for enabling or disabling certain builtin functions
   */
  public enum Phase { WORKSPACE, LOADING, ANALYSIS }

  /**
   * A mapping of bindings, either mutable or immutable according to an associated
   * {@link Mutability}. The order of the bindings within a single {@link Frame} is deterministic
   * but unspecified.
   *
   * <p>Any non-frozen {@link Frame} must have the same {@link Mutability} as the current {@link
   * Environment}, to avoid interference from other evaluation contexts. For example, a {@link
   * UserDefinedFunction} will close over the global frame of the {@link Environment} in which it
   * was defined. When the function is called from other {@link Environment}s (possibly
   * simultaneously), that global frame must already be frozen; a new local {@link Frame} is created
   * to represent the lexical scope of the function.
   *
   * <p>A {@link Frame} can have an associated "parent" {@link Frame}, which is used in {@link #get}
   * and {@link #getTransitiveBindings()}
   */
  public interface Frame extends Freezable {
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
     * @param env the {@link Environment} attempting the mutation
     * @param varname the name of the variable to be bound
     * @param value the value to bind to the variable
     */
    void put(Environment env, String varname, Object value) throws MutabilityException;

    /**
     * TODO(laurentlb): Remove this method when possible. It should probably not
     * be part of the public interface.
     */
    void remove(Environment env, String varname) throws MutabilityException;

    /**
     * Returns a map containing all bindings of this {@link Frame} and of its transitive
     * parents, taking into account shadowing precedence.
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

    static LexicalFrame createForUserDefinedFunctionCall(Mutability mutability, int numArgs) {
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
    public void put(Environment env, String varname, Object value) throws MutabilityException {
      Mutability.checkMutable(this, env.mutability());
      throw new IllegalStateException();
    }

    @Override
    public void remove(Environment env, String varname) throws MutabilityException {
      Mutability.checkMutable(this, env.mutability());
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
      this.bindings = new LinkedHashMap<>(initialCapacity);
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
    public void put(Environment env, String varname, Object value) throws MutabilityException {
      Mutability.checkMutable(this, env.mutability());
      bindings.put(varname, value);
    }

    @Override
    public void remove(Environment env, String varname) throws MutabilityException {
      Mutability.checkMutable(this, env.mutability());
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
   * A {@link Frame} that can have a parent {@link GlobalFrame} from which it inherits bindings.
   *
   * <p>Bindings in a {@link GlobalFrame} may shadow those inherited from its parents. A chain of
   * {@link GlobalFrame}s can represent different builtin scopes with a linear precedence ordering.
   *
   * <p>A {@link GlobalFrame} can also be constructed in a two-phase process. To do this, call the
   * nullary constructor to create an uninitialized {@link GlobalFrame}, then call
   * {@link #initialize}. It is illegal to use any other method in-between these two calls, or to
   * call {@link #initialize} on an already initialized {@link GlobalFrame}.
   */

  public static final class GlobalFrame implements Frame {
    /**
     * Final, except that it may be initialized after instantiation. Null mutability indicates that
     * this Frame is uninitialized.
     */
    @Nullable
    private Mutability mutability;

    /** Final, except that it may be initialized after instantiation. */
    @Nullable
    private GlobalFrame parent;

    /**
     * If this frame is a global frame, the label for the corresponding target, e.g. {@code
     * //foo:bar.bzl}.
     *
     * <p>Final, except that it may be initialized after instantiation.
     */
    @Nullable
    private Label label;

    /** Bindings are maintained in order of creation. */
    private final LinkedHashMap<String, Object> bindings;

    /** Constructs an uninitialized instance; caller must call {@link #initialize} before use. */
    public GlobalFrame() {
      this.mutability = null;
      this.parent = null;
      this.label = null;
      this.bindings = new LinkedHashMap<>();
    }

    public GlobalFrame(
        Mutability mutability,
        @Nullable GlobalFrame parent,
        @Nullable Label label,
        @Nullable Map<String, Object> bindings) {
      this.mutability = Preconditions.checkNotNull(mutability);
      this.parent = parent;
      this.label = label;
      this.bindings = new LinkedHashMap<>();
      if (bindings != null) {
        this.bindings.putAll(bindings);
      }
    }

    public GlobalFrame(Mutability mutability) {
      this(mutability, null, null, null);
    }

    public GlobalFrame(Mutability mutability, @Nullable GlobalFrame parent) {
      this(mutability, parent, null, null);
    }

    public GlobalFrame(Mutability mutability, @Nullable GlobalFrame parent, @Nullable Label label) {
      this(mutability, parent, label, null);
    }

    /** Constructs a global frame for the given builtin bindings. */
    public static GlobalFrame createForBuiltins(Map<String, Object> bindings) {
      Mutability mutability = Mutability.create("<builtins>").freeze();
      return new GlobalFrame(mutability, null, null, bindings);
    }

    private void checkInitialized() {
      Preconditions.checkNotNull(mutability, "Attempted to use Frame before initializing it");
    }

    public void initialize(
        Mutability mutability, @Nullable GlobalFrame parent,
        @Nullable Label label, Map<String, Object> bindings) {
      Preconditions.checkState(this.mutability == null,
          "Attempted to initialize an already initialized Frame");
      this.mutability = Preconditions.checkNotNull(mutability);
      this.parent = parent;
      this.label = label;
      this.bindings.putAll(bindings);
    }

    /**
     * Returns a new {@link GlobalFrame} that is a descendant of this one with {@link #label} set to
     * the given value.
     */
    public GlobalFrame withLabel(Label label) {
      checkInitialized();
      return new GlobalFrame(mutability, this, label);
    }

    /**
     * Returns the {@link Mutability} of this {@link GlobalFrame}, which may be different from its
     * parent's.
     */
    @Override
    public Mutability mutability() {
      checkInitialized();
      return mutability;
    }

    /** Returns the parent {@link GlobalFrame}, if it exists. */
    @Nullable
    public GlobalFrame getParent() {
      checkInitialized();
      return parent;
    }

    /**
     * Returns the label of this {@code Frame}, which may be null. Parent labels are not consulted.
     *
     * <p>Usually you want to use {@link #getTransitiveLabel}; this is just an accessor for
     * completeness.
     */
    @Nullable
    public Label getLabel() {
      checkInitialized();
      return label;
    }

    /**
     * Walks from this {@link GlobalFrame} up through transitive parents, and returns the first
     * non-null label found, or null if all labels are null.
     */
    @Nullable
    public Label getTransitiveLabel() {
      checkInitialized();
      if (label != null) {
        return label;
      } else if (parent != null) {
        return parent.getTransitiveLabel();
      } else {
        return null;
      }
    }

    /**
     * Returns a map of direct bindings of this {@link GlobalFrame}, ignoring parents.
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

    @Override
    public Map<String, Object> getTransitiveBindings() {
      checkInitialized();
      // Can't use ImmutableMap.Builder because it doesn't allow duplicates.
      LinkedHashMap<String, Object> collectedBindings = new LinkedHashMap<>();
      accumulateTransitiveBindings(collectedBindings);
      return collectedBindings;
    }

    private void accumulateTransitiveBindings(LinkedHashMap<String, Object> accumulator) {
      checkInitialized();
      // Put parents first, so child bindings take precedence.
      if (parent != null) {
        parent.accumulateTransitiveBindings(accumulator);
      }
      accumulator.putAll(bindings);
    }

    @Override
    public Object get(String varname) {
      checkInitialized();
      Object val = bindings.get(varname);
      if (val != null) {
        return val;
      }
      if (parent != null) {
        return parent.get(varname);
      }
      return null;
    }

    @Override
    public void put(Environment env, String varname, Object value)
        throws MutabilityException {
      checkInitialized();
      Mutability.checkMutable(this, env.mutability());
      bindings.put(varname, value);
    }

    @Override
    public void remove(Environment env, String varname) throws MutabilityException {
      checkInitialized();
      Mutability.checkMutable(this, env.mutability());
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

  /**
   * A Continuation contains data saved during a function call and restored when the function exits.
   */
  private static final class Continuation {
    /** The {@link BaseFunction} being evaluated that will return into this Continuation. */
    final BaseFunction function;

    /** The {@link FuncallExpression} to which this Continuation will return. */
    final FuncallExpression caller;

    /** The next Continuation after this Continuation. */
    @Nullable final Continuation continuation;

    /** The lexical Frame of the caller. */
    final LexicalFrame lexicalFrame;

    /** The global Frame of the caller. */
    final GlobalFrame globalFrame;

    /** The set of known global variables of the caller. */
    @Nullable final LinkedHashSet<String> knownGlobalVariables;

    Continuation(
        Continuation continuation,
        BaseFunction function,
        FuncallExpression caller,
        LexicalFrame lexicalFrame,
        GlobalFrame globalFrame,
        LinkedHashSet<String> knownGlobalVariables) {
      this.continuation = continuation;
      this.function = function;
      this.caller = caller;
      this.lexicalFrame = lexicalFrame;
      this.globalFrame = globalFrame;
      this.knownGlobalVariables = knownGlobalVariables;
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
     * Constructs using the bindings from the global definitions of the given {@link Environment},
     * and that {@code Environment}'s transitive hash code.
     */
    public Extension(Environment env) {
      this(ImmutableMap.copyOf(env.globalFrame.bindings), env.getTransitiveContentHashCode());
    }

    public String getTransitiveContentHashCode() {
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
        throw new IllegalStateException(String.format(
            "Expected an equal Extension, but got a %s instead of an Extension",
            obj == null ? "null" : obj.getClass().getName()));
      }
      Extension other = (Extension) obj;
      ImmutableMap<String, Object> otherBindings = other.getBindings();

      Set<String> names = bindings.keySet();
      Set<String> otherNames = otherBindings.keySet();
      if (!names.equals(otherNames)) {
        throw new IllegalStateException(String.format(
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
        throw new IllegalStateException(String.format(
            "Expected Extensions to be equal, but transitive content hashes don't match: %s != %s",
            transitiveContentHashCode,
            other.getTransitiveContentHashCode()));
      }
    }

    @Override
    public int hashCode() {
      return Objects.hash(bindings, transitiveContentHashCode);
    }
  }

  /**
   * Static Frame for lexical variables that are always looked up in the current Environment
   * or for the definition Environment of the function currently being evaluated.
   */
  private LexicalFrame lexicalFrame;

  /**
   * Static Frame for global variables; either the current lexical Frame if evaluation is currently
   * happening at the global scope of a BUILD file, or the global Frame at the time of function
   * definition if evaluation is currently happening in the body of a function. Thus functions can
   * close over other functions defined in the same file.
   */
  private GlobalFrame globalFrame;

  /**
   * Dynamic Frame for variables that are always looked up in the runtime Environment,
   * and never in the lexical or "global" Environment as it was at the time of function definition.
   * For instance, PACKAGE_NAME.
   */
  private final Frame dynamicFrame;

  /**
   * The semantics options that affect how Skylark code is evaluated.
   */
  private final SkylarkSemantics semantics;

  /**
   * An EventHandler for errors and warnings. This is not used in the BUILD language,
   * however it might be used in Skylark code called from the BUILD language, so shouldn't be null.
   */
  private final EventHandler eventHandler;

  /**
   * For each imported extension, a global Skylark frame from which to load() individual bindings.
   */
  private final Map<String, Extension> importedExtensions;

  /**
   * Is this Environment being executed during the loading phase? Many builtin functions are only
   * enabled during the loading phase, and check this flag.
   * TODO(laurentlb): Remove from Environment
   */
  private final Phase phase;

  /**
   * When in a lexical (Skylark) Frame, this set contains the variable names that are global,
   * as determined not by global declarations (not currently supported),
   * but by previous lookups that ended being global or dynamic.
   * This is necessary because if in a function definition something
   * reads a global variable after which a local variable with the same name is assigned an
   * Exception needs to be thrown.
   */
  @Nullable private LinkedHashSet<String> knownGlobalVariables;

  /**
   * When in a lexical (Skylark) frame, this lists the names of the functions in the call stack.
   * We currently use it to artificially disable recursion.
   */
  @Nullable private Continuation continuation;

  /**
   * Gets the label of the BUILD file that is using this environment. For example, if a target
   * //foo has a dependency on //bar which is a Skylark rule defined in //rules:my_rule.bzl being
   * evaluated in this environment, then this would return //foo.
   */
  @Nullable private final Label callerLabel;

  /**
   * Enters a scope by saving state to a new Continuation
   * @param function the function whose scope to enter
   * @param lexical the lexical frame to use
   * @param caller the source AST node for the caller
   * @param globals the global Frame that this function closes over from its definition Environment
   */
  void enterScope(
      BaseFunction function, LexicalFrame lexical, FuncallExpression caller, GlobalFrame globals) {
    continuation =
        new Continuation(
            continuation, function, caller, lexicalFrame, globalFrame, knownGlobalVariables);
    lexicalFrame = lexical;
    globalFrame = globals;
    knownGlobalVariables = new LinkedHashSet<>();
  }

  /**
   * Exits a scope by restoring state from the current continuation
   */
  void exitScope() {
    Preconditions.checkNotNull(continuation);
    lexicalFrame = continuation.lexicalFrame;
    globalFrame = continuation.globalFrame;
    knownGlobalVariables = continuation.knownGlobalVariables;
    continuation = continuation.continuation;
  }

  private final String transitiveHashCode;

  /**
   * Checks that the current Environment is in the loading or the workspace phase.
   * TODO(laurentlb): Move to SkylarkUtils
   *
   * @param symbol name of the function being only authorized thus.
   */
  public void checkLoadingOrWorkspacePhase(String symbol, Location loc) throws EvalException {
    if (phase == Phase.ANALYSIS) {
      throw new EvalException(loc, symbol + "() cannot be called during the analysis phase");
    }
  }

  /**
   * Checks that the current Environment is in the loading phase.
   * TODO(laurentlb): Move to SkylarkUtils
   *
   * @param symbol name of the function being only authorized thus.
   */
  public void checkLoadingPhase(String symbol, Location loc) throws EvalException {
    if (phase != Phase.LOADING) {
      throw new EvalException(loc, symbol + "() can only be called during the loading phase");
    }
  }

  /**
   * Is this a global Environment?
   * @return true if the current code is being executed at the top-level,
   * as opposed to inside the body of a function.
   */
  boolean isGlobal() {
    return lexicalFrame == null;
  }

  @Override
  public Mutability mutability() {
    // the mutability of the environment is that of its dynamic frame.
    return dynamicFrame.mutability();
  }

  /** Returns the current Frame, in which variable side-effects happen. */
  private Frame currentFrame() {
    return isGlobal() ? globalFrame : lexicalFrame;
  }

  /** Returns the global variables for the Environment (not including dynamic bindings). */
  public GlobalFrame getGlobals() {
    return globalFrame;
  }

  /**
   * Returns an EventHandler for errors and warnings.
   * The BUILD language doesn't use it directly, but can call Skylark code that does use it.
   * @return an EventHandler
   */
  public EventHandler getEventHandler() {
    return eventHandler;
  }

  /**
   * Returns if calling the supplied function would be a recursive call, or in other words if the
   * supplied function is already on the stack.
   */
  boolean isRecursiveCall(UserDefinedFunction function) {
    for (Continuation k = continuation; k != null; k = k.continuation) {
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

  /**
   * Returns the FuncallExpression and the BaseFunction for the top-level call being evaluated.
   */
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
   * Constructs an Environment. This is the main, most basic constructor.
   *
   * @param globalFrame a frame for the global Environment
   * @param dynamicFrame a frame for the dynamic Environment
   * @param eventHandler an EventHandler for warnings, errors, etc
   * @param importedExtensions Extension-s from which to import bindings with load()
   * @param fileContentHashCode a hash for the source file being evaluated, if any
   * @param phase the current phase
   * @param callerLabel the label this environment came from
   */
  private Environment(
      GlobalFrame globalFrame,
      LexicalFrame dynamicFrame,
      SkylarkSemantics semantics,
      EventHandler eventHandler,
      Map<String, Extension> importedExtensions,
      @Nullable String fileContentHashCode,
      Phase phase,
      @Nullable Label callerLabel) {
    this.globalFrame = Preconditions.checkNotNull(globalFrame);
    this.dynamicFrame = Preconditions.checkNotNull(dynamicFrame);
    Preconditions.checkArgument(!globalFrame.mutability().isFrozen());
    Preconditions.checkArgument(!dynamicFrame.mutability().isFrozen());
    this.semantics = semantics;
    this.eventHandler = eventHandler;
    this.importedExtensions = importedExtensions;
    this.phase = phase;
    this.callerLabel = callerLabel;
    this.transitiveHashCode =
        computeTransitiveContentHashCode(fileContentHashCode, importedExtensions);
  }

  /**
   * A Builder class for Environment.
   *
   * <p>The caller must explicitly set the semantics by calling either {@link #setSemantics} or
   * {@link #useDefaultSemantics}.
   */
  public static class Builder {
    private final Mutability mutability;
    private Phase phase = Phase.ANALYSIS;
    @Nullable private GlobalFrame parent;
    @Nullable private SkylarkSemantics semantics;
    @Nullable private EventHandler eventHandler;
    @Nullable private Map<String, Extension> importedExtensions;
    @Nullable private String fileContentHashCode;
    private Label label;

    Builder(Mutability mutability) {
      this.mutability = mutability;
    }

    /**
     * Obsolete, doesn't do anything.
     * TODO(laurentlb): To be removed once call-sites have been updated
     */
    public Builder setSkylark() {
      return this;
    }

    /** Enables loading or workspace phase only functions in this Environment. */
    public Builder setPhase(Phase phase) {
      Preconditions.checkState(this.phase == Phase.ANALYSIS);
      this.phase = phase;
      return this;
    }

    /** Inherits global bindings from the given parent Frame. */
    public Builder setGlobals(GlobalFrame parent) {
      Preconditions.checkState(this.parent == null);
      this.parent = parent;
      return this;
    }

    public Builder setSemantics(SkylarkSemantics semantics) {
      this.semantics = semantics;
      return this;
    }

    public Builder useDefaultSemantics() {
      this.semantics = SkylarkSemantics.DEFAULT_SEMANTICS;
      return this;
    }

    /** Sets an EventHandler for errors and warnings. */
    public Builder setEventHandler(EventHandler eventHandler) {
      Preconditions.checkState(this.eventHandler == null);
      this.eventHandler = eventHandler;
      return this;
    }

    /** Declares imported extensions for load() statements. */
    public Builder setImportedExtensions (Map<String, Extension> importMap) {
      Preconditions.checkState(this.importedExtensions == null);
      this.importedExtensions = importMap;
      return this;
    }

    /** Declares content hash for the source file for this Environment. */
    public Builder setFileContentHashCode(String fileContentHashCode) {
      this.fileContentHashCode = fileContentHashCode;
      return this;
    }

    /** Builds the Environment. */
    public Environment build() {
      Preconditions.checkArgument(!mutability.isFrozen());
      if (parent != null) {
        Preconditions.checkArgument(parent.mutability().isFrozen(), "parent frame must be frozen");
      }
      GlobalFrame globalFrame = new GlobalFrame(mutability, parent);
      LexicalFrame dynamicFrame = LexicalFrame.create(mutability);
      if (semantics == null) {
        throw new IllegalArgumentException("must call either setSemantics or useDefaultSemantics");
      }
      if (importedExtensions == null) {
        importedExtensions = ImmutableMap.of();
      }
      return new Environment(
          globalFrame,
          dynamicFrame,
          semantics,
          eventHandler,
          importedExtensions,
          fileContentHashCode,
          phase,
          label);
    }

    public Builder setCallerLabel(Label label) {
      this.label = label;
      return this;
    }
  }

  public static Builder builder(Mutability mutability) {
    return new Builder(mutability);
  }

  /**
   * Returns the caller's label.
   */
  public Label getCallerLabel() {
    return callerLabel;
  }

  /**
   * Sets a binding for a special dynamic variable in this Environment.
   * This is not for end-users, and will throw an AssertionError in case of conflict.
   * @param varname the name of the dynamic variable to be bound
   * @param value a value to bind to the variable
   * @return this Environment, in fluid style
   */
  public Environment setupDynamic(String varname, Object value) {
    if (dynamicFrame.get(varname) != null) {
      throw new AssertionError(
          String.format("Trying to bind dynamic variable '%s' but it is already bound",
              varname));
    }
    if (lexicalFrame != null && lexicalFrame.get(varname) != null) {
      throw new AssertionError(
          String.format("Trying to bind dynamic variable '%s' but it is already bound lexically",
              varname));
    }
    if (globalFrame.get(varname) != null) {
      throw new AssertionError(
          String.format("Trying to bind dynamic variable '%s' but it is already bound globally",
              varname));
    }
    try {
      dynamicFrame.put(this, varname, value);
    } catch (MutabilityException e) {
      // End users don't have access to setupDynamic, and it is an implementation error
      // if we encounter a mutability exception.
      throw new AssertionError(
          String.format(
              "Trying to bind dynamic variable '%s' in frozen environment %s", varname, this),
          e);
    }
    return this;
  }

  /** Remove variable from local bindings. */
  void removeLocalBinding(String varname) {
    try {
      currentFrame().remove(this, varname);
    } catch (MutabilityException e) {
      throw new AssertionError(e);
    }
  }

  /**
   * Modifies a binding in the current Frame of this Environment, as would an
   * {@link AssignmentStatement}. Does not try to modify an inherited binding.
   * This will shadow any inherited binding, which may be an error
   * that you want to guard against before calling this function.
   * @param varname the name of the variable to be bound
   * @param value the value to bind to the variable
   * @return this Environment, in fluid style
   */
  public Environment update(String varname, Object value) throws EvalException {
    Preconditions.checkNotNull(value, "update(value == null)");
    // prevents clashes between static and dynamic variables.
    if (dynamicFrame.get(varname) != null) {
      throw new EvalException(
          null, String.format("Trying to update special read-only global variable '%s'", varname));
    }
    if (isKnownGlobalVariable(varname)) {
      throw new EvalException(
          null, String.format("Trying to update read-only global variable '%s'", varname));
    }
    try {
      currentFrame().put(this, varname, Preconditions.checkNotNull(value));
    } catch (MutabilityException e) {
      // Note that since at this time we don't accept the global keyword, and don't have closures,
      // end users should never be able to mutate a frozen Environment, and a MutabilityException
      // is therefore a failed assertion for Bazel. However, it is possible to shadow a binding
      // imported from a parent Environment by updating the current Environment, which will not
      // trigger a MutabilityException.
      throw new AssertionError(
          Printer.format("Can't update %s to %r in frozen environment", varname, value),
          e);
    }
    return this;
  }

  public boolean hasVariable(String varname) {
    return lookup(varname) != null;
  }

  /**
   * Initializes a binding in this Environment. It is an error if the variable is already bound.
   * This is not for end-users, and will throw an AssertionError in case of conflict.
   * @param varname the name of the variable to be bound
   * @param value the value to bind to the variable
   * @return this Environment, in fluid style
   */
  public Environment setup(String varname, Object value) {
    if (hasVariable(varname)) {
      throw new AssertionError(String.format("variable '%s' already bound", varname));
    }
    return setupOverride(varname, value);
  }

  /**
   * Initializes a binding in this environment. Overrides any previous binding.
   * This is not for end-users, and will throw an AssertionError in case of conflict.
   * @param varname the name of the variable to be bound
   * @param value the value to bind to the variable
   * @return this Environment, in fluid style
   */
  public Environment setupOverride(String varname, Object value) {
    try {
      return update(varname, value);
    } catch (EvalException ee) {
      throw new AssertionError(ee);
    }
  }

  /**
   * Returns the value from the environment whose name is "varname" if it exists, otherwise null.
   */
  public Object lookup(String varname) {
    // Lexical frame takes precedence, then globals, then dynamics.
    if (lexicalFrame != null) {
      Object lexicalValue = lexicalFrame.get(varname);
      if (lexicalValue != null) {
        return lexicalValue;
      }
    }
    Object globalValue = globalFrame.get(varname);
    Object dynamicValue = dynamicFrame.get(varname);
    if (globalValue == null && dynamicValue == null) {
      return null;
    }
    if (knownGlobalVariables != null) {
      knownGlobalVariables.add(varname);
    }
    if (globalValue != null) {
      return globalValue;
    }
    return dynamicValue;
  }

  /**
   * Returns true if varname is a known global variable (i.e., it has been read in the context of
   * the current function).
   */
  boolean isKnownGlobalVariable(String varname) {
    return knownGlobalVariables != null && knownGlobalVariables.contains(varname);
  }

  public SkylarkSemantics getSemantics() {
    return semantics;
  }

  public void handleEvent(Event event) {
    eventHandler.handle(event);
  }

  /**
   * Returns a set of all names of variables that are accessible in this {@code Environment}, in a
   * deterministic order.
   */
  public Set<String> getVariableNames() {
    LinkedHashSet<String> vars = new LinkedHashSet<>();
    if (lexicalFrame != null) {
      vars.addAll(lexicalFrame.getTransitiveBindings().keySet());
    }
    vars.addAll(globalFrame.getTransitiveBindings().keySet());
    vars.addAll(dynamicFrame.getTransitiveBindings().keySet());
    return vars;
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
    return String.format("<Environment%s>", mutability());
  }

  /**
   * An Exception thrown when an attempt is made to import a symbol from a file
   * that was not properly loaded.
   */
  static class LoadFailedException extends Exception {
    LoadFailedException(String importString) {
      super(String.format("file '%s' was not correctly loaded. "
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
   * Returns a hash code calculated from the hash code of this Environment and the
   * transitive closure of other Environments it loads.
   */
  public String getTransitiveContentHashCode() {
    return transitiveHashCode;
  }

  /** A read-only {@link Environment.GlobalFrame} with False/True/None constants only. */
  static final GlobalFrame CONSTANTS_ONLY = createConstantsGlobals();

  /**
   * A read-only {@link Environment.GlobalFrame} with initial globals as defined in
   * MethodLibrary.
   */
  public static final GlobalFrame DEFAULT_GLOBALS = createDefaultGlobals();

  /** To be removed when all call-sites are updated. */
  public static final GlobalFrame SKYLARK = DEFAULT_GLOBALS;

  private static Environment.GlobalFrame createConstantsGlobals() {
    ImmutableMap.Builder<String, Object> builder = ImmutableMap.builder();
    Runtime.addConstantsToBuilder(builder);
    return GlobalFrame.createForBuiltins(builder.build());
  }

  private static Environment.GlobalFrame createDefaultGlobals() {
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
   * We do not reuse an existing unchecked exception type, because callers (e.g., test assertions)
   * need to be able to distinguish between organically occurring exceptions and exceptions thrown
   * by this handler.
   */
  public static final EventHandler FAIL_FAST_HANDLER = new EventHandler() {
    @Override
    public void handle(Event event) {
      if (EventKind.ERRORS_AND_WARNINGS.contains(event.getKind())) {
        throw new FailFastException(event.toString());
      }
    }
  };
}

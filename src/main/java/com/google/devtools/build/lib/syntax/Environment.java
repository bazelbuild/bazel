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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Mutability.Freezable;
import com.google.devtools.build.lib.syntax.Mutability.MutabilityException;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.SpellChecker;
import com.google.devtools.common.options.Options;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;
import javax.annotation.Nullable;

/**
 * An Environment is the main entry point to evaluating code in the BUILD language or Skylark. It
 * embodies all the state that is required to evaluate such code, except for the current instruction
 * pointer, which is an {@link ASTNode} whose {@link Statement#exec exec} or {@link Expression#eval
 * eval} method is invoked with this Environment, in a straightforward direct-style AST-walking
 * interpreter. {@link Continuation}-s are explicitly represented, but only partly, with another
 * part being implicit in a series of try-catch statements, to maintain the direct style. One
 * notable trick is how a {@link UserDefinedFunction} implements returning values as the function
 * catching a {@link ReturnStatement.ReturnException} thrown by a {@link ReturnStatement} in the
 * body.
 *
 * <p>Every Environment has a {@link Mutability} field, and must be used within a function that
 * creates and closes this {@link Mutability} with the try-with-resource pattern. This {@link
 * Mutability} is also used when initializing mutable objects within that Environment; when closed
 * at the end of the computation freezes the Environment and all those objects that then become
 * forever immutable. The pattern enforces the discipline that there should be no dangling mutable
 * Environment, or concurrency between interacting Environment-s. It is also an error to try to
 * mutate an Environment and its objects from another Environment, before the {@link Mutability} is
 * closed.
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
   * A mapping of bindings, along with a {@link Mutability} and a parent {@code Frame} from which to
   * inherit bindings. The order of the bindings within a single {@code Frame} is deterministic but
   * unspecified.
   *
   * <p>Each {@code Frame} can be thought of as either a lexical scope or a scope containing
   * predefined variables. Bindings in a {@code Frame} may shadow those inherited from its parents.
   * Thus, the chain of {@code Frame}s can represent a hierarchy of enclosing scopes, or a
   * collection of builtin modules with a linear precedence ordering.
   *
   * <p>Any non-frozen {@code Frame} must have the same {@code Mutability} as the current {@link
   * Environment}, to avoid interference from other evaluation contexts. For example, a {@link
   * UserDefinedFunction} will close over the global frame of the {@code Environment} in which it
   * was defined. When the function is called from other {@code Environment}s (possibly
   * simultaneously), that global frame must already be frozen; a new local {@code Frame} is created
   * to represent the lexical scope of the function.
   *
   * A {@code Frame} can also be constructed in a two-phase process. To do this, call the nullary
   * constructor to create an uninitialized {@code Frame}, then call {@link #initialize}. It is
   * illegal to use any other method in-between these two calls, or to call {@link #initialize} on
   * an already initialized {@code Frame}.
   */
  public static final class Frame implements Freezable {

    /**
     * Final, except that it may be initialized after instantiation. Null mutability indicates that
     * this Frame is uninitialized.
     */
    @Nullable
    private Mutability mutability;

    /** Final, except that it may be initialized after instantiation. */
    @Nullable
    private Frame parent;

    /**
     * If this frame is a global frame, the label for the corresponding target, e.g. {@code
     * //foo:bar.bzl}.
     *
     * <p>Final, except that it may be initialized after instantiation.
     */
    @Nullable
    private Label label;

    private final Map<String, Object> bindings;

    /** Constructs an uninitialized instance; caller must call {@link #initialize} before use. */
    public Frame() {
      this.mutability = null;
      this.parent = null;
      this.label = null;
      this.bindings = new LinkedHashMap<>();
    }

    public Frame(Mutability mutability, @Nullable Frame parent, @Nullable Label label) {
      this.mutability = Preconditions.checkNotNull(mutability);
      this.parent = parent;
      this.label = label;
      this.bindings = new LinkedHashMap<>();
    }

    public Frame(Mutability mutability) {
      this(mutability, null, null);
    }

    public Frame(Mutability mutability, Frame parent) {
      this(mutability, parent, null);
    }

    private void checkInitialized() {
      Preconditions.checkNotNull(mutability, "Attempted to use Frame before initializing it");
    }

    public void initialize(
        Mutability mutability, @Nullable Frame parent,
        @Nullable Label label, Map<String, Object> bindings) {
      Preconditions.checkState(this.mutability == null,
          "Attempted to initialize an already initialized Frame");
      this.mutability = Preconditions.checkNotNull(mutability);
      this.parent = parent;
      this.label = label;
      this.bindings.putAll(bindings);
    }

    /**
     * Returns a new {@code Frame} that is a copy of this one, but with {@code label} set to the
     * given value.
     */
    public Frame withLabel(Label label) {
      checkInitialized();
      return new Frame(mutability, this, label);
    }

    /**
     * Returns the {@link Mutability} of this {@code Frame}, which may be different from its
     * parent's.
     */
    @Override
    public Mutability mutability() {
      checkInitialized();
      return mutability;
    }

    /** Returns the parent {@code Frame}, if it exists. */
    @Nullable
    public Frame getParent() {
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
     * Walks from this {@code Frame} up through transitive parents, and returns the first non-null
     * label found, or null if all labels are null.
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
     * Returns a map of direct bindings of this {@code Frame}, ignoring parents.
     *
     * <p>For efficiency an unmodifiable view is returned. Callers should assume that the view is
     * invalidated by any subsequent modification to the {@code Frame}'s bindings.
     */
    public Map<String, Object> getBindings() {
      checkInitialized();
      return Collections.unmodifiableMap(bindings);
    }

    /**
     * Returns a map containing all bindings of this {@code Frame} and of its transitive parents,
     * taking into account shadowing precedence.
     */
    public Map<String, Object> getTransitiveBindings() {
      checkInitialized();
      // Can't use ImmutableMap.Builder because it doesn't allow duplicates.
      HashMap<String, Object> collectedBindings = new HashMap<>();
      accumulateTransitiveBindings(collectedBindings);
      return collectedBindings;
    }

    private void accumulateTransitiveBindings(Map<String, Object> accumulator) {
      checkInitialized();
      // Put parents first, so child bindings take precedence.
      if (parent != null) {
        parent.accumulateTransitiveBindings(accumulator);
      }
      accumulator.putAll(bindings);
    }

    /**
     * Gets a binding from the current {@code Frame} or one of its transitive parents.
     *
     * <p>In case of conflicts, the binding found in the {@code Frame} closest to the current one is
     * used; the remaining bindings are shadowed.
     *
     * @param varname the name of the variable to be bound
     * @return the value bound to the variable, or null if no binding is found
     */
    public Object get(String varname) {
      checkInitialized();
      if (bindings.containsKey(varname)) {
        return bindings.get(varname);
      }
      if (parent != null) {
        return parent.get(varname);
      }
      return null;
    }

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
    public void put(Environment env, String varname, Object value)
        throws MutabilityException {
      checkInitialized();
      Mutability.checkMutable(this, env.mutability());
      bindings.put(varname, value);
    }

    /**
     * TODO(laurentlb): Remove this method when possible. It should probably not
     * be part of the public interface.
     */
    void remove(Environment env, String varname) throws MutabilityException {
      checkInitialized();
      Mutability.checkMutable(this, env.mutability());
      bindings.remove(varname);
    }

    @Override
    public String toString() {
      if (mutability == null) {
        return "<Uninitialized Frame>";
      } else {
        return String.format("<Frame%s>", mutability());
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
    final Frame lexicalFrame;

    /** The global Frame of the caller. */
    final Frame globalFrame;

    /** The set of known global variables of the caller. */
    @Nullable final Set<String> knownGlobalVariables;

    Continuation(
        Continuation continuation,
        BaseFunction function,
        FuncallExpression caller,
        Frame lexicalFrame,
        Frame globalFrame,
        Set<String> knownGlobalVariables) {
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
  public static final class Extension {

    private final ImmutableMap<String, Object> bindings;

    /**
     * Cached hash code for the transitive content of this {@code Extension} and its dependencies.
     */
    private final String transitiveContentHashCode;

    /** Constructs with the given hash code and bindings. */
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

    @Override
    public int hashCode() {
      return Objects.hash(bindings, transitiveContentHashCode);
    }
  }

  /**
   * Static Frame for lexical variables that are always looked up in the current Environment
   * or for the definition Environment of the function currently being evaluated.
   */
  private Frame lexicalFrame;

  /**
   * Static Frame for global variables; either the current lexical Frame if evaluation is currently
   * happening at the global scope of a BUILD file, or the global Frame at the time of function
   * definition if evaluation is currently happening in the body of a function. Thus functions can
   * close over other functions defined in the same file.
   */
  private Frame globalFrame;

  /**
   * Dynamic Frame for variables that are always looked up in the runtime Environment,
   * and never in the lexical or "global" Environment as it was at the time of function definition.
   * For instance, PACKAGE_NAME.
   */
  private final Frame dynamicFrame;

  /**
   * The semantics options that affect how Skylark code is evaluated.
   */
  private final SkylarkSemanticsOptions semantics;

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
  @Nullable private Set<String> knownGlobalVariables;

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
   * @param caller the source AST node for the caller
   * @param globals the global Frame that this function closes over from its definition Environment
   */
  void enterScope(BaseFunction function, FuncallExpression caller, Frame globals) {
    continuation =
        new Continuation(
            continuation, function, caller, lexicalFrame, globalFrame, knownGlobalVariables);
    // TODO(bazel-team): What if instead of tracking both the lexical and global frames from the
    // Environment, we instead just tracked the current lexical frame, and made the global frame its
    // parent?
    lexicalFrame = new Frame(mutability(), null);
    globalFrame = globals;
    knownGlobalVariables = new HashSet<>();
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

  /** @return the current Frame, in which variable side-effects happen. */
  private Frame currentFrame() {
    return isGlobal() ? globalFrame : lexicalFrame;
  }

  /**
   * @return the global variables for the Environment (not including dynamic bindings).
   */
  public Frame getGlobals() {
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

  /** @return the current stack trace as a list of functions. */
  ImmutableList<BaseFunction> getStackTrace() {
    ImmutableList.Builder<BaseFunction> builder = new ImmutableList.Builder<>();
    for (Continuation k = continuation; k != null; k = k.continuation) {
      builder.add(k.function);
    }
    return builder.build().reverse();
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
      Frame globalFrame,
      Frame dynamicFrame,
      SkylarkSemanticsOptions semantics,
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
   * A Builder class for Environment
   */
  public static class Builder {
    private final Mutability mutability;
    private Phase phase = Phase.ANALYSIS;
    @Nullable private Frame parent;
    @Nullable private SkylarkSemanticsOptions semantics;
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
    public Builder setGlobals(Frame parent) {
      Preconditions.checkState(this.parent == null);
      this.parent = parent;
      return this;
    }

    public Builder setSemantics(SkylarkSemanticsOptions semantics) {
      this.semantics = semantics;
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
        Preconditions.checkArgument(parent.mutability().isFrozen());
      }
      Frame globalFrame = new Frame(mutability, parent);
      Frame dynamicFrame = new Frame(mutability, null);
      if (semantics == null) {
        semantics = Options.getDefaults(SkylarkSemanticsOptions.class);
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
   * @return the value from the environment whose name is "varname" if it exists, otherwise null.
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
   * @return true if varname is a known global variable,
   * because it has been read in the context of the current function.
   */
  boolean isKnownGlobalVariable(String varname) {
    return knownGlobalVariables != null && knownGlobalVariables.contains(varname);
  }

  public SkylarkSemanticsOptions getSemantics() {
    return semantics;
  }

  public void handleEvent(Event event) {
    eventHandler.handle(event);
  }

  /** Returns a set of all names of variables that are accessible in this {@code Environment}. */
  public Set<String> getVariableNames() {
    Set<String> vars = new HashSet<>();
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

  /** A read-only Environment.Frame with global constants in it only */
  static final Frame CONSTANTS_ONLY = createConstantsGlobals();

  /** A read-only Environment.Frame with initial globals */
  public static final Frame DEFAULT_GLOBALS = createDefaultGlobals();

  /** To be removed when all call-sites are updated. */
  public static final Frame SKYLARK = DEFAULT_GLOBALS;

  private static Environment.Frame createConstantsGlobals() {
    try (Mutability mutability = Mutability.create("CONSTANTS")) {
      Environment env = Environment.builder(mutability).build();
      Runtime.setupConstants(env);
      return env.getGlobals();
    }
  }

  private static Environment.Frame createDefaultGlobals() {
    try (Mutability mutability = Mutability.create("BUILD")) {
      Environment env = Environment.builder(mutability).build();
      Runtime.setupConstants(env);
      Runtime.setupMethodEnvironment(env, MethodLibrary.defaultGlobalFunctions);
      return env.getGlobals();
    }
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

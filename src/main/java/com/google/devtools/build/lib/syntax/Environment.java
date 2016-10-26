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

import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Mutability.Freezable;
import com.google.devtools.build.lib.syntax.Mutability.MutabilityException;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import javax.annotation.Nullable;

/**
 * An Environment is the main entry point to evaluating code in the BUILD language or Skylark.
 * It embodies all the state that is required to evaluate such code,
 * except for the current instruction pointer, which is an {@link ASTNode}
 * whose {@link Statement#exec exec} or {@link Expression#eval eval} method is invoked with
 * this Environment, in a straightforward direct-style AST-walking interpreter.
 * {@link Continuation}-s are explicitly represented, but only partly, with another part being
 * implicit in a series of try-catch statements, to maintain the direct style. One notable trick
 * is how a {@link UserDefinedFunction} implements returning values as the function catching a
 * {@link ReturnStatement.ReturnException} thrown by a {@link ReturnStatement} in the body.
 *
 * <p>Every Environment has a {@link Mutability} field, and must be used within a function that
 * creates and closes this {@link Mutability} with the try-with-resource pattern.
 * This {@link Mutability} is also used when initializing mutable objects within that Environment;
 * when closed at the end of the computation freezes the Environment and all those objects that
 * then become forever immutable. The pattern enforces the discipline that there should be no
 * dangling mutable Environment, or concurrency between interacting Environment-s.
 * It is also an error to try to mutate an Environment and its objects from another Environment,
 * before the {@link Mutability} is closed.
 *
 * <p>One creates an Environment using the {@link #builder} function, then
 * populates it with {@link #setup}, {@link #setupDynamic} and sometimes {@link #setupOverride},
 * before to evaluate code in it with {@link #eval}, or with {@link BuildFileAST#exec}
 * (where the AST was obtained by passing a {@link ValidationEnvironment} constructed from the
 * Environment to {@link BuildFileAST#parseBuildFile} or {@link BuildFileAST#parseSkylarkFile}).
 * When the computation is over, the frozen Environment can still be queried with {@link #lookup}.
 *
 * <p>Final fields of an Environment represent its dynamic state, i.e. state that remains the same
 * throughout a given evaluation context, and don't change with source code location,
 * while mutable fields embody its static state, that change with source code location.
 * The seeming paradox is that the words "dynamic" and "static" refer to the point of view
 * of the source code, and here we have a dual point of view.
 */
public final class Environment implements Freezable {

  /**
   * A phase for enabling or disabling certain builtin functions
   */
  public enum Phase { WORKSPACE, LOADING, ANALYSIS }

  /**
   * A Frame is a Map of bindings, plus a {@link Mutability} and a parent Frame
   * from which to inherit bindings.
   *
   * <p>A Frame contains bindings mapping variable name to variable value in a given scope.
   * It may also inherit bindings from a parent Frame corresponding to a parent scope,
   * which in turn may inherit bindings from its own parent, etc., transitively.
   * Bindings may shadow bindings from the parent. In Skylark, you may only mutate
   * bindings from the current Frame, which always got its {@link Mutability} with the
   * current {@link Environment}; but future extensions may make it more like Python
   * and allow mutation of bindings in outer Frame-s (or then again may not).
   *
   * <p>A Frame inherits the {@link Mutability} from the {@link Environment} in which it was
   * originally created. When that {@link Environment} is finalized and its {@link Mutability}
   * is closed, it becomes immutable, including the Frame, which can be shared in other
   * {@link Environment}-s. Indeed, a {@link UserDefinedFunction} will close over the global
   * Frame of its definition {@link Environment}, which will thus be reused (immutably)
   * in all any {@link Environment} in which this function is called, so it's important to
   * preserve the {@link Mutability} to make sure no Frame is modified after it's been finalized.
   */
  public static final class Frame implements Freezable {

    private final Mutability mutability;
    final Frame parent;
    final Map<String, Object> bindings = new HashMap<>();
    // The label for the target this frame is defined in (e.g., //foo:bar.bzl).
    @Nullable
    private Label label;

    private Frame(Mutability mutability, Frame parent) {
      this.mutability = mutability;
      this.parent = parent;
      this.label = parent == null ? null : parent.label;
    }

    @Override
    public final Mutability mutability() {
      return mutability;
    }

    /**
     * Attaches a label to an existing frame. This is used to get the repository a Skylark
     * extension is actually defined in.
     * @param label the label to attach.
     * @return a new Frame with the existing frame's properties plus the label.
     */
    public Frame setLabel(Label label) {
      Frame result = new Frame(mutability, this);
      result.label = label;
      return result;
    }

    /**
     * Returns the label for this frame.
     */
    @Nullable
    public final Label label() {
      return label;
    }

    /**
     * Gets a binding from the current frame or if not found its parent.
     * @param varname the name of the variable to be bound
     * @return the value bound to variable
     */
    public Object get(String varname) {
      if (bindings.containsKey(varname)) {
        return bindings.get(varname);
      }
      if (parent != null) {
        return parent.get(varname);
      }
      return null;
    }

    /**
     * Modifies a binding in the current Frame.
     * Does not try to modify an inherited binding.
     * This will shadow any inherited binding, which may be an error
     * that you want to guard against before calling this function.
     * @param env the Environment attempting the mutation
     * @param varname the name of the variable to be bound
     * @param value the value to bind to the variable
     */
    public void put(Environment env, String varname, Object value)
        throws MutabilityException {
      Mutability.checkMutable(this, env);
      bindings.put(varname, value);
    }

    /**
     * Adds the variable names of this Frame and its transitive parents to the given set.
     * This provides a O(n) way of extracting the list of all variables visible in an Environment.
     * @param vars the set of visible variables in the Environment, being computed.
     */
    void addVariableNamesTo(Set<String> vars) {
      vars.addAll(bindings.keySet());
      if (parent != null) {
        parent.addVariableNamesTo(vars);
      }
    }

    public Set<String> getDirectVariableNames() {
      return bindings.keySet();
    }

    @Override
    public String toString() {
      return String.format("<Frame%s>", mutability());
    }
  }

  /**
   * A Continuation contains data saved during a function call and restored when the function exits.
   */
  private static final class Continuation {
    /** The {@link BaseFunction} being evaluated that will return into this Continuation. */
    BaseFunction function;

    /** The {@link FuncallExpression} to which this Continuation will return. */
    FuncallExpression caller;

    /** The next Continuation after this Continuation. */
    @Nullable Continuation continuation;

    /** The lexical Frame of the caller. */
    Frame lexicalFrame;

    /** The global Frame of the caller. */
    Frame globalFrame;

    /** The set of known global variables of the caller. */
    @Nullable Set<String> knownGlobalVariables;

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

  // TODO(bazel-team): Fix this scary failure of serializability.
  // skyframe.SkylarkImportLookupFunction processes a .bzl and returns an Extension,
  // for use by whoever imports the .bzl file. Skyframe may subsequently serialize the results.
  // And it will fail to process these bindings, because they are inherited from a non-serializable
  // class (in previous versions of the code the serializable SkylarkEnvironment was inheriting
  // from the non-serializable Environment and being returned by said Function).
  // If we try to merge this otherwise redundant superclass into Extension, though,
  // skyframe experiences a massive failure to serialize things, and it's unclear how far
  // reaching the need to make things Serializable goes, though clearly we'll need to make
  // a whole lot of things Serializable, and for efficiency, we'll want to implement sharing
  // of imported values rather than a code explosion.
  private static class BaseExtension {
    final ImmutableMap<String, Object> bindings;
    BaseExtension(Environment env) {
      this.bindings = ImmutableMap.copyOf(env.globalFrame.bindings);
    }

    // Hack to allow serialization.
    BaseExtension() {
      this.bindings = ImmutableMap.of();
    }
  }

  /**
   * An Extension to be imported with load() into a BUILD or .bzl file.
   */
  public static final class Extension extends BaseExtension implements Serializable {

    private final String transitiveContentHashCode;

    /**
     * Constructs an Extension by extracting the new global definitions from an Environment.
     * Also caches a hash code for the transitive content of the file and its dependencies.
     * @param env the Environment from which to extract an Extension.
     */
    public Extension(Environment env) {
      super(env);
      this.transitiveContentHashCode = env.getTransitiveContentHashCode();
    }

    String getTransitiveContentHashCode() {
      return transitiveContentHashCode;
    }

    /** get the value bound to a variable in this Extension */
    public Object get(String varname) {
      return bindings.get(varname);
    }

    /** does this Extension contain a binding for the named variable? */
    public boolean containsKey(String varname) {
      return bindings.containsKey(varname);
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
   * An EventHandler for errors and warnings. This is not used in the BUILD language,
   * however it might be used in Skylark code called from the BUILD language, so shouldn't be null.
   */
  private final EventHandler eventHandler;

  /**
   * For each imported extension, a global Skylark frame from which to load() individual bindings.
   */
  private final Map<String, Extension> importedExtensions;

  /**
   * Is this Environment being executed in Skylark context?
   * TODO(laurentlb): Remove from Environment
   */
  private boolean isSkylark;

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
   * The path to the tools repository.
   * TODO(laurentlb): Remove from Environment
   */
  private final String toolsRepository;

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
   * Is this Environment being evaluated during the loading phase?
   * This is fixed during Environment setup, and enables various functions
   * that are not available during the analysis or workspace phase.
   */
  public Phase getPhase() {
    return phase;
  }

  /**
   * Checks that the current Environment is in the loading or the workspace phase.
   * @param symbol name of the function being only authorized thus.
   */
  public void checkLoadingOrWorkspacePhase(String symbol, Location loc) throws EvalException {
    if (phase == Phase.ANALYSIS) {
      throw new EvalException(loc, symbol + "() cannot be called during the analysis phase");
    }
  }

  /**
   * Checks that the current Environment is in the loading phase.
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
   * Constructs an Environment.
   * This is the main, most basic constructor.
   * @param globalFrame a frame for the global Environment
   * @param dynamicFrame a frame for the dynamic Environment
   * @param eventHandler an EventHandler for warnings, errors, etc
   * @param importedExtensions Extension-s from which to import bindings with load()
   * @param isSkylark true if in Skylark context
   * @param fileContentHashCode a hash for the source file being evaluated, if any
   * @param phase the current phase
   * @param callerLabel the label this environment came from
   */
  private Environment(
      Frame globalFrame,
      Frame dynamicFrame,
      EventHandler eventHandler,
      Map<String, Extension> importedExtensions,
      boolean isSkylark,
      @Nullable String fileContentHashCode,
      Phase phase,
      @Nullable Label callerLabel,
      String toolsRepository) {
    this.globalFrame = Preconditions.checkNotNull(globalFrame);
    this.dynamicFrame = Preconditions.checkNotNull(dynamicFrame);
    Preconditions.checkArgument(globalFrame.mutability().isMutable());
    Preconditions.checkArgument(dynamicFrame.mutability().isMutable());
    this.eventHandler = eventHandler;
    this.importedExtensions = importedExtensions;
    this.isSkylark = isSkylark;
    this.phase = phase;
    this.callerLabel = callerLabel;
    this.toolsRepository = toolsRepository;
    this.transitiveHashCode =
        computeTransitiveContentHashCode(fileContentHashCode, importedExtensions);
  }

  /**
   * A Builder class for Environment
   */
  public static class Builder {
    private final Mutability mutability;
    private boolean isSkylark = false;
    private Phase phase = Phase.ANALYSIS;
    @Nullable private Frame parent;
    @Nullable private EventHandler eventHandler;
    @Nullable private Map<String, Extension> importedExtensions;
    @Nullable private String fileContentHashCode;
    private Label label;
    private String toolsRepository;

    Builder(Mutability mutability) {
      this.mutability = mutability;
    }

    /** Enables Skylark for code read in this Environment. */
    public Builder setSkylark() {
      Preconditions.checkState(!isSkylark);
      isSkylark = true;
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

    /** Sets the path to the tools repository */
    public Builder setToolsRepository(String toolsRepository) {
      this.toolsRepository = toolsRepository;
      return this;
    }

    /** Builds the Environment. */
    public Environment build() {
      Preconditions.checkArgument(mutability.isMutable());
      if (parent != null) {
        Preconditions.checkArgument(!parent.mutability().isMutable());
      }
      Frame globalFrame = new Frame(mutability, parent);
      Frame dynamicFrame = new Frame(mutability, null);
      if (importedExtensions == null) {
        importedExtensions = ImmutableMap.of();
      }
      if (phase == Phase.LOADING) {
        Preconditions.checkState(this.toolsRepository != null);
      }
      return new Environment(
          globalFrame,
          dynamicFrame,
          eventHandler,
          importedExtensions,
          isSkylark,
          fileContentHashCode,
          phase,
          label,
          toolsRepository);
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
          Printer.format(
              "Trying to bind dynamic variable '%s' in frozen environment %r", varname, this),
          e);
    }
    return this;
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
    // Which Frame to lookup first doesn't matter because update prevents clashes.
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

  public void handleEvent(Event event) {
    eventHandler.handle(event);
  }

  /**
   * @return the (immutable) set of names of all variables defined in this
   * Environment. Exposed for testing.
   */
  @VisibleForTesting
  public Set<String> getVariableNames() {
    Set<String> vars = new HashSet<>();
    if (lexicalFrame != null) {
      lexicalFrame.addVariableNamesTo(vars);
    }
    globalFrame.addVariableNamesTo(vars);
    dynamicFrame.addVariableNamesTo(vars);
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

    LoadFailedException(String importString, String symbolString) {
      super(String.format("file '%s' does not contain symbol '%s'", importString, symbolString));
    }
  }

  void importSymbol(String importString, Identifier symbol, String nameInLoadedFile)
      throws LoadFailedException {
    Preconditions.checkState(isGlobal()); // loading is only allowed at global scope.

    if (!importedExtensions.containsKey(importString)) {
      throw new LoadFailedException(importString);
    }

    Extension ext = importedExtensions.get(importString);

    if (!ext.containsKey(nameInLoadedFile)) {
      throw new LoadFailedException(importString, nameInLoadedFile);
    }

    Object value = ext.get(nameInLoadedFile);

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


  /**
   * The fail fast handler, which throws a AssertionError whenever an error or warning occurs.
   */
  public static final EventHandler FAIL_FAST_HANDLER = new EventHandler() {
      @Override
      public void handle(Event event) {
        Preconditions.checkArgument(
            !EventKind.ERRORS_AND_WARNINGS.contains(event.getKind()), event);
      }
    };

  /**
   * Parses some String inputLines without a supporting file, returning statements only.
   * TODO(laurentlb): Remove from Environment
   * @param inputLines a list of lines of code
   */
  @VisibleForTesting
  public List<Statement> parseFile(String... inputLines) {
    ParserInputSource input = ParserInputSource.create(Joiner.on("\n").join(inputLines), null);
    List<Statement> statements;
    if (isSkylark) {
      Parser.ParseResult result = Parser.parseFileForSkylark(input, eventHandler);
      ValidationEnvironment valid = new ValidationEnvironment(this);
      valid.validateAst(result.statements, eventHandler);
      statements = result.statements;
    } else {
      statements = Parser.parseFile(input, eventHandler).statements;
    }
    // Force the validation of imports
    BuildFileAST.fetchLoads(statements, eventHandler);
    return statements;
  }

  /**
   * Evaluates code some String input without a supporting file.
   * TODO(laurentlb): Remove from Environment
   * @param input a list of lines of code to evaluate
   * @return the value of the last statement if it's an Expression or else null
   */
  @Nullable public Object eval(String... input) throws EvalException, InterruptedException {
    BuildFileAST ast;
    if (isSkylark) {
      ast = BuildFileAST.parseSkylarkString(eventHandler, input);
      ValidationEnvironment valid = new ValidationEnvironment(this);
      valid.validateAst(ast.getStatements(), eventHandler);
    } else {
      ast = BuildFileAST.parseBuildString(eventHandler, input);
    }
    return ast.eval(this);
  }

  public String getToolsRepository() {
    checkState(toolsRepository != null);
    return toolsRepository;
  }
}

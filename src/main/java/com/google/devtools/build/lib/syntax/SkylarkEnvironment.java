// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Fingerprint;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * The environment for Skylark.
 */
public class SkylarkEnvironment extends Environment implements Serializable {

  /**
   * This set contains the variable names of all the successful lookups from the global
   * environment. This is necessary because if in a function definition something
   * reads a global variable after which a local variable with the same name is assigned an
   * Exception needs to be thrown.
   */
  private final Set<String> readGlobalVariables = new HashSet<>();

  private ImmutableList<BaseFunction> stackTrace;

  @Nullable private String fileContentHashCode;

  /**
   * Creates a Skylark Environment for function calling, from the global Environment of the
   * caller Environment (which must be a Skylark Environment).
   */
  public static SkylarkEnvironment createEnvironmentForFunctionCalling(
      Environment callerEnv, SkylarkEnvironment definitionEnv,
      UserDefinedFunction function) throws EvalException {
    if (callerEnv.getStackTrace().contains(function)) {
      throw new EvalException(
          function.getLocation(),
          "Recursion was detected when calling '" + function.getName() + "' from '"
          + Iterables.getLast(callerEnv.getStackTrace()).getName() + "'");
    }
    ImmutableList<BaseFunction> stackTrace = new ImmutableList.Builder<BaseFunction>()
        .addAll(callerEnv.getStackTrace())
        .add(function)
        .build();
    SkylarkEnvironment childEnv =
        // Always use the caller Environment's EventHandler. We cannot assume that the
        // definition Environment's EventHandler is still working properly.
        new SkylarkEnvironment(definitionEnv, stackTrace, callerEnv.eventHandler);
    if (callerEnv.isLoadingPhase()) {
      childEnv.setLoadingPhase();
    }

    try {
      for (String varname : callerEnv.propagatingVariables) {
        childEnv.updateAndPropagate(varname, callerEnv.lookup(varname));
      }
    } catch (NoSuchVariableException e) {
      // This should never happen.
      throw new IllegalStateException(e);
    }
    return childEnv;
  }

  private SkylarkEnvironment(SkylarkEnvironment definitionEnv,
      ImmutableList<BaseFunction> stackTrace, EventHandler eventHandler) {
    super(definitionEnv.getGlobalEnvironment());
    this.stackTrace = stackTrace;
    this.eventHandler = Preconditions.checkNotNull(eventHandler,
        "EventHandler cannot be null in an Environment which calls into Skylark");
  }

  /**
   * Creates a global SkylarkEnvironment.
   */
  public SkylarkEnvironment(EventHandler eventHandler, String astFileContentHashCode) {
    super();
    stackTrace = ImmutableList.of();
    this.eventHandler = eventHandler;
    this.fileContentHashCode = astFileContentHashCode;
  }

  @VisibleForTesting
  public SkylarkEnvironment(EventHandler eventHandler) {
    this(eventHandler, null);
  }

  public SkylarkEnvironment(SkylarkEnvironment globalEnv) {
    super(globalEnv);
    stackTrace = ImmutableList.of();
    this.eventHandler = globalEnv.eventHandler;
  }

  @Override
  public ImmutableList<BaseFunction> getStackTrace() {
    return stackTrace;
  }

  /**
   * Clones this Skylark global environment.
   */
  public SkylarkEnvironment cloneEnv(EventHandler eventHandler) {
    Preconditions.checkArgument(isGlobal());
    SkylarkEnvironment newEnv = new SkylarkEnvironment(eventHandler, this.fileContentHashCode);
    for (Entry<String, Object> entry : env.entrySet()) {
      newEnv.env.put(entry.getKey(), entry.getValue());
    }
    return newEnv;
  }

  /**
   * Returns the global environment. Only works for Skylark environments. For the global Skylark
   * environment this method returns this Environment.
   */
  public SkylarkEnvironment getGlobalEnvironment() {
    // If there's a parent that's the global environment, otherwise this is.
    return parent != null ? (SkylarkEnvironment) parent : this;
  }

  /**
   * Returns true if this is a Skylark global environment.
   */
  @Override
  public boolean isGlobal() {
    return parent == null;
  }

  /**
   * Returns true if varname has been read as a global variable.
   */
  public boolean hasBeenReadGlobalVariable(String varname) {
    return readGlobalVariables.contains(varname);
  }

  @Override
  public boolean isSkylark() {
    return true;
  }

  /**
   * @return the value from the environment whose name is "varname".
   * @throws NoSuchVariableException if the variable is not defined in the environment.
   */
  @Override
  public Object lookup(String varname) throws NoSuchVariableException {
    Object value = env.get(varname);
    if (value == null) {
      if (parent != null && parent.hasVariable(varname)) {
        readGlobalVariables.add(varname);
        return parent.lookup(varname);
      }
      throw new NoSuchVariableException(varname);
    }
    return value;
  }

  /**
   * Like <code>lookup(String)</code>, but instead of throwing an exception in
   * the case where "varname" is not defined, "defaultValue" is returned instead.
   */
  @Override
  public Object lookup(String varname, Object defaultValue) {
    throw new UnsupportedOperationException();
  }

  /**
   * Returns the class of the variable or null if the variable does not exist. This function
   * works only in the local Environment, it doesn't check the global Environment.
   */
  public Class<?> getVariableType(String varname) {
    Object variable = env.get(varname);
    return variable != null ? EvalUtils.getSkylarkType(variable.getClass()) : null;
  }

  public void handleEvent(Event event) {
    eventHandler.handle(event);
  }

  /**
   * Returns a hash code calculated from the hash code of this Environment and the
   * transitive closure of other Environments it loads.
   */
  public String getTransitiveFileContentHashCode() {
    Fingerprint fingerprint = new Fingerprint();
    fingerprint.addString(Preconditions.checkNotNull(fileContentHashCode));
    // Calculate a new hash from the hash of the loaded Environments.
    for (SkylarkEnvironment env : importedExtensions.values()) {
      fingerprint.addString(env.getTransitiveFileContentHashCode());
    }
    return fingerprint.hexDigestAndReset();
  }
}

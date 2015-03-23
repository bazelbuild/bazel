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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.EvalException.EvalExceptionWithJavaCause;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ExecutionException;

/**
 * A function class for Skylark built in functions. Supports mandatory and optional arguments.
 * All usable arguments have to be specified. In case of ambiguous arguments (a parameter is
 * specified as positional and keyword arguments in the function call) an exception is thrown.
 */
public abstract class SkylarkFunction extends AbstractFunction {

  private ImmutableList<String> parameters;
  private ImmutableMap<String, SkylarkBuiltin.Param> parameterTypes;
  private int mandatoryParamNum;
  private boolean configured = false;
  private Class<?> objectType;
  private boolean onlyLoadingPhase;

  /**
   * Creates a SkylarkFunction with the given name.
   */
  public SkylarkFunction(String name) {
    super(name);
  }

  /**
   * Configures the parameter of this Skylark function using the annotation.
   */
  public void configure(SkylarkBuiltin annotation) {
    Preconditions.checkState(!configured);
    Preconditions.checkArgument(
        getName().equals(annotation.name()), "%s != %s", getName(), annotation.name());
    mandatoryParamNum = 0;
    ImmutableList.Builder<String> paramListBuilder = ImmutableList.builder();
    ImmutableMap.Builder<String, SkylarkBuiltin.Param> paramTypeBuilder = ImmutableMap.builder();
    for (SkylarkBuiltin.Param param : annotation.mandatoryParams()) {
      paramListBuilder.add(param.name());
      paramTypeBuilder.put(param.name(), param);
      mandatoryParamNum++;
    }
    for (SkylarkBuiltin.Param param : annotation.optionalParams()) {
      paramListBuilder.add(param.name());
      paramTypeBuilder.put(param.name(), param);
    }
    parameters = paramListBuilder.build();
    parameterTypes = paramTypeBuilder.build();
    this.objectType = annotation.objectType().equals(Object.class) ? null : annotation.objectType();
    this.onlyLoadingPhase = annotation.onlyLoadingPhase();
    configured = true;
  }

  /**
   * Returns true if the SkylarkFunction is configured.
   */
  public boolean isConfigured() {
    return configured;
  }

  @Override
  public Class<?> getObjectType() {
    return objectType;
  }

  public boolean isOnlyLoadingPhase() {
    return onlyLoadingPhase;
  }

  @Override
  public Object call(List<Object> args,
                     Map<String, Object> kwargs,
                     FuncallExpression ast,
                     Environment env)
      throws EvalException, InterruptedException {
    Preconditions.checkState(configured, "Function %s was not configured", getName());
    try {
      ImmutableMap.Builder<String, Object> arguments = new ImmutableMap.Builder<>();
      if (objectType != null && !FuncallExpression.isNamespace(objectType)) {
        args = new ArrayList<>(args); // args immutable, get a mutable copy.
        arguments.put("self", args.remove(0));
      }

      int maxParamNum = parameters.size();
      int paramNum = args.size() + kwargs.size();

      if (paramNum < mandatoryParamNum) {
        throw new EvalException(ast.getLocation(),
            String.format("incorrect number of arguments (got %s, expected at least %s)",
                paramNum, mandatoryParamNum));
      } else if (paramNum > maxParamNum) {
        throw new EvalException(ast.getLocation(),
            String.format("incorrect number of arguments (got %s, expected at most %s)",
                paramNum, maxParamNum));
      }

      for (int i = 0; i < mandatoryParamNum; i++) {
        Preconditions.checkState(i < args.size() || kwargs.containsKey(parameters.get(i)),
            String.format("missing mandatory parameter: %s", parameters.get(i)));
      }

      for (int i = 0; i < args.size(); i++) {
        checkTypeAndAddArg(parameters.get(i), args.get(i), arguments, ast.getLocation());
      }

      for (Entry<String, Object> kwarg : kwargs.entrySet()) {
        int idx = parameters.indexOf(kwarg.getKey());
        if (idx < 0) {
          throw new EvalException(ast.getLocation(),
              String.format("unknown keyword argument: %s", kwarg.getKey()));
        }
        if (idx < args.size()) {
          throw new EvalException(ast.getLocation(),
              String.format("ambiguous argument: %s", kwarg.getKey()));
        }
        checkTypeAndAddArg(kwarg.getKey(), kwarg.getValue(), arguments, ast.getLocation());
      }

      return call(arguments.build(), ast, env);
    } catch (ConversionException | IllegalArgumentException | IllegalStateException
        | ClassCastException | ClassNotFoundException | ExecutionException e) {
      if (e.getMessage() != null) {
        throw new EvalException(ast.getLocation(), e.getMessage());
      } else {
        // TODO(bazel-team): ideally this shouldn't happen, however we need this for debugging
        throw new EvalExceptionWithJavaCause(ast.getLocation(), e);
      }
    }
  }

  private void checkTypeAndAddArg(String paramName, Object value,
      ImmutableMap.Builder<String, Object> arguments, Location loc) throws EvalException {
    SkylarkBuiltin.Param param = parameterTypes.get(paramName);
    if (param.callbackEnabled() && value instanceof Function) {
      // If we pass a function as an argument we trust the Function implementation with the type
      // check. It's OK since the function needs to be called manually anyway.
      arguments.put(paramName, value);
      return;
    }
    checkType(getName(), paramName, SkylarkType.of(param.type(), param.generic1()),
        value, loc, param.doc());
    arguments.put(paramName, value);
  }

  public static void checkType(String functionName, String paramName,
      SkylarkType type, Object value, Location loc, String paramDoc) throws EvalException {
    if (type != null && value != null) { // TODO(bazel-team): should we give a pass to NONE here?
      if (!type.contains(value)) {
        throw new EvalException(loc, String.format(
            "expected %s for '%s' while calling %s but got %s instead: %s",
            type, paramName, functionName, EvalUtils.getDataTypeName(value, true), value));
      }
    }
  }

  /**
   * The actual function call. All positional and keyword arguments are put in the
   * arguments map.
   */
  protected abstract Object call(
      Map<String, Object> arguments, FuncallExpression ast, Environment env) throws EvalException,
      ConversionException,
      IllegalArgumentException,
      IllegalStateException,
      InterruptedException,
      ClassCastException,
      ClassNotFoundException,
      ExecutionException;

  /**
   * An intermediate class to provide a simpler interface for Skylark functions.
   */
  public abstract static class SimpleSkylarkFunction extends SkylarkFunction {

    public SimpleSkylarkFunction(String name) {
      super(name);
    }

    @Override
    protected final Object call(
        Map<String, Object> arguments, FuncallExpression ast, Environment env) throws EvalException,
        ConversionException,
        IllegalArgumentException,
        IllegalStateException,
        ClassCastException,
        ExecutionException {
      return call(arguments, ast.getLocation());
    }

    /**
     * The actual function call. All positional and keyword arguments are put in the
     * arguments map.
     */
    protected abstract Object call(Map<String, Object> arguments, Location loc)
        throws EvalException,
        ConversionException,
        IllegalArgumentException,
        IllegalStateException,
        ClassCastException,
        ExecutionException;
  }

  // TODO(bazel-team): this is only used in MixedModeFunctions in MethodLibrary, migrate those
  // to SkylarkFunction then remove this.
  public static <TYPE> TYPE cast(Object elem, Class<TYPE> type, String what, Location loc)
      throws EvalException {
    try {
      return type.cast(elem);
    } catch (ClassCastException e) {
      throw new EvalException(loc, String.format("expected %s for '%s' but got %s instead",
          EvalUtils.getDataTypeNameFromClass(type), what, EvalUtils.getDataTypeName(elem)));
    }
  }
}

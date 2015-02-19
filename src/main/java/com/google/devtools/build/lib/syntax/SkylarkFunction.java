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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Type;
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
    Preconditions.checkArgument(getName().equals(annotation.name()),
                                getName() + " != " + annotation.name());
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

    Preconditions.checkState(configured, "Function " + getName() + " was not configured");
    try {
      ImmutableMap.Builder<String, Object> arguments = new ImmutableMap.Builder<>();
      if (objectType != null && !FuncallExpression.isNamespace(objectType)) {
        args = new ArrayList<Object>(args); // args immutable, get a mutable copy.
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
    if (param.callbackEnabled() && Function.class.isAssignableFrom(value.getClass())) {
      // If we pass a function as an argument we trust the Function implementation with the type
      // check. It's OK since the function needs to be called manually anyway.
      arguments.put(paramName, value);
      return;
    }
    cast(getName(), paramName, param.type(), param.generic1(), value, loc, param.doc());
    arguments.put(paramName, value);
  }

  /**
   * Throws an EvalException of realValue is not of the expected type, otherwise returns realValue.
   * 
   * @param functionName - name of the function
   * @param paramName - name of the parameter
   * @param expectedType - the expected type of the parameter
   * @param expectedGenericType - the expected generic type of the parameter, or
   * Object.class if undefined
   * @param realValue - the actual value of the parameter
   * @param loc - the location info used in the EvalException
   * @param paramDoc - the documentation of the parameter to print in the error message
   */
  @SuppressWarnings("unchecked")
  public static <T> T cast(String functionName, String paramName,
      Class<T> expectedType, Class<?> expectedGenericType,
      Object realValue, Location loc, String paramDoc) throws EvalException {
    if (!(expectedType.isAssignableFrom(realValue.getClass()))) {
      throw new EvalException(loc, String.format("expected %s for '%s' but got %s instead\n"
          + "%s.%s: %s",
          EvalUtils.getDataTypeNameFromClass(expectedType), paramName,
          EvalUtils.getDataTypeName(realValue), functionName, paramName, paramDoc));
    }
    if (expectedType.equals(SkylarkList.class)) {
      checkGeneric(functionName, paramName, expectedType, expectedGenericType,
          realValue, ((SkylarkList) realValue).getGenericType(), loc, paramDoc);
    } else if (expectedType.equals(SkylarkNestedSet.class)) {
      checkGeneric(functionName, paramName, expectedType, expectedGenericType,
          realValue, ((SkylarkNestedSet) realValue).getGenericType(), loc, paramDoc);
    }
    return (T) realValue;
  }

  private static void checkGeneric(String functionName, String paramName,
      Class<?> expectedType, Class<?> expectedGenericType,
      Object realValue, Class<?> realGenericType,
      Location loc, String paramDoc) throws EvalException {
    if (!realGenericType.equals(Object.class)
        && !expectedGenericType.isAssignableFrom(realGenericType)) {
      String mainType = EvalUtils.getDataTypeNameFromClass(expectedType);
      throw new EvalException(loc, String.format(
          "expected %s of %ss for '%s' but got %s of %ss instead\n%s.%s: %s",
        mainType, EvalUtils.getDataTypeNameFromClass(expectedGenericType),
        paramName,
        EvalUtils.getDataTypeName(realValue), EvalUtils.getDataTypeNameFromClass(realGenericType),
        functionName, paramName, paramDoc));
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

  public static <TYPE> Iterable<TYPE> castList(Object obj, final Class<TYPE> type) {
    if (obj == null) {
      return ImmutableList.of();
    }
    return ((SkylarkList) obj).to(type);
  }

  public static <TYPE> Iterable<TYPE> castList(
      Object obj, final Class<TYPE> type, final String what) throws ConversionException {
    if (obj == null) {
      return ImmutableList.of();
    }
    return Iterables.transform(Type.LIST.convert(obj, what),
        new com.google.common.base.Function<Object, TYPE>() {
          @Override
          public TYPE apply(Object input) {
            try {
              return type.cast(input);
            } catch (ClassCastException e) {
              throw new IllegalArgumentException(String.format(
                  "expected %s type for '%s' but got %s instead",
                  EvalUtils.getDataTypeNameFromClass(type), what,
                  EvalUtils.getDataTypeName(input)));
            }
          }
    });
  }

  public static <KEY_TYPE, VALUE_TYPE> ImmutableMap<KEY_TYPE, VALUE_TYPE> toMap(
      Iterable<Map.Entry<KEY_TYPE, VALUE_TYPE>> obj) {
    ImmutableMap.Builder<KEY_TYPE, VALUE_TYPE> builder = ImmutableMap.builder();
    for (Map.Entry<KEY_TYPE, VALUE_TYPE> entry : obj) {
      builder.put(entry.getKey(), entry.getValue());
    }
    return builder.build();
  }

  public static <KEY_TYPE, VALUE_TYPE> Iterable<Map.Entry<KEY_TYPE, VALUE_TYPE>> castMap(Object obj,
      final Class<KEY_TYPE> keyType, final Class<VALUE_TYPE> valueType, final String what) {
    if (obj == null) {
      return ImmutableList.of();
    }
    if (!(obj instanceof Map<?, ?>)) {
      throw new IllegalArgumentException(String.format(
          "expected a dictionary for %s but got %s instead",
          what, EvalUtils.getDataTypeName(obj)));
    }
    return Iterables.transform(((Map<?, ?>) obj).entrySet(),
        new com.google.common.base.Function<Map.Entry<?, ?>, Map.Entry<KEY_TYPE, VALUE_TYPE>>() {
          // This is safe. We check the type of the key-value pairs for every entry in the Map.
          // In Map.Entry the key always has the type of the first generic parameter, the
          // value has the second.
          @SuppressWarnings("unchecked")
            @Override
            public Map.Entry<KEY_TYPE, VALUE_TYPE> apply(Map.Entry<?, ?> input) {
            if (keyType.isAssignableFrom(input.getKey().getClass())
                && valueType.isAssignableFrom(input.getValue().getClass())) {
              return (Map.Entry<KEY_TYPE, VALUE_TYPE>) input;
            }
            throw new IllegalArgumentException(String.format(
                "expected <%s, %s> type for '%s' but got <%s, %s> instead",
                keyType.getSimpleName(), valueType.getSimpleName(), what,
                EvalUtils.getDataTypeName(input.getKey()),
                EvalUtils.getDataTypeName(input.getValue())));
          }
        });
  }

  // TODO(bazel-team): this is only used in MixedModeFunctions in MethodLibrary, migrate those
  // to SkylarkFunction then remove this.
  public static <TYPE> TYPE cast(Object elem, Class<TYPE> type, String what, Location loc)
      throws EvalException {
    try {
      return type.cast(elem);
    } catch (ClassCastException e) {
      throw new EvalException(loc, String.format("expected %s for '%s' but got %s instead",
          type.getSimpleName(), what, EvalUtils.getDataTypeName(elem)));
    }
  }
}

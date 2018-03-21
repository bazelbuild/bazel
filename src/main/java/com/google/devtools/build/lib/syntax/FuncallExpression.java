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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkInterfaceUtils;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException.EvalExceptionWithJavaCause;
import com.google.devtools.build.lib.syntax.Runtime.NoneType;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.StringUtilities;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Syntax node for a function call expression. */
@AutoCodec
public final class FuncallExpression extends Expression {

  /**
   * A value class to store Methods with their corresponding SkylarkCallable annotations.
   * This is needed because the annotation is sometimes in a superclass.
   */
  public static final class MethodDescriptor {
    private final Method method;
    private final SkylarkCallable annotation;

    private MethodDescriptor(Method method, SkylarkCallable annotation) {
      this.method = method;
      this.annotation = annotation;
    }

    Method getMethod() {
      return method;
    }

    /**
     * Returns the SkylarkCallable annotation corresponding to this method.
     */
    public SkylarkCallable getAnnotation() {
      return annotation;
    }
  }

  private static final LoadingCache<Class<?>, Map<String, List<MethodDescriptor>>> methodCache =
      CacheBuilder.newBuilder()
          .initialCapacity(10)
          .maximumSize(100)
          .build(
              new CacheLoader<Class<?>, Map<String, List<MethodDescriptor>>>() {

                @Override
                public Map<String, List<MethodDescriptor>> load(Class<?> key) throws Exception {
                  Map<String, List<MethodDescriptor>> methodMap = new HashMap<>();
                  for (Method method : key.getMethods()) {
                    // Synthetic methods lead to false multiple matches
                    if (method.isSynthetic()) {
                      continue;
                    }
                    SkylarkCallable callable = SkylarkInterfaceUtils.getSkylarkCallable(method);
                    if (callable == null) {
                      continue;
                    }
                    String name = callable.name();
                    if (name.isEmpty()) {
                      name = StringUtilities.toPythonStyleFunctionName(method.getName());
                    }
                    if (methodMap.containsKey(name)) {
                      methodMap.get(name).add(new MethodDescriptor(method, callable));
                    } else {
                      methodMap.put(
                          name, Lists.newArrayList(new MethodDescriptor(method, callable)));
                    }
                  }
                  return ImmutableMap.copyOf(methodMap);
                }
              });

  private static final LoadingCache<Class<?>, Map<String, MethodDescriptor>> fieldCache =
      CacheBuilder.newBuilder()
          .initialCapacity(10)
          .maximumSize(100)
          .build(
              new CacheLoader<Class<?>, Map<String, MethodDescriptor>>() {

                @Override
                public Map<String, MethodDescriptor> load(Class<?> key) throws Exception {
                  ImmutableMap.Builder<String, MethodDescriptor> fieldMap = ImmutableMap.builder();
                  HashSet<String> fieldNamesForCollisions = new HashSet<>();
                  List<MethodDescriptor> fieldMethods =
                      methodCache
                          .get(key)
                          .values()
                          .stream()
                          .flatMap(List::stream)
                          .filter(
                              methodDescriptor -> methodDescriptor.getAnnotation().structField())
                          .collect(Collectors.toList());

                  for (MethodDescriptor fieldMethod : fieldMethods) {
                    SkylarkCallable callable = fieldMethod.getAnnotation();
                    String name = callable.name();
                    if (name.isEmpty()) {
                      name =
                          StringUtilities.toPythonStyleFunctionName(
                              fieldMethod.getMethod().getName());
                    }
                    // TODO(b/72113542): Validate with annotation processor instead of at runtime.
                    if (!fieldNamesForCollisions.add(name)) {
                      throw new IllegalArgumentException(
                          String.format(
                              "Class %s has two structField methods named %s defined",
                              key.getName(), name));
                    }
                    fieldMap.put(name, fieldMethod);
                  }
                  return fieldMap.build();
                }
              });

  /**
   * Returns a map of methods and corresponding SkylarkCallable annotations of the methods of the
   * classObj class reachable from Skylark.
   */
  public static ImmutableMap<Method, SkylarkCallable> collectSkylarkMethodsWithAnnotation(
      Class<?> classObj) {
    ImmutableSortedMap.Builder<Method, SkylarkCallable> methodMap
        = ImmutableSortedMap.orderedBy(Comparator.comparing(Object::toString));
    for (Method method : classObj.getMethods()) {
      // Synthetic methods lead to false multiple matches
      if (!method.isSynthetic()) {
        SkylarkCallable annotation = SkylarkInterfaceUtils.getSkylarkCallable(classObj, method);
        if (annotation != null) {
          methodMap.put(method, annotation);
        }
      }
    }
    return methodMap.build();
  }

  private static class ArgumentListConversionResult {
    private final ImmutableList<Object> arguments;
    private final String error;

    private ArgumentListConversionResult(ImmutableList<Object> arguments, String error) {
      this.arguments = arguments;
      this.error = error;
    }

    public static ArgumentListConversionResult fromArgumentList(ImmutableList<Object> arguments) {
      return new ArgumentListConversionResult(arguments, null);
    }

    public static ArgumentListConversionResult fromError(String error) {
      return new ArgumentListConversionResult(null, error);
    }

    public String getError() {
      return error;
    }

    public ImmutableList<Object> getArguments() {
      return arguments;
    }
  }

  /**
   * An exception class to handle exceptions in direct Java API calls.
   */
  public static final class FuncallException extends Exception {

    public FuncallException(String msg) {
      super(msg);
    }
  }

  private final Expression function;

  private final ImmutableList<Argument.Passed> arguments;

  private final int numPositionalArgs;

  public FuncallExpression(Expression function, ImmutableList<Argument.Passed> arguments) {
    this.function = Preconditions.checkNotNull(function);
    this.arguments = Preconditions.checkNotNull(arguments);
    this.numPositionalArgs = countPositionalArguments();
  }

  /** Returns the function that is called. */
  public Expression getFunction() {
    return this.function;
  }

  /**
   * Returns the number of positional arguments.
   */
  private int countPositionalArguments() {
    int num = 0;
    for (Argument.Passed arg : arguments) {
      if (arg.isPositional()) {
        num++;
      }
    }
    return num;
  }

  /**
   * Returns an (immutable, ordered) list of function arguments. The first n are
   * positional and the remaining ones are keyword args, where n =
   * getNumPositionalArguments().
   */
  public List<Argument.Passed> getArguments() {
    return Collections.unmodifiableList(arguments);
  }

  /**
   * Returns the number of arguments which are positional; the remainder are
   * keyword arguments.
   */
  public int getNumPositionalArguments() {
    return numPositionalArgs;
  }

  @Override
  public void prettyPrint(Appendable buffer) throws IOException {
    function.prettyPrint(buffer);
    buffer.append('(');
    String sep = "";
    for (Argument.Passed arg : arguments) {
      buffer.append(sep);
      arg.prettyPrint(buffer);
      sep = ", ";
    }
    buffer.append(')');
  }

  @Override
  public String toString() {
    Printer.LengthLimitedPrinter printer = new Printer.LengthLimitedPrinter();
    printer.append(function.toString());
    printer.printAbbreviatedList(arguments, "(", ", ", ")", null,
        Printer.SUGGESTED_CRITICAL_LIST_ELEMENTS_COUNT,
        Printer.SUGGESTED_CRITICAL_LIST_ELEMENTS_STRING_LENGTH);
    return printer.toString();
  }

  /** Returns the Skylark callable Method of objClass with structField=true and the given name. */
  public static MethodDescriptor getStructField(Class<?> objClass, String methodName) {
    try {
      return fieldCache.get(objClass).get(methodName);
    } catch (ExecutionException e) {
      throw new IllegalStateException("Method loading failed: " + e);
    }
  }

  /** Returns the list of names of Skylark callable Methods of objClass with structField=true. */
  public static Set<String> getStructFieldNames(Class<?> objClass) {
    try {
      return fieldCache.get(objClass).keySet();
    } catch (ExecutionException e) {
      throw new IllegalStateException("Method loading failed: " + e);
    }
  }

  /** Returns the list of Skylark callable Methods of objClass with the given name. */
  public static List<MethodDescriptor> getMethods(Class<?> objClass, String methodName) {
    try {
      return methodCache.get(objClass).get(methodName);
    } catch (ExecutionException e) {
      throw new IllegalStateException("Method loading failed: " + e);
    }
  }

  /**
   * Returns a set of the Skylark name of all Skylark callable methods for object of type {@code
   * objClass}.
   */
  public static Set<String> getMethodNames(Class<?> objClass) {
    try {
      return methodCache.get(objClass).keySet();
    } catch (ExecutionException e) {
      throw new IllegalStateException("Method loading failed: " + e);
    }
  }

  /**
   * Invokes the given structField=true method and returns the result.
   *
   * @param methodDescriptor the descriptor of the method to invoke
   * @param fieldName the name of the struct field
   * @param obj the object on which to invoke the method
   * @return the method return value
   * @throws EvalException if there was an issue evaluating the method
   */
  public static Object invokeStructField(
      MethodDescriptor methodDescriptor, String fieldName, Object obj) throws EvalException {
    Preconditions.checkArgument(methodDescriptor.getAnnotation().structField());
    return callMethod(methodDescriptor, fieldName, obj, new Object[0], Location.BUILTIN, null);
  }

  static Object callMethod(MethodDescriptor methodDescriptor, String methodName, Object obj,
      Object[] args, Location loc, Environment env) throws EvalException {
    try {
      Method method = methodDescriptor.getMethod();
      if (obj == null && !Modifier.isStatic(method.getModifiers())) {
        throw new EvalException(loc, "method '" + methodName + "' is not static");
      }
      // This happens when the interface is public but the implementation classes
      // have reduced visibility.
      method.setAccessible(true);
      Object result = method.invoke(obj, args);
      if (method.getReturnType().equals(Void.TYPE)) {
        return Runtime.NONE;
      }
      if (result == null) {
        if (methodDescriptor.getAnnotation().allowReturnNones()) {
          return Runtime.NONE;
        } else {
          throw new EvalException(
              loc,
              "method invocation returned None, please file a bug report: "
                  + methodName
                  + Printer.printAbbreviatedList(
                  ImmutableList.copyOf(args), "(", ", ", ")", null));
        }
      }
      // TODO(bazel-team): get rid of this, by having everyone use the Skylark data structures
      result = SkylarkType.convertToSkylark(result, method, env);
      if (result != null && !EvalUtils.isSkylarkAcceptable(result.getClass())) {
        throw new EvalException(
            loc,
            Printer.format(
                "method '%s' returns an object of invalid type %r", methodName, result.getClass()));
      }
      return result;
    } catch (IllegalAccessException e) {
      // TODO(bazel-team): Print a nice error message. Maybe the method exists
      // and an argument is missing or has the wrong type.
      throw new EvalException(loc, "Method invocation failed: " + e);
    } catch (InvocationTargetException e) {
      if (e.getCause() instanceof FuncallException) {
        throw new EvalException(loc, e.getCause().getMessage());
      } else if (e.getCause() != null) {
        throw new EvalExceptionWithJavaCause(loc, e.getCause());
      } else {
        // This is unlikely to happen
        throw new EvalException(loc, "method invocation failed: " + e);
      }
    }
  }

  // TODO(bazel-team): If there's exactly one usable method, this works. If there are multiple
  // matching methods, it still can be a problem. Figure out how the Java compiler does it
  // exactly and copy that behaviour.
  // Throws an EvalException when it cannot find a matching function.
  private Pair<MethodDescriptor, List<Object>> findJavaMethod(
      Class<?> objClass,
      String methodName,
      List<Object> args,
      Map<String, Object> kwargs,
      Environment environment)
      throws EvalException {
    Pair<MethodDescriptor, List<Object>> matchingMethod = null;
    List<MethodDescriptor> methods = getMethods(objClass, methodName);
    ArgumentListConversionResult argumentListConversionResult = null;
    if (methods != null) {
      for (MethodDescriptor method : methods) {
        if (method.getAnnotation().structField()) {
          // TODO(cparsons): Allow structField methods to accept interpreter-supplied arguments.
          return new Pair<>(method, null);
        } else {
          argumentListConversionResult = convertArgumentList(args, kwargs, method, environment);
          if (argumentListConversionResult.getArguments() != null) {
            if (matchingMethod == null) {
              matchingMethod = new Pair<>(method, argumentListConversionResult.getArguments());
            } else {
              throw new EvalException(
                  getLocation(),
                  String.format(
                      "type '%s' has multiple matches for function %s",
                      EvalUtils.getDataTypeNameFromClass(objClass),
                      formatMethod(methodName, args, kwargs)));
            }
          }
        }
      }
    }
    if (matchingMethod == null) {
      String errorMessage;
      if (ClassObject.class.isAssignableFrom(objClass)) {
        errorMessage = String.format("struct has no method '%s'", methodName);
      } else if (argumentListConversionResult == null
          || argumentListConversionResult.getError() == null) {
        errorMessage =
            String.format(
                "type '%s' has no method %s",
                EvalUtils.getDataTypeNameFromClass(objClass),
                formatMethod(methodName, args, kwargs));

      } else {
        errorMessage =
            String.format(
                "%s, in method call %s of '%s'",
                argumentListConversionResult.getError(),
                formatMethod(methodName, args, kwargs),
                EvalUtils.getDataTypeNameFromClass(objClass));
      }
      throw new EvalException(getLocation(), errorMessage);
    }
    return matchingMethod;
  }

  private static SkylarkType getType(Param param) {
    if (param.allowedTypes().length > 0) {
      Preconditions.checkState(Object.class.equals(param.type()));
      SkylarkType result = SkylarkType.BOTTOM;
      for (ParamType paramType : param.allowedTypes()) {
        SkylarkType t =
            paramType.generic1() != Object.class
                ? SkylarkType.of(paramType.type(), paramType.generic1())
                : SkylarkType.of(paramType.type());
        result = SkylarkType.Union.of(result, t);
      }
      return result;
    } else {
      SkylarkType type =
          param.generic1() != Object.class
              ? SkylarkType.of(param.type(), param.generic1())
              : SkylarkType.of(param.type());
      return type;
    }
  }

  /**
   * Constructs the parameters list to actually pass to the method, filling with default values if
   * any. If there is a type or argument mismatch, returns a result containing an error message.
   */
  private ArgumentListConversionResult convertArgumentList(
      List<Object> args,
      Map<String, Object> kwargs,
      MethodDescriptor method,
      Environment environment) {
    ImmutableList.Builder<Object> builder = ImmutableList.builder();
    Class<?>[] javaMethodSignatureParams = method.getMethod().getParameterTypes();
    SkylarkCallable callable = method.getAnnotation();
    int numExtraInterpreterParams = 0;
    numExtraInterpreterParams += callable.useLocation() ? 1 : 0;
    numExtraInterpreterParams += callable.useAst() ? 1 : 0;
    numExtraInterpreterParams += callable.useEnvironment() ? 1 : 0;

    int mandatoryPositionals = callable.mandatoryPositionals();
    if (mandatoryPositionals < 0) {
      if (callable.parameters().length > 0) {
        mandatoryPositionals = 0;
      } else {
        mandatoryPositionals = javaMethodSignatureParams.length - numExtraInterpreterParams;
      }
    }
    if (mandatoryPositionals > args.size()) {
      return ArgumentListConversionResult.fromError("too few arguments");
    }
    if (args.size() > mandatoryPositionals + callable.parameters().length) {
      return ArgumentListConversionResult.fromError("too many arguments");
    }
    // First process the legacy positional parameters.
    int i = 0;
    if (mandatoryPositionals > 0) {
      for (Class<?> param : javaMethodSignatureParams) {
        Object value = args.get(i);
        if (!param.isAssignableFrom(value.getClass())) {
          return ArgumentListConversionResult.fromError(
              String.format(
                  "Cannot convert parameter at position %d from type %s to type %s",
                  i, EvalUtils.getDataTypeName(value), param.toString()));
        }
        builder.add(value);
        i++;
        if (i >= mandatoryPositionals) {
          // Stops for specified parameters instead.
          break;
        }
      }
    }

    // Then the parameters specified in callable.parameters()
    Set<String> keys = new LinkedHashSet<>(kwargs.keySet());
    for (Param param : callable.parameters()) {
      SkylarkType type = getType(param);
      if (param.noneable()) {
        type = SkylarkType.Union.of(type, SkylarkType.NONE);
      }
      Object value = null;
      if (i < args.size()) {
        value = args.get(i);
        if (!param.positional()) {
          return ArgumentListConversionResult.fromError(
              String.format("Parameter '%s' is not positional", param.name()));
        } else if (!type.contains(value)) {
          return ArgumentListConversionResult.fromError(
              String.format(
                  "expected value of type '%s' for parameter '%s'",
                  type.toString(), param.name()));
        }
        i++;
      } else if (param.named() && keys.remove(param.name())) {
        // Named parameters
        value = kwargs.get(param.name());
        if (!type.contains(value)) {
          return ArgumentListConversionResult.fromError(
              String.format(
                  "expected value of type '%s' for parameter '%s'",
                  type.toString(), param.name()));
        }
      } else {
        // Use default value
        if (param.defaultValue().isEmpty()) {
          return ArgumentListConversionResult.fromError(
              String.format("parameter '%s' has no default value", param.name()));
        }
        value = SkylarkSignatureProcessor.getDefaultValue(param, null);
      }
      builder.add(value);
      if (!param.noneable() && value instanceof NoneType) {
        return ArgumentListConversionResult.fromError(
            String.format("parameter '%s' cannot be None", param.name()));
      }
    }
    if (i < args.size()) {
      return ArgumentListConversionResult.fromError("too many arguments");
    }
    if (!keys.isEmpty()) {
      return ArgumentListConversionResult.fromError(
          String.format("unexpected keyword%s %s",
              keys.size() > 1 ? "s" : "",
              Joiner.on(",").join(Iterables.transform(keys, s -> "'" + s + "'"))));
    }

    // Then add any skylark-info arguments (for example the Environment).
    if (callable.useLocation()) {
      builder.add(getLocation());
    }
    if (callable.useAst()) {
      builder.add(this);
    }
    if (callable.useEnvironment()) {
      builder.add(environment);
    }

    return ArgumentListConversionResult.fromArgumentList(builder.build());
  }

  private static String formatMethod(String name, List<Object> args, Map<String, Object> kwargs) {
    StringBuilder sb = new StringBuilder();
    sb.append(name).append("(");
    boolean first = true;
    for (Object obj : args) {
      if (!first) {
        sb.append(", ");
      }
      sb.append(EvalUtils.getDataTypeName(obj));
      first = false;
    }
    for (Map.Entry<String, Object> kwarg : kwargs.entrySet()) {
      if (!first) {
        sb.append(", ");
      }
      sb.append(EvalUtils.getDataTypeName(kwarg.getValue()));
      sb.append(" ");
      sb.append(kwarg.getKey());
      first = false;
    }
    return sb.append(")").toString();
  }

  /**
   * Add one named argument to the keyword map, and returns whether that name has been encountered
   * before.
   */
  private static boolean addKeywordArgAndCheckIfDuplicate(
      Map<String, Object> kwargs,
      String name,
      Object value) {
    return kwargs.put(name, value) != null;
  }

  /**
   * Add multiple arguments to the keyword map (**kwargs), and returns all the names of those
   * arguments that have been encountered before or {@code null} if there are no such names.
   */
  @Nullable
  private static ImmutableList<String> addKeywordArgsAndReturnDuplicates(
      Map<String, Object> kwargs,
      Object items,
      Location location)
      throws EvalException {
    if (!(items instanceof Map<?, ?>)) {
      throw new EvalException(
          location,
          "argument after ** must be a dictionary, not '" + EvalUtils.getDataTypeName(items) + "'");
    }
    ImmutableList.Builder<String> duplicatesBuilder = null;
    for (Map.Entry<?, ?> entry : ((Map<?, ?>) items).entrySet()) {
      if (!(entry.getKey() instanceof String)) {
        throw new EvalException(
            location,
            "keywords must be strings, not '" + EvalUtils.getDataTypeName(entry.getKey()) + "'");
      }
      String argName = (String) entry.getKey();
      if (addKeywordArgAndCheckIfDuplicate(kwargs, argName, entry.getValue())) {
        if (duplicatesBuilder == null) {
          duplicatesBuilder = ImmutableList.builder();
        }
        duplicatesBuilder.add(argName);
      }
    }
    return duplicatesBuilder == null ? null : duplicatesBuilder.build();
  }

  /**
   * Checks whether the given object is a {@link BaseFunction}.
   *
   * @throws EvalException If not a BaseFunction.
   */
  private static BaseFunction checkCallable(Object functionValue, Location location)
      throws EvalException {
    if (functionValue instanceof BaseFunction) {
      return (BaseFunction) functionValue;
    } else {
      throw new EvalException(
          location, "'" + EvalUtils.getDataTypeName(functionValue) + "' object is not callable");
    }
  }

  /**
   * Call a method depending on the type of an object it is called on.
   *
   * @param positionals The first object is expected to be the object the method is called on.
   * @param call the original expression that caused this call, needed for rules especially
   */
  private Object invokeObjectMethod(
      String method,
      ImmutableList<Object> positionals,
      ImmutableMap<String, Object> keyWordArgs,
      FuncallExpression call,
      Environment env)
      throws EvalException, InterruptedException {
    Location location = call.getLocation();
    Object value = positionals.get(0);
    ImmutableList<Object> positionalArgs = positionals.subList(1, positionals.size());
    BaseFunction function = Runtime.getBuiltinRegistry().getFunction(value.getClass(), method);
    Object fieldValue =
        (value instanceof ClassObject) ? ((ClassObject) value).getValue(method) : null;
    if (function != null) {
      if (!isNamespace(value.getClass())) {
        // Use self as an implicit parameter in front.
        positionalArgs = positionals;
      }
      return function.call(
          positionalArgs, ImmutableMap.copyOf(keyWordArgs), call, env);
    } else if (fieldValue != null) {
      if (!(fieldValue instanceof BaseFunction)) {
        throw new EvalException(
            location, String.format("struct field '%s' is not a function", method));
      }
      function = (BaseFunction) fieldValue;
      return function.call(
          positionalArgs, ImmutableMap.copyOf(keyWordArgs), call, env);
    } else {
      // When calling a Java method, the name is not in the Environment,
      // so evaluating 'function' would fail.
      Class<?> objClass;
      Object obj;
      if (value instanceof Class<?>) {
        // Static call
        obj = null;
        objClass = (Class<?>) value;
      } else {
        obj = value;
        objClass = value.getClass();
      }
      Pair<MethodDescriptor, List<Object>> javaMethod =
          call.findJavaMethod(objClass, method, positionalArgs, keyWordArgs, env);
      if (javaMethod.first.getAnnotation().structField()) {
        // Not a method but a callable attribute
        try {
          return callFunction(javaMethod.first.getMethod().invoke(obj), env);
        } catch (IllegalAccessException e) {
          throw new EvalException(getLocation(), "method invocation failed: " + e);
        } catch (InvocationTargetException e) {
          if (e.getCause() instanceof FuncallException) {
            throw new EvalException(getLocation(), e.getCause().getMessage());
          } else if (e.getCause() != null) {
            throw new EvalExceptionWithJavaCause(getLocation(), e.getCause());
          } else {
            // This is unlikely to happen
            throw new EvalException(getLocation(), "method invocation failed: " + e);
          }
        }
      }
      return callMethod(javaMethod.first, method, obj, javaMethod.second.toArray(), location, env);
    }
  }

  @SuppressWarnings("unchecked")
  private void evalArguments(ImmutableList.Builder<Object> posargs, Map<String, Object> kwargs,
      Environment env)
      throws EvalException, InterruptedException {
    // Optimize allocations for the common case where they are no duplicates.
    ImmutableList.Builder<String> duplicatesBuilder = null;
    // Iterate over the arguments. We assume all positional arguments come before any keyword
    // or star arguments, because the argument list was already validated by
    // Argument#validateFuncallArguments, as called by the Parser,
    // which should be the only place that build FuncallExpression-s.
    // Argument lists are typically short and functions are frequently called, so go by index
    // (O(1) for ImmutableList) to avoid the iterator overhead.
    for (int i = 0; i < arguments.size(); i++) {
      Argument.Passed arg = arguments.get(i);
      Object value = arg.getValue().eval(env);
      if (arg.isPositional()) {
        posargs.add(value);
      } else if (arg.isStar()) {  // expand the starArg
        if (!(value instanceof Iterable)) {
          throw new EvalException(
              getLocation(),
              "argument after * must be an iterable, not " + EvalUtils.getDataTypeName(value));
        }
        posargs.addAll((Iterable<Object>) value);
      } else if (arg.isStarStar()) {  // expand the kwargs
        ImmutableList<String> duplicates =
            addKeywordArgsAndReturnDuplicates(kwargs, value, getLocation());
        if (duplicates != null) {
          if (duplicatesBuilder == null) {
            duplicatesBuilder = ImmutableList.builder();
          }
          duplicatesBuilder.addAll(duplicates);
        }
      } else {
        if (addKeywordArgAndCheckIfDuplicate(kwargs, arg.getName(), value)) {
          if (duplicatesBuilder == null) {
            duplicatesBuilder = ImmutableList.builder();
          }
          duplicatesBuilder.add(arg.getName());
        }
      }
    }
    if (duplicatesBuilder != null) {
      ImmutableList<String> dups = duplicatesBuilder.build();
      throw new EvalException(
          getLocation(),
          "duplicate keyword"
              + (dups.size() > 1 ? "s" : "")
              + " '"
              + Joiner.on("', '").join(dups)
              + "' in call to "
              + function);
    }
  }

  @VisibleForTesting
  public static boolean isNamespace(Class<?> classObject) {
    return classObject.isAnnotationPresent(SkylarkModule.class)
        && classObject.getAnnotation(SkylarkModule.class).namespace();
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
    // TODO: Remove this special case once method resolution and invocation are supported as
    // separate steps.
    if (function instanceof DotExpression) {
      return invokeObjectMethod(env, (DotExpression) function);
    }
    Object funcValue = function.eval(env);
    return callFunction(funcValue, env);
  }

  /** Invokes object.function() and returns the result. */
  private Object invokeObjectMethod(Environment env, DotExpression dot)
      throws EvalException, InterruptedException {
    Object objValue = dot.getObject().eval(env);
    ImmutableList.Builder<Object> posargs = new ImmutableList.Builder<>();
    posargs.add(objValue);
    // We copy this into an ImmutableMap in the end, but we can't use an ImmutableMap.Builder, or
    // we'd still have to have a HashMap on the side for the sake of properly handling duplicates.
    Map<String, Object> kwargs = new LinkedHashMap<>();
    evalArguments(posargs, kwargs, env);
    return invokeObjectMethod(
        dot.getField().getName(), posargs.build(), ImmutableMap.copyOf(kwargs), this, env);
  }

  /**
   * Calls a function object
   */
  private Object callFunction(Object funcValue, Environment env)
      throws EvalException, InterruptedException {
    ImmutableList.Builder<Object> posargs = new ImmutableList.Builder<>();
    // We copy this into an ImmutableMap in the end, but we can't use an ImmutableMap.Builder, or
    // we'd still have to have a HashMap on the side for the sake of properly handling duplicates.
    Map<String, Object> kwargs = new LinkedHashMap<>();
    BaseFunction function = checkCallable(funcValue, getLocation());
    evalArguments(posargs, kwargs, env);
    return function.call(posargs.build(), ImmutableMap.copyOf(kwargs), this, env);
  }

  /**
   * Returns the value of the argument 'name' (or null if there is none).
   * This function is used to associate debugging information to rules created by skylark "macros".
   */
  @Nullable
  public String getNameArg() {
    for (Argument.Passed arg : arguments) {
      if (arg != null) {
        String name = arg.getName();
        if (name != null && name.equals("name")) {
          Expression expr = arg.getValue();
          return (expr instanceof StringLiteral) ? ((StringLiteral) expr).getValue() : null;
        }
      }
    }
    return null;
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.FUNCALL;
  }

  @Override
  protected boolean isNewScope() {
    return true;
  }
}

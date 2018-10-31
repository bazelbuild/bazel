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
import com.google.common.base.Throwables;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkInterfaceUtils;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException.EvalExceptionWithJavaCause;
import com.google.devtools.build.lib.syntax.Runtime.NoneType;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.util.Pair;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Syntax node for a function call expression. */
public final class FuncallExpression extends Expression {

  /**
   * Cache key for callable method lookup of skylark types. The key consists of the class of the
   * skylark type, and a skylark semantics object. The semantics object is required as part of the
   * key as certain methods of the class may be unavailable if certain semantics flags are flipped.
   */
  private static final class MethodDescriptorKey {
    private final Class<?> clazz;
    private final SkylarkSemantics semantics;

    private MethodDescriptorKey(Class<?> clazz, SkylarkSemantics semantics) {
      this.clazz = clazz;
      this.semantics = semantics;
    }

    public Class<?> getClazz() {
      return clazz;
    }

    public SkylarkSemantics getSemantics() {
      return semantics;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      MethodDescriptorKey that = (MethodDescriptorKey) o;
      return Objects.equals(clazz, that.clazz) && Objects.equals(semantics, that.semantics);
    }

    @Override
    public int hashCode() {
      return Objects.hash(clazz, semantics);
    }
  }

  private static final LoadingCache<MethodDescriptorKey, Optional<MethodDescriptor>> selfCallCache =
      CacheBuilder.newBuilder()
          .build(
              new CacheLoader<MethodDescriptorKey, Optional<MethodDescriptor>>() {
                @Override
                public Optional<MethodDescriptor> load(MethodDescriptorKey key) throws Exception {
                  Class<?> keyClass = key.getClazz();
                  SkylarkSemantics semantics = key.getSemantics();
                  MethodDescriptor returnValue = null;
                  for (Method method : sortMethodArrayByMethodName(keyClass.getMethods())) {
                    // Synthetic methods lead to false multiple matches
                    if (method.isSynthetic()) {
                      continue;
                    }
                    SkylarkCallable callable = SkylarkInterfaceUtils.getSkylarkCallable(method);
                    if (callable != null && callable.selfCall()) {
                      if (returnValue != null) {
                        throw new IllegalArgumentException(
                            String.format(
                                "Class %s has two selfCall methods defined", keyClass.getName()));
                      }
                      if (semantics.isFeatureEnabledBasedOnTogglingFlags(
                          callable.enableOnlyWithFlag(), callable.disableWithFlag())) {
                        returnValue = MethodDescriptor.of(method, callable);
                      }
                    }
                  }
                  return Optional.ofNullable(returnValue);
                }
              });

  private static final LoadingCache<MethodDescriptorKey, Map<String, MethodDescriptor>>
      methodCache =
          CacheBuilder.newBuilder()
              .build(
                  new CacheLoader<MethodDescriptorKey, Map<String, MethodDescriptor>>() {

                    @Override
                    public Map<String, MethodDescriptor> load(MethodDescriptorKey key)
                        throws Exception {
                      Class<?> keyClass = key.getClazz();
                      SkylarkSemantics semantics = key.getSemantics();
                      ImmutableMap.Builder<String, MethodDescriptor> methodMap =
                          ImmutableMap.builder();
                      for (Method method : sortMethodArrayByMethodName(keyClass.getMethods())) {
                        // Synthetic methods lead to false multiple matches
                        if (method.isSynthetic()) {
                          continue;
                        }
                        SkylarkCallable callable = SkylarkInterfaceUtils.getSkylarkCallable(method);
                        if (callable == null) {
                          continue;
                        }
                        if (callable.selfCall()) {
                          // Self-call java methods are not treated as methods of the skylark value.
                          continue;
                        }
                        if (semantics.isFeatureEnabledBasedOnTogglingFlags(
                            callable.enableOnlyWithFlag(), callable.disableWithFlag())) {
                          methodMap.put(callable.name(), MethodDescriptor.of(method, callable));
                        }
                      }
                      return methodMap.build();
                    }
                  });

  private static final LoadingCache<MethodDescriptorKey, Map<String, MethodDescriptor>> fieldCache =
      CacheBuilder.newBuilder()
          .build(
              new CacheLoader<MethodDescriptorKey, Map<String, MethodDescriptor>>() {

                @Override
                public Map<String, MethodDescriptor> load(MethodDescriptorKey key)
                    throws Exception {
                  ImmutableMap.Builder<String, MethodDescriptor> fieldMap = ImmutableMap.builder();
                  HashSet<String> fieldNamesForCollisions = new HashSet<>();
                  List<MethodDescriptor> fieldMethods =
                      methodCache.get(key).values().stream()
                          .filter(MethodDescriptor::isStructField)
                          .collect(Collectors.toList());

                  for (MethodDescriptor fieldMethod : fieldMethods) {
                    String name = fieldMethod.getName();
                    // TODO(b/72113542): Validate with annotation processor instead of at runtime.
                    if (!fieldNamesForCollisions.add(name)) {
                      throw new IllegalArgumentException(
                          String.format(
                              "Class %s has two structField methods named %s defined",
                              key.getClazz().getName(), name));
                    }
                    fieldMap.put(name, fieldMethod);
                  }
                  return fieldMap.build();
                }
              });

  // *args, **kwargs, location, ast, environment, skylark semantics
  private static final int EXTRA_ARGS_COUNT = 6;

  /**
   * Returns a map of methods and corresponding SkylarkCallable annotations of the methods of the
   * classObj class reachable from Skylark.
   */
  public static ImmutableMap<Method, SkylarkCallable> collectSkylarkMethodsWithAnnotation(
      Class<?> classObj) {
    ImmutableSortedMap.Builder<Method, SkylarkCallable> methodMap
        = ImmutableSortedMap.orderedBy(Comparator.comparing(Object::toString));
    for (Method method : sortMethodArrayByMethodName(classObj.getMethods())) {
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

  /** Sort Method arrays by their name for a deterministic ordering */
  private static Method[] sortMethodArrayByMethodName(Method[] methods) {
    Arrays.sort(methods, Comparator.comparing(Method::getName));
    return methods;
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

  /**
   * Returns either the class itself or, if the class is {@link String}, the proxy class
   * containing all 'string' methods.
   */
  private static Class<?> getClassOrProxyClass(Class<?> clazz) {
    return String.class.isAssignableFrom(clazz)
        ? StringModule.class
        : clazz;
  }

  /**
   * Returns the Skylark callable Method of objClass with structField=true and the given name.
   *
   * @deprecated use {@link #getStructField(SkylarkSemantics, Class, String)} instead
   */
  @Deprecated
  public static MethodDescriptor getStructField(Class<?> objClass, String methodName) {
    return getStructField(SkylarkSemantics.DEFAULT_SEMANTICS, objClass, methodName);
  }

  /** Returns the Skylark callable Method of objClass with structField=true and the given name. */
  public static MethodDescriptor getStructField(
      SkylarkSemantics semantics, Class<?> objClass, String methodName) {
    try {
      return fieldCache
          .get(new MethodDescriptorKey(getClassOrProxyClass(objClass), semantics))
          .get(methodName);
    } catch (ExecutionException e) {
      throw new IllegalStateException("Method loading failed: " + e);
    }
  }

  /**
   * Returns the list of names of Skylark callable Methods of objClass with structField=true.
   *
   * @deprecated use {@link #getStructFieldNames(SkylarkSemantics, Class)} instead
   */
  @Deprecated
  public static Set<String> getStructFieldNames(Class<?> objClass) {
    return getStructFieldNames(SkylarkSemantics.DEFAULT_SEMANTICS, objClass);
  }
  
  /** Returns the list of names of Skylark callable Methods of objClass with structField=true. */
  public static Set<String> getStructFieldNames(SkylarkSemantics semantics, Class<?> objClass) {
    try {
      return fieldCache
          .get(new MethodDescriptorKey(getClassOrProxyClass(objClass), semantics))
          .keySet();
    } catch (ExecutionException e) {
      throw new IllegalStateException("Method loading failed: " + e);
    }
  }

  /**
   * Returns the list of Skylark callable Methods of objClass with the given name.
   *
   * @deprecated use {@link #getMethods(SkylarkSemantics, Class, String)} instead
   */
  @Deprecated
  public static MethodDescriptor getMethod(Class<?> objClass, String methodName) {
    return getMethod(SkylarkSemantics.DEFAULT_SEMANTICS, objClass, methodName);
  }

  /** Returns the list of Skylark callable Methods of objClass with the given name. */
  public static MethodDescriptor getMethod(
      SkylarkSemantics semantics, Class<?> objClass, String methodName) {
    try {
      return methodCache
          .get(new MethodDescriptorKey(getClassOrProxyClass(objClass), semantics))
          .get(methodName);
    } catch (ExecutionException e) {
      throw new IllegalStateException("Method loading failed: " + e);
    }
  }

  /**
   * Returns a set of the Skylark name of all Skylark callable methods for object of type {@code
   * objClass}.
   *
   * @deprecated use {@link #getMethodNames(SkylarkSemantics, Class)} instead
   */
  @Deprecated
  public static Set<String> getMethodNames(Class<?> objClass) {
    return getMethodNames(SkylarkSemantics.DEFAULT_SEMANTICS, objClass);
  }

  /**
   * Returns a set of the Skylark name of all Skylark callable methods for object of type {@code
   * objClass}.
   */
  public static Set<String> getMethodNames(SkylarkSemantics semantics, Class<?> objClass) {
    try {
      return methodCache
          .get(new MethodDescriptorKey(getClassOrProxyClass(objClass), semantics))
          .keySet();
    } catch (ExecutionException e) {
      throw new IllegalStateException("Method loading failed: " + e);
    }
  }

  /**
   * Returns true if the given class has a method annotated with {@link SkylarkCallable} with {@link
   * SkylarkCallable#selfCall()} set to true.
   */
  public static boolean hasSelfCallMethod(SkylarkSemantics semantics, Class<?> objClass) {
    try {
      return selfCallCache.get(new MethodDescriptorKey(objClass, semantics)).isPresent();
    } catch (ExecutionException e) {
      throw new IllegalStateException("Method loading failed: " + e);
    }
  }

  /**
   * Returns a {@link BuiltinCallable} object representing a function which calls the selfCall java
   * method of the given object (the {@link SkylarkCallable} method with {@link
   * SkylarkCallable#selfCall()} set to true).
   *
   * @throws IllegalStateException if no such method exists for the object
   */
  public static BuiltinCallable getSelfCallMethod(SkylarkSemantics semantics, Object obj) {
    try {
      Optional<MethodDescriptor> selfCallDescriptor =
          selfCallCache.get(new MethodDescriptorKey(obj.getClass(), semantics));
      if (!selfCallDescriptor.isPresent()) {
        throw new IllegalStateException("Class " + obj.getClass() + " has no selfCall method");
      }
      MethodDescriptor descriptor = selfCallDescriptor.get();
      return new BuiltinCallable(descriptor.getName(), obj, descriptor);
    } catch (ExecutionException e) {
      throw new IllegalStateException("Method loading failed: " + e);
    }
  }

  /**
   * Returns a {@link BuiltinCallable} representing a {@link SkylarkCallable}-annotated instance
   * method of a given object with the given method name.
   */
  public static BuiltinCallable getBuiltinCallable(Object obj, String methodName) {
    Class<?> objClass = obj.getClass();
    MethodDescriptor methodDescriptor = getMethod(objClass, methodName);
    if (methodDescriptor == null) {
      throw new IllegalStateException(String.format(
          "Expected a method named '%s' in %s, but found none",
          methodName, objClass));
    }
    return new BuiltinCallable(methodName, obj, methodDescriptor);
  }

  /**
   * Invokes the given structField=true method and returns the result.
   *
   * <p>The given method must <b>not</b> require extra-interpreter parameters, such as
   * {@link Environment}. This method throws {@link IllegalArgumentException} for violations.</p>
   *
   * @param methodDescriptor the descriptor of the method to invoke
   * @param fieldName the name of the struct field
   * @param obj the object on which to invoke the method
   * @return the method return value
   * @throws EvalException if there was an issue evaluating the method
   */
  public static Object invokeStructField(
      MethodDescriptor methodDescriptor, String fieldName, Object obj)
      throws EvalException, InterruptedException {
    Preconditions.checkArgument(
        methodDescriptor.isStructField(), "Can only be invoked on structField callables");
    Preconditions.checkArgument(
        !methodDescriptor.isUseEnvironment()
            || !methodDescriptor.isUseSkylarkSemantics()
            || !methodDescriptor.isUseLocation(),
        "Cannot be invoked on structField callables with extra interpreter params");
    return methodDescriptor.call(obj, new Object[0], Location.BUILTIN, null);
  }

  // Throws an EvalException when there is no function matching the given name and arguments.
  private Pair<MethodDescriptor, List<Object>> findJavaMethod(
      Class<?> objClass,
      String methodName,
      List<Object> args,
      Map<String, Object> kwargs,
      Environment environment)
      throws EvalException {
    MethodDescriptor method = getMethod(environment.getSemantics(), objClass, methodName);
    ArgumentListConversionResult argumentListConversionResult = null;
    if (method != null) {
      if (method.isStructField()) {
        // This indicates a built-in structField which returns a function which may have
        // one or more arguments itself. For example, foo.bar('baz'), where foo.bar is a
        // structField returning a function. Calling the "bar" callable of foo should
        // not have 'baz' propagated, though extra interpreter arguments should be supplied.
        return new Pair<>(method, extraInterpreterArgs(method, null, getLocation(), environment));
      } else {
        argumentListConversionResult = convertArgumentList(args, kwargs, method, environment);

        if (argumentListConversionResult.getArguments() != null) {
          return new Pair<>(method, argumentListConversionResult.getArguments());
        } else {
          throw new EvalException(getLocation(),
              String.format(
                  "%s, in method call %s of '%s'",
                  argumentListConversionResult.getError(),
                  formatMethod(objClass, methodName, args, kwargs),
                  EvalUtils.getDataTypeNameFromClass(objClass)));
        }
      }
    } else { // method == null
      throw new EvalException(getLocation(),
          String.format(
              "type '%s' has no method %s",
              EvalUtils.getDataTypeNameFromClass(objClass),
              formatMethod(objClass, methodName, args, kwargs)));
    }
  }

  /**
   * Returns the extra interpreter arguments for the given {@link SkylarkCallable}, to be added at
   * the end of the argument list for the callable.
   *
   * <p>This method accepts null {@code ast} only if {@code callable.useAst()} is false. It is up to
   * the caller to validate this invariant.
   */
  public static List<Object> extraInterpreterArgs(
      MethodDescriptor method, @Nullable FuncallExpression ast, Location loc, Environment env) {
    ImmutableList.Builder<Object> builder = ImmutableList.builder();
    appendExtraInterpreterArgs(builder, method, ast, loc, env);
    return builder.build();
  }

  /**
   * Same as {@link #extraInterpreterArgs(MethodDescriptor, FuncallExpression, Location,
   * Environment)} but appends args to a passed {@code builder} to avoid unnecessary allocations of
   * intermediate instances.
   *
   * @see #extraInterpreterArgs(MethodDescriptor, FuncallExpression, Location, Environment)
   */
  private static void appendExtraInterpreterArgs(
      ImmutableList.Builder<Object> builder,
      MethodDescriptor method,
      @Nullable FuncallExpression ast,
      Location loc,
      Environment env) {
    if (method.isUseLocation()) {
      builder.add(loc);
    }
    if (method.isUseAst()) {
      if (ast == null) {
        throw new IllegalArgumentException("Callable expects to receive ast: " + method.getName());
      }
      builder.add(ast);
    }
    if (method.isUseEnvironment()) {
      builder.add(env);
    }
    if (method.isUseSkylarkSemantics()) {
      builder.add(env.getSemantics());
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
    ImmutableList<ParamDescriptor> parameters = method.getParameters();
    ImmutableList.Builder<Object> builder =
        ImmutableList.builderWithExpectedSize(parameters.size() + EXTRA_ARGS_COUNT);
    boolean acceptsExtraArgs = method.isAcceptsExtraArgs();
    boolean acceptsExtraKwargs = method.isAcceptsExtraKwargs();

    int argIndex = 0;

    // Process parameters specified in callable.parameters()
    Set<String> keys = new LinkedHashSet<>(kwargs.keySet());
    // Positional parameters are always enumerated before non-positional parameters,
    // And default-valued positional parameters are always enumerated after other positional
    // parameters. These invariants are validated by the SkylarkCallable annotation processor.
    // Index is used deliberately, since usage of iterators adds a significant overhead
    for (int i = 0; i < parameters.size(); ++i) {
      ParamDescriptor param = parameters.get(i);
      SkylarkType type = param.getSkylarkType();
      Object value;

      if (argIndex < args.size() && param.isPositional()) { // Positional args and params remain.
        value = args.get(argIndex);
        if (!type.contains(value)) {
          return ArgumentListConversionResult.fromError(
              String.format(
                  "expected value of type '%s' for parameter '%s'", type, param.getName()));
        }
        if (param.isNamed() && keys.contains(param.getName())) {
          return ArgumentListConversionResult.fromError(
              String.format("got multiple values for keyword argument '%s'", param.getName()));
        }
        argIndex++;
      } else { // No more positional arguments, or no more positional parameters.
        if (param.isNamed() && keys.remove(param.getName())) {
          // Param specified by keyword argument.
          value = kwargs.get(param.getName());
          if (!type.contains(value)) {
            return ArgumentListConversionResult.fromError(
                String.format(
                    "expected value of type '%s' for parameter '%s'", type, param.getName()));
          }
        } else { // Param not specified by user. Use default value.
          if (param.getDefaultValue().isEmpty()) {
            return ArgumentListConversionResult.fromError(
                String.format("parameter '%s' has no default value", param.getName()));
          }
          value =
              SkylarkSignatureProcessor.getDefaultValue(
                  param.getName(), param.getDefaultValue(), null);
        }
      }
      if (!param.isNoneable() && value instanceof NoneType) {
        return ArgumentListConversionResult.fromError(
            String.format("parameter '%s' cannot be None", param.getName()));
      }
      builder.add(value);
    }

    ImmutableList<Object> extraArgs = ImmutableList.of();
    if (argIndex < args.size()) {
      if (acceptsExtraArgs) {
        ImmutableList.Builder<Object> extraArgsBuilder =
            ImmutableList.builderWithExpectedSize(args.size() - argIndex);
        for (; argIndex < args.size(); argIndex++) {
          extraArgsBuilder.add(args.get(argIndex));
        }
        extraArgs = extraArgsBuilder.build();
      } else {
        return ArgumentListConversionResult.fromError(
            String.format(
                "expected no more than %s positional arguments, but got %s",
                argIndex, args.size()));
      }
    }
    ImmutableMap<String, Object> extraKwargs = ImmutableMap.of();
    if (!keys.isEmpty()) {
      if (acceptsExtraKwargs) {
        ImmutableMap.Builder<String, Object> extraKwargsBuilder =
            ImmutableMap.builderWithExpectedSize(keys.size());
        for (String key : keys) {
          extraKwargsBuilder.put(key, kwargs.get(key));
        }
        extraKwargs = extraKwargsBuilder.build();
      } else {
        return ArgumentListConversionResult.fromError(
            String.format(
                "unexpected keyword%s %s",
                keys.size() > 1 ? "s" : "",
                Joiner.on(",").join(Iterables.transform(keys, s -> "'" + s + "'"))));
      }
    }

    // Then add any skylark-interpreter arguments (for example kwargs or the Environment).
    if (acceptsExtraArgs) {
      builder.add(Tuple.copyOf(extraArgs));
    }
    if (acceptsExtraKwargs) {
      builder.add(SkylarkDict.copyOf(environment, extraKwargs));
    }
    appendExtraInterpreterArgs(builder, method, this, getLocation(), environment);

    return ArgumentListConversionResult.fromArgumentList(builder.build());
  }

  private static String formatMethod(
      Class<?> objClass, String name, List<Object> args, Map<String, Object> kwargs) {
    if (objClass == StringModule.class) {
      // StringModule is a special case, and begins with a String "self" parameter which should
      // be omitted from the method format.
      args = args.subList(1, args.size());
    }
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
   * Checks whether the given object is callable, either by being a {@link BaseFunction} or having a
   * {@link SkylarkCallable}-annotated method with selfCall = true.
   *
   * @return a BaseFunction object representing the callable function this object represents
   * @throws EvalException if the object is not callable.
   */
  private static BaseFunction checkCallable(
      SkylarkSemantics semantics, Object functionValue, Location location) throws EvalException {
    if (functionValue instanceof BaseFunction) {
      return (BaseFunction) functionValue;
    } else if (hasSelfCallMethod(semantics, functionValue.getClass())) {
      return getSelfCallMethod(semantics, functionValue);
    } else {
      throw new EvalException(
          location, "'" + EvalUtils.getDataTypeName(functionValue) + "' object is not callable");
    }
  }

  private boolean includeSelfAsArg(
      Object value, @Nullable BaseFunction globalBuiltinRegistryFunction) {
    if (value instanceof String) {
      // String is a special case, as it is treated like a skylark value but cannot be subclassed
      // in java. Callable functions which represent methods on string objects must thus be given
      // the string 'self' object.
      return true;
    }
    if (globalBuiltinRegistryFunction != null && !isNamespace(value.getClass())) {
      // Non-namespace objects which have registered static functions in the global builtin registry
      // need to have the instance object passed to the method invocation.
      // TODO(cparsons): Global builtin registry functions are going away, so this use-case is
      // deprecated.
      return true;
    }
    return false;
  }

  /**
   * Call a method depending on the type of an object it is called on.
   *
   * @param positionalArgs positional arguments to pass to the method
   * @param call the original expression that caused this call, needed for rules especially
   */
  private Object invokeObjectMethod(
      Object value,
      String method,
      @Nullable BaseFunction function,
      ImmutableList<Object> positionalArgs,
      ImmutableMap<String, Object> keyWordArgs,
      FuncallExpression call,
      Environment env)
      throws EvalException, InterruptedException {
    Location location = call.getLocation();
    if (function != null) {
      return function.call(positionalArgs, keyWordArgs, call, env);
    }
    Object fieldValue =
        (value instanceof ClassObject) ? ((ClassObject) value).getValue(method) : null;
    if (fieldValue != null) {
      if (!(fieldValue instanceof BaseFunction)) {
        throw new EvalException(
            location, String.format("struct field '%s' is not a function", method));
      }
      function = (BaseFunction) fieldValue;
      return function.call(positionalArgs, keyWordArgs, call, env);
    } else {
      // When calling a Java method, the name is not in the Environment,
      // so evaluating 'function' would fail.
      Class<?> objClass;
      Object obj;
      if (value instanceof Class<?>) {
        // Static call
        obj = null;
        objClass = (Class<?>) value;
      } else if (value instanceof String) {
        // String is special-cased, since it can't be subclassed. Methods on strings defer
        // to StringModule.
        obj = StringModule.INSTANCE;
        objClass = StringModule.class;
      } else {
        obj = value;
        objClass = value.getClass();
      }
      Pair<MethodDescriptor, List<Object>> javaMethod =
          call.findJavaMethod(objClass, method, positionalArgs, keyWordArgs, env);
      if (javaMethod.first.isStructField()) {
        // Not a method but a callable attribute
        Object func;
        try {
          func = javaMethod.first.invoke(obj);
        } catch (IllegalAccessException e) {
          throw new EvalException(getLocation(), "method invocation failed: " + e);
        } catch (InvocationTargetException e) {
          if (e.getCause() instanceof FuncallException) {
            throw new EvalException(getLocation(), e.getCause().getMessage());
          } else if (e.getCause() != null) {
            Throwables.throwIfInstanceOf(e.getCause(), InterruptedException.class);
            throw new EvalExceptionWithJavaCause(getLocation(), e.getCause());
          } else {
            // This is unlikely to happen
            throw new EvalException(getLocation(), "method invocation failed: " + e);
          }
        }
        return callFunction(func, env);
      }
      return javaMethod.first.call(obj, javaMethod.second.toArray(), location, env);
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
      } else if (arg.isStar()) { // expand the starArg
        if (!(value instanceof Iterable)) {
          throw new EvalException(
              getLocation(),
              "argument after * must be an iterable, not " + EvalUtils.getDataTypeName(value));
        }
        posargs.addAll((Iterable<Object>) value);
      } else if (arg.isStarStar()) { // expand the kwargs
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
    String method = dot.getField().getName();
    @Nullable
    BaseFunction function = Runtime.getBuiltinRegistry().getFunction(objValue.getClass(), method);
    ImmutableList.Builder<Object> posargs = new ImmutableList.Builder<>();
    if (includeSelfAsArg(objValue, function)) {
      posargs.add(objValue);
    }
    // We copy this into an ImmutableMap in the end, but we can't use an ImmutableMap.Builder, or
    // we'd still have to have a HashMap on the side for the sake of properly handling duplicates.
    Map<String, Object> kwargs = new LinkedHashMap<>();
    evalArguments(posargs, kwargs, env);
    return invokeObjectMethod(
        objValue, method, function, posargs.build(), ImmutableMap.copyOf(kwargs), this, env);
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
    BaseFunction function = checkCallable(env.getSemantics(), funcValue, getLocation());
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

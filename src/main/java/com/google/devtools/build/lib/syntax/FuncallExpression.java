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

import static com.google.devtools.build.lib.syntax.compiler.ByteCodeUtils.append;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException.EvalExceptionWithJavaCause;
import com.google.devtools.build.lib.syntax.Runtime.NoneType;
import com.google.devtools.build.lib.syntax.compiler.ByteCodeMethodCalls;
import com.google.devtools.build.lib.syntax.compiler.ByteCodeUtils;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo.AstAccessors;
import com.google.devtools.build.lib.syntax.compiler.NewObject;
import com.google.devtools.build.lib.syntax.compiler.Variable.InternalVariable;
import com.google.devtools.build.lib.syntax.compiler.VariableScope;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.StringUtilities;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;
import net.bytebuddy.description.type.TypeDescription;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.Removal;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.assign.TypeCasting;
import net.bytebuddy.implementation.bytecode.constant.TextConstant;

/**
 * Syntax node for a function call expression.
 */
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
                    SkylarkCallable callable = getAnnotationFromParentClass(method);
                    if (callable == null) {
                      continue;
                    }
                    Preconditions.checkArgument(
                        callable.parameters().length == 0 || !callable.structField(),
                        "Method "
                            + method
                            + " was annotated with both structField and parameters.");
                    if (callable.parameters().length > 0 || callable.mandatoryPositionals() >= 0) {
                      int nbArgs =
                          callable.parameters().length
                              + Math.max(0, callable.mandatoryPositionals());
                      Preconditions.checkArgument(
                          nbArgs == method.getParameterTypes().length,
                          "Method "
                              + method
                              + " was annotated for "
                              + nbArgs
                              + " arguments "
                              + "but accept only "
                              + method.getParameterTypes().length
                              + " arguments.");
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

  /**
   * Returns a map of methods and corresponding SkylarkCallable annotations of the methods of the
   * classObj class reachable from Skylark.
   */
  public static ImmutableMap<Method, SkylarkCallable> collectSkylarkMethodsWithAnnotation(
      Class<?> classObj) {
    ImmutableMap.Builder<Method, SkylarkCallable> methodMap = ImmutableMap.builder();
    for (Method method : classObj.getMethods()) {
      // Synthetic methods lead to false multiple matches
      if (!method.isSynthetic()) {
        SkylarkCallable annotation = getAnnotationFromParentClass(classObj, method);
        if (annotation != null) {
          methodMap.put(method, annotation);
        }
      }
    }
    return methodMap.build();
  }

  /**
   * Returns the {@link SkylarkCallable} annotation for the given method, if it exists, and
   * null otherwise. The method must be declared in {@code classObj} or one of its base classes
   * or interfaces. The first annotation of an overridden version of the method that is found
   * will be returned, starting with {@code classObj} and following its base classes and
   * interfaces recursively.
   */
  @Nullable
  public static SkylarkCallable getAnnotationFromParentClass(Class<?> classObj, Method method) {
    boolean keepLooking = false;
    try {
      Method superMethod = classObj.getMethod(method.getName(), method.getParameterTypes());
      if (classObj.isAnnotationPresent(SkylarkModule.class)
          && superMethod.isAnnotationPresent(SkylarkCallable.class)) {
        return superMethod.getAnnotation(SkylarkCallable.class);
      } else {
        keepLooking = true;
      }
    } catch (NoSuchMethodException e) {
      // The class might not have the specified method, so an exceptions is OK.
      keepLooking = true;
    }
    if (keepLooking) {
      if (classObj.getSuperclass() != null) {
        SkylarkCallable annotation =
            getAnnotationFromParentClass(classObj.getSuperclass(), method);
        if (annotation != null) {
          return annotation;
        }
      }
      for (Class<?> interfaceObj : classObj.getInterfaces()) {
        SkylarkCallable annotation = getAnnotationFromParentClass(interfaceObj, method);
        if (annotation != null) {
          return annotation;
        }
      }
    }
    return null;
  }

  /**
   * Convenience version of {@code getAnnotationsFromParentClass(Class, Method)} that uses
   * the declaring class of the method.
   */
  @Nullable
  public static SkylarkCallable getAnnotationFromParentClass(Method method) {
    return getAnnotationFromParentClass(method.getDeclaringClass(), method);
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

  @Nullable private final Expression obj;

  private final Identifier func;

  private final List<Argument.Passed> args;

  private final int numPositionalArgs;

  /**
   * Note: the grammar definition restricts the function value in a function
   * call expression to be a global identifier; however, the representation of
   * values in the interpreter is flexible enough to allow functions to be
   * arbitrary expressions. In any case, the "func" expression is always
   * evaluated, so functions and variables share a common namespace.
   */
  public FuncallExpression(@Nullable Expression obj, Identifier func,
                           List<Argument.Passed> args) {
    this.obj = obj;
    this.func = func;
    this.args = args; // we assume the parser validated it with Argument#validateFuncallArguments()
    this.numPositionalArgs = countPositionalArguments();
  }

  /**
   * Note: the grammar definition restricts the function value in a function
   * call expression to be a global identifier; however, the representation of
   * values in the interpreter is flexible enough to allow functions to be
   * arbitrary expressions. In any case, the "func" expression is always
   * evaluated, so functions and variables share a common namespace.
   */
  public FuncallExpression(Identifier func, List<Argument.Passed> args) {
    this(null, func, args);
  }

  /**
   * Returns the number of positional arguments.
   */
  private int countPositionalArguments() {
    int num = 0;
    for (Argument.Passed arg : args) {
      if (arg.isPositional()) {
        num++;
      }
    }
    return num;
  }

  /**
   * Returns the function expression.
   */
  public Identifier getFunction() {
    return func;
  }

  /**
   * Returns the object the function called on.
   * It's null if the function is not called on an object.
   */
  public Expression getObject() {
    return obj;
  }

  /**
   * Returns an (immutable, ordered) list of function arguments. The first n are
   * positional and the remaining ones are keyword args, where n =
   * getNumPositionalArguments().
   */
  public List<Argument.Passed> getArguments() {
    return Collections.unmodifiableList(args);
  }

  /**
   * Returns the number of arguments which are positional; the remainder are
   * keyword arguments.
   */
  public int getNumPositionalArguments() {
    return numPositionalArgs;
  }

  private String functionName() {
    return "function " + func.getName();
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    if (obj != null) {
      sb.append(obj).append(".");
    }
    sb.append(func);
    Printer.printList(sb, args, "(", ", ", ")", /* singletonTerminator */ null,
        Printer.SUGGESTED_CRITICAL_LIST_ELEMENTS_COUNT,
        Printer.SUGGESTED_CRITICAL_LIST_ELEMENTS_STRING_LENGTH);
    return sb.toString();
  }

  /**
   * Returns the list of Skylark callable Methods of objClass with the given name
   * and argument number.
   */
  public static List<MethodDescriptor> getMethods(Class<?> objClass, String methodName,
      Location loc) throws EvalException {
    try {
      return methodCache.get(objClass).get(methodName);
    } catch (ExecutionException e) {
      throw new EvalException(loc, "Method invocation failed: " + e);
    }
  }

  /**
   * Returns a set of the Skylark name of all Skylark callable methods for object of type {@code
   * objClass}.
   */
  public static Set<String> getMethodNames(Class<?> objClass) throws ExecutionException {
    return methodCache.get(objClass).keySet();
  }

  static Object callMethod(MethodDescriptor methodDescriptor, String methodName, Object obj,
      Object[] args, Location loc, Environment env) throws EvalException {
    try {
      Method method = methodDescriptor.getMethod();
      if (obj == null && !Modifier.isStatic(method.getModifiers())) {
        throw new EvalException(loc, "Method '" + methodName + "' is not static");
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
          throw new EvalException(loc,
              "Method invocation returned None, please contact Skylark developers: " + methodName
              + Printer.listString(ImmutableList.copyOf(args), "(", ", ", ")", null));
        }
      }
      // TODO(bazel-team): get rid of this, by having everyone use the Skylark data structures
      result = SkylarkType.convertToSkylark(result, method, env);
      if (result != null && !EvalUtils.isSkylarkAcceptable(result.getClass())) {
        throw new EvalException(loc, Printer.format(
            "Method '%s' returns an object of invalid type %r", methodName, result.getClass()));
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
        throw new EvalException(loc, "Method invocation failed: " + e);
      }
    }
  }

  // TODO(bazel-team): If there's exactly one usable method, this works. If there are multiple
  // matching methods, it still can be a problem. Figure out how the Java compiler does it
  // exactly and copy that behaviour.
  // Throws an EvalException when it cannot find a matching function.
  private Pair<MethodDescriptor, List<Object>> findJavaMethod(
      Class<?> objClass, String methodName, List<Object> args, Map<String, Object> kwargs)
      throws EvalException {
    Pair<MethodDescriptor, List<Object>> matchingMethod = null;
    List<MethodDescriptor> methods = getMethods(objClass, methodName, getLocation());
    ArgumentListConversionResult argumentListConversionResult = null;
    if (methods != null) {
      for (MethodDescriptor method : methods) {
        if (!method.getAnnotation().structField()) {
          argumentListConversionResult = convertArgumentList(args, kwargs, method);
          if (argumentListConversionResult.getArguments() != null) {
            if (matchingMethod == null) {
              matchingMethod =
                  new Pair<MethodDescriptor, List<Object>>(
                      method, argumentListConversionResult.getArguments());
            } else {
              throw new EvalException(
                  getLocation(),
                  String.format(
                      "Type %s has multiple matches for %s",
                      EvalUtils.getDataTypeNameFromClass(objClass), formatMethod(args, kwargs)));
            }
          }
        }
      }
    }
    if (matchingMethod == null) {
      String errorMessage;
      if (argumentListConversionResult == null || argumentListConversionResult.getError() == null) {
        errorMessage =
            String.format(
                "Type %s has no %s",
                EvalUtils.getDataTypeNameFromClass(objClass), formatMethod(args, kwargs));

      } else {
        errorMessage =
            String.format(
                "%s (in %s of %s).",
                argumentListConversionResult.getError(),
                formatMethod(args, kwargs),
                EvalUtils.getDataTypeNameFromClass(objClass));
      }
      throw new EvalException(getLocation(), errorMessage);
    }
    return matchingMethod;
  }

  private static SkylarkType getType(Param param) {
    SkylarkType type =
        param.generic1() != Object.class
            ? SkylarkType.of(param.type(), param.generic1())
            : SkylarkType.of(param.type());
    return type;
  }

  /**
   * Constructs the parameters list to actually pass to the method, filling with default values if
   * any. If the type does not match, returns null and fills in methodTypeMatchError with the
   * appropriate message.
   */
  private ArgumentListConversionResult convertArgumentList(
      List<Object> args, Map<String, Object> kwargs, MethodDescriptor method) {
    ImmutableList.Builder<Object> builder = ImmutableList.builder();
    Class<?>[] params = method.getMethod().getParameterTypes();
    SkylarkCallable callable = method.getAnnotation();
    int mandatoryPositionals = callable.mandatoryPositionals();
    if (mandatoryPositionals < 0) {
      if (callable.parameters().length > 0) {
        mandatoryPositionals = 0;
      } else {
        mandatoryPositionals = params.length;
      }
    }
    if (mandatoryPositionals > args.size()
        || args.size() > mandatoryPositionals + callable.parameters().length) {
      return ArgumentListConversionResult.fromError("Too many arguments");
    }
    // First process the legacy positional parameters.
    int i = 0;
    if (mandatoryPositionals > 0) {
      for (Class<?> param : params) {
        Object value = args.get(i);
        if (!param.isAssignableFrom(value.getClass())) {
          return ArgumentListConversionResult.fromError(
              String.format(
                  "Cannot convert parameter at position %d to type %s", i, param.toString()));
        }
        builder.add(value);
        i++;
        if (mandatoryPositionals >= 0 && i >= mandatoryPositionals) {
          // Stops for specified parameters instead.
          break;
        }
      }
    }

    // Then the parameters specified in callable.parameters()
    Set<String> keys = new HashSet<>(kwargs.keySet());
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
                  "Cannot convert parameter '%s' to type %s", param.name(), type.toString()));
        }
        i++;
      } else if (param.named() && keys.remove(param.name())) {
        // Named parameters
        value = kwargs.get(param.name());
        if (!type.contains(value)) {
          return ArgumentListConversionResult.fromError(
              String.format(
                  "Cannot convert parameter '%s' to type %s", param.name(), type.toString()));
        }
      } else {
        // Use default value
        if (param.defaultValue().isEmpty()) {
          return ArgumentListConversionResult.fromError(
              String.format("Parameter '%s' has no default value", param.name()));
        }
        value = SkylarkSignatureProcessor.getDefaultValue(param, null);
      }
      builder.add(value);
      if (!param.noneable() && value instanceof NoneType) {
        return ArgumentListConversionResult.fromError(
            String.format("Parameter '%s' cannot be None", param.name()));
      }
    }
    if (i < args.size() || !keys.isEmpty()) {
      return ArgumentListConversionResult.fromError("Too many arguments");
    }
    return ArgumentListConversionResult.fromArgumentList(builder.build());
  }

  private String formatMethod(List<Object> args, Map<String, Object> kwargs) {
    StringBuilder sb = new StringBuilder();
    sb.append(functionName()).append("(");
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
   * A {@link StackManipulation} invoking addKeywordArg.
   *
   * <p>Kept close to the definition of the method to avoid reflection errors when changing it.
   */
  private static final StackManipulation addKeywordArg =
      ByteCodeUtils.invoke(
          FuncallExpression.class,
          "addKeywordArg",
          Map.class,
          String.class,
          Object.class,
          ImmutableList.Builder.class);

  /**
   * Add one argument to the keyword map, registering a duplicate in case of conflict.
   *
   * <p>public for reflection by the compiler and calls from compiled functions
   */
  public static void addKeywordArg(
      Map<String, Object> kwargs,
      String name,
      Object value,
      ImmutableList.Builder<String> duplicates) {
    if (kwargs.put(name, value) != null) {
      duplicates.add(name);
    }
  }

  /**
   * A {@link StackManipulation} invoking addKeywordArgs.
   * <p>Kept close to the definition of the method to avoid reflection errors when changing it.
   */
  private static final StackManipulation addKeywordArgs =
      ByteCodeUtils.invoke(
          FuncallExpression.class,
          "addKeywordArgs",
          Map.class,
          Object.class,
          ImmutableList.Builder.class,
          Location.class);

  /**
   * Add multiple arguments to the keyword map (**kwargs), registering duplicates
   *
   * <p>public for reflection by the compiler and calls from compiled functions
   */
  public static void addKeywordArgs(
      Map<String, Object> kwargs,
      Object items,
      ImmutableList.Builder<String> duplicates,
      Location location)
      throws EvalException {
    if (!(items instanceof Map<?, ?>)) {
      throw new EvalException(
          location,
          "Argument after ** must be a dictionary, not " + EvalUtils.getDataTypeName(items));
    }
    for (Map.Entry<?, ?> entry : ((Map<?, ?>) items).entrySet()) {
      if (!(entry.getKey() instanceof String)) {
        throw new EvalException(
            location, "Keywords must be strings, not " + EvalUtils.getDataTypeName(entry.getKey()));
      }
      addKeywordArg(kwargs, (String) entry.getKey(), entry.getValue(), duplicates);
    }
  }

  /**
   * A {@link StackManipulation} invoking checkCallable.
   * <p>Kept close to the definition of the method to avoid reflection errors when changing it.
   */
  private static final StackManipulation checkCallable =
      ByteCodeUtils.invoke(FuncallExpression.class, "checkCallable", Object.class, Location.class);

  /**
   * Checks whether the given object is a {@link BaseFunction}.
   *
   * <p>Public for reflection by the compiler and access from generated byte code.
   *
   * @throws EvalException If not a BaseFunction.
   */
  public static BaseFunction checkCallable(Object functionValue, Location location)
      throws EvalException {
    if (functionValue instanceof BaseFunction) {
      return (BaseFunction) functionValue;
    } else {
      throw new EvalException(
          location, "'" + EvalUtils.getDataTypeName(functionValue) + "' object is not callable");
    }
  }

  /**
   * A {@link StackManipulation} invoking checkDuplicates.
   * <p>Kept close to the definition of the method to avoid reflection errors when changing it.
   */
  private static final StackManipulation checkDuplicates =
      ByteCodeUtils.invoke(
          FuncallExpression.class,
          "checkDuplicates",
          ImmutableList.Builder.class,
          String.class,
          Location.class);

  /**
   * Check the list from the builder and report an {@link EvalException} if not empty.
   *
   * <p>public for reflection by the compiler and calls from compiled functions
   */
  public static void checkDuplicates(
      ImmutableList.Builder<String> duplicates, String function, Location location)
      throws EvalException {
    List<String> dups = duplicates.build();
    if (!dups.isEmpty()) {
      throw new EvalException(
          location,
          "duplicate keyword"
              + (dups.size() > 1 ? "s" : "")
              + " '"
              + Joiner.on("', '").join(dups)
              + "' in call to "
              + function);
    }
  }

  /**
   * A {@link StackManipulation} invoking invokeObjectMethod.
   * <p>Kept close to the definition of the method to avoid reflection errors when changing it.
   */
  private static final StackManipulation invokeObjectMethod =
      ByteCodeUtils.invoke(
          FuncallExpression.class,
          "invokeObjectMethod",
          String.class,
          ImmutableList.class,
          ImmutableMap.class,
          FuncallExpression.class,
          Environment.class);

  /**
   * Call a method depending on the type of an object it is called on.
   *
   * <p>Public for reflection by the compiler and access from generated byte code.
   *
   * @param positionals The first object is expected to be the object the method is called on.
   * @param call the original expression that caused this call, needed for rules especially
   */
  public static Object invokeObjectMethod(
      String method,
      ImmutableList<Object> positionals,
      ImmutableMap<String, Object> keyWordArgs,
      FuncallExpression call,
      Environment env)
      throws EvalException, InterruptedException {
    Location location = call.getLocation();
    Object value = positionals.get(0);
    ImmutableList<Object> positionalArgs = positionals.subList(1, positionals.size());
    BaseFunction function = Runtime.getFunction(EvalUtils.getSkylarkType(value.getClass()), method);
    if (function != null) {
      if (!isNamespace(value.getClass())) {
        // Use self as an implicit parameter in front.
        positionalArgs = positionals;
      }
      return function.call(
          positionalArgs, ImmutableMap.<String, Object>copyOf(keyWordArgs), call, env);
    } else if (value instanceof ClassObject) {
      Object fieldValue = ((ClassObject) value).getValue(method);
      if (fieldValue == null) {
        throw new EvalException(location, String.format("struct has no method '%s'", method));
      }
      if (!(fieldValue instanceof BaseFunction)) {
        throw new EvalException(
            location, String.format("struct field '%s' is not a function", method));
      }
      function = (BaseFunction) fieldValue;
      return function.call(
          positionalArgs, ImmutableMap.<String, Object>copyOf(keyWordArgs), call, env);
    } else {
      // When calling a Java method, the name is not in the Environment,
      // so evaluating 'func' would fail.
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
          call.findJavaMethod(objClass, method, positionalArgs, keyWordArgs);
      return callMethod(javaMethod.first, method, obj, javaMethod.second.toArray(), location, env);
    }
  }

  @SuppressWarnings("unchecked")
  private void evalArguments(ImmutableList.Builder<Object> posargs, Map<String, Object> kwargs,
      Environment env)
      throws EvalException, InterruptedException {
    ImmutableList.Builder<String> duplicates = new ImmutableList.Builder<>();
    // Iterate over the arguments. We assume all positional arguments come before any keyword
    // or star arguments, because the argument list was already validated by
    // Argument#validateFuncallArguments, as called by the Parser,
    // which should be the only place that build FuncallExpression-s.
    for (Argument.Passed arg : args) {
      Object value = arg.getValue().eval(env);
      if (arg.isPositional()) {
        posargs.add(value);
      } else if (arg.isStar()) {  // expand the starArg
        if (value instanceof Iterable) {
          posargs.addAll((Iterable<Object>) value);
        }
      } else if (arg.isStarStar()) {  // expand the kwargs
        addKeywordArgs(kwargs, value, duplicates, getLocation());
      } else {
        addKeywordArg(kwargs, arg.getName(), value, duplicates);
      }
    }
    checkDuplicates(duplicates, func.getName(), getLocation());
  }

  @VisibleForTesting
  public static boolean isNamespace(Class<?> classObject) {
    return classObject.isAnnotationPresent(SkylarkModule.class)
        && classObject.getAnnotation(SkylarkModule.class).namespace();
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
    return (obj != null) ? invokeObjectMethod(env) : invokeGlobalFunction(env);
  }

  /**
   * Invokes obj.func() and returns the result.
   */
  private Object invokeObjectMethod(Environment env) throws EvalException, InterruptedException {
    Object objValue = obj.eval(env);
    ImmutableList.Builder<Object> posargs = new ImmutableList.Builder<>();
    posargs.add(objValue);
    // We copy this into an ImmutableMap in the end, but we can't use an ImmutableMap.Builder, or
    // we'd still have to have a HashMap on the side for the sake of properly handling duplicates.
    Map<String, Object> kwargs = new LinkedHashMap<>();
    evalArguments(posargs, kwargs, env);
    return invokeObjectMethod(
        func.getName(), posargs.build(), ImmutableMap.<String, Object>copyOf(kwargs), this, env);
  }

  /**
   * Invokes func() and returns the result.
   */
  private Object invokeGlobalFunction(Environment env) throws EvalException, InterruptedException {
    Object funcValue = func.eval(env);
    ImmutableList.Builder<Object> posargs = new ImmutableList.Builder<>();
    // We copy this into an ImmutableMap in the end, but we can't use an ImmutableMap.Builder, or
    // we'd still have to have a HashMap on the side for the sake of properly handling duplicates.
    Map<String, Object> kwargs = new HashMap<>();
    BaseFunction function = checkCallable(funcValue, getLocation());
    evalArguments(posargs, kwargs, env);
    return function.call(posargs.build(), ImmutableMap.<String, Object>copyOf(kwargs), this, env);
  }

  /**
   * Returns the value of the argument 'name' (or null if there is none).
   * This function is used to associate debugging information to rules created by skylark "macros".
   */
  @Nullable
  public String getNameArg() {
    for (Argument.Passed arg : args) {
      if (arg != null) {
        String name = arg.getName();
        if (name != null && name.equals("name")) {
          Expression expr = arg.getValue();
          return (expr != null && expr instanceof StringLiteral)
              ? ((StringLiteral) expr).getValue() : null;
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
  void validate(ValidationEnvironment env) throws EvalException {
    for (Argument.Passed arg : args) {
      arg.getValue().validate(env);
    }

    if (obj != null) {
      obj.validate(env);
    } else if (!env.hasSymbolInEnvironment(func.getName())) {
      throw new EvalException(getLocation(),
          String.format("function '%s' does not exist", func.getName()));
    }
  }

  @Override
  protected boolean isNewScope() {
    return true;
  }

  @Override
  ByteCodeAppender compile(VariableScope scope, DebugInfo debugInfo) throws EvalException {
    AstAccessors debugAccessors = debugInfo.add(this);
    List<ByteCodeAppender> code = new ArrayList<>();
    if (obj != null) {
      compileObjectMethodCall(scope, debugInfo, debugAccessors, code);
    } else {
      compileGlobalFunctionCall(scope, debugInfo, debugAccessors, code);
    }
    return ByteCodeUtils.compoundAppender(code);
  }

  /**
   * Add code that compiles the argument expressions.
   *
   * <p>The byte code leaves the arguments on the stack in order of:
   * positional arguments, key word arguments, this FuncallExpression, Environment
   * This is the order required by {@link #invokeObjectMethod} and
   *  {@link BaseFunction#call(List, Map, FuncallExpression, Environment)}.
   */
  private void compileArguments(
      VariableScope scope,
      DebugInfo debugInfo,
      AstAccessors debugAccessors,
      List<ByteCodeAppender> code)
      throws EvalException {
    InternalVariable positionalsBuilder = scope.freshVariable(ImmutableList.Builder.class);
    append(code, ByteCodeMethodCalls.BCImmutableList.builder);
    code.add(positionalsBuilder.store());

    InternalVariable keyWordArgs = scope.freshVariable(Map.class);
    append(code, NewObject.fromConstructor(HashMap.class).arguments());
    code.add(keyWordArgs.store());

    InternalVariable duplicatesBuilder =
        scope.freshVariable(new TypeDescription.ForLoadedType(ImmutableList.Builder.class));
    append(code, ByteCodeMethodCalls.BCImmutableList.builder);
    code.add(duplicatesBuilder.store());

    StackManipulation builderAdd =
        new StackManipulation.Compound(
            ByteCodeMethodCalls.BCImmutableList.Builder.add, Removal.SINGLE);

    // add an object the function is called on first
    if (obj != null) {
      append(code, positionalsBuilder.load());
      code.add(obj.compile(scope, debugInfo));
      append(code, builderAdd);
    }
    // add all arguments to their respective builder/map
    for (Argument.Passed arg : args) {
      ByteCodeAppender value = arg.getValue().compile(scope, debugInfo);
      if (arg.isPositional()) {
        append(code, positionalsBuilder.load());
        code.add(value);
        append(code, builderAdd);
      } else if (arg.isStar()) {
        // expand the starArg by adding all it's elements to the builder
        append(code, positionalsBuilder.load());
        code.add(value);
        append(
            code,
            TypeCasting.to(new TypeDescription.ForLoadedType(Iterable.class)),
            ByteCodeMethodCalls.BCImmutableList.Builder.addAll,
            Removal.SINGLE);
      } else if (arg.isStarStar()) {
        append(code, keyWordArgs.load());
        code.add(value);
        append(code, duplicatesBuilder.load(), debugAccessors.loadLocation, addKeywordArgs);
      } else {
        append(code, keyWordArgs.load(), new TextConstant(arg.getName()));
        code.add(value);
        append(code, duplicatesBuilder.load(), addKeywordArg);
      }
    }
    append(
        code,
        // check for duplicates in the key word arguments
        duplicatesBuilder.load(),
        new TextConstant(func.getName()),
        debugAccessors.loadLocation,
        checkDuplicates,
        // load the arguments in the correct order for invokeObjectMethod and BaseFunction.call
        positionalsBuilder.load(),
        ByteCodeMethodCalls.BCImmutableList.Builder.build,
        keyWordArgs.load(),
        ByteCodeMethodCalls.BCImmutableMap.copyOf,
        debugAccessors.loadAstNode,
        TypeCasting.to(new TypeDescription.ForLoadedType(FuncallExpression.class)),
        scope.loadEnvironment());
  }

  private void compileObjectMethodCall(
      VariableScope scope,
      DebugInfo debugInfo,
      AstAccessors debugAccessors,
      List<ByteCodeAppender> code)
      throws EvalException {
    append(code, new TextConstant(func.getName()));
    compileArguments(scope, debugInfo, debugAccessors, code);
    append(code, invokeObjectMethod);
  }

  private void compileGlobalFunctionCall(
      VariableScope scope,
      DebugInfo debugInfo,
      AstAccessors debugAccessors,
      List<ByteCodeAppender> code)
      throws EvalException {
    code.add(func.compile(scope, debugInfo));
    append(code, debugAccessors.loadLocation, checkCallable);
    compileArguments(scope, debugInfo, debugAccessors, code);
    append(code, BaseFunction.call);
  }
}

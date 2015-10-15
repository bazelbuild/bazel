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
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.EvalException.EvalExceptionWithJavaCause;
import com.google.devtools.build.lib.util.StringUtilities;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import javax.annotation.Nullable;

/**
 * Syntax node for a function call expression.
 */
public final class FuncallExpression extends Expression {

  private static enum ArgConversion {
    FROM_SKYLARK,
    TO_SKYLARK,
    NO_CONVERSION
  }

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
      .build(new CacheLoader<Class<?>, Map<String, List<MethodDescriptor>>>() {

        @Override
        public Map<String, List<MethodDescriptor>> load(Class<?> key) throws Exception {
          Map<String, List<MethodDescriptor>> methodMap = new HashMap<>();
          for (Method method : key.getMethods()) {
            // Synthetic methods lead to false multiple matches
            if (method.isSynthetic()) {
              continue;
            }
            SkylarkCallable callable = getAnnotationFromParentClass(
                  method.getDeclaringClass(), method);
            if (callable == null) {
              continue;
            }
            String name = callable.name();
            if (name.isEmpty()) {
              name = StringUtilities.toPythonStyleFunctionName(method.getName());
            }
            String signature = name + "#" + method.getParameterTypes().length;
            if (methodMap.containsKey(signature)) {
              methodMap.get(signature).add(new MethodDescriptor(method, callable));
            } else {
              methodMap.put(signature, Lists.newArrayList(new MethodDescriptor(method, callable)));
            }
          }
          return ImmutableMap.copyOf(methodMap);
        }
      });

  /**
   * Returns a map of methods and corresponding SkylarkCallable annotations
   * of the methods of the classObj class reachable from Skylark.
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

  @Nullable
  private static SkylarkCallable getAnnotationFromParentClass(Class<?> classObj, Method method) {
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
    String name = func.getName();
    if (name.equals("$slice")) {
      return "operator [:]";
    } else if (name.equals("$index")) {
      return "operator []";
    } else {
      return "function " + name;
    }
  }

  @Override
  public String toString() {
    if (func.getName().equals("$slice")) {
      return obj + "[" + args.get(0) + ":" + args.get(1) + "]";
    }
    if (func.getName().equals("$index")) {
      return obj + "[" + args.get(0) + "]";
    }
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
  public static List<MethodDescriptor> getMethods(Class<?> objClass, String methodName, int argNum,
      Location loc) throws EvalException {
    try {
      return methodCache.get(objClass).get(methodName + "#" + argNum);
    } catch (ExecutionException e) {
      throw new EvalException(loc, "Method invocation failed: " + e);
    }
  }

  /**
   * Returns the list of the Skylark name of all Skylark callable methods.
   */
  public static List<String> getMethodNames(Class<?> objClass)
      throws ExecutionException {
    List<String> names = new ArrayList<>();
    for (List<MethodDescriptor> methods : methodCache.get(objClass).values()) {
      for (MethodDescriptor method : methods) {
        // TODO(bazel-team): store the Skylark name in the MethodDescriptor. 
        String name = method.annotation.name();
        if (name.isEmpty()) {
          name = StringUtilities.toPythonStyleFunctionName(method.method.getName());
        }
        names.add(name);
      }
    }
    return names;
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
  private MethodDescriptor findJavaMethod(
      Class<?> objClass, String methodName, List<Object> args) throws EvalException {
    MethodDescriptor matchingMethod = null;
    List<MethodDescriptor> methods = getMethods(objClass, methodName, args.size(), getLocation());
    if (methods != null) {
      for (MethodDescriptor method : methods) {
        Class<?>[] params = method.getMethod().getParameterTypes();
        int i = 0;
        boolean matching = true;
        for (Class<?> param : params) {
          if (!param.isAssignableFrom(args.get(i).getClass())) {
            matching = false;
            break;
          }
          i++;
        }
        if (matching) {
          if (matchingMethod == null) {
            matchingMethod = method;
          } else {
            throw new EvalException(
                getLocation(),
                String.format(
                    "Type %s has multiple matches for %s",
                    EvalUtils.getDataTypeNameFromClass(objClass),
                    formatMethod(args)));
          }
        }
      }
    }
    if (matchingMethod != null && !matchingMethod.getAnnotation().structField()) {
      return matchingMethod;
    }
    throw new EvalException(
        getLocation(),
        String.format(
            "Type %s has no %s",
            EvalUtils.getDataTypeNameFromClass(objClass),
            formatMethod(args)));
  }

  private String formatMethod(List<Object> args) {
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
    return sb.append(")").toString();
  }

  /**
   * Add one argument to the keyword map, registering a duplicate in case of conflict.
   */
  private void addKeywordArg(Map<String, Object> kwargs, String name,
      Object value,  ImmutableList.Builder<String> duplicates) {
    if (kwargs.put(name, value) != null) {
      duplicates.add(name);
    }
  }

  /**
   * Add multiple arguments to the keyword map (**kwargs), registering duplicates
   */
  private void addKeywordArgs(Map<String, Object> kwargs,
      Object items, ImmutableList.Builder<String> duplicates) throws EvalException {
    if (!(items instanceof Map<?, ?>)) {
      throw new EvalException(getLocation(),
          "Argument after ** must be a dictionary, not " + EvalUtils.getDataTypeName(items));
    }
    for (Map.Entry<?, ?> entry : ((Map<?, ?>) items).entrySet()) {
      if (!(entry.getKey() instanceof String)) {
        throw new EvalException(getLocation(),
            "Keywords must be strings, not " + EvalUtils.getDataTypeName(entry.getKey()));
      }
      addKeywordArg(kwargs, (String) entry.getKey(), entry.getValue(), duplicates);
    }
  }

  @SuppressWarnings("unchecked")
  private void evalArguments(ImmutableList.Builder<Object> posargs, Map<String, Object> kwargs,
      Environment env, BaseFunction function)
      throws EvalException, InterruptedException {
    ArgConversion conversion = getArgConversion(function);
    ImmutableList.Builder<String> duplicates = new ImmutableList.Builder<>();
    // Iterate over the arguments. We assume all positional arguments come before any keyword
    // or star arguments, because the argument list was already validated by
    // Argument#validateFuncallArguments, as called by the Parser,
    // which should be the only place that build FuncallExpression-s.
    for (Argument.Passed arg : args) {
      Object value = arg.getValue().eval(env);
      if (conversion == ArgConversion.FROM_SKYLARK) {
        value = SkylarkType.convertFromSkylark(value);
      } else if (conversion == ArgConversion.TO_SKYLARK) {
        // We try to auto convert the type if we can.
        value = SkylarkType.convertToSkylark(value, env);
        // We call into Skylark so we need to be sure that the caller uses the appropriate types.
        SkylarkType.checkTypeAllowedInSkylark(value, getLocation());
      } // else NO_CONVERSION
      if (arg.isPositional()) {
        posargs.add(value);
      } else if (arg.isStar()) {  // expand the starArg
        if (value instanceof Iterable) {
          posargs.addAll((Iterable<Object>) value);
        }
      } else if (arg.isStarStar()) {  // expand the kwargs
        addKeywordArgs(kwargs, value, duplicates);
      } else {
        addKeywordArg(kwargs, arg.getName(), value, duplicates);
      }
    }
    List<String> dups = duplicates.build();
    if (!dups.isEmpty()) {
      throw new EvalException(getLocation(),
          "duplicate keyword" + (dups.size() > 1 ? "s" : "") + " '"
          + Joiner.on("', '").join(dups)
          + "' in call to " + func);
    }
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
    // We copy this into an ImmutableMap in the end, but we can't use an ImmutableMap.Builder, or
    // we'd still have to have a HashMap on the side for the sake of properly handling duplicates.
    Map<String, Object> kwargs = new HashMap<>();

    // Strings, lists and dictionaries (maps) have functions that we want to use in
    // MethodLibrary.
    // For other classes, we can call the Java methods.
    BaseFunction function =
        Runtime.getFunction(EvalUtils.getSkylarkType(objValue.getClass()), func.getName());
    if (function != null) {
      if (!isNamespace(objValue.getClass())) {
        // Add self as an implicit parameter in front.
        posargs.add(objValue);
      }
      evalArguments(posargs, kwargs, env, function);
      return convertFromSkylark(
          function.call(posargs.build(), ImmutableMap.<String, Object>copyOf(kwargs), this, env),
          env);
    } else if (objValue instanceof ClassObject) {
      Object fieldValue = ((ClassObject) objValue).getValue(func.getName());
      if (fieldValue == null) {
        throw new EvalException(
            getLocation(), String.format("struct has no method '%s'", func.getName()));
      }
      if (!(fieldValue instanceof BaseFunction)) {
        throw new EvalException(
            getLocation(), String.format("struct field '%s' is not a function", func.getName()));
      }
      function = (BaseFunction) fieldValue;
      evalArguments(posargs, kwargs, env, function);
      return convertFromSkylark(
          function.call(posargs.build(), ImmutableMap.<String, Object>copyOf(kwargs), this, env),
          env);
    } else if (env.isSkylark()) {
      // Only allow native Java calls when using Skylark
      // When calling a Java method, the name is not in the Environment,
      // so evaluating 'func' would fail.
      evalArguments(posargs, kwargs, env, null);
      Class<?> objClass;
      Object obj;
      if (objValue instanceof Class<?>) {
        // Static call
        obj = null;
        objClass = (Class<?>) objValue;
      } else {
        obj = objValue;
        objClass = objValue.getClass();
      }
      String name = func.getName();
      ImmutableList<Object> args = posargs.build();
      MethodDescriptor method = findJavaMethod(objClass, name, args);
      if (!kwargs.isEmpty()) {
        throw new EvalException(
            func.getLocation(),
            String.format(
                "Keyword arguments are not allowed when calling a java method"
                + "\nwhile calling method '%s' for type %s",
                name, EvalUtils.getDataTypeNameFromClass(objClass)));
      }
      return callMethod(method, name, obj, args.toArray(), getLocation(), env);
    } else {
      throw new EvalException(
          getLocation(),
          String.format("%s is not defined on object of type '%s'", functionName(),
              EvalUtils.getDataTypeName(objValue)));
    }
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
    if ((funcValue instanceof BaseFunction)) {
      BaseFunction function = (BaseFunction) funcValue;
      evalArguments(posargs, kwargs, env, function);
      return convertFromSkylark(
          function.call(posargs.build(), ImmutableMap.<String, Object>copyOf(kwargs), this, env),
          env);
    } else {
      throw new EvalException(
          getLocation(), "'" + EvalUtils.getDataTypeName(funcValue) + "' object is not callable");
    }
  }

  protected Object convertFromSkylark(Object returnValue, Environment env) throws EvalException {
    EvalUtils.checkNotNull(this, returnValue);
    if (!env.isSkylark()) {
      // The call happens in the BUILD language. Note that accessing "BUILD language" functions in
      // Skylark should never happen.
      return SkylarkType.convertFromSkylark(returnValue);
    }
    return returnValue;
  }

  private ArgConversion getArgConversion(BaseFunction function) {
    if (function == null) {
      // It means we try to call a Java function.
      return ArgConversion.FROM_SKYLARK;
    }
    // If we call a UserDefinedFunction we call into Skylark. If we call from Skylark
    // the argument conversion is invariant, but if we call from the BUILD language
    // we might need an auto conversion.
    return function instanceof UserDefinedFunction
        ? ArgConversion.TO_SKYLARK : ArgConversion.NO_CONVERSION;
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
}

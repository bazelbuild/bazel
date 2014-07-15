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

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.syntax.EvalException.EvalExceptionWithJavaCause;
import com.google.devtools.build.lib.util.StringUtilities;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

/**
 * Syntax node for a function call expression.
 */
public final class FuncallExpression extends Expression {

  private static final LoadingCache<Class<?>, Map<String, List<Method>>> methodCache =
      CacheBuilder.newBuilder()
      .initialCapacity(10)
      .maximumSize(100)
      .build(new CacheLoader<Class<?>, Map<String, List<Method>>>() {

        @Override
        public Map<String, List<Method>> load(Class<?> key) throws Exception {
          Map<String, List<Method>> methodMap = new HashMap<>();
          for (Method method : key.getMethods()) {
            // Synthetic methods lead to false multiple matches
            if (!method.isSynthetic()
                && isAnnotationPresentInParentClass(method.getDeclaringClass(), method)) {
              String signature = StringUtilities.toPythonStyleFunctionName(method.getName())
                  + "#" + method.getParameterTypes().length;
              if (methodMap.containsKey(signature)) {
                methodMap.get(signature).add(method);
              } else {
                methodMap.put(signature, Lists.newArrayList(method));
              }
            }
          }
          return ImmutableMap.copyOf(methodMap);
        }
      });

  /**
   * Returns a map of methods and corresponding SkylarkCallable annotations
   * of the methods of the classObj class reachable from Skylark.
   */
  public static Map<Method, SkylarkCallable> collectSkylarkMethodsWithAnnotation(
      Class<?> classObj) {
    Map<Method, SkylarkCallable> methodMap = new HashMap<>();
    for (Method method : classObj.getMethods()) {
      // Synthetic methods lead to false multiple matches
      if (!method.isSynthetic()) {
        SkylarkCallable annotation = getAnnotationFromParentClass(classObj, method);
        if (annotation != null) {
          methodMap.put(method, annotation);
        }
      }
    }
    return methodMap;
  }

  private static boolean isAnnotationPresentInParentClass(Class<?> classObj, Method method) {
    return getAnnotationFromParentClass(classObj, method) != null;
  }

  private static SkylarkCallable getAnnotationFromParentClass(Class<?> classObj, Method method) {
    boolean keepLooking = false;
    try {
      Method superMethod = classObj.getMethod(method.getName(), method.getParameterTypes());
      if (classObj.isAnnotationPresent(SkylarkBuiltin.class)
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
        SkylarkCallable annotation = getAnnotationFromParentClass(classObj.getSuperclass(), method);
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

  private final Expression obj;

  private final Ident func;

  private final List<Argument> args;

  private final int numPositionalArgs;

  /**
   * Note: the grammar definition restricts the function value in a function
   * call expression to be a global identifier; however, the representation of
   * values in the interpreter is flexible enough to allow functions to be
   * arbitrary expressions. In any case, the "func" expression is always
   * evaluated, so functions and variables share a common namespace.
   */
  public FuncallExpression(Expression obj, Ident func,
                           List<Argument> args) {
    this.obj = obj;
    this.func = func;
    this.args = args;
    this.numPositionalArgs = countPositionalArguments();
  }

  /**
   * Note: the grammar definition restricts the function value in a function
   * call expression to be a global identifier; however, the representation of
   * values in the interpreter is flexible enough to allow functions to be
   * arbitrary expressions. In any case, the "func" expression is always
   * evaluated, so functions and variables share a common namespace.
   */
  public FuncallExpression(Ident func, List<Argument> args) {
    this(null, func, args);
  }

  /**
   * Returns the number of positional arguments.
   */
  private int countPositionalArguments() {
    int num = 0;
    for (Argument arg: args) {
      if (arg.isPositional()) {
        num++;
      }
    }
    return num;
  }

  /**
   * Returns the function expression.
   */
  public Expression getFunction() {
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
  public List<Argument> getArguments() {
    return Collections.unmodifiableList(args);
  }

  /**
   * Returns the number of arguments which are positional; the remainder are
   * keyword arguments.
   */
  public int getNumPositionalArguments() {
    return numPositionalArgs;
  }

  @Override
  public String toString() {
    return func + "(" + args + ")";
  }

  // TODO(bazel-team): If there's exactly one usable method, this works. If there are multiple
  // matching methods, it still can be a problem. Figure out how the Java compiler does it
  // exactly and copy that behaviour.
  // TODO(bazel-team): check if this and SkylarkBuiltInFunctions.createObject can be merged.
  private Object invokeJavaMethod(
      Object obj, Class<?> objClass, String methodName, List<Object> args) throws EvalException {
    try {
      Method matchingMethod = null;
      List<Method> methods = methodCache.get(objClass).get(methodName + "#" + args.size());
      if (methods != null) {
        for (Method method : methods) {
          Class<?>[] params = method.getParameterTypes();
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
              throw new EvalException(func.getLocation(),
                  "Multiple matching methods for " + formatMethod(methodName, args) +
                  " in " + getClassName(objClass));
            }
          }
        }
      }
      if (matchingMethod != null) {
        if (obj == null && !Modifier.isStatic(matchingMethod.getModifiers())) {
          throw new EvalException(func.getLocation(), "Method '" + methodName + "' is not static.");
        }
        // This happens when the interface is public but the implementation classes
        // have reduced visibility.
        matchingMethod.setAccessible(true);
        Object result = matchingMethod.invoke(obj, args.toArray());
        if (result != null && !EvalUtils.isSkylarkImmutable(result.getClass())) {
          throw new EvalException(func.getLocation(), "Method '" + methodName
              + "' returns a mutable object (type of " + EvalUtils.getDatatypeName(result) + ").");
        }
        if (result instanceof NestedSet<?>) {
          // This is probably the most terrible hack ever written. However this is the last place
          // where we can infer generic type information, so SkylarkNestedSets can remain safe.
          // Eventually we should cache these info too like we cache Methods, and probably do
          // something with Lists and Maps too.
          ParameterizedType t = (ParameterizedType) matchingMethod.getGenericReturnType();
          return new SkylarkNestedSet((Class<?>) t.getActualTypeArguments()[0],
              (NestedSet<?>) result);
        }
        return result;
      } else {
        throw new EvalException(func.getLocation(), "No matching method found for "
            + formatMethod(methodName, args) + " in " + getClassName(objClass));
      }
    } catch (IllegalAccessException e) {
      // TODO(bazel-team): Print a nice error message. Maybe the method exists
      // and an argument is missing or has the wrong type.
      throw new EvalException(func.getLocation(), "Method invocation failed: " + e);
    } catch (InvocationTargetException e) {
      if (e.getCause() instanceof FuncallException) {
        throw new EvalException(func.getLocation(), e.getCause().getMessage());
      } else if (e.getCause() != null) {
        throw new EvalExceptionWithJavaCause(func.getLocation(), e.getCause());
      } else {
        // This is unlikely to happen
        throw new EvalException(func.getLocation(), "Method invocation failed: " + e);
      }
    } catch (ExecutionException e) {
      throw new EvalException(func.getLocation(), "Method invocation failed: " + e);
    }
  }

  private String getClassName(Class<?> classObject) {
    if (classObject.getSimpleName().isEmpty()) {
      return classObject.getName();
    } else {
      return classObject.getSimpleName();
    }
  }

  private String formatMethod(String methodName, List<Object> args) {
    StringBuilder sb = new StringBuilder();
    sb.append(methodName).append("(");
    boolean first = true;
    for (Object obj : args) {
      if (!first) {
        sb.append(", ");
      }
      sb.append(EvalUtils.getDatatypeName(obj));
      first = false;
    }
    return sb.append(")").toString();
  }

  private void evalArguments(List<Object> posargs, Map<String, Object> kwargs, Environment env)
      throws EvalException, InterruptedException {
    for (Argument arg : args) {
      Object value = arg.getValue().eval(env);
      if (arg.isPositional()) {
        posargs.add(value);
      } else {
        String name = arg.getName().getName();
        if (kwargs.put(name, value) != null) {
          throw new EvalException(func.getLocation(),
              "duplicate keyword '" + name + "' in call to '" + func + "'");
        }
      }
    }
  }

  @Override
  Object eval(Environment env) throws EvalException, InterruptedException {
    List<Object> posargs = new ArrayList<>();
    Map<String, Object> kwargs = new HashMap<>();

    if (obj != null) {
      Object objValue = obj.eval(env);
      if (env.isSkylarkEnabled() && objValue instanceof ClassObject) {
        // Accessing Skylark object fields
        evalArguments(posargs, kwargs, env);
        if (!kwargs.isEmpty() || !posargs.isEmpty()) {
          throw new EvalException(func.getLocation(),
              "Arguments are not allowed when accessing fields");
        }
        Object value = ((ClassObject) objValue).getValue(func.getName());
        if (value == null) {
          throw new EvalException(func.getLocation(),
              "Unknown struct field " + func.getName());
        }
        return value; 
      }
      // Strings, lists and dictionaries (maps) have functions that we want to use in MethodLibrary.
      // For other classes, we can call the Java methods.
      if (env.isSkylarkEnabled() &&
          !(objValue instanceof String || objValue instanceof List || objValue instanceof Map)) {

        // When calling a Java method, the name is not in the Environment, so
        // evaluating 'func' would fail.

        evalArguments(posargs, kwargs, env);
        if (!kwargs.isEmpty()) {
          throw new EvalException(func.getLocation(),
              "Keyword arguments are not allowed when calling a java method");
        }
        if (objValue instanceof Class<?>) {
          // Static Java method call
          return invokeJavaMethod(null, (Class<?>) objValue, func.getName(), posargs);
        } else {
          return invokeJavaMethod(objValue, objValue.getClass(), func.getName(), posargs);
        }

      } else {
        posargs.add(objValue);
      }
    }

    Object funcValue = func.eval(env);
    if (!(funcValue instanceof Function)) {
      throw new EvalException(func.getLocation(),
                              "'" + EvalUtils.getDatatypeName(funcValue)
                              + "' object is not callable");
    }
    Function function = (Function) funcValue;
    evalArguments(posargs, kwargs, env);
    return function.call(posargs, kwargs, this, env);
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  Class<?> validate(ValidationEnvironment env) throws EvalException {
    // TODO(bazel-team): implement semantical check.

    if (obj != null) {
      obj.validate(env);
    } else {
      // TODO(bazel-team): validate function calls on objects too.
      if (!env.hasVariable(func.getName())) {
        throw new EvalException(func.getLocation(),
            String.format("function '%s' does not exist", func.getName()));
      }
      Class<?> returnValue = env.getVartype(func.getName() + ".return");
      return returnValue;
    }
    return Object.class;
  }
}

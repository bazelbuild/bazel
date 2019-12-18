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

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkInterfaceUtils;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import com.google.devtools.build.lib.util.Pair;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;

/** Helper functions for implementing function calls. */
// TODO(adonovan): make this class private. Logically it is part of EvalUtils, and the public
// methods should move there, though some parts might better exposed as a group related to annotated
// methods. For ease of review, we'll do that in a follow-up change.
public final class CallUtils {

  private CallUtils() {} // uninstantiable

  private static CacheValue getCacheValue(Class<?> cls, StarlarkSemantics semantics) {
    if (cls == String.class) {
      cls = StringModule.class;
    }
    try {
      return cache.get(Pair.of(cls, semantics));
    } catch (ExecutionException ex) {
      throw new IllegalStateException("cache error", ex);
    }
  }

  // Information derived from a SkylarkCallable-annotated class and a StarlarkSemantics.
  // methods is a superset of fields.
  private static class CacheValue {
    @Nullable MethodDescriptor selfCall;
    ImmutableMap<String, MethodDescriptor> fields; // sorted by Java method name
    ImmutableMap<String, MethodDescriptor> methods; // sorted by Java method name
  }

  // A cache of information derived from a SkylarkCallable-annotated class and a StarlarkSemantics.
  private static final LoadingCache<Pair<Class<?>, StarlarkSemantics>, CacheValue> cache =
      CacheBuilder.newBuilder()
          .build(
              new CacheLoader<Pair<Class<?>, StarlarkSemantics>, CacheValue>() {
                @Override
                public CacheValue load(Pair<Class<?>, StarlarkSemantics> key) throws Exception {
                  Class<?> cls = key.first;
                  StarlarkSemantics semantics = key.second;

                  MethodDescriptor selfCall = null;
                  ImmutableMap.Builder<String, MethodDescriptor> methods = ImmutableMap.builder();
                  Map<String, MethodDescriptor> fields = new HashMap<>();

                  // Sort methods by Java name, for determinism.
                  Method[] classMethods = cls.getMethods();
                  Arrays.sort(classMethods, Comparator.comparing(Method::getName));
                  for (Method method : classMethods) {
                    // Synthetic methods lead to false multiple matches
                    if (method.isSynthetic()) {
                      continue;
                    }

                    // annotated?
                    SkylarkCallable callable = SkylarkInterfaceUtils.getSkylarkCallable(method);
                    if (callable == null) {
                      continue;
                    }

                    // enabled by semantics?
                    if (!semantics.isFeatureEnabledBasedOnTogglingFlags(
                        callable.enableOnlyWithFlag(), callable.disableWithFlag())) {
                      continue;
                    }

                    MethodDescriptor descriptor = MethodDescriptor.of(method, callable, semantics);

                    // self-call method?
                    if (callable.selfCall()) {
                      if (selfCall != null) {
                        throw new IllegalArgumentException(
                            String.format(
                                "Class %s has two selfCall methods defined", cls.getName()));
                      }
                      selfCall = descriptor;
                      continue;
                    }

                    // regular method
                    methods.put(callable.name(), descriptor);

                    // field method?
                    if (descriptor.isStructField()
                        && fields.put(callable.name(), descriptor) != null) {
                      // TODO(b/72113542): Validate with annotation processor instead of at runtime.
                      throw new IllegalArgumentException(
                          String.format(
                              "Class %s declares two structField methods named %s",
                              cls.getName(), callable.name()));
                    }
                  }

                  CacheValue value = new CacheValue();
                  value.selfCall = selfCall;
                  value.methods = methods.build();
                  value.fields = ImmutableMap.copyOf(fields);
                  return value;
                }
              });

  /**
   * Returns a map of methods and corresponding SkylarkCallable annotations of the methods of the
   * objClass class reachable from Skylark. Elements are sorted by Java method name (which is not
   * necessarily the same as Starlark attribute name).
   */
  // TODO(adonovan): eliminate sole use in skydoc.
  public static ImmutableMap<Method, SkylarkCallable> collectSkylarkMethodsWithAnnotation(
      Class<?> objClass) {
    ImmutableMap.Builder<Method, SkylarkCallable> result = ImmutableMap.builder();
    for (MethodDescriptor desc :
        getCacheValue(objClass, StarlarkSemantics.DEFAULT_SEMANTICS).methods.values()) {
      result.put(desc.getMethod(), desc.getAnnotation());
    }
    return result.build();
  }

  /**
   * Returns the value of the Starlark field of {@code x}, implemented by a Java method with a
   * {@code SkylarkCallable(structField=true)} annotation.
   */
  public static Object getField(StarlarkSemantics semantics, Object x, String fieldName)
      throws EvalException, InterruptedException {
    MethodDescriptor desc = getCacheValue(x.getClass(), semantics).fields.get(fieldName);
    if (desc == null) {
      throw new EvalException(
          null,
          String.format(
              "value of type %s has no .%s field", EvalUtils.getDataTypeName(x), fieldName));
    }
    return desc.callField(x, Location.BUILTIN, semantics, /*mu=*/ null);
  }

  /** Returns the names of the Starlark fields of {@code x} under the specified semantics. */
  public static ImmutableSet<String> getFieldNames(StarlarkSemantics semantics, Object x) {
    return getCacheValue(x.getClass(), semantics).fields.keySet();
  }

  /** Returns the SkylarkCallable-annotated method of objClass with the given name. */
  static MethodDescriptor getMethod(
      StarlarkSemantics semantics, Class<?> objClass, String methodName) {
    return getCacheValue(objClass, semantics).methods.get(methodName);
  }

  /**
   * Returns a set of the Skylark name of all Skylark callable methods for object of type {@code
   * objClass}.
   */
  static ImmutableSet<String> getMethodNames(StarlarkSemantics semantics, Class<?> objClass) {
    return getCacheValue(objClass, semantics).methods.keySet();
  }

  /**
   * Returns a {@link MethodDescriptor} object representing a function which calls the selfCall java
   * method of the given object (the {@link SkylarkCallable} method with {@link
   * SkylarkCallable#selfCall()} set to true). Returns null if no such method exists.
   */
  @Nullable
  static MethodDescriptor getSelfCallMethodDescriptor(
      StarlarkSemantics semantics, Class<?> objClass) {
    return getCacheValue(objClass, semantics).selfCall;
  }

  /**
   * Returns the annotation from the SkylarkCallable-annotated self-call method of the specified
   * class, or null if not found.
   */
  public static SkylarkCallable getSelfCallAnnotation(Class<?> objClass) {
    MethodDescriptor selfCall =
        getSelfCallMethodDescriptor(StarlarkSemantics.DEFAULT_SEMANTICS, objClass);
    return selfCall == null ? null : selfCall.getAnnotation();
  }

  /**
   * Converts Starlark-defined arguments to an array of argument {@link Object}s that may be passed
   * to a given callable-from-Starlark Java method.
   *
   * @param thread the Starlark thread for the call
   * @param methodName the named of the called method
   * @param call the syntax tree of the call expression
   * @param method a descriptor for a java method callable from Starlark
   * @param objClass the class of the java object on which to invoke this method
   * @param positional a list of positional arguments
   * @param named a list of named arguments, as alternating Strings/Objects. May contain dups.
   * @return the array of arguments which may be passed to {@link MethodDescriptor#call}
   * @throws EvalException if the given set of arguments are invalid for the given method. For
   *     example, if any arguments are of unexpected type, or not all mandatory parameters are
   *     specified by the user
   */
  // TODO(adonovan): move to BuiltinCallable
  static Object[] convertStarlarkArgumentsToJavaMethodArguments(
      StarlarkThread thread,
      String methodName,
      FuncallExpression call,
      MethodDescriptor method,
      Class<?> objClass,
      Object[] positional,
      Object[] named)
      throws EvalException {
    Preconditions.checkArgument(!method.isStructField(),
        "struct field methods should be handled by DotExpression separately");

    // TODO(adonovan): optimize and simplify this function and improve the error messages.
    // In particular, don't build a map unless isAcceptsExtraArgs();
    // instead, make two passes, the first over positional+named,
    // the second over the vacant parameters.

    LinkedHashMap<String, Object> kwargs = Maps.newLinkedHashMapWithExpectedSize(named.length / 2);
    for (int i = 0; i < named.length; i += 2) {
      String name = (String) named[i]; // safe
      Object value = named[i + 1];
      if (kwargs.put(name, value) != null) {
        throw new EvalException(
            null, String.format("duplicate argument '%s' in call to '%s'", name, methodName));
      }
    }

    ImmutableList<ParamDescriptor> parameters = method.getParameters();
    // *args, **kwargs, location, call, thread, skylark semantics
    // TODO(adonovan): opt: compute correct size and use an array.
    final int extraArgsCount = 6;
    List<Object> builder = new ArrayList<>(parameters.size() + extraArgsCount);

    int argIndex = 0;

    // Process parameters specified in callable.parameters()
    // Positional parameters are always enumerated before non-positional parameters,
    // And default-valued positional parameters are always enumerated after other positional
    // parameters. These invariants are validated by the SkylarkCallable annotation processor.
    // Index is used deliberately, since usage of iterators adds a significant overhead
    for (int i = 0; i < parameters.size(); ++i) {
      ParamDescriptor param = parameters.get(i);
      SkylarkType type = param.getSkylarkType();
      Object value;

      if (param.isDisabledInCurrentSemantics()) {
        value = evalDefault(param.getName(), param.getValueOverride());
        builder.add(value);
        continue;
      }

      Object namedValue = param.isNamed() ? kwargs.remove(param.getName()) : null;

      if (argIndex < positional.length
          && param.isPositional()) { // Positional args and params remain.
        value = positional[argIndex];
        if (!type.contains(value)) {
          throw argumentMismatchException(
              call,
              String.format(
                  "expected value of type '%s' for parameter '%s'", type, param.getName()),
              method,
              objClass);
        }
        if (namedValue != null) {
          throw argumentMismatchException(
              call,
              String.format("got multiple values for keyword argument '%s'", param.getName()),
              method,
              objClass);
        }
        argIndex++;
      } else { // No more positional arguments, or no more positional parameters.
        if (namedValue != null) {
          // Param specified by keyword argument.
          value = namedValue;
          if (!type.contains(value)) {
            throw argumentMismatchException(
                call,
                String.format(
                    "expected value of type '%s' for parameter '%s'", type, param.getName()),
                method,
                objClass);
          }
        } else { // Param not specified by user. Use default value.
          if (param.getDefaultValue().isEmpty()) {
            throw unspecifiedParameterException(call, param, method, objClass, kwargs);
          }
          value = evalDefault(param.getName(), param.getDefaultValue());
        }
      }
      if (value == Starlark.NONE && !param.isNoneable()) {
        throw argumentMismatchException(
            call,
            String.format("parameter '%s' cannot be None", param.getName()),
            method,
            objClass);
      }
      builder.add(value);
    }

    // *args
    if (method.isAcceptsExtraArgs()) {
      builder.add(Tuple.wrap(Arrays.copyOfRange(positional, argIndex, positional.length)));
    } else if (argIndex < positional.length) {
      throw argumentMismatchException(
          call,
          String.format(
              "expected no more than %s positional arguments, but got %s",
              argIndex, positional.length),
          method,
          objClass);
    }

    // **kwargs
    if (method.isAcceptsExtraKwargs()) {
      builder.add(Dict.wrap(thread.mutability(), kwargs));
    } else if (!kwargs.isEmpty()) {
      throw unexpectedKeywordArgumentException(call, kwargs.keySet(), method, objClass, thread);
    }

    // Add Location, FuncallExpression, and/or StarlarkThread.
    appendExtraInterpreterArgs(builder, method, call, call.getLocation(), thread);

    return builder.toArray();
  }

  private static EvalException unspecifiedParameterException(
      FuncallExpression call,
      ParamDescriptor param,
      MethodDescriptor method,
      Class<?> objClass,
      Map<String, Object> kwargs) {
    if (kwargs.containsKey(param.getName())) {
      return argumentMismatchException(
          call,
          String.format("parameter '%s' may not be specified by name", param.getName()),
          method,
          objClass);
    } else {
      return argumentMismatchException(
          call,
          String.format("parameter '%s' has no default value", param.getName()),
          method,
          objClass);
    }
  }

  private static EvalException unexpectedKeywordArgumentException(
      FuncallExpression call,
      Set<String> unexpectedKeywords,
      MethodDescriptor method,
      Class<?> objClass,
      StarlarkThread thread) {
    // Check if any of the unexpected keywords are for parameters which are disabled by the
    // current semantic flags. Throwing an error with information about the misconfigured
    // semantic flag is likely far more helpful.
    for (ParamDescriptor param : method.getParameters()) {
      if (param.isDisabledInCurrentSemantics() && unexpectedKeywords.contains(param.getName())) {
        FlagIdentifier flagIdentifier = param.getFlagResponsibleForDisable();
        // If the flag is True, it must be a deprecation flag. Otherwise it's an experimental flag.
        if (thread.getSemantics().flagValue(flagIdentifier)) {
          return new EvalException(
              call.getLocation(),
              String.format(
                  "parameter '%s' is deprecated and will be removed soon. It may be "
                      + "temporarily re-enabled by setting --%s=false",
                  param.getName(), flagIdentifier.getFlagName()));
        } else {
          return new EvalException(
              call.getLocation(),
              String.format(
                  "parameter '%s' is experimental and thus unavailable with the current "
                      + "flags. It may be enabled by setting --%s",
                  param.getName(), flagIdentifier.getFlagName()));
        }
      }
    }

    return argumentMismatchException(
        call,
        String.format(
            "unexpected keyword%s %s",
            unexpectedKeywords.size() > 1 ? "s" : "",
            Joiner.on(", ").join(Iterables.transform(unexpectedKeywords, s -> "'" + s + "'"))),
        method,
        objClass);
  }

  private static EvalException argumentMismatchException(
      FuncallExpression call,
      String errorDescription,
      MethodDescriptor methodDescriptor,
      Class<?> objClass) {
    if (methodDescriptor.isSelfCall() || SkylarkInterfaceUtils.hasSkylarkGlobalLibrary(objClass)) {
      return new EvalException(
          call.getLocation(),
          String.format(
              "%s, for call to function %s",
              errorDescription, formatMethod(objClass, methodDescriptor)));
    } else {
      return new EvalException(
          call.getLocation(),
          String.format(
              "%s, for call to method %s of '%s'",
              errorDescription,
              formatMethod(objClass, methodDescriptor),
              EvalUtils.getDataTypeNameFromClass(objClass)));
    }
  }

  private static void appendExtraInterpreterArgs(
      List<Object> builder,
      MethodDescriptor method,
      @Nullable FuncallExpression call,
      Location loc,
      StarlarkThread thread) {
    if (method.isUseLocation()) {
      builder.add(loc);
    }
    if (method.isUseAst()) {
      if (call == null) {
        throw new IllegalArgumentException("Callable expects to receive ast: " + method.getName());
      }
      builder.add(call);
    }
    if (method.isUseStarlarkThread()) {
      builder.add(thread);
    }
    if (method.isUseStarlarkSemantics()) {
      builder.add(thread.getSemantics());
    }
  }

  private static String formatMethod(Class<?> objClass, MethodDescriptor methodDescriptor) {
    ImmutableList.Builder<String> argTokens = ImmutableList.builder();
    // Skip first parameter ('self') for StringModule, as its a special case.
    Iterable<ParamDescriptor> parameters =
        objClass == StringModule.class
            ? Iterables.skip(methodDescriptor.getParameters(), 1)
            : methodDescriptor.getParameters();

    for (ParamDescriptor paramDescriptor : parameters) {
      if (!paramDescriptor.isDisabledInCurrentSemantics()) {
        if (paramDescriptor.getDefaultValue().isEmpty()) {
          argTokens.add(paramDescriptor.getName());
        } else {
          argTokens.add(paramDescriptor.getName() + " = " + paramDescriptor.getDefaultValue());
        }
      }
    }
    if (methodDescriptor.isAcceptsExtraArgs()) {
      argTokens.add("*args");
    }
    if (methodDescriptor.isAcceptsExtraKwargs()) {
      argTokens.add("**kwargs");
    }
    return methodDescriptor.getName() + "(" + Joiner.on(", ").join(argTokens.build()) + ")";
  }

  // A memoization of evalDefault, keyed by expression.
  // This cache is manually maintained (instead of using LoadingCache),
  // as default values may sometimes be recursively requested.
  private static final ConcurrentHashMap<String, Object> defaultValueCache =
      new ConcurrentHashMap<>();

  // Evaluates the default value expression for a parameter.
  private static Object evalDefault(String name, String expr) {
    if (expr.isEmpty()) {
      return Starlark.NONE;
    }
    Object x = defaultValueCache.get(expr);
    if (x != null) {
      return x;
    }
    try (Mutability mutability = Mutability.create("initialization")) {
      // Note that this Starlark thread ignores command line flags.
      StarlarkThread thread =
          StarlarkThread.builder(mutability)
              .useDefaultSemantics()
              .setGlobals(Module.createForBuiltins(Starlark.UNIVERSE))
              .build()
              .update("unbound", Starlark.UNBOUND);
      x = EvalUtils.eval(ParserInput.fromLines(expr), thread);
      defaultValueCache.put(expr, x);
      return x;
    } catch (Exception ex) {
      if (ex instanceof InterruptedException) {
        Thread.currentThread().interrupt();
      }
      throw new IllegalArgumentException(
          String.format(
              "while evaluating default value %s of parameter %s: %s",
              expr, name, ex.getMessage()));
    }
  }
}

// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;

/**
 * A value class to store Methods with their corresponding {@link SkylarkCallable} annotation
 * metadata. This is needed because the annotation is sometimes in a superclass.
 *
 * <p>The annotation metadata is duplicated in this class to avoid usage of Java dynamic proxies
 * which are ~7X slower.
 */
final class MethodDescriptor {
  private final Method method;
  private final SkylarkCallable annotation;

  private final String name;
  private final String doc;
  private final boolean documented;
  private final boolean structField;
  private final ImmutableList<ParamDescriptor> parameters;
  private final ParamDescriptor extraPositionals;
  private final ParamDescriptor extraKeywords;
  private final boolean selfCall;
  private final boolean allowReturnNones;
  private final boolean useLocation;
  private final boolean useAst;
  private final boolean useStarlarkThread;
  private final boolean useStarlarkSemantics;

  private MethodDescriptor(
      Method method,
      SkylarkCallable annotation,
      String name,
      String doc,
      boolean documented,
      boolean structField,
      ImmutableList<ParamDescriptor> parameters,
      ParamDescriptor extraPositionals,
      ParamDescriptor extraKeywords,
      boolean selfCall,
      boolean allowReturnNones,
      boolean useLocation,
      boolean useAst,
      boolean useStarlarkThread,
      boolean useStarlarkSemantics) {
    this.method = method;
    this.annotation = annotation;
    this.name = name;
    this.doc = doc;
    this.documented = documented;
    this.structField = structField;
    this.parameters = parameters;
    this.extraPositionals = extraPositionals;
    this.extraKeywords = extraKeywords;
    this.selfCall = selfCall;
    this.allowReturnNones = allowReturnNones;
    this.useLocation = useLocation;
    this.useAst = useAst;
    this.useStarlarkThread = useStarlarkThread;
    this.useStarlarkSemantics = useStarlarkSemantics;
  }

  /** Returns the SkylarkCallable annotation corresponding to this method. */
  SkylarkCallable getAnnotation() {
    return annotation;
  }

  /** @return Skylark method descriptor for provided Java method and signature annotation. */
  static MethodDescriptor of(
      Method method, SkylarkCallable annotation, StarlarkSemantics semantics) {
    // This happens when the interface is public but the implementation classes
    // have reduced visibility.
    method.setAccessible(true);
    return new MethodDescriptor(
        method,
        annotation,
        annotation.name(),
        annotation.doc(),
        annotation.documented(),
        annotation.structField(),
        Arrays.stream(annotation.parameters())
            .map(param -> ParamDescriptor.of(param, semantics))
            .collect(ImmutableList.toImmutableList()),
        ParamDescriptor.of(annotation.extraPositionals(), semantics),
        ParamDescriptor.of(annotation.extraKeywords(), semantics),
        annotation.selfCall(),
        annotation.allowReturnNones(),
        annotation.useLocation(),
        annotation.useAst(),
        annotation.useStarlarkThread(),
        annotation.useStarlarkSemantics());
  }

  /**
   * Invokes this method using {@code obj} as a target and {@code args} as arguments.
   *
   * <p>{@code obj} may be {@code null} in case this method is static. Methods with {@code void}
   * return type return {@code None} following Python convention.
   */
  Object call(Object obj, Object[] args, Location loc, StarlarkThread thread)
      throws EvalException, InterruptedException {
    Preconditions.checkNotNull(obj);
    Object result;
    try {
      result = method.invoke(obj, args);
    } catch (IllegalAccessException e) {
      // TODO(bazel-team): Print a nice error message. Maybe the method exists
      // and an argument is missing or has the wrong type.
      throw new EvalException(loc, "Method invocation failed: " + e);
    } catch (InvocationTargetException x) {
      Throwable e = x.getCause();
      if (e == null) {
        // This is unlikely to happen.
        throw new IllegalStateException(
            String.format(
                "causeless InvocationTargetException when calling %s with arguments %s at %s",
                obj, Arrays.toString(args), loc),
            x);
      }
      Throwables.propagateIfPossible(e, InterruptedException.class);
      if (e instanceof EvalException) {
        throw ((EvalException) e).ensureLocation(loc);
      }
      throw new EvalException(loc, null, e);
    }
    if (method.getReturnType().equals(Void.TYPE)) {
      return Starlark.NONE;
    }
    if (result == null) {
      // TODO(adonovan): eliminate allowReturnNones. Given that we convert
      // String/Integer/Boolean/List/Map, it seems obtuse to crash instead
      // of converting null too.
      if (isAllowReturnNones()) {
        return Starlark.NONE;
      } else {
        throw new IllegalStateException(
            "method invocation returned None: " + getName() + Tuple.copyOf(Arrays.asList(args)));
      }
    }

    // Careful: thread may be null when we are called by invokeStructField.
    return Starlark.fromJava(result, thread != null ? thread.mutability() : null);
  }

  /** @see SkylarkCallable#name() */
  String getName() {
    return name;
  }

  Method getMethod() {
    return method;
  }

  /** @see SkylarkCallable#structField() */
  boolean isStructField() {
    return structField;
  }

  /** @see SkylarkCallable#useStarlarkThread() */
  boolean isUseStarlarkThread() {
    return useStarlarkThread;
  }

  /** @see SkylarkCallable#useStarlarkSemantics() */
  boolean isUseStarlarkSemantics() {
    return useStarlarkSemantics;
  }

  /** @see SkylarkCallable#useLocation() */
  boolean isUseLocation() {
    return useLocation;
  }

  /** @see SkylarkCallable#allowReturnNones() */
  boolean isAllowReturnNones() {
    return allowReturnNones;
  }

  /** @see SkylarkCallable#useAst() */
  boolean isUseAst() {
    return useAst;
  }

  /** @see SkylarkCallable#extraPositionals() */
  ParamDescriptor getExtraPositionals() {
    return extraPositionals;
  }

  ParamDescriptor getExtraKeywords() {
    return extraKeywords;
  }

  /** @return {@code true} if this method accepts extra arguments ({@code *args}) */
  boolean isAcceptsExtraArgs() {
    return !getExtraPositionals().getName().isEmpty();
  }

  /** @see SkylarkCallable#extraKeywords() */
  boolean isAcceptsExtraKwargs() {
    return !getExtraKeywords().getName().isEmpty();
  }

  /** @see SkylarkCallable#parameters() */
  ImmutableList<ParamDescriptor> getParameters() {
    return parameters;
  }

  /** @see SkylarkCallable#documented() */
  boolean isDocumented() {
    return documented;
  }

  /** @see SkylarkCallable#doc() */
  String getDoc() {
    return doc;
  }

  /** @see SkylarkCallable#selfCall() */
  boolean isSelfCall() {
    return selfCall;
  }
}

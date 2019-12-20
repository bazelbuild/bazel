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
import javax.annotation.Nullable;

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
        annotation.useStarlarkThread(),
        annotation.useStarlarkSemantics());
  }

  private static final Object[] EMPTY = {};

  /** Calls this method, which must have {@code structField=true}. */
  Object callField(Object obj, Location loc, StarlarkSemantics semantics, @Nullable Mutability mu)
      throws EvalException, InterruptedException {
    if (!structField) {
      throw new IllegalStateException("not a struct field: " + name);
    }
    // Enforced by annotation processor.
    if (useStarlarkThread) {
      throw new IllegalStateException(
          "SkylarkCallable has structField and useStarlarkThread: " + name);
    }
    int nargs = (useLocation ? 1 : 0) + (useStarlarkSemantics ? 1 : 0);
    Object[] args = nargs == 0 ? EMPTY : new Object[nargs];
    // TODO(adonovan): used only once, repository_ctx.os. Abolish?
    if (useLocation) {
      args[0] = loc;
    }
    // TODO(adonovan): used only once, in getSkylarkLibrariesToLink. Abolish?
    if (useStarlarkSemantics) {
      args[nargs - 1] = semantics;
    }
    return call(obj, args, mu);
  }

  /**
   * Invokes this method using {@code obj} as a target and {@code args} as Java arguments.
   *
   * <p>Methods with {@code void} return type return {@code None} following Python convention.
   *
   * <p>The Mutability is used if it is necessary to allocate a Starlark copy of a Java result.
   */
  Object call(Object obj, Object[] args, @Nullable Mutability mu)
      throws EvalException, InterruptedException {
    Preconditions.checkNotNull(obj);
    Object result;
    try {
      result = method.invoke(obj, args);
    } catch (IllegalAccessException ex) {
      // The annotated processor ensures that annotated methods are accessible.
      throw new IllegalStateException(ex);

    } catch (InvocationTargetException ex) {
      Throwable e = ex.getCause();
      if (e == null) {
        throw new IllegalStateException(e);
      }
      // Don't intercept unchecked exceptions.
      Throwables.throwIfUnchecked(e);
      if (e instanceof EvalException) {
        throw (EvalException) e;
      } else if (e instanceof InterruptedException) {
        throw (InterruptedException) e;
      } else {
        // All other checked exceptions (e.g. LabelSyntaxException) are reported to Starlark.
        throw new EvalException(null, null, e);
      }
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

    return Starlark.fromJava(result, mu);
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

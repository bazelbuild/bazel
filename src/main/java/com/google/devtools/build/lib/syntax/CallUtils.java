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

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.StarlarkInterfaceUtils;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
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
      return cache.get(new Key(cls, semantics));
    } catch (ExecutionException ex) {
      throw new IllegalStateException("cache error", ex);
    }
  }

  // Key is a simple Pair<Class, StarlarkSemantics>.
  private static final class Key {
    final Class<?> cls;
    final StarlarkSemantics semantics;

    Key(Class<?> cls, StarlarkSemantics semantics) {
      this.cls = cls;
      this.semantics = semantics;
    }

    @Override
    public boolean equals(Object that) {
      return this == that
          || (that instanceof Key
              && this.cls.equals(((Key) that).cls)
              && this.semantics.equals(((Key) that).semantics));
    }

    @Override
    public int hashCode() {
      return 31 * cls.hashCode() + semantics.hashCode();
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
  private static final LoadingCache<Key, CacheValue> cache =
      CacheBuilder.newBuilder()
          .build(
              new CacheLoader<Key, CacheValue>() {
                @Override
                public CacheValue load(Key key) throws Exception {
                  MethodDescriptor selfCall = null;
                  ImmutableMap.Builder<String, MethodDescriptor> methods = ImmutableMap.builder();
                  Map<String, MethodDescriptor> fields = new HashMap<>();

                  // Sort methods by Java name, for determinism.
                  Method[] classMethods = key.cls.getMethods();
                  Arrays.sort(classMethods, Comparator.comparing(Method::getName));
                  for (Method method : classMethods) {
                    // Synthetic methods lead to false multiple matches
                    if (method.isSynthetic()) {
                      continue;
                    }

                    // annotated?
                    SkylarkCallable callable = StarlarkInterfaceUtils.getSkylarkCallable(method);
                    if (callable == null) {
                      continue;
                    }

                    // enabled by semantics?
                    if (!key.semantics.isFeatureEnabledBasedOnTogglingFlags(
                        callable.enableOnlyWithFlag(), callable.disableWithFlag())) {
                      continue;
                    }

                    MethodDescriptor descriptor =
                        MethodDescriptor.of(method, callable, key.semantics);

                    // self-call method?
                    if (callable.selfCall()) {
                      if (selfCall != null) {
                        throw new IllegalArgumentException(
                            String.format(
                                "Class %s has two selfCall methods defined", key.cls.getName()));
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
                              key.cls.getName(), callable.name()));
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
   * objClass class reachable from Starlark. Elements are sorted by Java method name (which is not
   * necessarily the same as Starlark attribute name).
   */
  // TODO(adonovan): eliminate sole use in skydoc.
  public static ImmutableMap<Method, SkylarkCallable> collectSkylarkMethodsWithAnnotation(
      Class<?> objClass) {
    ImmutableMap.Builder<Method, SkylarkCallable> result = ImmutableMap.builder();
    for (MethodDescriptor desc :
        getCacheValue(objClass, StarlarkSemantics.DEFAULT).methods.values()) {
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
      throw Starlark.errorf("value of type %s has no .%s field", Starlark.type(x), fieldName);
    }
    return desc.callField(x, semantics, /*mu=*/ null);
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
   * Returns a set of the Starlark name of all Starlark callable methods for object of type {@code
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
   * Returns a {@code selfCall=true} method for the given class under the given Starlark semantics,
   * or null if no such method exists.
   */
  @Nullable
  public static Method getSelfCallMethod(StarlarkSemantics semantics, Class<?> objClass) {
    MethodDescriptor descriptor = getCacheValue(objClass, semantics).selfCall;
    if (descriptor == null) {
      return null;
    }
    return descriptor.getMethod();
  }
}

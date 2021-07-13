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

package net.starlark.java.eval;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkAnnotations;
import net.starlark.java.annot.StarlarkMethod;

/**
 * A DescriptorCache is a cache of MethodDescriptors associated with a particular StarlarkSemantics.
 *
 * <p>Every {@link StarlarkThread} holds a {@code DescriptorCache} for its semantics.
 * A {@code DescriptorCache} may be shared between threads, so they are maintained
 * in a global map created on demand by {@link #forSemantics(StarlarkSemantics)}.
 */
final class DescriptorCache {

  final StarlarkSemantics semantics;

  // A cache of information derived from a StarlarkMethod-annotated class and a StarlarkSemantics.
  private final ConcurrentHashMap<Class<?>, CacheValue> cache;

  private DescriptorCache(StarlarkSemantics semantics) {
    this.semantics = semantics;
    cache = new ConcurrentHashMap<>();
  }

  private static final ConcurrentHashMap<StarlarkSemantics, DescriptorCache> pool =
      new ConcurrentHashMap<>();

  /** Get or create {@code DescriptorCache} for given semantics. */
  static DescriptorCache forSemantics(StarlarkSemantics semantics) {
    DescriptorCache caches = pool.get(semantics);
    if (caches == null) {
      caches = new DescriptorCache(semantics);
      DescriptorCache prev = pool.putIfAbsent(semantics, caches);
      if (prev != null) {
        caches = prev; // first thread wins
      }
    }
    return caches;
  }

  private CacheValue getCacheValue(Class<?> cls) {
    // Avoid computeIfAbsent! It is not reentrant,
    // and if getCacheValue is called before Starlark.UNIVERSE
    // is initialized then the computation will re-enter the cache.
    // (This is less likely now that CallUtils is private.)
    // See b/161479826 for history.
    //
    // Concurrent calls may result in duplicate computation.
    // If this is a performance concern, then we should use a CHM
    // of futures (see ch.9 of gopl.io) so that the computation
    // is not done in the critical section of the map stripe.
    CacheValue v = cache.get(cls);
    if (v == null) {
      v = buildCacheValue(cls);
      CacheValue prev = cache.putIfAbsent(cls, v);
      if (prev != null) {
        v = prev; // first thread wins
      }
    }
    return v;
  }

  // Information derived from a StarlarkMethod-annotated class and a StarlarkSemantics.
  private static class CacheValue {
    @Nullable MethodDescriptor selfCall;
    // All StarlarkMethod-annotated Java methods, sans selfCall, sorted by Java method name.
    ImmutableMap<String, MethodDescriptor> methods;
    // Subset of CacheValue.methods for which structField=True, sorted by Java method name.
    ImmutableMap<String, MethodDescriptor> fields;
  }

  private CacheValue buildCacheValue(Class<?> cls) {
    if (cls == String.class) {
      cls = StringModule.class;
    }

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
      StarlarkMethod callable = StarlarkAnnotations.getStarlarkMethod(method);
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
              String.format("Class %s has two selfCall methods defined", cls.getName()));
        }
        selfCall = descriptor;
        continue;
      }

      // regular method
      methods.put(callable.name(), descriptor);

      // field method?
      if (descriptor.isStructField() && fields.put(callable.name(), descriptor) != null) {
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

  /**
   * Returns the set of all StarlarkMethod-annotated Java methods (excluding the self-call method)
   * of the specified class.
   */
  ImmutableMap<String, MethodDescriptor> getAnnotatedMethods(Class<?> objClass) {
    return getCacheValue(objClass).methods;
  }

  /**
   * Returns the value of the Starlark field of {@code x}, implemented by a Java method with a
   * {@code StarlarkMethod(structField=true)} annotation.
   */
  Object getAnnotatedField(Object x, String fieldName)
      throws EvalException, InterruptedException {
    MethodDescriptor desc = getCacheValue(x.getClass()).fields.get(fieldName);
    if (desc == null) {
      throw Starlark.errorf("value of type %s has no .%s field", Starlark.type(x), fieldName);
    }
    return desc.callField(x, semantics, /*mu=*/ null);
  }

  /** Returns the names of the Starlark fields of {@code x} under the specified semantics. */
  ImmutableSet<String> getAnnotatedFieldNames(Object x) {
    return getCacheValue(x.getClass()).fields.keySet();
  }

  /**
   * Returns a {@link MethodDescriptor} object representing a function which calls the selfCall java
   * method of the given object (the {@link StarlarkMethod} method with {@link
   * StarlarkMethod#selfCall()} set to true). Returns null if no such method exists.
   */
  @Nullable
  MethodDescriptor getSelfCallMethodDescriptor(Class<?> objClass) {
    return getCacheValue(objClass).selfCall;
  }

  /**
   * Returns a {@code selfCall=true} method for the given class under the given Starlark semantics,
   * or null if no such method exists.
   */
  @Nullable
  Method getSelfCallMethod(Class<?> objClass) {
    MethodDescriptor descriptor = getCacheValue(objClass).selfCall;
    if (descriptor == null) {
      return null;
    }
    return descriptor.getMethod();
  }
}

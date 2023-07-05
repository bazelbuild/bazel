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

/** Helper functions for {@link StarlarkMethod}-annotated fields and methods. */
final class CallUtils {

  private CallUtils() {} // uninstantiable

  /**
   * Returns the {@link StarlarkClassDescriptor} for the given {@link StarlarkSemantics} and {@link
   * Class}.
   *
   * <p>This method is a hotspot! It's called on every function call and field access. A single
   * `bazel build` invocation can make tens or even hundreds of millions of calls to this method.
   */
  private static StarlarkClassDescriptor getStarlarkClassDescriptor(
      StarlarkSemantics semantics, Class<?> clazz) {
    if (clazz == String.class) {
      clazz = StringModule.class;
    }

    // We use two layers of caches, with the first layer being keyed by StarlarkSemantics and the
    // second layer being keyed by Class. This optimizes for the common case of very few different
    // StarlarkSemantics instances (typically, one) being in play. In contrast, if we used a single
    // cache data structure then we'd need to use a dedicated tuple object for the keys of that data
    // structure, and the GC churn and method call overhead become meaningful at scale.
    //
    // We implement each cache ourselves using CHM#get and CHM#putIfAbsent. We don't use
    // CHM#computeIfAbsent since it is not reentrant: If #getStarlarkClassDescriptor is called
    // before Starlark.UNIVERSE is initialized then the computation will re-enter the cache and have
    // a cycle; see b/161479826 for history.
    // TODO(bazel-team): Maybe the above cycle concern doesn't exist now that CallUtils is private.
    ConcurrentHashMap<Class<?>, StarlarkClassDescriptor> starlarkClassDescriptorCache =
        starlarkClassDescriptorCachesBySemantics.get(semantics);
    if (starlarkClassDescriptorCache == null) {
      starlarkClassDescriptorCache =
          new ConcurrentHashMap<>(
              // In May 2023, typical Bazel usage results in ~150 entries in this cache. Therefore
              // we presize the CHM accordingly to reduce the chance two entries use the same hash
              // bucket (in May 2023 this strategy was completely effective!). We used to use the
              // default capacity, and then the CHM would get dynamically resized to have 256
              // buckets, many of which had at least 2 entries which is suboptimal for such a hot
              // data structure.
              // TODO(bazel-team): Better would be to precompute the entire lookup table on server
              //  startup (best would be to do this at compile time via an annotation processor),
              //  rather than rely on it getting built-up dynamically as Starlark code gets
              //  evaluated over the lifetime of the server. This way there are no concurrency
              //  concerns, so we can use a more efficient data structure that doesn't need to
              //  handle concurrent writes.
              /* initialCapacity= */ 1000);
      ConcurrentHashMap<Class<?>, StarlarkClassDescriptor> prev =
          starlarkClassDescriptorCachesBySemantics.putIfAbsent(
              semantics, starlarkClassDescriptorCache);
      if (prev != null) {
        starlarkClassDescriptorCache = prev; // first thread wins
      }
    }

    StarlarkClassDescriptor starlarkClassDescriptor = starlarkClassDescriptorCache.get(clazz);
    if (starlarkClassDescriptor == null) {
      starlarkClassDescriptor = buildStarlarkClassDescriptor(semantics, clazz);
      StarlarkClassDescriptor prev =
          starlarkClassDescriptorCache.putIfAbsent(clazz, starlarkClassDescriptor);
      if (prev != null) {
        starlarkClassDescriptor = prev; // first thread wins
      }
    }
    return starlarkClassDescriptor;
  }

  /**
   * Information derived from a {@link Class} (that has methods annotated with {@link
   * StarlarkMethod}) based on a {@link StarlarkSemantics}.
   */
  private static class StarlarkClassDescriptor {
    @Nullable MethodDescriptor selfCall;

    /**
     * All {@link StarlarkMethod}-annotated Java methods, sans ones where {@code selfCall() ==
     * true}, sorted by Java method name.
     */
    ImmutableMap<String, MethodDescriptor> methods;
    /**
     * Submap of {@link #methods} for which {@code structField() == true}, sorted by Java method
     * name.
     */
    ImmutableMap<String, MethodDescriptor> fields;
  }

  /**
   * Two-layer cache of {@link #buildStarlarkClassDescriptor}, managed by {@link
   * #getStarlarkClassDescriptor}.
   */
  private static final ConcurrentHashMap<
          StarlarkSemantics, ConcurrentHashMap<Class<?>, StarlarkClassDescriptor>>
      starlarkClassDescriptorCachesBySemantics = new ConcurrentHashMap<>();

  private static StarlarkClassDescriptor buildStarlarkClassDescriptor(
      StarlarkSemantics semantics, Class<?> clazz) {
    MethodDescriptor selfCall = null;
    ImmutableMap.Builder<String, MethodDescriptor> methods = ImmutableMap.builder();
    Map<String, MethodDescriptor> fields = new HashMap<>();

    // Sort methods by Java name, for determinism.
    Method[] classMethods = clazz.getMethods();
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
              String.format("Class %s has two selfCall methods defined", clazz.getName()));
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
                clazz.getName(), callable.name()));
      }
    }

    StarlarkClassDescriptor starlarkClassDescriptor = new StarlarkClassDescriptor();
    starlarkClassDescriptor.selfCall = selfCall;
    starlarkClassDescriptor.methods = methods.buildOrThrow();
    starlarkClassDescriptor.fields = ImmutableMap.copyOf(fields);
    return starlarkClassDescriptor;
  }

  /**
   * Returns the set of all StarlarkMethod-annotated Java methods (excluding the self-call method)
   * of the specified class.
   */
  static ImmutableMap<String, MethodDescriptor> getAnnotatedMethods(
      StarlarkSemantics semantics, Class<?> objClass) {
    return getStarlarkClassDescriptor(semantics, objClass).methods;
  }

  /**
   * Returns the value of the Starlark field of {@code x}, implemented by a Java method with a
   * {@code StarlarkMethod(structField=true)} annotation.
   */
  static Object getAnnotatedField(StarlarkSemantics semantics, Object x, String fieldName)
      throws EvalException, InterruptedException {
    MethodDescriptor desc =
        getStarlarkClassDescriptor(semantics, x.getClass()).fields.get(fieldName);
    if (desc == null) {
      throw Starlark.errorf("value of type %s has no .%s field", Starlark.type(x), fieldName);
    }
    return desc.callField(x, semantics, /*mu=*/ null);
  }

  /** Returns the names of the Starlark fields of {@code x} under the specified semantics. */
  static ImmutableSet<String> getAnnotatedFieldNames(StarlarkSemantics semantics, Object x) {
    return getStarlarkClassDescriptor(semantics, x.getClass()).fields.keySet();
  }

  /**
   * Returns a {@link MethodDescriptor} object representing a function which calls the selfCall java
   * method of the given object (the {@link StarlarkMethod} method with {@link
   * StarlarkMethod#selfCall()} set to true). Returns null if no such method exists.
   */
  @Nullable
  static MethodDescriptor getSelfCallMethodDescriptor(
      StarlarkSemantics semantics, Class<?> objClass) {
    return getStarlarkClassDescriptor(semantics, objClass).selfCall;
  }

  /**
   * Returns a {@code selfCall=true} method for the given class under the given Starlark semantics,
   * or null if no such method exists.
   */
  @Nullable
  static Method getSelfCallMethod(StarlarkSemantics semantics, Class<?> objClass) {
    MethodDescriptor descriptor = getStarlarkClassDescriptor(semantics, objClass).selfCall;
    if (descriptor == null) {
      return null;
    }
    return descriptor.getMethod();
  }
}

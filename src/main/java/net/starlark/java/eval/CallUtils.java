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
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Comparator;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkAnnotations;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.syntax.TypeConstructor;

/** Helper functions for {@link StarlarkMethod}-annotated methods. */
final class CallUtils {

  private CallUtils() {} // uninstantiable

  /**
   * Returns the {@link ClassDescriptor} for the given {@link StarlarkSemantics} and {@link Class}.
   *
   * <p>This method is a hotspot! It's called on every function call and field access. A single
   * `bazel build` invocation can make tens or even hundreds of millions of calls to this method.
   */
  private static ClassDescriptor getClassDescriptor(StarlarkSemantics semantics, Class<?> clazz) {
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
    // CHM#computeIfAbsent since it is not reentrant: If #getClassDescriptor is called
    // before Starlark.UNIVERSE is initialized then the computation will re-enter the cache and have
    // a cycle; see b/161479826 for history.
    // TODO(bazel-team): Maybe the above cycle concern doesn't exist now that CallUtils is private.
    ConcurrentHashMap<Class<?>, ClassDescriptor> classDescriptorCache =
        classDescriptorCachesBySemantics.get(semantics.getClassDescriptorCacheKey());
    if (classDescriptorCache == null) {
      classDescriptorCache =
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
      ConcurrentHashMap<Class<?>, ClassDescriptor> prev =
          classDescriptorCachesBySemantics.putIfAbsent(semantics, classDescriptorCache);
      if (prev != null) {
        classDescriptorCache = prev; // first thread wins
      }
    }

    ClassDescriptor classDescriptor = classDescriptorCache.get(clazz);
    if (classDescriptor == null) {
      classDescriptor = buildClassDescriptor(semantics, clazz);
      ClassDescriptor prev = classDescriptorCache.putIfAbsent(clazz, classDescriptor);
      if (prev != null) {
        classDescriptor = prev; // first thread wins
      }
    }
    return classDescriptor;
  }

  /**
   * Describes the Starlark methods available for a particular Java class under a particular {@link
   * StarlarkSemantics}.
   *
   * <p>Generally, but not always (e.g. in the case of compilations of global functions like {@link
   * MethodLibrary}), instances of the Java class are valid as Starlark values.
   *
   * <p>Although a {@code ClassDescriptor} does not directly embed the {@code StarlarkSemantics},
   * its contents vary based on them. In contrast, {@link MethodDescriptor} and {@link
   * ParamDescriptor} do not vary with the semantics.
   */
  // TODO(bazel-team): For context on whether descriptors should depend on the StarlarkSemantics,
  // see #25743 and the discussion in cl/742265869. The history of this is that eliminating the
  // dependence on semantics made it simpler to obtain type information and avoid an overreliance on
  // StarlarkSemantics#DEFAULT. But embedding a semantics may make it simpler to give precise static
  // type information that takes into account flag-guarding. For the moment it suffices to store a
  // semantics in BuiltinFunction.
  private static class ClassDescriptor {
    /**
     * The descriptor for the unique {@code @StarlarkMethod}-annotated method on this class that has
     * {@link StarlarkMethod#selfCall} set to true (ex: "struct" in Bazel), or null if there is no
     * such method.
     */
    @Nullable MethodDescriptor selfCall;

    /**
     * A map of the method descriptors that are available as fields of this object.
     *
     * <p>This includes methods with {@link StarlarkMethod#structField} set to true, i.e.
     * non-callable Starlark fields.
     *
     * <p>The {@code selfCall} method is omitted (if one even exists). Any methods that are disabled
     * by flag guarding via the {@link StarlarkSemantics} are also omitted.
     *
     * <p>The map is keyed on the Starlark field name, and sorted by Java method name.
     */
    ImmutableMap<String, MethodDescriptor> methods;

    /**
     * The type constructor produced by augmenting this class's base type constructor with method
     * information; or null if this class cannot be used as a type.
     *
     * <p>See {@link StarlarkMethod#isTypeConstructor}.
     */
    @Nullable TypeConstructor typeConstructor;
  }

  /** Two-layer cache of {@link #buildClassDescriptor}, managed by {@link #getClassDescriptor}. */
  private static final ConcurrentHashMap<
          StarlarkSemantics, ConcurrentHashMap<Class<?>, ClassDescriptor>>
      classDescriptorCachesBySemantics = new ConcurrentHashMap<>();

  private static ClassDescriptor buildClassDescriptor(StarlarkSemantics semantics, Class<?> clazz) {
    MethodDescriptor selfCall = null;
    ImmutableMap.Builder<String, MethodDescriptor> methods = ImmutableMap.builder();

    TypeConstructor typeConstructor = getBaseTypeConstructor(clazz);
    // TODO: #28325 - Programmatically augment this type with the @StarlarkMethods.

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

      MethodDescriptor descriptor = MethodDescriptor.of(method, callable);

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
    }

    ClassDescriptor classDescriptor = new ClassDescriptor();
    classDescriptor.selfCall = selfCall;
    classDescriptor.methods = methods.buildOrThrow();
    classDescriptor.typeConstructor = typeConstructor;
    return classDescriptor;
  }

  /**
   * Returns the base type constructor identified by the given class's {@code
   * getBaseTypeConstructor()} static method, or null if it does not have one.
   *
   * <p>The base type constructor is not the final constructor stored on the {@link
   * ClassDescriptor}; it lacks type information about the class's methods.
   *
   * @throws IllegalArgumentException if the method exists but has an unexpected signature, or if it
   *     does not evaluate successfully
   */
  @Nullable
  private static TypeConstructor getBaseTypeConstructor(Class<?> clazz) {
    Method found = null;
    for (Method m : clazz.getDeclaredMethods()) {
      if (m.getName().equals("getBaseTypeConstructor")) {
        if (found != null) {
          throw new IllegalArgumentException(
              String.format(
                  "Class %s has multiple methods named getBaseTypeConstructor", clazz.getName()));
        }
        found = m;
      }
    }
    if (found == null) {
      return null;
    }

    // Signature check.
    if (!Modifier.isPublic(found.getModifiers())
        || !Modifier.isStatic(found.getModifiers())
        || !found.getReturnType().equals(TypeConstructor.class)
        || found.getParameterCount() != 0) {
      throw new IllegalArgumentException(
          String.format(
              "Method %s#getBaseTypeConstructor has an invalid signature; "
                  + "expected 'public static TypeConstructor getBaseTypeConstructor()'",
              clazz.getName()));
    }

    try {
      return (TypeConstructor) found.invoke(null);
    } catch (IllegalAccessException | InvocationTargetException | RuntimeException e) {
      throw new IllegalArgumentException(
          String.format("Error invoking %s#getBaseTypeConstructor", clazz.getName()), e);
    }
  }

  /**
   * Returns the type constructor associated with the given Java class under a given {@code
   * StarlarkSemantics}, or null if there is none.
   *
   * <p>An example would be getting the type constructor for the {@code list} type from the class
   * {@code StarlarkList}.
   *
   * <p>The returned constructor has complete type information about the available Starlark methods
   * of the class.
   */
  @Nullable
  static TypeConstructor getTypeConstructor(StarlarkSemantics semantics, Class<?> clazz) {
    return getClassDescriptor(semantics, clazz).typeConstructor;
  }

  /**
   * Returns the set of all StarlarkMethod-annotated Java methods (excluding the self-call method)
   * of the specified class.
   */
  static ImmutableMap<String, MethodDescriptor> getAnnotatedMethods(
      StarlarkSemantics semantics, Class<?> objClass) {
    return getClassDescriptor(semantics, objClass).methods;
  }

  /**
   * Returns a {@link MethodDescriptor} object representing a function which calls the selfCall java
   * method of the given object (the {@link StarlarkMethod} method with {@link
   * StarlarkMethod#selfCall()} set to true). Returns null if no such method exists.
   */
  @Nullable
  static MethodDescriptor getSelfCallMethodDescriptor(
      StarlarkSemantics semantics, Class<?> objClass) {
    return getClassDescriptor(semantics, objClass).selfCall;
  }

  /**
   * Returns a {@code selfCall=true} method for the given class under the given Starlark semantics,
   * or null if no such method exists.
   */
  @Nullable
  static Method getSelfCallMethod(StarlarkSemantics semantics, Class<?> objClass) {
    MethodDescriptor descriptor = getClassDescriptor(semantics, objClass).selfCall;
    if (descriptor == null) {
      return null;
    }
    return descriptor.getMethod();
  }
}

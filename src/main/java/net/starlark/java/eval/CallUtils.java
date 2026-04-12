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
import net.starlark.java.syntax.Types;

/**
 * Helper functions for {@link StarlarkMethod}-annotated methods.
 *
 * <p>This class is public for the benefit of serialization in Bazel. Other code outside the
 * Starlark interpreter should not rely on it.
 */
public final class CallUtils {

  private CallUtils() {} // uninstantiable

  /** A map for obtaining a {@link BuiltinManager} from a {@link StarlarkSemantics}. */
  // Historically, this code used to have a big map from (StarlarkSemantics, Class) pairs to
  // ClassDescriptors. This caused unnecessary GC churn and method call overhead for the dedicated
  // tuple objects, which became observable at scale. It was subsequently rewritten to be a
  // double-layer map from Semantics to Class to ClassDescriptor, which optimized for the common
  // case of few (typically just one) StarlarkSemantics instances. The inner map was then abstracted
  // into BuiltinManager.
  //
  // Avoid ConcurrentHashMap#computeIfAbsent because it is not reentrant: If a ClassDescriptor is
  // looked up before Starlark.UNIVERSE is initialized then the computation will re-enter the cache
  // and have a cycle; see b/161479826 for history.
  // TODO(bazel-team): Does the above cycle concern still exist?
  private static final ConcurrentHashMap<StarlarkSemantics, BuiltinManager> managerForSemantics =
      new ConcurrentHashMap<>();

  public static BuiltinManager getBuiltinManager(StarlarkSemantics semantics) {
    BuiltinManager manager = managerForSemantics.get(semantics.getBuiltinManagerCacheKey());
    if (manager == null) {
      manager = new BuiltinManager(semantics);
      BuiltinManager prev = managerForSemantics.putIfAbsent(semantics, manager);
      if (prev != null) {
        manager = prev; // first thread wins
      }
    }
    return manager;
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
    /** The manager that created this descriptor. Used for obtaining method type information. */
    @SuppressWarnings("UnusedVariable") // TODO: #28325 - Use it for obtaining StarlarkTypes.
    BuiltinManager manager;

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
     * The type constructor to be called when the Starlark symbol that acts as this class's Starlark
     * constructor appears in a type application expression; or null if this class cannot be used as
     * a Starlark type.
     *
     * <p>For example, for {@link StarlarkList}'s descriptor this is {@link Types#LIST_CONSTRUCTOR}.
     *
     * <p>See {@link StarlarkMethod#isTypeConstructor}.
     */
    @Nullable TypeConstructor typeConstructor;
  }

  /**
   * A manager for obtaining descriptors for native-defined Starlark objects and methods, under a
   * specific {@code StarlarkSemantics}.
   *
   * <p>This class is public for the benefit of serialization in Bazel. Other code outside the
   * Starlark interpreter should not rely on it.
   */
  public static class BuiltinManager {

    private final StarlarkSemantics semantics;

    private final ClassValue<ClassDescriptor> classDescriptorCache =
        new ClassValue<ClassDescriptor>() {
          @Override
          protected ClassDescriptor computeValue(Class<?> clazz) {
            if (clazz == String.class) {
              clazz = StringModule.class;
            }
            return buildClassDescriptor(BuiltinManager.this, clazz);
          }
        };

    private BuiltinManager(StarlarkSemantics semantics) {
      this.semantics = semantics;
    }

    StarlarkSemantics getSemantics() {
      return semantics;
    }

    /**
     * Returns the {@link ClassDescriptor} for the given {@link StarlarkSemantics} and {@link
     * Class}.
     *
     * <p>This method is a hotspot! It's called on every function call and field access. A single
     * `bazel build` invocation can make tens or even hundreds of millions of calls to this method.
     */
    private ClassDescriptor getClassDescriptor(Class<?> clazz) {
      return classDescriptorCache.get(clazz);
    }

    /**
     * Returns the type constructor associated with the given Java class under a given {@code
     * StarlarkSemantics}, or null if there is none.
     *
     * <p>An example would be getting the type constructor for the {@code list} type from the class
     * {@code StarlarkList}.
     *
     * <p>The returned constructor has complete type information about the available Starlark
     * methods of the class.
     */
    @Nullable
    TypeConstructor getTypeConstructor(Class<?> clazz) {
      return getClassDescriptor(clazz).typeConstructor;
    }

    /**
     * Returns the set of all StarlarkMethod-annotated Java methods (excluding the self-call method)
     * of the specified class.
     */
    ImmutableMap<String, MethodDescriptor> getAnnotatedMethods(Class<?> objClass) {
      return getClassDescriptor(objClass).methods;
    }

    /**
     * Returns a {@link MethodDescriptor} object representing a function which calls the selfCall
     * java method of the given object (the {@link StarlarkMethod} method with {@link
     * StarlarkMethod#selfCall()} set to true). Returns null if no such method exists.
     */
    @Nullable
    MethodDescriptor getSelfCallMethodDescriptor(Class<?> objClass) {
      return getClassDescriptor(objClass).selfCall;
    }

    /**
     * Returns a {@code selfCall=true} method for the given class under the given Starlark
     * semantics, or null if no such method exists.
     */
    @Nullable
    Method getSelfCallMethod(Class<?> objClass) {
      MethodDescriptor descriptor = getClassDescriptor(objClass).selfCall;
      if (descriptor == null) {
        return null;
      }
      return descriptor.getMethod();
    }
  }

  private static ClassDescriptor buildClassDescriptor(BuiltinManager manager, Class<?> clazz) {
    MethodDescriptor selfCall = null;
    ImmutableMap.Builder<String, MethodDescriptor> methods = ImmutableMap.builder();

    TypeConstructor typeConstructor = getAssociatedTypeConstructor(clazz);

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
      if (!manager
          .getSemantics()
          .isFeatureEnabledBasedOnTogglingFlags(
              callable.enableOnlyWithFlag(), callable.disableWithFlag())) {
        continue;
      }

      MethodDescriptor descriptor = MethodDescriptor.of(manager, method, callable);

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
    classDescriptor.manager = manager;
    classDescriptor.selfCall = selfCall;
    classDescriptor.methods = methods.buildOrThrow();
    classDescriptor.typeConstructor = typeConstructor;
    return classDescriptor;
  }

  /**
   * Returns the type constructor identified by calling the given class's {@code
   * getAssociatedTypeConstructor()} static method, or null if it does not have such a method.
   *
   * @throws IllegalArgumentException if the method exists but has an unexpected signature, or if it
   *     does not evaluate successfully
   */
  @Nullable
  private static TypeConstructor getAssociatedTypeConstructor(Class<?> clazz) {
    // Special-case bool, which is represented by Java booleans and does not have its own class.
    // (String.class does not need special-casing because it's already been replaced by
    // StringModule.class by this point.)
    if (clazz.equals(Boolean.class) || clazz.equals(boolean.class)) {
      return Types.BOOL_CONSTRUCTOR;
    }

    Method found = null;
    for (Method m : clazz.getDeclaredMethods()) {
      if (m.getName().equals("getAssociatedTypeConstructor")) {
        if (found != null) {
          throw new IllegalArgumentException(
              String.format(
                  "Class %s has multiple methods named getAssociatedTypeConstructor",
                  clazz.getName()));
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
              "Method %s#getAssociatedTypeConstructor has an invalid signature; "
                  + "expected 'public static TypeConstructor getAssociatedTypeConstructor()'",
              clazz.getName()));
    }

    try {
      return (TypeConstructor) found.invoke(null);
    } catch (IllegalAccessException | InvocationTargetException | RuntimeException e) {
      throw new IllegalArgumentException(
          String.format("Error invoking %s#getAssociatedTypeConstructor", clazz.getName()), e);
    }
  }
}

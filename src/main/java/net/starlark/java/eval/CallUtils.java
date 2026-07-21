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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.primitives.Booleans.falseFirst;
import static java.util.Comparator.comparing;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkAnnotations;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.syntax.StarlarkType;
import net.starlark.java.syntax.TypeConstructor;
import net.starlark.java.syntax.TypeContext;
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
    // TODO: b/513244797 - Eliminate need for getBuiltinManagerCacheKey.
    StarlarkSemantics key = semantics.getBuiltinManagerCacheKey();
    BuiltinManager manager = managerForSemantics.get(key);
    if (manager == null) {
      manager = new BuiltinManager(semantics);
      BuiltinManager prev = managerForSemantics.putIfAbsent(key, manager);
      if (prev != null) {
        manager = prev; // first thread wins
      }
    }
    return manager;
  }

  /**
   * Describes the Starlark methods - meaning methods annotated with {@link StarlarkMethod} -
   * available for a particular Java class under a particular {@link StarlarkSemantics}.
   *
   * <p>Generally, instances of this class are valid as Starlark values, and the class itself is
   * generally annotated with {@link StarlarkBuiltin}. However, there are exceptions. For example,
   * compilations of global functions, such as {@link MethodLibrary}, are not Starlark values. Some
   * internal {@link StarlarkValue} implementations, such as {@link Starlark.UnboundMarker}, do not
   * have a {@link StarlarkBuiltin} annotation. And {@link StringModule} cannot be used as a valid
   * Starlark value despite having a {@link StarlarkBuiltin} annotation.
   *
   * <p>Although a {@code ClassDescriptor} does not directly embed the {@code StarlarkSemantics},
   * its contents vary based on them. In contrast, {@link MethodDescriptor} and {@link
   * ParamDescriptor} do not vary with the semantics.
   */
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

    /**
     * The value of {@link ClassStarlarkType#getSupertypes} for this class's {@link
     * ClassStarlarkType} if it exists (i.e. if the {@link ClassStarlarkType} is non-null); or null
     * otherwise.
     *
     * <p>Needs to be stored outside the {@link ClassStarlarkType} to avoid circular dependencies
     * between a {@link ClassStarlarkType} and its methods' args/returns types.
     */
    @Nullable ImmutableList<StarlarkType> classStarlarkTypeSupertypes;
  }

  private static final class ClassStarlarkType extends StarlarkType {
    private final String name;
    private final Class<?> clazz;
    private final BuiltinManager manager;

    private ClassStarlarkType(String name, Class<?> clazz, BuiltinManager manager) {
      this.name = name;
      this.clazz = clazz;
      this.manager = manager;
    }

    // TODO: #28325 - Populate supertypes where possible. If a class implements eval.Sequence, its
    // ClassStarlarkType should have `Sequence` as a supertype. The StarlarkType hierarchy should be
    // compatible with the Java inheritance hierarchy.
    static ImmutableList<StarlarkType> buildSupertypes(
        Class<?> clazz, ClassDescriptor classDescriptor) {
      ImmutableList.Builder<StarlarkType> builder = ImmutableList.builder();
      if (classDescriptor.selfCall != null) {
        // Values of a self-call type are callable, with the self-call method's signature.
        builder.add(classDescriptor.selfCall.getStarlarkType());
      }
      if (StarlarkAnnotations.isAssignableToStructType(clazz)) {
        // Values of struct-like types are assignable to a struct type whose fields are the
        // class's structfield annotated methods.
        // TODO: #28325 - Do we need to support partial structs?
        ImmutableMap.Builder<String, StarlarkType> fields = ImmutableMap.builder();
        classDescriptor.methods.forEach(
            (methodName, desc) -> {
              if (desc.isStructField()) {
                fields.put(methodName, desc.getStarlarkType());
              }
            });
        builder.add(Types.struct(fields.buildOrThrow()));
      }
      return builder.build();
    }

    @Override
    public ImmutableList<StarlarkType> getSupertypes() {
      return checkNotNull(manager.getClassDescriptor(clazz).classStarlarkTypeSupertypes);
    }

    @Override
    @Nullable
    public StarlarkType getField(String name, TypeContext context) {
      @Nullable MethodDescriptor method = manager.getClassDescriptor(clazz).methods.get(name);
      return method == null ? null : method.getStarlarkType();
    }

    @Override
    public String toString() {
      return name;
    }
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

    private final ClassValue<ClassStarlarkType> classStarlarkTypeCache =
        new ClassValue<ClassStarlarkType>() {
          @Override
          @Nullable
          protected ClassStarlarkType computeValue(Class<?> clazz) {
            if (!wantClassStarlarkType(clazz)) {
              return null;
            }
            // TODO: #28325 - Use a superclass's/interface's ClassStarlarkType where it makes sense.
            // In particular, classes sharing the same most-proximate StarlarkBuiltin and the same
            // set of methods should share the same ClassStarlarkType.
            @Nullable StarlarkBuiltin annotation = StarlarkAnnotations.getStarlarkBuiltin(clazz);
            String typeName = annotation != null ? annotation.name() : clazz.getSimpleName();
            return new ClassStarlarkType(typeName, clazz, BuiltinManager.this);
          }
        };

    private BuiltinManager(StarlarkSemantics semantics) {
      this.semantics = semantics;
    }

    /**
     * Returns the Starlark type to be used for valid Starlark values of the given class which
     * doesn't override {@link StarlarkValue#getStarlarkType}; or null if it (or one of its
     * superclasses) does override {@link StarlarkValue#getStarlarkType}.
     */
    @Nullable
    StarlarkType getClassStarlarkType(Class<?> clazz) {
      return classStarlarkTypeCache.get(clazz);
    }

    StarlarkSemantics getSemantics() {
      return semantics;
    }

    /**
     * Returns the {@link ClassDescriptor} for the given {@link Class}, under the BuiltinManager's
     * {@link StarlarkSemantics}.
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
    LinkedHashMap<String, MethodDescriptor> methods = new LinkedHashMap<>();

    TypeConstructor typeConstructor = getAssociatedTypeConstructor(clazz);

    // Sort non-synthetic methods ahead of synthetic ones, then by Java name for determinism. A
    // public method inherited from a non-public superclass is exposed by Class.getMethods() only as
    // a synthetic bridge, so synthetic methods must not be skipped outright; processing
    // non-synthetic methods first lets a real method (with its non-erased signature) win over its
    // bridge, while a bridge still provides any name no non-synthetic method does.
    Method[] classMethods = clazz.getMethods();
    Arrays.sort(
        classMethods, comparing(Method::isSynthetic, falseFirst()).thenComparing(Method::getName));
    for (Method method : classMethods) {
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
        if (selfCall == null) {
          selfCall = descriptor;
        } else if (!method.isSynthetic()) {
          // Two distinct selfCall methods. (A synthetic bridge of the same method -- e.g. from a
          // covariant return override -- is not a conflict and is simply ignored.)
          throw new IllegalArgumentException(
              String.format("Class %s has two selfCall methods defined", clazz.getName()));
        }
        continue;
      }

      // regular method
      methods.putIfAbsent(callable.name(), descriptor);
    }

    ClassDescriptor classDescriptor = new ClassDescriptor();
    classDescriptor.manager = manager;
    classDescriptor.selfCall = selfCall;
    classDescriptor.methods = ImmutableMap.copyOf(methods);
    classDescriptor.typeConstructor = typeConstructor;
    if (wantClassStarlarkType(clazz)) {
      classDescriptor.classStarlarkTypeSupertypes =
          ClassStarlarkType.buildSupertypes(clazz, classDescriptor);
    }
    return classDescriptor;
  }

  /**
   * Returns true if a {@link ClassStarlarkType} should be generated for the given class. This is
   * the case if the class is a {@link StarlarkValue} and does not override {@link
   * StarlarkValue#getStarlarkType}.
   */
  private static boolean wantClassStarlarkType(Class<?> clazz) {
    if (!StarlarkValue.class.isAssignableFrom(clazz)) {
      return false;
    }
    @Nullable StarlarkBuiltin annotation = StarlarkAnnotations.getStarlarkBuiltin(clazz);
    if (annotation == null) {
      return true;
    }
    Method getter;
    try {
      // LINT.IfChange
      getter = clazz.getMethod("getStarlarkType", StarlarkSemantics.class);
      // LINT.ThenChange(//src/main/java/net/starlark/java/eval/StarlarkValue.java)
    } catch (NoSuchMethodException e) {
      // All StarlarkBuiltin-annotated classes must implement StarlarkValue and thus have a
      // getStarlarkType method.
      throw new IllegalStateException(
          String.format("%s missing getStarlarkType(StarlarkSemantics) method", clazz), e);
    }
    return getter.getDeclaringClass().equals(StarlarkValue.class);
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

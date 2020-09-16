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
import net.starlark.java.annot.StarlarkInterfaceUtils;
import net.starlark.java.annot.StarlarkMethod;

/** Helper functions for StarlarkMethod-annotated fields and methods. */
final class CallUtils {

  private CallUtils() {} // uninstantiable

  private static CacheValue getCacheValue(Class<?> cls, StarlarkSemantics semantics) {
    if (cls == String.class) {
      cls = StringModule.class;
    }
    Key key = new Key(cls, semantics);

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
    CacheValue v = cache.get(key);
    if (v == null) {
      v = buildCacheValue(key);
      CacheValue prev = cache.putIfAbsent(key, v);
      if (prev != null) {
        v = prev; // first thread wins
      }
    }
    return v;
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

  // Information derived from a StarlarkMethod-annotated class and a StarlarkSemantics.
  // methods is a superset of fields.
  private static class CacheValue {
    @Nullable MethodDescriptor selfCall;
    ImmutableMap<String, MethodDescriptor> fields; // sorted by Java method name
    ImmutableMap<String, MethodDescriptor> methods; // sorted by Java method name
  }

  // A cache of information derived from a StarlarkMethod-annotated class and a StarlarkSemantics.
  private static final ConcurrentHashMap<Key, CacheValue> cache = new ConcurrentHashMap<>();

  private static CacheValue buildCacheValue(Key key) {
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
      StarlarkMethod callable = StarlarkInterfaceUtils.getStarlarkMethod(method);
      if (callable == null) {
        continue;
      }

      // enabled by semantics?
      if (!key.semantics.isFeatureEnabledBasedOnTogglingFlags(
          callable.enableOnlyWithFlag(), callable.disableWithFlag())) {
        continue;
      }

      MethodDescriptor descriptor = MethodDescriptor.of(method, callable, key.semantics);

      // self-call method?
      if (callable.selfCall()) {
        if (selfCall != null) {
          throw new IllegalArgumentException(
              String.format("Class %s has two selfCall methods defined", key.cls.getName()));
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
                key.cls.getName(), callable.name()));
      }
    }

    CacheValue value = new CacheValue();
    value.selfCall = selfCall;
    value.methods = methods.build();
    value.fields = ImmutableMap.copyOf(fields);
    return value;
  }

  /**
   * Returns a map of methods and corresponding StarlarkMethod annotations of the methods of the
   * objClass class reachable from Starlark. Elements are sorted by Java method name (which is not
   * necessarily the same as Starlark attribute name).
   */
  static ImmutableMap<Method, StarlarkMethod> getAnnotatedMethods(Class<?> objClass) {
    ImmutableMap.Builder<Method, StarlarkMethod> result = ImmutableMap.builder();
    for (MethodDescriptor desc :
        getCacheValue(objClass, StarlarkSemantics.DEFAULT).methods.values()) {
      result.put(desc.getMethod(), desc.getAnnotation());
    }
    return result.build();
  }

  /**
   * Returns the value of the Starlark field of {@code x}, implemented by a Java method with a
   * {@code StarlarkMethod(structField=true)} annotation.
   */
  static Object getAnnotatedField(StarlarkSemantics semantics, Object x, String fieldName)
      throws EvalException, InterruptedException {
    MethodDescriptor desc = getCacheValue(x.getClass(), semantics).fields.get(fieldName);
    if (desc == null) {
      throw Starlark.errorf("value of type %s has no .%s field", Starlark.type(x), fieldName);
    }
    return desc.callField(x, semantics, /*mu=*/ null);
  }

  /** Returns the names of the Starlark fields of {@code x} under the specified semantics. */
  static ImmutableSet<String> getAnnotatedFieldNames(StarlarkSemantics semantics, Object x) {
    return getCacheValue(x.getClass(), semantics).fields.keySet();
  }

  /** Returns the StarlarkMethod-annotated method of objClass with the given name. */
  static MethodDescriptor getAnnotatedMethod(
      StarlarkSemantics semantics, Class<?> objClass, String methodName) {
    return getCacheValue(objClass, semantics).methods.get(methodName);
  }

  /**
   * Returns a set of the Starlark name of all Starlark callable methods for object of type {@code
   * objClass}.
   */
  static ImmutableSet<String> getAnnotatedMethodNames(
      StarlarkSemantics semantics, Class<?> objClass) {
    return getCacheValue(objClass, semantics).methods.keySet();
  }

  /**
   * Returns a {@link MethodDescriptor} object representing a function which calls the selfCall java
   * method of the given object (the {@link StarlarkMethod} method with {@link
   * StarlarkMethod#selfCall()} set to true). Returns null if no such method exists.
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
  static Method getSelfCallMethod(StarlarkSemantics semantics, Class<?> objClass) {
    MethodDescriptor descriptor = getCacheValue(objClass, semantics).selfCall;
    if (descriptor == null) {
      return null;
    }
    return descriptor.getMethod();
  }
}

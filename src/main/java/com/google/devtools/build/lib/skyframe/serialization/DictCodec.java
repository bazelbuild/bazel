// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.base.Throwables.throwIfUnchecked;
import static com.google.devtools.build.lib.skyframe.serialization.MapHelpers.deserializeMapEntries;
import static com.google.devtools.build.lib.skyframe.serialization.MapHelpers.serializeMapEntries;
import static java.lang.invoke.MethodType.methodType;

import com.google.errorprone.annotations.Keep;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import net.starlark.java.eval.CompactImmutableDict;
import net.starlark.java.eval.Dict;

/**
 * A canonicalizing codec for {@link Dict} instances.
 *
 * <p>Serializes only the dictionary entries in iteration order (requiring the dictionary to be
 * frozen and not a {@link Dict.ImmutableKeyTrackingDict}), and deserializes into the canonical
 * {@link CompactImmutableDict} representation for its size.
 *
 * <p>To support reference cycles while keeping {@link CompactImmutableDict}'s implementation
 * classes private and their fields {@code final}, instances are constructed and populated via
 * {@link MethodHandle}s (subject to JEP 500 restrictions on final field mutation).
 */
@Keep
final class DictCodec extends AsyncObjectCodec<Dict<Object, Object>> {

  /**
   * {@code (Object, Object) -> Dict} constructing an unpopulated {@code SingletonImmutableDict}.
   */
  private static final MethodHandle NEW_SINGLETON;

  private static final MethodHandle SET_SINGLETON_KEY;
  private static final MethodHandle SET_SINGLETON_VALUE;

  /**
   * {@code (Object, Object, Object, Object) -> Dict} constructing an unpopulated {@code
   * DoubletonImmutableDict}.
   */
  private static final MethodHandle NEW_DOUBLETON;

  private static final MethodHandle SET_DOUBLETON_KEY1;
  private static final MethodHandle SET_DOUBLETON_VALUE1;
  private static final MethodHandle SET_DOUBLETON_KEY2;
  private static final MethodHandle SET_DOUBLETON_VALUE2;

  /** {@code (Object[], Object[]) -> Dict} wrapping the arrays in a {@code LinearImmutableDict}. */
  private static final MethodHandle NEW_LINEAR;

  /**
   * {@code (Object[], Object[], int[]) -> Dict} wrapping the arrays in a {@code HashImmutableDict}.
   */
  private static final MethodHandle NEW_HASH;

  /** {@code (Object[]) -> int[]} computing the open-addressing hash table for {@code ks}. */
  private static final MethodHandle CREATE_HASH_TABLE;

  private static final MethodHandle SET_HASH_TABLE;

  static {
    try {
      String prefix = CompactImmutableDict.class.getName() + "$";
      Class<?> singletonClass = Class.forName(prefix + "SingletonImmutableDict");
      Class<?> doubletonClass = Class.forName(prefix + "DoubletonImmutableDict");
      Class<?> linearClass = Class.forName(prefix + "LinearImmutableDict");
      Class<?> hashClass = Class.forName(prefix + "HashImmutableDict");

      MethodHandles.Lookup lookup =
          MethodHandles.privateLookupIn(CompactImmutableDict.class, MethodHandles.lookup());

      NEW_SINGLETON = findConstructor(lookup, singletonClass, Object.class, Object.class);
      SET_SINGLETON_KEY = findSetter(singletonClass, "k", Object.class);
      SET_SINGLETON_VALUE = findSetter(singletonClass, "v", Object.class);

      NEW_DOUBLETON =
          findConstructor(
              lookup, doubletonClass, Object.class, Object.class, Object.class, Object.class);
      SET_DOUBLETON_KEY1 = findSetter(doubletonClass, "k1", Object.class);
      SET_DOUBLETON_VALUE1 = findSetter(doubletonClass, "v1", Object.class);
      SET_DOUBLETON_KEY2 = findSetter(doubletonClass, "k2", Object.class);
      SET_DOUBLETON_VALUE2 = findSetter(doubletonClass, "v2", Object.class);

      NEW_LINEAR = findConstructor(lookup, linearClass, Object[].class, Object[].class);

      NEW_HASH = findConstructor(lookup, hashClass, Object[].class, Object[].class, int[].class);
      CREATE_HASH_TABLE =
          lookup.findStatic(hashClass, "createTable", methodType(int[].class, Object[].class));
      SET_HASH_TABLE = findSetter(hashClass, "table", int[].class);
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  private static MethodHandle findConstructor(
      MethodHandles.Lookup lookup, Class<?> clazz, Class<?>... parameterTypes)
      throws ReflectiveOperationException {
    return lookup
        .findConstructor(clazz, methodType(void.class, parameterTypes))
        .asType(methodType(Dict.class, parameterTypes));
  }

  private static MethodHandle findSetter(Class<?> clazz, String fieldName, Class<?> fieldType)
      throws ReflectiveOperationException {
    Field field = clazz.getDeclaredField(fieldName);
    field.setAccessible(true);
    return MethodHandles.lookup()
        .unreflectSetter(field)
        .asType(methodType(void.class, Dict.class, fieldType));
  }

  // Class literals cannot be parameterized. The cast is safe because the codec handles all Dict
  // instances, treating their keys and values as Objects.
  @Override
  @SuppressWarnings("unchecked")
  public Class<Dict<Object, Object>> getEncodedClass() {
    return (Class<Dict<Object, Object>>) (Class<?>) Dict.class;
  }

  @Override
  public void serialize(
      SerializationContext context, Dict<Object, Object> dict, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    if (!dict.isImmutable()) {
      throw new SerializationException("Trying to serialize unfrozen Dict: " + dict);
    }
    if (dict instanceof Dict.ImmutableKeyTrackingDict) {
      // ImmutableKeyTrackingDict backs ctx.var only when --include_config_fragments_provider is
      // enabled (an internal developer feature, also set implicitly by
      // `cquery --show_config_fragments`). Its access tracking is only meaningful while the rule
      // implementation runs, and nothing prevents a rule from capturing ctx.var into a provider,
      // so we reject it rather than persisting it; builds with that flag are not expected to
      // upload to the remote analysis cache.
      //
      // Avoid formatting dict in the message: toString() calls entrySet(), mutating accessedKeys.
      throw new SerializationException("Trying to serialize ImmutableKeyTrackingDict");
    }
    codedOut.writeInt32NoTag(dict.size());
    serializeMapEntries(context, dict, codedOut);
  }

  @Override
  public Dict<Object, Object> deserializeAsync(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int size = codedIn.readInt32();
    if (size < 0) {
      throw new SerializationException("Expected non-negative size: " + size);
    }
    return switch (size) {
      case 0 -> {
        // Dict.empty() is deliberately not a @SerializationConstant: constants are matched by
        // identity, so the shared instance would serialize differently from other empty dicts
        // (e.g. frozen mutable dicts), breaking canonicalization.
        Dict<Object, Object> empty = Dict.empty();
        context.registerInitialValue(empty);
        yield empty;
      }
      case 1 -> {
        Dict<Object, Object> dict = newSingleton();
        context.registerInitialValue(dict);
        context.deserialize(codedIn, dict, DictCodec::setSingletonKey);
        context.deserialize(codedIn, dict, DictCodec::setSingletonValue);
        yield dict;
      }
      case 2 -> {
        Dict<Object, Object> dict = newDoubleton();
        context.registerInitialValue(dict);
        context.deserialize(codedIn, dict, DictCodec::setDoubletonKey1);
        context.deserialize(codedIn, dict, DictCodec::setDoubletonValue1);
        context.deserialize(codedIn, dict, DictCodec::setDoubletonKey2);
        context.deserialize(codedIn, dict, DictCodec::setDoubletonValue2);
        yield dict;
      }
      default -> {
        Object[] ks = new Object[size];
        Object[] vs = new Object[size];
        if (size <= CompactImmutableDict.MAX_LINEAR_SIZE) {
          Dict<Object, Object> dict = newLinear(ks, vs);
          context.registerInitialValue(dict);
          deserializeMapEntries(context, codedIn, ks, vs);
          yield dict;
        }
        Dict<Object, Object> dict = newHash(ks, vs);
        context.registerInitialValue(dict);
        deserializeMapEntries(context, codedIn, ks, vs, () -> initializeHashTable(dict, ks));
        yield dict;
      }
    };
  }

  // The helpers below invoke method handles with invokeExact, which is declared to throw
  // Throwable. None of the underlying constructors, static methods, or field setters declare
  // checked exceptions, so only unchecked exceptions (RuntimeException or Error) can be thrown.

  // Safe because the newly created dict holds Object keys and values.
  @SuppressWarnings("unchecked")
  private static Dict<Object, Object> newSingleton() {
    try {
      return (Dict<Object, Object>) NEW_SINGLETON.invokeExact((Object) null, (Object) null);
    } catch (Throwable t) {
      throwIfUnchecked(t);
      throw new AssertionError("SingletonImmutableDict(Object, Object) threw checked exception", t);
    }
  }

  private static void setSingletonKey(Dict<Object, Object> dict, Object key) {
    try {
      SET_SINGLETON_KEY.invokeExact(dict, key);
    } catch (Throwable t) {
      throwIfUnchecked(t);
      throw new AssertionError("setting SingletonImmutableDict.k threw checked exception", t);
    }
  }

  private static void setSingletonValue(Dict<Object, Object> dict, Object value) {
    try {
      SET_SINGLETON_VALUE.invokeExact(dict, value);
    } catch (Throwable t) {
      throwIfUnchecked(t);
      throw new AssertionError("setting SingletonImmutableDict.v threw checked exception", t);
    }
  }

  // Safe because the newly created dict holds Object keys and values.
  @SuppressWarnings("unchecked")
  private static Dict<Object, Object> newDoubleton() {
    try {
      return (Dict<Object, Object>)
          NEW_DOUBLETON.invokeExact((Object) null, (Object) null, (Object) null, (Object) null);
    } catch (Throwable t) {
      throwIfUnchecked(t);
      throw new AssertionError(
          "DoubletonImmutableDict(Object, Object, Object, Object) threw checked exception", t);
    }
  }

  private static void setDoubletonKey1(Dict<Object, Object> dict, Object key1) {
    try {
      SET_DOUBLETON_KEY1.invokeExact(dict, key1);
    } catch (Throwable t) {
      throwIfUnchecked(t);
      throw new AssertionError("setting DoubletonImmutableDict.k1 threw checked exception", t);
    }
  }

  private static void setDoubletonValue1(Dict<Object, Object> dict, Object value1) {
    try {
      SET_DOUBLETON_VALUE1.invokeExact(dict, value1);
    } catch (Throwable t) {
      throwIfUnchecked(t);
      throw new AssertionError("setting DoubletonImmutableDict.v1 threw checked exception", t);
    }
  }

  private static void setDoubletonKey2(Dict<Object, Object> dict, Object key2) {
    try {
      SET_DOUBLETON_KEY2.invokeExact(dict, key2);
    } catch (Throwable t) {
      throwIfUnchecked(t);
      throw new AssertionError("setting DoubletonImmutableDict.k2 threw checked exception", t);
    }
  }

  private static void setDoubletonValue2(Dict<Object, Object> dict, Object value2) {
    try {
      SET_DOUBLETON_VALUE2.invokeExact(dict, value2);
    } catch (Throwable t) {
      throwIfUnchecked(t);
      throw new AssertionError("setting DoubletonImmutableDict.v2 threw checked exception", t);
    }
  }

  // Safe because the newly created dict wraps Object[] key and value arrays.
  @SuppressWarnings("unchecked")
  private static Dict<Object, Object> newLinear(Object[] ks, Object[] vs) {
    try {
      return (Dict<Object, Object>) NEW_LINEAR.invokeExact(ks, vs);
    } catch (Throwable t) {
      throwIfUnchecked(t);
      throw new AssertionError(
          "LinearImmutableDict(Object[], Object[]) threw checked exception", t);
    }
  }

  // Safe because the newly created dict wraps Object[] key and value arrays.
  @SuppressWarnings("unchecked")
  private static Dict<Object, Object> newHash(Object[] ks, Object[] vs) {
    try {
      return (Dict<Object, Object>) NEW_HASH.invokeExact(ks, vs, (int[]) null);
    } catch (Throwable t) {
      throwIfUnchecked(t);
      throw new AssertionError(
          "HashImmutableDict(Object[], Object[], int[]) threw checked exception", t);
    }
  }

  private static void initializeHashTable(Dict<Object, Object> dict, Object[] ks) {
    try {
      int[] table = (int[]) CREATE_HASH_TABLE.invokeExact(ks);
      SET_HASH_TABLE.invokeExact(dict, table);
    } catch (Throwable t) {
      throwIfUnchecked(t);
      throw new AssertionError("initializing HashImmutableDict.table threw checked exception", t);
    }
  }
}

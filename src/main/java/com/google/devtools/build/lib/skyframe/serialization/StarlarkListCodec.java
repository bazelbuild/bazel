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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.base.Throwables.throwIfUnchecked;
import static com.google.devtools.build.lib.skyframe.serialization.ArrayProcessor.deserializeObjectArray;
import static java.lang.invoke.MethodType.methodType;

import com.google.errorprone.annotations.Keep;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import net.starlark.java.eval.StarlarkList;

/**
 * A canonicalizing codec for {@link StarlarkList} instances.
 *
 * <p>Serializes only the list elements (requiring the list to be frozen), and deserializes into the
 * canonical immutable {@link StarlarkList} representation for its size.
 *
 * <p>To support reference cycles while keeping {@link StarlarkList}'s implementation classes
 * package-private and their fields {@code final}, instances are constructed and populated via
 * {@link MethodHandle}s.
 */
@Keep
final class StarlarkListCodec extends AsyncObjectCodec<StarlarkList<Object>> {

  /**
   * {@code (Object) -> StarlarkList} constructing an unpopulated {@code
   * ImmutableSingletonStarlarkList}.
   */
  private static final MethodHandle NEW_SINGLETON;

  /**
   * {@code (StarlarkList, Object) -> void} setting the final {@code elem} field on {@code
   * ImmutableSingletonStarlarkList}.
   */
  private static final MethodHandle SET_SINGLETON_ELEM;

  /**
   * {@code (Object[]) -> StarlarkList} wrapping the array in a {@code
   * RegularImmutableStarlarkList}.
   */
  private static final MethodHandle NEW_REGULAR;

  static {
    try {
      Class<?> singletonClass = StarlarkList.immutableOf("a").getClass();
      Class<?> regularClass = StarlarkList.immutableOf("a", "b").getClass();

      MethodHandles.Lookup lookup =
          MethodHandles.privateLookupIn(StarlarkList.class, MethodHandles.lookup());

      NEW_SINGLETON =
          lookup
              .findConstructor(singletonClass, methodType(void.class, Object.class))
              .asType(methodType(StarlarkList.class, Object.class));

      Field elem = singletonClass.getDeclaredField("elem");
      elem.setAccessible(true);
      SET_SINGLETON_ELEM =
          MethodHandles.lookup()
              .unreflectSetter(elem)
              .asType(methodType(void.class, StarlarkList.class, Object.class));

      NEW_REGULAR =
          lookup
              .findConstructor(regularClass, methodType(void.class, Object[].class))
              .asType(methodType(StarlarkList.class, Object[].class));
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  // Class literals cannot be parameterized. The cast is safe because the codec handles all
  // StarlarkList instances, treating their elements as Objects.
  @Override
  @SuppressWarnings("unchecked")
  public Class<StarlarkList<Object>> getEncodedClass() {
    return (Class<StarlarkList<Object>>) (Class<?>) StarlarkList.class;
  }

  @Override
  public void serialize(
      SerializationContext context, StarlarkList<Object> list, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    if (!list.isImmutable()) {
      throw new SerializationException("Trying to serialize unfrozen StarlarkList: " + list);
    }
    int size = list.size();
    codedOut.writeInt32NoTag(size);
    for (int i = 0; i < size; i++) {
      context.serialize(list.get(i), codedOut);
    }
  }

  @Override
  public StarlarkList<Object> deserializeAsync(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int size = codedIn.readInt32();
    return switch (size) {
      case 0 -> {
        // StarlarkList.empty() is deliberately not a @SerializationConstant: constants are matched
        // by identity, so the shared instance would serialize differently from other empty lists
        // (e.g. frozen mutable lists), breaking canonicalization.
        StarlarkList<Object> empty = StarlarkList.empty();
        context.registerInitialValue(empty);
        yield empty;
      }
      case 1 -> {
        StarlarkList<Object> list = newSingleton();
        context.registerInitialValue(list);
        context.deserialize(codedIn, list, StarlarkListCodec::setSingletonElem);
        yield list;
      }
      default -> {
        Object[] elems = new Object[size];
        StarlarkList<Object> list = newRegular(elems);
        context.registerInitialValue(list);
        deserializeObjectArray(context, codedIn, elems, size);
        yield list;
      }
    };
  }

  // The helpers below invoke method handles with invokeExact, which is declared to throw
  // Throwable. None of the underlying constructors or field setters declare checked exceptions,
  // so only unchecked exceptions (RuntimeException or Error) can be thrown.

  // Safe because the newly created list holds an Object element.
  @SuppressWarnings("unchecked")
  private static StarlarkList<Object> newSingleton() {
    try {
      return (StarlarkList<Object>) NEW_SINGLETON.invokeExact((Object) null);
    } catch (Throwable t) {
      throwIfUnchecked(t);
      throw new AssertionError("ImmutableSingletonStarlarkList(Object) threw checked exception", t);
    }
  }

  private static void setSingletonElem(StarlarkList<Object> list, Object elem) {
    try {
      SET_SINGLETON_ELEM.invokeExact(list, elem);
    } catch (Throwable t) {
      throwIfUnchecked(t);
      throw new AssertionError(
          "setting ImmutableSingletonStarlarkList.elem threw checked exception", t);
    }
  }

  // Safe because the newly created list wraps an Object[] element array.
  @SuppressWarnings("unchecked")
  private static StarlarkList<Object> newRegular(Object[] elems) {
    try {
      return (StarlarkList<Object>) NEW_REGULAR.invokeExact(elems);
    } catch (Throwable t) {
      throwIfUnchecked(t);
      throw new AssertionError("RegularImmutableStarlarkList(Object[]) threw checked exception", t);
    }
  }
}

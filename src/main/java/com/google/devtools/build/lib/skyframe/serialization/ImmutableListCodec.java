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

import com.google.common.collect.ImmutableList;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;

/**
 * An {@link AsyncObjectCodec} for {@link ImmutableList}.
 *
 * <p>This codec is necessary because {@link ImmutableList}:
 *
 * <ul>
 *   <li>has a number of hidden subclasses; and
 *   <li>marks important fields transient.
 * </ul>
 *
 * <p>To support cycles, we instantiate the ImmutableList before deserializing its elements and then
 * fill in the elements as they are themselves deserialized by mutating the supposedly-immutable
 * ImmutableList. This requires reching into its internals.
 *
 * <ul>
 *   <li>Lists with two or more elements are created by calling the {@code RegularImmutableList}
 *       constructor, which wraps the given array without copying it. The elements are then
 *       deserialized directly into that array.
 *   <li>Lists with a single element are created as a {@code SingletonImmutableList} with a
 *       placeholder element. The final {@code element} field is then overwritten reflectively once
 *       the actual element is available. Note that this is subject to JEP 500 restrictions on final
 *       field mutation.
 * </ul>
 */
@SuppressWarnings("rawtypes") // Intentional erasure of ImmutableList.
public class ImmutableListCodec extends AsyncObjectCodec<ImmutableList> {
  public ImmutableListCodec() {}

  private static final Class<? extends ImmutableList> SINGLETON_IMMUTABLE_LIST_CLASS =
      ImmutableList.<Integer>of(0).getClass();
  private static final Class<? extends ImmutableList> REGULAR_IMMUTABLE_LIST_CLASS =
      ImmutableList.<Integer>of(0, 1).getClass();

  /** Temporary element of a singleton list until its actual element is available. */
  private static final Object PLACEHOLDER = new Object();

  /** {@code (Object[]) -> ImmutableList}, wrapping the array without copying it. */
  private static final MethodHandle NEW_REGULAR_IMMUTABLE_LIST;

  /** {@code (ImmutableList, Object) -> void}, setting the final {@code element} field. */
  private static final MethodHandle SET_SINGLETON_ELEMENT;

  static {
    try {
      NEW_REGULAR_IMMUTABLE_LIST =
          MethodHandles.privateLookupIn(REGULAR_IMMUTABLE_LIST_CLASS, MethodHandles.lookup())
              .findConstructor(REGULAR_IMMUTABLE_LIST_CLASS, methodType(void.class, Object[].class))
              .asType(methodType(ImmutableList.class, Object[].class));

      Field element = SINGLETON_IMMUTABLE_LIST_CLASS.getDeclaredField("element");
      element.setAccessible(true);
      SET_SINGLETON_ELEMENT =
          MethodHandles.lookup()
              .unreflectSetter(element)
              .asType(methodType(void.class, ImmutableList.class, Object.class));
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  @Override
  public Class<ImmutableList> getEncodedClass() {
    return ImmutableList.class;
  }

  @Override
  public void serialize(
      SerializationContext context, ImmutableList object, CodedOutputStream codedOut)
      throws IOException, SerializationException {
    codedOut.writeInt32NoTag(object.size());
    for (Object obj : object) {
      context.serialize(obj, codedOut);
    }
  }

  @Override
  public ImmutableList deserializeAsync(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws IOException, SerializationException {
    int size = codedIn.readInt32();
    if (size == 0) {
      return ImmutableList.of();
    }

    if (size == 1) {
      ImmutableList list = ImmutableList.of(PLACEHOLDER);
      context.registerInitialValue(list);
      context.deserialize(codedIn, list, ImmutableListCodec::setSingletonElement);
      return list;
    }

    Object[] elements = new Object[size];
    ImmutableList list = newRegularImmutableList(elements);
    context.registerInitialValue(list);
    deserializeObjectArray(context, codedIn, elements, size);
    return list;
  }

  // The two helpers below invoke method handles with invokeExact, which is declared to throw
  // Throwable. The underlying targets, the RegularImmutableList(Object[]) constructor and a setter
  // for the SingletonImmutableList.element field, declare no checked exceptions, so only unchecked
  // exceptions (RuntimeException or Error) can actually be thrown. These are rethrown as is: they
  // indicate either a programming error, such as a WrongMethodTypeException, or a JVM error, such
  // as an OutOfMemoryError, neither of which should be converted into a checked exception. Any
  // other Throwable is impossible, hence the AssertionError.
  //
  // Note that if final field mutation is ever denied (JEP 500), the failure happens when the setter
  // is created in the static initializer, not when it is invoked.

  private static ImmutableList newRegularImmutableList(Object[] elements) {
    try {
      return (ImmutableList) NEW_REGULAR_IMMUTABLE_LIST.invokeExact(elements);
    } catch (Throwable t) {
      throwIfUnchecked(t);
      throw new AssertionError("RegularImmutableList(Object[]) threw a checked exception", t);
    }
  }

  private static void setSingletonElement(ImmutableList list, Object element) {
    try {
      SET_SINGLETON_ELEMENT.invokeExact(list, element);
    } catch (Throwable t) {
      throwIfUnchecked(t);
      throw new AssertionError(
          "setting SingletonImmutableList.element threw a checked exception", t);
    }
  }
}

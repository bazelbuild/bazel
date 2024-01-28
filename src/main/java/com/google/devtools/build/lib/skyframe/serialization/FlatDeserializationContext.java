// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.protobuf.CodedInputStream;
import java.io.IOException;

/**
 * Context provided to {@link ObjectCodec}s where deserialization depends deeply on child values.
 *
 * <p>This is used, for example, when deserializing keys or interned objects that cannot tolerate
 * partially deserialized values. Note that using {@link FlatDeserializationContext} precludes
 * object-cycles.
 */
public interface FlatDeserializationContext extends SerializationDependencyProvider {
  /** Defines a way to set a field in a given object. */
  interface FieldSetter<T> {
    /**
     * Sets a field of {@code obj}.
     *
     * @param target the object that accepts the field value.
     * @param fieldValue the non-null field value.
     */
    void set(T target, Object fieldValue) throws SerializationException;
  }

  /**
   * Parses the next object from {@code codedIn} and sets it in {@code obj} at {@code offset}.
   *
   * <p>Unlike {@code AsyncDeserializationContext#deserialize}, {@link #deserializeFully}
   * <i>requires</i> the requested child objects to be completely deserialized before they are
   * passed to the caller.
   *
   * <p>Under asynchrony, it's common for child values to be instantiated before they are fully
   * formed. In most cases, it is safe to set references to incompletely deserialized child objects.
   * However, in the case of sets or set-like containers, for example map keys, inserting a
   * partially formed object into the container will yield an incorrect result. For similar reasons,
   * an object containing partially formed values should not be interned. In such cases, {@link
   * #deserializeFully} should be used instead.
   *
   * <p>This effect is observable the following cases.
   *
   * <ul>
   *   <li>{@link InterningObjectCodec} is only permitted to deserialize values fully as it only
   *       sees {@link FlatDeserializationContext}. This means that the {@link
   *       InterningObjectCodec#intern} only ever observes fully deserialized values.
   *   <li>In {@link DeferredObjectCodec}s, when the supplier returned by {@link
   *       DeferredObjectCodec#deserializeDeferred} is called, any child value requested via {@link
   *       #deserializeFully} will be fully deserialized. It's possible for {@link
   *       DeferredObjectCodec} to request a combination of fully deserialized and partially
   *       deserialized values. When the supplier is invoked, values requested via {@link
   *       AsyncDeserializationContext#deserialize} might be incompletely deserialized. Such partial
   *       deserialization is essential for deserializating cyclic data structures.
   *   <li>{@code setter} or {@code done} callbacks are be able to observe differences between
   *       partially and fully deserialized values.
   * </ul>
   */
  void deserializeFully(CodedInputStream codedIn, Object obj, long offset)
      throws IOException, SerializationException;

  /**
   * Parses the next object from {@code codedIn} and sets it in {@code obj} using {@code setter}.
   *
   * <p>As with other methods in this class, the setting occurs only after the child object is fully
   * deserialized. The setter is not called if the child is null.
   */
  <T> void deserializeFully(CodedInputStream codedIn, T obj, FieldSetter<? super T> setter)
      throws IOException, SerializationException;

  /**
   * Similar to the {@link #deserializeFully} method above, but provides a {@code done} callback.
   *
   * @param done a callback invoked by the context after the requested value is set that the codec
   *     can use for reference counting
   */
  void deserializeFully(CodedInputStream codedIn, Object obj, long offset, Runnable done)
      throws IOException, SerializationException;
}

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
 * Context provided to {@link ObjectCodec} implementations with methods compatible with asynchrony.
 *
 * <p>The {@link #deserialize} signatures are defined in such a way that the context may decide when
 * to make values available.
 *
 * <p>The semantics of {@link #deserialize} can be divided into two cases.
 *
 * <ul>
 *   <li><b>Acyclic</b>: any asynchronous activity needed for deserialization is guaranteed to have
 *       completed prior to setting the value. Since there are no cycles, this is straightforward to
 *       implement by bottom-up futures-chaining. This works for any acyclic backreferences by
 *       allowing those backreferences to be stored as futures.
 *   <li><b>Cyclic</b>: when there are object graph cycles, it means that a node has a reference to
 *       one of its ancestors. In this case, during deserialization, the node will observe a
 *       partially formed ancestor value, defined by {@link #registerInitialValue}. It's impossible
 *       to guarantee that the provided value is complete due to the cycle.
 * </ul>
 */
// TODO(b/297857068): Once all codecs are migrated, this should replace the existing
// DeserializationContext as the interface to codecs.
public interface AsyncDeserializationContext extends SerializationDependencyProvider {
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
   * Registers an initial value for the currently deserializing value, for use by child objects that
   * may have references to it.
   *
   * <p>This is a noop when memoization is disabled.
   */
  void registerInitialValue(Object initialValue);

  /**
   * Parses the next object from {@code codedIn} and sets it in {@code obj} using {@code setter}.
   *
   * <p>Deserialization may complete asynchronously, for example, when the input requires a Skyframe
   * lookup to compute.
   *
   * <p>No value is written when the resulting value is null.
   */
  <T> void deserialize(CodedInputStream codedIn, T obj, FieldSetter<? super T> setter)
      throws IOException, SerializationException;

  /**
   * Parses the next object from {@code codedIn} and writes it into {@code obj} at {@code offset}.
   *
   * <p>This is an overload of {@link #deserialize(CodedInputStream, Object, FieldSetter)} that uses
   * an offset instead and avoids forcing the caller to perform a per-component allocation when
   * deserializing an array. It has similar behavior. The result can be written asynchronously or
   * not at all if its value was null.
   */
  void deserialize(CodedInputStream codedIn, Object obj, long offset)
      throws IOException, SerializationException;

  /**
   * Similar to the {@code offset} based {@link #deserialize} above, but includes a {@code done}
   * callback.
   *
   * <p>The {@code done} callback is called once the assignment is complete, which is useful for
   * container codecs that perform reference counting. The {@code done} callback is always called,
   * even if the deserialized value is null.
   */
  void deserialize(CodedInputStream codedIn, Object obj, long offset, Runnable done)
      throws IOException, SerializationException;
}

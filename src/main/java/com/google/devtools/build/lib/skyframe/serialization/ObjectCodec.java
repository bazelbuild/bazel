// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.List;

/**
 * Generic object serialization/deserialization. Implementations should serialize values
 * deterministically.
 */
public interface ObjectCodec<T> {
  /**
   * Returns the class of the objects serialized/deserialized by this codec.
   *
   * <p>This is useful for automatically dispatching to the correct codec, e.g. in {@link
   * ObjectCodecs}.
   *
   * <p>If {@link T} is an interface, then this codec will never be used by the auto-registration
   * framework in {@link ObjectCodecRegistry} unless it is explicitly invoked or {@link
   * #additionalEncodedClasses} is non-empty, since the {@link ObjectCodecRegistry} traverses the
   * concrete class hierarchy looking for matches, and will never come to an interface.
   */
  Class<? extends T> getEncodedClass();

  /**
   * Returns additional subtypes of {@code T} that may be serialized/deserialized using this codec
   * without loss of information.
   *
   * <p>This method is intended for when {@code T} has multiple concrete implementations whose
   * details are known to the codec but not to the codec dispatching mechanism. It signals that the
   * dispatcher may choose to use this codec for the subtype, rather than raise {@link
   * SerializationException.NoCodecException}.
   *
   * <p>This method should not be used if the codec's serialization and deserialization methods
   * perform their own dispatching to other codecs for subtypes of {@code T}.
   *
   * <p>{@code T} itself should not be included in the returned list.
   */
  default List<Class<? extends T>> additionalEncodedClasses() {
    return ImmutableList.of();
  }

  /**
   * Serializes {@code obj}, inverse of {@link #deserialize}.
   *
   * @param context {@link SerializationContext} providing additional information to the
   *     serialization process
   * @param obj the object to serialize
   * @param codedOut the {@link CodedOutputStream} to write this object into. Implementations need
   *     not call {@link CodedOutputStream#flush()}, this should be handled by the caller.
   * @throws SerializationException on failure to serialize
   * @throws IOException on {@link IOException} during serialization
   */
  void serialize(SerializationContext context, T obj, CodedOutputStream codedOut)
      throws SerializationException, IOException;

  /**
   * Deserializes from {@code codedIn}, inverse of {@link #serialize}.
   *
   * @param context {@link DeserializationContext} for providing additional information to the
   *     deserialization process.
   * @param codedIn the {@link CodedInputStream} to read the serialized object from
   * @return the object deserialized from {@code codedIn}
   * @throws SerializationException on failure to deserialize
   * @throws IOException on {@link IOException} during deserialization
   */
  T deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException;

  /**
   * Returns the memoization strategy for this codec.
   *
   * <p>If set to {@link MemoizationStrategy#MEMOIZE_BEFORE}, then {@link
   * DeserializationContext#registerInitialValue} must be called first in the {@link #deserialize}
   * method, before delegating to any other codecs.
   *
   * <p>Implementations of this method should just return a constant, since the choice of strategy
   * is usually intrinsic to {@link T}.
   */
  default MemoizationStrategy getStrategy() {
    return MemoizationStrategy.MEMOIZE_AFTER;
  }

  /** Indicates how an {@link ObjectCodec} is memoized. */
  enum MemoizationStrategy {
    /**
     * Indicates that memoization is not directly used by this codec.
     *
     * <p>Codecs with this strategy will always serialize payloads, never backreferences, even if
     * the same value has been serialized before. This does not apply to other codecs that are
     * delegated to within this codec. Deserialization behaves analogously.
     *
     * <p>This strategy is useful for codecs that write very little data themselves, but that still
     * delegate to other codecs.
     */
    DO_NOT_MEMOIZE,

    /**
     * Indicates that the value is memoized before recursing to its children, so that it is
     * available to form cyclic references from its children. If this strategy is used, {@link
     * DeserializationContext#registerInitialValue} must be called during the {@link #deserialize}
     * method.
     *
     * <p>This should be used for all types where it is feasible to provide an initial value. Any
     * cycle that does not go through at least one {@code MEMOIZE_BEFORE} type of value (e.g., a
     * pathological self-referential tuple) is unserializable.
     */
    MEMOIZE_BEFORE,

    /**
     * Indicates that the value is memoized after recursing to its children, so that it cannot be
     * referred to until after it has been constructed (regardless of whether its children are still
     * under construction).
     *
     * <p>This is typically used for immutable types, since they cannot be created by mutating an
     * initial value.
     */
    MEMOIZE_AFTER
  }
}

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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * Wrapper for the minutiae of serializing and deserializing objects using {@link ObjectCodec}s,
 * serving as a layer between the streaming-oriented {@link ObjectCodec} interface and users.
 */
public class ObjectCodecs {
  // TODO(shahan): when per-invocation state is needed, for example, memoization, these may
  // need to be constructed each time.
  private final SerializationContext serializationContext;
  private final DeserializationContext deserializationContext;

  /**
   * Creates an instance using the supplied {@link ObjectCodecRegistry} for looking up {@link
   * ObjectCodec}s.
   */
  public ObjectCodecs(
      ObjectCodecRegistry codecRegistry, ImmutableMap<Class<?>, Object> dependencies) {
    serializationContext = new SerializationContext(codecRegistry, dependencies);
    deserializationContext = new DeserializationContext(codecRegistry, dependencies);
  }

  public ByteString serialize(Object subject) throws SerializationException {
    ByteString.Output resultOut = ByteString.newOutput();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(resultOut);
    try {
      serializationContext.serialize(subject, codedOut);
      codedOut.flush();
      return resultOut.toByteString();
    } catch (IOException e) {
      throw new SerializationException("Failed to serialize " + subject, e);
    }
  }

  public void serialize(
      Object subject, CodedOutputStream codedOut, MemoizationPermission memoizationPermission)
      throws SerializationException {
    SerializationContext context = serializationContext;
    if (memoizationPermission == MemoizationPermission.DISABLED) {
      context = context.disableMemoization();
    }
    try {
      context.serialize(subject, codedOut);
    } catch (IOException e) {
      throw new SerializationException("Failed to serialize " + subject, e);
    }
  }

  /**
   * Controls whether memoization can occur for serialization/deserialization. Should be allowed
   * unless bit-equivalence is needed.
   */
  public enum MemoizationPermission {
    ALLOWED,
    DISABLED
  }

  public Object deserialize(ByteString data) throws SerializationException {
    return deserialize(data.newCodedInput(), MemoizationPermission.ALLOWED);
  }

  public Object deserialize(CodedInputStream codedIn, MemoizationPermission memoizationPermission)
      throws SerializationException {
    DeserializationContext context = deserializationContext;
    if (memoizationPermission == MemoizationPermission.DISABLED) {
      context = context.disableMemoization();
    }
    try {
      return context.deserialize(codedIn);
    } catch (IOException e) {
      throw new SerializationException("Failed to deserialize data", e);
    }
  }

  @VisibleForTesting
  public SerializationContext getSerializationContextForTesting() {
    return serializationContext;
  }

  @VisibleForTesting
  public DeserializationContext getDeserializationContextForTesting() {
    return deserializationContext;
  }
}

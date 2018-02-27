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

  private final ObjectCodecRegistry codecRegistry;
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
    this.codecRegistry = codecRegistry;
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

  public void serialize(Object subject, CodedOutputStream codedOut) throws SerializationException {
    try {
      serializationContext.serialize(subject, codedOut);
    } catch (IOException e) {
      throw new SerializationException("Failed to serialize " + subject, e);
    }
  }

  /**
   * Serialize {@code subject}, using the serialization strategy determined by {@code classifier},
   * returning a {@link ByteString} containing the serialized representation.
   */
  public ByteString serialize(String classifier, Object subject) throws SerializationException {
    ByteString.Output resultOut = ByteString.newOutput();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(resultOut);
    ObjectCodec<?> codec = codecRegistry.getCodecDescriptor(classifier).getCodec();
    try {
      doSerialize(classifier, codec, subject, codedOut);
      codedOut.flush();
      return resultOut.toByteString();
    } catch (IOException e) {
      throw new SerializationException(
          "Failed to serialize " + subject + " using " + codec + " for " + classifier, e);
    }
  }

  /**
   * Similar to {@link #serialize(String, Object)}, except allows the caller to specify a {@link
   * CodedOutputStream} to serialize {@code subject} to. Has less object overhead than {@link
   * #serialize(String, Object)} and as such is preferrable when serializing objects in bulk.
   *
   * <p>{@code codedOut} is not flushed by this method.
   */
  public void serialize(String classifier, Object subject, CodedOutputStream codedOut)
      throws SerializationException {
    ObjectCodec<?> codec = codecRegistry.getCodecDescriptor(classifier).getCodec();
    try {
      doSerialize(classifier, codec, subject, codedOut);
    } catch (IOException e) {
      throw new SerializationException(
          "Failed to serialize " + subject + " using " + codec + " for " + classifier, e);
    }
  }

  public Object deserialize(ByteString data) throws SerializationException {
    return deserialize(data.newCodedInput());
  }

  public Object deserialize(CodedInputStream codedIn) throws SerializationException {
    try {
      return deserializationContext.deserialize(codedIn);
    } catch (IOException e) {
      throw new SerializationException("Failed to deserialize data", e);
    }
  }

  /**
   * Deserialize {@code data} using the serialization strategy determined by {@code classifier}.
   * {@code classifier} should be the utf-8 encoded {@link ByteString} representation of the {@link
   * String} classifier used to serialize {@code data}. This is preferred since callers typically
   * have parsed {@code classifier} from a protocol buffer, for which {@link ByteString}s are
   * cheaper to use.
   */
  public Object deserialize(ByteString classifier, ByteString data) throws SerializationException {
    return deserialize(classifier, data.newCodedInput());
  }

  /**
   * Similar to {@link #deserialize(ByteString, ByteString)}, except allows the caller to specify a
   * {@link CodedInputStream} to deserialize data from. This is useful for decoding objects
   * serialized in bulk by {@link #serialize(String, Object, CodedOutputStream)}.
   */
  public Object deserialize(ByteString classifier, CodedInputStream codedIn)
      throws SerializationException {
    ObjectCodec<?> codec = codecRegistry.getCodecDescriptor(classifier).getCodec();
    // If safe, this will allow CodedInputStream to return a direct view of the underlying bytes
    // in some situations, bypassing a copy.
    codedIn.enableAliasing(true);
    try {
      Object result = codec.deserialize(deserializationContext, codedIn);
      if (result == null) {
        throw new NullPointerException(
            "ObjectCodec " + codec + " for " + classifier.toStringUtf8() + " returned null");
      }
      return result;
    } catch (IOException e) {
      throw new SerializationException(
          "Failed to deserialize data using " + codec + " for " + classifier.toStringUtf8(), e);
    }
  }

  private <T> void doSerialize(
      String classifier, ObjectCodec<T> codec, Object subject, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    try {
      codec.serialize(serializationContext, codec.getEncodedClass().cast(subject), codedOut);
    } catch (ClassCastException e) {
      throw new SerializationException(
          "Codec "
              + codec
              + " for "
              + classifier
              + " is incompatible with "
              + subject
              + " (of type "
              + subject.getClass().getName()
              + ")",
          e);
    }
  }
}

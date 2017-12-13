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
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Wrapper for the minutiae of serializing and deserializing objects using {@link ObjectCodec}s,
 * serving as a layer between the streaming-oriented {@link ObjectCodec} interface and users.
 * Handles the mapping and selection of custom serialization implementations, falling back on less
 * performant java serialization by default when no better option is available and it is allowed by
 * the configuration.
 *
 * <p>To use, create a {@link ObjectCodecs.Builder} and add custom classifier to {@link ObjectCodec}
 * mappings using {@link ObjectCodecs.Builder#add} directly or by using one of the convenience
 * builders returned by {@link ObjectCodecs.Builder#asSkyFunctionNameKeyedBuilder()} or
 * {@link ObjectCodecs.Builder#asClassKeyedBuilder()}. The provided mappings are then used to
 * determine serialization/deserialization logic. For example:
 *
 * <pre>{@code
 * // Create an instance for which anything identified as "foo" will use FooCodec.
 * ObjectCodecs objectCodecs = ObjectCodecs.newBuilder()
 *     .add("foo", new FooCodec())
 *     .build();
 *
 * // This will use the custom supplied FooCodec to serialize obj:
 * ByteString serialized = objectCodecs.serialize("foo", obj);
 * Object deserialized = objectCodecs.deserialize(ByteString.copyFromUtf8("foo"), serialized);
 *
 * // This will use default java object serialization to serialize obj:
 * ByteString serialized = objectCodecs.serialize("bar", obj);
 * Object deserialized = objectCodecs.deserialize(ByteString.copyFromUtf8("bar"), serialized);
 * }</pre>
 *
 * <p>Classifiers will typically be class names or SkyFunction names.
 */
public class ObjectCodecs {

  private static final ObjectCodec<Object> DEFAULT_CODEC = new JavaSerializableCodec();

  /** Create new ObjectCodecs.Builder, the preferred instantiation method. */
  // TODO(janakr,michajlo): Specialize builders into ones keyed by class (even if the class isn't
  // the one specified by the codec) and ones keyed by string, and expose a getClassifier() method
  // for ObjectCodecs keyed by class.
  public static ObjectCodecs.Builder newBuilder() {
    return new Builder();
  }

  private final Map<String, ObjectCodec<?>> stringMappedCodecs;
  private final Map<ByteString, ObjectCodec<?>> byteStringMappedCodecs;
  private final boolean allowDefaultCodec;

  private ObjectCodecs(Map<String, ObjectCodec<?>> codecs, boolean allowDefaultCodec) {
    this.stringMappedCodecs = codecs;
    this.byteStringMappedCodecs = makeByteStringMappedCodecs(codecs);
    this.allowDefaultCodec = allowDefaultCodec;
  }

  /**
   * Serialize {@code subject}, using the serialization strategy determined by {@code classifier},
   * returning a {@link ByteString} containing the serialized representation.
   */
  public ByteString serialize(String classifier, Object subject) throws SerializationException {
    ByteString.Output resultOut = ByteString.newOutput();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(resultOut);
    ObjectCodec<?> codec = getCodec(classifier);
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
    ObjectCodec<?> codec = getCodec(classifier);
    try {
      doSerialize(classifier, codec, subject, codedOut);
    } catch (IOException e) {
      throw new SerializationException(
          "Failed to serialize " + subject + " using " + codec + " for " + classifier, e);
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
    ObjectCodec<?> codec = getCodec(classifier);
    // If safe, this will allow CodedInputStream to return a direct view of the underlying bytes
    // in some situations, bypassing a copy.
    codedIn.enableAliasing(true);
    try {
      Object result = codec.deserialize(codedIn);
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

  private ObjectCodec<?> getCodec(String classifier)
      throws SerializationException.NoCodecException {
    ObjectCodec<?> result = stringMappedCodecs.get(classifier);
    if (result != null) {
      return result;
    } else if (allowDefaultCodec) {
      return DEFAULT_CODEC;
    } else {
      throw new SerializationException.NoCodecException(
          "No codec available for " + classifier + " and default fallback disabled");
    }
  }

  private ObjectCodec<?> getCodec(ByteString classifier) throws SerializationException {
    ObjectCodec<?> result = byteStringMappedCodecs.get(classifier);
    if (result != null) {
      return result;
    } else if (allowDefaultCodec) {
      return DEFAULT_CODEC;
    } else {
      throw new SerializationException(
          "No codec available for " + classifier.toStringUtf8() + " and default fallback disabled");
    }
  }

  private static <T> void doSerialize(
      String classifier, ObjectCodec<T> codec, Object subject, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    try {
      codec.serialize(codec.getEncodedClass().cast(subject), codedOut);
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

  /** Builder for {@link ObjectCodecs}. */
  static class Builder {
    private final ImmutableMap.Builder<String, ObjectCodec<?>> codecsBuilder =
        ImmutableMap.builder();
    private boolean allowDefaultCodec = true;

    private Builder() {}

    /**
     * Add custom serialization strategy ({@code codec}) for {@code classifier}.
     *
     * <p>Intended for package-internal usage only. Consider using the specialized build types
     * returned by {@link #asClassKeyedBuilder()} or {@link #asSkyFunctionNameKeyedBuilder()}
     * before using this method.
     */
    Builder add(String classifier, ObjectCodec<?> codec) {
      codecsBuilder.put(classifier, codec);
      return this;
    }

    /** Set whether or not we allow fallback to the default codec, java serialization. */
    public Builder setAllowDefaultCodec(boolean allowDefaultCodec) {
      this.allowDefaultCodec = allowDefaultCodec;
      return this;
    }

    /** Wrap this builder with a {@link ClassKeyedBuilder}. */
    public ClassKeyedBuilder asClassKeyedBuilder() {
      return new ClassKeyedBuilder(this);
    }

    /** Wrap this builder with a {@link SkyFunctionNameKeyedBuilder}. */
    public SkyFunctionNameKeyedBuilder asSkyFunctionNameKeyedBuilder() {
      return new SkyFunctionNameKeyedBuilder(this);
    }

    public ObjectCodecs build() {
      return new ObjectCodecs(codecsBuilder.build(), allowDefaultCodec);
    }
  }

  /** Convenience builder for adding codecs classified by class name. */
  static class ClassKeyedBuilder {
    private final Builder underlying;

    private ClassKeyedBuilder(Builder underlying) {
      this.underlying = underlying;
    }

    public <T> ClassKeyedBuilder add(Class<? extends T> clazz, ObjectCodec<T> codec) {
      underlying.add(clazz.getName(), codec);
      return this;
    }

    public ObjectCodecs build() {
      return underlying.build();
    }
  }

  /** Convenience builder for adding codecs classified by SkyFunctionName. */
  static class SkyFunctionNameKeyedBuilder {
    private final Builder underlying;

    private SkyFunctionNameKeyedBuilder(Builder underlying) {
      this.underlying = underlying;
    }

    public SkyFunctionNameKeyedBuilder add(SkyFunctionName skyFuncName, ObjectCodec<?> codec) {
      underlying.add(skyFuncName.getName(), codec);
      return this;
    }

    public ObjectCodecs build() {
      return underlying.build();
    }
  }

  private static Map<ByteString, ObjectCodec<?>> makeByteStringMappedCodecs(
      Map<String, ObjectCodec<?>> stringMappedCodecs) {
    ImmutableMap.Builder<ByteString, ObjectCodec<?>> result = ImmutableMap.builder();
    for (Entry<String, ObjectCodec<?>> entry : stringMappedCodecs.entrySet()) {
      result.put(ByteString.copyFromUtf8(entry.getKey()), entry.getValue());
    }
    return result.build();
  }

}

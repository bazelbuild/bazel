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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.protobuf.ByteString;
import java.util.Map;
import java.util.Map.Entry;
import javax.annotation.Nullable;

/**
 * Registry class for handling {@link ObjectCodec} mappings. Codecs are indexed by {@link String}
 * classifiers and assigned deterministic numeric identifiers for more compact on-the-wire
 * representation if desired.
 */
class ObjectCodecRegistry {

  static Builder newBuilder() {
    return new Builder();
  }

  private final ImmutableMap<String, CodecDescriptor> stringMappedCodecs;
  private final ImmutableMap<ByteString, CodecDescriptor> byteStringMappedCodecs;
  private final ImmutableList<CodecDescriptor> tagMappedCodecs;
  @Nullable
  private final CodecDescriptor defaultCodecDescriptor;

  private ObjectCodecRegistry(Map<String, ObjectCodec<?>> codecs, boolean allowDefaultCodec) {
    ImmutableMap.Builder<String, CodecDescriptor> codecMappingsBuilder = ImmutableMap.builder();
    int nextTag = 0;
    for (String classifier : ImmutableList.sortedCopyOf(codecs.keySet())) {
      codecMappingsBuilder.put(classifier, new CodecDescriptor(nextTag, codecs.get(classifier)));
      nextTag++;
    }
    this.stringMappedCodecs = codecMappingsBuilder.build();
    this.byteStringMappedCodecs = makeByteStringMappedCodecs(stringMappedCodecs);

    this.defaultCodecDescriptor = allowDefaultCodec
        ? new CodecDescriptor(nextTag, new JavaSerializableCodec())
        : null;
    this.tagMappedCodecs = makeTagMappedCodecs(stringMappedCodecs, defaultCodecDescriptor);
  }

  /** Returns the {@link CodecDescriptor} associated with the supplied classifier. */
  public CodecDescriptor getCodecDescriptor(String classifier)
      throws SerializationException.NoCodecException {
    CodecDescriptor result = stringMappedCodecs.getOrDefault(classifier, defaultCodecDescriptor);
    if (result != null) {
      return result;
    } else {
      throw new SerializationException.NoCodecException(
          "No codec available for " + classifier + " and default fallback disabled");
    }
  }

  /**
   * Returns the {@link CodecDescriptor} associated with the supplied classifier. This method is a
   * specialization of {@link #getCodecDescriptor(String)} for performance purposes.
   */
  public CodecDescriptor getCodecDescriptor(ByteString classifier)
      throws SerializationException.NoCodecException {
    CodecDescriptor result =
        byteStringMappedCodecs.getOrDefault(classifier, defaultCodecDescriptor);
    if (result != null) {
      return result;
    } else {
      throw new SerializationException.NoCodecException(
          "No codec available for " + classifier.toStringUtf8() + " and default fallback disabled");
    }
  }

  /** Returns the {@link CodecDescriptor} associated with the supplied tag. */
  public CodecDescriptor getCodecDescriptorByTag(int tag)
      throws SerializationException.NoCodecException {
    if (tag < 0 || tag > tagMappedCodecs.size()) {
      throw new SerializationException.NoCodecException("No codec available for tag " + tag);
    }

    CodecDescriptor result = tagMappedCodecs.get(tag);
    if (result != null) {
      return result;
    } else {
      throw new SerializationException.NoCodecException("No codec available for tag " + tag);
    }
  }

  /** Describes encoding logic. */
  static class CodecDescriptor {
    private final int tag;
    private final ObjectCodec<?> codec;

    private CodecDescriptor(int tag, ObjectCodec<?> codec) {
      this.tag = tag;
      this.codec = codec;
    }

    /**
     * Unique identifier identifying the associated codec. Intended to be used as a compact
     * on-the-wire representation of an encoded object's type.
     */
    int getTag() {
      return tag;
    }

    ObjectCodec<?> getCodec() {
      return codec;
    }
  }

  /** Builder for {@link ObjectCodecRegistry}. */
  static class Builder {
    private final ImmutableMap.Builder<String, ObjectCodec<?>> codecsBuilder =
        ImmutableMap.builder();
    private boolean allowDefaultCodec = true;

    private Builder() {}

    /**
     * Add custom serialization strategy ({@code codec}) for {@code classifier}.
     *
     * <p>Intended for package-internal usage only. Consider using the specialized build types
     * returned by {@link #asClassKeyedBuilder()} before using this method.
     */
    Builder add(String classifier, ObjectCodec<?> codec) {
      codecsBuilder.put(classifier, codec);
      return this;
    }

    /**
     * Set whether or not we allow fallback to java serialization when no matching codec is found.
     */
    public Builder setAllowDefaultCodec(boolean allowDefaultCodec) {
      this.allowDefaultCodec = allowDefaultCodec;
      return this;
    }

    /** Wrap this builder with a {@link ClassKeyedBuilder}. */
    public ClassKeyedBuilder asClassKeyedBuilder() {
      return new ClassKeyedBuilder(this);
    }

    public ObjectCodecRegistry build() {
      return new ObjectCodecRegistry(codecsBuilder.build(), allowDefaultCodec);
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

    public ObjectCodecRegistry build() {
      return underlying.build();
    }
  }

  private static ImmutableMap<ByteString, CodecDescriptor> makeByteStringMappedCodecs(
      Map<String, CodecDescriptor> stringMappedCodecs) {
    ImmutableMap.Builder<ByteString, CodecDescriptor> result = ImmutableMap.builder();
    for (Entry<String, CodecDescriptor> entry : stringMappedCodecs.entrySet()) {
      result.put(ByteString.copyFromUtf8(entry.getKey()), entry.getValue());
    }
    return result.build();
  }

  private static ImmutableList<CodecDescriptor> makeTagMappedCodecs(
      Map<String, CodecDescriptor> codecs,
      @Nullable CodecDescriptor defaultCodecDescriptor) {
    CodecDescriptor[] codecTable =
        new CodecDescriptor[codecs.size() + (defaultCodecDescriptor != null ? 1 : 0)];
    for (Entry<String, CodecDescriptor> entry : codecs.entrySet()) {
      codecTable[entry.getValue().getTag()] = entry.getValue();
    }

    if (defaultCodecDescriptor != null) {
      codecTable[defaultCodecDescriptor.getTag()] = defaultCodecDescriptor;
    }
    return ImmutableList.copyOf(codecTable);
  }
}

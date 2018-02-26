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
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Map.Entry;
import javax.annotation.Nullable;

/**
 * Registry class for handling {@link ObjectCodec} mappings. Codecs are indexed by {@link String}
 * classifiers and assigned deterministic numeric identifiers for more compact on-the-wire
 * representation if desired.
 */
public class ObjectCodecRegistry {

  static Builder newBuilder() {
    return new Builder();
  }

  private final ImmutableMap<String, CodecDescriptor> stringMappedCodecs;
  private final ImmutableMap<ByteString, CodecDescriptor> byteStringMappedCodecs;
  private final ImmutableList<CodecDescriptor> tagMappedCodecs;
  @Nullable
  private final CodecDescriptor defaultCodecDescriptor;
  private final IdentityHashMap<Object, Integer> constantsMap;
  private final ImmutableList<Object> constants;
  private final int constantsStartTag;

  private ObjectCodecRegistry(
      Map<String, CodecHolder> codecs, ImmutableList<Object> constants, boolean allowDefaultCodec) {
    ImmutableMap.Builder<String, CodecDescriptor> codecMappingsBuilder = ImmutableMap.builder();
    int nextTag = 1; // 0 is reserved for null.
    for (String classifier : ImmutableList.sortedCopyOf(codecs.keySet())) {
      codecMappingsBuilder.put(classifier, codecs.get(classifier).createDescriptor(nextTag));
      nextTag++;
    }
    this.stringMappedCodecs = codecMappingsBuilder.build();

    this.byteStringMappedCodecs = makeByteStringMappedCodecs(stringMappedCodecs);

    this.defaultCodecDescriptor =
        allowDefaultCodec
            ? new TypedCodecDescriptor<>(nextTag++, new JavaSerializableCodec())
            : null;
    this.tagMappedCodecs = makeTagMappedCodecs(stringMappedCodecs, defaultCodecDescriptor);
    constantsStartTag = nextTag;
    constantsMap = new IdentityHashMap<>();
    for (Object constant : constants) {
      constantsMap.put(constant, nextTag++);
    }
    this.constants = constants;
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

  /**
   * Returns a {@link CodecDescriptor} for the given type.
   *
   * <p>Falls back to a codec for the nearest super type of type. Failing that, may fall back to the
   * registry's default codec.
   */
  public CodecDescriptor getCodecDescriptor(Class<?> type)
      throws SerializationException.NoCodecException {
    // TODO(blaze-team): consider caching this traversal.
    for (Class<?> nextType = type; nextType != null; nextType = nextType.getSuperclass()) {
      CodecDescriptor result = stringMappedCodecs.get(nextType.getName());
      if (result != null) {
        return result;
      }
    }
    if (defaultCodecDescriptor == null) {
      throw new SerializationException.NoCodecException(
          "No codec available for " + type + " and default fallback disabled");
    }
    return defaultCodecDescriptor;
  }

  @Nullable
  Object maybeGetConstantByTag(int tag) {
    return tag < constantsStartTag || tag - constantsStartTag >= constants.size()
        ? null
        : constants.get(tag - constantsStartTag);
  }

  @Nullable
  Integer maybeGetTagForConstant(Object object) {
    return constantsMap.get(object);
  }

  /** Returns the {@link CodecDescriptor} associated with the supplied tag. */
  public CodecDescriptor getCodecDescriptorByTag(int tag)
      throws SerializationException.NoCodecException {
    int tagOffset = tag - 1;
    if (tagOffset < 0 || tagOffset > tagMappedCodecs.size()) {
      throw new SerializationException.NoCodecException("No codec available for tag " + tag);
    }

    CodecDescriptor result = tagMappedCodecs.get(tagOffset);
    if (result != null) {
      return result;
    } else {
      throw new SerializationException.NoCodecException("No codec available for tag " + tag);
    }
  }

  /**
   * Creates a builder using the current contents of this registry.
   *
   * <p>This is much more efficient than scanning multiple times.
   */
  Builder getBuilder() {
    Builder builder = newBuilder();
    builder.setAllowDefaultCodec(defaultCodecDescriptor != null);
    for (Map.Entry<String, CodecDescriptor> entry : stringMappedCodecs.entrySet()) {
      builder.add(entry.getKey(), entry.getValue().getCodec());
    }
    return builder;
  }

  /** Describes encoding logic. */
  static interface CodecDescriptor {
    void serialize(SerializationContext context, Object obj, CodedOutputStream codedOut)
        throws IOException, SerializationException;

    Object deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws IOException, SerializationException;

    /**
     * Unique identifier identifying the associated codec.
     *
     * <p>Intended to be used as a compact on-the-wire representation of an encoded object's type.
     *
     * <p>Returns a value ≥ 1.
     *
     * <p>0 is a special tag representing null while negative numbers are reserved for
     * backreferences.
     */
    int getTag();

    /** Returns the underlying codec. */
    ObjectCodec<?> getCodec();
  }

  private static class TypedCodecDescriptor<T> implements CodecDescriptor {
    private final int tag;
    private final ObjectCodec<T> codec;

    private TypedCodecDescriptor(int tag, ObjectCodec<T> codec) {
      this.tag = tag;
      this.codec = codec;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void serialize(SerializationContext context, Object obj, CodedOutputStream codedOut)
        throws IOException, SerializationException {
      codec.serialize(context, (T) obj, codedOut);
    }

    @Override
    public T deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws IOException, SerializationException {
      return codec.deserialize(context, codedIn);
    }

    @Override
    public int getTag() {
      return tag;
    }

    @Override
    public ObjectCodec<T> getCodec() {
      return codec;
    }
  }

  private interface CodecHolder {
    CodecDescriptor createDescriptor(int tag);
  }

  private static class TypedCodecHolder<T> implements CodecHolder {
    private final ObjectCodec<T> codec;

    private TypedCodecHolder(ObjectCodec<T> codec) {
      this.codec = codec;
    }

    @Override
    public CodecDescriptor createDescriptor(int tag) {
      return new TypedCodecDescriptor<T>(tag, codec);
    }
  }

  /** Builder for {@link ObjectCodecRegistry}. */
  public static class Builder {
    private final ImmutableMap.Builder<String, CodecHolder> codecsBuilder = ImmutableMap.builder();
    private final ImmutableList.Builder<Object> constantsBuilder = ImmutableList.builder();
    private boolean allowDefaultCodec = true;

    /**
     * Add custom serialization strategy ({@code codec}) for {@code classifier}.
     *
     * <p>Intended for package-internal usage only. Consider using the specialized build types
     * returned by {@link #asClassKeyedBuilder()} before using this method.
     */
    <T> Builder add(String classifier, ObjectCodec<T> codec) {
      codecsBuilder.put(classifier, new TypedCodecHolder<>(codec));
      return this;
    }

    public <T> Builder add(Class<? extends T> type, ObjectCodec<T> codec) {
      add(type.getName(), codec);
      return this;
    }

    /**
     * Set whether or not we allow fallback to java serialization when no matching codec is found.
     */
    public Builder setAllowDefaultCodec(boolean allowDefaultCodec) {
      this.allowDefaultCodec = allowDefaultCodec;
      return this;
    }

    public Builder addConstant(Object object) {
      constantsBuilder.add(object);
      return this;
    }

    /** Wrap this builder with a {@link ClassKeyedBuilder}. */
    public ClassKeyedBuilder asClassKeyedBuilder() {
      return new ClassKeyedBuilder(this);
    }

    public ObjectCodecRegistry build() {
      return new ObjectCodecRegistry(
          codecsBuilder.build(), constantsBuilder.build(), allowDefaultCodec);
    }
  }

  /** Convenience builder for adding codecs classified by class name. */
  static class ClassKeyedBuilder {
    private final Builder underlying;

    private ClassKeyedBuilder(Builder underlying) {
      this.underlying = underlying;
    }

    public <T> ClassKeyedBuilder add(Class<? extends T> clazz, ObjectCodec<T> codec) {
      underlying.add(clazz, codec);
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
      codecTable[entry.getValue().getTag() - 1] = entry.getValue();
    }

    if (defaultCodecDescriptor != null) {
      codecTable[defaultCodecDescriptor.getTag() - 1] = defaultCodecDescriptor;
    }
    return ImmutableList.copyOf(codecTable);
  }
}

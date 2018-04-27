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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.function.Supplier;
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

  private final boolean allowDefaultCodec;

  private final ImmutableMap<Class<?>, CodecDescriptor> classMappedCodecs;
  private final ImmutableList<CodecDescriptor> tagMappedCodecs;

  private final int referenceConstantsStartTag;
  private final IdentityHashMap<Object, Integer> referenceConstantsMap;
  private final ImmutableList<Object> referenceConstants;

  private final int valueConstantsStartTag;
  private final ImmutableMap<Class<?>, ImmutableMap<Object, Integer>> valueConstantsMap;
  private final ImmutableList<Object> valueConstants;

  /** This is sorted, but we need index-based access. */
  private final ImmutableList<String> classNames;

  private final IdentityHashMap<String, Supplier<CodecDescriptor>> dynamicCodecs;

  private ObjectCodecRegistry(
      ImmutableSet<ObjectCodec<?>> memoizingCodecs,
      ImmutableList<Object> referenceConstants,
      ImmutableList<Object> valueConstants,
      ImmutableSortedSet<String> classNames,
      ImmutableList<String> blacklistedClassNamePrefixes,
      boolean allowDefaultCodec) {
    this.allowDefaultCodec = allowDefaultCodec;

    int nextTag = 1; // 0 is reserved for null.
    ImmutableMap.Builder<Class<?>, CodecDescriptor> memoizingCodecsBuilder =
        ImmutableMap.builderWithExpectedSize(memoizingCodecs.size());
    ImmutableList.Builder<CodecDescriptor> tagMappedMemoizingCodecsBuilder =
        ImmutableList.builderWithExpectedSize(memoizingCodecs.size());
    nextTag =
        processCodecs(
            memoizingCodecs, nextTag, tagMappedMemoizingCodecsBuilder, memoizingCodecsBuilder);

    this.classMappedCodecs = memoizingCodecsBuilder.build();
    this.tagMappedCodecs = tagMappedMemoizingCodecsBuilder.build();

    referenceConstantsStartTag = nextTag;
    referenceConstantsMap = new IdentityHashMap<>();
    for (Object constant : referenceConstants) {
      referenceConstantsMap.put(constant, nextTag++);
    }
    this.referenceConstants = referenceConstants;

    valueConstantsStartTag = nextTag;

    HashMap<Class<?>, HashMap<Object, Integer>> valuesBuilder = new HashMap<>();
    for (Object constant : valueConstants) {
      valuesBuilder
          .computeIfAbsent(constant.getClass(), k -> new HashMap<>())
          .put(constant, nextTag++);
    }
    this.valueConstantsMap =
        valuesBuilder
            .entrySet()
            .stream()
            .collect(
                ImmutableMap.toImmutableMap(
                    Map.Entry::getKey, e -> ImmutableMap.copyOf(e.getValue())));
    this.valueConstants = valueConstants;
    this.classNames =
        classNames
            .stream()
            .filter((str) -> isAllowed(str, blacklistedClassNamePrefixes))
            .collect(ImmutableList.toImmutableList());
    this.dynamicCodecs = createDynamicCodecs(this.classNames, nextTag);
  }

  public CodecDescriptor getCodecDescriptorForObject(Object obj)
      throws SerializationException.NoCodecException {
    Class<?> type = obj.getClass();
    CodecDescriptor descriptor = getCodecDescriptor(type);
    if (descriptor != null) {
      return descriptor;
    }
    if (!allowDefaultCodec) {
      throw new SerializationException.NoCodecException(
          "No codec available for " + type + " and default fallback disabled");
    }
    if (obj instanceof Enum) {
      // Enums must be serialized using declaring class.
      type = ((Enum) obj).getDeclaringClass();
    }
    return getDynamicCodecDescriptor(type.getName());
  }

  /**
   * Returns a {@link CodecDescriptor} for the given type or null if none found.
   *
   * <p>Also checks if there are codecs for a superclass of the given type.
   */
  private @Nullable CodecDescriptor getCodecDescriptor(Class<?> type) {
    // TODO(blaze-team): consider caching this traversal.
    for (Class<?> nextType = type; nextType != null; nextType = nextType.getSuperclass()) {
      CodecDescriptor result = classMappedCodecs.get(nextType);
      if (result != null) {
        return result;
      }
    }
    return null;
  }

  @Nullable
  Object maybeGetConstantByTag(int tag) {
    if (referenceConstantsStartTag <= tag
        && tag < referenceConstantsStartTag + referenceConstants.size()) {
      return referenceConstants.get(tag - referenceConstantsStartTag);
    }
    if (valueConstantsStartTag <= tag && tag < valueConstantsStartTag + valueConstants.size()) {
      return valueConstants.get(tag - valueConstantsStartTag);
    }
    return null;
  }

  @Nullable
  Integer maybeGetTagForConstant(Object object) {
    Integer result = referenceConstantsMap.get(object);
    if (result != null) {
      return result;
    }
    ImmutableMap<Object, Integer> valueConstantsForClass = valueConstantsMap.get(object.getClass());
    if (valueConstantsForClass == null) {
      return null;
    }
    return valueConstantsForClass.get(object);
  }

  /** Returns the {@link CodecDescriptor} associated with the supplied tag. */
  public CodecDescriptor getCodecDescriptorByTag(int tag)
      throws SerializationException.NoCodecException {
    int tagOffset = tag - 1; // 0 reserved for null
    if (tagOffset < 0) {
      throw new SerializationException.NoCodecException("No codec available for tag " + tag);
    }
    if (tagOffset < tagMappedCodecs.size()) {
      return tagMappedCodecs.get(tagOffset);
    }

    tagOffset -= tagMappedCodecs.size();
    tagOffset -= referenceConstants.size();
    tagOffset -= valueConstants.size();
    if (!allowDefaultCodec || tagOffset < 0 || tagOffset >= classNames.size()) {
      throw new SerializationException.NoCodecException("No codec available for tag " + tag);
    }
    return getDynamicCodecDescriptor(classNames.get(tagOffset));
  }

  /**
   * Creates a builder using the current contents of this registry.
   *
   * <p>This is much more efficient than scanning multiple times.
   */
  @VisibleForTesting
  public Builder getBuilder() {
    Builder builder = newBuilder();
    builder.setAllowDefaultCodec(allowDefaultCodec);
    for (Map.Entry<Class<?>, CodecDescriptor> entry : classMappedCodecs.entrySet()) {
      builder.add(entry.getValue().getCodec());
    }

    for (Object constant : referenceConstants) {
      builder.addReferenceConstant(constant);
    }

    for (Object constant : valueConstants) {
      builder.addValueConstant(constant);
    }

    for (String className : classNames) {
      builder.addClassName(className);
    }
    return builder;
  }

  ImmutableList<String> classNames() {
    return classNames;
  }

  /** Describes encoding logic. */
  interface CodecDescriptor {
    void serialize(SerializationContext context, Object obj, CodedOutputStream codedOut)
        throws IOException, SerializationException;

    Object deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws IOException, SerializationException;

    /**
     * Unique identifier for the associated codec.
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

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this).add("codec", codec).add("tag", tag).toString();
    }
  }

  /** Builder for {@link ObjectCodecRegistry}. */
  public static class Builder {
    private final Map<Class<?>, ObjectCodec<?>> codecs = new HashMap<>();
    private final ImmutableList.Builder<Object> referenceConstantsBuilder = ImmutableList.builder();
    private final ImmutableList.Builder<Object> valueConstantsBuilder = ImmutableList.builder();
    private final ImmutableSortedSet.Builder<String> classNames = ImmutableSortedSet.naturalOrder();
    private final ImmutableList.Builder<String> blacklistedClassNamePrefixes =
        ImmutableList.builder();
    private boolean allowDefaultCodec = true;

    /**
     * Adds the given codec. If a codec for this codec's encoded class already exists in the
     * registry, it is overwritten.
     */
    public Builder add(ObjectCodec<?> codec) {
      codecs.put(codec.getEncodedClass(), codec);
      return this;
    }

    /**
     * Set whether or not we allow fallback to java serialization when no matching codec is found.
     */
    public Builder setAllowDefaultCodec(boolean allowDefaultCodec) {
      this.allowDefaultCodec = allowDefaultCodec;
      return this;
    }

    /**
     * Adds a constant value by reference. Any value encountered during serialization which {@code
     * == object} will be replaced by {@code object} upon deserialization. Interned objects and
     * effective singletons are ideal for reference constants.
     *
     * <p>These constants should be interned or effectively interned: it should not be possible to
     * create objects that should be considered equal in which one has an element of this list and
     * the other does not, since that would break bit-for-bit equality of the objects' serialized
     * bytes when used in {@link com.google.devtools.build.skyframe.SkyKey}s.
     *
     * <p>Note that even {@link Boolean} does not satisfy this constraint, since {@code new
     * Boolean(true)} is allowed, but upon deserialization, when a {@code boolean} is boxed to a
     * {@link Boolean}, it will always be {@link Boolean#TRUE} or {@link Boolean#FALSE}.
     *
     * <p>The same is not true for an empty {@link ImmutableList}, since an empty non-{@link
     * ImmutableList} will not serialize to an {@link ImmutableList}, and so won't be deserialized
     * to an empty {@link ImmutableList}. If an object has a list field, and one codepath passes in
     * an empty {@link ArrayList} and another passes in an empty {@link ImmutableList}, and two
     * objects constructed in this way can be considered equal, then those two objects already do
     * not serialize bit-for-bit identical disregarding this list of constants, since the list
     * object's codec will be different for the two objects.
     */
    public Builder addReferenceConstant(Object object) {
      referenceConstantsBuilder.add(object);
      return this;
    }

    /**
     * Adds a constant value. Any value encountered during serialization which has the same class as
     * {@code object} and {@link Object#equals} {@code object} will be replaced by {@code object}
     * upon deserialization. These objects should therefore be indistinguishable, and unequal
     * objects should quickly compare unequal (it is ok for equal objects to be relatively expensive
     * to compare equal, if that is still less expensive than the cost of serializing the object).
     * Short {@link String} objects are ideal for value constants.
     */
    public Builder addValueConstant(Object object) {
      valueConstantsBuilder.add(object);
      return this;
    }

    public Builder addClassName(String className) {
      classNames.add(className);
      return this;
    }

    public Builder blacklistClassNamePrefix(String classNamePrefix) {
      blacklistedClassNamePrefixes.add(classNamePrefix);
      return this;
    }

    public ObjectCodecRegistry build() {
      return new ObjectCodecRegistry(
          ImmutableSet.copyOf(codecs.values()),
          referenceConstantsBuilder.build(),
          valueConstantsBuilder.build(),
          classNames.build(),
          blacklistedClassNamePrefixes.build(),
          allowDefaultCodec);
    }
  }

  private static int processCodecs(
      Iterable<? extends ObjectCodec<?>> memoizingCodecs,
      int nextTag,
      ImmutableList.Builder<CodecDescriptor> tagMappedCodecsBuilder,
      ImmutableMap.Builder<Class<?>, CodecDescriptor> codecsBuilder) {
    for (ObjectCodec<?> codec :
        ImmutableList.sortedCopyOf(
            Comparator.comparing(o -> o.getEncodedClass().getName()), memoizingCodecs)) {
      CodecDescriptor codecDescriptor = new TypedCodecDescriptor<>(nextTag++, codec);
      tagMappedCodecsBuilder.add(codecDescriptor);
      codecsBuilder.put(codec.getEncodedClass(), codecDescriptor);
      for (Class<?> otherClass : codec.additionalEncodedClasses()) {
        codecsBuilder.put(otherClass, codecDescriptor);
      }
    }
    return nextTag;
  }

  private static IdentityHashMap<String, Supplier<CodecDescriptor>> createDynamicCodecs(
      ImmutableList<String> classNames, int nextTag) {
    IdentityHashMap<String, Supplier<CodecDescriptor>> dynamicCodecs =
        new IdentityHashMap<>(classNames.size());
    for (String className : classNames) {
      int tag = nextTag++;
      dynamicCodecs.put(
          className, Suppliers.memoize(() -> createDynamicCodecDescriptor(tag, className)));
    }
    return dynamicCodecs;
  }

  private static boolean isAllowed(
      String className, ImmutableList<String> blacklistedClassNamePefixes) {
    for (String blacklistedClassNamePrefix : blacklistedClassNamePefixes) {
      if (className.startsWith(blacklistedClassNamePrefix)) {
        return false;
      }
    }
    return true;
  }

  /** For enums, this method must only be called for the declaring class. */
  private static CodecDescriptor createDynamicCodecDescriptor(int tag, String className) {
    try {
      Class<?> type = Class.forName(className);
      if (type.isEnum()) {
        return createCodecDescriptorForEnum(tag, type);
      }
      return new TypedCodecDescriptor<>(tag, new DynamicCodec(Class.forName(className)));
    } catch (ReflectiveOperationException e) {
      new SerializationException("Could not create codec for type: " + className, e)
          .printStackTrace();
      return null;
    }
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private static CodecDescriptor createCodecDescriptorForEnum(int tag, Class<?> enumType) {
    return new TypedCodecDescriptor(tag, new EnumCodec(enumType));
  }

  private CodecDescriptor getDynamicCodecDescriptor(String className)
      throws SerializationException.NoCodecException {
    Supplier<CodecDescriptor> supplier = dynamicCodecs.get(className);
    if (supplier == null) {
      throw new SerializationException.NoCodecException(
          "No default codec available for " + className);
    }
    CodecDescriptor descriptor = supplier.get();
    if (descriptor == null) {
      throw new SerializationException.NoCodecException(
          "There was a problem creating a codec for " + className + " check logs for details.");
    }
    return descriptor;
  }
}

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

import com.google.common.base.MoreObjects;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * Registry class for handling {@link ObjectCodec} mappings. Codecs are indexed by {@link String}
 * classifiers and assigned deterministic numeric identifiers for more compact on-the-wire
 * representation if desired.
 */
public class ObjectCodecRegistry {
  /** Creates a new, empty builder. */
  public static Builder newBuilder() {
    return new Builder();
  }

  private final boolean allowDefaultCodec;

  private final ConcurrentMap<Class<?>, CodecDescriptor> classMappedCodecs;
  private final ImmutableList<CodecDescriptor> tagMappedCodecs;

  private final int referenceConstantsStartTag;
  private final IdentityHashMap<Object, Integer> referenceConstantsMap;
  private final ImmutableList<Object> referenceConstants;

  /** This is sorted, but we need index-based access. */
  private final ImmutableList<String> classNames;

  private final IdentityHashMap<String, Supplier<CodecDescriptor>> dynamicCodecs;

  private ObjectCodecRegistry(
      ImmutableSet<ObjectCodec<?>> memoizingCodecs,
      ImmutableList<Object> referenceConstants,
      ImmutableSortedSet<String> classNames,
      ImmutableList<String> blacklistedClassNamePrefixes,
      boolean allowDefaultCodec) {
    this.allowDefaultCodec = allowDefaultCodec;

    int nextTag = 1; // 0 is reserved for null.
    this.classMappedCodecs =
        new ConcurrentHashMap<>(
            memoizingCodecs.size(), 0.75f, Runtime.getRuntime().availableProcessors());
    ImmutableList.Builder<CodecDescriptor> tagMappedMemoizingCodecsBuilder =
        ImmutableList.builderWithExpectedSize(memoizingCodecs.size());
    nextTag =
        processCodecs(memoizingCodecs, nextTag, tagMappedMemoizingCodecsBuilder, classMappedCodecs);
    this.tagMappedCodecs = tagMappedMemoizingCodecsBuilder.build();

    referenceConstantsStartTag = nextTag;
    referenceConstantsMap = new IdentityHashMap<>();
    for (Object constant : referenceConstants) {
      referenceConstantsMap.put(constant, nextTag++);
    }
    this.referenceConstants = referenceConstants;

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
      type = ((Enum<?>) obj).getDeclaringClass();
    }
    return getDynamicCodecDescriptor(type.getName(), type);
  }

  /**
   * Returns a {@link CodecDescriptor} for the given type or null if none found.
   *
   * <p>Also checks if there are codecs for a superclass of the given type.
   */
  private @Nullable CodecDescriptor getCodecDescriptor(Class<?> type) {
    for (Class<?> nextType = type; nextType != null; nextType = nextType.getSuperclass()) {
      CodecDescriptor result = classMappedCodecs.get(nextType);
      if (result != null) {
        if (nextType != type) {
          classMappedCodecs.put(type, result);
        }
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
    return null;
  }

  @Nullable
  Integer maybeGetTagForConstant(Object object) {
    return referenceConstantsMap.get(object);
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
    if (!allowDefaultCodec || tagOffset < 0 || tagOffset >= classNames.size()) {
      throw new SerializationException.NoCodecException("No codec available for tag " + tag);
    }
    return getDynamicCodecDescriptor(classNames.get(tagOffset), /*type=*/ null);
  }

  /**
   * Creates a builder using the current contents of this registry.
   *
   * <p>This is much more efficient than scanning multiple times.
   */
  public Builder getBuilder() {
    Builder builder = newBuilder();
    builder.setAllowDefaultCodec(allowDefaultCodec);
    for (Map.Entry<Class<?>, CodecDescriptor> entry : classMappedCodecs.entrySet()) {
      builder.add(entry.getValue().getCodec());
    }

    for (Object constant : referenceConstants) {
      builder.addReferenceConstant(constant);
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

    public Builder addReferenceConstants(Iterable<?> referenceConstants) {
      referenceConstantsBuilder.addAll(referenceConstants);
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
          classNames.build(),
          blacklistedClassNamePrefixes.build(),
          allowDefaultCodec);
    }
  }

  private static int processCodecs(
      Iterable<? extends ObjectCodec<?>> memoizingCodecs,
      int nextTag,
      ImmutableList.Builder<CodecDescriptor> tagMappedCodecsBuilder,
      ConcurrentMap<Class<?>, CodecDescriptor> codecsBuilder) {
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

  private CodecDescriptor getDynamicCodecDescriptor(String className, @Nullable Class<?> type)
      throws SerializationException.NoCodecException {
    Supplier<CodecDescriptor> supplier = dynamicCodecs.get(className);
    if (supplier != null) {
      CodecDescriptor descriptor = supplier.get();
      if (descriptor == null) {
        throw new SerializationException.NoCodecException(
            "There was a problem creating a codec for " + className + ". Check logs for details",
            type);
      }
      return descriptor;
    }
    if (type != null && LambdaCodec.isProbablyLambda(type)) {
      if (Serializable.class.isAssignableFrom(type)) {
        // LambdaCodec is hidden away as a codec for Serializable. This avoids special-casing it in
        // all places we look up a codec, and doesn't clash with anything else because Serializable
        // is an interface, not a class.
        return classMappedCodecs.get(Serializable.class);
      } else {
        throw new SerializationException.NoCodecException(
            "No default codec available for "
                + className
                + ". If this is a lambda, try casting it to (type & Serializable), like "
                + "(Supplier<String> & Serializable)",
            type);
      }
    }
    throw new SerializationException.NoCodecException(
        "No default codec available for " + className, type);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("allowDefaultCodec", allowDefaultCodec)
        .add("classMappedCodecs.size", classMappedCodecs.size())
        .add("tagMappedCodecs.size", tagMappedCodecs.size())
        .add("referenceConstantsStartTag", referenceConstantsStartTag)
        .add("referenceConstants.size", referenceConstants.size())
        .add("classNames.size", classNames.size())
        .add("dynamicCodecs.size", dynamicCodecs.size())
        .toString();
  }
}

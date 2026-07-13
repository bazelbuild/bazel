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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.Streams.stream;
import static java.util.Comparator.comparing;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Streams;
import com.google.common.io.ByteStreams;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.MessageLite;
import java.io.IOException;
import java.io.Serializable;
import java.security.DigestOutputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
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
  private final ImmutableList<CodecDescriptor> unstableTagMappedCodecs;
  private final ImmutableList<CodecDescriptor> stablePublicTagMappedCodecs;
  private final ImmutableList<CodecDescriptor> stablePrivateTagMappedCodecs;

  private final IdentityHashMap<Object, Integer> referenceConstantsMap;
  private final ImmutableList<Object> unstableReferenceConstants;
  private final ImmutableList<ConstantDescriptor> stablePublicReferenceConstants;
  private final ImmutableList<ConstantDescriptor> stablePrivateReferenceConstants;

  /** This is sorted, but we need index-based access. */
  private final ImmutableList<String> classNames;

  private final IdentityHashMap<String, Supplier<CodecDescriptor>> dynamicCodecs;

  @Nullable private final byte[] checksum;

  private ObjectCodecRegistry(
      ImmutableSet<ObjectCodec<?>> unstableCodecs,
      @Nullable ImmutableList<CodecDescriptor> stablePublicCodecs,
      @Nullable ImmutableList<CodecDescriptor> stablePrivateCodecs,
      ImmutableList<Object> unstableReferenceConstants,
      @Nullable ImmutableList<ConstantDescriptor> stablePublicReferenceConstants,
      @Nullable ImmutableList<ConstantDescriptor> stablePrivateReferenceConstants,
      ImmutableSortedSet<String> classNames,
      ImmutableList<String> excludedClassNamePrefixes,
      boolean allowDefaultCodec,
      boolean computeChecksum)
      throws IOException, NoSuchAlgorithmException {
    // Mimic what com.google.devtools.build.lib.util.Fingerprint does. Using it directly would
    // require untangling a circular dependency.
    MessageDigest messageDigest = null;
    CodedOutputStream checksum = null;
    if (computeChecksum) {
      messageDigest = MessageDigest.getInstance("SHA-256");
      checksum =
          CodedOutputStream.newInstance(
              new DigestOutputStream(ByteStreams.nullOutputStream(), messageDigest),
              /* bufferSize= */ 1024);
      checksum.writeBoolNoTag(allowDefaultCodec);
    }
    this.allowDefaultCodec = allowDefaultCodec;

    int nextTag = 1; // 0 is reserved for null.
    // Codec initialization.
    this.classMappedCodecs =
        new ConcurrentHashMap<>(
            unstableCodecs.size(), 0.75f, Runtime.getRuntime().availableProcessors());
    ImmutableList.Builder<CodecDescriptor> tagMappedMemoizingCodecsBuilder =
        ImmutableList.builderWithExpectedSize(unstableCodecs.size());
    nextTag =
        processCodecs(
            unstableCodecs, nextTag, tagMappedMemoizingCodecsBuilder, classMappedCodecs, checksum);
    this.unstableTagMappedCodecs = tagMappedMemoizingCodecsBuilder.build();

    if (stablePublicCodecs != null) {
      for (CodecDescriptor codec : stablePublicCodecs) {
        classMappedCodecs.put(codec.codec().getEncodedClass(), codec);
        for (Class<?> otherClass : codec.codec().additionalEncodedClasses()) {
          classMappedCodecs.put(otherClass, codec);
        }
      }
      this.stablePublicTagMappedCodecs = stablePublicCodecs;
    } else {
      this.stablePublicTagMappedCodecs = ImmutableList.of();
    }

    if (stablePrivateCodecs != null) {
      for (CodecDescriptor codec : stablePrivateCodecs) {
        classMappedCodecs.put(codec.codec().getEncodedClass(), codec);
        for (Class<?> otherClass : codec.codec().additionalEncodedClasses()) {
          classMappedCodecs.put(otherClass, codec);
        }
      }
      this.stablePrivateTagMappedCodecs = stablePrivateCodecs;
    } else {
      this.stablePrivateTagMappedCodecs = ImmutableList.of();
    }
    // Reference constant initialization.
    referenceConstantsMap = new IdentityHashMap<>();
    {
      int unstableConstantTag = 0;
      for (Object constant : unstableReferenceConstants) {
        referenceConstantsMap.put(
            constant, WireType.ConstantWireType.UNSTABLE.getTypedTagNumber(unstableConstantTag));
        addToChecksum(checksum, unstableConstantTag, constant.getClass().getName());
        unstableConstantTag++;
      }
      this.unstableReferenceConstants = unstableReferenceConstants;
    }
    // If an unstableReferenceConstant also appears in a stable reference list, the stable tag will
    // be used.
    if (stablePublicReferenceConstants != null) {
      for (ConstantDescriptor constant : stablePublicReferenceConstants) {
        referenceConstantsMap.put(constant.constant(), constant.getTypedTagNumber());
      }
      this.stablePublicReferenceConstants = stablePublicReferenceConstants;
    } else {
      this.stablePublicReferenceConstants = ImmutableList.of();
    }
    if (stablePrivateReferenceConstants != null) {
      for (ConstantDescriptor constant : stablePrivateReferenceConstants) {
        referenceConstantsMap.put(constant.constant(), constant.getTypedTagNumber());
      }
      this.stablePrivateReferenceConstants = stablePrivateReferenceConstants;
    } else {
      this.stablePrivateReferenceConstants = ImmutableList.of();
    }
    this.classNames =
        classNames.stream()
            .filter((str) -> isAllowed(str, excludedClassNamePrefixes))
            .collect(toImmutableList());
    this.dynamicCodecs = createDynamicCodecs(this.classNames, nextTag, checksum);
    if (computeChecksum) {
      checksum.flush();
      this.checksum = messageDigest.digest();
    } else {
      this.checksum = null;
    }
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
  @Nullable
  private CodecDescriptor getCodecDescriptor(Class<?> type) {
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
  Object maybeGetConstantByTag(int typedTag) throws SerializationException {
    int wireType = WireType.getWireTypeIndex(typedTag);
    // Micro optimization: switching on compile-time constants to get a jump table.
    return switch (wireType) {
      case WireType.UNSTABLE_CONSTANT_VALUE -> {
        int tag = WireType.getTagNumber(typedTag);
        if (tag < unstableReferenceConstants.size()) {
          yield unstableReferenceConstants.get(tag);
        }
        yield null;
      }
      case WireType.STABLE_CONSTANT_VALUE ->
          getStableReferenceConstantValue(stablePublicReferenceConstants, typedTag);
      case WireType.STABLE_PRIVATE_CONSTANT_VALUE ->
          getStableReferenceConstantValue(stablePrivateReferenceConstants, typedTag);
      case WireType.BACKREFERENCE_VALUE,
          WireType.UNSTABLE_CODEC_VALUE,
          WireType.STABLE_CODEC_VALUE,
          WireType.STABLE_PRIVATE_CODEC_VALUE ->
          null;
      default -> {
        throw new SerializationException("Unknown wire type: " + wireType);
      }
    };
  }

  @Nullable
  public Integer maybeGetTagForConstant(Object object) {
    @Nullable Integer tag = referenceConstantsMap.get(object);
    if (tag == null) {
      return null;
    }
    return tag;
  }

  /** Returns the {@link CodecDescriptor} associated with the supplied tag. */
  CodecDescriptor getCodecDescriptorByTag(int typedTag)
      throws SerializationException.NoCodecException {
    int wireType = WireType.getWireTypeIndex(typedTag);
    int tag = WireType.getTagNumber(typedTag);
    // Micro optimization: switching on compile-time constants to get a jump table.
    return switch (wireType) {
      case WireType.UNSTABLE_CODEC_VALUE -> {
        int tagOffset = tag - 1; // 0 reserved for null
        if (tagOffset < 0) {
          throw new SerializationException.NoCodecException("No codec available for tag " + tag);
        }
        if (tagOffset < unstableTagMappedCodecs.size()) {
          yield unstableTagMappedCodecs.get(tagOffset);
        }
        tagOffset -= unstableTagMappedCodecs.size();
        if (!allowDefaultCodec || tagOffset < 0 || tagOffset >= classNames.size()) {
          throw new SerializationException.NoCodecException(
              "No unstable codec available for tag " + tag);
        }
        yield getDynamicCodecDescriptor(classNames.get(tagOffset), /* type= */ null);
      }
      case WireType.STABLE_CODEC_VALUE ->
          getCodecDescriptorForTag(stablePublicTagMappedCodecs, tag);
      case WireType.STABLE_PRIVATE_CODEC_VALUE ->
          getCodecDescriptorForTag(stablePrivateTagMappedCodecs, tag);
      case WireType.BACKREFERENCE_VALUE,
          WireType.UNSTABLE_CONSTANT_VALUE,
          WireType.STABLE_CONSTANT_VALUE,
          WireType.STABLE_PRIVATE_CONSTANT_VALUE ->
          throw new SerializationException.NoCodecException(
              "called getCodecDescriptorByTag for WireType: " + WireType.fromValue(wireType));
      default ->
          throw new SerializationException.NoCodecException(
              "No codec available for tag " + tag + " with wire type " + wireType);
    };
  }

  /**
   * Returns a checksum computed from the tag mappings that make up this registry.
   *
   * <p>The checksum can be used to ensure consistent serialization semantics across servers.
   *
   * <p>Returns {@code null} if this instance was not configured to compute a checksum via {@link
   * Builder#computeChecksum(boolean)}.
   */
  @Nullable
  public byte[] getChecksum() {
    return checksum == null ? null : checksum.clone();
  }

  /**
   * Creates a builder using the current contents of this registry.
   *
   * <p>This is much more efficient than scanning multiple times.
   */
  public Builder getBuilder() {
    Builder builder = newBuilder();
    builder.setAllowDefaultCodec(allowDefaultCodec);
    for (CodecDescriptor codecDescriptor : classMappedCodecs.values()) {
      if (codecDescriptor.wireType() == WireType.CodecWireType.UNSTABLE) {
        builder.add(codecDescriptor.codec());
      }
    }
    if (!stablePublicTagMappedCodecs.isEmpty()) {
      builder.setStablePublicCodecDescriptors(stablePublicTagMappedCodecs);
    }
    if (!stablePrivateTagMappedCodecs.isEmpty()) {
      builder.setStablePrivateCodecDescriptors(stablePrivateTagMappedCodecs);
    }

    for (Object constant : unstableReferenceConstants) {
      builder.addReferenceConstant(constant);
    }
    if (!stablePublicReferenceConstants.isEmpty()) {
      builder.setStablePublicReferenceConstantDescriptors(stablePublicReferenceConstants);
    }
    if (!stablePrivateReferenceConstants.isEmpty()) {
      builder.setStablePrivateReferenceConstantDescriptors(stablePrivateReferenceConstants);
    }

    for (String className : classNames) {
      builder.addClassName(className);
    }
    return builder;
  }

  ImmutableList<String> classNames() {
    return classNames;
  }

  /** Returns the codec value associated with the given tag, checking for unused tag numbers. */
  private static CodecDescriptor getCodecDescriptorForTag(
      ImmutableList<CodecDescriptor> tagMappedCodecs, int tag)
      throws SerializationException.NoCodecException {
    if (tag < tagMappedCodecs.size()) {
      return tagMappedCodecs.get(tag);
    }
    throw new SerializationException.NoCodecException("No codec available for tag " + tag);
  }

  /** Returns the constant value associated with the given tag, checking for unused tag numbers. */
  private static Object getStableReferenceConstantValue(
      ImmutableList<ConstantDescriptor> stableReferenceConstantList, int typedTag)
      throws SerializationException {
    int tag = WireType.getTagNumber(typedTag);
    if (tag >= stableReferenceConstantList.size()) {
      throw new SerializationException("Unknown constant tag: " + tag);
    }
    return stableReferenceConstantList.get(tag).constant();
  }

  /**
   * Describes encoding logic.
   *
   * @param wireType The codec wire type of the codec.
   * @param tag Unique identifier for the associated codec. Intended to be used as a compact
   *     on-the-wire representation of an encoded object's type. Returns a value ≥ 1. 0 is a special
   *     tag representing null.
   * @param codec The underlying codec.
   */
  record CodecDescriptor(WireType.CodecWireType wireType, int tag, ObjectCodec<?> codec) {
    CodecDescriptor {
      // Check that the tag is not a reserved value.
      int minTag = wireType == WireType.CodecWireType.UNSTABLE ? 1 : 0;
      Preconditions.checkArgument(tag >= minTag);
    }

    /** Returns the combined wire type and tag number for this codec descriptor. */
    int getTypedTagNumber() {
      return wireType.getTypedTagNumber(tag);
    }
  }

  /**
   * Describes encoding logic.
   *
   * @param wireType The wire type of the reference constant.
   * @param tag Unique identifier for the associated constant. Intended to be used as a compact
   *     on-the-wire representation of an encoded object's type. *May* be zero, unlike in {@link
   *     CodecDescriptor}.
   * @param constant The underlying constant.
   */
  public record ConstantDescriptor(WireType.ConstantWireType wireType, int tag, Object constant) {

    public ConstantDescriptor {
      // Check that the tag is not a reserved value.
      Preconditions.checkArgument(tag >= 0);
    }

    /** Returns the combined wire type and tag number for this constant descriptor. */
    int getTypedTagNumber() {
      return wireType.getTypedTagNumber(tag);
    }
  }

  /** Builder for {@link ObjectCodecRegistry}. */
  public static class Builder {
    private final Map<Class<?>, ObjectCodec<?>> unstableCodecs = new HashMap<>();
    private ImmutableList<CodecDescriptor> stablePublicCodecs = null;
    private ImmutableList<CodecDescriptor> stablePrivateCodecs = null;
    private final ImmutableList.Builder<Object> unstableReferenceConstantsBuilder =
        ImmutableList.builder();
    private ImmutableList<ConstantDescriptor> stablePublicReferenceConstants = null;
    private ImmutableList<ConstantDescriptor> stablePrivateReferenceConstants = null;
    private final ImmutableSortedSet.Builder<String> classNames = ImmutableSortedSet.naturalOrder();
    private final ImmutableList.Builder<String> excludedClassNamePrefixes = ImmutableList.builder();
    private boolean allowDefaultCodec = true;
    private boolean computeChecksum = false;

    /**
     * Adds the given codec. If a codec for this codec's encoded class already exists in the
     * registry, it is overwritten.
     */
    @CanIgnoreReturnValue
    public Builder add(ObjectCodec<?> codec) {
      unstableCodecs.put(codec.getEncodedClass(), codec);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setStablePublicCodecs(List<ObjectCodec<?>> newStablePublicCodecs) {
      return setStablePublicCodecDescriptors(
          makeCodecDescriptors(newStablePublicCodecs, WireType.CodecWireType.STABLE_PUBLIC));
    }

    @CanIgnoreReturnValue
    public Builder setStablePrivateCodecs(List<ObjectCodec<?>> newStablePrivateCodecs) {
      return setStablePrivateCodecDescriptors(
          makeCodecDescriptors(newStablePrivateCodecs, WireType.CodecWireType.STABLE_PRIVATE));
    }

    @CanIgnoreReturnValue
    Builder setStablePublicCodecDescriptors(ImmutableList<CodecDescriptor> newStablePublicCodecs) {
      checkState(stablePublicCodecs == null, "Cannot set stable public codec descriptors twice");
      newStablePublicCodecs.forEach(
          c -> checkArgument(c.wireType() == WireType.CodecWireType.STABLE_PUBLIC));
      stablePublicCodecs = newStablePublicCodecs;
      return this;
    }

    @CanIgnoreReturnValue
    Builder setStablePrivateCodecDescriptors(
        ImmutableList<CodecDescriptor> newStablePrivateCodecs) {
      checkState(stablePrivateCodecs == null, "Cannot set stable private codec descriptors twice");
      newStablePrivateCodecs.forEach(
          c -> checkArgument(c.wireType() == WireType.CodecWireType.STABLE_PRIVATE));
      stablePrivateCodecs = newStablePrivateCodecs;
      return this;
    }

    private static ImmutableList<CodecDescriptor> makeCodecDescriptors(
        List<ObjectCodec<?>> referenceCodecs, WireType.CodecWireType wireType) {
      return Streams.mapWithIndex(
              referenceCodecs.stream(),
              (o, i) -> new CodecDescriptor(wireType, (int) i, checkNotNull(o)))
          .collect(toImmutableList());
    }

    /**
     * Set whether or not we allow fallback to java serialization when no matching codec is found.
     */
    @CanIgnoreReturnValue
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
    @CanIgnoreReturnValue
    public Builder addReferenceConstant(Object object) {
      unstableReferenceConstantsBuilder.add(object);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addReferenceConstants(Iterable<?> referenceConstants) {
      unstableReferenceConstantsBuilder.addAll(referenceConstants);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setStablePublicReferenceConstants(
        List<Object> newStablePublicReferenceConstants) {
      return setStablePublicReferenceConstantDescriptors(
          makeConstantDescriptors(
              newStablePublicReferenceConstants, WireType.ConstantWireType.STABLE_PUBLIC));
    }

    @CanIgnoreReturnValue
    public Builder setStablePrivateReferenceConstants(
        List<Object> newStablePrivateReferenceConstants) {
      return setStablePrivateReferenceConstantDescriptors(
          makeConstantDescriptors(
              newStablePrivateReferenceConstants, WireType.ConstantWireType.STABLE_PRIVATE));
    }

    @CanIgnoreReturnValue
    Builder setStablePublicReferenceConstantDescriptors(
        ImmutableList<ConstantDescriptor> newStablePublicReferenceConstants) {
      checkState(
          stablePublicReferenceConstants == null,
          "Cannot set stable public reference constant descriptors twice");
      newStablePublicReferenceConstants.forEach(
          c -> checkArgument(c.wireType() == WireType.ConstantWireType.STABLE_PUBLIC));
      stablePublicReferenceConstants = newStablePublicReferenceConstants;
      return this;
    }

    @CanIgnoreReturnValue
    Builder setStablePrivateReferenceConstantDescriptors(
        ImmutableList<ConstantDescriptor> newStablePrivateReferenceConstants) {
      checkState(
          stablePrivateReferenceConstants == null,
          "Cannot set stable private reference constant descriptors twice");
      newStablePrivateReferenceConstants.forEach(
          c -> checkArgument(c.wireType() == WireType.ConstantWireType.STABLE_PRIVATE));
      stablePrivateReferenceConstants = newStablePrivateReferenceConstants;
      return this;
    }

    private static ImmutableList<ConstantDescriptor> makeConstantDescriptors(
        List<Object> referenceConstants, WireType.ConstantWireType wireType) {
      return Streams.mapWithIndex(
              referenceConstants.stream(),
              (o, i) -> new ConstantDescriptor(wireType, (int) i, checkNotNull(o)))
          .collect(toImmutableList());
    }

    @CanIgnoreReturnValue
    public Builder addClassName(String className) {
      classNames.add(className);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder excludeClassNamePrefix(String classNamePrefix) {
      excludedClassNamePrefixes.add(classNamePrefix);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder computeChecksum(boolean computeChecksum) {
      this.computeChecksum = computeChecksum;
      return this;
    }

    public ObjectCodecRegistry build() {
      var classNamesBuilt = classNames.build();
      try {
        return new ObjectCodecRegistry(
            ImmutableSet.copyOf(unstableCodecs.values()),
            stablePublicCodecs,
            stablePrivateCodecs,
            unstableReferenceConstantsBuilder.build(),
            stablePublicReferenceConstants,
            stablePrivateReferenceConstants,
            classNamesBuilt,
            excludedClassNamePrefixes.build(),
            allowDefaultCodec,
            computeChecksum);
      } catch (IOException | NoSuchAlgorithmException e) {
        throw new IllegalStateException("Unexpected exception while building codec registry", e);
      }
    }
  }

  private static int processCodecs(
      Iterable<? extends ObjectCodec<?>> memoizingCodecs,
      int nextTag,
      ImmutableList.Builder<CodecDescriptor> tagMappedCodecsBuilder,
      ConcurrentMap<Class<?>, CodecDescriptor> codecsBuilder,
      @Nullable CodedOutputStream checksum)
      throws IOException {
    // First, register all codecs and their monotonically increasing tag numbers in a stable
    // alphabetic sort order, using their primary encoded class as the key.
    var sortedCodecDescriptors =
        Streams.mapWithIndex(
                // Sort the codecs by their primary encoded class name.
                stream(memoizingCodecs).sorted(comparing(o -> o.getEncodedClass().getName())),
                // Then create a codec descriptor for each codec.
                (codec, idx) ->
                    // idx is small enough to be casted from long to int without loss of
                    // information.
                    new CodecDescriptor(
                        WireType.CodecWireType.UNSTABLE, (int) idx + nextTag, codec))
            .collect(toImmutableList());

    // Then, perform checksumming and check that there's a unique codec descriptor for each encoded
    // class.
    for (CodecDescriptor codecDescriptor : sortedCodecDescriptors) {
      addToChecksum(checksum, codecDescriptor.tag(), codecDescriptor.codec().getClass().getName());

      CodecDescriptor previousCodecDescriptor =
          codecsBuilder.put(codecDescriptor.codec().getEncodedClass(), codecDescriptor);
      Preconditions.checkState(
          previousCodecDescriptor == null,
          "found duplicate codec descriptor for %s, was: %s, new: %s",
          codecDescriptor.codec().getEncodedClass(),
          previousCodecDescriptor,
          codecDescriptor);
    }

    // Finally, for all codec descriptors, map their additional encoded classes, and overwrite
    // any existing descriptor mappings.
    for (CodecDescriptor codecDescriptor : sortedCodecDescriptors) {
      for (Class<?> otherClass : codecDescriptor.codec().additionalEncodedClasses()) {
        codecsBuilder.put(otherClass, codecDescriptor);
      }
    }

    // Append all new descriptors into the builder.
    tagMappedCodecsBuilder.addAll(sortedCodecDescriptors);

    return nextTag + sortedCodecDescriptors.size();
  }

  private static IdentityHashMap<String, Supplier<CodecDescriptor>> createDynamicCodecs(
      ImmutableList<String> classNames, int nextTag, @Nullable CodedOutputStream checksum)
      throws IOException {
    IdentityHashMap<String, Supplier<CodecDescriptor>> dynamicCodecs =
        new IdentityHashMap<>(classNames.size());
    for (String className : classNames) {
      int tag = nextTag++;
      dynamicCodecs.put(
          className, Suppliers.memoize(() -> createDynamicCodecDescriptor(tag, className)));
      addToChecksum(checksum, tag, className);
    }
    return dynamicCodecs;
  }

  private static void addToChecksum(@Nullable CodedOutputStream checksum, int tag, String className)
      throws IOException {
    if (checksum != null) {
      checksum.writeInt32NoTag(tag);

      // Trim class names of lambdas to the enclosing class. The lambda class itself is named
      // nondeterministically.
      int lambdaIndex = className.indexOf("$$Lambda");
      if (lambdaIndex != -1) {
        className = className.substring(0, lambdaIndex);
      }
      checksum.writeStringNoTag(className);
    }
  }

  private static boolean isAllowed(
      String className, ImmutableList<String> excludedClassNamePefixes) {
    for (String excludedClassNamePrefix : excludedClassNamePefixes) {
      if (className.startsWith(excludedClassNamePrefix)) {
        return false;
      }
    }
    return true;
  }

  /** For enums, this method must only be called for the declaring class. */
  @Nullable
  private static CodecDescriptor createDynamicCodecDescriptor(int tag, String className) {
    try {
      Class<?> type = Class.forName(className);
      if (type.isEnum()) {
        return createCodecDescriptorForEnum(tag, type);
      }
      if (MessageLite.class.isAssignableFrom(type)) {
        return createCodecDescriptorForProto(tag, type);
      }
      return new CodecDescriptor(WireType.CodecWireType.UNSTABLE, tag, new DynamicCodec(type));
    } catch (ReflectiveOperationException e) {
      new SerializationException("Could not create codec for type: " + className, e)
          .printStackTrace();
      return null;
    }
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private static CodecDescriptor createCodecDescriptorForEnum(int tag, Class<?> enumType) {
    return new CodecDescriptor(WireType.CodecWireType.UNSTABLE, tag, new EnumCodec(enumType));
  }

  @SuppressWarnings("unchecked")
  private static CodecDescriptor createCodecDescriptorForProto(int tag, Class<?> protoType) {
    return new CodecDescriptor(
        WireType.CodecWireType.UNSTABLE,
        tag,
        new MessageLiteCodec((Class<? extends MessageLite>) protoType));
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
        .add("checksum", checksum)
        .add("allowDefaultCodec", allowDefaultCodec)
        .add("classMappedCodecs.size", classMappedCodecs.size())
        .add("unstableTagMappedCodecs.size", unstableTagMappedCodecs.size())
        .add("stablePublicTagMappedCodecs.size", stablePublicTagMappedCodecs.size())
        .add("stablePrivateTagMappedCodecs.size", stablePrivateTagMappedCodecs.size())
        .add("referenceConstants.size", unstableReferenceConstants.size())
        .add("stablePublicReferenceConstants.size", stablePublicReferenceConstants.size())
        .add("stablePrivateReferenceConstants.size", stablePrivateReferenceConstants.size())
        .add("classNames.size", classNames.size())
        .add("dynamicCodecs.size", dynamicCodecs.size())
        .toString();
  }
}

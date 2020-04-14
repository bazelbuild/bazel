/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.langmodel;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableMap;
import java.util.Arrays;
import java.util.Collection;
import org.objectweb.asm.Type;

/**
 * Represents the identifiable name of a Java class or interface with convenient conversions among
 * different names.
 */
@AutoValue
public abstract class ClassName implements TypeMappable<ClassName> {

  public static final String IN_PROCESS_LABEL = "__desugar__/";

  private static final String IMMUTABLE_LABEL_LABEL = "__final__/";

  private static final String TYPE_ADAPTER_PACKAGE_ROOT =
      "com/google/devtools/build/android/desugar/typeadapter/";

  public static final TypeMapper IN_PROCESS_LABEL_STRIPPER =
      new TypeMapper(className -> className.stripPackagePrefix(IN_PROCESS_LABEL));

  public static final TypeMapper IMMUTABLE_LABEL_STRIPPER =
      new TypeMapper(className -> className.stripPackagePrefix(IMMUTABLE_LABEL_LABEL));

  private static final String TYPE_ADAPTER_SUFFIX = "Adapter";

  private static final String TYPE_CONVERTER_SUFFIX = "Converter";

  /**
   * The primitive type as specified at
   * https://docs.oracle.com/javase/specs/jvms/se11/html/jvms-2.html#jvms-2.3
   */
  private static final ImmutableMap<String, Type> PRIMITIVES_TYPES =
      ImmutableMap.<String, Type>builder()
          .put("V", Type.VOID_TYPE)
          .put("Z", Type.BOOLEAN_TYPE)
          .put("C", Type.CHAR_TYPE)
          .put("B", Type.BYTE_TYPE)
          .put("S", Type.SHORT_TYPE)
          .put("I", Type.INT_TYPE)
          .put("F", Type.FLOAT_TYPE)
          .put("J", Type.LONG_TYPE)
          .put("D", Type.DOUBLE_TYPE)
          .build();
  /**
   * The primitive type as specified at
   * https://docs.oracle.com/javase/specs/jvms/se11/html/jvms-2.html#jvms-2.3
   */
  private static final ImmutableMap<ClassName, ClassName> PRIMITIVES_TO_BOXED_TYPES =
      ImmutableMap.<ClassName, ClassName>builder()
          .put(ClassName.create(Type.VOID_TYPE), ClassName.create("java/lang/Void"))
          .put(ClassName.create(Type.BOOLEAN_TYPE), ClassName.create("java/lang/Boolean"))
          .put(ClassName.create(Type.CHAR_TYPE), ClassName.create("java/lang/Character"))
          .put(ClassName.create(Type.BYTE_TYPE), ClassName.create("java/lang/Byte"))
          .put(ClassName.create(Type.SHORT_TYPE), ClassName.create("java/lang/Short"))
          .put(ClassName.create(Type.INT_TYPE), ClassName.create("java/lang/Integer"))
          .put(ClassName.create(Type.FLOAT_TYPE), ClassName.create("java/lang/Float"))
          .put(ClassName.create(Type.LONG_TYPE), ClassName.create("java/lang/Long"))
          .put(ClassName.create(Type.DOUBLE_TYPE), ClassName.create("java/lang/Double"))
          .build();

  private static final ImmutableBiMap<String, String> SHADOWED_TO_MIRRORED_TYPE_PREFIX_MAPPINGS =
      ImmutableBiMap.<String, String>builder()
          .put("java/time/", "j$/time/")
          .put("java/lang/Double8", "j$/lang/Double8")
          .put("java/lang/Integer8", "j$/lang/Integer8")
          .put("java/lang/Long8", "j$/lang/Long8")
          .put("java/lang/Math8", "j$/lang/Math8")
          .put("java/io/Desugar", "j$/io/Desugar")
          .put("java/io/UncheckedIOException", "j$/io/UncheckedIOException")
          .put("java/util/stream/", "j$/util/stream/")
          .put("java/util/function/", "j$/util/function/")
          .put("java/util/Desugar", "j$/util/Desugar")
          .put("java/util/DoubleSummaryStatistics", "j$/util/DoubleSummaryStatistics")
          .put("java/util/IntSummaryStatistics", "j$/util/IntSummaryStatistics")
          .put("java/util/LongSummaryStatistics", "j$/util/LongSummaryStatistics")
          .put("java/util/Objects", "j$/util/Objects")
          .put("java/util/Optional", "j$/util/Optional")
          .put("java/util/PrimitiveIterator", "j$/util/PrimitiveIterator")
          .put("java/util/Spliterator", "j$/util/Spliterator")
          .put("java/util/StringJoiner", "j$/util/StringJoiner")
          .put("java/util/concurrent/ConcurrentHashMap", "j$/util/concurrent/ConcurrentHashMap")
          .put("java/util/concurrent/ThreadLocalRandom", "j$/util/concurrent/ThreadLocalRandom")
          .put(
              "java/util/concurrent/atomic/DesugarAtomic",
              "j$/util/concurrent/atomic/DesugarAtomic")
          .put("javadesugar/testing/", "jd$/testing/")
          .build();

  public static final TypeMapper SHADOWED_TO_MIRRORED_TYPE_MAPPER =
      new TypeMapper(ClassName::shadowedToMirrored);
  public static final TypeMapper IMMUTABLE_LABEL_ATTACHER =
      new TypeMapper(ClassName::withCoreTypeImmutableLabel);

  public static ClassName create(String binaryName) {
    checkArgument(
        !binaryName.contains("."),
        "Expected a binary/internal class name ('/'-delimited) instead of a qualified name."
            + " Actual: (%s)",
        binaryName);
    return new AutoValue_ClassName(binaryName);
  }

  public static ClassName create(Class<?> clazz) {
    return create(Type.getType(clazz));
  }

  public static ClassName create(Type asmType) {
    return create(asmType.getInternalName());
  }

  public static ClassName fromClassFileName(String fileName) {
    checkArgument(
        fileName.endsWith(".class"), "Expected a class file (*.class). Actual: (%s).", fileName);
    return ClassName.create(fileName.substring(0, fileName.length() - ".class".length()));
  }

  private static void checkPackagePrefixFormat(String prefix) {
    checkArgument(
        prefix.isEmpty() || prefix.endsWith("/"),
        "Expected (%s) to be a package prefix of ending with '/'.",
        prefix);
    checkArgument(
        !prefix.contains("."),
        "Expected a '/'-delimited binary name instead of a '.'-delimited qualified name for %s",
        prefix);
  }

  /**
   * The textual binary name used to index the class name, as defined at,
   * https://docs.oracle.com/javase/specs/jvms/se11/html/jvms-4.html#jvms-4.2.1
   */
  public abstract String binaryName();

  public final Type toAsmObjectType() {
    return isPrimitive() ? PRIMITIVES_TYPES.get(binaryName()) : Type.getObjectType(binaryName());
  }

  public final ClassName toBoxedType() {
    checkState(isPrimitive(), "Expected a primitive type for type boxing, but got %s", this);
    return PRIMITIVES_TO_BOXED_TYPES.get(this);
  }

  public final boolean isPrimitive() {
    return PRIMITIVES_TYPES.containsKey(binaryName());
  }

  public final boolean isWideType() {
    return "D".equals(binaryName()) || "J".equals(binaryName());
  }

  public final boolean isBoxedType() {
    return PRIMITIVES_TO_BOXED_TYPES.containsValue(this);
  }

  public final String qualifiedName() {
    return binaryName().replace('/', '.');
  }

  public ClassName innerClass(String innerClassSimpleName) {
    return ClassName.create(binaryName() + '$' + innerClassSimpleName);
  }

  public final String getPackageName() {
    String binaryName = binaryName();
    int i = binaryName.lastIndexOf('/');
    return i < 0 ? "" : binaryName.substring(0, i + 1);
  }

  public final String simpleName() {
    String binaryName = binaryName();
    int i = binaryName.lastIndexOf('/');
    return i < 0 ? binaryName : binaryName.substring(i + 1);
  }

  public final ClassName withSimpleNameSuffix(String suffix) {
    return ClassName.create(binaryName() + suffix);
  }

  public final String classFilePathName() {
    return binaryName() + ".class";
  }

  public final boolean hasInProcessLabel() {
    return hasPackagePrefix(IN_PROCESS_LABEL);
  }

  public final boolean hasImmutableLabel() {
    return hasPackagePrefix(IMMUTABLE_LABEL_LABEL);
  }

  /**
   * Returns a new instance of {@link ClassName} that represents the owner class of a single adapter
   * method for an Android SDK API.
   *
   * <p>The implementation has to guarantee generating different class names for different target
   * methods to be adapted, including overloaded API methods, in order to avoid adapter class name
   * clashing from separate compilation units.
   */
  final ClassName typeAdapterOwner(String encodedMethodTag) {
    checkState(
        !hasInProcessLabel() && !hasImmutableLabel(),
        "Expected a label-free type: Actual(%s)",
        this);
    checkState(
        isInPackageEligibleForTypeAdapter(),
        "Expected an Android SDK type to have an adapter: Actual (%s)",
        this);
    String binaryName =
        String.format(
            "%s%s$%x$%s",
            TYPE_ADAPTER_PACKAGE_ROOT,
            binaryName(),
            encodedMethodTag.hashCode(),
            TYPE_ADAPTER_SUFFIX);
    return ClassName.create(binaryName);
  }

  /**
   * Returns a new instance of {@code ClassName} that represents the owner class with conversion
   * methods between JDK built-in types and desguar-mirrored types.
   */
  public final ClassName typeConverterOwner() {
    checkState(
        !hasInProcessLabel() && !hasImmutableLabel(),
        "Expected a label-free type: Actual(%s)",
        this);
    checkState(
        isDesugarShadowedType(),
        "Expected an JDK built-in type to have an converter: Actual (%s)",
        this);
    return withSimpleNameSuffix(TYPE_CONVERTER_SUFFIX).withPackagePrefix(TYPE_ADAPTER_PACKAGE_ROOT);
  }

  /**
   * Returns a new instance of {@code ClassName} attached with an immutable label which marks the
   * type is not subject to further desugar operations until the final label striping.
   */
  public final ClassName withCoreTypeImmutableLabel() {
    return isDesugarShadowedType() ? withPackagePrefix(IMMUTABLE_LABEL_LABEL) : this;
  }

  /**
   * Returns a new instance of {@code ClassName} that is the desugar-mirrored core type (e.g. {@code
   * j$/time/MonthDay}) of the current shadowed built-in core type, assuming {@code this} instance
   * is a desugared-shadowed built-in core type.
   */
  public final ClassName shadowedToMirrored() {
    return SHADOWED_TO_MIRRORED_TYPE_PREFIX_MAPPINGS.keySet().stream()
        .filter(this::hasPackagePrefix)
        .map(
            prefix ->
                replacePackagePrefix(prefix, SHADOWED_TO_MIRRORED_TYPE_PREFIX_MAPPINGS.get(prefix)))
        .findAny()
        .orElse(this);
  }

  /**
   * Returns a new instance of {@code ClassName} that is a shadowed built-in core type (e.g. {@code
   * java/time/MonthDay}) of the current desugar-mirrored core type, assuming {@code this} instance
   * is a desugar-mirrored core type.
   */
  public final ClassName mirroredToShadowed() {
    ImmutableBiMap<String, String> verbatimTypeMappings =
        SHADOWED_TO_MIRRORED_TYPE_PREFIX_MAPPINGS.inverse();
    return verbatimTypeMappings.keySet().stream()
        .filter(this::hasPackagePrefix)
        .map(prefix -> replacePackagePrefix(prefix, verbatimTypeMappings.get(prefix)))
        .findAny()
        .orElse(this);
  }

  public final ClassName withPackagePrefix(String prefix) {
    checkPackagePrefixFormat(prefix);
    return ClassName.create(prefix + binaryName());
  }

  public final boolean hasPackagePrefix(String prefix) {
    return binaryName().startsWith(prefix);
  }

  public final boolean hasAnyPackagePrefix(String... prefixes) {
    return Arrays.stream(prefixes).anyMatch(this::hasPackagePrefix);
  }

  public final boolean hasAnyPackagePrefix(Collection<String> prefixes) {
    return prefixes.stream().anyMatch(this::hasPackagePrefix);
  }

  public final boolean isDesugarEligible() {
    return !isInDesugarRuntimeLibrary();
  }

  public final boolean isInPackageEligibleForTypeAdapter() {
    // TODO(b/152573900): Update to hasPackagePrefix("android/") once all package-wise incremental
    // rollouts are complete.

    return hasAnyPackagePrefix(
        "android/testing/",
        "android/accessibilityservice/AccessibilityService",
        "android/app/admin/FreezePeriod",
        "android/app/role/RoleManager",
        "android/app/usage/UsageStatsManager",
        "android/hardware/display/AmbientBrightnessDayStats",
        "android/os/SystemClock",
        "android/service/voice/VoiceInteractionSession",
        "android/service/voice/VoiceInteractionSession",
        "android/telephony/SubscriptionPlan$Builder",
        "android/telephony/TelephonyManager",
        "android/view/textclassifier/ConversationActions$Message",
        "android/view/textclassifier/TextClassification$Request",
        "android/view/textclassifier/TextLinks");
  }

  public final boolean isInDesugarRuntimeLibrary() {
    return hasAnyPackagePrefix(
        "com/google/devtools/build/android/desugar/runtime/", TYPE_ADAPTER_PACKAGE_ROOT);
  }

  public final boolean isDesugarShadowedType() {
    return hasAnyPackagePrefix(SHADOWED_TO_MIRRORED_TYPE_PREFIX_MAPPINGS.keySet());
  }

  public final boolean isDesugarMirroredType() {
    return hasAnyPackagePrefix(SHADOWED_TO_MIRRORED_TYPE_PREFIX_MAPPINGS.values());
  }

  private ClassName stripPackagePrefix(String prefix) {
    return hasPackagePrefix(prefix) ? stripRequiredPackagePrefix(prefix) : this;
  }

  private ClassName stripRequiredPackagePrefix(String prefix) {
    return replacePackagePrefix(/* originalPrefix= */ prefix, /* targetPrefix= */ "");
  }

  private ClassName replacePackagePrefix(String originalPrefix, String targetPrefix) {
    checkState(
        hasPackagePrefix(originalPrefix),
        "Expected %s to have a package prefix of (%s) before stripping.",
        this,
        originalPrefix);
    checkPackagePrefixFormat(targetPrefix);
    return ClassName.create(targetPrefix + binaryName().substring(originalPrefix.length()));
  }

  @Override
  public ClassName acceptTypeMapper(TypeMapper typeMapper) {
    return typeMapper.map(this);
  }
}

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

  private static final String TYPE_ADAPTER_PACKAGE_ROOT = "desugar/runtime/typeadapter/";

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

  private static final ImmutableBiMap<String, String> DELIVERY_TYPE_MAPPINGS =
      ImmutableBiMap.<String, String>builder()
          .put("java/", "j$/")
          .put("javadesugar/", "jd$/")
          .build();

  public static final TypeMapper IN_PROCESS_LABEL_STRIPPER =
      new TypeMapper(ClassName::verbatimName);

  public static final TypeMapper DELIVERY_TYPE_MAPPER =
      new TypeMapper(ClassName::verbatimToDelivery);

  public static final TypeMapper VERBATIM_TYPE_MAPPER =
      new TypeMapper(ClassName::deliveryToVerbatim);

  /**
   * The textual binary name used to index the class name, as defined at,
   * https://docs.oracle.com/javase/specs/jvms/se11/html/jvms-4.html#jvms-4.2.1
   */
  public abstract String binaryName();

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

  private ClassName stripInProcessLabel() {
    return stripPackagePrefix(IN_PROCESS_LABEL);
  }

  private ClassName stripInProcessLabelIfAny() {
    return hasInProcessLabel() ? stripPackagePrefix(IN_PROCESS_LABEL) : this;
  }

  /** Strips out in-process labels if any. */
  public final ClassName verbatimName() {
    return stripInProcessLabelIfAny().deliveryToVerbatim();
  }

  public final ClassName typeAdapterOwner() {
    return verbatimName().withSimpleNameSuffix("Adapter").prependPrefix(TYPE_ADAPTER_PACKAGE_ROOT);
  }

  public final ClassName typeConverterOwner() {
    return verbatimName()
        .withSimpleNameSuffix("Converter")
        .prependPrefix(TYPE_ADAPTER_PACKAGE_ROOT);
  }

  public final ClassName verbatimToDelivery() {
    return DELIVERY_TYPE_MAPPINGS.keySet().stream()
        .filter(this::hasPackagePrefix)
        .map(prefix -> replacePackagePrefix(prefix, DELIVERY_TYPE_MAPPINGS.get(prefix)))
        .findAny()
        .orElse(this);
  }

  public final ClassName deliveryToVerbatim() {
    ImmutableBiMap<String, String> verbatimTypeMappings = DELIVERY_TYPE_MAPPINGS.inverse();
    return verbatimTypeMappings.keySet().stream()
        .filter(this::hasPackagePrefix)
        .map(prefix -> replacePackagePrefix(prefix, verbatimTypeMappings.get(prefix)))
        .findAny()
        .orElse(this);
  }

  public final ClassName prependPrefix(String prefix) {
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

  public final ClassName stripPackagePrefix(String prefix) {
    return replacePackagePrefix(/* originalPrefix= */ prefix, /* targetPrefix= */ "");
  }

  public final ClassName replacePackagePrefix(String originalPrefix, String targetPrefix) {
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

  public boolean isInProcessCoreType() {
    return hasInProcessLabel() && stripInProcessLabel().isVerbatimCoreType();
  }

  public boolean isVerbatimCoreType() {
    return hasAnyPackagePrefix(DELIVERY_TYPE_MAPPINGS.keySet());
  }

  public boolean isDeliveryCoreType() {
    return hasAnyPackagePrefix(DELIVERY_TYPE_MAPPINGS.values());
  }
}

/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package com.google.devtools.build.android.desugar.corelibadapter;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.android.desugar.langmodel.ClassName.TYPE_ADAPTER_PACKAGE_ROOT;
import static com.google.devtools.build.android.desugar.langmodel.ClassName.TYPE_CONVERTER_SUFFIX;
import static org.objectweb.asm.ClassReader.SKIP_CODE;
import static org.objectweb.asm.ClassReader.SKIP_DEBUG;
import static org.objectweb.asm.ClassReader.SKIP_FRAMES;
import static org.objectweb.asm.Opcodes.ACC_PROTECTED;
import static org.objectweb.asm.Opcodes.ACC_PUBLIC;

import com.google.auto.value.AutoValue;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.android.desugar.io.JarItem;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.FieldKey;
import com.google.devtools.build.android.desugar.langmodel.MethodDeclInfo;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import java.io.IOError;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.FieldVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/**
 * Checks a) the Java platform types with supported desugar-shadowed/mirrored type converters under
 * {@link com.google.devtools.build.android.desugar.typeadapter} subpackages, against b) the
 * <i>desugar-shadowable Java platform types</i> referenced by any Android platform method headers,
 * and reports an missing error if there is any Java Platform type absent in a) but present in b).
 *
 * <p>A type is a <i>desugar-shadowable Java platform type</i> if and only if the type tests {@code
 * true} for {@link ClassName#isDesugarShadowedType()}.
 */
@RunWith(JUnit4.class)
public class ShadowedPlatformTypeConverterCoverageTest {

  private static final Splitter SPACE_SPLITTER = Splitter.on(" ").trimResults();
  private static final ImmutableList<String> PLATFORM_JAR_PATHS =
      ImmutableList.copyOf(SPACE_SPLITTER.splitToList(System.getProperty("platform_jars")));
  private static final String TYPE_CONVERTER_JAR_PATH = System.getProperty("type_converter_jar");

  private ImmutableMultimap<ClassName, MethodHeaderTypeTrackingLabel>
      shadowedTypesOnPlatformMethodHeaders;
  private ImmutableMultimap<ClassName, FieldTypeTrackingLabel> shadowedTypesOnPlatformFields;

  private final ImmutableSet<ClassName> shadowedTypesWithTypeConverterSupport =
      getShadowedTypesWithTypeConverterSupport();

  @Before
  public void setUp() throws Exception {
    ImmutableMultimap.Builder<ClassName, MethodHeaderTypeTrackingLabel> shadowedMethods =
        ImmutableMultimap.builder();
    ImmutableMultimap.Builder<ClassName, FieldTypeTrackingLabel> shadowedFields =
        ImmutableMultimap.builder();
    loadShadowedTypesOnPlatformClassMembers(shadowedMethods, shadowedFields);
    shadowedTypesOnPlatformMethodHeaders = shadowedMethods.build();
    shadowedTypesOnPlatformFields = shadowedFields.build();
  }

  @Test
  public void checkTypeConverterSupport_allAndroidPlatformMethodHeadersCovered() {
    ImmutableSet<ClassName> shadowedTypesOnPlatform = shadowedTypesOnPlatformMethodHeaders.keySet();
    Set<ClassName> shadowedTypesMissingTypeConverter =
        Sets.difference(shadowedTypesOnPlatform, shadowedTypesWithTypeConverterSupport);

    assertWithMessage(
            String.format(
                "Desugar-shadowable platform types missing a type converter: \n%s\n",
                shadowedTypesMissingTypeConverter.stream()
                    .flatMap(type -> shadowedTypesOnPlatformMethodHeaders.get(type).stream())
                    .collect(toImmutableList())))
        .that(shadowedTypesMissingTypeConverter)
        .isEmpty();
  }

  @Test
  public void checkTypeConverterSupport_allAndroidPlatformFieldsCovered() {
    ImmutableSet<ClassName> shadowedTypesOnPlatform = shadowedTypesOnPlatformFields.keySet();
    Set<ClassName> shadowedTypesMissingTypeConverter =
        Sets.difference(shadowedTypesOnPlatform, shadowedTypesWithTypeConverterSupport);

    assertWithMessage(
            String.format(
                "Desugar-shadowable platform types missing a type converter: \n%s\n",
                shadowedTypesMissingTypeConverter.stream()
                    .flatMap(type -> shadowedTypesOnPlatformFields.get(type).stream())
                    .collect(toImmutableList())))
        .that(shadowedTypesMissingTypeConverter)
        .isEmpty();
  }

  private static void loadShadowedTypesOnPlatformClassMembers(
      ImmutableMultimap.Builder<ClassName, MethodHeaderTypeTrackingLabel>
          shadowedMethodTypesBuilder,
      ImmutableMultimap.Builder<ClassName, FieldTypeTrackingLabel> shadowedFieldTypesBuilder) {
    PLATFORM_JAR_PATHS.stream()
        .flatMap(jarTextPath -> JarItem.jarItemStream(Paths.get(jarTextPath)))
        .filter(
            jarItem ->
                jarItem.jarEntryName().endsWith(".class")
                    && !jarItem.jarEntryName().startsWith("META-INF/"))
        .forEach(
            jarItem -> {
              try (InputStream inputStream = jarItem.getInputStream()) {
                ClassReader cr = new ClassReader(inputStream);
                ClassVisitor cv =
                    new ClassMemberHeaderClassVisitor(
                        shadowedMethodTypesBuilder, shadowedFieldTypesBuilder, jarItem.jarPath());
                cr.accept(cv, SKIP_CODE | SKIP_DEBUG | SKIP_FRAMES);
              } catch (IOException e) {
                throw new IOError(e);
              }
            });
  }

  private static ImmutableSet<ClassName> getShadowedTypesWithTypeConverterSupport() {
    String typeConverterClassFileSuffix = TYPE_CONVERTER_SUFFIX + ".class";
    int typeConverterClassFileSuffixLength = typeConverterClassFileSuffix.length();
    return JarItem.jarItemStream(Paths.get(TYPE_CONVERTER_JAR_PATH))
        .map(JarItem::jarEntryName)
        .filter(
            jarEntryName ->
                jarEntryName.startsWith(TYPE_ADAPTER_PACKAGE_ROOT)
                    && jarEntryName.endsWith(typeConverterClassFileSuffix))
        .map(
            jarEntryName ->
                ClassName.create(
                    jarEntryName.substring(
                        TYPE_ADAPTER_PACKAGE_ROOT.length(),
                        jarEntryName.length() - typeConverterClassFileSuffixLength)))
        .collect(toImmutableSet());
  }

  private static class ClassMemberHeaderClassVisitor extends ClassVisitor {

    private final ImmutableMultimap.Builder<ClassName, MethodHeaderTypeTrackingLabel>
        shadowedMethodTypes;
    private final ImmutableMultimap.Builder<ClassName, FieldTypeTrackingLabel> shadowedFieldTypes;
    private final Path containgJar;

    private ClassName className;
    private int classAccess;

    ClassMemberHeaderClassVisitor(
        ImmutableMultimap.Builder<ClassName, MethodHeaderTypeTrackingLabel> shadowedMethodTypes,
        ImmutableMultimap.Builder<ClassName, FieldTypeTrackingLabel> shadowedFieldTypes,
        Path containingJar) {
      super(Opcodes.ASM7);
      this.shadowedMethodTypes = shadowedMethodTypes;
      this.shadowedFieldTypes = shadowedFieldTypes;
      this.containgJar = containingJar;
    }

    @Override
    public void visit(
        int version,
        int access,
        String name,
        String signature,
        String superName,
        String[] interfaces) {
      super.visit(version, classAccess, name, signature, superName, interfaces);
      className = ClassName.create(name);
      classAccess = access;
    }

    @Override
    public FieldVisitor visitField(
        int access, String name, String descriptor, String signature, Object value) {
      FieldKey fieldKey = FieldKey.create(className, name, descriptor);
      ClassName fieldType = fieldKey.getFieldTypeName();
      if (className.isAndroidDomainType()
          && ((access & ACC_PUBLIC) != 0 || (access & ACC_PROTECTED) != 0)
          && fieldType.isDesugarShadowedType()) {
        FieldTypeTrackingLabel trackingLabel =
            FieldTypeTrackingLabel.builder()
                .setField(fieldKey)
                .setShadowedType(fieldType)
                .setJarPath(containgJar)
                .build();
        shadowedFieldTypes.put(trackingLabel.shadowedType(), trackingLabel);
      }
      return super.visitField(access, name, descriptor, signature, value);
    }

    @Override
    public MethodVisitor visitMethod(
        int access, String name, String descriptor, String signature, String[] exceptions) {
      if (className.isAndroidDomainType()) {
        MethodDeclInfo methodDeclInfo =
            MethodDeclInfo.create(
                MethodKey.create(className, name, descriptor),
                classAccess,
                access,
                signature,
                exceptions);
        if (methodDeclInfo.isPublicAccess() || methodDeclInfo.isProtectedAccess()) {
          ClassName returnType = methodDeclInfo.returnTypeName();
          if (returnType.isDesugarShadowedType()) {
            MethodHeaderTypeTrackingLabel trackingLabel =
                MethodHeaderTypeTrackingLabel.builder()
                    .setShadowedType(returnType)
                    .setMethod(methodDeclInfo)
                    .setAtReturnType(true)
                    .setParameterTypePosition(-1)
                    .setExceptionTypePosition(-1)
                    .setJarPath(containgJar)
                    .build();
            shadowedMethodTypes.put(trackingLabel.shadowedType(), trackingLabel);
          }

          ImmutableList<ClassName> argumentTypes = methodDeclInfo.argumentTypeNames();
          for (int i = 0; i < argumentTypes.size(); i++) {
            ClassName parameterType = argumentTypes.get(i);
            if (parameterType.isDesugarShadowedType()) {
              MethodHeaderTypeTrackingLabel trackingLabel =
                  MethodHeaderTypeTrackingLabel.builder()
                      .setShadowedType(parameterType)
                      .setMethod(methodDeclInfo)
                      .setAtReturnType(false)
                      .setParameterTypePosition(i)
                      .setExceptionTypePosition(-1)
                      .setJarPath(containgJar)
                      .build();
              shadowedMethodTypes.put(trackingLabel.shadowedType(), trackingLabel);
            }
          }

          ImmutableList<ClassName> exceptionTypes = methodDeclInfo.argumentTypeNames();
          for (int i = 0; i < exceptionTypes.size(); i++) {
            ClassName exceptionType = exceptionTypes.get(i);
            if (exceptionType.isDesugarShadowedType()) {
              MethodHeaderTypeTrackingLabel trackingLabel =
                  MethodHeaderTypeTrackingLabel.builder()
                      .setShadowedType(exceptionType)
                      .setMethod(methodDeclInfo)
                      .setAtReturnType(false)
                      .setParameterTypePosition(-1)
                      .setExceptionTypePosition(i)
                      .setJarPath(containgJar)
                      .build();
              shadowedMethodTypes.put(trackingLabel.shadowedType(), trackingLabel);
            }
          }
        }
      }
      return super.visitMethod(access, name, descriptor, signature, exceptions);
    }
  }

  /** Tracks the origin of a field type. */
  @AutoValue
  abstract static class FieldTypeTrackingLabel {
    abstract ClassName shadowedType();

    abstract FieldKey field();

    abstract Path jarPath();

    static Builder builder() {
      return new AutoValue_ShadowedPlatformTypeConverterCoverageTest_FieldTypeTrackingLabel
          .Builder();
    }

    @AutoValue.Builder
    abstract static class Builder {

      abstract Builder setShadowedType(ClassName value);

      abstract Builder setField(FieldKey value);

      abstract Builder setJarPath(Path value);

      abstract FieldTypeTrackingLabel build();
    }
  }

  /** Tracks the origin of a method header type, including parameter, return and exception types. */
  @AutoValue
  abstract static class MethodHeaderTypeTrackingLabel {
    abstract ClassName shadowedType();

    abstract MethodDeclInfo method();

    abstract boolean atReturnType();

    abstract int parameterTypePosition();

    abstract int exceptionTypePosition();

    abstract Path jarPath();

    static Builder builder() {
      return new AutoValue_ShadowedPlatformTypeConverterCoverageTest_MethodHeaderTypeTrackingLabel
          .Builder();
    }

    @AutoValue.Builder
    public abstract static class Builder {

      abstract Builder setShadowedType(ClassName value);

      abstract Builder setMethod(MethodDeclInfo value);

      abstract Builder setAtReturnType(boolean value);

      abstract Builder setParameterTypePosition(int value);

      abstract Builder setExceptionTypePosition(int value);

      abstract Builder setJarPath(Path value);

      abstract MethodHeaderTypeTrackingLabel build();
    }
  }
}

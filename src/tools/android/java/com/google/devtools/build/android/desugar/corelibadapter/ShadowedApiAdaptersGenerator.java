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

package com.google.devtools.build.android.desugar.corelibadapter;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static org.objectweb.asm.Opcodes.ACC_ABSTRACT;
import static org.objectweb.asm.Opcodes.ACC_PUBLIC;
import static org.objectweb.asm.Opcodes.ACC_STATIC;
import static org.objectweb.asm.Opcodes.ACC_SYNTHETIC;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.android.desugar.io.FileContentProvider;
import com.google.devtools.build.android.desugar.io.MapBasedClassFileProvider;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.MethodDeclInfo;
import com.google.devtools.build.android.desugar.langmodel.MethodInvocationSite;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import java.util.stream.Stream;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;

/**
 * Generates type adapter classes with methods that bridge the interactions between
 * desugared-mirrored types and their desugar-shadowed built-in types. Together with the necessary
 * type converter classes, the class delivers the generated adapter classes to runtime library.
 */
public final class ShadowedApiAdaptersGenerator {

  private static final int TYPE_ADAPTER_CLASS_ACCESS = ACC_PUBLIC | ACC_ABSTRACT | ACC_SYNTHETIC;
  private static final int TYPE_CONVERSION_METHOD_ACCESS = ACC_PUBLIC | ACC_STATIC | ACC_SYNTHETIC;

  /** A pre-collected record that tracks the adapter method requests from invocation sites. */
  private final InvocationSiteTransformationRecord invocationAdapterSites;

  /** A record with evolving map values that track adapter classes to be generated. */
  private final ImmutableMap<ClassName, ClassWriter> typeAdapters;

  private ShadowedApiAdaptersGenerator(
      InvocationSiteTransformationRecord invocationAdapterSites,
      ImmutableMap<ClassName, ClassWriter> typeAdapters) {
    this.invocationAdapterSites = invocationAdapterSites;
    this.typeAdapters = typeAdapters;
  }

  /** The public factory method to construct {@link ShadowedApiAdaptersGenerator}. */
  public static ShadowedApiAdaptersGenerator create(
      InvocationSiteTransformationRecord callSiteTransformations) {
    return emitClassWriters(callSiteTransformations).emitAdapterMethods().closeClassWriters();
  }

  private static ShadowedApiAdaptersGenerator emitClassWriters(
      InvocationSiteTransformationRecord callSiteTransformations) {
    return new ShadowedApiAdaptersGenerator(
        callSiteTransformations,
        callSiteTransformations.adapterReplacements().stream()
            .map(ShadowedApiAdapterHelper::getAdapterInvocationSite)
            .map(MethodInvocationSite::owner)
            .distinct()
            .collect(
                toImmutableMap(
                    className -> className,
                    ShadowedApiAdaptersGenerator::createAdapterClassWriter)));
  }

  private static ClassWriter createAdapterClassWriter(ClassName className) {
    ClassWriter cw = new ClassWriter(ClassWriter.COMPUTE_MAXS);
    cw.visit(
        Opcodes.V1_7,
        TYPE_ADAPTER_CLASS_ACCESS,
        className.binaryName(),
        /* signature= */ null,
        /* superName= */ "java/lang/Object",
        /* interfaces= */ new String[0]);
    return cw;
  }

  /** Returns desugar-shadowed API adapters with desugar-mirrored types. */
  public MapBasedClassFileProvider getApiAdapters() {
    return MapBasedClassFileProvider.builder()
        .setTag("ShadowedApiAdapters")
        .setFileContents(
            Maps.transformEntries(
                typeAdapters,
                (className, cw) ->
                    FileContentProvider.fromBytes(className.classFilePathName(), cw.toByteArray())))
        .build();
  }

  /**
   * Returns type conversion classes that converts between a desugar-shadowed type and its
   * deusgar-mirrored counterpart.
   */
  public ImmutableList<ClassName> getTypeConverters() {
    return Stream.concat(
            invocationAdapterSites.inlineConversions().stream(),
            invocationAdapterSites.adapterReplacements().stream())
        .flatMap(
            site ->
                Stream.concat(Stream.of(site.returnTypeName()), site.argumentTypeNames().stream()))
        .filter(ClassName::isDesugarShadowedType)
        .distinct()
        .map(ClassName::typeConverterOwner)
        .collect(toImmutableList());
  }

  private ShadowedApiAdaptersGenerator emitAdapterMethods() {
    for (MethodInvocationSite invocationSite : invocationAdapterSites.adapterReplacements()) {
      MethodInvocationSite adapterSite =
          ShadowedApiAdapterHelper.getAdapterInvocationSite(invocationSite);
      ClassName adapterOwner = adapterSite.owner();
      ClassWriter cv =
          checkNotNull(
              typeAdapters.get(adapterOwner),
              "Expected a class writer present before writing its methods. Requested adapter"
                  + " owner: (%s). Available adapter owners: (%s).",
              adapterOwner,
              typeAdapters);
      MethodKey adapterMethodKey = adapterSite.method();
      MethodDeclInfo adapterMethodDecl =
          MethodDeclInfo.create(
              adapterMethodKey,
              TYPE_ADAPTER_CLASS_ACCESS,
              TYPE_CONVERSION_METHOD_ACCESS,
              /* signature= */ null,
              /* exceptions= */ new String[] {});
      MethodVisitor mv = adapterMethodDecl.accept(cv);
      mv.visitCode();

      int slotOffset = 0;
      for (Type argType : adapterMethodDecl.argumentTypes()) {
        ClassName argTypeName = ClassName.create(argType);
        mv.visitVarInsn(argType.getOpcode(Opcodes.ILOAD), slotOffset);
        if (argTypeName.isDesugarMirroredType()) {
          MethodInvocationSite conversion =
              ShadowedApiAdapterHelper.mirroredToShadowedTypeConversionSite(argTypeName);
          conversion.accept(mv);
        }
        slotOffset += argType.getSize();
      }

      invocationSite.accept(mv);

      ClassName adapterReturnTypeName = adapterMethodDecl.returnTypeName();
      if (adapterReturnTypeName.isDesugarMirroredType()) {
        MethodInvocationSite conversion =
            ShadowedApiAdapterHelper.shadowedToMirroredTypeConversionSite(
                adapterReturnTypeName.mirroredToShadowed());
        conversion.accept(mv);
      }

      mv.visitInsn(adapterMethodDecl.returnType().getOpcode(Opcodes.IRETURN));

      mv.visitMaxs(slotOffset, slotOffset);
      mv.visitEnd();
    }
    return this;
  }

  private ShadowedApiAdaptersGenerator closeClassWriters() {
    typeAdapters.values().forEach(ClassVisitor::visitEnd);
    return this;
  }
}

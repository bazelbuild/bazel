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

package com.google.devtools.build.android.desugar.typehierarchy;

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.MethodDeclInfo;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import com.google.devtools.build.android.desugar.typehierarchy.TypeHierarchy.TypeHierarchyBuilder;
import java.util.Arrays;
import javax.annotation.Nullable;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;

/** Collects the type hierarchy information with ASM framework's class visitor. */
class TypeHierarchyClassVisitor extends ClassVisitor {

  private final TypeHierarchyBuilder typeHierarchyBuilder;
  private final String resourcePath;

  private HierarchicalTypeKey hierarchicalType;
  private int classAccess;

  TypeHierarchyClassVisitor(
      int api,
      String resourcePath,
      TypeHierarchyBuilder typeHierarchyBuilder,
      ClassVisitor classVisitor) {
    super(api, classVisitor);
    this.resourcePath = resourcePath;
    this.typeHierarchyBuilder = typeHierarchyBuilder;
  }

  @Override
  public void visit(
      int version,
      int access,
      String name,
      @Nullable String signature,
      @Nullable String superName,
      @Nullable String[] interfaces) {
    hierarchicalType = HierarchicalTypeKey.create(ClassName.create(name));
    checkState(
        resourcePath.equals(hierarchicalType.type().classFilePathName()),
        "Expected resource path %s to match class name %s.",
        resourcePath,
        hierarchicalType);
    classAccess = access;
    HierarchicalTypeKey superClassType =
        superName == null
            ? HierarchicalTypeKey.SENTINEL
            : HierarchicalTypeKey.create(ClassName.create(superName));
    typeHierarchyBuilder.putDirectSuperClass(hierarchicalType, superClassType);
    ImmutableSet<HierarchicalTypeKey> superInterfaceTypes =
        interfaces == null
            ? ImmutableSet.of()
            : Arrays.stream(interfaces)
                .map(ClassName::create)
                .map(HierarchicalTypeKey::create)
                .collect(toImmutableSet());
    typeHierarchyBuilder.putDirectInterfaces(hierarchicalType, superInterfaceTypes);
    super.visit(version, access, name, signature, superName, interfaces);
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String descriptor, String signature, String[] exceptions) {
    MethodDeclInfo methodDeclInfo =
        MethodDeclInfo.create(
            MethodKey.create(hierarchicalType.type(), name, descriptor),
            classAccess,
            access,
            signature,
            exceptions);
    if (!methodDeclInfo.isPrivateAccess() && !methodDeclInfo.isStaticMethod()) {
      typeHierarchyBuilder.putMethod(methodDeclInfo);
    }
    return super.visitMethod(access, name, descriptor, signature, exceptions);
  }
}

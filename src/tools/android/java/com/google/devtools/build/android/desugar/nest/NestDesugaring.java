// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.desugar.nest;

import static com.google.devtools.build.android.desugar.langmodel.LangModelConstants.NEST_COMPANION_CLASS_SIMPLE_NAME;
import static org.objectweb.asm.Opcodes.ACC_ABSTRACT;
import static org.objectweb.asm.Opcodes.ACC_INTERFACE;
import static org.objectweb.asm.Opcodes.ACC_SYNTHETIC;

import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.FieldKey;
import com.google.devtools.build.android.desugar.langmodel.MethodDeclInfo;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import javax.annotation.Nullable;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.FieldVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/**
 * A class visitor that rewrites visited classes and interfaces with nested-based access [JEP 181].
 * For any private class member with access from another class or interface within the same nest,
 * The visitor generates visibility-broadened bridge methods and update all call sites. For any
 * private interface member within access from another class or interface within the same nest, the
 * visitor broadens the visibility of private interface methods, renames these interface members and
 * update their call sites.
 */
public final class NestDesugaring extends ClassVisitor {

  private static final int NEST_COMPANION_CLASS_ACCESS_CODE =
      Opcodes.ACC_SYNTHETIC | Opcodes.ACC_ABSTRACT | Opcodes.ACC_STATIC;

  private final NestDigest nestDigest;

  private FieldAccessBridgeEmitter fieldAccessBridgeEmitter;
  private MethodAccessorEmitter methodAccessorEmitter;

  private ClassName className;
  private int classAccess;
  @Nullable private ClassVisitor nestCompanionVisitor;

  /** Whether the class being visited is a nest host with with a nest companion class. */
  private boolean isNestHostWithNestCompanion;

  public NestDesugaring(ClassVisitor classVisitor, NestDigest nestDigest) {
    super(Opcodes.ASM7, classVisitor);
    this.nestDigest = nestDigest;
  }

  @Override
  public void visit(
      int version,
      int access,
      String name,
      String signature,
      String superName,
      String[] interfaces) {
    className = ClassName.create(name);
    classAccess = access;
    nestCompanionVisitor = nestDigest.getCompanionClassWriter(className);
    isNestHostWithNestCompanion =
        (nestCompanionVisitor != null) && className.equals(nestDigest.nestHost(className).get());
    fieldAccessBridgeEmitter = new FieldAccessBridgeEmitter();
    methodAccessorEmitter = new MethodAccessorEmitter(nestDigest);
    super.visit(
        Math.min(version, NestDesugarConstants.MIN_VERSION),
        access,
        name,
        signature,
        superName,
        interfaces);
    if (isNestHostWithNestCompanion) {
      nestCompanionVisitor.visit(
          Math.min(version, NestDesugarConstants.MIN_VERSION),
          ACC_SYNTHETIC | ACC_ABSTRACT,
          nestDigest.nestCompanion(className).binaryName(),
          /* signature= */ null,
          /* superName= */ "java/lang/Object",
          /* interfaces= */ new String[0]);
    }
  }

  @Override
  public FieldVisitor visitField(
      int access, String name, String descriptor, String signature, Object value) {
    FieldKey fieldKey = FieldKey.create(className, name, descriptor);
    // Generates necessary bridge methods for this field, including field getter methods and field
    // setter methods. See FieldAccessBridgeEmitter.
    nestDigest
        .findAllMemberUseKinds(fieldKey)
        .forEach(useKind -> fieldKey.accept(useKind, fieldAccessBridgeEmitter, cv));
    return super.visitField(access, name, descriptor, signature, value);
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String descriptor, String signature, String[] exceptions) {
    MethodKey methodKey = MethodKey.create(className, name, descriptor);
    if (nestDigest.hasAnyTrackingReason(methodKey)) {
      MethodDeclInfo methodDeclInfo =
          MethodDeclInfo.create(methodKey, classAccess, access, signature, exceptions);
      // For interfaces, the following .accept converts the original method into a
      // visibility-broadened renamed method in-place while preserving the method body
      // implementation; for classes, the following .accept generates synthetic bridge methods and
      // preserve the original methods. The returned method visitor is subject to further desugar
      // operations for interfaces. See {@link MethodAccessorEmitter}
      MethodVisitor targetMethodVisitor = methodDeclInfo.accept(methodAccessorEmitter, cv);

      // For classes, we continue to desugar the code of the bridge's originating method. For
      // interfaces, we continue to desugar the header-modified methods. The code of the bridge
      // method is one simple delegation call to its originating method without any code subject to
      // desugar.
      MethodVisitor primaryMethodVisitor =
          (classAccess & ACC_INTERFACE) != 0
              ? targetMethodVisitor
              : super.visitMethod(access, name, descriptor, signature, exceptions);
      return new NestBridgeRefConverter(primaryMethodVisitor, methodKey, nestDigest);
    }
    // In absence of method invocation record, fallback to the delegate method visitor.
    MethodVisitor mv = super.visitMethod(access, name, descriptor, signature, exceptions);
    return mv == null ? null : new NestBridgeRefConverter(mv, methodKey, nestDigest);
  }

  @Override
  public void visitNestHost(String nestHost) {
    // Intentionally omit super call to avoid writing NestHost attribute to the class file.
  }

  @Override
  public void visitNestMember(String nestMember) {
    // Intentionally omit super call to avoid writing the NestMembers attribute to the class file.
  }

  @Override
  public void visitSource(String source, String debug) {
    super.visitSource(source, debug);
    if (isNestHostWithNestCompanion) {
      nestCompanionVisitor.visitSource(source, debug);
    }
  }

  @Override
  public void visitEnd() {
    if (isNestHostWithNestCompanion) {
      ClassName nestCompanion = nestDigest.nestCompanion(className);
      // In the nest companion class, marks its outer class as the nest host.
      nestCompanionVisitor.visitInnerClass(
          nestCompanion.binaryName(),
          className.binaryName(),
          NEST_COMPANION_CLASS_SIMPLE_NAME,
          NEST_COMPANION_CLASS_ACCESS_CODE);
      // In the nest host class, marks the nest companion as one of its inner classes.
      cv.visitInnerClass(
          nestCompanion.binaryName(),
          className.binaryName(),
          NEST_COMPANION_CLASS_SIMPLE_NAME,
          NEST_COMPANION_CLASS_ACCESS_CODE);
    }
    super.visitEnd();
  }
}

// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.r8.desugar;

import static com.google.common.truth.Truth.assertThat;
import static org.objectweb.asm.Opcodes.ACC_ABSTRACT;
import static org.objectweb.asm.Opcodes.ACC_INTERFACE;
import static org.objectweb.asm.Opcodes.ASM7;

import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.Handle;
import org.objectweb.asm.MethodVisitor;

/**
 * ASM visitor to collect summary information from class files for checking that desugaring has been
 * applied
 */
public class DesugarInfoCollector extends ClassVisitor {
  private int largestClassFileVersion;
  private int numberOfInvokeDynamic;
  private int numberOfDefaultMethods;
  private int numberOfDesugaredLambdas;
  private int numberOfCompanionClasses;

  private int currentClassAccess;

  public DesugarInfoCollector() {
    this(null);
  }

  public DesugarInfoCollector(ClassVisitor classVisitor) {
    super(ASM7, classVisitor);
  }

  public int getNumberOfInvokeDynamic() {
    return numberOfInvokeDynamic;
  }

  public int getNumberOfDefaultMethods() {
    return numberOfDefaultMethods;
  }

  public int getLargestMajorClassFileVersion() {
    return computeMajorClassFileVersion(largestClassFileVersion);
  }

  public int getNumberOfDesugaredLambdas() {
    return numberOfDesugaredLambdas;
  }

  public int getNumberOfCompanionClasses() {
    return numberOfCompanionClasses;
  }

  @Override
  public void visit(
      int version,
      int access,
      java.lang.String name,
      java.lang.String signature,
      java.lang.String superName,
      java.lang.String[] interfaces) {
    super.visit(version, access, name, signature, superName, interfaces);
    assertThat(computeMinorClassFileVersion(version)).isEqualTo(0);
    largestClassFileVersion = Math.max(version, largestClassFileVersion);
    if (classNameFromBinaryName(name).contains("$$ExternalSyntheticLambda")) {
      numberOfDesugaredLambdas++;
    }
    if (name.endsWith("$-CC")) {
      numberOfCompanionClasses++;
    }
    currentClassAccess = access;
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String descriptor, String signature, String[] exceptions) {
    MethodVisitor mv = super.visitMethod(access, name, descriptor, signature, exceptions);
    if (isInterface(currentClassAccess) && !isAbstract(access)) {
      numberOfDefaultMethods++;
    }
    return new LambdaUseMethodVisitor(api, mv);
  }

  private class LambdaUseMethodVisitor extends MethodVisitor {

    LambdaUseMethodVisitor(int api, MethodVisitor methodVisitor) {
      super(api, methodVisitor);
    }

    @Override
    public void visitInvokeDynamicInsn(String name, String desc, Handle bsm, Object... bsmArgs) {
      super.visitInvokeDynamicInsn(name, desc, bsm, bsmArgs);
      numberOfInvokeDynamic++;
    }
  }

  private static int computeMinorClassFileVersion(int version) {
    return (version >> 16) & 0xffff;
  }

  private static int computeMajorClassFileVersion(int version) {
    return version & 0xffff;
  }

  private static boolean isAbstract(int access) {
    return (access & ACC_ABSTRACT) != 0;
  }

  private static boolean isInterface(int access) {
    return (access & ACC_INTERFACE) != 0;
  }

  private static String classNameFromBinaryName(String name) {
    int index = name.lastIndexOf('/');
    if (index == -1) {
      return name;
    }
    return name.substring(index + 1);
  }
}

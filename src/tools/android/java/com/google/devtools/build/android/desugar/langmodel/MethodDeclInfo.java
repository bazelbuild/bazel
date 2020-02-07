/*
 * Copyright 2019 The Bazel Authors. All rights reserved.
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

import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/** A unit data object represents a class or interface declaration. */
// TODO(deltazulu): Consider @AutoValue-ize this class. (String[] as attribute is not supported).
public final class MethodDeclInfo implements TypeMappable<MethodDeclInfo> {
  public final MethodKey methodKey;
  public final int ownerAccess;
  public final int memberAccess;
  public final String signature;
  public final String[] exceptions;

  public MethodDeclInfo(
      MethodKey methodKey,
      int ownerAccess,
      int memberAccess,
      String signature,
      String[] exceptions) {
    this.methodKey = methodKey;
    this.ownerAccess = ownerAccess;
    this.memberAccess = memberAccess;
    this.signature = signature;
    this.exceptions = exceptions;
  }

  public final MethodVisitor accept(ClassVisitor cv) {
    return cv.visitMethod(
        memberAccess, methodKey.descriptor(), methodKey.descriptor(), signature, exceptions);
  }

  public final <R, P> R accept(MethodDeclVisitor<R, ? super MethodDeclInfo, P> visitor, P param) {
    if (methodKey.isConstructor()) {
      return visitor.visitClassConstructor(this, param);
    }
    boolean isInterface = (ownerAccess & Opcodes.ACC_INTERFACE) != 0;
    boolean isStatic = (memberAccess & Opcodes.ACC_STATIC) != 0;
    if (isInterface) {
      return isStatic
          ? visitor.visitInterfaceStaticMethod(this, param)
          : visitor.visitInterfaceInstanceMethod(this, param);
    } else {
      return isStatic
          ? visitor.visitClassStaticMethod(this, param)
          : visitor.visitClassInstanceMethod(this, param);
    }
  }

  @Override
  public MethodDeclInfo acceptTypeMapper(TypeMapper typeMapper) {
    return new MethodDeclInfo(
        methodKey.acceptTypeMapper(typeMapper), ownerAccess, memberAccess, signature, exceptions);
  }
}

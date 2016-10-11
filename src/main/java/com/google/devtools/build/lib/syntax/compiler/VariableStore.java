// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax.compiler;

import com.google.devtools.build.lib.syntax.compiler.Variable.InternalVariable;
import com.google.devtools.build.lib.syntax.compiler.Variable.SkylarkVariable;
import com.google.devtools.build.lib.util.Preconditions;
import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.description.type.TypeDescription;
import net.bytebuddy.implementation.Implementation.Context;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.StackManipulation.Size;
import net.bytebuddy.implementation.bytecode.StackSize;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

// TODO(klaasb) javadoc
enum VariableStore {
  INTEGER(Opcodes.ISTORE, StackSize.SINGLE),
  LONG(Opcodes.LSTORE, StackSize.DOUBLE),
  FLOAT(Opcodes.FSTORE, StackSize.SINGLE),
  DOUBLE(Opcodes.DSTORE, StackSize.DOUBLE),
  REFERENCE(Opcodes.ASTORE, StackSize.SINGLE);

  private final int opCode;
  private final Size size;

  private VariableStore(int opCode, StackSize size) {
    this.opCode = opCode;
    this.size = size.toIncreasingSize();
  }

  // TODO(klaasb) javadoc
  private VariableIndexStore into(int index) {
    return new VariableIndexStore(index);
  }

  // TODO(klaasb) javadoc
  public static VariableIndexStore into(SkylarkVariable variable) {
    return REFERENCE.into(variable.index);
  }

  // TODO(klaasb) javadoc
  public static VariableIndexStore into(InternalVariable variable) {
    return forType(variable.type).into(variable.index);
  }

  // TODO(klaasb) javadoc
  class VariableIndexStore implements ByteCodeAppender {

    private final int operandIndex;

    private VariableIndexStore(int operandIndex) {
      this.operandIndex = operandIndex;
    }

    @Override
    public ByteCodeAppender.Size apply(
        MethodVisitor methodVisitor,
        Context implementationContext,
        MethodDescription instrumentedMethod) {
      methodVisitor.visitVarInsn(opCode, operandIndex);
      return new ByteCodeAppender.Size(
          size.getMaximalSize(), Math.max(instrumentedMethod.getStackSize(), operandIndex + 1));
    }

    @Override
    public String toString() {
      return "VariableStore(" + opCode + ", " + operandIndex + ")";
    }
  }

  /**
   * Selects the correct VariableStore value for the given type
   */
  public static VariableStore forType(TypeDescription type) {
    if (type.isPrimitive()) {
      if (type.represents(long.class)) {
        return LONG;
      } else if (type.represents(double.class)) {
        return DOUBLE;
      } else if (type.represents(float.class)) {
        return FLOAT;
      } else {
        Preconditions.checkArgument(
            !type.represents(void.class), "Variables can't be of void type");
        return INTEGER;
      }
    } else {
      return REFERENCE;
    }
  }
}

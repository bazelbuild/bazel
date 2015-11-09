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

import com.google.devtools.build.lib.syntax.Operator;
import net.bytebuddy.implementation.Implementation.Context;
import net.bytebuddy.implementation.bytecode.StackManipulation;

import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/**
 * Implements byte code for gotos and conditional jumps.
 */
public class Jump implements StackManipulation {

  protected final int opCode;
  protected final Label target;
  protected final Size stackSizeChange;

  private Jump(int opCode, Label target, Size stackSizeChange) {
    this.opCode = opCode;
    this.target = target;
    this.stackSizeChange = stackSizeChange;
  }

  @Override
  public boolean isValid() {
    return true;
  }

  @Override
  public Size apply(MethodVisitor methodVisitor, Context implementationContext) {
    methodVisitor.visitJumpInsn(opCode, target);
    return stackSizeChange;
  }

  @Override
  public String toString() {
    return "Jump(" + opCode + ", " + target + ")";
  }

  /**
   * Builds a conditional jump for two int operands on the stack.
   */
  public static JumpWithoutTarget ifIntOperands(PrimitiveComparison comparison) {
    return new JumpWithoutTarget(Opcodes.IF_ICMPEQ + comparison.ordinal(), new Size(-2, 0));
  }

  /**
   * Builds a conditional jump for one int operand from the stack compared to zero.
   */
  public static JumpWithoutTarget ifIntOperandToZero(PrimitiveComparison comparison) {
    return new JumpWithoutTarget(Opcodes.IFEQ + comparison.ordinal(), new Size(-1, 0));
  }

  /**
   * Builds a conditional jump for two reference type operands from the stack.
   */
  public static JumpWithoutTarget ifReferenceOperands(ReferenceComparison comparison) {
    return new JumpWithoutTarget(Opcodes.IF_ACMPEQ + comparison.ordinal(), new Size(-2, 0));
  }

  /**
   * Builds a conditional jump for one reference type operand from the stack compared to null.
   */
  public static JumpWithoutTarget ifReferenceOperandToNull(ReferenceComparison comparison) {
    return new JumpWithoutTarget(Opcodes.IFNULL + comparison.ordinal(), new Size(-1, 0));
  }

  /**
   * Builds an unconditional jump to the target label.
   */
  public static Jump to(Label target) {
    return new Jump(Opcodes.GOTO, target, new Size(0, 0));
  }

  /**
   * Builds an unconditional jump to the label added by the given {@link LabelAdder}.
   */
  public static Jump to(LabelAdder target) {
    return to(target.getLabel());
  }

  /**
   * Builder helper class for partially built jumps from conditionals.
   *
   * <p>Allows adding a jump target label.
   */
  public static final class JumpWithoutTarget {

    protected final int opCode;
    protected final Size stackSizeChange;

    private JumpWithoutTarget(int opCode, Size stackSizeChange) {
      this.opCode = opCode;
      this.stackSizeChange = stackSizeChange;
    }

    /**
     * Builds a jump to the given target and the previously initialized conditional.
     */
    public Jump to(LabelAdder target) {
      return new Jump(opCode, target.getLabel(), stackSizeChange);
    }
  }

  /**
   * All primitive comparisons for which there are byte code equivalents.
   */
  public enum PrimitiveComparison {
    EQUAL,
    NOT_EQUAL,
    LESS,
    GREATER_EQUAL,
    GREATER,
    LESS_EQUAL;

    public static PrimitiveComparison forOperator(Operator operator) {
      switch (operator) {
        case LESS:
          return LESS;
        case LESS_EQUALS:
          return PrimitiveComparison.LESS_EQUAL;
        case GREATER:
          return GREATER;
        case GREATER_EQUALS:
          return GREATER_EQUAL;
        default:
          throw new Error("unreachable code");
      }
    }
  }

  /**
   * All reference comparisons for which there are byte code equivalents.
   */
  public enum ReferenceComparison {
    EQUAL,
    NOT_EQUAL;
  }
}

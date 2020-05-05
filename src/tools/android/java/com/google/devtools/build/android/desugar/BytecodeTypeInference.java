// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.android.desugar.io.BitFlags.isStatic;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.errorprone.annotations.CheckReturnValue;
import java.util.ArrayList;
import java.util.Optional;
import org.objectweb.asm.Handle;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;

/**
 * Perform type inference for byte code (local variables and operand stack) with the help of stack
 * map frames.
 *
 * <p>Note: This class only guarantees the correctness of reference types, but not the primitive
 * types, though they might be correct too.
 */
@CheckReturnValue // a good practice in general, and preparation for copying this into Truth
final class BytecodeTypeInference extends MethodVisitor {

  private boolean used = false;
  private final ArrayList<InferredType> localVariableSlots;
  private final ArrayList<InferredType> operandStack = new ArrayList<>();
  private FrameInfo previousFrame;
  /** For debugging purpose. */
  private final String methodSignature;

  BytecodeTypeInference(int access, String owner, String name, String methodDescriptor) {
    super(Opcodes.ASM8);
    localVariableSlots = createInitialLocalVariableTypes(access, owner, name, methodDescriptor);
    previousFrame = FrameInfo.create(ImmutableList.copyOf(localVariableSlots), ImmutableList.of());
    this.methodSignature = owner + "." + name + methodDescriptor;
  }

  void setDelegateMethodVisitor(MethodVisitor visitor) {
    mv = visitor;
  }

  @Override
  public void visitCode() {
    checkState(!used, "Cannot reuse this method visitor.");
    used = true;
    super.visitCode();
  }

  /** Returns the type of a value in the operand. 0 means the top of the stack. */
  InferredType getTypeOfOperandFromTop(int offsetFromTop) {
    int index = operandStack.size() - 1 - offsetFromTop;
    checkState(
        index >= 0,
        "Invalid offset %s in the list of size %s. The current method is %s",
        offsetFromTop,
        operandStack.size(),
        methodSignature);
    return operandStack.get(index);
  }

  String getOperandStackAsString() {
    return operandStack.toString();
  }

  String getLocalsAsString() {
    return localVariableSlots.toString();
  }

  @Override
  public void visitInsn(int opcode) {
    switch (opcode) {
      case Opcodes.NOP:
      case Opcodes.INEG:
      case Opcodes.LNEG:
      case Opcodes.FNEG:
      case Opcodes.DNEG:
      case Opcodes.I2B:
      case Opcodes.I2C:
      case Opcodes.I2S:
      case Opcodes.RETURN:
        break;
      case Opcodes.ACONST_NULL:
        push(InferredType.NULL);
        break;
      case Opcodes.ICONST_M1:
      case Opcodes.ICONST_0:
      case Opcodes.ICONST_1:
      case Opcodes.ICONST_2:
      case Opcodes.ICONST_3:
      case Opcodes.ICONST_4:
      case Opcodes.ICONST_5:
        push(InferredType.INT);
        break;
      case Opcodes.LCONST_0:
      case Opcodes.LCONST_1:
        push(InferredType.LONG);
        push(InferredType.TOP);
        break;
      case Opcodes.FCONST_0:
      case Opcodes.FCONST_1:
      case Opcodes.FCONST_2:
        push(InferredType.FLOAT);
        break;
      case Opcodes.DCONST_0:
      case Opcodes.DCONST_1:
        push(InferredType.DOUBLE);
        push(InferredType.TOP);
        break;
      case Opcodes.IALOAD:
      case Opcodes.BALOAD:
      case Opcodes.CALOAD:
      case Opcodes.SALOAD:
        pop(2);
        push(InferredType.INT);
        break;
      case Opcodes.LALOAD:
      case Opcodes.D2L:
        pop(2);
        push(InferredType.LONG);
        push(InferredType.TOP);
        break;
      case Opcodes.DALOAD:
      case Opcodes.L2D:
        pop(2);
        push(InferredType.DOUBLE);
        push(InferredType.TOP);
        break;
      case Opcodes.AALOAD:
        InferredType arrayType = pop(2);
        InferredType elementType = arrayType.getElementTypeIfArrayOrThrow();
        push(elementType);
        break;
      case Opcodes.IASTORE:
      case Opcodes.BASTORE:
      case Opcodes.CASTORE:
      case Opcodes.SASTORE:
      case Opcodes.FASTORE:
      case Opcodes.AASTORE:
        pop(3);
        break;
      case Opcodes.LASTORE:
      case Opcodes.DASTORE:
        pop(4);
        break;
      case Opcodes.POP:
      case Opcodes.IRETURN:
      case Opcodes.FRETURN:
      case Opcodes.ARETURN:
      case Opcodes.ATHROW:
      case Opcodes.MONITORENTER:
      case Opcodes.MONITOREXIT:
        pop();
        break;
      case Opcodes.POP2:
      case Opcodes.LRETURN:
      case Opcodes.DRETURN:
        pop(2);
        break;
      case Opcodes.DUP:
        push(top());
        break;
      case Opcodes.DUP_X1:
        {
          InferredType top = pop();
          InferredType next = pop();
          push(top);
          push(next);
          push(top);
          break;
        }
      case Opcodes.DUP_X2:
        {
          InferredType top = pop();
          InferredType next = pop();
          InferredType bottom = pop();
          push(top);
          push(bottom);
          push(next);
          push(top);
          break;
        }
      case Opcodes.DUP2:
        {
          InferredType top = pop();
          InferredType next = pop();
          push(next);
          push(top);
          push(next);
          push(top);
          break;
        }
      case Opcodes.DUP2_X1:
        {
          InferredType top = pop();
          InferredType next = pop();
          InferredType bottom = pop();
          push(next);
          push(top);
          push(bottom);
          push(next);
          push(top);
          break;
        }
      case Opcodes.DUP2_X2:
        {
          InferredType t1 = pop();
          InferredType t2 = pop();
          InferredType t3 = pop();
          InferredType t4 = pop();
          push(t2);
          push(t1);
          push(t4);
          push(t3);
          push(t2);
          push(t1);
          break;
        }
      case Opcodes.SWAP:
        {
          InferredType top = pop();
          InferredType next = pop();
          push(top);
          push(next);
          break;
        }
      case Opcodes.IADD:
      case Opcodes.ISUB:
      case Opcodes.IMUL:
      case Opcodes.IDIV:
      case Opcodes.IREM:
      case Opcodes.ISHL:
      case Opcodes.ISHR:
      case Opcodes.IUSHR:
      case Opcodes.IAND:
      case Opcodes.IOR:
      case Opcodes.IXOR:
      case Opcodes.L2I:
      case Opcodes.D2I:
      case Opcodes.FCMPL:
      case Opcodes.FCMPG:
        pop(2);
        push(InferredType.INT);
        break;

      case Opcodes.LADD:
      case Opcodes.LSUB:
      case Opcodes.LMUL:
      case Opcodes.LDIV:
      case Opcodes.LREM:
      case Opcodes.LAND:
      case Opcodes.LOR:
      case Opcodes.LXOR:
        pop(4);
        push(InferredType.LONG);
        push(InferredType.TOP);
        break;

      case Opcodes.LSHL:
      case Opcodes.LSHR:
      case Opcodes.LUSHR:
        pop(3);
        push(InferredType.LONG);
        push(InferredType.TOP);
        break;
      case Opcodes.I2L:
      case Opcodes.F2L:
        pop();
        push(InferredType.LONG);
        push(InferredType.TOP);
        break;
      case Opcodes.I2F:
        pop();
        push(InferredType.FLOAT);
        break;

      case Opcodes.LCMP:
      case Opcodes.DCMPG:
      case Opcodes.DCMPL:
        pop(4);
        push(InferredType.INT);
        break;

      case Opcodes.I2D:
      case Opcodes.F2D:
        pop();
        push(InferredType.DOUBLE);
        push(InferredType.TOP);
        break;
      case Opcodes.F2I:
      case Opcodes.ARRAYLENGTH:
        pop();
        push(InferredType.INT);
        break;
      case Opcodes.FALOAD:
      case Opcodes.FADD:
      case Opcodes.FSUB:
      case Opcodes.FMUL:
      case Opcodes.FDIV:
      case Opcodes.FREM:
      case Opcodes.L2F:
      case Opcodes.D2F:
        pop(2);
        push(InferredType.FLOAT);
        break;

      case Opcodes.DADD:
      case Opcodes.DSUB:
      case Opcodes.DMUL:
      case Opcodes.DDIV:
      case Opcodes.DREM:
        pop(4);
        push(InferredType.DOUBLE);
        push(InferredType.TOP);
        break;
      default:
        throw new RuntimeException("Unhandled opcode " + opcode);
    }
    super.visitInsn(opcode);
  }

  @Override
  public void visitIntInsn(int opcode, int operand) {
    switch (opcode) {
      case Opcodes.BIPUSH:
      case Opcodes.SIPUSH:
        push(InferredType.INT);
        break;
      case Opcodes.NEWARRAY:
        pop();
        switch (operand) {
          case Opcodes.T_BOOLEAN:
            pushDescriptor("[Z");
            break;
          case Opcodes.T_CHAR:
            pushDescriptor("[C");
            break;
          case Opcodes.T_FLOAT:
            pushDescriptor("[F");
            break;
          case Opcodes.T_DOUBLE:
            pushDescriptor("[D");
            break;
          case Opcodes.T_BYTE:
            pushDescriptor("[B");
            break;
          case Opcodes.T_SHORT:
            pushDescriptor("[S");
            break;
          case Opcodes.T_INT:
            pushDescriptor("[I");
            break;
          case Opcodes.T_LONG:
            pushDescriptor("[J");
            break;
          default:
            throw new RuntimeException("Unhandled operand value: " + operand);
        }
        break;
      default:
        throw new RuntimeException("Unhandled opcode " + opcode);
    }
    super.visitIntInsn(opcode, operand);
  }

  @Override
  public void visitVarInsn(int opcode, int var) {
    switch (opcode) {
      case Opcodes.ILOAD:
        push(InferredType.INT);
        break;
      case Opcodes.LLOAD:
        push(InferredType.LONG);
        push(InferredType.TOP);
        break;
      case Opcodes.FLOAD:
        push(InferredType.FLOAT);
        break;
      case Opcodes.DLOAD:
        push(InferredType.DOUBLE);
        push(InferredType.TOP);
        break;
      case Opcodes.ALOAD:
        push(getLocalVariableType(var));
        break;
      case Opcodes.ISTORE:
      case Opcodes.FSTORE:
      case Opcodes.ASTORE:
        {
          InferredType type = pop();
          setLocalVariableTypes(var, type);
          break;
        }
      case Opcodes.LSTORE:
      case Opcodes.DSTORE:
        {
          InferredType type = pop(2);
          setLocalVariableTypes(var, type);
          setLocalVariableTypes(var + 1, InferredType.TOP);
          break;
        }
      case Opcodes.RET:
        throw new RuntimeException("The instruction RET is not supported");
      default:
        throw new RuntimeException("Unhandled opcode " + opcode);
    }
    super.visitVarInsn(opcode, var);
  }

  @Override
  public void visitTypeInsn(int opcode, String type) {
    String descriptor = convertToDescriptor(type);
    switch (opcode) {
      case Opcodes.NEW:
        // This should be UNINITIALIZED(label). Okay for type inference.
        pushDescriptor(descriptor);
        break;
      case Opcodes.ANEWARRAY:
        pop();
        pushDescriptor('[' + descriptor);
        break;
      case Opcodes.CHECKCAST:
        pop();
        pushDescriptor(descriptor);
        break;
      case Opcodes.INSTANCEOF:
        pop();
        push(InferredType.INT);
        break;
      default:
        throw new RuntimeException("Unhandled opcode " + opcode);
    }
    super.visitTypeInsn(opcode, type);
  }

  @Override
  public void visitFieldInsn(int opcode, String owner, String name, String desc) {
    switch (opcode) {
      case Opcodes.GETSTATIC:
        pushDescriptor(desc);
        break;
      case Opcodes.PUTSTATIC:
        popDescriptor(desc);
        break;
      case Opcodes.GETFIELD:
        pop();
        pushDescriptor(desc);
        break;
      case Opcodes.PUTFIELD:
        popDescriptor(desc);
        pop();
        break;
      default:
        throw new RuntimeException(
            "Unhandled opcode " + opcode + ", owner=" + owner + ", name=" + name + ", desc" + desc);
    }
    super.visitFieldInsn(opcode, owner, name, desc);
  }

  @Override
  public void visitMethodInsn(int opcode, String owner, String name, String desc, boolean itf) {
    if (opcode == Opcodes.INVOKESPECIAL && "<init>".equals(name)) {
      int argumentSize = (Type.getArgumentsAndReturnSizes(desc) >> 2);
      InferredType receiverType = getTypeOfOperandFromTop(argumentSize - 1);
      if (receiverType.isUninitialized()) {
        InferredType realType = InferredType.create('L' + owner + ';');
        replaceUninitializedTypeInStack(receiverType, realType);
      }
    }
    switch (opcode) {
      case Opcodes.INVOKESPECIAL:
      case Opcodes.INVOKEVIRTUAL:
      case Opcodes.INVOKESTATIC:
      case Opcodes.INVOKEINTERFACE:
        popDescriptor(desc);
        if (opcode != Opcodes.INVOKESTATIC) {
          pop(); // Pop receiver.
        }
        pushDescriptor(desc);
        break;
      default:
        throw new RuntimeException(
            String.format(
                "Unhandled opcode %s, owner=%s, name=%s, desc=%s, itf=%s",
                opcode, owner, name, desc, itf));
    }
    super.visitMethodInsn(opcode, owner, name, desc, itf);
  }

  @Override
  public void visitInvokeDynamicInsn(String name, String desc, Handle bsm, Object... bsmArgs) {
    popDescriptor(desc);
    pushDescriptor(desc);
    super.visitInvokeDynamicInsn(name, desc, bsm, bsmArgs);
  }

  @Override
  public void visitJumpInsn(int opcode, Label label) {
    switch (opcode) {
      case Opcodes.IFEQ:
      case Opcodes.IFNE:
      case Opcodes.IFLT:
      case Opcodes.IFGE:
      case Opcodes.IFGT:
      case Opcodes.IFLE:
        pop();
        break;
      case Opcodes.IF_ICMPEQ:
      case Opcodes.IF_ICMPNE:
      case Opcodes.IF_ICMPLT:
      case Opcodes.IF_ICMPGE:
      case Opcodes.IF_ICMPGT:
      case Opcodes.IF_ICMPLE:
      case Opcodes.IF_ACMPEQ:
      case Opcodes.IF_ACMPNE:
        pop(2);
        break;
      case Opcodes.GOTO:
        break;
      case Opcodes.JSR:
        throw new RuntimeException("The JSR instruction is not supported.");
      case Opcodes.IFNULL:
      case Opcodes.IFNONNULL:
        pop(1);
        break;
      default:
        throw new RuntimeException("Unhandled opcode " + opcode);
    }
    super.visitJumpInsn(opcode, label);
  }

  @Override
  public void visitLdcInsn(Object cst) {
    if (cst instanceof Integer) {
      push(InferredType.INT);
    } else if (cst instanceof Float) {
      push(InferredType.FLOAT);
    } else if (cst instanceof Long) {
      push(InferredType.LONG);
      push(InferredType.TOP);
    } else if (cst instanceof Double) {
      push(InferredType.DOUBLE);
      push(InferredType.TOP);
    } else if (cst instanceof String) {
      pushDescriptor("Ljava/lang/String;");
    } else if (cst instanceof Type) {
      pushDescriptor(((Type) cst).getDescriptor());
    } else if (cst instanceof Handle) {
      pushDescriptor("Ljava/lang/invoke/MethodHandle;");
    } else {
      throw new RuntimeException("Cannot handle constant " + cst + " for LDC instruction");
    }
    super.visitLdcInsn(cst);
  }

  @Override
  public void visitIincInsn(int var, int increment) {
    setLocalVariableTypes(var, InferredType.INT);
    super.visitIincInsn(var, increment);
  }

  @Override
  public void visitTableSwitchInsn(int min, int max, Label dflt, Label... labels) {
    pop();
    super.visitTableSwitchInsn(min, max, dflt, labels);
  }

  @Override
  public void visitLookupSwitchInsn(Label dflt, int[] keys, Label[] labels) {
    pop();
    super.visitLookupSwitchInsn(dflt, keys, labels);
  }

  @Override
  public void visitMultiANewArrayInsn(String desc, int dims) {
    pop(dims);
    pushDescriptor(desc);
    super.visitMultiANewArrayInsn(desc, dims);
  }

  @Override
  public void visitFrame(int type, int nLocal, Object[] local, int nStack, Object[] stack) {
    switch (type) {
      case Opcodes.F_NEW:
        // Expanded form.
        previousFrame =
            FrameInfo.create(
                convertTypesInStackMapFrame(nLocal, local),
                convertTypesInStackMapFrame(nStack, stack));
        break;
      case Opcodes.F_SAME:
        // This frame type indicates that the frame has exactly the same local variables as the
        // previous frame and that the operand stack is empty.
        previousFrame = FrameInfo.create(previousFrame.locals(), ImmutableList.of());
        break;
      case Opcodes.F_SAME1:
        // This frame type indicates that the frame has exactly the same local variables as the
        // previous frame and that the operand stack has one entry.
        previousFrame =
            FrameInfo.create(previousFrame.locals(), convertTypesInStackMapFrame(nStack, stack));
        break;
      case Opcodes.F_APPEND:
        // This frame type indicates that the frame has the same locals as the previous frame except
        // that k additional locals are defined, and that the operand stack is empty.
        previousFrame =
            FrameInfo.create(
                appendArrayToList(previousFrame.locals(), nLocal, local), ImmutableList.of());
        break;
      case Opcodes.F_CHOP:
        // This frame type indicates that the frame has the same local variables as the previous
        // frame except that the last k local variables are absent, and that the operand stack is
        // empty.
        previousFrame =
            FrameInfo.create(
                removeBackFromList(previousFrame.locals(), nLocal), ImmutableList.of());
        break;
      case Opcodes.F_FULL:
        previousFrame =
            FrameInfo.create(
                convertTypesInStackMapFrame(nLocal, local),
                convertTypesInStackMapFrame(nStack, stack));
        break;
      default:
        // continue below
    }
    // Update types for operand stack and local variables.
    operandStack.clear();
    operandStack.addAll(previousFrame.stack());
    localVariableSlots.clear();
    localVariableSlots.addAll(previousFrame.locals());
    super.visitFrame(type, nLocal, local, nStack, stack);
  }

  private static String convertToDescriptor(String type) {
    return (type.length() > 1 && type.charAt(0) != '[') ? 'L' + type + ';' : type;
  }

  private void push(InferredType type) {
    operandStack.add(type);
  }

  private void replaceUninitializedTypeInStack(InferredType oldType, InferredType newType) {
    checkArgument(oldType.isUninitialized(), "The old type is NOT uninitialized. %s", oldType);
    for (int i = 0, size = operandStack.size(); i < size; ++i) {
      InferredType type = operandStack.get(i);
      if (type.equals(oldType)) {
        operandStack.set(i, newType);
      }
    }
  }

  private void pushDescriptor(String desc) {
    int index = desc.charAt(0) == '(' ? desc.indexOf(')') + 1 : 0;
    switch (desc.charAt(index)) {
      case 'V':
        return;
      case 'Z':
      case 'C':
      case 'B':
      case 'S':
      case 'I':
        push(InferredType.INT);
        break;
      case 'F':
        push(InferredType.FLOAT);
        break;
      case 'D':
        push(InferredType.DOUBLE);
        push(InferredType.TOP);
        break;
      case 'J':
        push(InferredType.LONG);
        push(InferredType.TOP);
        break;
      case 'L':
      case '[':
        push(InferredType.create(desc.substring(index)));
        break;
      default:
        throw new RuntimeException("Unhandled type: " + desc);
    }
  }

  @CanIgnoreReturnValue
  private InferredType pop() {
    return pop(1);
  }

  private void popDescriptor(String desc) {
    char c = desc.charAt(0);
    switch (c) {
      case '(':
        int argumentSize = (Type.getArgumentsAndReturnSizes(desc) >> 2) - 1;
        if (argumentSize > 0) {
          pop(argumentSize);
        }
        break;
      case 'J':
      case 'D':
        pop(2);
        break;
      default:
        pop(1);
        break;
    }
  }

  private InferredType getLocalVariableType(int index) {
    checkState(
        index < localVariableSlots.size(),
        "Cannot find type for var %s in method %s",
        index,
        methodSignature);
    return localVariableSlots.get(index);
  }

  private void setLocalVariableTypes(int index, InferredType type) {
    while (localVariableSlots.size() <= index) {
      localVariableSlots.add(InferredType.TOP);
    }
    localVariableSlots.set(index, type);
  }

  private InferredType top() {
    return operandStack.get(operandStack.size() - 1);
  }

  /** Pop elements from the end of the operand stack, and return the last popped element. */
  @CanIgnoreReturnValue
  private InferredType pop(int count) {
    checkArgument(
        count >= 1, "The count should be at least one: %s (In %s)", count, methodSignature);
    checkState(
        operandStack.size() >= count,
        "There are no enough elements in the stack. count=%s, stack=%s (In %s)",
        count,
        operandStack,
        methodSignature);
    int expectedLastIndex = operandStack.size() - count - 1;
    InferredType lastPopped = null;
    for (int i = operandStack.size() - 1; i > expectedLastIndex; --i) {
      lastPopped = operandStack.remove(i);
    }
    return lastPopped;
  }

  /**
   * Create the types of local variables at the very beginning of the method with the information of
   * the declaring class and the method descriptor.
   */
  private static ArrayList<InferredType> createInitialLocalVariableTypes(
      int access, String ownerClass, String methodName, String methodDescriptor) {
    ArrayList<InferredType> types = new ArrayList<>();

    if (!isStatic(access)) {
      // Instance method, and this is the receiver
      types.add(InferredType.create(convertToDescriptor(ownerClass)));
    }
    Type[] argumentTypes = Type.getArgumentTypes(methodDescriptor);
    for (Type argumentType : argumentTypes) {
      switch (argumentType.getSort()) {
        case Type.BOOLEAN:
        case Type.BYTE:
        case Type.CHAR:
        case Type.SHORT:
        case Type.INT:
          types.add(InferredType.INT);
          break;
        case Type.FLOAT:
          types.add(InferredType.FLOAT);
          break;
        case Type.LONG:
          types.add(InferredType.LONG);
          types.add(InferredType.TOP);
          break;
        case Type.DOUBLE:
          types.add(InferredType.DOUBLE);
          types.add(InferredType.TOP);
          break;
        case Type.ARRAY:
        case Type.OBJECT:
          types.add(InferredType.create(argumentType.getDescriptor()));
          break;
        default:
          throw new RuntimeException(
              "Unhandled argument type: "
                  + argumentType
                  + " in "
                  + ownerClass
                  + "."
                  + methodName
                  + methodDescriptor);
      }
    }
    return types;
  }

  private static ImmutableList<InferredType> removeBackFromList(
      ImmutableList<InferredType> list, int countToRemove) {
    int origSize = list.size();
    int index = origSize - 1;

    while (index >= 0 && countToRemove > 0) {
      InferredType type = list.get(index);
      if (type.equals(InferredType.TOP) && index > 0 && list.get(index - 1).isCategory2()) {
        --index; // A category 2 takes two slots.
      }
      --index; // Eat this local variable.
      --countToRemove;
    }
    checkState(
        countToRemove == 0,
        "countToRemove is %s but not 0. index=%s, list=%s",
        countToRemove,
        index,
        list);
    return list.subList(0, index + 1);
  }

  private ImmutableList<InferredType> appendArrayToList(
      ImmutableList<InferredType> list, int size, Object[] array) {
    ImmutableList.Builder<InferredType> builder = ImmutableList.builder();
    builder.addAll(list);
    for (int i = 0; i < size; ++i) {
      InferredType type = convertTypeInStackMapFrame(array[i]);
      builder.add(type);
      if (type.isCategory2()) {
        builder.add(InferredType.TOP);
      }
    }
    return builder.build();
  }

  /** Convert the type in stack map frame to inference type. */
  private InferredType convertTypeInStackMapFrame(Object typeInStackMapFrame) {
    if (typeInStackMapFrame == Opcodes.TOP) {
      return InferredType.TOP;
    } else if (typeInStackMapFrame == Opcodes.INTEGER) {
      return InferredType.INT;
    } else if (typeInStackMapFrame == Opcodes.FLOAT) {
      return InferredType.FLOAT;
    } else if (typeInStackMapFrame == Opcodes.DOUBLE) {
      return InferredType.DOUBLE;
    } else if (typeInStackMapFrame == Opcodes.LONG) {
      return InferredType.LONG;
    } else if (typeInStackMapFrame == Opcodes.NULL) {
      return InferredType.NULL;
    } else if (typeInStackMapFrame == Opcodes.UNINITIALIZED_THIS) {
      return InferredType.UNINITIALIZED_THIS;
    } else if (typeInStackMapFrame instanceof String) {
      String referenceTypeName = (String) typeInStackMapFrame;
      if (referenceTypeName.charAt(0) == '[') {
        return InferredType.create(referenceTypeName);
      } else {
        return InferredType.create('L' + referenceTypeName + ';');
      }
    } else if (typeInStackMapFrame instanceof Label) {
      return InferredType.UNINITIALIZED;
    } else {
      throw new RuntimeException(
          "Cannot reach here. Unhandled element: value="
              + typeInStackMapFrame
              + ", class="
              + typeInStackMapFrame.getClass()
              + ". The current method being desugared is "
              + methodSignature);
    }
  }

  private ImmutableList<InferredType> convertTypesInStackMapFrame(int size, Object[] array) {
    ImmutableList.Builder<InferredType> builder = ImmutableList.builder();
    for (int i = 0; i < size; ++i) {
      InferredType type = convertTypeInStackMapFrame(array[i]);
      builder.add(type);
      if (type.isCategory2()) {
        builder.add(InferredType.TOP);
      }
    }
    return builder.build();
  }

  /** A value class to represent a frame. */
  @AutoValue
  abstract static class FrameInfo {

    static FrameInfo create(ImmutableList<InferredType> locals, ImmutableList<InferredType> stack) {
      return new AutoValue_BytecodeTypeInference_FrameInfo(locals, stack);
    }

    abstract ImmutableList<InferredType> locals();

    abstract ImmutableList<InferredType> stack();
  }

  /** This is the type used for type inference. */
  @AutoValue
  abstract static class InferredType {

    static final String UNINITIALIZED_PREFIX = "UNINIT@";

    static final InferredType BOOLEAN = new AutoValue_BytecodeTypeInference_InferredType("Z");
    static final InferredType BYTE = new AutoValue_BytecodeTypeInference_InferredType("B");
    static final InferredType INT = new AutoValue_BytecodeTypeInference_InferredType("I");
    static final InferredType FLOAT = new AutoValue_BytecodeTypeInference_InferredType("F");
    static final InferredType LONG = new AutoValue_BytecodeTypeInference_InferredType("J");
    static final InferredType DOUBLE = new AutoValue_BytecodeTypeInference_InferredType("D");
    /** Not a real value. */
    static final InferredType TOP = new AutoValue_BytecodeTypeInference_InferredType("TOP");
    /** The value NULL */
    static final InferredType NULL = new AutoValue_BytecodeTypeInference_InferredType("NULL");

    static final InferredType UNINITIALIZED_THIS =
        new AutoValue_BytecodeTypeInference_InferredType("UNINITIALIZED_THIS");

    static final InferredType UNINITIALIZED =
        new AutoValue_BytecodeTypeInference_InferredType(UNINITIALIZED_PREFIX);

    /** Create a type for a value. */
    static InferredType create(String descriptor) {
      if (UNINITIALIZED_PREFIX.equals(descriptor)) {
        return UNINITIALIZED;
      }
      char firstChar = descriptor.charAt(0);
      if (firstChar == 'L' || firstChar == '[') {
        // Reference, array.
        return new AutoValue_BytecodeTypeInference_InferredType(descriptor);
      }
      switch (descriptor) {
        case "Z":
          return BOOLEAN;
        case "B":
          return BYTE;
        case "I":
          return INT;
        case "F":
          return FLOAT;
        case "J":
          return LONG;
        case "D":
          return DOUBLE;
        case "TOP":
          return TOP;
        case "NULL":
          return NULL;
        case "UNINITIALIZED_THIS":
          return UNINITIALIZED_THIS;
        default:
          throw new RuntimeException("Invalid descriptor: " + descriptor);
      }
    }

    abstract String descriptor();

    @Override
    public String toString() {
      return descriptor();
    }

    /** Is a category 2 value? */
    boolean isCategory2() {
      String descriptor = descriptor();
      return descriptor.equals("J") || descriptor.equals("D");
    }

    /** If the type is an array, return the element type. Otherwise, throw an exception. */
    InferredType getElementTypeIfArrayOrThrow() {
      String descriptor = descriptor();
      checkState(descriptor.charAt(0) == '[', "This type %s is not an array.", this);
      return create(descriptor.substring(1));
    }

    /** Is an uninitialized value? */
    boolean isUninitialized() {
      return descriptor().startsWith(UNINITIALIZED_PREFIX);
    }

    /** Is a null value? */
    boolean isNull() {
      return NULL.equals(this);
    }

    /**
     * If this type is a reference type, then return the internal name. Otherwise, returns empty.
     */
    Optional<String> getInternalName() {
      String descriptor = descriptor();
      int length = descriptor.length();
      if (length > 0 && descriptor.charAt(0) == 'L' && descriptor.charAt(length - 1) == ';') {
        return Optional.of(descriptor.substring(1, length - 1));
      } else {
        return Optional.empty();
      }
    }
  }
}

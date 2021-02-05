/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.classfile.instruction;

import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.ClassUtil;

/**
 * This Instruction represents an instruction that refers to an entry in the
 * constant pool.
 *
 * @author Eric Lafortune
 */
public class ConstantInstruction extends Instruction
implements   ConstantVisitor
{
    public int constantIndex;
    public int constant;


    // Fields acting as return parameters for the ConstantVisitor methods.
    private int parameterStackDelta;
    private int typeStackDelta;


    /**
     * Creates an uninitialized ConstantInstruction.
     */
    public ConstantInstruction() {}


    /**
     * Creates a new ConstantInstruction with the given opcode and constant pool
     * index.
     */
    public ConstantInstruction(byte opcode, int constantIndex)
    {
        this(opcode, constantIndex, 0);
    }


    /**
     * Creates a new ConstantInstruction with the given opcode, constant pool
     * index, and constant.
     */
    public ConstantInstruction(byte opcode, int constantIndex, int constant)
    {
        this.opcode        = opcode;
        this.constantIndex = constantIndex;
        this.constant      = constant;
    }


    /**
     * Copies the given instruction into this instruction.
     * @param constantInstruction the instruction to be copied.
     * @return this instruction.
     */
    public ConstantInstruction copy(ConstantInstruction constantInstruction)
    {
        this.opcode        = constantInstruction.opcode;
        this.constantIndex = constantInstruction.constantIndex;
        this.constant      = constantInstruction.constant;

        return this;
    }


    // Implementations for Instruction.

    public byte canonicalOpcode()
    {
        // Remove the _w extension, if any.
        return
            opcode == InstructionConstants.OP_LDC_W ? InstructionConstants.OP_LDC :
                                                      opcode;
    }

    public Instruction shrink()
    {
        // Do we need a short index or a long index?
        if (requiredConstantIndexSize() == 1)
        {
            // Can we replace the long instruction by a short instruction?
            if (opcode == InstructionConstants.OP_LDC_W)
            {
                opcode = InstructionConstants.OP_LDC;
            }
        }
        else
        {
            // Should we replace the short instruction by a long instruction?
            if (opcode == InstructionConstants.OP_LDC)
            {
                opcode = InstructionConstants.OP_LDC_W;
            }
        }

        return this;
    }

    protected void readInfo(byte[] code, int offset)
    {
        int constantIndexSize = constantIndexSize();
        int constantSize      = constantSize();

        constantIndex = readValue(code, offset, constantIndexSize);  offset += constantIndexSize;
        constant      = readValue(code, offset, constantSize);
    }


    protected void writeInfo(byte[] code, int offset)
    {
        int constantIndexSize = constantIndexSize();
        int constantSize      = constantSize();

        if (requiredConstantIndexSize() > constantIndexSize)
        {
            throw new IllegalArgumentException("Instruction has invalid constant index size ("+this.toString(offset)+")");
        }

        writeValue(code, offset, constantIndex, constantIndexSize); offset += constantIndexSize;
        writeValue(code, offset, constant,      constantSize);
    }


    public int length(int offset)
    {
        return 1 + constantIndexSize() + constantSize();
    }


    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, InstructionVisitor instructionVisitor)
    {
        instructionVisitor.visitConstantInstruction(clazz, method, codeAttribute, offset, this);
    }


    public int stackPopCount(Clazz clazz)
    {
        int stackPopCount = super.stackPopCount(clazz);

        // Some special cases.
        switch (opcode)
        {
            case InstructionConstants.OP_MULTIANEWARRAY:
                // For each dimension, an integer size is popped from the stack.
                stackPopCount += constant;
                break;

            case InstructionConstants.OP_PUTSTATIC:
            case InstructionConstants.OP_PUTFIELD:
                // The field value is be popped from the stack.
                clazz.constantPoolEntryAccept(constantIndex, this);
                stackPopCount += typeStackDelta;
                break;

            case InstructionConstants.OP_INVOKEVIRTUAL:
            case InstructionConstants.OP_INVOKESPECIAL:
            case InstructionConstants.OP_INVOKESTATIC:
            case InstructionConstants.OP_INVOKEINTERFACE:
            case InstructionConstants.OP_INVOKEDYNAMIC:
                // Some parameters may be popped from the stack.
                clazz.constantPoolEntryAccept(constantIndex, this);
                stackPopCount += parameterStackDelta;
                break;
        }

        return stackPopCount;
    }


    public int stackPushCount(Clazz clazz)
    {
        int stackPushCount = super.stackPushCount(clazz);

        // Some special cases.
        switch (opcode)
        {
            case InstructionConstants.OP_GETSTATIC:
            case InstructionConstants.OP_GETFIELD:
            case InstructionConstants.OP_INVOKEVIRTUAL:
            case InstructionConstants.OP_INVOKESPECIAL:
            case InstructionConstants.OP_INVOKESTATIC:
            case InstructionConstants.OP_INVOKEINTERFACE:
            case InstructionConstants.OP_INVOKEDYNAMIC:
                // The field value or a return value may be pushed onto the stack.
                clazz.constantPoolEntryAccept(constantIndex, this);
                stackPushCount += typeStackDelta;
                break;
        }

        return stackPushCount;
    }


    // Implementations for ConstantVisitor.

    public void visitIntegerConstant(Clazz clazz, IntegerConstant integerConstant) {}
    public void visitLongConstant(Clazz clazz, LongConstant longConstant) {}
    public void visitFloatConstant(Clazz clazz, FloatConstant floatConstant) {}
    public void visitDoubleConstant(Clazz clazz, DoubleConstant doubleConstant) {}
    public void visitPrimitiveArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant) {}
    public void visitStringConstant(Clazz clazz, StringConstant stringConstant) {}
    public void visitUtf8Constant(Clazz clazz, Utf8Constant utf8Constant) {}
    public void visitMethodHandleConstant(Clazz clazz, MethodHandleConstant methodHandleConstant) {}
    public void visitClassConstant(Clazz clazz, ClassConstant classConstant) {}
    public void visitMethodTypeConstant(Clazz clazz, MethodTypeConstant methodTypeConstant) {}
    public void visitModuleConstant(Clazz clazz, ModuleConstant moduleConstant) {}
    public void visitPackageConstant(Clazz clazz, PackageConstant packageConstant) {}


    public void visitFieldrefConstant(Clazz clazz, FieldrefConstant fieldrefConstant)
    {
        String type = fieldrefConstant.getType(clazz);

        typeStackDelta = ClassUtil.internalTypeSize(ClassUtil.internalMethodReturnType(type));
    }


    public void visitDynamicConstant(Clazz clazz, DynamicConstant dynamicConstant)
    {
        clazz.constantPoolEntryAccept(dynamicConstant.u2nameAndTypeIndex, this);
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        clazz.constantPoolEntryAccept(invokeDynamicConstant.u2nameAndTypeIndex, this);
    }


    public void visitInterfaceMethodrefConstant(Clazz clazz, InterfaceMethodrefConstant interfaceMethodrefConstant)
    {
        clazz.constantPoolEntryAccept(interfaceMethodrefConstant.u2nameAndTypeIndex, this);
    }


    public void visitMethodrefConstant(Clazz clazz, MethodrefConstant methodrefConstant)
    {
        clazz.constantPoolEntryAccept(methodrefConstant.u2nameAndTypeIndex, this);
    }


    public void visitNameAndTypeConstant(Clazz clazz, NameAndTypeConstant nameAndTypeConstant)
    {
        String type = nameAndTypeConstant.getType(clazz);

        parameterStackDelta = ClassUtil.internalMethodParameterSize(type);
        typeStackDelta      = ClassUtil.internalTypeSize(ClassUtil.internalMethodReturnType(type));
    }


    // Implementations for Object.

    public String toString()
    {
        return getName()+" #"+constantIndex+(constantSize() == 0 ? "" : ", "+constant);
    }


    // Small utility methods.

    /**
     * Returns the constant pool index size for this instruction.
     */
    private int constantIndexSize()
    {
        return opcode == InstructionConstants.OP_LDC ? 1 :
                                                       2;
    }


    /**
     * Returns the constant size for this instruction.
     */
    private int constantSize()
    {
        return opcode == InstructionConstants.OP_MULTIANEWARRAY  ? 1 :
               opcode == InstructionConstants.OP_INVOKEDYNAMIC ||
               opcode == InstructionConstants.OP_INVOKEINTERFACE ? 2 :
                                                                   0;
    }


    /**
     * Computes the required constant pool index size for this instruction's
     * constant pool index.
     */
    private int requiredConstantIndexSize()
    {
        return (constantIndex &   0xff) == constantIndex ? 1 :
               (constantIndex & 0xffff) == constantIndex ? 2 :
                                                           4;
    }
}

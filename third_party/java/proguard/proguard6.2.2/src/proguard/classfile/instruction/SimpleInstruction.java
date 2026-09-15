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
import proguard.classfile.instruction.visitor.InstructionVisitor;

/**
 * This Instruction represents a simple instruction without variable arguments
 * or constant pool references.
 *
 * @author Eric Lafortune
 */
public class SimpleInstruction extends Instruction
{
    public int constant;


    /**
     * Creates an uninitialized SimpleInstruction.
     */
    public SimpleInstruction() {}


    /**
     * Creates a new SimpleInstruction with the given opcode.
     */
    public SimpleInstruction(byte opcode)
    {
        this(opcode, embeddedConstant(opcode));
    }


    /**
     * Creates a new SimpleInstruction with the given opcode and constant.
     */
    public SimpleInstruction(byte opcode, int constant)
    {
        this.opcode   = opcode;
        this.constant = constant;
    }


    /**
     * Copies the given instruction into this instruction.
     * @param simpleInstruction the instruction to be copied.
     * @return this instruction.
     */
    public SimpleInstruction copy(SimpleInstruction simpleInstruction)
    {
        this.opcode   = simpleInstruction.opcode;
        this.constant = simpleInstruction.constant;

        return this;
    }


    /**
     * Return the embedded constant of the given opcode, or 0 if the opcode
     * doesn't have one.
     */
    private static int embeddedConstant(byte opcode)
    {
        switch (opcode)
        {
            case InstructionConstants.OP_ICONST_M1: return -1;

            case InstructionConstants.OP_ICONST_1:
            case InstructionConstants.OP_LCONST_1:
            case InstructionConstants.OP_FCONST_1:
            case InstructionConstants.OP_DCONST_1: return 1;

            case InstructionConstants.OP_ICONST_2:
            case InstructionConstants.OP_FCONST_2: return 2;

            case InstructionConstants.OP_ICONST_3: return 3;

            case InstructionConstants.OP_ICONST_4: return 4;

            case InstructionConstants.OP_ICONST_5: return 5;

            default: return 0;
        }
    }


    // Implementations for Instruction.

    public byte canonicalOpcode()
    {
        // Replace any _1, _2, _3,... extension by _0.
        switch (opcode)
        {
            case InstructionConstants.OP_ICONST_M1:
            case InstructionConstants.OP_ICONST_0:
            case InstructionConstants.OP_ICONST_1:
            case InstructionConstants.OP_ICONST_2:
            case InstructionConstants.OP_ICONST_3:
            case InstructionConstants.OP_ICONST_4:
            case InstructionConstants.OP_ICONST_5:
            case InstructionConstants.OP_BIPUSH:
            case InstructionConstants.OP_SIPUSH:   return InstructionConstants.OP_ICONST_0;

            case InstructionConstants.OP_LCONST_0:
            case InstructionConstants.OP_LCONST_1: return InstructionConstants.OP_LCONST_0;

            case InstructionConstants.OP_FCONST_0:
            case InstructionConstants.OP_FCONST_1:
            case InstructionConstants.OP_FCONST_2: return InstructionConstants.OP_FCONST_0;

            case InstructionConstants.OP_DCONST_0:
            case InstructionConstants.OP_DCONST_1: return InstructionConstants.OP_DCONST_0;

            default: return opcode;
        }
    }

    public Instruction shrink()
    {
        // Reconstruct the opcode of the shortest instruction, if there are
        // any alternatives.
        switch (opcode)
        {
            case InstructionConstants.OP_ICONST_M1:
            case InstructionConstants.OP_ICONST_0:
            case InstructionConstants.OP_ICONST_1:
            case InstructionConstants.OP_ICONST_2:
            case InstructionConstants.OP_ICONST_3:
            case InstructionConstants.OP_ICONST_4:
            case InstructionConstants.OP_ICONST_5:
            case InstructionConstants.OP_BIPUSH:
            case InstructionConstants.OP_SIPUSH:
                switch (requiredConstantSize())
                {
                    case 0:
                        opcode = (byte)(InstructionConstants.OP_ICONST_0 + constant);
                        break;
                    case 1:
                        opcode = InstructionConstants.OP_BIPUSH;
                        break;
                    case 2:
                        opcode = InstructionConstants.OP_SIPUSH;
                        break;
                }
                break;

            case InstructionConstants.OP_LCONST_0:
            case InstructionConstants.OP_LCONST_1:
                opcode = (byte)(InstructionConstants.OP_LCONST_0 + constant);
                break;

            case InstructionConstants.OP_FCONST_0:
            case InstructionConstants.OP_FCONST_1:
            case InstructionConstants.OP_FCONST_2:
                opcode = (byte)(InstructionConstants.OP_FCONST_0 + constant);
                break;

            case InstructionConstants.OP_DCONST_0:
            case InstructionConstants.OP_DCONST_1:
                opcode = (byte)(InstructionConstants.OP_DCONST_0 + constant);
                break;
        }

        return this;
    }

    protected void readInfo(byte[] code, int offset)
    {
        int constantSize = constantSize();

        // Also initialize embedded constants that are different from 0.
        constant = constantSize == 0 ?
            embeddedConstant(opcode) :
            readSignedValue(code, offset, constantSize);
    }


    protected void writeInfo(byte[] code, int offset)
    {
        int constantSize = constantSize();

        if (requiredConstantSize() > constantSize)
        {
            throw new IllegalArgumentException("Instruction has invalid constant size ("+this.toString(offset)+")");
        }

        writeSignedValue(code, offset, constant, constantSize);
    }


    public int length(int offset)
    {
        return 1 + constantSize();
    }


    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, InstructionVisitor instructionVisitor)
    {
        instructionVisitor.visitSimpleInstruction(clazz, method, codeAttribute, offset, this);
    }


    // Implementations for Object.

    public String toString()
    {
        return getName() +
               (constantSize() > 0 ? " "+constant : "");
    }


    // Small utility methods.

    /**
     * Returns the constant size for this instruction.
     */
    private int constantSize()
    {
        return opcode == InstructionConstants.OP_BIPUSH ||
               opcode == InstructionConstants.OP_NEWARRAY ? 1 :
               opcode == InstructionConstants.OP_SIPUSH   ? 2 :
                                                            0;
    }


    /**
     * Computes the required constant size for this instruction.
     */
    private int requiredConstantSize()
    {
        return constant >= -1 && constant <= 5  ? 0 :
               (byte)constant  == constant      ? 1 :
               (short)constant == constant      ? 2 :
                                                  4;
    }
}

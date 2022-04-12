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
 * This Instruction represents an instruction that refers to a variable on the
 * local variable stack.
 *
 * @author Eric Lafortune
 */
public class VariableInstruction extends Instruction
{
    public boolean wide;
    public int     variableIndex;
    public int     constant;


    /**
     * Creates an uninitialized VariableInstruction.
     */
    public VariableInstruction() {}


    public VariableInstruction(boolean wide)
    {
        this.wide = wide;
    }


    public VariableInstruction(byte opcode)
    {
        this(opcode, embeddedVariable(opcode), 0);
    }


    public VariableInstruction(byte opcode,
                               int  variableIndex)
    {
        this(opcode, variableIndex, 0);
    }


    public VariableInstruction(byte opcode,
                               int  variableIndex,
                               int  constant)
    {
        this.opcode        = opcode;
        this.variableIndex = variableIndex;
        this.constant      = constant;
        this.wide          = requiredVariableIndexSize() > 1 ||
                             requiredConstantSize()      > 1;
    }


    /**
     * Copies the given instruction into this instruction.
     * @param variableInstruction the instruction to be copied.
     * @return this instruction.
     */
    public VariableInstruction copy(VariableInstruction variableInstruction)
    {
        this.opcode        = variableInstruction.opcode;
        this.variableIndex = variableInstruction.variableIndex;
        this.constant      = variableInstruction.constant;
        this.wide          = variableInstruction.wide;

        return this;
    }


    /**
     * Return the embedded variable of the given opcode, or 0 if the opcode
     * doesn't have one.
     */
    private static int embeddedVariable(byte opcode)
    {
        switch (opcode)
        {
            case InstructionConstants.OP_ILOAD_1:
            case InstructionConstants.OP_LLOAD_1:
            case InstructionConstants.OP_FLOAD_1:
            case InstructionConstants.OP_DLOAD_1:
            case InstructionConstants.OP_ALOAD_1:
            case InstructionConstants.OP_ISTORE_1:
            case InstructionConstants.OP_LSTORE_1:
            case InstructionConstants.OP_FSTORE_1:
            case InstructionConstants.OP_DSTORE_1:
            case InstructionConstants.OP_ASTORE_1: return 1;

            case InstructionConstants.OP_ILOAD_2:
            case InstructionConstants.OP_LLOAD_2:
            case InstructionConstants.OP_FLOAD_2:
            case InstructionConstants.OP_DLOAD_2:
            case InstructionConstants.OP_ALOAD_2:
            case InstructionConstants.OP_ISTORE_2:
            case InstructionConstants.OP_LSTORE_2:
            case InstructionConstants.OP_FSTORE_2:
            case InstructionConstants.OP_DSTORE_2:
            case InstructionConstants.OP_ASTORE_2: return 2;

            case InstructionConstants.OP_ILOAD_3:
            case InstructionConstants.OP_LLOAD_3:
            case InstructionConstants.OP_FLOAD_3:
            case InstructionConstants.OP_DLOAD_3:
            case InstructionConstants.OP_ALOAD_3:
            case InstructionConstants.OP_ISTORE_3:
            case InstructionConstants.OP_LSTORE_3:
            case InstructionConstants.OP_FSTORE_3:
            case InstructionConstants.OP_DSTORE_3:
            case InstructionConstants.OP_ASTORE_3: return 3;

            default: return 0;
        }
    }


    /**
     * Returns whether this instruction stores the value of a variable.
     * The value is false for the ret instruction, but true for the iinc
     * instruction.
     */
    public boolean isStore()
    {
        // A store instruction can be recognized as follows. Note that this
        // excludes the ret instruction, which has a negative opcode.
        return opcode >= InstructionConstants.OP_ISTORE ||
               opcode == InstructionConstants.OP_IINC;
    }


    /**
     * Returns whether this instruction loads the value of a variable.
     * The value is true for the ret instruction and for the iinc
     * instruction.
     */
    public boolean isLoad()
    {
        // A load instruction can be recognized as follows. Note that this
        // includes the ret instruction, which has a negative opcode.
        return opcode < InstructionConstants.OP_ISTORE;
    }


    // Implementations for Instruction.

    public byte canonicalOpcode()
    {
        // Remove the _0, _1, _2, _3 extension, if any.
        switch (opcode)
        {
            case InstructionConstants.OP_ILOAD_0:
            case InstructionConstants.OP_ILOAD_1:
            case InstructionConstants.OP_ILOAD_2:
            case InstructionConstants.OP_ILOAD_3: return InstructionConstants.OP_ILOAD;
            case InstructionConstants.OP_LLOAD_0:
            case InstructionConstants.OP_LLOAD_1:
            case InstructionConstants.OP_LLOAD_2:
            case InstructionConstants.OP_LLOAD_3: return InstructionConstants.OP_LLOAD;
            case InstructionConstants.OP_FLOAD_0:
            case InstructionConstants.OP_FLOAD_1:
            case InstructionConstants.OP_FLOAD_2:
            case InstructionConstants.OP_FLOAD_3: return InstructionConstants.OP_FLOAD;
            case InstructionConstants.OP_DLOAD_0:
            case InstructionConstants.OP_DLOAD_1:
            case InstructionConstants.OP_DLOAD_2:
            case InstructionConstants.OP_DLOAD_3: return InstructionConstants.OP_DLOAD;
            case InstructionConstants.OP_ALOAD_0:
            case InstructionConstants.OP_ALOAD_1:
            case InstructionConstants.OP_ALOAD_2:
            case InstructionConstants.OP_ALOAD_3: return InstructionConstants.OP_ALOAD;

            case InstructionConstants.OP_ISTORE_0:
            case InstructionConstants.OP_ISTORE_1:
            case InstructionConstants.OP_ISTORE_2:
            case InstructionConstants.OP_ISTORE_3: return InstructionConstants.OP_ISTORE;
            case InstructionConstants.OP_LSTORE_0:
            case InstructionConstants.OP_LSTORE_1:
            case InstructionConstants.OP_LSTORE_2:
            case InstructionConstants.OP_LSTORE_3: return InstructionConstants.OP_LSTORE;
            case InstructionConstants.OP_FSTORE_0:
            case InstructionConstants.OP_FSTORE_1:
            case InstructionConstants.OP_FSTORE_2:
            case InstructionConstants.OP_FSTORE_3: return InstructionConstants.OP_FSTORE;
            case InstructionConstants.OP_DSTORE_0:
            case InstructionConstants.OP_DSTORE_1:
            case InstructionConstants.OP_DSTORE_2:
            case InstructionConstants.OP_DSTORE_3: return InstructionConstants.OP_DSTORE;
            case InstructionConstants.OP_ASTORE_0:
            case InstructionConstants.OP_ASTORE_1:
            case InstructionConstants.OP_ASTORE_2:
            case InstructionConstants.OP_ASTORE_3: return InstructionConstants.OP_ASTORE;

            default: return opcode;
        }
    }

    public Instruction shrink()
    {
        opcode = canonicalOpcode();

        // Is this instruction pointing to a variable with index from 0 to 3?
        if (variableIndex <= 3)
        {
            switch (opcode)
            {
                case InstructionConstants.OP_ILOAD: opcode = (byte)(InstructionConstants.OP_ILOAD_0 + variableIndex); break;
                case InstructionConstants.OP_LLOAD: opcode = (byte)(InstructionConstants.OP_LLOAD_0 + variableIndex); break;
                case InstructionConstants.OP_FLOAD: opcode = (byte)(InstructionConstants.OP_FLOAD_0 + variableIndex); break;
                case InstructionConstants.OP_DLOAD: opcode = (byte)(InstructionConstants.OP_DLOAD_0 + variableIndex); break;
                case InstructionConstants.OP_ALOAD: opcode = (byte)(InstructionConstants.OP_ALOAD_0 + variableIndex); break;

                case InstructionConstants.OP_ISTORE: opcode = (byte)(InstructionConstants.OP_ISTORE_0 + variableIndex); break;
                case InstructionConstants.OP_LSTORE: opcode = (byte)(InstructionConstants.OP_LSTORE_0 + variableIndex); break;
                case InstructionConstants.OP_FSTORE: opcode = (byte)(InstructionConstants.OP_FSTORE_0 + variableIndex); break;
                case InstructionConstants.OP_DSTORE: opcode = (byte)(InstructionConstants.OP_DSTORE_0 + variableIndex); break;
                case InstructionConstants.OP_ASTORE: opcode = (byte)(InstructionConstants.OP_ASTORE_0 + variableIndex); break;
            }
        }

        // Only make the instruction wide if necessary.
        wide = requiredVariableIndexSize() > 1 ||
               requiredConstantSize()      > 1;

        return this;
    }


    protected boolean isWide()
    {
        return wide;
    }


    protected void readInfo(byte[] code, int offset)
    {
        int variableIndexSize = variableIndexSize();
        int constantSize      = constantSize();

        // Also initialize embedded variable indexes.
        if (variableIndexSize == 0)
        {
            // An embedded variable index can be decoded as follows.
            variableIndex = opcode < InstructionConstants.OP_ISTORE_0 ?
                (opcode - InstructionConstants.OP_ILOAD_0 ) & 3 :
                (opcode - InstructionConstants.OP_ISTORE_0) & 3;
        }
        else
        {
            variableIndex = readValue(code, offset, variableIndexSize); offset += variableIndexSize;
        }

        constant = readSignedValue(code, offset, constantSize);
    }


    protected void writeInfo(byte[] code, int offset)
    {
        int variableIndexSize = variableIndexSize();
        int constantSize      = constantSize();

        if (requiredVariableIndexSize() > variableIndexSize)
        {
            throw new IllegalArgumentException("Instruction has invalid variable index size ("+this.toString(offset)+")");
        }

        if (requiredConstantSize() > constantSize)
        {
            throw new IllegalArgumentException("Instruction has invalid constant size ("+this.toString(offset)+")");
        }

        writeValue(code, offset, variableIndex, variableIndexSize); offset += variableIndexSize;
        writeSignedValue(code, offset, constant, constantSize);
    }


    public int length(int offset)
    {
        return (wide ? 2 : 1) + variableIndexSize() + constantSize();
    }


    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, InstructionVisitor instructionVisitor)
    {
        instructionVisitor.visitVariableInstruction(clazz, method, codeAttribute, offset, this);
    }


    // Implementations for Object.

    public String toString()
    {
        return getName() +
               (wide ? "_w" : "") +
               " v"+variableIndex +
               (constantSize() > 0 ? ", "+constant : "");
    }


    // Small utility methods.

    /**
     * Returns the variable index size for this instruction.
     */
    private int variableIndexSize()
    {
        return (opcode >= InstructionConstants.OP_ILOAD_0 &&
                opcode <= InstructionConstants.OP_ALOAD_3) ||
               (opcode >= InstructionConstants.OP_ISTORE_0 &&
                opcode <= InstructionConstants.OP_ASTORE_3) ? 0 :
               wide                                         ? 2 :
                                                              1;
    }


    /**
     * Computes the required variable index size for this instruction's variable
     * index.
     */
    private int requiredVariableIndexSize()
    {
        return (variableIndex &    0x3) == variableIndex ? 0 :
               (variableIndex &   0xff) == variableIndex ? 1 :
               (variableIndex & 0xffff) == variableIndex ? 2 :
                                                           4;

    }


    /**
     * Returns the constant size for this instruction.
     */
    private int constantSize()
    {
        return opcode != InstructionConstants.OP_IINC ? 0 :
               wide                                   ? 2 :
                                                        1;
    }


    /**
     * Computes the required constant size for this instruction's constant.
     */
    private int requiredConstantSize()
    {
        return opcode != InstructionConstants.OP_IINC ? 0 :
               (byte)constant  == constant            ? 1 :
               (short)constant == constant            ? 2 :
                                                        4;
    }
}

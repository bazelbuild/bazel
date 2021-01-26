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
public class LookUpSwitchInstruction extends SwitchInstruction
{
    public int[] cases;


    /**
     * Creates an uninitialized LookUpSwitchInstruction.
     */
    public LookUpSwitchInstruction() {}


    /**
     * Creates a new LookUpSwitchInstruction with the given arguments.
     */
    public LookUpSwitchInstruction(byte  opcode,
                                   int   defaultOffset,
                                   int[] cases,
                                   int[] jumpOffsets)
    {
        this.opcode        = opcode;
        this.defaultOffset = defaultOffset;
        this.cases         = cases;
        this.jumpOffsets   = jumpOffsets;
    }


    /**
     * Copies the given instruction into this instruction.
     * @param lookUpSwitchInstruction the instruction to be copied.
     * @return this instruction.
     */
    public LookUpSwitchInstruction copy(LookUpSwitchInstruction lookUpSwitchInstruction)
    {
        this.opcode        = lookUpSwitchInstruction.opcode;
        this.defaultOffset = lookUpSwitchInstruction.defaultOffset;
        this.cases         = lookUpSwitchInstruction.cases;
        this.jumpOffsets   = lookUpSwitchInstruction.jumpOffsets;

        return this;
    }


    // Implementations for Instruction.

    public Instruction shrink()
    {
        // There aren't any ways to shrink this instruction.
        return this;
    }

    protected void readInfo(byte[] code, int offset)
    {
        // Skip up to three padding bytes.
        offset += -offset & 3;

        // Read the two 32-bit arguments.
        defaultOffset       = readInt(code, offset); offset += 4;
        int jumpOffsetCount = readInt(code, offset); offset += 4;

        // Read the matches-offset pairs.
        cases       = new int[jumpOffsetCount];
        jumpOffsets = new int[jumpOffsetCount];

        for (int index = 0; index < jumpOffsetCount; index++)
        {
            cases[index]       = readInt(code, offset); offset += 4;
            jumpOffsets[index] = readInt(code, offset); offset += 4;
        }
    }


    protected void writeInfo(byte[] code, int offset)
    {
        // Write up to three padding bytes.
        while ((offset & 3) != 0)
        {
            writeByte(code, offset++, 0);
        }

        // Write the two 32-bit arguments.
        writeInt(code, offset, defaultOffset); offset += 4;
        writeInt(code, offset, cases.length);  offset += 4;

        // Write the matches-offset pairs.
        for (int index = 0; index < cases.length; index++)
        {
            writeInt(code, offset, cases[index]);       offset += 4;
            writeInt(code, offset, jumpOffsets[index]); offset += 4;
        }
    }


    public int length(int offset)
    {
        return 1 + (-(offset+1) & 3) + 8 + cases.length * 8;
    }


    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, InstructionVisitor instructionVisitor)
    {
        instructionVisitor.visitLookUpSwitchInstruction(clazz, method, codeAttribute, offset, this);
    }
}

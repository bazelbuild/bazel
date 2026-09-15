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
public class TableSwitchInstruction extends SwitchInstruction
{
    public int lowCase;
    public int highCase;


    /**
     * Creates an uninitialized TableSwitchInstruction.
     */
    public TableSwitchInstruction() {}


    /**
     * Creates a new TableSwitchInstruction with the given arguments.
     */
    public TableSwitchInstruction(byte  opcode,
                                  int   defaultOffset,
                                  int   lowCase,
                                  int   highCase,
                                  int[] jumpOffsets)
    {
        this.opcode        = opcode;
        this.defaultOffset = defaultOffset;
        this.lowCase       = lowCase;
        this.highCase      = highCase;
        this.jumpOffsets   = jumpOffsets;
    }


    /**
     * Copies the given instruction into this instruction.
     * @param tableSwitchInstruction the instruction to be copied.
     * @return this instruction.
     */
    public TableSwitchInstruction copy(TableSwitchInstruction tableSwitchInstruction)
    {
        this.opcode        = tableSwitchInstruction.opcode;
        this.defaultOffset = tableSwitchInstruction.defaultOffset;
        this.lowCase       = tableSwitchInstruction.lowCase;
        this.highCase      = tableSwitchInstruction.highCase;
        this.jumpOffsets   = tableSwitchInstruction.jumpOffsets;

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

        // Read the three 32-bit arguments.
        defaultOffset = readInt(code, offset); offset += 4;
        lowCase       = readInt(code, offset); offset += 4;
        highCase      = readInt(code, offset); offset += 4;

        // Read the jump offsets.
        jumpOffsets = new int[highCase - lowCase + 1];

        for (int index = 0; index < jumpOffsets.length; index++)
        {
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

        // Write the three 32-bit arguments.
        writeInt(code, offset, defaultOffset); offset += 4;
        writeInt(code, offset, lowCase);       offset += 4;
        writeInt(code, offset, highCase);      offset += 4;

        // Write the jump offsets.
        int length = highCase - lowCase + 1;
        for (int index = 0; index < length; index++)
        {
            writeInt(code, offset, jumpOffsets[index]); offset += 4;
        }
    }


    public int length(int offset)
    {
        return 1 + (-(offset+1) & 3) + 12 + (highCase - lowCase + 1) * 4;
    }


    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, InstructionVisitor instructionVisitor)
    {
        instructionVisitor.visitTableSwitchInstruction(clazz, method, codeAttribute, offset, this);
    }
}

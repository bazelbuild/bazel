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

/**
 * This Instruction represents a simple instruction without variable arguments
 * or constant pool references.
 *
 * @author Eric Lafortune
 */
public abstract class SwitchInstruction extends Instruction
{
    public int   defaultOffset;
    public int[] jumpOffsets;


    /**
     * Creates an uninitialized SwitchInstruction.
     */
    public SwitchInstruction() {}


    /**
     * Creates a new SwitchInstruction with the given arguments.
     */
    public SwitchInstruction(byte  opcode,
                             int   defaultOffset,
                             int[] jumpOffsets)
    {
        this.opcode        = opcode;
        this.defaultOffset = defaultOffset;
        this.jumpOffsets   = jumpOffsets;
    }


    /**
     * Copies the given instruction into this instruction.
     * @param switchInstruction the instruction to be copied.
     * @return this instruction.
     */
    public SwitchInstruction copy(SwitchInstruction switchInstruction)
    {
        this.opcode        = switchInstruction.opcode;
        this.defaultOffset = switchInstruction.defaultOffset;
        this.jumpOffsets   = switchInstruction.jumpOffsets;

        return this;
    }


    // Implementations for Instruction.

    public String toString(int offset)
    {
        return "["+offset+"] "+toString()+" (target="+(offset+defaultOffset)+")";
    }


    // Implementations for Object.

    public String toString()
    {
        return getName()+" ("+jumpOffsets.length+" offsets, default="+defaultOffset+")";
    }
}

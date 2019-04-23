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
 * This interface describes an instruction that branches to a given offset in
 * the code.
 *
 * @author Eric Lafortune
 */
public class BranchInstruction extends Instruction
{
    public int branchOffset;


    /**
     * Creates an uninitialized BranchInstruction.
     */
    public BranchInstruction() {}


    /**
     * Creates a BranchInstruction with the given branch offset.
     * The branch offset is relative to this instruction's offset.
     */
    public BranchInstruction(byte opcode, int branchOffset)
    {
        this.opcode       = opcode;
        this.branchOffset = branchOffset;
    }


    /**
     * Copies the given instruction into this instruction.
     * @param branchInstruction the instruction to be copied.
     * @return this instruction.
     */
    public BranchInstruction copy(BranchInstruction branchInstruction)
    {
        this.opcode       = branchInstruction.opcode;
        this.branchOffset = branchInstruction.branchOffset;

        return this;
    }


    // Implementations for Instruction.

    public byte canonicalOpcode()
    {
        // Remove the _w extension, if any.
        switch (opcode)
        {
            case InstructionConstants.OP_GOTO_W: return InstructionConstants.OP_GOTO;

            case InstructionConstants.OP_JSR_W: return InstructionConstants.OP_JSR;

            default: return opcode;
        }
    }

    public Instruction shrink()
    {
        // Do we need an ordinary branch or a wide branch?
        if (requiredBranchOffsetSize() == 2)
        {
            // Can we replace the wide branch by an ordinary branch?
            if      (opcode == InstructionConstants.OP_GOTO_W)
            {
                opcode = InstructionConstants.OP_GOTO;
            }
            else if (opcode == InstructionConstants.OP_JSR_W)
            {
                opcode = InstructionConstants.OP_JSR;
            }
        }
        else
        {
            // Can we provide a wide branch?
            if      (opcode == InstructionConstants.OP_GOTO ||
                     opcode == InstructionConstants.OP_GOTO_W)
            {
                opcode = InstructionConstants.OP_GOTO_W;
            }
            else if (opcode == InstructionConstants.OP_JSR)
            {
                opcode = InstructionConstants.OP_JSR_W;
            }
            else
            {
                throw new IllegalArgumentException("Branch instruction can't be widened ("+this.toString()+")");
            }
        }

        return this;
    }

    protected void readInfo(byte[] code, int offset)
    {
        branchOffset = readSignedValue(code, offset, branchOffsetSize());
    }


    protected void writeInfo(byte[] code, int offset)
    {
        if (requiredBranchOffsetSize() > branchOffsetSize())
        {
            throw new IllegalArgumentException("Instruction has invalid branch offset size ("+this.toString(offset)+")");
        }

        writeSignedValue(code, offset, branchOffset, branchOffsetSize());
    }


    public int length(int offset)
    {
        return 1 + branchOffsetSize();
    }


    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, InstructionVisitor instructionVisitor)
    {
        instructionVisitor.visitBranchInstruction(clazz, method, codeAttribute, offset, this);
    }


    public String toString(int offset)
    {
        return "["+offset+"] "+toString()+" (target="+(offset+branchOffset)+")";
    }


    // Implementations for Object.

    public String toString()
    {
        return getName()+" "+(branchOffset >= 0 ? "+" : "")+branchOffset;
    }


    // Small utility methods.

    /**
     * Returns the branch offset size for this instruction.
     */
    private int branchOffsetSize()
    {
        return opcode == InstructionConstants.OP_GOTO_W ||
               opcode == InstructionConstants.OP_JSR_W  ? 4 :
                                                          2;
    }


    /**
     * Computes the required branch offset size for this instruction's branch
     * offset.
     */
    private int requiredBranchOffsetSize()
    {
        return (short)branchOffset == branchOffset ? 2 : 4;
    }
}

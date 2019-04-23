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
package proguard.optimize.peephole;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;

import java.util.Arrays;

/**
 * This AttributeVisitor finds all instruction offsets, branch targets, and
 * exception targets in the CodeAttribute objects that it visits.
 *
 * @author Eric Lafortune
 */
public class ReachableCodeMarker
extends      SimplifiedVisitor
implements   AttributeVisitor,
             InstructionVisitor,
             ExceptionInfoVisitor
{
    private boolean[] isReachable = new boolean[ClassConstants.TYPICAL_CODE_LENGTH];

    private boolean next;
    private boolean evaluateExceptions;


    /**
     * Returns whether the instruction at the given offset is reachable in
     * the CodeAttribute that was visited most recently.
     */
    public boolean isReachable(int offset)
    {
        return isReachable[offset];
    }


    /**
     * Returns whether any of the instructions at the given offsets are
     * reachable in the CodeAttribute that was visited most recently.
     */
    public boolean isReachable(int startOffset, int endOffset)
    {
        // Check if any of the instructions is reachable.
        for (int offset = startOffset; offset < endOffset; offset++)
        {
            if (isReachable[offset])
            {
                return true;
            }
        }

        return false;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Make sure there is a sufficiently large array.
        int codeLength = codeAttribute.u4codeLength;
        if (isReachable.length < codeLength)
        {
            // Create a new array.
            isReachable = new boolean[codeLength];
        }
        else
        {
            // Reset the array.
            Arrays.fill(isReachable, 0, codeLength, false);
        }

        // Mark the code, starting at the entry point.
        markCode(clazz, method, codeAttribute, 0);

        // Mark the exception handlers, iterating as long as necessary.
        do
        {
            evaluateExceptions = false;

            codeAttribute.exceptionsAccept(clazz, method, this);
        }
        while (evaluateExceptions);
    }


    // Implementations for InstructionVisitor.

    public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
    {
        byte opcode = simpleInstruction.opcode;
        if (opcode == InstructionConstants.OP_IRETURN ||
            opcode == InstructionConstants.OP_LRETURN ||
            opcode == InstructionConstants.OP_FRETURN ||
            opcode == InstructionConstants.OP_DRETURN ||
            opcode == InstructionConstants.OP_ARETURN ||
            opcode == InstructionConstants.OP_RETURN  ||
            opcode == InstructionConstants.OP_ATHROW)
        {
            next = false;
        }
    }


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
    }


    public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
    {
        if (variableInstruction.opcode == InstructionConstants.OP_RET)
        {
            next = false;
        }
    }


    public void visitBranchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction branchInstruction)
    {
        // Mark the branch target.
        markBranchTarget(clazz,
                         method,
                         codeAttribute,
                         offset + branchInstruction.branchOffset);

        byte opcode = branchInstruction.opcode;
        if (opcode == InstructionConstants.OP_GOTO ||
            opcode == InstructionConstants.OP_GOTO_W)
        {
            next = false;
        }
    }


    public void visitAnySwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SwitchInstruction switchInstruction)
    {
        // Mark the branch targets of the default jump offset.
        markBranchTarget(clazz,
                         method,
                         codeAttribute,
                         offset + switchInstruction.defaultOffset);

        // Mark the branch targets of the jump offsets.
        markBranchTargets(clazz,
                          method,
                          codeAttribute,
                          offset,
                          switchInstruction.jumpOffsets);

        next = false;
    }


    // Implementations for ExceptionInfoVisitor.

    public void visitExceptionInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, ExceptionInfo exceptionInfo)
    {
        // Mark the exception handler if it's relevant.
        if (!isReachable(exceptionInfo.u2handlerPC) &&
            isReachable(exceptionInfo.u2startPC, exceptionInfo.u2endPC))
        {
            markCode(clazz, method, codeAttribute, exceptionInfo.u2handlerPC);

            evaluateExceptions = true;
        }
    }


    // Small utility methods.

    /**
     * Marks the branch targets of the given jump offsets for the instruction
     * at the given offset.
     */
    private void markBranchTargets(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int[] jumpOffsets)
    {
        for (int index = 0; index < jumpOffsets.length; index++)
        {
            markCode(clazz, method, codeAttribute, offset + jumpOffsets[index]);
        }
    }


    /**
     * Marks the branch target at the given offset.
     */
    private void markBranchTarget(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset)
    {
        boolean oldNext = next;

        markCode(clazz, method, codeAttribute, offset);

        next = oldNext;
    }


    /**
     * Marks the code starting at the given offset.
     */
    private void markCode(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset)
    {
        boolean oldNext = next;

        byte[] code = codeAttribute.code;

        // Continue with the current instruction as long as we haven't marked it
        // yet.
        while (!isReachable[offset])
        {
            // Get the current instruction.
            Instruction instruction = InstructionFactory.create(code, offset);

            // Mark it as reachable.
            isReachable[offset] = true;

            // By default, we'll assume we can continue with the next
            // instruction in a moment.
            next = true;

            // Mark the branch targets, if any.
            instruction.accept(clazz, method, codeAttribute, offset, this);

            // Can we really continue with the next instruction?
            if (!next)
            {
                break;
            }

            // Go to the next instruction.
            offset += instruction.length(offset);
        }

        next = oldNext;
    }
}

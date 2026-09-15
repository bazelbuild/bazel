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
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.optimize.evaluation.InitializationFinder;

import java.util.Arrays;

/**
 * This AttributeVisitor finds all instruction offsets, branch targets, and
 * exception targets in the CodeAttribute objects that it visits.
 *
 * @see InitializationFinder
 * @author Eric Lafortune
 */
public class BranchTargetFinder
extends      SimplifiedVisitor
implements   AttributeVisitor,
             InstructionVisitor,
             ExceptionInfoVisitor,
             ConstantVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("btf") != null;
    //*/

    // We'll explicitly mark instructions that are not part of a subroutine,
    // with NO_SUBROUTINE. Subroutines may just branch back into normal code
    // (e.g. due to a break instruction in Java code), and we want to avoid
    // marking such normal code as subroutine. The first mark wins, so we're
    // assuming that such code is marked as normal code before it is marked
    // as subroutine.
    public static final int UNKNOWN       = -1;
    public static final int NO_SUBROUTINE = -2;

    private static final short INSTRUCTION           = 1 <<  0;
    private static final short CREATION              = 1 <<  1;
    private static final short INITIALIZER           = 1 <<  2;
    private static final short BRANCH_ORIGIN         = 1 <<  3;
    private static final short BRANCH_TARGET         = 1 <<  4;
    private static final short AFTER_BRANCH          = 1 <<  5;
    private static final short EXCEPTION_START       = 1 <<  6;
    private static final short EXCEPTION_END         = 1 <<  7;
    private static final short EXCEPTION_HANDLER     = 1 <<  8;
    private static final short SUBROUTINE_INVOCATION = 1 <<  9;
    private static final short SUBROUTINE_RETURNING  = 1 << 10;


    private short[] instructionMarks      = new short[ClassConstants.TYPICAL_CODE_LENGTH + 1];
    private int[]   subroutineStarts      = new int[ClassConstants.TYPICAL_CODE_LENGTH];
    private int[]   subroutineEnds        = new int[ClassConstants.TYPICAL_CODE_LENGTH];
    private boolean containsSubroutines;

    private boolean repeat;
    private int     currentSubroutineStart;
    private boolean isInitializer;


    /**
     * Returns whether there is an instruction at the given offset in the
     * CodeAttribute that was visited most recently.
     */
    public boolean isInstruction(int offset)
    {
        return (instructionMarks[offset] & INSTRUCTION) != 0;
    }


    /**
     * Returns whether the instruction at the given offset creates a new,
     * uninitialized object instance, in the CodeAttribute that was visited
     * most recently.
     */
    public boolean isCreation(int offset)
    {
        return (instructionMarks[offset] & CREATION) != 0;
    }


    /**
     * Returns whether the instruction at the given offset is the special
     * invocation of an instance initializer, in the CodeAttribute that was
     * visited most recently.
     */
    public boolean isInitializer(int offset)
    {
        return (instructionMarks[offset] & INITIALIZER) != 0;
    }


    /**
     * Returns whether the instruction at the given offset is the target of
     * any kind in the CodeAttribute that was visited most recently.
     */
    public boolean isTarget(int offset)
    {
        return offset == 0 ||
               (instructionMarks[offset] & (BRANCH_TARGET   |
                                            EXCEPTION_START |
                                            EXCEPTION_END   |
                                            EXCEPTION_HANDLER)) != 0;
    }


    /**
     * Returns whether the instruction at the given offset is the origin of a
     * branch instruction in the CodeAttribute that was visited most recently.
     */
    public boolean isBranchOrigin(int offset)
    {
        return (instructionMarks[offset] & BRANCH_ORIGIN) != 0;
    }


    /**
     * Returns whether the instruction at the given offset is the target of a
     * branch instruction in the CodeAttribute that was visited most recently.
     */
    public boolean isBranchTarget(int offset)
    {
        return (instructionMarks[offset] & BRANCH_TARGET) != 0;
    }


    /**
     * Returns whether the instruction at the given offset comes right after a
     * definite branch instruction in the CodeAttribute that was visited most
     * recently.
     */
    public boolean isAfterBranch(int offset)
    {
        return (instructionMarks[offset] & AFTER_BRANCH) != 0;
    }


    /**
     * Returns whether the instruction at the given offset is the start of an
     * exception try block in the CodeAttribute that was visited most recently.
     */
    public boolean isExceptionStart(int offset)
    {
        return (instructionMarks[offset] & EXCEPTION_START) != 0;
    }


    /**
     * Returns whether the instruction at the given offset is the end of an
     * exception try block in the CodeAttribute that was visited most recently.
     */
    public boolean isExceptionEnd(int offset)
    {
        return (instructionMarks[offset] & EXCEPTION_END) != 0;
    }


    /**
     * Returns whether the instruction at the given offset is the start of an
     * exception handler in the CodeAttribute that was visited most recently.
     */
    public boolean isExceptionHandler(int offset)
    {
        return (instructionMarks[offset] & EXCEPTION_HANDLER) != 0;
    }


    /**
     * Returns whether the instruction at the given offset is a subroutine
     * invocation in the CodeAttribute that was visited most recently.
     */
    public boolean isSubroutineInvocation(int offset)
    {
        return (instructionMarks[offset] & SUBROUTINE_INVOCATION) != 0;
    }


    /**
     * Returns whether the instruction at the given offset is the start of a
     * subroutine in the CodeAttribute that was visited most recently.
     */
    public boolean isSubroutineStart(int offset)
    {
        return subroutineStarts[offset] == offset;
    }


    /**
     * Returns whether the instruction at the given offset is part of a
     * subroutine in the CodeAttribute that was visited most recently.
     */
    public boolean isSubroutine(int offset)
    {
        return subroutineStarts[offset] >= 0;
    }


    /**
     * Returns whether the subroutine at the given offset is ever returning
     * by means of a regular 'ret' instruction.
     */
    public boolean isSubroutineReturning(int offset)
    {
        return (instructionMarks[offset] & SUBROUTINE_RETURNING) != 0;
    }


    /**
     * Returns the start offset of the subroutine at the given offset, in the
     * CodeAttribute that was visited most recently.
     */
    public int subroutineStart(int offset)
    {
        return subroutineStarts[offset];
    }


    /**
     * Returns the offset after the subroutine at the given offset, in the
     * CodeAttribute that was visited most recently.
     */
    public int subroutineEnd(int offset)
    {
        return subroutineEnds[offset];
    }


//    /**
//     * Returns the instruction offset at which the object instance that is
//     * created at the given 'new' instruction offset is initialized, or
//     * <code>NONE</code> if it is not being created.
//     */
//    public int initializationOffset(int offset)
//    {
//        return initializationOffsets[offset];
//    }


//    /**
//     * Returns whether the method is an instance initializer, in the
//     * CodeAttribute that was visited most recently.
//     */
//    public boolean isInitializer()
//    {
//        return superInitializationOffset != NONE;
//    }


//    /**
//     * Returns the instruction offset at which this initializer is calling
//     * the "super" or "this" initializer method, or <code>NONE</code> if it is
//     * not an initializer.
//     */
//    public int superInitializationOffset()
//    {
//        return superInitializationOffset;
//    }


    /**
     * Returns whether the method contains subroutines, in the CodeAttribute
     * that was visited most recently.
     */
    public boolean containsSubroutines()
    {
        return containsSubroutines;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
//        DEBUG =
//            clazz.getName().equals("abc/Def") &&
//            method.getName(clazz).equals("abc");

        // Make sure there are sufficiently large arrays.
        int codeLength = codeAttribute.u4codeLength;
        if (subroutineStarts.length < codeLength)
        {
            // Create new arrays.
            instructionMarks      = new short[codeLength + 1];
            subroutineStarts      = new int[codeLength];
            subroutineEnds        = new int[codeLength];
//            initializationOffsets = new int[codeLength];

            // Reset the arrays.
            Arrays.fill(subroutineStarts,      0, codeLength, UNKNOWN);
            Arrays.fill(subroutineEnds,        0, codeLength, UNKNOWN);
//            Arrays.fill(initializationOffsets, 0, codeLength, NONE);
        }
        else
        {
            // Reset the arrays.
            Arrays.fill(instructionMarks,      0, codeLength, (short)0);
            Arrays.fill(subroutineStarts,      0, codeLength, UNKNOWN);
            Arrays.fill(subroutineEnds,        0, codeLength, UNKNOWN);
//            Arrays.fill(initializationOffsets, 0, codeLength, NONE);

            instructionMarks[codeLength] = 0;
        }

//        superInitializationOffset = NONE;
        containsSubroutines       = false;

        // Iterate until all subroutines have been fully marked.
        do
        {
            repeat                    = false;
            currentSubroutineStart    = NO_SUBROUTINE;

            // Mark branch targets by going over all instructions.
            codeAttribute.instructionsAccept(clazz, method, this);

            // Mark branch targets in the exception table.
            codeAttribute.exceptionsAccept(clazz, method, this);
        }
        while (repeat);

        // The end of the code is a branch target sentinel.
        instructionMarks[codeLength] = BRANCH_TARGET;

        if (containsSubroutines)
        {
            // Set the subroutine returning flag and the subroutine end at each
            // subroutine start.
            int previousSubroutineStart = NO_SUBROUTINE;


            for (int offset = 0; offset < codeLength; offset++)
            {
                if (isInstruction(offset))
                {
                    int subroutineStart = subroutineStarts[offset];

                    if (subroutineStart >= 0 &&
                        isSubroutineReturning(offset))
                    {
                        instructionMarks[subroutineStart] |= SUBROUTINE_RETURNING;
                    }

                    if (previousSubroutineStart >= 0)
                    {
                        subroutineEnds[previousSubroutineStart] = offset;
                    }

                    previousSubroutineStart = subroutineStart;
                }
            }

            if (previousSubroutineStart >= 0)
            {
                subroutineEnds[previousSubroutineStart] = codeLength;
            }

            // Set the subroutine returning flag and the subroutine end at each
            // subroutine instruction, based on the marks at the subroutine
            // start.
            for (int offset = 0; offset < codeLength; offset++)
            {
                if (isSubroutine(offset))
                {
                    int subroutineStart = subroutineStarts[offset];

                    if (isSubroutineReturning(subroutineStart))
                    {
                        instructionMarks[offset] |= SUBROUTINE_RETURNING;
                    }

                    subroutineEnds[offset] = subroutineEnds[subroutineStart];
                }
            }
        }

        if (DEBUG)
        {
            System.out.println();
            System.out.println("Branch targets: "+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz));

            for (int index = 0; index < codeLength; index++)
            {
                if (isInstruction(index))
                {
                    System.out.println("" +
                                       (isInitializer(index)          ? 'I' : '-') +
                                       (isBranchOrigin(index)         ? 'B' : '-') +
                                       (isAfterBranch(index)          ? 'b' : '-') +
                                       (isBranchTarget(index)         ? 'T' : '-') +
                                       (isExceptionStart(index)       ? 'E' : '-') +
                                       (isExceptionEnd(index)         ? 'e' : '-') +
                                       (isExceptionHandler(index)     ? 'H' : '-') +
                                       (isSubroutineInvocation(index) ? 'J' : '-') +
                                       (isSubroutineStart(index)      ? 'S' : '-') +
                                       (isSubroutineReturning(index)  ? 'r' : '-') +
                                       (isSubroutine(index)           ? " ["+subroutineStart(index)+" -> "+subroutineEnd(index)+"]" : "") +
                                       InstructionFactory.create(codeAttribute.code, index).toString(index));
                }
            }
        }
    }


    // Implementations for InstructionVisitor.

    public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
    {
        // Mark the instruction.
        instructionMarks[offset] |= INSTRUCTION;

        // Check if this is an instruction of a subroutine.
        checkSubroutine(offset);

        byte opcode = simpleInstruction.opcode;
        if (opcode == InstructionConstants.OP_IRETURN ||
            opcode == InstructionConstants.OP_LRETURN ||
            opcode == InstructionConstants.OP_FRETURN ||
            opcode == InstructionConstants.OP_DRETURN ||
            opcode == InstructionConstants.OP_ARETURN ||
            opcode == InstructionConstants.OP_RETURN  ||
            opcode == InstructionConstants.OP_ATHROW)
        {
            // Mark the branch origin.
            markBranchOrigin(offset);

            // Mark the next instruction.
            markAfterBranchOrigin(offset + simpleInstruction.length(offset));
        }
    }


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        // Mark the instruction.
        instructionMarks[offset] |= INSTRUCTION;

        // Check if this is an instruction of a subroutine.
        checkSubroutine(offset);

        byte opcode = constantInstruction.opcode;
        if (opcode == InstructionConstants.OP_NEW)
        {
            // Mark the creation.
            instructionMarks[offset] |= CREATION;
        }
        else if (opcode == InstructionConstants.OP_INVOKESPECIAL)
        {
            // Is it calling an instance initializer?
            isInitializer = false;
            clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
            if (isInitializer)
            {
                // Mark the initializer.
                instructionMarks[offset] |= INITIALIZER;
            }
        }
    }


    public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
    {
        // Mark the instruction.
        instructionMarks[offset] |= INSTRUCTION;

        // Check if this is an instruction of a subroutine.
        checkSubroutine(offset);

        if (variableInstruction.opcode == InstructionConstants.OP_RET)
        {
            // Mark the method.
            containsSubroutines = true;

            // Mark the branch origin.
            markBranchOrigin(offset);

            // Mark the subroutine return at its return instruction.
            instructionMarks[offset] |= SUBROUTINE_RETURNING;

            // Mark the next instruction.
            markAfterBranchOrigin(offset + variableInstruction.length(offset));
        }
    }


    public void visitBranchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction branchInstruction)
    {
        int branchOffset = branchInstruction.branchOffset;
        int targetOffset = offset + branchOffset;

        // Mark the branch origin.
        markBranchOrigin(offset);

        // Check if this is an instruction of a subroutine.
        checkSubroutine(offset);

        // Mark the branch target.
        markBranchTarget(offset, branchOffset);

        byte opcode = branchInstruction.opcode;
        if (opcode == InstructionConstants.OP_JSR ||
            opcode == InstructionConstants.OP_JSR_W)
        {
            // Mark the method.
            containsSubroutines = true;

            // Mark the subroutine invocation.
            instructionMarks[offset] |= SUBROUTINE_INVOCATION;

            // Mark the new subroutine start.
            markBranchSubroutineStart(offset, branchOffset, targetOffset);
        }
        else if (currentSubroutineStart != UNKNOWN)
        {
            // Mark the continued subroutine start.
            markBranchSubroutineStart(offset, branchOffset, currentSubroutineStart);
        }

        if (opcode == InstructionConstants.OP_GOTO ||
            opcode == InstructionConstants.OP_GOTO_W)
        {
            // Mark the next instruction.
            markAfterBranchOrigin(offset + branchInstruction.length(offset));
        }
    }


    public void visitAnySwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SwitchInstruction switchInstruction)
    {
        // Mark the branch origin.
        markBranchOrigin(offset);

        // Check if this is an instruction of a subroutine.
        checkSubroutine(offset);

        // Mark the branch targets of the default jump offset.
        markBranch(offset, switchInstruction.defaultOffset);

        // Mark the branch targets of the jump offsets.
        markBranches(offset, switchInstruction.jumpOffsets);

        // Mark the next instruction.
        markAfterBranchOrigin(offset + switchInstruction.length(offset));
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitMethodrefConstant(Clazz clazz, MethodrefConstant methodrefConstant)
    {
        // Remember whether the method is an initializer.
        isInitializer = methodrefConstant.getName(clazz).equals(ClassConstants.METHOD_NAME_INIT);
    }


    // Implementations for ExceptionInfoVisitor.

    public void visitExceptionInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, ExceptionInfo exceptionInfo)
    {
        int startPC   = exceptionInfo.u2startPC;
        int endPC     = exceptionInfo.u2endPC;
        int handlerPC = exceptionInfo.u2handlerPC;

        // Mark the exception offsets.
        instructionMarks[startPC]   |= EXCEPTION_START;
        instructionMarks[endPC]     |= EXCEPTION_END;
        instructionMarks[handlerPC] |= EXCEPTION_HANDLER;

        // Mark the handler as part of a subroutine if necessary.
        if (subroutineStarts[handlerPC] == UNKNOWN &&
            subroutineStarts[startPC]   != UNKNOWN)
        {
            subroutineStarts[handlerPC] = subroutineStarts[startPC];

            // We'll have to go over all instructions again.
            repeat = true;
        }
    }


    // Small utility methods.

    /**
     * Marks the branch targets and their subroutine starts at the given
     * offsets.
     */
    private void markBranches(int offset, int[] jumpOffsets)
    {
        for (int index = 0; index < jumpOffsets.length; index++)
        {
            markBranch(offset, jumpOffsets[index]);
        }
    }


    /**
     * Marks the branch target and its subroutine start at the given offset.
     */
    private void markBranch(int offset, int jumpOffset)
    {
        markBranchTarget(offset, jumpOffset);

        if (currentSubroutineStart != UNKNOWN)
        {
            markBranchSubroutineStart(offset, jumpOffset, currentSubroutineStart);
        }
    }

    /**
     * Marks the branch origin at the given offset.
     */
    private void markBranchOrigin(int offset)
    {
        instructionMarks[offset] |= INSTRUCTION | BRANCH_ORIGIN;
    }


    /**
     * Marks the branch target at the given offset.
     */
    private void markBranchTarget(int offset, int jumpOffset)
    {
        int targetOffset = offset + jumpOffset;

        instructionMarks[targetOffset] |= BRANCH_TARGET;
    }


    /**
     * Marks the subroutine start at the given offset, if applicable.
     */
    private void markBranchSubroutineStart(int offset,
                                           int jumpOffset,
                                           int subroutineStart)
    {
        int targetOffset = offset + jumpOffset;

        // Are we marking a subroutine and branching to an offset that hasn't
        // been marked yet?
        if (subroutineStarts[targetOffset] == UNKNOWN)
        {
            // Is it a backward branch?
            if (jumpOffset < 0)
            {
                // Remember the smallest subroutine start.
                if (subroutineStart > targetOffset)
                {
                    subroutineStart = targetOffset;
                }

                // We'll have to go over all instructions again.
                repeat = true;
            }

            // Mark the subroutine start of the target.
            subroutineStarts[targetOffset] = subroutineStart;
        }
    }


    /**
     * Marks the instruction at the given offset, after a branch.
     */
    private void markAfterBranchOrigin(int nextOffset)
    {
        instructionMarks[nextOffset] |= AFTER_BRANCH;

        // Stop marking a subroutine.
        currentSubroutineStart = UNKNOWN;
    }


    /**
     * Checks if the specified instruction is inside a subroutine.
     */
    private void checkSubroutine(int offset)
    {
        // Are we inside a previously marked subroutine?
        if (subroutineStarts[offset] != UNKNOWN)
        {
            // Start marking a subroutine.
            currentSubroutineStart = subroutineStarts[offset];
        }

        // Are we marking a subroutine?
        else if (currentSubroutineStart != UNKNOWN)
        {
            // Mark the subroutine start.
            subroutineStarts[offset] = currentSubroutineStart;
        }
    }
}

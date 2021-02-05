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
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.editor.CodeAttributeEditor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This AttributeVisitor redirects unconditional branches so any common code
 * is shared, and the code preceding the branch can be removed, in the code
 * attributes that it visits.
 *
 * @author Eric Lafortune
 */
public class GotoCommonCodeReplacer
extends      SimplifiedVisitor
implements   AttributeVisitor,
             InstructionVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = true;
    //*/


    private final InstructionVisitor  extraInstructionVisitor;

    private final BranchTargetFinder  branchTargetFinder  = new BranchTargetFinder();
    private final CodeAttributeEditor codeAttributeEditor = new CodeAttributeEditor();


    /**
     * Creates a new GotoCommonCodeReplacer.
     * @param extraInstructionVisitor an optional extra visitor for all replaced
     *                                goto instructions.
     */
    public GotoCommonCodeReplacer(InstructionVisitor  extraInstructionVisitor)
    {
        this.extraInstructionVisitor = extraInstructionVisitor;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
//        DEBUG =
//            clazz.getName().equals("abc/Def") &&
//            method.getName(clazz).equals("abc");

        // Mark all branch targets.
        branchTargetFinder.visitCodeAttribute(clazz, method, codeAttribute);

        // Reset the code attribute editor.
        codeAttributeEditor.reset(codeAttribute.u4codeLength);

        // Remap the variables of the instructions.
        codeAttribute.instructionsAccept(clazz, method, this);

        // Apply the code atribute editor.
        codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitBranchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction branchInstruction)
    {
        // Check if the instruction is an unconditional goto instruction that
        // isn't the target of a branch itself.
        byte opcode = branchInstruction.opcode;
        if ((opcode == InstructionConstants.OP_GOTO ||
             opcode == InstructionConstants.OP_GOTO_W) &&
            !branchTargetFinder.isBranchTarget(offset))
        {
            int branchOffset = branchInstruction.branchOffset;
            int targetOffset = offset + branchOffset;

            // Get the number of common bytes.
            int commonCount = commonByteCodeCount(codeAttribute, offset, targetOffset);

            if (commonCount > 0 &&
                !exceptionBoundary(codeAttribute, offset, targetOffset))
            {
                if (DEBUG)
                {
                    System.out.println("GotoCommonCodeReplacer: "+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+" (["+(offset-commonCount)+"] - "+branchInstruction.toString(offset)+" -> "+targetOffset+")");
                }

                // Delete the common instructions.
                for (int delta = 0; delta <= commonCount; delta++)
                {
                    int deleteOffset = offset - delta;
                    if (branchTargetFinder.isInstruction(deleteOffset))
                    {
                        codeAttributeEditor.clearModifications(deleteOffset);
                        codeAttributeEditor.deleteInstruction(deleteOffset);
                    }
                }

                // Redirect the goto instruction, if it is still necessary.
                int newBranchOffset = branchOffset - commonCount;
                if (newBranchOffset != branchInstruction.length(offset))
                {
                    Instruction newGotoInstruction =
                         new BranchInstruction(opcode, newBranchOffset);
                    codeAttributeEditor.replaceInstruction(offset,
                                                           newGotoInstruction);
                }

                // Visit the instruction, if required.
                if (extraInstructionVisitor != null)
                {
                    extraInstructionVisitor.visitBranchInstruction(clazz, method, codeAttribute, offset, branchInstruction);
                }
            }
        }
    }


    // Small utility methods.

    /**
     * Returns the number of common bytes preceding the given offsets,
     * avoiding branches and exception blocks.
     */
    private int commonByteCodeCount(CodeAttribute codeAttribute, int offset1, int offset2)
    {
        // Find the block of common instructions preceding it.
        byte[] code = codeAttribute.code;

        int successfulDelta = 0;

        for (int delta = 1;
             delta <= offset1 &&
             delta <= offset2 &&
             offset2 - delta != offset1;
             delta++)
        {
            int newOffset1 = offset1 - delta;
            int newOffset2 = offset2 - delta;

            // Is the code identical at both offsets?
            if (code[newOffset1] != code[newOffset2])
            {
                break;
            }

            // Are there instructions at either offset but not both?
            if (branchTargetFinder.isInstruction(newOffset1) ^
                branchTargetFinder.isInstruction(newOffset2))
            {
                break;
            }

            // Are there instructions at both offsets?
            if (branchTargetFinder.isInstruction(newOffset1) &&
                branchTargetFinder.isInstruction(newOffset2))
            {
                // Are the offsets involved in some branches?
                // Note that the preverifier doesn't like initializer
                // invocations to be moved around.
                // Also note that the preverifier doesn't like pop instructions
                // that work on different operands.
                if (branchTargetFinder.isBranchOrigin(newOffset1)   ||
                    branchTargetFinder.isBranchTarget(newOffset1)   ||
                    branchTargetFinder.isExceptionStart(newOffset1) ||
                    branchTargetFinder.isExceptionEnd(newOffset1)   ||
                    branchTargetFinder.isInitializer(newOffset1)    ||
                    branchTargetFinder.isExceptionStart(newOffset2) ||
                    branchTargetFinder.isExceptionEnd(newOffset2)   ||
                    isPop(code[newOffset1]))
                {
                    break;
                }

                // Make sure the new branch target was a branch target before,
                // in order not to introduce new entries in the stack map table.
                if (branchTargetFinder.isBranchTarget(newOffset2))
                {
                    successfulDelta = delta;
                }

                if (branchTargetFinder.isBranchTarget(newOffset1))
                {
                    break;
                }
            }
        }

        return successfulDelta;
    }


    /**
     * Returns whether the given opcode represents a pop instruction that must
     * get a consistent type (pop, pop2, arraylength).
     */
    private boolean isPop(byte opcode)
    {
        return opcode == InstructionConstants.OP_POP  ||
               opcode == InstructionConstants.OP_POP2 ||
               opcode == InstructionConstants.OP_ARRAYLENGTH;
    }


    /**
     * Returns the whether there is a boundary of an exception block between
     * the given offsets (including both).
     */
    private boolean exceptionBoundary(CodeAttribute codeAttribute, int offset1, int offset2)
    {
        // Swap the offsets if the second one is smaller than the first one.
        if (offset2 < offset1)
        {
            int offset = offset1;
            offset1 = offset2;
            offset2 = offset;
        }

        // Check if there is a boundary of an exception block.
        for (int offset = offset1; offset <= offset2; offset++)
        {
            if (branchTargetFinder.isExceptionStart(offset) ||
                branchTargetFinder.isExceptionEnd(offset))
            {
                return true;
            }
        }

        return false;
    }
}

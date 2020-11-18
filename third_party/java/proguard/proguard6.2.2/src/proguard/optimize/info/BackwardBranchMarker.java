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
package proguard.optimize.info;

import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This InstructionVisitor marks all methods that branch backward in any of the
 * instructions that it visits.
 *
 * @author Eric Lafortune
 */
public class BackwardBranchMarker
extends      SimplifiedVisitor
implements   InstructionVisitor
{
    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitBranchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction branchInstruction)
    {
        markBackwardBranch(method, branchInstruction.branchOffset);
    }


    public void visitAnySwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SwitchInstruction switchInstruction)
    {
        markBackwardBranch(method, switchInstruction.defaultOffset);

        for (int index = 0; index < switchInstruction.jumpOffsets.length; index++)
        {
            markBackwardBranch(method, switchInstruction.jumpOffsets[index]);
        }
    }


    // Small utility methods.

    /**
     * Marks the given method if the given branch offset is negative.
     */
    private void markBackwardBranch(Method method, int branchOffset)
    {
        if (branchOffset < 0)
        {
            setBranchesBackward(method);
        }
    }


    private static void setBranchesBackward(Method method)
    {
        ProgramMethodOptimizationInfo.getProgramMethodOptimizationInfo(method).setBranchesBackward();
    }


    public static boolean branchesBackward(Method method)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).branchesBackward();
    }
}

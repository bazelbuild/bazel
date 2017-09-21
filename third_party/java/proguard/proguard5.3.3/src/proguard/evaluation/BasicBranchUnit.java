/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2017 Eric Lafortune @ GuardSquare
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
package proguard.evaluation;

import proguard.classfile.Clazz;
import proguard.classfile.attribute.CodeAttribute;
import proguard.evaluation.value.InstructionOffsetValue;

/**
 * This BranchUnit remembers the branch unit commands that are invoked on it.
 * I doesn't consider conditions when branching.
 *
 * @author Eric Lafortune
 */
public class BasicBranchUnit
implements   BranchUnit
{
    private boolean                wasCalled;
    private InstructionOffsetValue traceBranchTargets;


    /**
     * Resets the flag that tells whether any of the branch unit commands was
     * called.
     */
    public void resetCalled()
    {
        wasCalled = false;
    }

    /**
     * Sets the flag that tells whether any of the branch unit commands was
     * called.
     */
    protected void setCalled()
    {
        wasCalled = true;
    }

    /**
     * Returns whether any of the branch unit commands was called.
     */
    public boolean wasCalled()
    {
        return wasCalled;
    }


    /**
     * Sets the initial branch targets, which will be updated as the branch
     * methods of the branch unit are called.
     */
    public void setTraceBranchTargets(InstructionOffsetValue branchTargets)
    {
        this.traceBranchTargets = branchTargets;
    }

    public InstructionOffsetValue getTraceBranchTargets()
    {
        return traceBranchTargets;
    }


    // Implementations for BranchUnit.

    public void branch(Clazz         clazz,
                       CodeAttribute codeAttribute,
                       int           offset,
                       int           branchTarget)
    {
        // Override the branch targets.
        traceBranchTargets = new InstructionOffsetValue(branchTarget);

        wasCalled = true;
    }


    public void branchConditionally(Clazz         clazz,
                                    CodeAttribute codeAttribute,
                                    int           offset,
                                    int           branchTarget,
                                    int           conditional)
    {
        // Accumulate the branch targets.
        traceBranchTargets =
            traceBranchTargets.generalize(new InstructionOffsetValue(branchTarget)).instructionOffsetValue();

        wasCalled = true;
    }


    public void returnFromMethod()
    {
        // Stop processing this block.
        traceBranchTargets = InstructionOffsetValue.EMPTY_VALUE;

        wasCalled = true;
    }


    public void throwException()
    {
        // Stop processing this block.
        traceBranchTargets = InstructionOffsetValue.EMPTY_VALUE;

        wasCalled = true;
    }
}

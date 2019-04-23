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
package proguard.evaluation;

import proguard.classfile.Clazz;
import proguard.classfile.attribute.CodeAttribute;
import proguard.evaluation.value.InstructionOffsetValue;

/**
 * This BranchUnit remembers the branch unit commands that are invoked on it.
 * It doesn't consider conditions when branching.
 *
 * @author Eric Lafortune
 */
public class BasicBranchUnit
implements   BranchUnit
{
    protected InstructionOffsetValue traceBranchTargets;
    protected boolean                wasCalled;


    /**
     * Resets the accumulated branch targets and the flag that tells whether
     * any of the branch unit methods was called.
     */
    public void reset()
    {
        traceBranchTargets = InstructionOffsetValue.EMPTY_VALUE;

        wasCalled = false;
    }

    /**
     * Returns whether any of the branch unit methods was called.
     */
    public boolean wasCalled()
    {
        return wasCalled;
    }


    /**
     * Returns the accumulated branch targets that were passed to the branch
     * unit methods.
     */
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
            traceBranchTargets.add(branchTarget);

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

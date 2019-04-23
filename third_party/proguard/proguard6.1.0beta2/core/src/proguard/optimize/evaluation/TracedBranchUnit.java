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
package proguard.optimize.evaluation;

import proguard.classfile.Clazz;
import proguard.classfile.attribute.CodeAttribute;
import proguard.evaluation.BasicBranchUnit;
import proguard.evaluation.value.Value;

/**
 * This BranchUnit remembers the branch unit commands that are invoked on it.
 *
 * @author Eric Lafortune
 */
class   TracedBranchUnit
extends BasicBranchUnit
{
    private boolean isFixed;


    // Implementations for BasicBranchUnit.

    public void reset()
    {
        super.reset();

        isFixed = false;
    }


    // Implementations for BranchUnit.

    public void branch(Clazz         clazz,
                       CodeAttribute codeAttribute,
                       int           offset,
                       int           branchTarget)
    {
        super.branch(clazz, codeAttribute, offset, branchTarget);

        isFixed = true;
    }


    public void branchConditionally(Clazz         clazz,
                                    CodeAttribute codeAttribute,
                                    int           offset,
                                    int           branchTarget,
                                    int           conditional)
    {
        if      (conditional == Value.ALWAYS)
        {
            // Always branch.
            super.branch(clazz, codeAttribute, offset, branchTarget);

            isFixed = true;
        }
        else if (conditional == Value.MAYBE)
        {
            if (!isFixed)
            {
                // Maybe branch.
                super.branchConditionally(clazz, codeAttribute, offset, branchTarget, conditional);
            }
        }
        else
        {
            super.wasCalled = true;
        }
    }
}

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
package proguard.optimize;

import proguard.classfile.*;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.MemberVisitor;
import proguard.evaluation.value.Value;
import proguard.optimize.evaluation.StoringInvocationUnit;

/**
 * This <code>MemberVisitor</code> delegates its visits to program class members
 * to another given <code>MemberVisitor</code>, but only when the visited
 * class member has been marked as a constant.
 *
 * @see StoringInvocationUnit
 * @author Eric Lafortune
 */
public class ConstantMemberFilter
extends      SimplifiedVisitor
implements   MemberVisitor
{
    private final MemberVisitor constantMemberVisitor;


    /**
     * Creates a new ConstantMemberFilter.
     * @param constantMemberVisitor the <code>MemberVisitor</code> to which
     *                              visits to constant members will be delegated.
     */
    public ConstantMemberFilter(MemberVisitor constantMemberVisitor)
    {
        this.constantMemberVisitor = constantMemberVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        Value value = StoringInvocationUnit.getFieldValue(programField);
        if (value != null &&
            value.isParticular())
        {
            constantMemberVisitor.visitProgramField(programClass, programField);
        }
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        Value value = StoringInvocationUnit.getMethodReturnValue(programMethod);
        if (value != null &&
            value.isParticular())
        {
            constantMemberVisitor.visitProgramMethod(programClass, programMethod);
        }
    }
}

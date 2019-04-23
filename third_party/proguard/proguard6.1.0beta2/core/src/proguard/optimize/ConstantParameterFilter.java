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
import proguard.classfile.util.*;
import proguard.classfile.visitor.MemberVisitor;
import proguard.evaluation.value.Value;
import proguard.optimize.evaluation.StoringInvocationUnit;

/**
 * This <code>MemberVisitor</code> delegates its visits to program methods
 * to another given <code>MemberVisitor</code>, for each method parameter
 * that has been marked as constant.
 *
 * @see StoringInvocationUnit
 * @author Eric Lafortune
 */
public class ConstantParameterFilter
extends      SimplifiedVisitor
implements   MemberVisitor
{
    private final MemberVisitor constantParameterVisitor;


    /**
     * Creates a new ConstantParameterFilter.
     * @param constantParameterVisitor the <code>MemberVisitor</code> to which
     *                                 visits will be delegated.
     */
    public ConstantParameterFilter(MemberVisitor constantParameterVisitor)
    {
        this.constantParameterVisitor = constantParameterVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        // All parameters of non-static methods are shifted by one in the local
        // variable frame.
        boolean isStatic =
            (programMethod.getAccessFlags() & ClassConstants.ACC_STATIC) != 0;

        int parameterStart = isStatic ? 0 : 1;
        int parameterCount =
            ClassUtil.internalMethodParameterCount(programMethod.getDescriptor(programClass),
                                                   isStatic);

        for (int index = parameterStart; index < parameterCount; index++)
        {
            Value value = StoringInvocationUnit.getMethodParameterValue(programMethod, index);
            if (value != null &&
                value.isParticular())
            {
                constantParameterVisitor.visitProgramMethod(programClass, programMethod);
            }
        }
    }
}
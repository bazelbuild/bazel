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
import proguard.classfile.attribute.BootstrapMethodInfo;
import proguard.classfile.attribute.visitor.BootstrapMethodInfoVisitor;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.MemberVisitor;
import proguard.optimize.info.*;
import proguard.optimize.peephole.VariableShrinker;

/**
 * This BootstrapMethodInfoVisitor removes unused constant arguments from
 * bootstrap method entries that it visits.
 *
 * @see ParameterUsageMarker
 * @see VariableUsageMarker
 * @see VariableShrinker
 * @author Eric Lafortune
 */
public class BootstrapMethodArgumentShrinker
extends      SimplifiedVisitor
implements   BootstrapMethodInfoVisitor,
             ConstantVisitor,
             MemberVisitor
{
    private long usedParameters;


    // Implementations for BootstrapMethodInfoVisitor.

    public void visitBootstrapMethodInfo(Clazz clazz, BootstrapMethodInfo bootstrapMethodInfo)
    {
        // Check which method parameters are used.
        usedParameters = -1L;
        clazz.constantPoolEntryAccept(bootstrapMethodInfo.u2methodHandleIndex, this);

        // Remove the unused arguments.
        int   methodArgumentCount = bootstrapMethodInfo.u2methodArgumentCount;
        int[] methodArguments     = bootstrapMethodInfo.u2methodArguments;

        int newArgumentIndex = 0;

        for (int argumentIndex = 0; argumentIndex < methodArgumentCount; argumentIndex++)
        {
            if (argumentIndex >= 64 ||
                (usedParameters & (1L << argumentIndex)) != 0L)
            {
                methodArguments[newArgumentIndex++] = methodArguments[argumentIndex];
            }
        }

        // Update the number of arguments.
        bootstrapMethodInfo.u2methodArgumentCount = newArgumentIndex;
    }


    // Implementations for ConstantVisitor.

    public void visitMethodHandleConstant(Clazz clazz, MethodHandleConstant methodHandleConstant)
    {
        // Check the referenced bootstrap method.
        clazz.constantPoolEntryAccept(methodHandleConstant.u2referenceIndex, this);
    }


    public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
    {
        // Check the referenced class member itself.
        refConstant.referencedMemberAccept(this);
    }


    // Implementations for MemberVisitor.

    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod) {}


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        usedParameters = ParameterUsageMarker.getUsedParameters(programMethod);
    }
}

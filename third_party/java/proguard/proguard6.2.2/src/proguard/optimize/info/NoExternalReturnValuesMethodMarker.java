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
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.MemberVisitor;

/**
 * This MemberVisitor marks all methods that it visits as not having any
 * return values that are external reference values (only parameters or new
 * instances). It will make the ParameterEscapeMarker consider them as
 * such without further analysis.
 *
 * @see ParameterEscapeMarker
 * @author Eric Lafortune
 */
public class NoExternalReturnValuesMethodMarker
extends      SimplifiedVisitor
implements   MemberVisitor
{
    // Implementations for MemberVisitor.

    public void visitAnyMember(Clazz Clazz, Member member)
    {
        // Ignore any attempts to mark fields.
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        markNoExternalReturnValues(programMethod);
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        markNoExternalReturnValues(libraryMethod);
    }


    // Small utility methods.

    private static void markNoExternalReturnValues(Method method)
    {
        MethodOptimizationInfo.getMethodOptimizationInfo(method).setNoExternalReturnValues();
    }


    public static boolean hasNoExternalReturnValues(Method method)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).hasNoExternalReturnValues();
    }
}

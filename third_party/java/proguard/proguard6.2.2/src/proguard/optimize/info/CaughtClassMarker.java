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
import proguard.classfile.visitor.ClassVisitor;

/**
 * This ClassVisitor marks all program classes that it visits as caught.
 * This means that these classes are exception classes that occur in exception
 * handlers.
 *
 * @author Eric Lafortune
 */
public class CaughtClassMarker
implements   ClassVisitor
{
    // Implementations for ClassVisitor.

    public void visitLibraryClass(LibraryClass libraryClass) {}

    public void visitProgramClass(ProgramClass programClass)
    {
        setCaught(programClass);
    }


    // Small utility methods.

    private static void setCaught(Clazz clazz)
    {
        ProgramClassOptimizationInfo.getProgramClassOptimizationInfo(clazz).setCaught();
    }


    public static boolean isCaught(Clazz clazz)
    {
        return ClassOptimizationInfo.getClassOptimizationInfo(clazz).isCaught();
    }
}
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
 * This ClassVisitor marks all program classes that it visits with a given
 * flag for simple enums.
 *
 * @author Eric Lafortune
 */
public class SimpleEnumMarker
implements   ClassVisitor
{
    private final boolean simple;


    /**
     * Creates a new SimpleEnumMarker that marks visited classes with the
     * given flag.
     */
    public SimpleEnumMarker(boolean simple)
    {
        this.simple = simple;
    }


    // Implementations for ClassVisitor.

    public void visitLibraryClass(LibraryClass libraryClass) {}

    public void visitProgramClass(ProgramClass programClass)
    {
        setSimpleEnum(programClass);
    }


    // Small utility methods.

    private void setSimpleEnum(Clazz clazz)
    {
        ProgramClassOptimizationInfo.getProgramClassOptimizationInfo(clazz).setSimpleEnum(simple);
    }


    public static boolean isSimpleEnum(Clazz clazz)
    {
        return ClassOptimizationInfo.getClassOptimizationInfo(clazz).isSimpleEnum();
    }
}
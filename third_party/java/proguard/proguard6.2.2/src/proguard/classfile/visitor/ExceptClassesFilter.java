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
package proguard.classfile.visitor;

import proguard.classfile.*;

/**
 * This <code>ClassVisitor</code> delegates its visits to another given
 * <code>ClassVisitor</code>, except for classes are in a given list.
 *
 * @author Eric Lafortune
 */
public class ExceptClassesFilter implements ClassVisitor
{
    private final Clazz[]      exceptClasses;
    private final ClassVisitor classVisitor;


    /**
     * Creates a new ExceptClassesFilter.
     * @param exceptClasses the classes that will not be visited.
     * @param classVisitor  the <code>ClassVisitor</code> to which visits will
     *                      be delegated.
     */
    public ExceptClassesFilter(Clazz[]      exceptClasses,
                               ClassVisitor classVisitor)
    {
        this.exceptClasses = exceptClasses;
        this.classVisitor  = classVisitor;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        if (!present(programClass))
        {
            classVisitor.visitProgramClass(programClass);
        }
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        if (!present(libraryClass))
        {
            classVisitor.visitLibraryClass(libraryClass);
        }
    }


    // Small utility methods.

    private boolean present(Clazz clazz)
    {
        if (exceptClasses == null)
        {
            return false;
        }

        for (int index = 0; index < exceptClasses.length; index++)
        {
            if (exceptClasses[index].equals(clazz))
            {
                return true;
            }
        }

        return false;
    }
}
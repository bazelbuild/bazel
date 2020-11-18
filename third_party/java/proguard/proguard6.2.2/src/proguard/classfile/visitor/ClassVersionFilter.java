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
 * This <code>ClassVisitor</code> delegates its visits to program classes to
 * another given <code>ClassVisitor</code>, but only when the class version
 * number of the visited program class lies in a given range.
 *
 * @author Eric Lafortune
 */
public class ClassVersionFilter implements ClassVisitor
{
    private final int          minimumClassVersion;
    private final int          maximumClassVersion;
    private final ClassVisitor classVisitor;


    /**
     * Creates a new ClassVersionFilter.
     * @param minimumClassVersion the minimum class version number.
     * @param classVisitor        the <code>ClassVisitor</code> to which visits
     *                            will be delegated.
     */
    public ClassVersionFilter(int          minimumClassVersion,
                              ClassVisitor classVisitor)
    {
        this(minimumClassVersion, Integer.MAX_VALUE, classVisitor);
    }


    /**
     * Creates a new ClassVersionFilter.
     * @param minimumClassVersion the minimum class version number.
     * @param maximumClassVersion the maximum class version number.
     * @param classVisitor        the <code>ClassVisitor</code> to which visits
     *                            will be delegated.
     */
    public ClassVersionFilter(int          minimumClassVersion,
                              int          maximumClassVersion,
                              ClassVisitor classVisitor)
    {
        this.minimumClassVersion = minimumClassVersion;
        this.maximumClassVersion = maximumClassVersion;
        this.classVisitor        = classVisitor;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        if (programClass.u4version >= minimumClassVersion &&
            programClass.u4version <= maximumClassVersion)
        {
            classVisitor.visitProgramClass(programClass);
        }
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        // Library classes don't have version numbers.
    }
}

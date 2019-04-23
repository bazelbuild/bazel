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
import proguard.classfile.visitor.ClassVisitor;

/**
 * This ClassVisitor delegates its visits to one of two ClassVisitor's,
 * depending on whether the visited class is kept or not.
 *
 * @see KeepMarker
 *
 * @author Eric Lafortune
 */
public class KeptClassFilter
implements   ClassVisitor
{
    private final ClassVisitor acceptedVisitor;
    private final ClassVisitor rejectedVisitor;


    /**
     * Creates a new KeptClassFilter.
     *
     * @param acceptedVisitor the class visitor to which accepted (kept)
     *                        classes will be delegated.
     */
    public KeptClassFilter(ClassVisitor acceptedVisitor)
    {
        this(acceptedVisitor, null);
    }

    /**
     * Creates a new KeptClassFilter.
     *
     * @param acceptedVisitor the class visitor to which accepted (kept)
     *                        classes will be delegated.
     * @param rejectedVisitor the class visitor to which rejected (unkept)
     *                        classes will be delegated.
     */
    public KeptClassFilter(ClassVisitor acceptedVisitor,
                           ClassVisitor rejectedVisitor)
    {
        this.acceptedVisitor = acceptedVisitor;
        this.rejectedVisitor = rejectedVisitor;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        ClassVisitor delegateVisitor = selectVisitor(programClass);
        if (delegateVisitor != null) {
            delegateVisitor.visitProgramClass(programClass);
        }
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        ClassVisitor delegateVisitor = selectVisitor(libraryClass);
        if (delegateVisitor != null) {
            delegateVisitor.visitLibraryClass(libraryClass);
        }
    }


    // Small utility methods.

    private ClassVisitor selectVisitor(Clazz clazz)
    {
        return KeepMarker.isKept(clazz) ? acceptedVisitor : rejectedVisitor;
    }
}
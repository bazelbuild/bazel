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
package proguard.optimize.peephole;

import proguard.classfile.*;
import proguard.classfile.visitor.ClassVisitor;

/**
 * This ClassVisitor delegates its visits to one of two other given
 * ClassVisitor instances, depending on whether the classes are marked to be
 * retargeted or not.
 *
 * @see ClassMerger
 *
 * @author Eric Lafortune
 */
public class RetargetedClassFilter
implements   ClassVisitor
{
    private final ClassVisitor retargetedClassVisitor;
    private final ClassVisitor otherClassVisitor;


    /**
     * Creates a new RetargetedClassFilter.
     *
     * @param retargetedClassVisitor the class visitor to which visits to
     *                               classes that are marked to be retargeted
     *                               will be delegated.
     */
    public RetargetedClassFilter(ClassVisitor retargetedClassVisitor)
    {
        this(retargetedClassVisitor, null);
    }


    /**
     * Creates a new RetargetedClassFilter.
     *
     * @param retargetedClassVisitor the class visitor to which visits to
     *                               classes that are marked to be retargeted
     *                               will be delegated.
     * @param otherClassVisitor      the class visitor to which visits to
     *                               classes that are not marked to be
     *                               retargeted will be delegated.
     */
    public RetargetedClassFilter(ClassVisitor retargetedClassVisitor,
                                 ClassVisitor otherClassVisitor)
    {
        this.retargetedClassVisitor = retargetedClassVisitor;
        this.otherClassVisitor      = otherClassVisitor;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Is the class marked to be retargeted?
        ClassVisitor classVisitor = ClassMerger.getTargetClass(programClass) != null ?
            retargetedClassVisitor : otherClassVisitor;

        if (classVisitor != null)
        {
            classVisitor.visitProgramClass(programClass);
        }
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        // A library class can't be retargeted.
        if (otherClassVisitor != null)
        {
            otherClassVisitor.visitLibraryClass(libraryClass);
        }
    }
}

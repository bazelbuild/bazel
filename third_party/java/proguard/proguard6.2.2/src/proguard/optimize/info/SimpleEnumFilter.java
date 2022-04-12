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
 * This ClassVisitor delegates its visits to one of two other given
 * ClassVisitor instances, depending on whether the classes are marked
 * as simple enums or not.
 *
 * @see SimpleEnumMarker
 *
 * @author Eric Lafortune
 */
public class SimpleEnumFilter
implements   ClassVisitor
{
    private final ClassVisitor simpleEnumClassVisitor;
    private final ClassVisitor otherClassVisitor;


    /**
     * Creates a new SimpleEnumClassFilter.
     *
     * @param simpleEnumClassVisitor the class visitor to which visits to
     *                               classes that are marked to be simpleEnum
     *                               will be delegated.
     */
    public SimpleEnumFilter(ClassVisitor simpleEnumClassVisitor)
    {
        this(simpleEnumClassVisitor, null);
    }


    /**
     * Creates a new SimpleEnumClassFilter.
     *
     * @param simpleEnumClassVisitor the class visitor to which visits to
     *                               classes that are marked as simple enums
     *                               will be delegated.
     * @param otherClassVisitor      the class visitor to which visits to
     *                               classes that are not marked as simple
     *                               enums will be delegated.
     */
    public SimpleEnumFilter(ClassVisitor simpleEnumClassVisitor,
                            ClassVisitor otherClassVisitor)
    {
        this.simpleEnumClassVisitor = simpleEnumClassVisitor;
        this.otherClassVisitor      = otherClassVisitor;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Is the class marked as a simple enum?
        ClassVisitor classVisitor = SimpleEnumMarker.isSimpleEnum(programClass) ?
            simpleEnumClassVisitor : otherClassVisitor;

        if (classVisitor != null)
        {
            classVisitor.visitProgramClass(programClass);
        }
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        // A library class can't be marked as a simple enum.
        if (otherClassVisitor != null)
        {
            otherClassVisitor.visitLibraryClass(libraryClass);
        }
    }
}
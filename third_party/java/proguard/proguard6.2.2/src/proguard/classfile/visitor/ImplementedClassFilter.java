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
 * This <code>ClassVisitor</code> delegates its visits to one of two given
 * <code>ClassVisitor</code>s, depending on whether the visited classes
 * extend/implement a given class or not.
 *
 * Filter:
 * - accepted: the visited class extends/implements the given class.
 * - rejected: the visited class does not extend/implement the given class.
 *
 * @author Eric Lafortune
 */
public class ImplementedClassFilter implements ClassVisitor
{
    private final Clazz        implementedClass;
    private final boolean      includeImplementedClass;
    private final ClassVisitor acceptedClassVisitor;
    private final ClassVisitor rejectedClassVisitor;


    /**
     * Creates a new ImplementedClassFilter.
     *
     * @param implementedClass        the class whose implementations will
     *                                be accepted.
     * @param includeImplementedClass if true, the implemented class itself
     *                                will also be accepted, otherwise it
     *                                will be rejected.
     * @param acceptedClassVisitor    the <code>ClassVisitor</code> to which
     *                                visits of classes implementing the given
     *                                class will be delegated.
     * @param rejectedClassVisistor   the <code>ClassVisitor</code> to which
     *                                visits of classes not implementing the
     *                                given class will be delegated.
     */
    public ImplementedClassFilter(Clazz        implementedClass,
                                  boolean      includeImplementedClass,
                                  ClassVisitor acceptedClassVisitor,
                                  ClassVisitor rejectedClassVisistor)
    {
        this.implementedClass = implementedClass;
        this.includeImplementedClass = includeImplementedClass;
        this.acceptedClassVisitor = acceptedClassVisitor;
        this.rejectedClassVisitor = rejectedClassVisistor;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        ClassVisitor visitor = delegateVisitor(programClass);
        if (visitor != null)
        {
            visitor.visitProgramClass(programClass);
        }
    }

    public void visitLibraryClass(LibraryClass libraryClass)
    {
        ClassVisitor visitor = delegateVisitor(libraryClass);
        if (visitor != null)
        {
            visitor.visitLibraryClass(libraryClass);
        }
    }


    // Small utility methods.

    private ClassVisitor delegateVisitor(Clazz clazz)
    {
        return clazz.extendsOrImplements(implementedClass) &&
               (clazz != implementedClass || includeImplementedClass) ?
            acceptedClassVisitor : rejectedClassVisitor;
    }
}

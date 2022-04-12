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
 * This <code>ClassVisitor</code> lets a given <code>ClassVisitor</code>
 * optionally travel to the visited class, its superclass, its interfaces, and
 * its subclasses.
 *
 * @author Eric Lafortune
 */
public class ClassHierarchyTraveler implements ClassVisitor
{
    private final boolean visitThisClass;
    private final boolean visitSuperClass;
    private final boolean visitInterfaces;
    private final boolean visitSubclasses;

    private final ClassVisitor classVisitor;


    /**
     * Creates a new ClassHierarchyTraveler.
     * @param visitThisClass  specifies whether to visit the originally visited
     *                        classes.
     * @param visitSuperClass specifies whether to visit the super classes of
     *                        the visited classes.
     * @param visitInterfaces specifies whether to visit the interfaces of
     *                        the visited classes.
     * @param visitSubclasses specifies whether to visit the subclasses of
     *                        the visited classes.
     * @param classVisitor    the <code>ClassVisitor</code> to
     *                        which visits will be delegated.
     */
    public ClassHierarchyTraveler(boolean      visitThisClass,
                                  boolean      visitSuperClass,
                                  boolean      visitInterfaces,
                                  boolean      visitSubclasses,
                                  ClassVisitor classVisitor)
    {
        this.visitThisClass  = visitThisClass;
        this.visitSuperClass = visitSuperClass;
        this.visitInterfaces = visitInterfaces;
        this.visitSubclasses = visitSubclasses;

        this.classVisitor = classVisitor;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        programClass.hierarchyAccept(visitThisClass,
                                     visitSuperClass,
                                     visitInterfaces,
                                     visitSubclasses,
                                     classVisitor);
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        libraryClass.hierarchyAccept(visitThisClass,
                                     visitSuperClass,
                                     visitInterfaces,
                                     visitSubclasses,
                                     classVisitor);
    }
}

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
package proguard.classfile.constant.visitor;

import proguard.classfile.*;
import proguard.classfile.visitor.ClassVisitor;

/**
 * This ClassVisitor lets a given ConstantVisitor visit all the constant pool
 * entries of the super class and interfaces of the program classes it visits.
 *
 * @author Eric Lafortune
 */
public class SuperClassConstantVisitor implements ClassVisitor
{
    private final boolean         visitSuperClassConstants;
    private final boolean         visitInterfaceConstants;
    private final ConstantVisitor constantVisitor;


    /**
     * Creates a new SuperClassConstantVisitor.
     */
    public SuperClassConstantVisitor(boolean         visitSuperClassConstants,
                                     boolean         visitInterfaceConstants,
                                     ConstantVisitor constantVisitor)
    {
        this.visitSuperClassConstants = visitSuperClassConstants;
        this.visitInterfaceConstants  = visitInterfaceConstants;
        this.constantVisitor          = constantVisitor;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        if (visitSuperClassConstants)
        {
            programClass.superClassConstantAccept(constantVisitor);
        }

        if (visitInterfaceConstants)
        {
            programClass.interfaceConstantsAccept(constantVisitor);
        }
    }


    public void visitLibraryClass(LibraryClass libraryClass) {}
}

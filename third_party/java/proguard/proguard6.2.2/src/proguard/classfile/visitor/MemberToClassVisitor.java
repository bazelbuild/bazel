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
 * This MemberVisitor delegates all visits to a given ClassVisitor.
 * The latter visits the class of each visited class member.
 *
 * @author Eric Lafortune
 */
public class MemberToClassVisitor implements MemberVisitor
{
    private final ClassVisitor classVisitor;


    public MemberToClassVisitor(ClassVisitor classVisitor)
    {
        this.classVisitor = classVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        classVisitor.visitProgramClass(programClass);
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        classVisitor.visitProgramClass(programClass);
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
    {
        classVisitor.visitLibraryClass(libraryClass);
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        classVisitor.visitLibraryClass(libraryClass);
    }
}

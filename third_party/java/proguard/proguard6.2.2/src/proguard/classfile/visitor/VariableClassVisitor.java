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
 * This ClassVisitor delegates all method calls to a ClassVisitor
 * that can be changed at any time.
 *
 * @author Eric Lafortune
 */
public class VariableClassVisitor implements ClassVisitor
{
    private ClassVisitor classVisitor;


    public VariableClassVisitor()
    {
        this(null);
    }


    public VariableClassVisitor(ClassVisitor classVisitor)
    {
        this.classVisitor = classVisitor;
    }


    public void setClassVisitor(ClassVisitor classVisitor)
    {
        this.classVisitor = classVisitor;
    }

    public ClassVisitor getClassVisitor()
    {
        return classVisitor;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        if (classVisitor != null)
        {
            classVisitor.visitProgramClass(programClass);
        }
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        if (classVisitor != null)
        {
            classVisitor.visitLibraryClass(libraryClass);
        }
    }
}

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
 * This ClassPoolVisitor and ClassVisitor remembers the ClassPool instances
 * that it visits and applies the given ClassPoolVisitor to the most
 * recently remembered one, every time it visits a Clazz instance.
 *
 * @author Eric Lafortune
 */
public class ClassPoolClassVisitor
implements   ClassPoolVisitor,
             ClassVisitor
{
    private ClassPoolVisitor classPoolVisitor;
    private ClassPool classPool;


    /**
     * Creates a new ClassPoolClassVisitor.
     * @param classPoolVisitor
     */
    public ClassPoolClassVisitor(ClassPoolVisitor classPoolVisitor)
    {
        this.classPoolVisitor = classPoolVisitor;
    }


    // Implementations for ClassPoolVisitor.

    public void visitClassPool(ClassPool classPool)
    {
        this.classPool = classPool;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        classPoolVisitor.visitClassPool(classPool);
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        classPoolVisitor.visitClassPool(classPool);
    }
}

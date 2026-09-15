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
 * This <code>ClassVisitor</code> delegates its visits to one of two
 * <code>ClassVisitor</code> instances, depending on whether the name of
 * the visited class file is present in a given <code>ClassPool</code> or not.
 *
 * @author Eric Lafortune
 */
public class ClassPresenceFilter implements ClassVisitor
{
    private final ClassPool    classPool;
    private final ClassVisitor presentClassVisitor;
    private final ClassVisitor missingClassVisitor;


    /**
     * Creates a new ClassPresenceFilter.
     * @param classPool           the <code>ClassPool</code> in which the
     *                            presence will be tested.
     * @param presentClassVisitor the <code>ClassVisitor</code> to which visits
     *                            of present class files will be delegated.
     * @param missingClassVisitor the <code>ClassVisitor</code> to which visits
     *                            of missing class files will be delegated.
     */
    public ClassPresenceFilter(ClassPool    classPool,
                               ClassVisitor presentClassVisitor,
                               ClassVisitor missingClassVisitor)
    {
        this.classPool           = classPool;
        this.presentClassVisitor = presentClassVisitor;
        this.missingClassVisitor = missingClassVisitor;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        ClassVisitor classFileVisitor = classFileVisitor(programClass);

        if (classFileVisitor != null)
        {
            classFileVisitor.visitProgramClass(programClass);
        }
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        ClassVisitor classFileVisitor = classFileVisitor(libraryClass);

        if (classFileVisitor != null)
        {
            classFileVisitor.visitLibraryClass(libraryClass);
        }
    }


    // Small utility methods.

    /**
     * Returns the appropriate <code>ClassVisitor</code>.
     */
    private ClassVisitor classFileVisitor(Clazz clazz)
    {
        return classPool.getClass(clazz.getName()) != null ?
            presentClassVisitor :
            missingClassVisitor;
    }
}

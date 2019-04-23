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
 * This <code>ClassVisitor</code> delegates its visits to another given
 * <code>ClassVisitor</code>, but only for functional interfaces, that
 * is, interface classes that have exactly one abstract method.
 *
 * @author Eric Lafortune
 */
public class FunctionalInterfaceFilter implements ClassVisitor
{
    private final ClassVisitor classVisitor;


    /**
     * Creates a new ProgramClassFilter.
     * @param classVisitor the <code>ClassVisitor</code> to which visits
     *                     will be delegated.
     */
    public FunctionalInterfaceFilter(ClassVisitor classVisitor)
    {
        this.classVisitor = classVisitor;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        if (isFunctionalInterface(programClass))
        {
            classVisitor.visitProgramClass(programClass);
        }
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        if (isFunctionalInterface(libraryClass))
        {
            classVisitor.visitLibraryClass(libraryClass);
        }
    }


    // Small utility methods.

    private boolean isFunctionalInterface(Clazz clazz)
    {
        // Is it an interface?
        if ((clazz.getAccessFlags() & ClassConstants.ACC_INTERFACE) == 0)
        {
            return false;
        }

        // Count the abstract methods in the interface hierarchy.
        // Subtract any corresponding default methods, since only abstract
        // methods that don't have a default implementation count.
        // TODO: Find a better way to count default methods, since there may be more of them for a single abstract method, or we can find one via different paths.
        MemberCounter abstractMethodCounter  = new MemberCounter();
        MemberCounter defaultMethodCounter   = new MemberCounter();
        clazz.hierarchyAccept(true, false, true, false,
                              new AllMethodVisitor(
                              new MemberAccessFilter(ClassConstants.ACC_ABSTRACT, 0,
                              new MultiMemberVisitor(
                                  abstractMethodCounter,
                                  new SimilarMemberVisitor(clazz, true, true, true, false,
                                  new MemberAccessFilter(0, ClassConstants.ACC_ABSTRACT,
                                  defaultMethodCounter))
                              ))));

        return abstractMethodCounter.getCount() - defaultMethodCounter.getCount() == 1;
    }
}

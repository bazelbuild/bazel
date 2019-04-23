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
 * travel to the first concrete subclasses down in its hierarchy of abstract
 * classes and concrete classes.
 *
 * @author Eric Lafortune
 */
public class ConcreteClassDownTraveler
implements   ClassVisitor
{
    private final ClassVisitor classVisitor;


    /**
     * Creates a new ConcreteClassDownTraveler.
     * @param classVisitor     the <code>ClassVisitor</code> to
     *                         which visits will be delegated.
     */
    public ConcreteClassDownTraveler(ClassVisitor classVisitor)
    {
        this.classVisitor = classVisitor;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Is this an abstract class or an interface?
        if ((programClass.getAccessFlags() &
             (ClassConstants.ACC_INTERFACE |
              ClassConstants.ACC_ABSTRACT)) != 0)
        {
            // Travel down the hierarchy.
            Clazz[] subClasses = programClass.subClasses;
            if (subClasses != null)
            {
                for (int index = 0; index < subClasses.length; index++)
                {
                    subClasses[index].accept(this);
                }
            }
        }
        else
        {
            // Visit the class. Don't descend any further.
            programClass.accept(classVisitor);
        }
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        // Is this an abstract class or interface?
        if ((libraryClass.getAccessFlags() &
             (ClassConstants.ACC_INTERFACE |
              ClassConstants.ACC_ABSTRACT)) != 0)
        {
            // Travel down the hierarchy.
            Clazz[] subClasses = libraryClass.subClasses;
            if (subClasses != null)
            {
                for (int index = 0; index < subClasses.length; index++)
                {
                    subClasses[index].accept(this);
                }
            }
        }
        else
        {
            // Visit the class. Don't descend any further.
            libraryClass.accept(classVisitor);
        }
    }
}

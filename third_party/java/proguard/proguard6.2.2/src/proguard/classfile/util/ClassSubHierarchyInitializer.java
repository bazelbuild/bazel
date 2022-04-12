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
package proguard.classfile.util;

import proguard.classfile.*;
import proguard.classfile.visitor.ClassVisitor;

/**
 * This ClassVisitor adds all classes that it visits to the list of subclasses
 * of their superclass. These subclass lists make it more convenient to travel
 *
 * @author Eric Lafortune
 */
public class ClassSubHierarchyInitializer
implements   ClassVisitor
{
    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Add this class to the subclasses of its superclass.
        addSubclass(programClass, programClass.getSuperClass());

        // Add this class to the subclasses of its interfaces.
        for (int index = 0; index < programClass.u2interfacesCount; index++)
        {
            addSubclass(programClass, programClass.getInterface(index));
        }
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        // Add this class to the subclasses of its superclass,
        addSubclass(libraryClass, libraryClass.superClass);

        // Add this class to the subclasses of its interfaces.
        Clazz[] interfaceClasses = libraryClass.interfaceClasses;
        if (interfaceClasses != null)
        {
            for (int index = 0; index < interfaceClasses.length; index++)
            {
                // Add this class to the subclasses of the interface class.
                addSubclass(libraryClass, interfaceClasses[index]);
            }
        }
    }


    // Small utility methods.

    private void addSubclass(Clazz subclass, Clazz clazz)
    {
        if (clazz != null)
        {
            clazz.addSubClass(subclass);
        }
    }
}

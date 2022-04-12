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
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.visitor.*;

/**
 * This utility class can find the nest host class names of given classes.
 *
 * @author Eric Lafortune
 */
public class NestHostFinder
extends      SimplifiedVisitor
implements   ClassVisitor,
             AttributeVisitor
{
    private String nestHostClassName;


    /**
     * Returns whether the two given classes are in the same nest.
     */
    public boolean inSameNest(Clazz class1, Clazz class2)
    {
        // Are the classes the same?
        if (class1.equals(class2))
        {
            return true;
        }

        // Do the classes have the same nest host?
        String nestHostClassName1 = findNestHostClassName(class1);
        String nestHostClassName2 = findNestHostClassName(class2);

        return nestHostClassName1.equals(nestHostClassName2);
    }


    /**
     * Returns the class name of the nest host of the given class.
     * This may be the class itself, if the class doesn't have a nest host
     * attribute (including for class versions below Java 11 and for library
     * classes).
     */
    public String findNestHostClassName(Clazz clazz)
    {
        // The default is the name of the class itself.
        nestHostClassName = clazz.getName();

        // Look for an explicit attribute.
        clazz.accept(this);

        // Return the found name.
        return nestHostClassName;
    }


    // Implementations for ClassVisitor.


    public void visitProgramClass(ProgramClass programClass)
    {
        // The nest host attribute only exists since Java 10.
        if (programClass.u4version >= ClassConstants.CLASS_VERSION_10)
        {
            programClass.attributesAccept(this);
        }
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        // Library classes don't store their versions or attributes.
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitNestHostAttribute(Clazz clazz, NestHostAttribute nestHostAttribute)
    {
        // Remember the class name of the nest host.
        nestHostClassName = clazz.getClassName(nestHostAttribute.u2hostClassIndex);
    }
}

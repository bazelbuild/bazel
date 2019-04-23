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
package proguard.obfuscate;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;

import java.util.Arrays;

/**
 * This ClassVisitor removes attributes that are not marked as being used or
 * required.
 *
 * @see AttributeUsageMarker
 *
 * @author Eric Lafortune
 */
public class AttributeShrinker
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor,
             AttributeVisitor
{
    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Compact the array for class attributes.
        programClass.u2attributesCount =
            shrinkArray(programClass.attributes,
                        programClass.u2attributesCount);

        // Compact the attributes in fields, methods, and class attributes,
        programClass.fieldsAccept(this);
        programClass.methodsAccept(this);
        programClass.attributesAccept(this);
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        // Library classes are left unchanged.
    }


    // Implementations for MemberVisitor.

    public void visitProgramMember(ProgramClass programClass, ProgramMember programMember)
    {
        // Compact the attributes array.
        programMember.u2attributesCount =
            shrinkArray(programMember.attributes,
                        programMember.u2attributesCount);

        // Compact any attributes of the remaining attributes.
        programMember.attributesAccept(programClass, this);
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Compact the attributes array.
        codeAttribute.u2attributesCount =
            shrinkArray(codeAttribute.attributes,
                        codeAttribute.u2attributesCount);
    }


    // Small utility methods.

    /**
     * Removes all VisitorAccepter objects that are not marked as being used
     * from the given array.
     * @return the new number of VisitorAccepter objects.
     */
    private static int shrinkArray(VisitorAccepter[] array, int length)
    {
        int counter = 0;

        // Shift the used objects together.
        for (int index = 0; index < length; index++)
        {
            if (AttributeUsageMarker.isUsed(array[index]))
            {
                array[counter++] = array[index];
            }
        }

        // Clear the remaining array elements.
        Arrays.fill(array, counter, length, null);

        return counter;
    }
}

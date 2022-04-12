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
package proguard.classfile;

import proguard.classfile.attribute.Attribute;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.visitor.*;

/**
 * Representation of a method from a program class.
 *
 * @author Eric Lafortune
 */
public class ProgramMethod extends ProgramMember implements Method
{
    private static final Attribute[] EMPTY_ATTRIBUTES = new Attribute[0];


    /**
     * An extra field containing all the classes referenced in the
     * descriptor string. This field is filled out by the {@link
     * proguard.classfile.util.ClassReferenceInitializer ClassReferenceInitializer}.
     * The size of the array is the number of classes in the descriptor.
     * Primitive types and arrays of primitive types are ignored.
     * Unknown classes are represented as null values.
     */
    public Clazz[] referencedClasses;


    /**
     * Creates an uninitialized ProgramMethod.
     */
    public ProgramMethod()
    {
    }


    /**
     * Creates an initialized ProgramMethod without attributes.
     */
    public ProgramMethod(int     u2accessFlags,
                         int     u2nameIndex,
                         int     u2descriptorIndex,
                         Clazz[] referencedClasses)
    {
        this(u2accessFlags,
             u2nameIndex,
             u2descriptorIndex,
             0,
             EMPTY_ATTRIBUTES,
             referencedClasses);
    }


    /**
     * Creates an initialized ProgramMethod.
     */
    public ProgramMethod(int         u2accessFlags,
                         int         u2nameIndex,
                         int         u2descriptorIndex,
                         int         u2attributesCount,
                         Attribute[] attributes,
                         Clazz[]     referencedClasses)
    {
        super(u2accessFlags, u2nameIndex, u2descriptorIndex, u2attributesCount, attributes);

        this.referencedClasses = referencedClasses;
    }


    // Implementations for ProgramMember.

    public void accept(ProgramClass programClass, MemberVisitor memberVisitor)
    {
        memberVisitor.visitProgramMethod(programClass, this);
    }


    public void attributesAccept(ProgramClass programClass, AttributeVisitor attributeVisitor)
    {
        for (int index = 0; index < u2attributesCount; index++)
        {
            attributes[index].accept(programClass, this, attributeVisitor);
        }
    }


    // Implementations for Member.

    public void referencedClassesAccept(ClassVisitor classVisitor)
    {
        if (referencedClasses != null)
        {
            for (int index = 0; index < referencedClasses.length; index++)
            {
                if (referencedClasses[index] != null)
                {
                    referencedClasses[index].accept(classVisitor);
                }
            }
        }
    }
}

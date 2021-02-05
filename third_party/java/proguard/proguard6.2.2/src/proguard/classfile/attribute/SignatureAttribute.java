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
package proguard.classfile.attribute;

import proguard.classfile.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.visitor.ClassVisitor;

/**
 * This Attribute represents a signature attribute.
 *
 * @author Eric Lafortune
 */
public class SignatureAttribute extends Attribute
{
    public int u2signatureIndex;

    /**
     * An extra field containing all the classes referenced in the
     * signature string. This field is filled out by the {@link
     * proguard.classfile.util.ClassReferenceInitializer ClassReferenceInitializer}.
     * The size of the array is the number of classes in the signature.
     * Primitive types and arrays of primitive types are ignored.
     * Unknown classes are represented as null values.
     */
    public Clazz[] referencedClasses;


    /**
     * Creates an uninitialized SignatureAttribute.
     */
    public SignatureAttribute()
    {
    }


    /**
     * Creates an initialized SignatureAttribute.
     */
    public SignatureAttribute(int u2attributeNameIndex,
                              int u2signatureIndex)
    {
        super(u2attributeNameIndex);

        this.u2signatureIndex = u2signatureIndex;
    }


    /**
     * Returns the signature.
     */
    public String getSignature(Clazz clazz)
    {
        return clazz.getString(u2signatureIndex);
    }


    /**
     * Lets the Clazz objects referenced in the signature string accept the
     * given visitor.
     */
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


    // Implementations for Attribute.

    public void accept(Clazz clazz, AttributeVisitor attributeVisitor)
    {
        attributeVisitor.visitSignatureAttribute(clazz, this);
    }

    public void accept(Clazz clazz, Field field, AttributeVisitor attributeVisitor)
    {
        attributeVisitor.visitSignatureAttribute(clazz, field, this);
    }

    public void accept(Clazz clazz, Method method, AttributeVisitor attributeVisitor)
    {
        attributeVisitor.visitSignatureAttribute(clazz, method, this);
    }
 }

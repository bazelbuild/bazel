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

/**
 * This Attribute represents an unknown attribute.
 *
 * @author Eric Lafortune
 */
public class UnknownAttribute extends Attribute
{
    public final int    u4attributeLength;
    public       byte[] info;


    /**
     * Creates an uninitialized UnknownAttribute with the specified name and
     * length.
     */
    public UnknownAttribute(int u2attributeNameIndex,
                            int attributeLength)
    {
        this(u2attributeNameIndex, attributeLength, null);
    }


    /**
     * Creates an initialized UnknownAttribute.
     */
    public UnknownAttribute(int    u2attributeNameIndex,
                            int    u4attributeLength,
                            byte[] info)
    {
        super(u2attributeNameIndex);

        this.u4attributeLength = u4attributeLength;
        this.info              = info;
    }


    // Implementations for Attribute.

    public void accept(Clazz clazz, AttributeVisitor attributeVisitor)
    {
        attributeVisitor.visitUnknownAttribute(clazz, this);
    }

    public void accept(Clazz clazz, Field field, AttributeVisitor attributeVisitor)
    {
        attributeVisitor.visitUnknownAttribute(clazz, this);
    }

    public void accept(Clazz clazz, Method method, AttributeVisitor attributeVisitor)
    {
        attributeVisitor.visitUnknownAttribute(clazz, this);
    }

    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, AttributeVisitor attributeVisitor)
    {
        attributeVisitor.visitUnknownAttribute(clazz, this);
    }
}

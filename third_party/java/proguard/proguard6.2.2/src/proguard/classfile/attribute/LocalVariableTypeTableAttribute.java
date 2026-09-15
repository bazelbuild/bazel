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
import proguard.classfile.attribute.visitor.*;

/**
 * This Attribute represents a local variable table type attribute.
 *
 * @author Eric Lafortune
 */
public class LocalVariableTypeTableAttribute extends Attribute
{
    public int                     u2localVariableTypeTableLength;
    public LocalVariableTypeInfo[] localVariableTypeTable;


    /**
     * Creates an uninitialized LocalVariableTypeTableAttribute.
     */
    public LocalVariableTypeTableAttribute()
    {
    }


    /**
     * Creates an initialized LocalVariableTypeTableAttribute.
     */
    public LocalVariableTypeTableAttribute(int                     u2attributeNameIndex,
                                           int                     u2localVariableTypeTableLength,
                                           LocalVariableTypeInfo[] localVariableTypeTable)
    {
        super(u2attributeNameIndex);

        this.u2localVariableTypeTableLength = u2localVariableTypeTableLength;
        this.localVariableTypeTable         = localVariableTypeTable;
    }


    // Implementations for Attribute.

    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, AttributeVisitor attributeVisitor)
    {
        attributeVisitor.visitLocalVariableTypeTableAttribute(clazz, method, codeAttribute, this);
    }


    /**
     * Applies the given visitor to all local variable types.
     */
    public void localVariablesAccept(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeInfoVisitor localVariableTypeInfoVisitor)
    {
        for (int index = 0; index < u2localVariableTypeTableLength; index++)
        {
            // We don't need double dispatching here, since there is only one
            // type of LocalVariableTypeInfo.
            localVariableTypeInfoVisitor.visitLocalVariableTypeInfo(clazz, method, codeAttribute, localVariableTypeTable[index]);
        }
    }
}

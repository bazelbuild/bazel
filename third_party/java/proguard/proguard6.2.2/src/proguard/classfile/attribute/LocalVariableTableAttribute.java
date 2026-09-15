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
 * This Attribute represents a local variable table attribute.
 *
 * @author Eric Lafortune
 */
public class LocalVariableTableAttribute extends Attribute
{
    public int                 u2localVariableTableLength;
    public LocalVariableInfo[] localVariableTable;


    /**
     * Creates an uninitialized LocalVariableTableAttribute.
     */
    public LocalVariableTableAttribute()
    {
    }


    /**
     * Creates an initialized LocalVariableTableAttribute.
     */
    public LocalVariableTableAttribute(int                 u2attributeNameIndex,
                                       int                 u2localVariableTableLength,
                                       LocalVariableInfo[] localVariableTable)
    {
        super(u2attributeNameIndex);

        this.u2localVariableTableLength = u2localVariableTableLength;
        this.localVariableTable         = localVariableTable;
    }


    // Implementations for Attribute.

    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, AttributeVisitor attributeVisitor)
    {
        attributeVisitor.visitLocalVariableTableAttribute(clazz, method, codeAttribute, this);
    }


    /**
     * Applies the given visitor to all local variables.
     */
    public void localVariablesAccept(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableInfoVisitor localVariableInfoVisitor)
    {
        for (int index = 0; index < u2localVariableTableLength; index++)
        {
            // We don't need double dispatching here, since there is only one
            // type of LocalVariableInfo.
            localVariableInfoVisitor.visitLocalVariableInfo(clazz, method, codeAttribute, localVariableTable[index]);
        }
    }
}

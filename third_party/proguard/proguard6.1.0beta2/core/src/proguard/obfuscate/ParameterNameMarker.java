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

/**
 * This AttributeVisitor trims and marks all local variable (type) table
 * attributes that it visits. It keeps parameter names and types and removes
 * the ordinary local variable names and types.
 *
 * @author Eric Lafortune
 */
public class ParameterNameMarker
extends      SimplifiedVisitor
implements   AttributeVisitor
{
    private final AttributeVisitor attributeUsageMarker;


    /**
     * Constructs a new ParameterNameMarker.
     * @param attributeUsageMarker the marker that will be used to mark
     *                             attributes containing local variable info.
     */
    public ParameterNameMarker(AttributeVisitor attributeUsageMarker)
    {
        this.attributeUsageMarker = attributeUsageMarker;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        if (!AttributeUsageMarker.isUsed(localVariableTableAttribute) &&
            hasParameters(clazz, method))
        {
            // Shift the entries that start at offset 0 to the front.
            int newIndex = 0;

            for (int index = 0; index < localVariableTableAttribute.u2localVariableTableLength; index++)
            {
                LocalVariableInfo localVariableInfo =
                    localVariableTableAttribute.localVariableTable[index];

                if (localVariableInfo.u2startPC == 0)
                {
                    localVariableTableAttribute.localVariableTable[newIndex++] =
                        localVariableInfo;
                }
            }

            // Trim the table.
            localVariableTableAttribute.u2localVariableTableLength = newIndex;

            // Mark the table if there are any entries.
            if (newIndex > 0)
            {
                attributeUsageMarker.visitLocalVariableTableAttribute(clazz, method, codeAttribute, localVariableTableAttribute);
            }
        }
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        if (!AttributeUsageMarker.isUsed(localVariableTypeTableAttribute) &&
            hasParameters(clazz, method))
        {
            // Shift the entries that start at offset 0 to the front.
            int newIndex = 0;

            for (int index = 0; index < localVariableTypeTableAttribute.u2localVariableTypeTableLength; index++)
            {
                LocalVariableTypeInfo localVariableTypeInfo =
                    localVariableTypeTableAttribute.localVariableTypeTable[index];

                if (localVariableTypeInfo.u2startPC == 0)
                {
                    localVariableTypeTableAttribute.localVariableTypeTable[newIndex++] =
                        localVariableTypeInfo;
                }
            }

            // Trim the table.
            localVariableTypeTableAttribute.u2localVariableTypeTableLength = newIndex;

            // Mark the table if there are any entries.
            if (newIndex > 0)
            {
                attributeUsageMarker.visitLocalVariableTypeTableAttribute(clazz, method, codeAttribute, localVariableTypeTableAttribute);
            }
        }
    }


    // Small utility methods.

    private boolean hasParameters(Clazz clazz, Method method)
    {
        return method.getDescriptor(clazz).charAt(1) != ClassConstants.METHOD_ARGUMENTS_CLOSE;
    }
}
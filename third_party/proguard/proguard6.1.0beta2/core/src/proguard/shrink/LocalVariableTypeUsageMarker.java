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
package proguard.shrink;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.Constant;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.ClassVisitor;

/**
 * This AttributeVisitor recursively marks all information that points to used
 * classes, in the LocalVariableTable and LocalVariableTypeTable attributes that
 * it visits.
 *
 * @see UsageMarker
 *
 * @author Eric Lafortune
 */
public class LocalVariableTypeUsageMarker
extends      SimplifiedVisitor
implements   AttributeVisitor,
             LocalVariableInfoVisitor,
             LocalVariableTypeInfoVisitor,
             ClassVisitor,
             ConstantVisitor
{
    private final UsageMarker usageMarker;

    // Fields acting as return values for several visitor methods.
    private boolean tableUsed;
    private boolean variableInfoUsed;


    /**
     * Creates a new LocalVariableTypeUsageMarker.
     * @param usageMarker the usage marker that is used to mark the classes
     *                    and class members.
     */
    public LocalVariableTypeUsageMarker(UsageMarker usageMarker)
    {
        this.usageMarker = usageMarker;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        // Check and mark the individual entries.
        tableUsed = false;
        localVariableTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);

        // Mark the table if any of the entries is marked.
        if (tableUsed)
        {
            usageMarker.markAsUsed(localVariableTableAttribute);

            markConstant(clazz, localVariableTableAttribute.u2attributeNameIndex);
        }
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        // Check and mark the individual entries.
        tableUsed = false;
        localVariableTypeTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);

        // Mark the table if any of the entries is marked.
        if (tableUsed)
        {
            usageMarker.markAsUsed(localVariableTypeTableAttribute);

            markConstant(clazz, localVariableTypeTableAttribute.u2attributeNameIndex);
        }
    }


    // Implementations for LocalVariableInfoVisitor.

    public void visitLocalVariableInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableInfo localVariableInfo)
    {
        // Only keep the local variable info if all of its classes are used.
        variableInfoUsed = true;
        localVariableInfo.referencedClassAccept(this);

        if (variableInfoUsed)
        {
            // We got a positive used flag, so the local variable info is useful.
            usageMarker.markAsUsed(localVariableInfo);

            markConstant(clazz, localVariableInfo.u2nameIndex);
            markConstant(clazz, localVariableInfo.u2descriptorIndex);

            tableUsed = true;
        }
    }


    // Implementations for LocalVariableTypeInfoVisitor.

    public void visitLocalVariableTypeInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeInfo localVariableTypeInfo)
    {
        // Only keep the local variable info if all of its classes are used.
        variableInfoUsed = true;
        localVariableTypeInfo.referencedClassesAccept(this);

        if (variableInfoUsed)
        {
            // We got a positive used flag, so the local variable info is useful.
            usageMarker.markAsUsed(localVariableTypeInfo);

            markConstant(clazz, localVariableTypeInfo.u2nameIndex);
            markConstant(clazz, localVariableTypeInfo.u2signatureIndex);

            tableUsed = true;
        }
    }


    // Implementations for ClassVisitor.

    public void visitLibraryClass(LibraryClass libraryClass) {}


    public void visitProgramClass(ProgramClass programClass)
    {
        // Don't keep the local variable info if one of its classes is not used.
        if (!usageMarker.isUsed(programClass))
        {
            variableInfoUsed = false;
        }
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant)
    {
        usageMarker.markAsUsed(constant);
    }


    // Small utility methods.

    /**
     * Marks the given constant pool entry of the given class.
     */
    private void markConstant(Clazz clazz, int index)
    {
         clazz.constantPoolEntryAccept(index, this);
    }
}

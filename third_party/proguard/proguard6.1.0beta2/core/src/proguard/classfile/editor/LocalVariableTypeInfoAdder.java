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
package proguard.classfile.editor;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.LocalVariableTypeInfoVisitor;

/**
 * This LocalVariableTypeInfoVisitor adds all local variable types that it
 * visits to the given target local variable type attribute.
 */
public class LocalVariableTypeInfoAdder
implements   LocalVariableTypeInfoVisitor
{
    private final ConstantAdder                     constantAdder;
    private final LocalVariableTypeTableAttributeEditor localVariableTypeTableAttributeEditor;


    /**
     * Creates a new LocalVariableTypeInfoAdder that will copy local variable
     * types into the given target local variable type table.
     */
    public LocalVariableTypeInfoAdder(ProgramClass                    targetClass,
                                      LocalVariableTypeTableAttribute targetLocalVariableTypeTableAttribute)
    {
        this.constantAdder                         = new ConstantAdder(targetClass);
        this.localVariableTypeTableAttributeEditor = new LocalVariableTypeTableAttributeEditor(targetLocalVariableTypeTableAttribute);
    }


    // Implementations for LocalVariableTypeInfoVisitor.

    public void visitLocalVariableTypeInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeInfo localVariableTypeInfo)
    {
        // Create a new local variable type.
        LocalVariableTypeInfo newLocalVariableTypeInfo =
            new LocalVariableTypeInfo(localVariableTypeInfo.u2startPC,
                                      localVariableTypeInfo.u2length,
                                      constantAdder.addConstant(clazz, localVariableTypeInfo.u2nameIndex),
                                      constantAdder.addConstant(clazz, localVariableTypeInfo.u2signatureIndex),
                                      localVariableTypeInfo.u2index);

        // TODO: Clone array.
        newLocalVariableTypeInfo.referencedClasses = localVariableTypeInfo.referencedClasses;

        // Add it to the target.
        localVariableTypeTableAttributeEditor.addLocalVariableTypeInfo(newLocalVariableTypeInfo);
    }
}
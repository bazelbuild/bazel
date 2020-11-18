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
import proguard.classfile.attribute.visitor.LocalVariableInfoVisitor;

/**
 * This LocalVariableInfoVisitor adds all local variables that it visits to the
 * given target local variable table attribute.
 */
public class LocalVariableInfoAdder
implements   LocalVariableInfoVisitor
{
    private final ConstantAdder                     constantAdder;
    private final LocalVariableTableAttributeEditor localVariableTableAttributeEditor;


    /**
     * Creates a new LocalVariableInfoAdder that will copy local variables
     * into the given target local variable table.
     */
    public LocalVariableInfoAdder(ProgramClass                targetClass,
                                  LocalVariableTableAttribute targetLocalVariableTableAttribute)
    {
        this.constantAdder                     = new ConstantAdder(targetClass);
        this.localVariableTableAttributeEditor = new LocalVariableTableAttributeEditor(targetLocalVariableTableAttribute);
    }


    // Implementations for LocalVariableInfoVisitor.

    public void visitLocalVariableInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableInfo localVariableInfo)
    {
        // Create a new local variable.
        LocalVariableInfo newLocalVariableInfo =
            new LocalVariableInfo(localVariableInfo.u2startPC,
                                  localVariableInfo.u2length,
                                  constantAdder.addConstant(clazz, localVariableInfo.u2nameIndex),
                                  constantAdder.addConstant(clazz, localVariableInfo.u2descriptorIndex),
                                  localVariableInfo.u2index);

        newLocalVariableInfo.referencedClass = localVariableInfo.referencedClass;

        // Add it to the target.
        localVariableTableAttributeEditor.addLocalVariableInfo(newLocalVariableInfo);
    }
}
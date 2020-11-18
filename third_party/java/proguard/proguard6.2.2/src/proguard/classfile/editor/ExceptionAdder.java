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
import proguard.classfile.attribute.ExceptionsAttribute;
import proguard.classfile.constant.ClassConstant;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This ConstantVisitor adds all class constants that it visits to the given
 * target exceptions attribute.
 *
 * @author Eric Lafortune
 */
public class ExceptionAdder
extends      SimplifiedVisitor
implements   ConstantVisitor
{
    private final ConstantAdder             constantAdder;
    private final ExceptionsAttributeEditor exceptionsAttributeEditor;


    /**
     * Creates a new ExceptionAdder that will copy classes into the given
     * target exceptions attribute.
     */
    public ExceptionAdder(ProgramClass        targetClass,
                          ExceptionsAttribute targetExceptionsAttribute)
    {
        constantAdder             = new ConstantAdder(targetClass);
        exceptionsAttributeEditor = new ExceptionsAttributeEditor(targetExceptionsAttribute);
    }


    // Implementations for ConstantVisitor.

    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        // Add a class constant to the constant pool.
        constantAdder.visitClassConstant(clazz, classConstant);

        // Add the index of the class constant to the list of exceptions.
        exceptionsAttributeEditor.addException(constantAdder.getConstantIndex());
    }
}

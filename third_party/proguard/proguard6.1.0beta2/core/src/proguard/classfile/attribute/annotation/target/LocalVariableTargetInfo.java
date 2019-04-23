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
package proguard.classfile.attribute.annotation.target;

import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.attribute.annotation.TypeAnnotation;
import proguard.classfile.attribute.annotation.target.visitor.*;

/**
 * Representation of a local variable annotation target.
 *
 * @author Eric Lafortune
 */
public class LocalVariableTargetInfo extends TargetInfo
{
    public int                          u2tableLength;
    public LocalVariableTargetElement[] table;


    /**
     * Creates an uninitialized LocalVariableTargetInfo.
     */
    public LocalVariableTargetInfo()
    {
    }


    /**
     * Creates a partially initialized LocalVariableTargetInfo.
     */
    public LocalVariableTargetInfo(byte u1targetType)
    {
        super(u1targetType);
    }


    /**
     * Creates an initialized LocalVariableTargetInfo.
     */
    public LocalVariableTargetInfo(byte                         u1targetType,
                                   int                          u2tableLength,
                                   LocalVariableTargetElement[] table)
    {
        super(u1targetType);

        this.u2tableLength = u2tableLength;
        this.table         = table;
    }


    /**
     * Applies the given visitor to all target elements.
     */
    public void targetElementsAccept(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, LocalVariableTargetElementVisitor localVariableTargetElementVisitor)
    {
        for (int index = 0; index < u2tableLength; index++)
        {
            // We don't need double dispatching here, since there is only one
            // type of TypePathInfo.
            localVariableTargetElementVisitor.visitLocalVariableTargetElement(clazz, method, codeAttribute, typeAnnotation, this, table[index]);
        }
    }


    // Implementations for TargetInfo.

    /**
     * Lets the visitor visit, with Method and CodeAttribute null.
     */
    public void accept(Clazz clazz, TypeAnnotation typeAnnotation, TargetInfoVisitor targetInfoVisitor)
    {
        targetInfoVisitor.visitLocalVariableTargetInfo(clazz, null, null, typeAnnotation, this);
    }


    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, TargetInfoVisitor targetInfoVisitor)
    {
        targetInfoVisitor.visitLocalVariableTargetInfo(clazz, method, codeAttribute, typeAnnotation, this);
    }
}

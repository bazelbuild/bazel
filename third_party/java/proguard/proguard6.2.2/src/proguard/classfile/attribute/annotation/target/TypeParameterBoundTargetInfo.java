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
import proguard.classfile.attribute.annotation.TypeAnnotation;
import proguard.classfile.attribute.annotation.target.visitor.TargetInfoVisitor;

/**
 * Representation of a type parameter bound annotation target.
 *
 * @author Eric Lafortune
 */
public class TypeParameterBoundTargetInfo extends TargetInfo
{
    public int u1typeParameterIndex;
    public int u1boundIndex;


    /**
     * Creates an uninitialized TypeParameterBoundTargetInfo.
     */
    public TypeParameterBoundTargetInfo()
    {
    }


    /**
     * Creates a partially initialized TypeParameterBoundTargetInfo.
     */
    public TypeParameterBoundTargetInfo(byte u1targetType)
    {
        super(u1targetType);
    }


    /**
     * Creates an initialized TypeParameterBoundTargetInfo.
     */
    public TypeParameterBoundTargetInfo(byte u1targetType,
                                        int  u1typeParameterIndex,
                                        int  u1boundIndex)
    {
        super(u1targetType);

        this.u1typeParameterIndex = u1typeParameterIndex;
        this.u1boundIndex         = u1boundIndex;
    }


    // Implementations for TargetInfo.

    public void accept(Clazz clazz, TypeAnnotation typeAnnotation, TargetInfoVisitor targetInfoVisitor)
    {
        targetInfoVisitor.visitTypeParameterBoundTargetInfo(clazz, typeAnnotation, this);
    }


    public void accept(Clazz clazz, Field field, TypeAnnotation typeAnnotation, TargetInfoVisitor targetInfoVisitor)
    {
        targetInfoVisitor.visitTypeParameterBoundTargetInfo(clazz, field, typeAnnotation, this);
    }


    public void accept(Clazz clazz, Method method, TypeAnnotation typeAnnotation, TargetInfoVisitor targetInfoVisitor)
    {
        targetInfoVisitor.visitTypeParameterBoundTargetInfo(clazz, method, typeAnnotation, this);
    }
}

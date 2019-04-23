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
 * Representation of a 'throws' annotation target.
 *
 * @author Eric Lafortune
 */
public class ThrowsTargetInfo extends TargetInfo
{
    public int u2throwsTypeIndex;


    /**
     * Creates an uninitialized ThrowsTargetInfo.
     */
    public ThrowsTargetInfo()
    {
    }


    /**
     * Creates a partially initialized ThrowsTargetInfo.
     */
    public ThrowsTargetInfo(byte u1targetType)
    {
        super(u1targetType);
    }


    /**
     * Creates an initialized ThrowsTargetInfo.
     */
    public ThrowsTargetInfo(byte u1targetType,
                            int  u2throwsTypeIndex)
    {
        super(u1targetType);

        this.u2throwsTypeIndex = u2throwsTypeIndex;
    }


    // Implementations for TargetInfo.

    /**
     * Lets the visitor visit, with Method null.
     */
    public void accept(Clazz clazz, TypeAnnotation typeAnnotation, TargetInfoVisitor targetInfoVisitor)
    {
        targetInfoVisitor.visitThrowsTargetInfo(clazz, null, typeAnnotation, this);
    }


    public void accept(Clazz clazz, Method method, TypeAnnotation typeAnnotation, TargetInfoVisitor targetInfoVisitor)
    {
        targetInfoVisitor.visitThrowsTargetInfo(clazz, method, typeAnnotation, this);
    }
}

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
import proguard.classfile.attribute.annotation.target.visitor.TargetInfoVisitor;

/**
 * Representation of an offset annotation target.
 *
 * @author Eric Lafortune
 */
public class TypeArgumentTargetInfo extends TargetInfo
{
    public int u2offset;
    public int u1typeArgumentIndex;


    /**
     * Creates an uninitialized TypeArgumentTargetInfo.
     */
    public TypeArgumentTargetInfo()
    {
    }


    /**
     * Creates a partially initialized TypeArgumentTargetInfo.
     */
    public TypeArgumentTargetInfo(byte u1targetType)
    {
        super(u1targetType);
    }


    /**
     * Creates an initialized TypeArgumentTargetInfo.
     */
    public TypeArgumentTargetInfo(byte u1targetType,
                                  int  u2offset,
                                  int  u1typeArgumentIndex)
    {
        super(u1targetType);

        this.u2offset            = u2offset;
        this.u1typeArgumentIndex = u1typeArgumentIndex;
    }


    // Implementations for TargetInfo.

    /**
     * Lets the visitor visit, with Method and CodeAttribute null.
     */
    public void accept(Clazz clazz, TypeAnnotation typeAnnotation, TargetInfoVisitor targetInfoVisitor)
    {
        targetInfoVisitor.visitTypeArgumentTargetInfo(clazz, null, null, typeAnnotation, this);
    }


    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, TargetInfoVisitor targetInfoVisitor)
    {
        targetInfoVisitor.visitTypeArgumentTargetInfo(clazz, method, codeAttribute, typeAnnotation, this);
    }
}

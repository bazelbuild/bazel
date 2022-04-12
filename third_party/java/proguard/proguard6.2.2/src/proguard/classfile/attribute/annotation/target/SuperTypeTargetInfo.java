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

import proguard.classfile.Clazz;
import proguard.classfile.attribute.annotation.TypeAnnotation;
import proguard.classfile.attribute.annotation.target.visitor.TargetInfoVisitor;

/**
 * Representation of a super type annotation target.
 *
 * @author Eric Lafortune
 */
public class SuperTypeTargetInfo extends TargetInfo
{
    public static final int EXTENDS_INDEX = 65535;


    public int u2superTypeIndex;


    /**
     * Creates an uninitialized SuperTypeTargetInfo.
     */
    public SuperTypeTargetInfo()
    {
    }


    /**
     * Creates a partially initialized SuperTypeTargetInfo.
     */
    public SuperTypeTargetInfo(byte u1targetType)
    {
        super(u1targetType);
    }


    /**
     * Creates an initialized SuperTypeTargetInfo.
     */
    public SuperTypeTargetInfo(byte u1targetType,
                               int  u2superTypeIndex)
    {
        super(u1targetType);

        this.u2superTypeIndex = u2superTypeIndex;
    }


    // Implementations for TargetInfo.

    public void accept(Clazz clazz, TypeAnnotation typeAnnotation, TargetInfoVisitor targetInfoVisitor)
    {
        targetInfoVisitor.visitSuperTypeTargetInfo(clazz, typeAnnotation, this);
    }
}

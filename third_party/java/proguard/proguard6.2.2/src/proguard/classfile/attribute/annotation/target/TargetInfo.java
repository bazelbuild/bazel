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
 * Representation of an annotation target.
 *
 * @author Eric Lafortune
 */
public abstract class TargetInfo
{
    public byte u1targetType;


    /**
     * Creates an uninitialized TargetInfo.
     */
    protected TargetInfo()
    {
    }


    /**
     * Creates an initialized TargetInfo.
     */
    protected TargetInfo(byte u1targetType)
    {
        this.u1targetType = u1targetType;
    }


    /**
     * Returns the type of the target.
     */
    public byte getTargetType()
    {
        return u1targetType;
    }


    // Methods to be implemented by extensions.

    /**
     * Accepts the given visitor, in the context of a type annotation on a class.
     */
    public void accept(Clazz clazz,                                             TypeAnnotation typeAnnotation, TargetInfoVisitor targetInfoVisitor)
    {
        throw new UnsupportedOperationException("Unsupported type annotation [0x"+Integer.toHexString(u1targetType)+"] on a class");
    }

    /**
     * Accepts the given visitor, in the context of a type annotation on a field.
     */
    public void accept(Clazz clazz, Field field,                                TypeAnnotation typeAnnotation, TargetInfoVisitor targetInfoVisitor)
    {
        throw new UnsupportedOperationException("Unsupported type annotation [0x"+Integer.toHexString(u1targetType)+"] on a field");
    }

    /**
     * Accepts the given visitor, in the context of a type annotation on a method.
     */
    public void accept(Clazz clazz, Method method,                              TypeAnnotation typeAnnotation, TargetInfoVisitor targetInfoVisitor)
    {
        throw new UnsupportedOperationException("Unsupported type annotation [0x"+Integer.toHexString(u1targetType)+"] on a method");
    }

    /**
     * Accepts the given visitor, in the context of a type annotation code.
     */
    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, TargetInfoVisitor targetInfoVisitor)
    {
        throw new UnsupportedOperationException("Unsupported type annotation [0x"+Integer.toHexString(u1targetType)+"] on code");
    }
}

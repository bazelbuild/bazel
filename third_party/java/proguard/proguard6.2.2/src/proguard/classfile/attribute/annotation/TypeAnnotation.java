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
package proguard.classfile.attribute.annotation;

import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.attribute.annotation.target.TargetInfo;
import proguard.classfile.attribute.annotation.target.visitor.TargetInfoVisitor;
import proguard.classfile.attribute.annotation.visitor.*;

/**
 * Representation of a type annotation.
 *
 * @author Eric Lafortune
 */
public class TypeAnnotation extends Annotation
{
    public TargetInfo     targetInfo;
    public TypePathInfo[] typePath;


    /**
     * Creates an uninitialized TypeAnnotation.
     */
    public TypeAnnotation()
    {
    }


    /**
     * Creates an initialized TypeAnnotation.
     */
    public TypeAnnotation(int            u2typeIndex,
                          int            u2elementValuesCount,
                          ElementValue[] elementValues,
                          TargetInfo     targetInfo,
                          TypePathInfo[] typePath)
    {
        super(u2typeIndex, u2elementValuesCount, elementValues);

        this.targetInfo = targetInfo;
        this.typePath   = typePath;
    }


    /**
     * Applies the given visitor to the target info.
     */
    public void targetInfoAccept(Clazz clazz, TargetInfoVisitor targetInfoVisitor)
    {
        // We don't need double dispatching here, since there is only one
        // type of TypePathInfo.
        targetInfo.accept(clazz, this, targetInfoVisitor);
    }


    /**
     * Applies the given visitor to the target info.
     */
    public void targetInfoAccept(Clazz clazz, Field field, TargetInfoVisitor targetInfoVisitor)
    {
        // We don't need double dispatching here, since there is only one
        // type of TypePathInfo.
        targetInfo.accept(clazz, field, this, targetInfoVisitor);
    }


    /**
     * Applies the given visitor to the target info.
     */
    public void targetInfoAccept(Clazz clazz, Method method, TargetInfoVisitor targetInfoVisitor)
    {
        // We don't need double dispatching here, since there is only one
        // type of TypePathInfo.
        targetInfo.accept(clazz, method, this, targetInfoVisitor);
    }


    /**
     * Applies the given visitor to the target info.
     */
    public void targetInfoAccept(Clazz clazz, Method method, CodeAttribute codeAttribute, TargetInfoVisitor targetInfoVisitor)
    {
        // We don't need double dispatching here, since there is only one
        // type of TypePathInfo.
        targetInfo.accept(clazz, method, codeAttribute, this, targetInfoVisitor);
    }


    /**
     * Applies the given visitor to all type path elements.
     */
    public void typePathInfosAccept(Clazz clazz, TypePathInfoVisitor typePathVisitor)
    {
        for (int index = 0; index < typePath.length; index++)
        {
            // We don't need double dispatching here, since there is only one
            // type of TypePathInfo.
            typePathVisitor.visitTypePathInfo(clazz, this, typePath[index]);
        }
    }


    /**
     * Applies the given visitor to all type path elements.
     */
    public void typePathInfosAccept(Clazz clazz, Field field, TypePathInfoVisitor typePathVisitor)
    {
        for (int index = 0; index < typePath.length; index++)
        {
            // We don't need double dispatching here, since there is only one
            // type of TypePathInfo.
            typePathVisitor.visitTypePathInfo(clazz, field, this, typePath[index]);
        }
    }


    /**
     * Applies the given visitor to all type path elements.
     */
    public void typePathInfosAccept(Clazz clazz, Method method, TypePathInfoVisitor typePathVisitor)
    {
        for (int index = 0; index < typePath.length; index++)
        {
            // We don't need double dispatching here, since there is only one
            // type of TypePathInfo.
            typePathVisitor.visitTypePathInfo(clazz, method, this, typePath[index]);
        }
    }


    /**
     * Applies the given visitor to all type path elements.
     */
    public void typePathInfosAccept(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, TypePathInfoVisitor typePathVisitor)
    {
        for (int index = 0; index < typePath.length; index++)
        {
            // We don't need double dispatching here, since there is only one
            // type of TypePathInfo.
            typePathVisitor.visitTypePathInfo(clazz, method, codeAttribute, typeAnnotation, typePath[index]);
        }
    }
}

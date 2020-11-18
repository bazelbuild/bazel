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
package proguard.classfile.visitor;

import proguard.classfile.Clazz;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.*;

/**
 * This ConstantVisitor lets a given ClassVisitor visit all the referenced
 * classes that are returned by the dynamic constants and invoke dynamic
 * constants that it visits.
 *
 * @author Eric Lafortune
 */
public class DynamicReturnedClassVisitor
extends      SimplifiedVisitor
implements   ConstantVisitor
{
    protected final ClassVisitor classVisitor;


    public DynamicReturnedClassVisitor(ClassVisitor classVisitor)
    {
        this.classVisitor = classVisitor;
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitDynamicConstant(Clazz clazz, DynamicConstant dynamicConstant)
    {
        // Is the method returning a class type?
        Clazz[] referencedClasses = dynamicConstant.referencedClasses;
        if (referencedClasses != null    &&
            referencedClasses.length > 0 &&
            ClassUtil.isInternalClassType(ClassUtil.internalMethodReturnType(dynamicConstant.getType(clazz))))
        {
            // Let the visitor visit the return type class, if any.
            Clazz referencedClass = referencedClasses[referencedClasses.length - 1];
            if (referencedClass != null)
            {
                referencedClass.accept(classVisitor);
            }
        }
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        // Is the method returning a class type?
        Clazz[] referencedClasses = invokeDynamicConstant.referencedClasses;
        if (referencedClasses != null    &&
            referencedClasses.length > 0 &&
            ClassUtil.isInternalClassType(ClassUtil.internalMethodReturnType(invokeDynamicConstant.getType(clazz))))
        {
            // Let the visitor visit the return type class, if any.
            Clazz referencedClass = referencedClasses[referencedClasses.length - 1];
            if (referencedClass != null)
            {
                referencedClass.accept(classVisitor);
            }
        }
    }
}

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
package proguard.classfile.constant.visitor;

import proguard.classfile.Clazz;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This ConstantVisitor and BootstrapMethodInfoVisitor travels from any invoke
 * dynamic constants or bootstrap method info entries that it visits to their
 * bootstrap method handle constants, and applies a given constant visitor.
 *
 * @author Eric Lafortune
 */
public class BootstrapMethodHandleTraveler
extends      SimplifiedVisitor
implements   ConstantVisitor,
             AttributeVisitor,
             BootstrapMethodInfoVisitor
{
    private ConstantVisitor bootstrapMethodHandleVisitor;

    // Field serving as a method argument.
    int bootstrapMethodAttributeIndex;


    /**
     * Creates a new BootstrapMethodHandleVisitor that will delegate to the
     * given constant visitor.
     */
    public BootstrapMethodHandleTraveler(ConstantVisitor bootstrapMethodHandleVisitor)
    {
        this.bootstrapMethodHandleVisitor = bootstrapMethodHandleVisitor;
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitDynamicConstant(Clazz clazz, DynamicConstant dynamicConstant)
    {
        // Pass the method index.
        bootstrapMethodAttributeIndex =
            dynamicConstant.u2bootstrapMethodAttributeIndex;

        // Delegate to the bootstrap method.
        clazz.attributesAccept(this);
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        // Pass the method index.
        bootstrapMethodAttributeIndex =
            invokeDynamicConstant.u2bootstrapMethodAttributeIndex;

        // Delegate to the bootstrap method.
        clazz.attributesAccept(this);
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitBootstrapMethodsAttribute(Clazz clazz, BootstrapMethodsAttribute bootstrapMethodsAttribute)
    {
        // Check bootstrap methods.
        bootstrapMethodsAttribute.bootstrapMethodEntryAccept(clazz,
                                                             bootstrapMethodAttributeIndex,
                                                             this);
    }


    // Implementations for BootstrapMethodInfoVisitor.

    public void visitBootstrapMethodInfo(Clazz clazz, BootstrapMethodInfo bootstrapMethodInfo)
    {
        // Check bootstrap method.
        clazz.constantPoolEntryAccept(bootstrapMethodInfo.u2methodHandleIndex,
                                      bootstrapMethodHandleVisitor);
    }
}

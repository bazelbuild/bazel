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
package proguard.classfile.constant;

import proguard.classfile.*;
import proguard.classfile.constant.visitor.*;
import proguard.classfile.visitor.ClassVisitor;

/**
 * This Constant represents a dynamic constant in the constant pool.
 *
 * @author Eric Lafortune
 */
public class DynamicConstant extends Constant
{
    public int u2bootstrapMethodAttributeIndex;
    public int u2nameAndTypeIndex;

    /**
     * An extra field pointing to the Clazz objects referenced in the
     * descriptor string. This field is filled out by the <code>{@link
     * proguard.classfile.util.ClassReferenceInitializer ClassReferenceInitializer}</code>.
     * References to primitive types are ignored.
     */
    public Clazz[] referencedClasses;


    /**
     * Creates an uninitialized InvokeDynamicConstant.
     */
    public DynamicConstant()
    {
    }


    /**
     * Creates a new InvokeDynamicConstant with the given bootstrap method
     * and name-and-type indices.
     * @param u2bootstrapMethodAttributeIndex the index of the bootstrap method
     *                                        entry in the bootstrap methods
     *                                        attribute.
     * @param u2nameAndTypeIndex              the index of the name and type
     *                                        entry in the constant pool.
     * @param referencedClasses               the classes referenced by the
     *                                        type.
     */
    public DynamicConstant(int     u2bootstrapMethodAttributeIndex,
                           int     u2nameAndTypeIndex,
                           Clazz[] referencedClasses)
    {
        this.u2bootstrapMethodAttributeIndex = u2bootstrapMethodAttributeIndex;
        this.u2nameAndTypeIndex              = u2nameAndTypeIndex;
        this.referencedClasses               = referencedClasses;
    }


    /**
     * Returns the index of the bootstrap method in the bootstrap methods
     * attribute of the class.
     */
    public int getBootstrapMethodAttributeIndex()
    {
        return u2bootstrapMethodAttributeIndex;
    }

    /**
     * Returns the name-and-type index.
     */
    public int getNameAndTypeIndex()
    {
        return u2nameAndTypeIndex;
    }

    /**
     * Returns the method name.
     */
    public String getName(Clazz clazz)
    {
        return clazz.getName(u2nameAndTypeIndex);
    }

    /**
     * Returns the method type.
     */
    public String getType(Clazz clazz)
    {
        return clazz.getType(u2nameAndTypeIndex);
    }


    /**
     * Lets the Clazz objects referenced in the descriptor string
     * accept the given visitor.
     */
    public void referencedClassesAccept(ClassVisitor classVisitor)
    {
        if (referencedClasses != null)
        {
            for (int index = 0; index < referencedClasses.length; index++)
            {
                if (referencedClasses[index] != null)
                {
                    referencedClasses[index].accept(classVisitor);
                }
            }
        }
    }


    /**
     * Lets the bootstrap method handle constant accept the given visitor.
     */
    public void bootstrapMethodHandleAccept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        new BootstrapMethodHandleTraveler(constantVisitor).visitDynamicConstant(clazz, this);
    }


    // Implementations for Constant.

    public int getTag()
    {
        return ClassConstants.CONSTANT_Dynamic;
    }

    public void accept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        constantVisitor.visitDynamicConstant(clazz, this);
    }
}

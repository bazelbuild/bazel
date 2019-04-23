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
import proguard.classfile.visitor.*;

/**
 * This Constant represents a ref constant in the constant pool.
 *
 * @author Eric Lafortune
 */
public abstract class RefConstant extends Constant
{
    public int u2classIndex;
    public int u2nameAndTypeIndex;

    /**
     * An extra field pointing to the referenced Clazz object.
     * This field is typically filled out by the <code>{@link
     * proguard.classfile.util.ClassReferenceInitializer
     * ClassReferenceInitializer}</code>.
     */
    public Clazz referencedClass;

    /**
     * An extra field optionally pointing to the referenced Member object.
     * This field is typically filled out by the <code>{@link
     * proguard.classfile.util.ClassReferenceInitializer
     * ClassReferenceInitializer}</code>.
     */
    public Member referencedMember;


    protected RefConstant()
    {
    }


    /**
     * Returns the class index.
     */
    public int getClassIndex()
    {
        return u2classIndex;
    }

    /**
     * Returns the name-and-type index.
     */
    public int getNameAndTypeIndex()
    {
        return u2nameAndTypeIndex;
    }

    /**
     * Sets the name-and-type index.
     */
    public void setNameAndTypeIndex(int index)
    {
        u2nameAndTypeIndex = index;
    }

    /**
     * Returns the class name.
     */
    public String getClassName(Clazz clazz)
    {
        return clazz.getClassName(u2classIndex);
    }

    /**
     * Returns the method/field name.
     */
    public String getName(Clazz clazz)
    {
        return clazz.getName(u2nameAndTypeIndex);
    }

    /**
     * Returns the type.
     */
    public String getType(Clazz clazz)
    {
        return clazz.getType(u2nameAndTypeIndex);
    }


    /**
     * Lets the referenced class accept the given visitor.
     */
    public void referencedClassAccept(ClassVisitor classVisitor)
    {
        if (referencedClass != null)
        {
            referencedClass.accept(classVisitor);
        }
    }


    /**
     * Lets the referenced class member accept the given visitor.
     */
    public void referencedMemberAccept(MemberVisitor memberVisitor)
    {
        if (referencedMember != null)
        {
            referencedMember.accept(referencedClass,
                                    memberVisitor);
        }
    }
}

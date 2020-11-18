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
import proguard.classfile.constant.visitor.ConstantVisitor;

/**
 * This Constant represents a method handle constant in the constant pool.
 *
 * @author Eric Lafortune
 */
public class MethodHandleConstant extends Constant
{
    public int u1referenceKind;
    public int u2referenceIndex;


    /**
     * An extra field pointing to the java.lang.invoke.MethodHandle Clazz object.
     * This field is typically filled out by the <code>{@link
     * proguard.classfile.util.ClassReferenceInitializer
     * ClassReferenceInitializer}</code>.
     */
    public Clazz javaLangInvokeMethodHandleClass;


    /**
     * Creates an uninitialized MethodHandleConstant.
     */
    public MethodHandleConstant()
    {
    }


    /**
     * Creates a new MethodHandleConstant with the given type and method ref
     * index.
     * @param u1referenceKind  the reference kind.
     * @param u2referenceIndex the index of the field ref constant, interface
     *                         method ref constant, or method ref constant in
     *                         the constant pool.
     */
    public MethodHandleConstant(int u1referenceKind, int u2referenceIndex)
    {
        this.u1referenceKind  = u1referenceKind;
        this.u2referenceIndex = u2referenceIndex;
    }


    /**
     * Returns the kind of reference to which this constant is pointing.
     * @return One of
     *         {@link ClassConstants#REF_getField        },
     *         {@link ClassConstants#REF_getStatic       },
     *         {@link ClassConstants#REF_putField        },
     *         {@link ClassConstants#REF_putStatic       },
     *         {@link ClassConstants#REF_invokeVirtual   },
     *         {@link ClassConstants#REF_invokeStatic    },
     *         {@link ClassConstants#REF_invokeSpecial   },
     *         {@link ClassConstants#REF_newInvokeSpecial}, or
     *         {@link ClassConstants#REF_invokeInterface }.
     */
    public int getReferenceKind()
    {
        return u1referenceKind;
    }

    /**
     * Returns the field ref, interface method ref, or method ref index.
     */
    public int getReferenceIndex()
    {
        return u2referenceIndex;
    }


    /**
     * Returns the class name.
     */
    public String getClassName(Clazz clazz)
    {
        return clazz.getRefClassName(u2referenceIndex);
    }

    /**
     * Returns the method/field name.
     */
    public String getName(Clazz clazz)
    {
        return clazz.getRefName(u2referenceIndex);
    }

    /**
     * Returns the type.
     */
    public String getType(Clazz clazz)
    {
        return clazz.getRefType(u2referenceIndex);
    }


    /**
     * Applies the given constant pool visitor to the reference.
     */
    public void referenceAccept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        clazz.constantPoolEntryAccept(u2referenceIndex,
                                      constantVisitor);
    }


    // Implementations for Constant.

    public int getTag()
    {
        return ClassConstants.CONSTANT_MethodHandle;
    }

    public void accept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        constantVisitor.visitMethodHandleConstant(clazz, this);
    }
}

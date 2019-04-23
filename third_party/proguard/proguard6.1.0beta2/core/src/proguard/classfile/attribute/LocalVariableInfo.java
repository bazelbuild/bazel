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
package proguard.classfile.attribute;

import proguard.classfile.*;
import proguard.classfile.visitor.ClassVisitor;

/**
 * Representation of an Local Variable table entry.
 *
 * @author Eric Lafortune
 */
public class LocalVariableInfo implements VisitorAccepter, Comparable
{
    public int u2startPC;
    public int u2length;
    public int u2nameIndex;
    public int u2descriptorIndex;
    public int u2index;

    /**
     * An extra field pointing to the referenced Clazz object.
     * This field is typically filled out by the <code>{@link
     * proguard.classfile.util.ClassReferenceInitializer
     * ClassReferenceInitializer}</code>.
     */
    public Clazz referencedClass;

    /**
     * An extra field in which visitors can store information.
     */
    public Object visitorInfo;


    /**
     * Creates an uninitialized LocalVariableInfo.
     */
    public LocalVariableInfo()
    {
    }


    /**
     * Creates an initialized LocalVariableInfo.
     */
    public LocalVariableInfo(int   u2startPC,
                             int   u2length,
                             int   u2nameIndex,
                             int   u2descriptorIndex,
                             int   u2index)
    {
        this.u2startPC         = u2startPC;
        this.u2length          = u2length;
        this.u2nameIndex       = u2nameIndex;
        this.u2descriptorIndex = u2descriptorIndex;
        this.u2index           = u2index;
    }


    /**
     * Returns the name.
     */
    public String getName(Clazz clazz)
    {
        return clazz.getString(u2nameIndex);
    }


    /**
     * Returns the descriptor.
     */
    public String getDescriptor(Clazz clazz)
    {
        return clazz.getString(u2descriptorIndex);
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


    // Implementations for VisitorAccepter.

    public Object getVisitorInfo()
    {
        return visitorInfo;
    }

    public void setVisitorInfo(Object visitorInfo)
    {
        this.visitorInfo = visitorInfo;
    }


    // Implementations for Comparable.

    public int compareTo(Object object)
    {
        LocalVariableInfo other = (LocalVariableInfo)object;

        return
            this.u2startPC          < other.u2startPC          ? -1 : this.u2startPC          > other.u2startPC          ? 1 :
            this.u2index            < other.u2index            ? -1 : this.u2index            > other.u2index            ? 1 :
            this.u2length           < other.u2length           ? -1 : this.u2length           > other.u2length           ? 1 :
            this.u2descriptorIndex  < other.u2descriptorIndex  ? -1 : this.u2descriptorIndex  > other.u2descriptorIndex  ? 1 :
            this.u2nameIndex        < other.u2nameIndex        ? -1 : this.u2nameIndex        > other.u2nameIndex        ? 1 :
                                                                                                                           0;
    }
}

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
package proguard.classfile.attribute.preverification;

import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.attribute.preverification.visitor.StackMapFrameVisitor;

/**
 * This abstract class represents a stack map frame. Specific types
 * of entries are subclassed from it.
 *
 * @author Eric Lafortune
 */
public abstract class StackMapFrame implements VisitorAccepter
{
    public static final int SAME_ZERO_FRAME          =   0;
    public static final int SAME_ONE_FRAME           =  64;
    public static final int SAME_ONE_FRAME_EXTENDED  = 247;
    public static final int LESS_ZERO_FRAME          = 248;
    public static final int SAME_ZERO_FRAME_EXTENDED = 251;
    public static final int MORE_ZERO_FRAME          = 252;
    public static final int FULL_FRAME               = 255;


    public int u2offsetDelta;

    /**
     * An extra field in which visitors can store information.
     */
    public Object visitorInfo;



    /**
     * Returns the bytecode offset delta relative to the previous stack map
     * frame.
     */
    public int getOffsetDelta()
    {
        return u2offsetDelta;
    }


    // Abstract methods to be implemented by extensions.

    /**
     * Returns the stack map frame tag that specifies the entry type.
     */
    public abstract int getTag();


    /**
     * Accepts the given visitor.
     */
    public abstract void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, StackMapFrameVisitor stackMapFrameVisitor);


    // Implementations for VisitorAccepter.

    public Object getVisitorInfo()
    {
        return visitorInfo;
    }

    public void setVisitorInfo(Object visitorInfo)
    {
        this.visitorInfo = visitorInfo;
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        if (object == null ||
            this.getClass() != object.getClass())
        {
            return false;
        }

        StackMapFrame other = (StackMapFrame)object;

        return this.u2offsetDelta == other.u2offsetDelta;
    }


    public int hashCode()
    {
        return getClass().hashCode() ^
               u2offsetDelta;
    }


    public String toString()
    {
        return "[" + u2offsetDelta + "] ";
    }
}

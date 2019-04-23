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
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.preverification.visitor.StackMapFrameVisitor;
import proguard.classfile.attribute.visitor.AttributeVisitor;

/**
 * This Attribute represents an exceptions attribute.
 *
 * @author Eric Lafortune
 */
public class StackMapAttribute extends Attribute
{
    public int         u2stackMapFramesCount;
    public FullFrame[] stackMapFrames;


    /**
     * Creates an uninitialized ExceptionsAttribute.
     */
    public StackMapAttribute()
    {
    }


    /**
     * Creates a StackMapTableAttribute with the given stack map frames.
     */
    public StackMapAttribute(FullFrame[] stackMapFrames)
    {
        this(stackMapFrames.length, stackMapFrames);
    }


    /**
     * Creates a StackMapTableAttribute with the given stack map frames.
     */
    public StackMapAttribute(int         stackMapFramesCount,
                             FullFrame[] stackMapFrames)
    {
        this.u2stackMapFramesCount = stackMapFramesCount;
        this.stackMapFrames        = stackMapFrames;
    }


    // Implementations for Attribute.

    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, AttributeVisitor attributeVisitor)
    {
        attributeVisitor.visitStackMapAttribute(clazz, method, codeAttribute, this);
    }


    /**
     * Applies the given stack map frame visitor to all stack map frames.
     */
    public void stackMapFramesAccept(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapFrameVisitor stackMapFrameVisitor)
    {
        for (int index = 0; index < u2stackMapFramesCount; index++)
        {
            FullFrame stackMapFrame = stackMapFrames[index];

            // We don't need double dispatching here, since there is only one
            // type of StackMapFrame.
            stackMapFrameVisitor.visitFullFrame(clazz, method, codeAttribute, stackMapFrame.getOffsetDelta(), stackMapFrame);
        }
    }
}

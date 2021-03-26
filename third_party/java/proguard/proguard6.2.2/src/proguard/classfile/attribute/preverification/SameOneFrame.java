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
import proguard.classfile.attribute.preverification.visitor.*;

/**
 * This StackMapFrame represents a "same locals 1 stack item frame" or a
 * "same locals 1 stack item frame extended".
 *
 * @author Eric Lafortune
 */
public class SameOneFrame extends StackMapFrame
{
    public VerificationType stackItem;


    /**
     * Creates an uninitialized SameOneFrame.
     */
    public SameOneFrame()
    {
    }


    /**
     * Creates a SameOneFrame with the given tag.
     */
    public SameOneFrame(int tag)
    {
        u2offsetDelta = tag - SAME_ONE_FRAME;
    }


    /**
     * Creates a SameOneFrame with the given stack verification type.
     */
    public SameOneFrame(VerificationType stackItem)
    {
        this.stackItem = stackItem;
    }


    /**
     * Applies the given verification type visitor to the stack item.
     */
    public void stackItemAccept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VerificationTypeVisitor verificationTypeVisitor)
    {
        stackItem.accept(clazz, method, codeAttribute, offset, verificationTypeVisitor);
    }


    // Implementations for StackMapFrame.

    public int getTag()
    {
        return u2offsetDelta < 64 ?
            SAME_ONE_FRAME + u2offsetDelta :
            SAME_ONE_FRAME_EXTENDED;
    }


    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, StackMapFrameVisitor stackMapFrameVisitor)
    {
        stackMapFrameVisitor.visitSameOneFrame(clazz, method, codeAttribute, offset, this);
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        if (!super.equals(object))
        {
            return false;
        }

        SameOneFrame other = (SameOneFrame)object;

        return this.u2offsetDelta == other.u2offsetDelta &&
               this.stackItem.equals(other.stackItem);
    }


    public int hashCode()
    {
        return super.hashCode() ^ stackItem.hashCode();
    }


    public String toString()
    {
        return super.toString()+"Var: ..., Stack: ["+stackItem.toString()+"]";
    }
}

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
 * This StackMapFrame represents a "full frame".
 *
 * @author Eric Lafortune
 */
public class FullFrame extends StackMapFrame
{
    public int                variablesCount;
    public VerificationType[] variables;
    public int                stackCount;
    public VerificationType[] stack;


    /**
     * Creates an uninitialized FullFrame.
     */
    public FullFrame()
    {
    }


    /**
     * Creates a FullFrame with the given variables and stack.
     */
    public FullFrame(int                offsetDelta,
                     VerificationType[] variables,
                     VerificationType[] stack)
    {
        this(offsetDelta,
             variables.length,
             variables,
             stack.length,
             stack);
    }


    /**
     * Creates a FullFrame with the given variables and stack.
     */
    public FullFrame(int                offsetDelta,
                     int                variablesCount,
                     VerificationType[] variables,
                     int                stackCount,
                     VerificationType[] stack)
    {
        this.u2offsetDelta  = offsetDelta;
        this.variablesCount = variablesCount;
        this.variables      = variables;
        this.stackCount     = stackCount;
        this.stack          = stack;
    }


    /**
     * Applies the given verification type visitor to all variables.
     */
    public void variablesAccept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VerificationTypeVisitor verificationTypeVisitor)
    {
        for (int index = 0; index < variablesCount; index++)
        {
            variables[index].variablesAccept(clazz, method, codeAttribute, offset, index, verificationTypeVisitor);
        }
    }


    /**
     * Applies the given verification type visitor to all stack.
     */
    public void stackAccept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VerificationTypeVisitor verificationTypeVisitor)
    {
        for (int index = 0; index < stackCount; index++)
        {
            stack[index].stackAccept(clazz, method, codeAttribute, offset, index, verificationTypeVisitor);
        }
    }


    // Implementations for StackMapFrame.

    public int getTag()
    {
        return FULL_FRAME;
    }


    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, StackMapFrameVisitor stackMapFrameVisitor)
    {
        stackMapFrameVisitor.visitFullFrame(clazz, method, codeAttribute, offset, this);
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        if (!super.equals(object))
        {
            return false;
        }

        FullFrame other = (FullFrame)object;

        if (this.u2offsetDelta  != other.u2offsetDelta  ||
            this.variablesCount != other.variablesCount ||
            this.stackCount     != other.stackCount)
        {
            return false;
        }

        for (int index = 0; index < variablesCount; index++)
        {
            VerificationType thisType  = this.variables[index];
            VerificationType otherType = other.variables[index];

            if (!thisType.equals(otherType))
            {
                return false;
            }
        }

        for (int index = 0; index < stackCount; index++)
        {
            VerificationType thisType  = this.stack[index];
            VerificationType otherType = other.stack[index];

            if (!thisType.equals(otherType))
            {
                return false;
            }
        }

        return true;
    }


    public int hashCode()
    {
        int hashCode = super.hashCode();

        for (int index = 0; index < variablesCount; index++)
        {
            hashCode ^= variables[index].hashCode();
        }

        for (int index = 0; index < stackCount; index++)
        {
            hashCode ^= stack[index].hashCode();
        }

        return hashCode;
    }


    public String toString()
    {
        StringBuffer buffer = new StringBuffer(super.toString()).append("Var: ");

        for (int index = 0; index < variablesCount; index++)
        {
            buffer = buffer.append('[')
                           .append(variables[index].toString())
                           .append(']');
        }

        buffer.append(", Stack: ");

        for (int index = 0; index < stackCount; index++)
        {
            buffer = buffer.append('[')
                           .append(stack[index].toString())
                           .append(']');
        }

        return buffer.toString();
    }
}

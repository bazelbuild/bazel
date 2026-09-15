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
import proguard.classfile.attribute.preverification.visitor.VerificationTypeVisitor;

/**
 * This VerificationType represents a Uninitialized type.
 *
 * @author Eric Lafortune
 */
public class UninitializedType extends VerificationType
{
    public int u2newInstructionOffset;


    /**
     * Creates an uninitialized UninitializedType.
     */
    public UninitializedType()
    {
    }


    /**
     * Creates an UninitializedType pointing to the given 'new' instruction.
     */
    public UninitializedType(int u2newInstructionOffset)
    {
        this.u2newInstructionOffset = u2newInstructionOffset;
    }


    // Implementations for VerificationType.

    public int getTag()
    {
        return UNINITIALIZED_TYPE;
    }


    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, int instructionOffset, VerificationTypeVisitor verificationTypeVisitor)
    {
        verificationTypeVisitor.visitUninitializedType(clazz, method, codeAttribute, instructionOffset, this);
    }


    public void stackAccept(Clazz clazz, Method method, CodeAttribute codeAttribute, int instructionOffset, int stackIndex, VerificationTypeVisitor verificationTypeVisitor)
    {
        verificationTypeVisitor.visitStackUninitializedType(clazz, method, codeAttribute, instructionOffset, stackIndex, this);
    }


    public void variablesAccept(Clazz clazz, Method method, CodeAttribute codeAttribute, int instructionOffset, int variableIndex, VerificationTypeVisitor verificationTypeVisitor)
    {
        verificationTypeVisitor.visitVariablesUninitializedType(clazz, method, codeAttribute, instructionOffset, variableIndex, this);
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        if (!super.equals(object))
        {
            return false;
        }

        UninitializedType other = (UninitializedType)object;

        return this.u2newInstructionOffset == other.u2newInstructionOffset;
    }


    public int hashCode()
    {
        return super.hashCode() ^
               u2newInstructionOffset;
    }


    public String toString()
    {
        return "u:" + u2newInstructionOffset;
    }
}

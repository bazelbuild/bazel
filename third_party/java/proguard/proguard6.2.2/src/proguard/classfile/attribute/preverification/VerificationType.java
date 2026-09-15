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
 * This abstract class represents a verification type of a local variable or
 * a stack element. Specific verification types are subclassed from it.
 *
 * @author Eric Lafortune
 */
public abstract class VerificationType implements VisitorAccepter
{
    public static final int TOP_TYPE                = 0;
    public static final int INTEGER_TYPE            = 1;
    public static final int FLOAT_TYPE              = 2;
    public static final int DOUBLE_TYPE             = 3;
    public static final int LONG_TYPE               = 4;
    public static final int NULL_TYPE               = 5;
    public static final int UNINITIALIZED_THIS_TYPE = 6;
    public static final int OBJECT_TYPE             = 7;
    public static final int UNINITIALIZED_TYPE      = 8;


    /**
     * An extra field in which visitors can store information.
     */
    public Object visitorInfo;


    /**
     * Returns the tag of the verification type.
     */
    public abstract int getTag();


    /**
     * Accepts the given visitor in the context of a method's code, either on
     * a stack or as a variable.
     */
    public abstract void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, int instructionOffset, VerificationTypeVisitor verificationTypeVisitor);


    /**
     * Accepts the given visitor in the context of a stack in a method's code .
     */
    public abstract void stackAccept(Clazz clazz, Method method, CodeAttribute codeAttribute, int instructionOffset, int stackIndex, VerificationTypeVisitor verificationTypeVisitor);


    /**
     * Accepts the given visitor in the context of a variable in a method's code.
     */
    public abstract void variablesAccept(Clazz clazz, Method method, CodeAttribute codeAttribute, int instructionOffset, int variableIndex, VerificationTypeVisitor verificationTypeVisitor);


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
        return object != null &&
               this.getClass() == object.getClass();
    }


    public int hashCode()
    {
        return this.getClass().hashCode();
    }
}

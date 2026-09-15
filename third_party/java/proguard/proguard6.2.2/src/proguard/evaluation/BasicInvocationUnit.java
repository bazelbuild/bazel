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
package proguard.evaluation;

import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.constant.*;
import proguard.classfile.util.ClassUtil;
import proguard.classfile.visitor.MemberVisitor;
import proguard.evaluation.value.*;

/**
 * This InvocationUnit sets up the variables for entering a method,
 * and it updates the stack for the invocation of a class member,
 * using simple values.
 *
 * @author Eric Lafortune
 */
public class BasicInvocationUnit
extends      SimplifiedInvocationUnit
implements   InvocationUnit,
             MemberVisitor
{
    protected final ValueFactory valueFactory;

    // Field acting as parameter between the visitor methods.
    private Clazz returnTypeClass;


    /**
     * Creates a new BasicInvocationUnit with the given value factory.
     */
    public BasicInvocationUnit(ValueFactory valueFactory)
    {
        this.valueFactory = valueFactory;
    }


    // Implementations for SimplifiedInvocationUnit.

    public Value getExceptionValue(Clazz         clazz,
                                   ClassConstant catchClassConstant)
    {
        String catchClassName = catchClassConstant != null ?
            catchClassConstant.getName(clazz) :
            ClassConstants.NAME_JAVA_LANG_THROWABLE;

        Clazz catchClass = catchClassConstant != null ?
            catchClassConstant.referencedClass :
            null;

        return valueFactory.createReferenceValue(catchClassName,
                                                 catchClass,
                                                 true,
                                                 false);
    }


    public void setFieldClassValue(Clazz          clazz,
                                   RefConstant    refConstant,
                                   ReferenceValue value)
    {
        // We don't care about the new value.
    }


    public Value getFieldClassValue(Clazz       clazz,
                                    RefConstant refConstant,
                                    String      type)
    {
        // Try to figure out the class of the return type.
        returnTypeClass = null;
        refConstant.referencedMemberAccept(this);

        return valueFactory.createValue(type,
                                        returnTypeClass,
                                        true,
                                        true);
    }


    public void setFieldValue(Clazz       clazz,
                              RefConstant refConstant,
                              Value       value)
    {
        // We don't care about the new field value.
    }


    public Value getFieldValue(Clazz       clazz,
                               RefConstant refConstant,
                               String      type)
    {
        // Try to figure out the class of the return type.
        returnTypeClass = null;
        refConstant.referencedMemberAccept(this);

        return valueFactory.createValue(type,
                                        returnTypeClass,
                                        true,
                                        true);
    }


    public void setMethodParameterValue(Clazz       clazz,
                                        RefConstant refConstant,
                                        int         parameterIndex,
                                        Value       value)
    {
        // We don't care about the parameter value.
    }


    public Value getMethodParameterValue(Clazz  clazz,
                                         Method method,
                                         int    parameterIndex,
                                         String type,
                                         Clazz  referencedClass)
    {
        // A "this" parameter can never be null.
        boolean isThis =
            parameterIndex == 0 &&
            (method.getAccessFlags() & ClassConstants.ACC_STATIC) == 0;

        return valueFactory.createValue(type,
                                        referencedClass,
                                        true,
                                        !isThis);
    }


    public void setMethodReturnValue(Clazz  clazz,
                                     Method method,
                                     Value  value)
    {
        // We don't care about the return value.
    }


    public Value getMethodReturnValue(Clazz       clazz,
                                      RefConstant refConstant,
                                      String      type)
    {
        // Try to figure out the class of the return type.
        returnTypeClass = null;
        refConstant.referencedMemberAccept(this);

        return valueFactory.createValue(type,
                                        returnTypeClass,
                                        true,
                                        true);
    }


    /**
     * Returns the return value of the specified method.
     */
    public Value getMethodReturnValue(Clazz                 clazz,
                                      InvokeDynamicConstant invokeDynamicConstant,
                                      String                type)
    {
        // Try to figure out the class of the return type.
        Clazz[] referencedClasses = invokeDynamicConstant.referencedClasses;

        Clazz referencedClass =
            referencedClasses != null &&
            ClassUtil.isInternalClassType(type) ?
                referencedClasses[referencedClasses.length - 1] :
                null;

        return valueFactory.createValue(type,
                                        referencedClass,
                                        true,
                                        true);
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        returnTypeClass = programField.referencedClass;
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        Clazz[] referencedClasses = programMethod.referencedClasses;
        if (referencedClasses != null &&
            ClassUtil.isInternalClassType(programMethod.getDescriptor(programClass)))
        {
            returnTypeClass = referencedClasses[referencedClasses.length - 1];
        }
    }


    public void visitLibraryField(LibraryClass programClass, LibraryField libraryField)
    {
        returnTypeClass = libraryField.referencedClass;
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        Clazz[] referencedClasses = libraryMethod.referencedClasses;
        if (referencedClasses != null &&
            ClassUtil.isInternalClassType(libraryMethod.getDescriptor(libraryClass)))
        {
            returnTypeClass = referencedClasses[referencedClasses.length - 1];
        }
    }
}

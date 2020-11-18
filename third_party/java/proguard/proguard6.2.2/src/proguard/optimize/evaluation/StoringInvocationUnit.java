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
package proguard.optimize.evaluation;

import proguard.classfile.*;
import proguard.classfile.constant.RefConstant;
import proguard.evaluation.BasicInvocationUnit;
import proguard.evaluation.value.*;
import proguard.optimize.KeepMarker;
import proguard.optimize.info.*;

/**
 * This InvocationUnit stores parameter values and return values with the
 * methods that are invoked.
 *
 * @see LoadingInvocationUnit
 * @author Eric Lafortune
 */
public class StoringInvocationUnit
extends      BasicInvocationUnit
{
    private boolean storeFieldValues;
    private boolean storeMethodParameterValues;
    private boolean storeMethodReturnValues;


    /**
     * Creates a new StoringInvocationUnit with the given value factory.
     */
    public StoringInvocationUnit(ValueFactory valueFactory)
    {
        this(valueFactory, true, true, true);
    }


    /**
     * Creates a new StoringInvocationUnit with the given value factory, for
     * storing the specified values.
     */
    public StoringInvocationUnit(ValueFactory valueFactory,
                                 boolean      storeFieldValues,
                                 boolean      storeMethodParameterValues,
                                 boolean      storeMethodReturnValues)
    {
        super(valueFactory);

        this.storeFieldValues           = storeFieldValues;
        this.storeMethodParameterValues = storeMethodParameterValues;
        this.storeMethodReturnValues    = storeMethodReturnValues;
    }


    // Implementations for BasicInvocationUnit.

    public void setFieldClassValue(Clazz          clazz,
                                   RefConstant    refConstant,
                                   ReferenceValue value)
    {
        if (storeFieldValues)
        {
            Member referencedMember = refConstant.referencedMember;
            if (referencedMember != null)
            {
                generalizeFieldClassValue((Field)referencedMember, value);
            }
        }
    }


    public void setFieldValue(Clazz       clazz,
                              RefConstant refConstant,
                              Value       value)
    {
        if (storeFieldValues)
        {
            Member referencedMember = refConstant.referencedMember;
            if (referencedMember != null)
            {
                generalizeFieldValue((Field)referencedMember, value);
            }
        }
    }


    public void setMethodParameterValue(Clazz       clazz,
                                        RefConstant refConstant,
                                        int         parameterIndex,
                                        Value       value)
    {
        if (storeMethodParameterValues)
        {
            Member referencedMember = refConstant.referencedMember;
            if (referencedMember != null)
            {
                generalizeMethodParameterValue((Method)referencedMember,
                                               parameterIndex,
                                               value);
            }
        }
    }


    public void setMethodReturnValue(Clazz  clazz,
                                     Method method,
                                     Value  value)
    {
        if (storeMethodReturnValues)
        {
            generalizeMethodReturnValue(method, value);
        }
    }


    // Small utility methods.

    private static void generalizeFieldClassValue(Field field, ReferenceValue value)
    {
        if (!KeepMarker.isKept(field))
        {
            ProgramFieldOptimizationInfo.getProgramFieldOptimizationInfo(field).generalizeReferencedClass(value);
        }
    }


    public static ReferenceValue getFieldClassValue(Field field)
    {
        return FieldOptimizationInfo.getFieldOptimizationInfo(field).getReferencedClass();
    }


    private static void generalizeFieldValue(Field field, Value value)
    {
        if (!KeepMarker.isKept(field))
        {
            ProgramFieldOptimizationInfo.getProgramFieldOptimizationInfo(field).generalizeValue(value);
        }
    }


    public static Value getFieldValue(Field field)
    {
        return FieldOptimizationInfo.getFieldOptimizationInfo(field).getValue();
    }


    private static void generalizeMethodParameterValue(Method method, int parameterIndex, Value value)
    {
        if (!KeepMarker.isKept(method))
        {
            ProgramMethodOptimizationInfo.getProgramMethodOptimizationInfo(method).generalizeParameterValue(parameterIndex, value);
        }
    }


    public static Value getMethodParameterValue(Method method, int parameterIndex)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).getParameterValue(parameterIndex);
    }


    private static void generalizeMethodReturnValue(Method method, Value value)
    {
        if (!KeepMarker.isKept(method))
        {
            ProgramMethodOptimizationInfo.getProgramMethodOptimizationInfo(method).generalizeReturnValue(value);
        }
    }


    public static Value getMethodReturnValue(Method method)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).getReturnValue();
    }
}

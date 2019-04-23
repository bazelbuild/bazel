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

/**
 * This InvocationUnit loads parameter values and return values that were
 * previously stored with the methods that are invoked.
 *
 * @see StoringInvocationUnit
 * @author Eric Lafortune
 */
public class LoadingInvocationUnit
extends      BasicInvocationUnit
{
    private final boolean loadFieldValues;
    private final boolean loadMethodParameterValues;
    private final boolean loadMethodReturnValues;


    /**
     * Creates a new LoadingInvocationUnit with the given value factory.
     */
    public LoadingInvocationUnit(ValueFactory valueFactory)
    {
        this(valueFactory, true, true, true);
    }


    /**
     * Creates a new LoadingInvocationUnit with the given value factory, for
     * loading the specified values.
     */
    public LoadingInvocationUnit(ValueFactory valueFactory,
                                 boolean      loadFieldValues,
                                 boolean      loadMethodParameterValues,
                                 boolean      loadMethodReturnValues)
    {
        super(valueFactory);

        this.loadFieldValues           = loadFieldValues;
        this.loadMethodParameterValues = loadMethodParameterValues;
        this.loadMethodReturnValues    = loadMethodReturnValues;
    }


    // Implementations for BasicInvocationUnit.

    public Value getFieldClassValue(Clazz       clazz,
                                    RefConstant refConstant,
                                    String      type)
    {
        if (loadFieldValues)
        {
            // Do we know this field?
            Member referencedMember = refConstant.referencedMember;
            if (referencedMember != null)
            {
                // Retrieve the stored field class value.
                ReferenceValue value = StoringInvocationUnit.getFieldClassValue((Field)referencedMember);
                if (value != null)
                {
                    return value;
                }
            }
        }

        return super.getFieldClassValue(clazz, refConstant, type);
    }


    public Value getFieldValue(Clazz       clazz,
                               RefConstant refConstant,
                               String      type)
    {
        if (loadFieldValues)
        {
            // Do we know this field?
            Member referencedMember = refConstant.referencedMember;
            if (referencedMember != null)
            {
                // Retrieve the stored field value.
                Value value = StoringInvocationUnit.getFieldValue((Field)referencedMember);
                if (value != null)
                {
                    return value;
                }
            }
        }

        return super.getFieldValue(clazz, refConstant, type);
    }


    public Value getMethodParameterValue(Clazz  clazz,
                                         Method method,
                                         int    parameterIndex,
                                         String type,
                                         Clazz  referencedClass)
    {
        if (loadMethodParameterValues)
        {
            // Retrieve the stored method parameter value.
            Value value = StoringInvocationUnit.getMethodParameterValue(method, parameterIndex);
            if (value != null)
            {
                return value;
            }
        }

        return super.getMethodParameterValue(clazz,
                                             method,
                                             parameterIndex,
                                             type,
                                             referencedClass);
    }


    public Value getMethodReturnValue(Clazz       clazz,
                                      RefConstant refConstant,
                                      String      type)
    {
        if (loadMethodReturnValues)
        {
            // Do we know this method?
            Member referencedMember = refConstant.referencedMember;
            if (referencedMember != null)
            {
                // Retrieve the stored method return value.
                Value value = StoringInvocationUnit.getMethodReturnValue((Method)referencedMember);
                if (value != null)
                {
                    return value;
                }
            }
        }

        return super.getMethodReturnValue(clazz,
                                          refConstant,
                                          type);
    }
}

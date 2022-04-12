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
package proguard.evaluation.value;

import proguard.classfile.*;
import proguard.classfile.util.ClassUtil;

/**
 * This class provides methods to create and reuse Value objects.
 *
 * @author Eric Lafortune
 */
public class BasicValueFactory
implements   ValueFactory
{
    // Shared copies of Value objects, to avoid creating a lot of objects.
    static final IntegerValue   INTEGER_VALUE   = new UnknownIntegerValue();
    static final LongValue      LONG_VALUE      = new UnknownLongValue();
    static final FloatValue     FLOAT_VALUE     = new UnknownFloatValue();
    static final DoubleValue    DOUBLE_VALUE    = new UnknownDoubleValue();
    static final ReferenceValue REFERENCE_VALUE = new UnknownReferenceValue();


    // Implementations for BasicValueFactory.

    public Value createValue(String  type,
                             Clazz   referencedClass,
                             boolean mayBeExtension,
                             boolean mayBeNull)
    {
        switch (type.charAt(0))
        {
            case ClassConstants.TYPE_VOID:    return null;
            case ClassConstants.TYPE_BOOLEAN:
            case ClassConstants.TYPE_BYTE:
            case ClassConstants.TYPE_CHAR:
            case ClassConstants.TYPE_SHORT:
            case ClassConstants.TYPE_INT:     return createIntegerValue();
            case ClassConstants.TYPE_LONG:    return createLongValue();
            case ClassConstants.TYPE_FLOAT:   return createFloatValue();
            case ClassConstants.TYPE_DOUBLE:  return createDoubleValue();
            default:                          return createReferenceValue(ClassUtil.isInternalArrayType(type) ?
                                                                            type :
                                                                            ClassUtil.internalClassNameFromClassType(type),
                                                                          referencedClass,
                                                                          mayBeExtension,
                                                                          mayBeNull);
        }
    }


    public IntegerValue createIntegerValue()
    {
        return INTEGER_VALUE;
    }


    public IntegerValue createIntegerValue(int value)
    {
        return createIntegerValue();
    }


    public IntegerValue createIntegerValue(int min, int max)
    {
        return createIntegerValue();
    }


    public LongValue createLongValue()
    {
        return LONG_VALUE;
    }


    public LongValue createLongValue(long value)
    {
        return createLongValue();
    }


    public FloatValue createFloatValue()
    {
        return FLOAT_VALUE;
    }


    public FloatValue createFloatValue(float value)
    {
        return createFloatValue();
    }


    public DoubleValue createDoubleValue()
    {
        return DOUBLE_VALUE;
    }


    public DoubleValue createDoubleValue(double value)
    {
        return createDoubleValue();
    }


    public ReferenceValue createReferenceValue()
    {
        return REFERENCE_VALUE;
    }


    public ReferenceValue createReferenceValueNull()
    {
        return REFERENCE_VALUE;
    }


    public ReferenceValue createReferenceValue(String  type,
                                               Clazz   referencedClass,
                                               boolean mayBeExtension,
                                               boolean mayBeNull)
    {
        return createReferenceValue();
    }


    public ReferenceValue createArrayReferenceValue(String       type,
                                                    Clazz        referencedClass,
                                                    IntegerValue arrayLength)
    {
        return createReferenceValue(type, referencedClass, false, false);
    }


    public ReferenceValue createArrayReferenceValue(String       type,
                                                    Clazz        referencedClass,
                                                    IntegerValue arrayLength,
                                                    Value        elementValue)
    {
        return createArrayReferenceValue(type, referencedClass, arrayLength);
    }
}

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
 * Its ReferenceValue objects have types if they represent primitive arrays.
 *
 * @author Eric Lafortune
 */
public class PrimitiveTypedReferenceValueFactory
extends      BasicValueFactory
{
    static final ReferenceValue REFERENCE_VALUE_NULL = new TypedReferenceValue(null, null, false, true);

    // Implementations for BasicValueFactory.


    public ReferenceValue createReferenceValueNull()
    {
        return REFERENCE_VALUE_NULL;
    }


    public ReferenceValue createReferenceValue(String type,
                                               Clazz referencedClass,
                                               boolean mayBeExtension,
                                               boolean mayBeNull)
    {
        return type == null ? REFERENCE_VALUE_NULL :
            !ClassUtil.isInternalArrayType(type) ||
             ClassUtil.isInternalClassType(type) ? REFERENCE_VALUE :
                                                   new TypedReferenceValue(type, referencedClass, mayBeExtension, mayBeNull);
    }


    public ReferenceValue createArrayReferenceValue(String       type,
                                                    Clazz        referencedClass,
                                                    IntegerValue arrayLength)
    {
        return type == null ?
            REFERENCE_VALUE_NULL :
            new ArrayReferenceValue(ClassConstants.TYPE_ARRAY + type,
                                    referencedClass,
                                    false,
                                    arrayLength);
    }

}
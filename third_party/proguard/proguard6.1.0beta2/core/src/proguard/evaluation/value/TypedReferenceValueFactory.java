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

/**
 * This class provides methods to create and reuse Value objects.
 * Its ReferenceValue objects have types.
 *
 * @author Eric Lafortune
 */
public class TypedReferenceValueFactory
extends      BasicValueFactory
{
    static final ReferenceValue REFERENCE_VALUE_NULL                        = new TypedReferenceValue(null, null, false, true);
    static final ReferenceValue REFERENCE_VALUE_JAVA_LANG_OBJECT_MAYBE_NULL = new TypedReferenceValue(ClassConstants.NAME_JAVA_LANG_OBJECT, null, true, true);
    static final ReferenceValue REFERENCE_VALUE_JAVA_LANG_OBJECT_NOT_NULL   = new TypedReferenceValue(ClassConstants.NAME_JAVA_LANG_OBJECT, null, true, false);


    // Implementations for BasicValueFactory.

    public ReferenceValue createReferenceValueNull()
    {
        return REFERENCE_VALUE_NULL;
    }


    public ReferenceValue createReferenceValue(String  type,
                                               Clazz   referencedClass,
                                               boolean mayBeExtension,
                                               boolean mayBeNull)
    {
        return type == null                                       ? REFERENCE_VALUE_NULL                                                      :
               !type.equals(ClassConstants.NAME_JAVA_LANG_OBJECT) ||
               !mayBeExtension                                    ? new TypedReferenceValue(type, referencedClass, mayBeExtension, mayBeNull) :
               mayBeNull                                          ? REFERENCE_VALUE_JAVA_LANG_OBJECT_MAYBE_NULL                               :
                                                                    REFERENCE_VALUE_JAVA_LANG_OBJECT_NOT_NULL;
    }


    public ReferenceValue createArrayReferenceValue(String       type,
                                                    Clazz        referencedClass,
                                                    IntegerValue arrayLength)
    {
        return createArrayReferenceValue(type,
                                         referencedClass,
                                         arrayLength,
                                         createValue(type,
                                                     referencedClass,
                                                     true,
                                                     true));
    }


    public ReferenceValue createArrayReferenceValue(String       type,
                                                    Clazz        referencedClass,
                                                    IntegerValue arrayLength,
                                                    Value        elementValue)
    {
        return createReferenceValue(ClassConstants.TYPE_ARRAY + type,
                                    referencedClass,
                                    false,
                                    false);
    }
}
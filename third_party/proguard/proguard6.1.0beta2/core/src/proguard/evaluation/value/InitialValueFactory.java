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

import proguard.classfile.ClassConstants;

/**
 * This value factory creates initial values for fields and array elements,
 * with the help of a given value factory. Note that this class itself doesn't
 * implement ValueFactory.
 *
 * @author Eric Lafortune
 */
public class InitialValueFactory
{
    private final ValueFactory valueFactory;


    /**
     * Creates a new InitialValueFactory.
     * @param valueFactory the value factory that will actually create the
     *                     values.
     */
    public InitialValueFactory(ValueFactory valueFactory)
    {
        this.valueFactory = valueFactory;
    }


    /**
     * Creates an initial value (0, 0L, 0.0f, 0.0, null) of the given type.
     */
    public Value createValue(String type)
    {
        switch (type.charAt(0))
        {
            case ClassConstants.TYPE_BOOLEAN:
            case ClassConstants.TYPE_BYTE:
            case ClassConstants.TYPE_CHAR:
            case ClassConstants.TYPE_SHORT:
            case ClassConstants.TYPE_INT:
                return valueFactory.createIntegerValue(0);

            case ClassConstants.TYPE_LONG:
                return valueFactory.createLongValue(0L);

            case ClassConstants.TYPE_FLOAT:
                return valueFactory.createFloatValue(0.0f);

            case ClassConstants.TYPE_DOUBLE:
                return valueFactory.createDoubleValue(0.0);

            case ClassConstants.TYPE_CLASS_START:
            case ClassConstants.TYPE_ARRAY:
                return valueFactory.createReferenceValueNull();

            default:
                throw new IllegalArgumentException("Invalid type ["+type+"]");
        }
    }
}

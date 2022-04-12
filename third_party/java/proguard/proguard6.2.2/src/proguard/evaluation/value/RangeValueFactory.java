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

import proguard.classfile.Clazz;

/**
 * This class provides methods to create and reuse Value objects that have
 * known ranges.
 *
 * @author Eric Lafortune
 */
public class RangeValueFactory
extends      ParticularValueFactory
implements   ValueFactory
{
    // Shared copies of Value objects, to avoid creating a lot of objects.
    static final IntegerValue INTEGER_VALUE_BYTE  = new RangeIntegerValue(Byte.MIN_VALUE,      Byte.MAX_VALUE);
    static final IntegerValue INTEGER_VALUE_CHAR  = new RangeIntegerValue(Character.MIN_VALUE, Character.MAX_VALUE);
    static final IntegerValue INTEGER_VALUE_SHORT = new RangeIntegerValue(Short.MIN_VALUE,     Short.MAX_VALUE);


    /**
     * Creates a new RangeValueFactory.
     */
    public RangeValueFactory()
    {
        super();
    }


    /**
     * Creates a new RangeValueFactory that delegates to the given
     * value factory for creating reference values.
     */
    public RangeValueFactory(ValueFactory referenceValueFactory)
    {
        super(referenceValueFactory);
    }


    // Implementations for ValueFactory.

    public IntegerValue createIntegerValue(int min, int max)
    {
        return min == max ?
            new ParticularIntegerValue(min) :
            new RangeIntegerValue(min, max);
    }
}

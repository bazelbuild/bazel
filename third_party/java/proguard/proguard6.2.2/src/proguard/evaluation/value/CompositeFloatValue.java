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

/**
 * This FloatValue represents the result of a binary operation on two float
 * values.
 *
 * @author Eric Lafortune
 */
final class CompositeFloatValue extends SpecificFloatValue
{
    public static final byte ADD       = '+';
    public static final byte SUBTRACT  = '-';
    public static final byte MULTIPLY  = '*';
    public static final byte DIVIDE    = '/';
    public static final byte REMAINDER = '%';


    private final FloatValue floatValue1;
    private final byte       operation;
    private final FloatValue floatValue2;


    /**
     * Creates a new composite float value of the two given float values
     * and the given operation.
     */
    public CompositeFloatValue(FloatValue floatValue1,
                               byte       operation,
                               FloatValue floatValue2)
    {
        this.floatValue1 = floatValue1;
        this.operation   = operation;
        this.floatValue2 = floatValue2;
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        return this == object ||
               super.equals(object) &&
               this.floatValue1.equals(((CompositeFloatValue)object).floatValue1) &&
               this.operation       == ((CompositeFloatValue)object).operation    &&
               this.floatValue2.equals(((CompositeFloatValue)object).floatValue2);
    }


    public int hashCode()
    {
        return super.hashCode() ^
               floatValue1.hashCode() ^
               floatValue2.hashCode();
    }


    public String toString()
    {
        return "("+floatValue1+((char)operation)+floatValue2+")";
    }
}
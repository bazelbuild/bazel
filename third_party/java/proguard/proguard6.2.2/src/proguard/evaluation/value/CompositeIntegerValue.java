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
 * This IntegerValue represents the result of a binary operation on two integer
 * values.
 *
 * @author Eric Lafortune
 */
final class CompositeIntegerValue extends SpecificIntegerValue
{
    public static final byte ADD                  = '+';
    public static final byte SUBTRACT             = '-';
    public static final byte MULTIPLY             = '*';
    public static final byte DIVIDE               = '/';
    public static final byte REMAINDER            = '%';
    public static final byte SHIFT_LEFT           = '<';
    public static final byte SHIFT_RIGHT          = '>';
    public static final byte UNSIGNED_SHIFT_RIGHT = '}';
    public static final byte AND                  = '&';
    public static final byte OR                   = '|';
    public static final byte XOR                  = '^';


    private final IntegerValue integerValue1;
    private final byte         operation;
    private final IntegerValue integerValue2;


    /**
     * Creates a new composite integer value of the two given integer values
     * and the given operation.
     */
    public CompositeIntegerValue(IntegerValue integerValue1,
                                 byte         operation,
                                 IntegerValue integerValue2)
    {
        this.integerValue1 = integerValue1;
        this.operation     = operation;
        this.integerValue2 = integerValue2;
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        return this == object ||
               super.equals(object) &&
               this.integerValue1.equals(((CompositeIntegerValue)object).integerValue1) &&
               this.operation         == ((CompositeIntegerValue)object).operation      &&
               this.integerValue2.equals(((CompositeIntegerValue)object).integerValue2);
    }


    public int hashCode()
    {
        return super.hashCode() ^
               integerValue1.hashCode() ^
               integerValue2.hashCode();
    }


    public String toString()
    {
        return "("+integerValue1+((char)operation)+integerValue2+")";
    }
}
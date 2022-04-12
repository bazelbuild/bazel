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
 * This LongValue represents the result of a binary operation on two long
 * values.
 *
 * @author Eric Lafortune
 */
final class CompositeLongValue extends SpecificLongValue
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


    private final LongValue longValue1;
    private final byte      operation;
    private final Value longValue2;


    /**
     * Creates a new composite long value of the two given long values
     * and the given operation.
     */
    public CompositeLongValue(LongValue longValue1,
                              byte      operation,
                              Value     longValue2)
    {
        this.longValue1 = longValue1;
        this.operation  = operation;
        this.longValue2 = longValue2;
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        return this == object ||
               super.equals(object) &&
               this.longValue1.equals(((CompositeLongValue)object).longValue1) &&
               this.operation      == ((CompositeLongValue)object).operation   &&
               this.longValue2.equals(((CompositeLongValue)object).longValue2);
    }


    public int hashCode()
    {
        return super.hashCode() ^
               longValue1.hashCode() ^
               longValue2.hashCode();
    }


    public String toString()
    {
        return "("+longValue1+((char)operation)+longValue2+")";
    }
}
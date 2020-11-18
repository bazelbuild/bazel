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
 * This DoubleValue represents the result of a binary operation on two double
 * values.
 *
 * @author Eric Lafortune
 */
final class CompositeDoubleValue extends SpecificDoubleValue
{
    public static final byte ADD       = '+';
    public static final byte SUBTRACT  = '-';
    public static final byte MULTIPLY  = '*';
    public static final byte DIVIDE    = '/';
    public static final byte REMAINDER = '%';


    private final DoubleValue doubleValue1;
    private final byte        operation;
    private final DoubleValue doubleValue2;


    /**
     * Creates a new composite double value of the two given double values
     * and the given operation.
     */
    public CompositeDoubleValue(DoubleValue doubleValue1,
                                byte        operation,
                                DoubleValue doubleValue2)
    {
        this.doubleValue1 = doubleValue1;
        this.operation    = operation;
        this.doubleValue2 = doubleValue2;
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        return this == object ||
               super.equals(object) &&
               this.doubleValue1.equals(((CompositeDoubleValue)object).doubleValue1) &&
               this.operation        == ((CompositeDoubleValue)object).operation     &&
               this.doubleValue2.equals(((CompositeDoubleValue)object).doubleValue2);
    }


    public int hashCode()
    {
        return super.hashCode() ^
               doubleValue1.hashCode() ^
               doubleValue2.hashCode();
    }


    public String toString()
    {
        return "("+doubleValue1+((char)operation)+doubleValue2+")";
    }
}
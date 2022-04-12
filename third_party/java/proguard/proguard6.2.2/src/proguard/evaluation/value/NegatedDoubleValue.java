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
 * This DoubleValue represents a double value that is negated.
 *
 * @author Eric Lafortune
 */
final class NegatedDoubleValue extends SpecificDoubleValue
{
    private final DoubleValue doubleValue;


    /**
     * Creates a new negated double value of the given double value.
     */
    public NegatedDoubleValue(DoubleValue doubleValue)
    {
        this.doubleValue = doubleValue;
    }


    // Implementations of unary methods of DoubleValue.

    public DoubleValue negate()
    {
        return doubleValue;
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        return this == object ||
               super.equals(object) &&
               this.doubleValue.equals(((NegatedDoubleValue)object).doubleValue);
    }


    public int hashCode()
    {
        return super.hashCode() ^
               doubleValue.hashCode();
    }


    public String toString()
    {
        return "-"+doubleValue;
    }
}
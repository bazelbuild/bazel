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
 * This IntegerValue represents a integer value that is negated.
 *
 * @author Eric Lafortune
 */
final class NegatedIntegerValue extends SpecificIntegerValue
{
    private final IntegerValue integerValue;


    /**
     * Creates a new negated integer value of the given integer value.
     */
    public NegatedIntegerValue(IntegerValue integerValue)
    {
        this.integerValue = integerValue;
    }


    // Implementations of unary methods of IntegerValue.

    public IntegerValue negate()
    {
        return integerValue;
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        return this == object ||
               super.equals(object) &&
               this.integerValue.equals(((NegatedIntegerValue)object).integerValue);
    }


    public int hashCode()
    {
        return super.hashCode() ^
               integerValue.hashCode();
    }


    public String toString()
    {
        return "-"+integerValue;
    }
}
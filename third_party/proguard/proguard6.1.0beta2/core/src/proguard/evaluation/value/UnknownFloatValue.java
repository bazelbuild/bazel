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
 * This class represents a partially evaluated float value.
 *
 * @author Eric Lafortune
 */
public class UnknownFloatValue extends FloatValue
{
    // Basic unary methods.

    public FloatValue negate()
    {
        return this;
    }

    public IntegerValue convertToInteger()
    {
        return BasicValueFactory.INTEGER_VALUE;
    }

    public LongValue convertToLong()
    {
        return BasicValueFactory.LONG_VALUE;
    }

    public DoubleValue convertToDouble()
    {
        return BasicValueFactory.DOUBLE_VALUE;
    }


    // Basic binary methods.

    public FloatValue generalize(FloatValue other)
    {
        return this;
    }

    public FloatValue add(FloatValue other)
    {
        return this;
    }

    public FloatValue subtract(FloatValue other)
    {
        return this;
    }

    public FloatValue subtractFrom(FloatValue other)
    {
        return this;
    }

    public FloatValue multiply(FloatValue other)
    {
        return this;
    }

    public FloatValue divide(FloatValue other)
    {
        return this;
    }

    public FloatValue divideOf(FloatValue other)
    {
        return this;
    }

    public FloatValue remainder(FloatValue other)
    {
        return this;
    }

    public FloatValue remainderOf(FloatValue other)
    {
        return this;
    }

    public IntegerValue compare(FloatValue other)
    {
        return BasicValueFactory.INTEGER_VALUE;
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        return object != null &&
               this.getClass() == object.getClass();
    }


    public int hashCode()
    {
        return this.getClass().hashCode();
    }


    public String toString()
    {
        return "f";
    }
}
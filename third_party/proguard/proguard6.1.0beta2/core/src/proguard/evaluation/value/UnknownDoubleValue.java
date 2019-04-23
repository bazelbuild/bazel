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
 * This class represents a partially evaluated double value.
 *
 * @author Eric Lafortune
 */
public class UnknownDoubleValue extends DoubleValue
{
    // Basic unary methods.

    public DoubleValue negate()
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

    public FloatValue convertToFloat()
    {
        return BasicValueFactory.FLOAT_VALUE;
    }


    // Basic binary methods.

    public DoubleValue generalize(DoubleValue other)
    {
        return this;
    }

    public DoubleValue add(DoubleValue other)
    {
        return this;
    }

    public DoubleValue subtract(DoubleValue other)
    {
        return this;
    }

    public DoubleValue subtractFrom(DoubleValue other)
    {
        return this;
    }

    public DoubleValue multiply(DoubleValue other)
    {
        return this;
    }

    public DoubleValue divide(DoubleValue other)
    {
        return this;
    }

    public DoubleValue divideOf(DoubleValue other)
    {
        return this;
    }

    public DoubleValue remainder(DoubleValue other)
    {
        return this;
    }

    public DoubleValue remainderOf(DoubleValue other)
    {
        return this;
    }

    public IntegerValue compare(DoubleValue other)
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
        return "d";
    }
}
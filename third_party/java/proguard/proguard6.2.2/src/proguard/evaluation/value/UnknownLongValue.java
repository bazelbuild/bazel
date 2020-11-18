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
 * This class represents a partially evaluated long value.
 *
 * @author Eric Lafortune
 */
public class UnknownLongValue extends LongValue
{
    // Basic unary methods.

    public LongValue negate()
    {
        return this;
    }

    public IntegerValue convertToInteger()
    {
        return BasicValueFactory.INTEGER_VALUE;
    }

    public FloatValue convertToFloat()
    {
        return BasicValueFactory.FLOAT_VALUE;
    }

    public DoubleValue convertToDouble()
    {
        return BasicValueFactory.DOUBLE_VALUE;
    }


    // Basic binary methods.

    public LongValue generalize(LongValue other)
    {
        return this;
    }

    public LongValue add(LongValue other)
    {
        return this;
    }

    public LongValue subtract(LongValue other)
    {
        return this;
    }

    public LongValue subtractFrom(LongValue other)
    {
        return this;
    }

    public LongValue multiply(LongValue other)
    throws ArithmeticException
    {
        return this;
    }

    public LongValue divide(LongValue other)
    throws ArithmeticException
    {
        return this;
    }

    public LongValue divideOf(LongValue other)
    throws ArithmeticException
    {
        return this;
    }

    public LongValue remainder(LongValue other)
    throws ArithmeticException
    {
        return this;
    }

    public LongValue remainderOf(LongValue other)
    throws ArithmeticException
    {
        return this;
    }

    public LongValue shiftLeft(IntegerValue other)
    {
        return this;
    }

    public LongValue shiftRight(IntegerValue other)
    {
        return this;
    }

    public LongValue unsignedShiftRight(IntegerValue other)
    {
        return this;
    }

    public LongValue and(LongValue other)
    {
        return this;
    }

    public LongValue or(LongValue other)
    {
        return this;
    }

    public LongValue xor(LongValue other)
    {
        return this;
    }

    public IntegerValue compare(LongValue other)
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
        return "l";
    }
}
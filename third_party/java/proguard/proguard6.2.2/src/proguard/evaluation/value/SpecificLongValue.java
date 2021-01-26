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
 * This LongValue represents a specific long value.
 *
 * @author Eric Lafortune
 */
abstract class SpecificLongValue extends LongValue
{
    // Implementations of unary methods of LongValue.

    public LongValue negate()
    {
        return new NegatedLongValue(this);
    }

    public IntegerValue convertToInteger()
    {
        return new ConvertedIntegerValue(this);
    }

    public FloatValue convertToFloat()
    {
        return new ConvertedFloatValue(this);
    }

    public DoubleValue convertToDouble()
    {
        return new ConvertedDoubleValue(this);
    }


    // Implementations of binary methods of LongValue.

    public LongValue generalize(LongValue other)
    {
        return other.generalize(this);
    }

    public LongValue add(LongValue other)
    {
        return other.add(this);
    }

    public LongValue subtract(LongValue other)
    {
        return other.subtractFrom(this);
    }

    public LongValue subtractFrom(LongValue other)
    {
        return other.subtract(this);
    }

    public LongValue multiply(LongValue other)
    {
        return other.multiply(this);
    }

    public LongValue divide(LongValue other)
    throws ArithmeticException
    {
        return other.divideOf(this);
    }

    public LongValue divideOf(LongValue other)
    throws ArithmeticException
    {
        return other.divide(this);
    }

    public LongValue remainder(LongValue other)
    throws ArithmeticException
    {
        return other.remainderOf(this);
    }

    public LongValue remainderOf(LongValue other)
    throws ArithmeticException
    {
        return other.remainder(this);
    }

    public LongValue shiftLeft(IntegerValue other)
    {
        return other.shiftLeftOf(this);
    }

    public LongValue shiftRight(IntegerValue other)
    {
        return other.shiftRightOf(this);
    }

    public LongValue unsignedShiftRight(IntegerValue other)
    {
        return other.unsignedShiftRightOf(this);
    }

    public LongValue and(LongValue other)
    {
        return other.and(this);
    }

    public LongValue or(LongValue other)
    {
        return other.or(this);
    }

    public LongValue xor(LongValue other)
    {
        return other.xor(this);
    }

    public IntegerValue compare(LongValue other)
    {
        return other.compareReverse(this);
    }


    // Implementations of binary LongValue methods with SpecificLongValue
    // arguments.

    public LongValue generalize(SpecificLongValue other)
    {
        return this.equals(other) ? this : BasicValueFactory.LONG_VALUE;
    }

    public LongValue add(SpecificLongValue other)
    {
        return new CompositeLongValue(this, CompositeLongValue.ADD, other);
    }

    public LongValue subtract(SpecificLongValue other)
    {
        return this.equals(other) ?
            ParticularValueFactory.LONG_VALUE_0 :
            new CompositeLongValue(this, CompositeLongValue.SUBTRACT, other);
    }

    public LongValue subtractFrom(SpecificLongValue other)
    {
        return this.equals(other) ?
            ParticularValueFactory.LONG_VALUE_0 :
            new CompositeLongValue(other, CompositeLongValue.SUBTRACT, this);
    }

    public LongValue multiply(SpecificLongValue other)
    {
        return new CompositeLongValue(this, CompositeLongValue.MULTIPLY, other);
    }

    public LongValue divide(SpecificLongValue other)
    throws ArithmeticException
    {
        return new CompositeLongValue(this, CompositeLongValue.DIVIDE, other);
    }

    public LongValue divideOf(SpecificLongValue other)
    throws ArithmeticException
    {
        return new CompositeLongValue(other, CompositeLongValue.DIVIDE, this);
    }

    public LongValue remainder(SpecificLongValue other)
    throws ArithmeticException
    {
        return new CompositeLongValue(this, CompositeLongValue.REMAINDER, other);
    }

    public LongValue remainderOf(SpecificLongValue other)
    throws ArithmeticException
    {
        return new CompositeLongValue(other, CompositeLongValue.REMAINDER, this);
    }

    public LongValue shiftLeft(SpecificLongValue other)
    {
        return new CompositeLongValue(this, CompositeLongValue.SHIFT_LEFT, other);
    }

    public LongValue shiftRight(SpecificLongValue other)
    {
        return new CompositeLongValue(this, CompositeLongValue.SHIFT_RIGHT, other);
    }

    public LongValue unsignedShiftRight(SpecificLongValue other)
    {
        return new CompositeLongValue(this, CompositeLongValue.UNSIGNED_SHIFT_RIGHT, other);
    }

    public LongValue and(SpecificLongValue other)
    {
        return this.equals(other) ?
            this :
            new CompositeLongValue(other, CompositeLongValue.AND, this);
    }

    public LongValue or(SpecificLongValue other)
    {
        return this.equals(other) ?
            this :
            new CompositeLongValue(other, CompositeLongValue.OR, this);
    }

    public LongValue xor(SpecificLongValue other)
    {
        return this.equals(other) ?
            ParticularValueFactory.LONG_VALUE_0 :
            new CompositeLongValue(other, CompositeLongValue.XOR, this);
    }

    public IntegerValue compare(SpecificLongValue other)
    {
        return new ComparisonValue(this, other);
    }


    // Implementations for Value.

    public boolean isSpecific()
    {
        return true;
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
}

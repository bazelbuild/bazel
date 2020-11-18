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
 * This LongValue represents a particular long value.
 *
 * @author Eric Lafortune
 */
final class ParticularLongValue extends SpecificLongValue
{
    private final long value;


    /**
     * Creates a new particular long value.
     */
    public ParticularLongValue(long value)
    {
        this.value = value;
    }


    // Implementations for LongValue.

    public long value()
    {
        return value;
    }


    // Implementations of unary methods of LongValue.

    public LongValue negate()
    {
        return new ParticularLongValue(-value);
    }

    public IntegerValue convertToInteger()
    {
        return new ParticularIntegerValue((int)value);
    }

    public FloatValue convertToFloat()
    {
        return new ParticularFloatValue((float)value);
    }

    public DoubleValue convertToDouble()
    {
        return new ParticularDoubleValue((double)value);
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


    // Implementations of binary LongValue methods with ParticularLongValue
    // arguments.

    public LongValue generalize(ParticularLongValue other)
    {
        return generalize((SpecificLongValue)other);
    }

    public LongValue add(ParticularLongValue other)
    {
        return new ParticularLongValue(this.value + other.value);
    }

    public LongValue subtract(ParticularLongValue other)
    {
        return new ParticularLongValue(this.value - other.value);
    }

    public LongValue subtractFrom(ParticularLongValue other)
    {
        return new ParticularLongValue(other.value - this.value);
    }

    public LongValue multiply(ParticularLongValue other)
    {
        return new ParticularLongValue(this.value * other.value);
    }

    public LongValue divide(ParticularLongValue other)
    throws ArithmeticException
    {
        return new ParticularLongValue(this.value / other.value);
    }

    public LongValue divideOf(ParticularLongValue other)
    throws ArithmeticException
    {
        return new ParticularLongValue(other.value / this.value);
    }

    public LongValue remainder(ParticularLongValue other)
    throws ArithmeticException
    {
        return new ParticularLongValue(this.value % other.value);
    }

    public LongValue remainderOf(ParticularLongValue other)
    throws ArithmeticException
    {
        return new ParticularLongValue(other.value % this.value);
    }

    public LongValue shiftLeft(ParticularIntegerValue other)
    {
        return new ParticularLongValue(this.value << other.value());
    }

    public LongValue shiftRight(ParticularIntegerValue other)
    {
        return new ParticularLongValue(this.value >> other.value());
    }

    public LongValue unsignedShiftRight(ParticularIntegerValue other)
    {
        return new ParticularLongValue(this.value >>> other.value());
    }

    public LongValue and(ParticularLongValue other)
    {
        return new ParticularLongValue(this.value & other.value);
    }

    public LongValue or(ParticularLongValue other)
    {
        return new ParticularLongValue(this.value | other.value);
    }

    public LongValue xor(ParticularLongValue other)
    {
        return new ParticularLongValue(this.value ^ other.value);
    }


    // Implementations for Value.

    public boolean isParticular()
    {
        return true;
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        return super.equals(object) &&
               this.value == ((ParticularLongValue)object).value;
    }


    public int hashCode()
    {
        return this.getClass().hashCode() ^
               (int)value;
    }


    public String toString()
    {
        return value+"L";
    }
}
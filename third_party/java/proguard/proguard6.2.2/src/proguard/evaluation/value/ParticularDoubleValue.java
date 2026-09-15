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
 * This DoubleValue represents a particular double value.
 *
 * @author Eric Lafortune
 */
final class ParticularDoubleValue extends SpecificDoubleValue
{
    private final double value;


    /**
     * Creates a new particular double value.
     */
    public ParticularDoubleValue(double value)
    {
        this.value = value;
    }


    // Implementations for DoubleValue.

    public double value()
    {
        return value;
    }


    // Implementations of unary methods of DoubleValue.

    public DoubleValue negate()
    {
        return new ParticularDoubleValue(-value);
    }

    public IntegerValue convertToInteger()
    {
        return new ParticularIntegerValue((int)value);
    }

    public LongValue convertToLong()
    {
        return new ParticularLongValue((long)value);
    }

    public FloatValue convertToFloat()
    {
        return new ParticularFloatValue((float)value);
    }


    // Implementations of binary methods of DoubleValue.

    public DoubleValue generalize(DoubleValue other)
    {
        return other.generalize(this);
    }

    public DoubleValue add(DoubleValue other)
    {
        // Careful: -0.0 + 0.0 == 0.0
        //return value == 0.0 ? other : other.add(this);
        return other.add(this);
    }

    public DoubleValue subtract(DoubleValue other)
    {
        // Careful: -0.0 + 0.0 == 0.0
        //return value == 0.0 ? other.negate() : other.subtractFrom(this);
        return other.subtractFrom(this);
    }

    public DoubleValue subtractFrom(DoubleValue other)
    {
        // Careful: -0.0 + 0.0 == 0.0
        //return value == 0.0 ? other : other.subtract(this);
        return other.subtract(this);
    }

    public DoubleValue multiply(DoubleValue other)
    {
        return other.multiply(this);
    }

    public DoubleValue divide(DoubleValue other)
    {
        return other.divideOf(this);
    }

    public DoubleValue divideOf(DoubleValue other)
    {
        return other.divide(this);
    }

    public DoubleValue remainder(DoubleValue other)
    {
        return other.remainderOf(this);
    }

    public DoubleValue remainderOf(DoubleValue other)
    {
        return other.remainder(this);
    }

    public IntegerValue compare(DoubleValue other)
    {
        return other.compareReverse(this);
    }


    // Implementations of binary DoubleValue methods with ParticularDoubleValue
    // arguments.

    public DoubleValue generalize(ParticularDoubleValue other)
    {
        // Also handle NaN and Infinity.
        return Double.doubleToRawLongBits(this.value) ==
               Double.doubleToRawLongBits(other.value) ?
                   this : BasicValueFactory.DOUBLE_VALUE;
    }

    public DoubleValue add(ParticularDoubleValue other)
    {
        return new ParticularDoubleValue(this.value + other.value);
    }

    public DoubleValue subtract(ParticularDoubleValue other)
    {
        return new ParticularDoubleValue(this.value - other.value);
    }

    public DoubleValue subtractFrom(ParticularDoubleValue other)
    {
        return new ParticularDoubleValue(other.value - this.value);
    }

    public DoubleValue multiply(ParticularDoubleValue other)
    {
        return new ParticularDoubleValue(this.value * other.value);
    }

    public DoubleValue divide(ParticularDoubleValue other)
    {
        return new ParticularDoubleValue(this.value / other.value);
    }

    public DoubleValue divideOf(ParticularDoubleValue other)
    {
        return new ParticularDoubleValue(other.value / this.value);
    }

    public DoubleValue remainder(ParticularDoubleValue other)
    {
        return new ParticularDoubleValue(this.value % other.value);
    }

    public DoubleValue remainderOf(ParticularDoubleValue other)
    {
        return new ParticularDoubleValue(other.value % this.value);
    }

    public IntegerValue compare(ParticularDoubleValue other)
    {
        return this.value <  other.value ? ParticularValueFactory.INTEGER_VALUE_M1 :
               this.value == other.value ? ParticularValueFactory.INTEGER_VALUE_0  :
                                           ParticularValueFactory.INTEGER_VALUE_1;
    }


    // Implementations for Value.

    public boolean isParticular()
    {
        return true;
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
       // Also handle NaN and Infinity.
       return super.equals(object) &&
              Double.doubleToLongBits(this.value) ==
              Double.doubleToLongBits(((ParticularDoubleValue)object).value);
    }


    public int hashCode()
    {
        return super.hashCode() ^
               (int)Double.doubleToLongBits(value);
    }


    public String toString()
    {
        return value+"d";
    }
}
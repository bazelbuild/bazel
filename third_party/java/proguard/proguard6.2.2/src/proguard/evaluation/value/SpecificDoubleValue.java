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
 * This DoubleValue represents a specific double value.
 *
 * @author Eric Lafortune
 */
abstract class SpecificDoubleValue extends DoubleValue
{
    // Implementations of unary methods of DoubleValue.

    public DoubleValue negate()
    {
        return new NegatedDoubleValue(this);
    }

    public IntegerValue convertToInteger()
    {
        return new ConvertedIntegerValue(this);
    }

    public LongValue convertToLong()
    {
        return new ConvertedLongValue(this);
    }

    public FloatValue convertToFloat()
    {
        return new ConvertedFloatValue(this);
    }


    // Implementations of binary methods of DoubleValue.

    public DoubleValue generalize(DoubleValue other)
    {
        return other.generalize(this);
    }

    public DoubleValue add(DoubleValue other)
    {
        return other.add(this);
    }

    public DoubleValue subtract(DoubleValue other)
    {
        return other.subtractFrom(this);
    }

    public DoubleValue subtractFrom(DoubleValue other)
    {
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


    // Implementations of binary DoubleValue methods with SpecificDoubleValue
    // arguments.

    public DoubleValue generalize(SpecificDoubleValue other)
    {
        return this.equals(other) ? this : BasicValueFactory.DOUBLE_VALUE;
    }

    public DoubleValue add(SpecificDoubleValue other)
    {
        return new CompositeDoubleValue(this, CompositeDoubleValue.ADD, other);
    }

    public DoubleValue subtract(SpecificDoubleValue other)
    {
        return new CompositeDoubleValue(this, CompositeDoubleValue.SUBTRACT, other);
    }

    public DoubleValue subtractFrom(SpecificDoubleValue other)
    {
        return new CompositeDoubleValue(other, CompositeDoubleValue.SUBTRACT, this);
    }

    public DoubleValue multiply(SpecificDoubleValue other)
    {
        return new CompositeDoubleValue(this, CompositeDoubleValue.MULTIPLY, other);
    }

    public DoubleValue divide(SpecificDoubleValue other)
    {
        return new CompositeDoubleValue(this, CompositeDoubleValue.DIVIDE, other);
    }

    public DoubleValue divideOf(SpecificDoubleValue other)
    {
        return new CompositeDoubleValue(other, CompositeDoubleValue.DIVIDE, this);
    }

    public DoubleValue remainder(SpecificDoubleValue other)
    {
        return new CompositeDoubleValue(this, CompositeDoubleValue.REMAINDER, other);
    }

    public DoubleValue remainderOf(SpecificDoubleValue other)
    {
        return new CompositeDoubleValue(other, CompositeDoubleValue.REMAINDER, this);
    }

    public IntegerValue compare(SpecificDoubleValue other)
    {
        return BasicValueFactory.INTEGER_VALUE;

        // Not handling NaN properly.
        //return this.equals(other) ?
        //    ParticularValueFactory.INTEGER_VALUE_0 :
        //    new ComparisonValue(this, other);
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

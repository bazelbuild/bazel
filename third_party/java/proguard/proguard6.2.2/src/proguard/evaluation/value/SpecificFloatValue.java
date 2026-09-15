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
 * This FloatValue represents a specific float value.
 *
 * @author Eric Lafortune
 */
abstract class SpecificFloatValue extends FloatValue
{
    // Implementations of unary methods of FloatValue.

    public FloatValue negate()
    {
        return new NegatedFloatValue(this);
    }

    public IntegerValue convertToInteger()
    {
        return new ConvertedIntegerValue(this);
    }

    public LongValue convertToLong()
    {
        return new ConvertedLongValue(this);
    }

    public DoubleValue convertToDouble()
    {
        return new ConvertedDoubleValue(this);
    }


    // Implementations of binary methods of FloatValue.

    public FloatValue generalize(FloatValue other)
    {
        return other.generalize(this);
    }

    public FloatValue add(FloatValue other)
    {
        return other.add(this);
    }

    public FloatValue subtract(FloatValue other)
    {
        return other.subtractFrom(this);
    }

    public FloatValue subtractFrom(FloatValue other)
    {
        return other.subtract(this);
    }

    public FloatValue multiply(FloatValue other)
    {
        return other.multiply(this);
    }

    public FloatValue divide(FloatValue other)
    {
        return other.divideOf(this);
    }

    public FloatValue divideOf(FloatValue other)
    {
        return other.divide(this);
    }

    public FloatValue remainder(FloatValue other)
    {
        return other.remainderOf(this);
    }

    public FloatValue remainderOf(FloatValue other)
    {
        return other.remainder(this);
    }

    public IntegerValue compare(FloatValue other)
    {
        return other.compareReverse(this);
    }


    // Implementations of binary FloatValue methods with SpecificFloatValue
    // arguments.

    public FloatValue generalize(SpecificFloatValue other)
    {
        return this.equals(other) ? this : BasicValueFactory.FLOAT_VALUE;
    }

    public FloatValue add(SpecificFloatValue other)
    {
        return new CompositeFloatValue(this, CompositeFloatValue.ADD, other);
    }

    public FloatValue subtract(SpecificFloatValue other)
    {
        return new CompositeFloatValue(this, CompositeFloatValue.SUBTRACT, other);
    }

    public FloatValue subtractFrom(SpecificFloatValue other)
    {
        return new CompositeFloatValue(other, CompositeFloatValue.SUBTRACT, this);
    }

    public FloatValue multiply(SpecificFloatValue other)
    {
        return new CompositeFloatValue(this, CompositeFloatValue.MULTIPLY, other);
    }

    public FloatValue divide(SpecificFloatValue other)
    {
        return new CompositeFloatValue(this, CompositeFloatValue.DIVIDE, other);
    }

    public FloatValue divideOf(SpecificFloatValue other)
    {
        return new CompositeFloatValue(other, CompositeFloatValue.DIVIDE, this);
    }

    public FloatValue remainder(SpecificFloatValue other)
    {
        return new CompositeFloatValue(this, CompositeFloatValue.REMAINDER, other);
    }

    public FloatValue remainderOf(SpecificFloatValue other)
    {
        return new CompositeFloatValue(other, CompositeFloatValue.REMAINDER, this);
    }

    public IntegerValue compare(SpecificFloatValue other)
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

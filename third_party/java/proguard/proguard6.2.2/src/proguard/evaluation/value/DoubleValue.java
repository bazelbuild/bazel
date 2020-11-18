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

import proguard.classfile.ClassConstants;

/**
 * This class represents a partially evaluated double value.
 *
 * @author Eric Lafortune
 */
public abstract class DoubleValue extends Category2Value
{
    /**
     * Returns the specific double value, if applicable.
     */
    public double value()
    {
        return 0.0;
    }


    // Basic unary methods.

    /**
     * Returns the negated value of this DoubleValue.
     */
    public abstract DoubleValue negate();

    /**
     * Converts this DoubleValue to an IntegerValue.
     */
    public abstract IntegerValue convertToInteger();

    /**
     * Converts this DoubleValue to a LongValue.
     */
    public abstract LongValue convertToLong();

    /**
     * Converts this DoubleValue to a FloatValue.
     */
    public abstract FloatValue convertToFloat();


    // Basic binary methods.

    /**
     * Returns the generalization of this DoubleValue and the given other
     * DoubleValue.
     */
    public abstract DoubleValue generalize(DoubleValue other);


    /**
     * Returns the sum of this DoubleValue and the given DoubleValue.
     */
    public abstract DoubleValue add(DoubleValue other);

    /**
     * Returns the difference of this DoubleValue and the given DoubleValue.
     */
    public abstract DoubleValue subtract(DoubleValue other);

    /**
     * Returns the difference of the given DoubleValue and this DoubleValue.
     */
    public abstract DoubleValue subtractFrom(DoubleValue other);

    /**
     * Returns the product of this DoubleValue and the given DoubleValue.
     */
    public abstract DoubleValue multiply(DoubleValue other);

    /**
     * Returns the quotient of this DoubleValue and the given DoubleValue.
     */
    public abstract DoubleValue divide(DoubleValue other);

    /**
     * Returns the quotient of the given DoubleValue and this DoubleValue.
     */
    public abstract DoubleValue divideOf(DoubleValue other);

    /**
     * Returns the remainder of this DoubleValue divided by the given DoubleValue.
     */
    public abstract DoubleValue remainder(DoubleValue other);

    /**
     * Returns the remainder of the given DoubleValue divided by this DoubleValue.
     */
    public abstract DoubleValue remainderOf(DoubleValue other);

    /**
     * Returns an IntegerValue with value -1, 0, or 1, if this DoubleValue is
     * less than, equal to, or greater than the given DoubleValue, respectively.
     */
    public abstract IntegerValue compare(DoubleValue other);


    // Derived binary methods.

    /**
     * Returns an IntegerValue with value 1, 0, or -1, if this DoubleValue is
     * less than, equal to, or greater than the given DoubleValue, respectively.
     */
    public final IntegerValue compareReverse(DoubleValue other)
    {
        return compare(other).negate();
    }


    // Similar binary methods, but this time with more specific arguments.

    /**
     * Returns the generalization of this DoubleValue and the given other
     * SpecificDoubleValue.
     */
    public DoubleValue generalize(SpecificDoubleValue other)
    {
        return generalize((DoubleValue)other);
    }


    /**
     * Returns the sum of this DoubleValue and the given SpecificDoubleValue.
     */
    public DoubleValue add(SpecificDoubleValue other)
    {
        return add((DoubleValue)other);
    }

    /**
     * Returns the difference of this DoubleValue and the given SpecificDoubleValue.
     */
    public DoubleValue subtract(SpecificDoubleValue other)
    {
        return subtract((DoubleValue)other);
    }

    /**
     * Returns the difference of the given SpecificDoubleValue and this DoubleValue.
     */
    public DoubleValue subtractFrom(SpecificDoubleValue other)
    {
        return subtractFrom((DoubleValue)other);
    }

    /**
     * Returns the product of this DoubleValue and the given SpecificDoubleValue.
     */
    public DoubleValue multiply(SpecificDoubleValue other)
    {
        return multiply((DoubleValue)other);
    }

    /**
     * Returns the quotient of this DoubleValue and the given SpecificDoubleValue.
     */
    public DoubleValue divide(SpecificDoubleValue other)
    {
        return divide((DoubleValue)other);
    }

    /**
     * Returns the quotient of the given SpecificDoubleValue and this
     * DoubleValue.
     */
    public DoubleValue divideOf(SpecificDoubleValue other)
    {
        return divideOf((DoubleValue)other);
    }

    /**
     * Returns the remainder of this DoubleValue divided by the given
     * SpecificDoubleValue.
     */
    public DoubleValue remainder(SpecificDoubleValue other)
    {
        return remainder((DoubleValue)other);
    }

    /**
     * Returns the remainder of the given SpecificDoubleValue and this
     * DoubleValue.
     */
    public DoubleValue remainderOf(SpecificDoubleValue other)
    {
        return remainderOf((DoubleValue)other);
    }

    /**
     * Returns an IntegerValue with value -1, 0, or 1, if this DoubleValue is
     * less than, equal to, or greater than the given SpecificDoubleValue,
     * respectively.
     */
    public IntegerValue compare(SpecificDoubleValue other)
    {
        return compare((DoubleValue)other);
    }


    // Derived binary methods.

    /**
     * Returns an IntegerValue with value 1, 0, or -1, if this DoubleValue is
     * less than, equal to, or greater than the given SpecificDoubleValue,
     * respectively.
     */
    public final IntegerValue compareReverse(SpecificDoubleValue other)
    {
        return compare(other).negate();
    }


    // Similar binary methods, but this time with particular arguments.

    /**
     * Returns the generalization of this DoubleValue and the given other
     * ParticularDoubleValue.
     */
    public DoubleValue generalize(ParticularDoubleValue other)
    {
        return generalize((SpecificDoubleValue)other);
    }


    /**
     * Returns the sum of this DoubleValue and the given ParticularDoubleValue.
     */
    public DoubleValue add(ParticularDoubleValue other)
    {
        return add((SpecificDoubleValue)other);
    }

    /**
     * Returns the difference of this DoubleValue and the given ParticularDoubleValue.
     */
    public DoubleValue subtract(ParticularDoubleValue other)
    {
        return subtract((SpecificDoubleValue)other);
    }

    /**
     * Returns the difference of the given ParticularDoubleValue and this DoubleValue.
     */
    public DoubleValue subtractFrom(ParticularDoubleValue other)
    {
        return subtractFrom((SpecificDoubleValue)other);
    }

    /**
     * Returns the product of this DoubleValue and the given ParticularDoubleValue.
     */
    public DoubleValue multiply(ParticularDoubleValue other)
    {
        return multiply((SpecificDoubleValue)other);
    }

    /**
     * Returns the quotient of this DoubleValue and the given ParticularDoubleValue.
     */
    public DoubleValue divide(ParticularDoubleValue other)
    {
        return divide((SpecificDoubleValue)other);
    }

    /**
     * Returns the quotient of the given ParticularDoubleValue and this
     * DoubleValue.
     */
    public DoubleValue divideOf(ParticularDoubleValue other)
    {
        return divideOf((SpecificDoubleValue)other);
    }

    /**
     * Returns the remainder of this DoubleValue divided by the given
     * ParticularDoubleValue.
     */
    public DoubleValue remainder(ParticularDoubleValue other)
    {
        return remainder((SpecificDoubleValue)other);
    }

    /**
     * Returns the remainder of the given ParticularDoubleValue and this
     * DoubleValue.
     */
    public DoubleValue remainderOf(ParticularDoubleValue other)
    {
        return remainderOf((SpecificDoubleValue)other);
    }

    /**
     * Returns an IntegerValue with value -1, 0, or 1, if this DoubleValue is
     * less than, equal to, or greater than the given ParticularDoubleValue,
     * respectively.
     */
    public IntegerValue compare(ParticularDoubleValue other)
    {
        return compare((SpecificDoubleValue)other);
    }


    // Derived binary methods.

    /**
     * Returns an IntegerValue with value 1, 0, or -1, if this DoubleValue is
     * less than, equal to, or greater than the given ParticularDoubleValue,
     * respectively.
     */
    public final IntegerValue compareReverse(ParticularDoubleValue other)
    {
        return compare(other).negate();
    }


    // Implementations for Value.

    public final DoubleValue doubleValue()
    {
        return this;
    }

    public final Value generalize(Value other)
    {
        return this.generalize(other.doubleValue());
    }

    public final int computationalType()
    {
        return TYPE_DOUBLE;
    }

    public final String internalType()
    {
        return String.valueOf(ClassConstants.TYPE_DOUBLE);
    }
}

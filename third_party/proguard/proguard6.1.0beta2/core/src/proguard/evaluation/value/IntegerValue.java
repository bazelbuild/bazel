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
 * This class represents a partially evaluated integer value.
 *
 * @author Eric Lafortune
 */
public abstract class IntegerValue extends Category1Value
{
    /**
     * Returns the specific integer value, if applicable.
     */
    public int value()
    {
        return 0;
    }


    // Basic unary methods.

    /**
     * Returns the negated value of this IntegerValue.
     */
    public abstract IntegerValue negate();

    /**
     * Converts this IntegerValue to a byte IntegerValue.
     */
    public abstract IntegerValue convertToByte();

    /**
     * Converts this IntegerValue to a character IntegerValue.
     */
    public abstract IntegerValue convertToCharacter();

    /**
     * Converts this IntegerValue to a short IntegerValue.
     */
    public abstract IntegerValue convertToShort();

    /**
     * Converts this IntegerValue to a LongValue.
     */
    public abstract LongValue convertToLong();

    /**
     * Converts this IntegerValue to a FloatValue.
     */
    public abstract FloatValue convertToFloat();

    /**
     * Converts this IntegerValue to a DoubleValue.
     */
    public abstract DoubleValue convertToDouble();


    // Basic binary methods.

    /**
     * Returns the generalization of this IntegerValue and the given other
     * IntegerValue.
     */
    public abstract IntegerValue generalize(IntegerValue other);

    /**
     * Returns the sum of this IntegerValue and the given IntegerValue.
     */
    public abstract IntegerValue add(IntegerValue other);

    /**
     * Returns the difference of this IntegerValue and the given IntegerValue.
     */
    public abstract IntegerValue subtract(IntegerValue other);

    /**
     * Returns the difference of the given IntegerValue and this IntegerValue.
     */
    public abstract IntegerValue subtractFrom(IntegerValue other);

    /**
     * Returns the product of this IntegerValue and the given IntegerValue.
     */
    public abstract IntegerValue multiply(IntegerValue other)
    throws ArithmeticException;

    /**
     * Returns the quotient of this IntegerValue and the given IntegerValue.
     */
    public abstract IntegerValue divide(IntegerValue other)
    throws ArithmeticException;

    /**
     * Returns the quotient of the given IntegerValue and this IntegerValue.
     */
    public abstract IntegerValue divideOf(IntegerValue other)
    throws ArithmeticException;

    /**
     * Returns the remainder of this IntegerValue divided by the given
     * IntegerValue.
     */
    public abstract IntegerValue remainder(IntegerValue other)
    throws ArithmeticException;

    /**
     * Returns the remainder of the given IntegerValue divided by this
     * IntegerValue.
     */
    public abstract IntegerValue remainderOf(IntegerValue other)
    throws ArithmeticException;

    /**
     * Returns this IntegerValue, shifted left by the given IntegerValue.
     */
    public abstract IntegerValue shiftLeft(IntegerValue other);

    /**
     * Returns this IntegerValue, shifted right by the given IntegerValue.
     */
    public abstract IntegerValue shiftRight(IntegerValue other);

    /**
     * Returns this unsigned IntegerValue, shifted left by the given
     * IntegerValue.
     */
    public abstract IntegerValue unsignedShiftRight(IntegerValue other);

    /**
     * Returns the given IntegerValue, shifted left by this IntegerValue.
     */
    public abstract IntegerValue shiftLeftOf(IntegerValue other);

    /**
     * Returns the given IntegerValue, shifted right by this IntegerValue.
     */
    public abstract IntegerValue shiftRightOf(IntegerValue other);

    /**
     * Returns the given unsigned IntegerValue, shifted left by this
     * IntegerValue.
     */
    public abstract IntegerValue unsignedShiftRightOf(IntegerValue other);

    /**
     * Returns the given LongValue, shifted left by this IntegerValue.
     */
    public abstract LongValue shiftLeftOf(LongValue other);

    /**
     * Returns the given LongValue, shifted right by this IntegerValue.
     */
    public abstract LongValue shiftRightOf(LongValue other);

    /**
     * Returns the given unsigned LongValue, shifted right by this IntegerValue.
     */
    public abstract LongValue unsignedShiftRightOf(LongValue other);

    /**
     * Returns the logical <i>and</i> of this IntegerValue and the given
     * IntegerValue.
     */
    public abstract IntegerValue and(IntegerValue other);

    /**
     * Returns the logical <i>or</i> of this IntegerValue and the given
     * IntegerValue.
     */
    public abstract IntegerValue or(IntegerValue other);

    /**
     * Returns the logical <i>xor</i> of this IntegerValue and the given
     * IntegerValue.
     */
    public abstract IntegerValue xor(IntegerValue other);

    /**
     * Returns whether this IntegerValue and the given IntegerValue are equal:
     * <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public abstract int equal(IntegerValue other);

    /**
     * Returns whether this IntegerValue is less than the given IntegerValue:
     * <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public abstract int lessThan(IntegerValue other);

    /**
     * Returns whether this IntegerValue is less than or equal to the given
     * IntegerValue: <code>NEVER</code>, <code>MAYBE</code>, or
     * <code>ALWAYS</code>.
     */
    public abstract int lessThanOrEqual(IntegerValue other);


    // Derived binary methods.

    /**
     * Returns whether this IntegerValue and the given IntegerValue are different:
     * <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public final int notEqual(IntegerValue other)
    {
        return -equal(other);
    }

    /**
     * Returns whether this IntegerValue is greater than the given IntegerValue:
     * <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public final int greaterThan(IntegerValue other)
    {
        return -lessThanOrEqual(other);
    }

    /**
     * Returns whether this IntegerValue is greater than or equal to the given IntegerValue:
     * <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public final int greaterThanOrEqual(IntegerValue other)
    {
        return -lessThan(other);
    }


    // Similar binary methods, but this time with unknown arguments.

    /**
     * Returns the generalization of this IntegerValue and the given other
     * UnknownIntegerValue.
     */
    public IntegerValue generalize(UnknownIntegerValue other)
    {
        return generalize((IntegerValue)other);
    }


    /**
     * Returns the sum of this IntegerValue and the given UnknownIntegerValue.
     */
    public IntegerValue add(UnknownIntegerValue other)
    {
        return add((IntegerValue)other);
    }

    /**
     * Returns the difference of this IntegerValue and the given UnknownIntegerValue.
     */
    public IntegerValue subtract(UnknownIntegerValue other)
    {
        return subtract((IntegerValue)other);
    }

    /**
     * Returns the difference of the given UnknownIntegerValue and this IntegerValue.
     */
    public IntegerValue subtractFrom(UnknownIntegerValue other)
    {
        return subtractFrom((IntegerValue)other);
    }

    /**
     * Returns the product of this IntegerValue and the given UnknownIntegerValue.
     */
    public IntegerValue multiply(UnknownIntegerValue other)
    {
        return multiply((IntegerValue)other);
    }

    /**
     * Returns the quotient of this IntegerValue and the given
     * UnknownIntegerValue.
     */
    public IntegerValue divide(UnknownIntegerValue other)
    {
        return divide((IntegerValue)other);
    }

    /**
     * Returns the quotient of the given UnknownIntegerValue and this
     * IntegerValue.
     */
    public IntegerValue divideOf(UnknownIntegerValue other)
    {
        return divideOf((IntegerValue)other);
    }

    /**
     * Returns the remainder of this IntegerValue divided by the given
     * UnknownIntegerValue.
     */
    public IntegerValue remainder(UnknownIntegerValue other)
    {
        return remainder((IntegerValue)other);
    }

    /**
     * Returns the remainder of the given UnknownIntegerValue divided by this
     * IntegerValue.
     */
    public IntegerValue remainderOf(UnknownIntegerValue other)
    {
        return remainderOf((IntegerValue)other);
    }

    /**
     * Returns this IntegerValue, shifted left by the given UnknownIntegerValue.
     */
    public IntegerValue shiftLeft(UnknownIntegerValue other)
    {
        return shiftLeft((IntegerValue)other);
    }

    /**
     * Returns this IntegerValue, shifted right by the given UnknownIntegerValue.
     */
    public IntegerValue shiftRight(UnknownIntegerValue other)
    {
        return shiftRight((IntegerValue)other);
    }

    /**
     * Returns this unsigned IntegerValue, shifted right by the given
     * UnknownIntegerValue.
     */
    public IntegerValue unsignedShiftRight(UnknownIntegerValue other)
    {
        return unsignedShiftRight((IntegerValue)other);
    }

    /**
     * Returns the given UnknownIntegerValue, shifted left by this IntegerValue.
     */
    public IntegerValue shiftLeftOf(UnknownIntegerValue other)
    {
        return shiftLeftOf((IntegerValue)other);
    }

    /**
     * Returns the given UnknownIntegerValue, shifted right by this IntegerValue.
     */
    public IntegerValue shiftRightOf(UnknownIntegerValue other)
    {
        return shiftRightOf((IntegerValue)other);
    }

    /**
     * Returns the given unsigned UnknownIntegerValue, shifted right by this
     * IntegerValue.
     */
    public IntegerValue unsignedShiftRightOf(UnknownIntegerValue other)
    {
        return unsignedShiftRightOf((IntegerValue)other);
    }

    /**
     * Returns the given UnknownLongValue, shifted left by this IntegerValue.
     */
    public LongValue shiftLeftOf(UnknownLongValue other)
    {
        return shiftLeftOf((LongValue)other);
    }

    /**
     * Returns the given UnknownLongValue, shifted right by this IntegerValue.
     */
    public LongValue shiftRightOf(UnknownLongValue other)
    {
        return shiftRightOf((LongValue)other);
    }

    /**
     * Returns the given unsigned UnknownLongValue, shifted right by this
     * IntegerValue.
     */
    public LongValue unsignedShiftRightOf(UnknownLongValue other)
    {
        return unsignedShiftRightOf((LongValue)other);
    }

    /**
     * Returns the logical <i>and</i> of this IntegerValue and the given
     * UnknownIntegerValue.
     */
    public IntegerValue and(UnknownIntegerValue other)
    {
        return and((IntegerValue)other);
    }

    /**
     * Returns the logical <i>or</i> of this IntegerValue and the given
     * UnknownIntegerValue.
     */
    public IntegerValue or(UnknownIntegerValue other)
    {
        return or((IntegerValue)other);
    }

    /**
     * Returns the logical <i>xor</i> of this IntegerValue and the given
     * UnknownIntegerValue.
     */
    public IntegerValue xor(UnknownIntegerValue other)
    {
        return xor((IntegerValue)other);
    }

    /**
     * Returns whether this IntegerValue and the given UnknownIntegerValue are
     * equal: <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public int equal(UnknownIntegerValue other)
    {
        return equal((IntegerValue)other);
    }

    /**
     * Returns whether this IntegerValue is less than the given
     * UnknownIntegerValue: <code>NEVER</code>, <code>MAYBE</code>, or
     * <code>ALWAYS</code>.
     */
    public int lessThan(UnknownIntegerValue other)
    {
        return lessThan((IntegerValue)other);
    }

    /**
     * Returns whether this IntegerValue is less than or equal to the given
     * UnknownIntegerValue: <code>NEVER</code>, <code>MAYBE</code>, or
     * <code>ALWAYS</code>.
     */
    public int lessThanOrEqual(UnknownIntegerValue other)
    {
        return lessThanOrEqual((IntegerValue)other);
    }


    // Derived binary methods.

    /**
     * Returns whether this IntegerValue and the given UnknownIntegerValue are
     * different: <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public final int notEqual(UnknownIntegerValue other)
    {
        return -equal(other);
    }

    /**
     * Returns whether this IntegerValue is greater than the given
     * UnknownIntegerValue: <code>NEVER</code>, <code>MAYBE</code>, or
     * <code>ALWAYS</code>.
     */
    public final int greaterThan(UnknownIntegerValue other)
    {
        return -lessThanOrEqual(other);
    }

    /**
     * Returns whether this IntegerValue is greater than or equal to the given
     * UnknownIntegerValue: <code>NEVER</code>, <code>MAYBE</code>, or
     * <code>ALWAYS</code>.
     */
    public final int greaterThanOrEqual(UnknownIntegerValue other)
    {
        return -lessThan(other);
    }


    // Similar binary methods, but this time with specific arguments.

    /**
     * Returns the generalization of this IntegerValue and the given other
     * SpecificIntegerValue.
     */
    public IntegerValue generalize(SpecificIntegerValue other)
    {
        return generalize((IntegerValue)other);
    }


    /**
     * Returns the sum of this IntegerValue and the given SpecificIntegerValue.
     */
    public IntegerValue add(SpecificIntegerValue other)
    {
        return add((IntegerValue)other);
    }

    /**
     * Returns the difference of this IntegerValue and the given SpecificIntegerValue.
     */
    public IntegerValue subtract(SpecificIntegerValue other)
    {
        return subtract((IntegerValue)other);
    }

    /**
     * Returns the difference of the given SpecificIntegerValue and this IntegerValue.
     */
    public IntegerValue subtractFrom(SpecificIntegerValue other)
    {
        return subtractFrom((IntegerValue)other);
    }

    /**
     * Returns the product of this IntegerValue and the given SpecificIntegerValue.
     */
    public IntegerValue multiply(SpecificIntegerValue other)
    {
        return multiply((IntegerValue)other);
    }

    /**
     * Returns the quotient of this IntegerValue and the given
     * SpecificIntegerValue.
     */
    public IntegerValue divide(SpecificIntegerValue other)
    {
        return divide((IntegerValue)other);
    }

    /**
     * Returns the quotient of the given SpecificIntegerValue and this
     * IntegerValue.
     */
    public IntegerValue divideOf(SpecificIntegerValue other)
    {
        return divideOf((IntegerValue)other);
    }

    /**
     * Returns the remainder of this IntegerValue divided by the given
     * SpecificIntegerValue.
     */
    public IntegerValue remainder(SpecificIntegerValue other)
    {
        return remainder((IntegerValue)other);
    }

    /**
     * Returns the remainder of the given SpecificIntegerValue divided by this
     * IntegerValue.
     */
    public IntegerValue remainderOf(SpecificIntegerValue other)
    {
        return remainderOf((IntegerValue)other);
    }

    /**
     * Returns this IntegerValue, shifted left by the given SpecificIntegerValue.
     */
    public IntegerValue shiftLeft(SpecificIntegerValue other)
    {
        return shiftLeft((IntegerValue)other);
    }

    /**
     * Returns this IntegerValue, shifted right by the given SpecificIntegerValue.
     */
    public IntegerValue shiftRight(SpecificIntegerValue other)
    {
        return shiftRight((IntegerValue)other);
    }

    /**
     * Returns this unsigned IntegerValue, shifted right by the given
     * SpecificIntegerValue.
     */
    public IntegerValue unsignedShiftRight(SpecificIntegerValue other)
    {
        return unsignedShiftRight((IntegerValue)other);
    }

    /**
     * Returns the given SpecificIntegerValue, shifted left by this IntegerValue.
     */
    public IntegerValue shiftLeftOf(SpecificIntegerValue other)
    {
        return shiftLeftOf((IntegerValue)other);
    }

    /**
     * Returns the given SpecificIntegerValue, shifted right by this IntegerValue.
     */
    public IntegerValue shiftRightOf(SpecificIntegerValue other)
    {
        return shiftRightOf((IntegerValue)other);
    }

    /**
     * Returns the given unsigned SpecificIntegerValue, shifted right by this
     * IntegerValue.
     */
    public IntegerValue unsignedShiftRightOf(SpecificIntegerValue other)
    {
        return unsignedShiftRightOf((IntegerValue)other);
    }

    /**
     * Returns the given SpecificLongValue, shifted left by this IntegerValue.
     */
    public LongValue shiftLeftOf(SpecificLongValue other)
    {
        return shiftLeftOf((LongValue)other);
    }

    /**
     * Returns the given SpecificLongValue, shifted right by this IntegerValue.
     */
    public LongValue shiftRightOf(SpecificLongValue other)
    {
        return shiftRightOf((LongValue)other);
    }

    /**
     * Returns the given unsigned SpecificLongValue, shifted right by this
     * IntegerValue.
     */
    public LongValue unsignedShiftRightOf(SpecificLongValue other)
    {
        return unsignedShiftRightOf((LongValue)other);
    }

    /**
     * Returns the logical <i>and</i> of this IntegerValue and the given
     * SpecificIntegerValue.
     */
    public IntegerValue and(SpecificIntegerValue other)
    {
        return and((IntegerValue)other);
    }

    /**
     * Returns the logical <i>or</i> of this IntegerValue and the given
     * SpecificIntegerValue.
     */
    public IntegerValue or(SpecificIntegerValue other)
    {
        return or((IntegerValue)other);
    }

    /**
     * Returns the logical <i>xor</i> of this IntegerValue and the given
     * SpecificIntegerValue.
     */
    public IntegerValue xor(SpecificIntegerValue other)
    {
        return xor((IntegerValue)other);
    }

    /**
     * Returns whether this IntegerValue and the given SpecificIntegerValue are
     * equal: <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public int equal(SpecificIntegerValue other)
    {
        return equal((IntegerValue)other);
    }

    /**
     * Returns whether this IntegerValue is less than the given
     * SpecificIntegerValue: <code>NEVER</code>, <code>MAYBE</code>, or
     * <code>ALWAYS</code>.
     */
    public int lessThan(SpecificIntegerValue other)
    {
        return lessThan((IntegerValue)other);
    }

    /**
     * Returns whether this IntegerValue is less than or equal to the given
     * SpecificIntegerValue: <code>NEVER</code>, <code>MAYBE</code>, or
     * <code>ALWAYS</code>.
     */
    public int lessThanOrEqual(SpecificIntegerValue other)
    {
        return lessThanOrEqual((IntegerValue)other);
    }


    // Derived binary methods.

    /**
     * Returns whether this IntegerValue and the given SpecificIntegerValue are
     * different: <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public final int notEqual(SpecificIntegerValue other)
    {
        return -equal(other);
    }

    /**
     * Returns whether this IntegerValue is greater than the given
     * SpecificIntegerValue: <code>NEVER</code>, <code>MAYBE</code>, or
     * <code>ALWAYS</code>.
     */
    public final int greaterThan(SpecificIntegerValue other)
    {
        return -lessThanOrEqual(other);
    }

    /**
     * Returns whether this IntegerValue is greater than or equal to the given
     * SpecificIntegerValue: <code>NEVER</code>, <code>MAYBE</code>, or
     * <code>ALWAYS</code>.
     */
    public final int greaterThanOrEqual(SpecificIntegerValue other)
    {
        return -lessThan(other);
    }


    // Similar binary methods, but this time with particular arguments.

    /**
     * Returns the generalization of this IntegerValue and the given other
     * ParticularIntegerValue.
     */
    public IntegerValue generalize(ParticularIntegerValue other)
    {
        return generalize((SpecificIntegerValue)other);
    }


    /**
     * Returns the sum of this IntegerValue and the given ParticularIntegerValue.
     */
    public IntegerValue add(ParticularIntegerValue other)
    {
        return add((SpecificIntegerValue)other);
    }

    /**
     * Returns the difference of this IntegerValue and the given ParticularIntegerValue.
     */
    public IntegerValue subtract(ParticularIntegerValue other)
    {
        return subtract((SpecificIntegerValue)other);
    }

    /**
     * Returns the difference of the given ParticularIntegerValue and this IntegerValue.
     */
    public IntegerValue subtractFrom(ParticularIntegerValue other)
    {
        return subtractFrom((SpecificIntegerValue)other);
    }

    /**
     * Returns the product of this IntegerValue and the given ParticularIntegerValue.
     */
    public IntegerValue multiply(ParticularIntegerValue other)
    {
        return multiply((SpecificIntegerValue)other);
    }

    /**
     * Returns the quotient of this IntegerValue and the given
     * ParticularIntegerValue.
     */
    public IntegerValue divide(ParticularIntegerValue other)
    {
        return divide((SpecificIntegerValue)other);
    }

    /**
     * Returns the quotient of the given ParticularIntegerValue and this
     * IntegerValue.
     */
    public IntegerValue divideOf(ParticularIntegerValue other)
    {
        return divideOf((SpecificIntegerValue)other);
    }

    /**
     * Returns the remainder of this IntegerValue divided by the given
     * ParticularIntegerValue.
     */
    public IntegerValue remainder(ParticularIntegerValue other)
    {
        return remainder((SpecificIntegerValue)other);
    }

    /**
     * Returns the remainder of the given ParticularIntegerValue divided by this
     * IntegerValue.
     */
    public IntegerValue remainderOf(ParticularIntegerValue other)
    {
        return remainderOf((SpecificIntegerValue)other);
    }

    /**
     * Returns this IntegerValue, shifted left by the given ParticularIntegerValue.
     */
    public IntegerValue shiftLeft(ParticularIntegerValue other)
    {
        return shiftLeft((SpecificIntegerValue)other);
    }

    /**
     * Returns this IntegerValue, shifted right by the given ParticularIntegerValue.
     */
    public IntegerValue shiftRight(ParticularIntegerValue other)
    {
        return shiftRight((SpecificIntegerValue)other);
    }

    /**
     * Returns this unsigned IntegerValue, shifted right by the given
     * ParticularIntegerValue.
     */
    public IntegerValue unsignedShiftRight(ParticularIntegerValue other)
    {
        return unsignedShiftRight((SpecificIntegerValue)other);
    }

    /**
     * Returns the given ParticularIntegerValue, shifted left by this IntegerValue.
     */
    public IntegerValue shiftLeftOf(ParticularIntegerValue other)
    {
        return shiftLeftOf((SpecificIntegerValue)other);
    }

    /**
     * Returns the given ParticularIntegerValue, shifted right by this IntegerValue.
     */
    public IntegerValue shiftRightOf(ParticularIntegerValue other)
    {
        return shiftRightOf((SpecificIntegerValue)other);
    }

    /**
     * Returns the given unsigned ParticularIntegerValue, shifted right by this
     * IntegerValue.
     */
    public IntegerValue unsignedShiftRightOf(ParticularIntegerValue other)
    {
        return unsignedShiftRightOf((SpecificIntegerValue)other);
    }

    /**
     * Returns the given ParticularLongValue, shifted left by this IntegerValue.
     */
    public LongValue shiftLeftOf(ParticularLongValue other)
    {
        return shiftLeftOf((SpecificLongValue)other);
    }

    /**
     * Returns the given ParticularLongValue, shifted right by this IntegerValue.
     */
    public LongValue shiftRightOf(ParticularLongValue other)
    {
        return shiftRightOf((SpecificLongValue)other);
    }

    /**
     * Returns the given unsigned ParticularLongValue, shifted right by this
     * IntegerValue.
     */
    public LongValue unsignedShiftRightOf(ParticularLongValue other)
    {
        return unsignedShiftRightOf((SpecificLongValue)other);
    }

    /**
     * Returns the logical <i>and</i> of this IntegerValue and the given
     * ParticularIntegerValue.
     */
    public IntegerValue and(ParticularIntegerValue other)
    {
        return and((SpecificIntegerValue)other);
    }

    /**
     * Returns the logical <i>or</i> of this IntegerValue and the given
     * ParticularIntegerValue.
     */
    public IntegerValue or(ParticularIntegerValue other)
    {
        return or((SpecificIntegerValue)other);
    }

    /**
     * Returns the logical <i>xor</i> of this IntegerValue and the given
     * ParticularIntegerValue.
     */
    public IntegerValue xor(ParticularIntegerValue other)
    {
        return xor((SpecificIntegerValue)other);
    }

    /**
     * Returns whether this IntegerValue and the given ParticularIntegerValue are
     * equal: <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public int equal(ParticularIntegerValue other)
    {
        return equal((SpecificIntegerValue)other);
    }

    /**
     * Returns whether this IntegerValue is less than the given
     * ParticularIntegerValue: <code>NEVER</code>, <code>MAYBE</code>, or
     * <code>ALWAYS</code>.
     */
    public int lessThan(ParticularIntegerValue other)
    {
        return lessThan((SpecificIntegerValue)other);
    }

    /**
     * Returns whether this IntegerValue is less than or equal to the given
     * ParticularIntegerValue: <code>NEVER</code>, <code>MAYBE</code>, or
     * <code>ALWAYS</code>.
     */
    public int lessThanOrEqual(ParticularIntegerValue other)
    {
        return lessThanOrEqual((SpecificIntegerValue)other);
    }


    // Derived binary methods.

    /**
     * Returns whether this IntegerValue and the given ParticularIntegerValue are
     * different: <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public final int notEqual(ParticularIntegerValue other)
    {
        return -equal(other);
    }

    /**
     * Returns whether this IntegerValue is greater than the given
     * ParticularIntegerValue: <code>NEVER</code>, <code>MAYBE</code>, or
     * <code>ALWAYS</code>.
     */
    public final int greaterThan(ParticularIntegerValue other)
    {
        return -lessThanOrEqual(other);
    }

    /**
     * Returns whether this IntegerValue is greater than or equal to the given
     * ParticularIntegerValue: <code>NEVER</code>, <code>MAYBE</code>, or
     * <code>ALWAYS</code>.
     */
    public final int greaterThanOrEqual(ParticularIntegerValue other)
    {
        return -lessThan(other);
    }


    // Similar binary methods, but this time with range arguments.

    /**
     * Returns the generalization of this IntegerValue and the given other
     * RangeIntegerValue.
     */
    public IntegerValue generalize(RangeIntegerValue other)
    {
        return generalize((IntegerValue)other);
    }


    /**
     * Returns the sum of this IntegerValue and the given RangeIntegerValue.
     */
    public IntegerValue add(RangeIntegerValue other)
    {
        return add((IntegerValue)other);
    }

    /**
     * Returns the difference of this IntegerValue and the given RangeIntegerValue.
     */
    public IntegerValue subtract(RangeIntegerValue other)
    {
        return subtract((IntegerValue)other);
    }

    /**
     * Returns the difference of the given RangeIntegerValue and this IntegerValue.
     */
    public IntegerValue subtractFrom(RangeIntegerValue other)
    {
        return subtractFrom((IntegerValue)other);
    }

    /**
     * Returns the product of this IntegerValue and the given RangeIntegerValue.
     */
    public IntegerValue multiply(RangeIntegerValue other)
    {
        return multiply((IntegerValue)other);
    }

    /**
     * Returns the quotient of this IntegerValue and the given
     * RangeIntegerValue.
     */
    public IntegerValue divide(RangeIntegerValue other)
    {
        return divide((IntegerValue)other);
    }

    /**
     * Returns the quotient of the given RangeIntegerValue and this
     * IntegerValue.
     */
    public IntegerValue divideOf(RangeIntegerValue other)
    {
        return divideOf((IntegerValue)other);
    }

    /**
     * Returns the remainder of this IntegerValue divided by the given
     * RangeIntegerValue.
     */
    public IntegerValue remainder(RangeIntegerValue other)
    {
        return remainder((IntegerValue)other);
    }

    /**
     * Returns the remainder of the given RangeIntegerValue divided by this
     * IntegerValue.
     */
    public IntegerValue remainderOf(RangeIntegerValue other)
    {
        return remainderOf((IntegerValue)other);
    }

    /**
     * Returns this IntegerValue, shifted left by the given RangeIntegerValue.
     */
    public IntegerValue shiftLeft(RangeIntegerValue other)
    {
        return shiftLeft((IntegerValue)other);
    }

    /**
     * Returns this IntegerValue, shifted right by the given RangeIntegerValue.
     */
    public IntegerValue shiftRight(RangeIntegerValue other)
    {
        return shiftRight((IntegerValue)other);
    }

    /**
     * Returns this unsigned IntegerValue, shifted right by the given
     * RangeIntegerValue.
     */
    public IntegerValue unsignedShiftRight(RangeIntegerValue other)
    {
        return unsignedShiftRight((IntegerValue)other);
    }

    /**
     * Returns the given RangeIntegerValue, shifted left by this IntegerValue.
     */
    public IntegerValue shiftLeftOf(RangeIntegerValue other)
    {
        return shiftLeftOf((IntegerValue)other);
    }

    /**
     * Returns the given RangeIntegerValue, shifted right by this IntegerValue.
     */
    public IntegerValue shiftRightOf(RangeIntegerValue other)
    {
        return shiftRightOf((IntegerValue)other);
    }

    /**
     * Returns the given unsigned RangeIntegerValue, shifted right by this
     * IntegerValue.
     */
    public IntegerValue unsignedShiftRightOf(RangeIntegerValue other)
    {
        return unsignedShiftRightOf((IntegerValue)other);
    }

    /**
     * Returns the logical <i>and</i> of this IntegerValue and the given
     * RangeIntegerValue.
     */
    public IntegerValue and(RangeIntegerValue other)
    {
        return and((IntegerValue)other);
    }

    /**
     * Returns the logical <i>or</i> of this IntegerValue and the given
     * RangeIntegerValue.
     */
    public IntegerValue or(RangeIntegerValue other)
    {
        return or((IntegerValue)other);
    }

    /**
     * Returns the logical <i>xor</i> of this IntegerValue and the given
     * RangeIntegerValue.
     */
    public IntegerValue xor(RangeIntegerValue other)
    {
        return xor((IntegerValue)other);
    }

    /**
     * Returns whether this IntegerValue and the given RangeIntegerValue are
     * equal: <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public int equal(RangeIntegerValue other)
    {
        return equal((IntegerValue)other);
    }

    /**
     * Returns whether this IntegerValue is less than the given
     * RangeIntegerValue: <code>NEVER</code>, <code>MAYBE</code>, or
     * <code>ALWAYS</code>.
     */
    public int lessThan(RangeIntegerValue other)
    {
        return lessThan((IntegerValue)other);
    }

    /**
     * Returns whether this IntegerValue is less than or equal to the given
     * RangeIntegerValue: <code>NEVER</code>, <code>MAYBE</code>, or
     * <code>ALWAYS</code>.
     */
    public int lessThanOrEqual(RangeIntegerValue other)
    {
        return lessThanOrEqual((IntegerValue)other);
    }


    // Derived binary methods.

    /**
     * Returns whether this IntegerValue and the given RangeIntegerValue are
     * different: <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public final int notEqual(RangeIntegerValue other)
    {
        return -equal(other);
    }

    /**
     * Returns whether this IntegerValue is greater than the given
     * RangeIntegerValue: <code>NEVER</code>, <code>MAYBE</code>, or
     * <code>ALWAYS</code>.
     */
    public final int greaterThan(RangeIntegerValue other)
    {
        return -lessThanOrEqual(other);
    }

    /**
     * Returns whether this IntegerValue is greater than or equal to the given
     * RangeIntegerValue: <code>NEVER</code>, <code>MAYBE</code>, or
     * <code>ALWAYS</code>.
     */
    public final int greaterThanOrEqual(RangeIntegerValue other)
    {
        return -lessThan(other);
    }


    // Implementations for Value.

    public final IntegerValue integerValue()
    {
        return this;
    }

    public final Value generalize(Value other)
    {
        return this.generalize(other.integerValue());
    }

    public final int computationalType()
    {
        return TYPE_INTEGER;
    }

    public final String internalType()
    {
        return String.valueOf(ClassConstants.TYPE_INT);
    }
}

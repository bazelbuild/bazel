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
 * This IntegerValue represents a known range of integer values.
 * It currently is not a SpecificIntegerValue, so it doesn't have
 * a particular value nor an identifier.
 *
 * This class handles interactions with:
 * - ParticularIntegerValue
 * - RangeIntegerValue
 *
 * It reverses and delegates interactions with:
 * - IntegerValue (in general)
 *
 * It notably doesn't handle interactions with:
 * - UnknownInteger
 * - SpecificInteger (in general)
 *
 * @author Eric Lafortune
 */
public final class RangeIntegerValue extends IntegerValue
{
    private final int min;
    private final int max;


    /**
     * Creates a new range of integer values.
     */
    public RangeIntegerValue(int min, int max)
    {
        this.min = min;
        this.max = max;
    }


    // Implementations for IntegerValue.

    public int value()
    {
        return min;
    }


    // Implementations of unary methods of IntegerValue.

    public IntegerValue negate()
    {
        return new RangeIntegerValue(
            min == Integer.MIN_VALUE ? Integer.MIN_VALUE :
                -max,
            -min);
    }

    public IntegerValue convertToByte()
    {
        return min >= Byte.MIN_VALUE &&
               max <= Byte.MAX_VALUE ?
            this :
            RangeValueFactory.INTEGER_VALUE_BYTE;
    }

    public IntegerValue convertToCharacter()
    {
        return min >= Character.MIN_VALUE &&
               max <= Character.MAX_VALUE ?
            this :
            RangeValueFactory.INTEGER_VALUE_CHAR;
    }

    public IntegerValue convertToShort()
    {
        return min >= Short.MIN_VALUE &&
               max <= Short.MAX_VALUE ?
            this :
            RangeValueFactory.INTEGER_VALUE_SHORT;
    }

    public LongValue convertToLong()
    {
        return BasicValueFactory.LONG_VALUE;
    }

    public FloatValue convertToFloat()
    {
        return BasicValueFactory.FLOAT_VALUE;
    }

    public DoubleValue convertToDouble()
    {
        return BasicValueFactory.DOUBLE_VALUE;
    }


    // Implementations of binary methods of IntegerValue.

    public IntegerValue generalize(IntegerValue other)
    {
        return other.generalize(this);
    }

    public IntegerValue add(IntegerValue other)
    {
        return other.add(this);
    }

    public IntegerValue subtract(IntegerValue other)
    {
        return other.subtractFrom(this);
    }

    public IntegerValue subtractFrom(IntegerValue other)
    {
        return other.subtract(this);
    }

    public IntegerValue multiply(IntegerValue other)
    {
        return other.multiply(this);
    }

    public IntegerValue divide(IntegerValue other)
    throws ArithmeticException
    {
        return other.divideOf(this);
    }

    public IntegerValue divideOf(IntegerValue other)
    throws ArithmeticException
    {
        return other.divide(this);
    }

    public IntegerValue remainder(IntegerValue other)
    throws ArithmeticException
    {
        return other.remainderOf(this);
    }

    public IntegerValue remainderOf(IntegerValue other)
    throws ArithmeticException
    {
        return other.remainder(this);
    }

    public IntegerValue shiftLeft(IntegerValue other)
    {
        return other.shiftLeftOf(this);
    }

    public IntegerValue shiftLeftOf(IntegerValue other)
    {
        return other.shiftLeft(this);
    }

    public IntegerValue shiftRight(IntegerValue other)
    {
        return other.shiftRightOf(this);
    }

    public IntegerValue shiftRightOf(IntegerValue other)
    {
        return other.shiftRight(this);
    }

    public IntegerValue unsignedShiftRight(IntegerValue other)
    {
        return other.unsignedShiftRightOf(this);
    }

    public IntegerValue unsignedShiftRightOf(IntegerValue other)
    {
        return other.unsignedShiftRight(this);
    }

    public LongValue shiftLeftOf(LongValue other)
    {
        return other.shiftLeft(this);
    }

    public LongValue shiftRightOf(LongValue other)
    {
        return other.shiftRight(this);
    }

    public LongValue unsignedShiftRightOf(LongValue other)
    {
        return other.unsignedShiftRight(this);
    }

    public IntegerValue and(IntegerValue other)
    {
        return other.and(this);
    }

    public IntegerValue or(IntegerValue other)
    {
        return other.or(this);
    }

    public IntegerValue xor(IntegerValue other)
    {
        return other.xor(this);
    }

    public int equal(IntegerValue other)
    {
        return other.equal(this);
    }

    public int lessThan(IntegerValue other)
    {
        return other.greaterThan(this);
    }

    public int lessThanOrEqual(IntegerValue other)
    {
        return other.greaterThanOrEqual(this);
    }


    // Implementations of binary IntegerValue methods with ParticularIntegerValue
    // arguments.

    public IntegerValue generalize(ParticularIntegerValue other)
    {
        // Extend the range if necessary.
        int value = other.value();
        return value < min ? new RangeIntegerValue(value, max) :
               value > max ? new RangeIntegerValue(min, value) :
                             this;
    }

    public IntegerValue add(ParticularIntegerValue other)
    {
        int value = other.value();
        return
            // Check simple cases.
            value == 0 ? this :
            // Check for overflow.
            (value > 0 ? max + value < max :
                         min + value > min) ?
                BasicValueFactory.INTEGER_VALUE :
            // Transform the range.
            new RangeIntegerValue(min + value, max + value);
    }

    public IntegerValue subtract(ParticularIntegerValue other)
    {
        int value = other.value();
        return
            // Check simple cases.
            value == 0 ? this :
            // Check for overflow.
            (value < 0 ? max - value < max :
                         min - value > min) ?
                BasicValueFactory.INTEGER_VALUE :
            // Transform the range.
            new RangeIntegerValue(min - value, max - value);
    }

    public IntegerValue subtractFrom(ParticularIntegerValue other)
    {
        int value = other.value();
        return
            // Check for overflow.
            (long)value - (long)max != (long)(value - max) ||
            (long)value - (long)min != (long)(value - min) ?
                BasicValueFactory.INTEGER_VALUE :
            // Transform the range.
            new RangeIntegerValue(value - max, value - min);
    }

    public IntegerValue multiply(ParticularIntegerValue other)
    {
        int value = other.value();
        return
            // Check simple cases.
            value == 0 ? other :
            value == 1 ? this  :
            // Check for overflow.
            (long)min * (long)value != (long)(min * value) ||
            (long)max * (long)value != (long)(max * value) ?
                BasicValueFactory.INTEGER_VALUE :
            // Transform the range.
            // Check if the interval is inverted.
            value < 0 ?
                new RangeIntegerValue(max * value, min * value) :
                new RangeIntegerValue(min * value, max * value);
    }

    public IntegerValue divide(ParticularIntegerValue other)
    throws ArithmeticException
    {
        int value = other.value();
        return
            // Check simple cases.
            value == 0 ? BasicValueFactory.INTEGER_VALUE :
            value == 1 ? this  :
            // Check for overflow.
            (long)min / (long)value != (long)(min / value) ||
            (long)max / (long)value != (long)(max / value) ?
                BasicValueFactory.INTEGER_VALUE :
            // Transform the range.
            // Check if the interval is inverted.
            value < 0 ?
                new RangeIntegerValue(max / value, min / value) :
                new RangeIntegerValue(min / value, max / value);
    }

    public IntegerValue divideOf(ParticularIntegerValue other)
    throws ArithmeticException
    {
        int value = other.value();
        return
            // Check simple cases.
            min <= 0 &&
            max >= 0 ? BasicValueFactory.INTEGER_VALUE :
            // Check for overflow.
            (long)value / (long)min != (long)(value / min) ||
            (long)value / (long)max != (long)(value / max) ?
                BasicValueFactory.INTEGER_VALUE :
            // Transform the range.
            // Check if the interval is inverted.
            value < 0 ^ min < 0  ?
                new RangeIntegerValue(value / min, value / max) :
                new RangeIntegerValue(value / max, value / min);
    }

    public IntegerValue remainder(ParticularIntegerValue other)
    throws ArithmeticException
    {
        int value = other.value();
        return
            // Check difficult cases.
            value <= 0 ||
            min < 0     ? BasicValueFactory.INTEGER_VALUE :
            // Check simple cases.
            max < value ? this :
                          new RangeIntegerValue(0, value - 1);
    }

    public IntegerValue remainderOf(ParticularIntegerValue other)
    throws ArithmeticException
    {
        int value = other.value();
        return
            // Check difficult cases.
            value < 0 ||
            min <= 0    ? BasicValueFactory.INTEGER_VALUE :
            // Check simple cases.
            value < min ? other :
            value < max ? new RangeIntegerValue(0, value) :
                          new RangeIntegerValue(0, max - 1);
    }

    public IntegerValue shiftLeft(ParticularIntegerValue other)
    {
        int value = other.value();
        return
            // Check simple cases.
            (value & 0x1f) == 0 ? this :
            // Check for overflow.
            (long)min << value != (long)(min << value) ||
            (long)max << value != (long)(max << value) ?
                BasicValueFactory.INTEGER_VALUE :
                new RangeIntegerValue(min << value, max << value);
    }

    public IntegerValue shiftRight(ParticularIntegerValue other)
    {
        int value = other.value();
        return
            // Check simple cases.
            (value & 0x1f) == 0 ? this :
                new RangeIntegerValue(min >> value, max >> value);
    }

    public IntegerValue unsignedShiftRight(ParticularIntegerValue other)
    {
        int value = other.value();
        return
            // Check simple cases.
            (value & 0x1f) == 0 ? this :
            min < 0 ?
            max > 0 ?
                // The negative-to-positive case.
                new RangeIntegerValue(0, Integer.MIN_VALUE >>> value) :
                // The all-negative case.
                new RangeIntegerValue(max >>> value, min >>> value)   :
                // The all-positive case.
                new RangeIntegerValue(min >>> value, max >>> value);
    }

    public IntegerValue shiftLeftOf(ParticularIntegerValue other)
    {
        int value = other.value();
        return
            // Check simple cases.
            value == 0 ? other :
            // Check for overflow.
            min <   0 ||
            max >= 32 ||
            (long)value << max != (long)(value << max) ?
                BasicValueFactory.INTEGER_VALUE :
            value < 0 ?
                new RangeIntegerValue(value << max, value << min) :
                new RangeIntegerValue(value << min, value << max);
    }

    public IntegerValue shiftRightOf(ParticularIntegerValue other)
    {
        int value = other.value();
        return
            // Check simple cases.
            value == 0 ? other :
            // Check for overflow.
            min <   0 ||
            max >= 32 ?
                BasicValueFactory.INTEGER_VALUE :
            value < 0 ?
                new RangeIntegerValue(value >> min, value >> max) :
                new RangeIntegerValue(value >> max, value >> min);
    }

    public IntegerValue unsignedShiftRightOf(ParticularIntegerValue other)
    {
        int value = other.value();
        return
            // Check simple cases.
            value == 0 ? other :
            // Check for overflow.
            min <   0 ||
            max >= 32 ?
                BasicValueFactory.INTEGER_VALUE :
                new RangeIntegerValue(value >>> max, value >>> min);
    }

    public LongValue shiftLeftOf(ParticularLongValue other)
    {
        long value = other.value();
        return
            // Check simple cases.
            value == 0L ? other :
                          BasicValueFactory.LONG_VALUE;
    }

    public LongValue shiftRightOf(ParticularLongValue other)
    {
        long value = other.value();
        return
            // Check simple cases.
            value == 0L ? other :
                          BasicValueFactory.LONG_VALUE;
    }

    public LongValue unsignedShiftRightOf(ParticularLongValue other)
    {
        long value = other.value();
        return
            // Check simple cases.
            value == 0L ? other :
                          BasicValueFactory.LONG_VALUE;
    }

    public IntegerValue and(ParticularIntegerValue other)
    {
        int value = other.value();
        return
            // Check simple cases.
            value ==  0 ? other :
            value == -1 ? this  :
            // Check difficult cases.
            value > 0 ?
                new RangeIntegerValue(0, value) :
                BasicValueFactory.INTEGER_VALUE;
    }

    public IntegerValue or(ParticularIntegerValue other)
    {
        int value = other.value();
        return
            // Check simple cases.
            value ==  0 ? this  :
            value == -1 ? other :
            // Check difficult cases.
            value < 0 ?
                new RangeIntegerValue(value, -1) :
                BasicValueFactory.INTEGER_VALUE;
    }

    public IntegerValue xor(ParticularIntegerValue other)
    {
        int value = other.value();
        return
            // Check simple cases.
            value == 0 ? this :
                         BasicValueFactory.INTEGER_VALUE;
    }

    public int equal(ParticularIntegerValue other)
    {
        int value = other.value();
        return
            min == value &&
            max == value ? ALWAYS :
            value < min ||
            value > max  ? NEVER :
                           MAYBE;
    }

    public int lessThan(ParticularIntegerValue other)
    {
        int value = other.value();
        return
            max < value  ? ALWAYS :
            value <= min ? NEVER  :
                          MAYBE;
    }

    public int lessThanOrEqual(ParticularIntegerValue other)
    {
        int value = other.value();
        return
            max <= value ? ALWAYS :
            value < min  ? NEVER  :
                           MAYBE;
    }


    // Implementations of binary IntegerValue methods with RangeIntegerValue
    // arguments.

    public IntegerValue generalize(RangeIntegerValue other)
    {
        // Extend the range if necessary.
        return
            // Does this range cover the other range?
            this.min <= other.min &&
            this.max >= other.max ? this :
            // Does the other range cover this range?
            other.min <= this.min &&
            other.max >= this.max ? other :
            // Extend the range.
            new RangeIntegerValue(Math.min(this.min, other.min),
                                  Math.max(this.max, other.max));
    }

    public IntegerValue add(RangeIntegerValue other)
    {
        return
            // Check for overflow.
            (long)this.min + (long)other.min != (long)(this.min + other.min) ||
            (long)this.max + (long)other.max != (long)(this.max + other.max) ?
                BasicValueFactory.INTEGER_VALUE :
            // Transform the range.
            new RangeIntegerValue(this.min + other.min,
                                  this.max + other.max);
    }

    public IntegerValue subtract(RangeIntegerValue other)
    {
        return
            // Check for overflow.
            (long)this.min - (long)other.max != (long)(this.min - other.max) ||
            (long)this.max - (long)other.min != (long)(this.max - other.min) ?
                BasicValueFactory.INTEGER_VALUE :
            // Transform the range.
            new RangeIntegerValue(this.min - other.max,
                                  this.max - other.min);
    }

    public IntegerValue subtractFrom(RangeIntegerValue other)
    {
        return
            // Check for overflow.
            (long)other.min - (long)this.max != (long)(other.min - this.max) ||
            (long)other.max - (long)this.min != (long)(other.max - this.min) ?
                BasicValueFactory.INTEGER_VALUE :
            // Transform the range.
            new RangeIntegerValue(other.min - this.max,
                                  other.max - this.min);
    }

    public IntegerValue multiply(RangeIntegerValue other)
    {
        return BasicValueFactory.INTEGER_VALUE;
    }

    public IntegerValue divide(RangeIntegerValue other)
    throws ArithmeticException
    {
        return BasicValueFactory.INTEGER_VALUE;
    }

    public IntegerValue divideOf(RangeIntegerValue other)
    throws ArithmeticException
    {
        return BasicValueFactory.INTEGER_VALUE;
    }

    public IntegerValue remainder(RangeIntegerValue other)
    throws ArithmeticException
    {
        return BasicValueFactory.INTEGER_VALUE;
    }

    public IntegerValue remainderOf(RangeIntegerValue other)
    throws ArithmeticException
    {
        return BasicValueFactory.INTEGER_VALUE;
    }

    public IntegerValue shiftLeft(RangeIntegerValue other)
    {
        return BasicValueFactory.INTEGER_VALUE;
    }

    public IntegerValue shiftRight(RangeIntegerValue other)
    {
        return BasicValueFactory.INTEGER_VALUE;
    }

    public IntegerValue unsignedShiftRight(RangeIntegerValue other)
    {
        return BasicValueFactory.INTEGER_VALUE;
    }

    public IntegerValue shiftLeftOf(RangeIntegerValue other)
    {
        return BasicValueFactory.INTEGER_VALUE;
    }

    public IntegerValue shiftRightOf(RangeIntegerValue other)
    {
        return BasicValueFactory.INTEGER_VALUE;
    }

    public IntegerValue unsignedShiftRightOf(RangeIntegerValue other)
    {
        return BasicValueFactory.INTEGER_VALUE;
    }

    public IntegerValue and(RangeIntegerValue other)
    {
        return BasicValueFactory.INTEGER_VALUE;
    }

    public IntegerValue or(RangeIntegerValue other)
    {
        return BasicValueFactory.INTEGER_VALUE;
    }

    public IntegerValue xor(RangeIntegerValue other)
    {
        return BasicValueFactory.INTEGER_VALUE;
    }

    public int equal(RangeIntegerValue other)
    {
        return
            this.min == this.max  &&
            this.min == other.min &&
            this.min == other.max ? ALWAYS :
            this.max < other.min ||
            other.max < this.min  ? NEVER :
                                    MAYBE;
    }

    public int lessThan(RangeIntegerValue other)
    {
        return
            this.max < other.min  ? ALWAYS :
            other.max <= this.min ? NEVER  :
                                    MAYBE;
    }

    public int lessThanOrEqual(RangeIntegerValue other)
    {
        return
            this.max <= other.min ? ALWAYS :
            other.max < this.min  ? NEVER  :
                                    MAYBE;
    }


    // Implementations for Value.

    public boolean isParticular()
    {
        return min == max;
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        return super.equals(object) &&
               this.min == ((RangeIntegerValue)object).min &&
               this.max == ((RangeIntegerValue)object).max;
    }


    public int hashCode()
    {
        return this.getClass().hashCode() ^
               min ^
               max;
    }


    public String toString()
    {
        return min + ".." + max;
    }
}
/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2017 Eric Lafortune @ GuardSquare
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
 * This class represents a partially evaluated instruction offset. It can
 * contain 0 or more specific instruction offsets.
 *
 * @author Eric Lafortune
 */
public class InstructionOffsetValue extends Category1Value
{
    public static final InstructionOffsetValue EMPTY_VALUE = new InstructionOffsetValue();


    private int[] values;


    private InstructionOffsetValue()
    {
    }


    public InstructionOffsetValue(int value)
    {
        this.values = new int[] { value };
    }


    public InstructionOffsetValue(int[] values)
    {
        this.values = values;
    }


    public int instructionOffsetCount()
    {
        return values == null ? 0 : values.length;
    }


    public int instructionOffset(int index)
    {
        return values[index];
    }


    /**
     * Returns whether the given value is present in this list of instruction
     * offsets.
     */
    public boolean contains(int value)
    {
        if (values != null)
        {
            for (int index = 0; index < values.length; index++)
            {
                if (values[index] == value)
                {
                    return true;
                }
            }
        }

        return false;
    }


    /**
     * Returns the minimum value from this list of instruction offsets.
     * Returns <code>Integer.MAX_VALUE</code> if the list is empty.
     */
    public int minimumValue()
    {
        int minimumValue = Integer.MAX_VALUE;

        if (values != null)
        {
            for (int index = 0; index < values.length; index++)
            {
                int value = values[index];

                if (minimumValue > value)
                {
                    minimumValue = value;
                }
            }
        }

        return minimumValue;
    }


    /**
     * Returns the maximum value from this list of instruction offsets.
     * Returns <code>Integer.MIN_VALUE</code> if the list is empty.
     */
    public int maximumValue()
    {
        int maximumValue = Integer.MIN_VALUE;

        if (values != null)
        {
            for (int index = 0; index < values.length; index++)
            {
                int value = values[index];

                if (maximumValue < value)
                {
                    maximumValue = value;
                }
            }
        }

        return maximumValue;
    }


    /**
     * Returns the generalization of this InstructionOffsetValue and the given
     * other InstructionOffsetValue. The values of the other InstructionOffsetValue
     * are guaranteed to remain at the end of the list, in the same order.
     */
    public final Value generalize(InstructionOffsetValue other)
    {
        // If the values array of either is null, we can return the other one.
        int[] thisValues = this.values;
        if (thisValues == null)
        {
            return other;
        }

        int[] otherValues = other.values;
        if (otherValues == null)
        {
            return this;
        }

        // Compute the length of the union of the arrays.
        int newLength = thisValues.length;
        for (int index = 0; index < otherValues.length; index++)
        {
            if (!this.contains(otherValues[index]))
            {
                newLength++;
            }
        }

        // If the length of the union array is equal to the length of the other
        // values array, we can return it.
        if (newLength == otherValues.length)
        {
            return other;
        }

        // If the length of the union array is equal to the length of this
        // values array, we can return it. We have to make sure that the other
        // values are at the end. We'll just test one special case, with a
        // single other value.
        if (newLength == this.values.length &&
            otherValues.length == 1 &&
            thisValues[thisValues.length-1] == otherValues[0])
        {
            return this;
        }

        // Create the union array.
        int newIndex = 0;
        int[] newValues = new int[newLength];

        // Is the length of the union array is equal to the sum of the lengths?
        if (newLength == thisValues.length + otherValues.length)
        {
            // We can just copy all values, because they are unique.
            System.arraycopy(thisValues, 0, newValues, 0, thisValues.length);

            newIndex = thisValues.length;
        }
        else
        {
            // Copy the values that are different from the other array.
            for (int index = 0; index < thisValues.length; index++)
            {
                if (!other.contains(thisValues[index]))
                {
                    newValues[newIndex++] = thisValues[index];
                }
            }
        }

        // Copy the values from the other array.
        System.arraycopy(otherValues, 0, newValues, newIndex, otherValues.length);

        return new InstructionOffsetValue(newValues);
    }


    // Implementations for Value.

    public final InstructionOffsetValue instructionOffsetValue()
    {
        return this;
    }

    public boolean isSpecific()
    {
        return true;
    }

    public boolean isParticular()
    {
        return true;
    }

    public final Value generalize(Value other)
    {
        return this.generalize(other.instructionOffsetValue());
    }

    public final int computationalType()
    {
        return TYPE_INSTRUCTION_OFFSET;
    }

    public final String internalType()
    {
        return String.valueOf(ClassConstants.TYPE_INT);
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        if (object == null ||
            this.getClass() != object.getClass())
        {
            return false;
        }

        InstructionOffsetValue other = (InstructionOffsetValue)object;
        if (this.values == other.values)
        {
            return true;
        }

        if (this.values  == null ||
            other.values == null ||
            this.values.length != other.values.length)
        {
            return false;
        }

        for (int index = 0; index < other.values.length; index++)
        {
            if (!this.contains(other.values[index]))
            {
                return false;
            }
        }

        return true;
    }


    public int hashCode()
    {
        int hashCode = this.getClass().hashCode();

        if (values != null)
        {
            for (int index = 0; index < values.length; index++)
            {
                hashCode ^= values[index];
            }
        }

        return hashCode;
    }


    public String toString()
    {
        StringBuffer buffer = new StringBuffer();

        if (values != null)
        {
            for (int index = 0; index < values.length; index++)
            {
                if (index > 0)
                {
                    buffer.append(',');
                }
                buffer.append(values[index]);
            }
        }

        return buffer.append(':').toString();
    }
}

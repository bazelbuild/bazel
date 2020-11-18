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
 * This class represents a partially evaluated instruction offset. It can
 * contain 0 or more specific instruction offsets. Each instruction offset
 * can be flagged as an ordinary offset, a method parameter, a method return
 * value, a field value, a new instance value, or an exception handler.
 *
 * @author Eric Lafortune
 */
public class InstructionOffsetValue extends Category1Value
{
    private static final int[]                  EMPTY_OFFSETS = new int[0];
    public  static final InstructionOffsetValue EMPTY_VALUE   = new InstructionOffsetValue(EMPTY_OFFSETS);

    public static final int INSTRUCTION_OFFSET_MASK = 0x01ffffff;
    public static final int METHOD_PARAMETER        = 0x01000000; // Method parameter indices are not really instruction offsets.
    public static final int METHOD_RETURN_VALUE     = 0x02000000;
    public static final int FIELD_VALUE             = 0x04000000;
    public static final int NEW_INSTANCE            = 0x08000000;
    public static final int CAST                    = 0x10000000;
    public static final int EXCEPTION_HANDLER       = 0x20000000;


    private int[] values;


    /**
     * Creates a new InstructionOffsetValue with the given instruction offset.
     */
    public InstructionOffsetValue(int value)
    {
        this.values = new int[] { value };
    }


    /**
     * Creates a new InstructionOffsetValue with the given list of instruction
     * offsets.
     */
    public InstructionOffsetValue(int[] values)
    {
        this.values = values;
    }


    /**
     * Returns the number of instruction offsets of this value.
     */
    public int instructionOffsetCount()
    {
        return values.length;
    }


    /**
     * Returns the specified instruction offset of this value.
     */
    public int instructionOffset(int index)
    {
        return values[index] & INSTRUCTION_OFFSET_MASK;
    }


    /**
     * Returns whether the given value is present in this list of instruction
     * offsets.
     */
    public boolean contains(int value)
    {
        for (int index = 0; index < values.length; index++)
        {
            if (values[index] == value)
            {
                return true;
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

        for (int index = 0; index < values.length; index++)
        {
            int value = values[index] & INSTRUCTION_OFFSET_MASK;

            if (minimumValue > value)
            {
                minimumValue = value;
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

        for (int index = 0; index < values.length; index++)
        {
            int value = values[index] & INSTRUCTION_OFFSET_MASK;

            if (maximumValue < value)
            {
                maximumValue = value;
            }
        }

        return maximumValue;
    }


    /**
     * Returns whether the specified instruction offset corresponds to a method
     * parameter.
     */
    public boolean isMethodParameter(int index)
    {
        return (values[index] & METHOD_PARAMETER) != 0;
    }


    /**
     * Returns the specified method parameter (assuming it is one).
     */
    public int methodParameter(int index)
    {
        return values[index] & ~METHOD_PARAMETER;
    }


    /**
     * Returns whether the specified instruction offset corresponds to a method
     * return value.
     */
    public boolean isMethodReturnValue(int index)
    {
        return (values[index] & METHOD_RETURN_VALUE) != 0;
    }


    /**
     * Returns whether the specified instruction offset corresponds to a field
     * value.
     */
    public boolean isFieldValue(int index)
    {
        return (values[index] & FIELD_VALUE) != 0;
    }


    /**
     * Returns whether the specified instruction offset corresponds to a new
     * instance.
     */
    public boolean isNewinstance(int index)
    {
        return (values[index] & NEW_INSTANCE) != 0;
    }


    /**
     * Returns whether the specified instruction offset corresponds to a cast.
     */
    public boolean isCast(int index)
    {
        return (values[index] & CAST) != 0;
    }


    /**
     * Returns whether the specified instruction offset corresponds to an
     * exception handler.
     */
    public boolean isExceptionHandler(int index)
    {
        return (values[index] & EXCEPTION_HANDLER) != 0;
    }


    /**
     * Returns an InstructionOffsetValue that contains the instructions offsets
     * of this value and the given instruction offset.
     */
    public InstructionOffsetValue add(int value)
    {
        if (contains(value))
        {
            return this;
        }

        int[] newValues = new int[values.length+1];
        System.arraycopy(values, 0, newValues, 0, values.length);
        newValues[values.length] = value;

        return new InstructionOffsetValue(newValues);
    }


    /**
     * Returns an InstructionOffsetValue that contains the instructions offsets
     * of this value but not the given instruction offset.
     */
    public InstructionOffsetValue remove(int value)
    {
        for (int index = 0; index < values.length; index++)
        {
            if (values[index] == value)
            {
                int[] newValues = new int[values.length-1];
                System.arraycopy(values, 0, newValues, 0, index);
                System.arraycopy(values, index+1, newValues, index, values.length-index-1);

                return new InstructionOffsetValue(newValues);
            }
        }

        return this;
    }


    /**
     * Returns the generalization of this InstructionOffsetValue and the given
     * other InstructionOffsetValue. The values of the other InstructionOffsetValue
     * are guaranteed to remain at the end of the list, in the same order.
     */
    public final InstructionOffsetValue generalize(InstructionOffsetValue other)
    {
        // If the values array of either is empty, we can return the other one.
        int[] thisValues = this.values;
        if (thisValues.length == 0)
        {
            return other;
        }

        int[] otherValues = other.values;
        if (otherValues.length == 0)
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

                if (values[index] < 0)
                {
                    buffer.append(values[index]);
                }
                else
                {
                    if (isMethodParameter(index))
                    {
                        buffer.append('P');
                    }

                    if (isMethodReturnValue(index))
                    {
                        buffer.append('M');
                    }

                    if (isFieldValue(index))
                    {
                        buffer.append('F');
                    }

                    if (isNewinstance(index))
                    {
                        buffer.append('N');
                    }

                    if (isCast(index))
                    {
                        buffer.append('C');
                    }

                    if (isExceptionHandler(index))
                    {
                        buffer.append('E');
                    }

                    buffer.append(values[index] & 0xffff);
                }
            }
        }

        return buffer.append(':').toString();
    }
}

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
package proguard.evaluation;

import proguard.evaluation.value.*;

import java.util.Arrays;

/**
 * This class represents a local variable frame that contains <code>Value</code>
 * objects. Values are generalizations of all values that have been stored in
 * the respective variables.
 *
 * @author Eric Lafortune
 */
public class Variables
{
    private static final TopValue TOP_VALUE = new TopValue();


    protected Value[] values;
    protected int     size;


    /**
     * Creates a new Variables object with a given maximum number of variables.
     */
    public Variables(int size)
    {
        this.values = new Value[size];
        this.size   = size;
    }


    /**
     * Creates a Variables object that is a copy of the given Variables object.
     */
    public Variables(Variables variables)
    {
        // Create the values array.
        this(variables.size);

        // Copy the values.
        initialize(variables);
    }


    /**
     * Resets this Variables object, so that it can be reused.
     */
    public void reset(int size)
    {
        // Is the values array large enough?
        if (values.length < size)
        {
            // Create a new one.
            values = new Value[size];
        }
        else
        {
            // Clear the old variables.
            Arrays.fill(values, 0, this.size, null);
        }

        this.size = size;
    }


    /**
     * Initializes the values of this Variables object with the values of the
     * given Variables object. The other object may have fewer values, in which
     * case the remaining values are left unchanged.
     */
    public void initialize(Variables other)
    {
        if (this.size < other.size)
        {
            throw new IllegalArgumentException("Variable frame is too small ["+this.size+"] compared to other frame ["+other.size+"]");
        }

        // Copy the values.
        System.arraycopy(other.values, 0, this.values, 0, other.size);
    }


    /**
     * Generalizes the values of this Variables object with the values of the
     * given Variables object.
     * @param clearConflictingOtherVariables specifies whether the other
     *                                       variables should be cleared too,
     *                                       in case of conflicts.
     * @return whether the generalization has made any difference.
     */
    public boolean generalize(Variables other,
                              boolean   clearConflictingOtherVariables)
    {
        if (this.size != other.size)
        {
            throw new IllegalArgumentException("Variable frames have different sizes ["+this.size+"] and ["+other.size+"]");
        }

        boolean changed = false;

        for (int index = 0; index < size; index++)
        {
            Value thisValue  = this.values[index];
            Value otherValue = other.values[index];

            // Occasionally, two values of different types might be present
            // in the same variable in a variable frame (corresponding to
            // two local variables that share the same index), at some point
            // outside of their scopes. Don't generalize the variable then,
            // but let it clear instead.
            if (thisValue  != null &&
                otherValue != null &&
                thisValue.computationalType() == otherValue.computationalType())
            {
                Value newValue = thisValue.generalize(otherValue);

                changed = changed || !thisValue.equals(newValue);

                this.values[index] = newValue;
            }
            else
            {
                changed = changed || thisValue != null;

                this.values[index] = null;

                if (clearConflictingOtherVariables)
                {
                    other.values[index] = null;
                }
            }
        }

        return changed;
    }


    /**
     * Returns the number of variables.
     */
    public int size()
    {
        return size;
    }


    /**
     * Gets the Value of the variable with the given index, without disturbing it.
     */
    public Value getValue(int index)
    {
        if (index < 0 ||
            index >= size)
        {
            throw new IndexOutOfBoundsException("Variable index ["+index+"] out of bounds ["+size+"]");
        }

        return values[index];
    }


    /**
     * Stores the given Value at the given variable index.
     */
    public void store(int index, Value value)
    {
        if (index < 0 ||
            index >= size)
        {
            throw new IndexOutOfBoundsException("Variable index ["+index+"] out of bounds ["+size+"]");
        }

        // Store the value.
        values[index] = value;

        // Account for the extra space required by Category 2 values.
        if (value.isCategory2())
        {
            values[index + 1] = TOP_VALUE;
        }
    }


    /**
     * Loads the Value from the variable with the given index.
     */
    public Value load(int index)
    {
        if (index < 0 ||
            index >= size)
        {
            throw new IndexOutOfBoundsException("Variable index ["+index+"] out of bounds ["+size+"]");
        }

        return values[index];
    }


    // Load methods that provide convenient casts to the expected value types.

    /**
     * Loads the IntegerValue from the variable with the given index.
     */
    public IntegerValue iload(int index)
    {
        return load(index).integerValue();
    }


    /**
     * Loads the LongValue from the variable with the given index.
     */
    public LongValue lload(int index)
    {
        return load(index).longValue();
    }


    /**
     * Loads the FloatValue from the variable with the given index.
     */
    public FloatValue fload(int index)
    {
        return load(index).floatValue();
    }


    /**
     * Loads the DoubleValue from the variable with the given index.
     */
    public DoubleValue dload(int index)
    {
        return load(index).doubleValue();
    }


    /**
     * Loads the ReferenceValue from the variable with the given index.
     */
    public ReferenceValue aload(int index)
    {
        return load(index).referenceValue();
    }


    /**
     * Loads the InstructionOffsetValue from the variable with the given index.
     */
    public InstructionOffsetValue oload(int index)
    {
        return load(index).instructionOffsetValue();
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        if (object == null ||
            this.getClass() != object.getClass())
        {
            return false;
        }

        Variables other = (Variables)object;

        if (this.size != other.size)
        {
            return false;
        }

        for (int index = 0; index < size; index++)
        {
            Value thisValue  = this.values[index];
            Value otherValue = other.values[index];

            // Occasionally, two values of different types might be
            // present in the same variable in a variable frame
            // (corresponding to two local variables that share the
            // same index), at some point outside of their scopes.
            // We'll ignore these.
            if (thisValue  != null &&
                otherValue != null &&
                thisValue.computationalType() == otherValue.computationalType() &&
                !thisValue.equals(otherValue))
            {
                return false;
            }
        }

        return true;
    }


    public int hashCode()
    {
        int hashCode = size;

        for (int index = 0; index < size; index++)
        {
            Value value = values[index];
            if (value != null)
            {
                hashCode ^= value.hashCode();
            }
        }

        return hashCode;
    }


    public String toString()
    {
        StringBuffer buffer = new StringBuffer();

        for (int index = 0; index < size; index++)
        {
            Value value = values[index];
            buffer = buffer.append('[')
                           .append(value == null ? "empty" : value.toString())
                           .append(']');
        }

        return buffer.toString();
    }
}

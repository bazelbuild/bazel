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
 * This class represents an operand stack that contains <code>Value</code>
 * objects.
 *
 * @author Eric Lafortune
 */
public class Stack
{
    private static final TopValue TOP_VALUE = new TopValue();


    protected Value[] values;
    protected int     currentSize;
    protected int     actualMaxSize;


    /**
     * Creates a new Stack with a given maximum size, accounting for the double
     * space required by Category 2 values.
     */
    public Stack(int maxSize)
    {
        values = new Value[maxSize];
    }


    /**
     * Creates a Stack that is a copy of the given Stack.
     */
    public Stack(Stack stack)
    {
        // Create the values array.
        this(stack.values.length);

        // Copy the stack contents.
        copy(stack);
    }


    /**
     * Returns the actual maximum stack size that was required for all stack
     * operations, accounting for the double space required by Category 2 values.
     */
    public int getActualMaxSize()
    {
        return actualMaxSize;
    }


    /**
     * Resets this Stack, so that it can be reused.
     */
    public void reset(int maxSize)
    {
        // Is the values array large enough?
        if (values.length < maxSize)
        {
            // Create a new one.
            values = new Value[maxSize];
        }

        // Clear the sizes.
        clear();

        actualMaxSize = 0;
    }


    /**
     * Copies the values of the given Stack into this Stack.
     */
    public void copy(Stack other)
    {
        // Is the values array large enough?
        if (values.length < other.values.length)
        {
            // Create a new one.
            values = new Value[other.values.length];
        }

        // Copy the stack contents.
        System.arraycopy(other.values, 0, this.values, 0, other.currentSize);

        // Copy the sizes.
        currentSize   = other.currentSize;
        actualMaxSize = other.actualMaxSize;
    }


    /**
     * Generalizes the values of this Stack with the values of the given Stack.
     * The stacks must have the same current sizes.
     * @return whether the generalization has made any difference.
     */
    public boolean generalize(Stack other)
    {
        if (this.currentSize != other.currentSize)
        {
            throw new IllegalArgumentException("Stacks have different current sizes ["+this.currentSize+"] and ["+other.currentSize+"]");
        }

        boolean changed = false;

        // Generalize the stack values.
        for (int index = 0; index < currentSize; index++)
        {
            Value thisValue  = this.values[index];

            if (thisValue != null)
            {
                Value newValue = null;

                Value otherValue = other.values[index];

                if (otherValue != null)
                {
                    newValue = thisValue.generalize(otherValue);
                }

                changed = changed || !thisValue.equals(newValue);

                values[index] = newValue;
            }
        }

        // Check if the other stack extends beyond this one.
        if (this.actualMaxSize < other.actualMaxSize)
        {
            this.actualMaxSize = other.actualMaxSize;
        }

        return changed;
    }


    /**
     * Clears the stack.
     */
    public void clear()
    {
        // Clear the stack contents.
        Arrays.fill(values, 0, currentSize, null);

        currentSize = 0;
    }


    /**
     * Returns the number of elements currently on the stack, accounting for the
     * double space required by Category 2 values.
     */
    public int size()
    {
        return currentSize;
    }


    /**
     * Gets the specified Value from the stack, without disturbing it.
     * @param index the index of the stack element, counting from the bottom
     *              of the stack.
     * @return the value at the specified position.
     */
    public Value getBottom(int index)
    {
        return values[index];
    }


    /**
     * Sets the specified Value on the stack, without disturbing it.
     * @param index the index of the stack element, counting from the bottom
     *              of the stack.
     * @param value the value to set.
     */
    public void setBottom(int index, Value value)
    {
        values[index] = value;
    }


    /**
     * Gets the specified Value from the stack, without disturbing it.
     * @param index the index of the stack element, counting from the top
     *              of the stack.
     * @return the value at the specified position.
     */
    public Value getTop(int index)
    {
        return values[currentSize - index - 1];
    }


    /**
     * Sets the specified Value on the stack, without disturbing it.
     * @param index the index of the stack element, counting from the top
     *              of the stack.
     * @param value the value to set.
     */
    public void setTop(int index, Value value)
    {
        values[currentSize - index - 1] = value;
    }


    /**
     * Removes the specified Value from the stack.
     * @param index the index of the stack element, counting from the top
     *              of the stack.
     */
    public void removeTop(int index)
    {
        System.arraycopy(values, currentSize - index,
                         values, currentSize - index - 1,
                         index);
        currentSize--;
    }


    /**
     * Pushes the given Value onto the stack.
     */
    public void push(Value value)
    {
        // Account for the extra space required by Category 2 values.
        if (value.isCategory2())
        {
            values[currentSize++] = TOP_VALUE;
        }

        // Push the value.
        values[currentSize++] = value;

        // Update the maximum actual size;
        if (actualMaxSize < currentSize)
        {
            actualMaxSize = currentSize;
        }
    }


    /**
     * Pops the top Value from the stack.
     */
    public Value pop()
    {
        Value value = values[--currentSize];

        values[currentSize] = null;

        // Account for the extra space required by Category 2 values.
        if (value.isCategory2())
        {
            values[--currentSize] = null;
        }

        return value;
    }


    // Pop methods that provide convenient casts to the expected value types.

    /**
     * Pops the top IntegerValue from the stack.
     */
    public IntegerValue ipop()
    {
        return pop().integerValue();
    }


    /**
     * Pops the top LongValue from the stack.
     */
    public LongValue lpop()
    {
        return pop().longValue();
    }


    /**
     * Pops the top FloatValue from the stack.
     */
    public FloatValue fpop()
    {
        return pop().floatValue();
    }


    /**
     * Pops the top DoubleValue from the stack.
     */
    public DoubleValue dpop()
    {
        return pop().doubleValue();
    }


    /**
     * Pops the top ReferenceValue from the stack.
     */
    public ReferenceValue apop()
    {
        return pop().referenceValue();
    }


    /**
     * Pops the top InstructionOffsetValue from the stack.
     */
    public InstructionOffsetValue opop()
    {
        return pop().instructionOffsetValue();
    }


    /**
     * Pops the top category 1 value from the stack.
     */
    public void pop1()
    {
        values[--currentSize] = null;
    }


    /**
     * Pops the top category 2 value from the stack (or alternatively, two
     * Category 1 stack elements).
     */
    public void pop2()
    {
        values[--currentSize] = null;
        values[--currentSize] = null;
    }


    /**
     * Duplicates the top Category 1 value.
     */
    public void dup()
    {
        values[currentSize] = values[currentSize - 1].category1Value();

        currentSize++;

        // Update the maximum actual size;
        if (actualMaxSize < currentSize)
        {
            actualMaxSize = currentSize;
        }
    }


    /**
     * Duplicates the top Category 1 value, one Category 1 element down the
     * stack.
     */
    public void dup_x1()
    {
        values[currentSize]     = values[currentSize - 1].category1Value();
        values[currentSize - 1] = values[currentSize - 2].category1Value();
        values[currentSize - 2] = values[currentSize    ];

        currentSize++;

        // Update the maximum actual size;
        if (actualMaxSize < currentSize)
        {
            actualMaxSize = currentSize;
        }
    }


    /**
     * Duplicates the top Category 1 value, two Category 1 elements (or one
     * Category 2 element) down the stack.
     */
    public void dup_x2()
    {
        values[currentSize]     = values[currentSize - 1].category1Value();
        values[currentSize - 1] = values[currentSize - 2];
        values[currentSize - 2] = values[currentSize - 3];
        values[currentSize - 3] = values[currentSize    ];

        currentSize++;

        // Update the maximum actual size;
        if (actualMaxSize < currentSize)
        {
            actualMaxSize = currentSize;
        }
    }

    /**
     * Duplicates the top Category 2 value (or alternatively, the equivalent
     * Category 1 stack elements).
     */
    public void dup2()
    {
        values[currentSize    ] = values[currentSize - 2];
        values[currentSize + 1] = values[currentSize - 1];

        currentSize += 2;

        // Update the maximum actual size;
        if (actualMaxSize < currentSize)
        {
            actualMaxSize = currentSize;
        }
    }


    /**
     * Duplicates the top Category 2 value, one Category 1 element down the
     * stack (or alternatively, the equivalent Category 1 stack values).
     */
    public void dup2_x1()
    {
        values[currentSize + 1] = values[currentSize - 1];
        values[currentSize    ] = values[currentSize - 2];
        values[currentSize - 1] = values[currentSize - 3];
        values[currentSize - 2] = values[currentSize + 1];
        values[currentSize - 3] = values[currentSize    ];

        currentSize += 2;

        // Update the maximum actual size;
        if (actualMaxSize < currentSize)
        {
            actualMaxSize = currentSize;
        }
    }


    /**
     * Duplicates the top Category 2 value, one Category 2 stack element down
     * the stack (or alternatively, the equivalent Category 1 stack values).
     */
    public void dup2_x2()
    {
        values[currentSize + 1] = values[currentSize - 1];
        values[currentSize    ] = values[currentSize - 2];
        values[currentSize - 1] = values[currentSize - 3];
        values[currentSize - 2] = values[currentSize - 4];
        values[currentSize - 3] = values[currentSize + 1];
        values[currentSize - 4] = values[currentSize    ];

        currentSize += 2;

        // Update the maximum actual size;
        if (actualMaxSize < currentSize)
        {
            actualMaxSize = currentSize;
        }
    }


    /**
     * Swaps the top two Category 1 values.
     */
    public void swap()
    {
        Value value1 = values[currentSize - 1].category1Value();
        Value value2 = values[currentSize - 2].category1Value();

        values[currentSize - 1] = value2;
        values[currentSize - 2] = value1;
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        if (object == null ||
            this.getClass() != object.getClass())
        {
            return false;
        }

        Stack other = (Stack)object;

        if (this.currentSize != other.currentSize)
        {
            return false;
        }

        for (int index = 0; index < currentSize; index++)
        {
            Value thisValue  = this.values[index];
            Value otherValue = other.values[index];
            if (thisValue == null ? otherValue != null :
                                    !thisValue.equals(otherValue))
            {
                return false;
            }
        }

        return true;
    }


    public int hashCode()
    {
        int hashCode = currentSize;

        for (int index = 0; index < currentSize; index++)
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

        for (int index = 0; index < currentSize; index++)
        {
            Value value = values[index];
            buffer = buffer.append('[')
                           .append(value == null ? "empty" : value.toString())
                           .append(']');
        }

        return buffer.toString();
    }
}

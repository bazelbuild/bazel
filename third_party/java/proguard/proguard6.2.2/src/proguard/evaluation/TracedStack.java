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

import proguard.evaluation.value.Value;

/**
 * This Stack saves additional information with stack elements, to keep track
 * of their origins.
 * <p>
 * The stack stores a given producer Value along with each Value it stores.
 * It then generalizes a given collected Value with the producer Value
 * of each Value it loads. The producer Value and the initial collected Value
 * can be set. The generalized collected Value can be retrieved, either taking
 * into account dup/swap instructions as proper instructions or ignoring them.
 *
 * @author Eric Lafortune
 */
public class TracedStack extends Stack
{
    private Value producerValue;
    private Stack producerStack;
    private Stack actualProducerStack;


    /**
     * Creates a new TracedStack with a given maximum size.
     */
    public TracedStack(int maxSize)
    {
        super(maxSize);

        producerStack       = new Stack(maxSize);
        actualProducerStack = new Stack(maxSize);
    }


    /**
     * Creates a new TracedStack that is a copy of the given TracedStack.
     */
    public TracedStack(TracedStack tracedStack)
    {
        super(tracedStack);

        producerStack       = new Stack(tracedStack.producerStack);
        actualProducerStack = new Stack(tracedStack.actualProducerStack);
    }


    /**
     * Sets the Value that will be stored along with all push and pop
     * instructions.
     */
    public void setProducerValue(Value producerValue)
    {
        this.producerValue = producerValue;
    }


    /**
     * Gets the specified producer Value from the stack, without disturbing it.
     * @param index the index of the stack element, counting from the bottom
     *              of the stack.
     * @return the producer value at the specified position.
     */
    public Value getBottomProducerValue(int index)
    {
        return producerStack.getBottom(index);
    }


    /**
     * Gets the specified actual producer Value from the stack, ignoring
     * dup/swap instructions, without disturbing it.
     * @param index the index of the stack element, counting from the bottom
     *              of the stack.
     * @return the producer value at the specified position.
     */
    public Value getBottomActualProducerValue(int index)
    {
        return actualProducerStack.getBottom(index);
    }


    /**
     * Gets the specified producer Value from the stack, without disturbing it.
     * @param index the index of the stack element, counting from the top
     *              of the stack.
     * @return the producer value at the specified position.
     */
    public Value getTopProducerValue(int index)
    {
        return producerStack.getTop(index);
    }


    /**
     * Gets the specified actual producer Value from the stack, ignoring
     * dup/swap instructions, without disturbing it.
     * @param index the index of the stack element, counting from the top
     *              of the stack.
     * @return the producer value at the specified position.
     */
    public Value getTopActualProducerValue(int index)
    {
        return actualProducerStack.getTop(index);
    }


    // Implementations for Stack.

    public void reset(int size)
    {
        super.reset(size);

        producerStack.reset(size);
        actualProducerStack.reset(size);
    }

    public void copy(TracedStack other)
    {
        super.copy(other);

        producerStack.copy(other.producerStack);
        actualProducerStack.copy(other.actualProducerStack);
    }

    public boolean generalize(TracedStack other)
    {
        return
            super.generalize(other) |
            producerStack.generalize(other.producerStack) |
            actualProducerStack.generalize(other.actualProducerStack);
    }

    public void clear()
    {
        super.clear();

        producerStack.clear();
        actualProducerStack.clear();
    }

    public void removeTop(int index)
    {
        super.removeTop(index);

        producerStack.removeTop(index);
        actualProducerStack.removeTop(index);
    }

    public void push(Value value)
    {
        super.push(value);

        producerPush();

        // Account for the extra space required by Category 2 values.
        if (value.isCategory2())
        {
            producerPush();
        }
    }

    public Value pop()
    {
        Value value = super.pop();

        producerPop();

        // Account for the extra space required by Category 2 values.
        if (value.isCategory2())
        {
            producerPop();
        }

        return value;
    }

    public void pop1()
    {
        super.pop1();

        producerPop();
    }

    public void pop2()
    {
        super.pop2();

        producerPop();
        producerPop();
    }

    public void dup()
    {
        super.dup();

        producerStack.pop();
        producerStack.push(producerValue);
        producerStack.push(producerValue);

        actualProducerStack.dup();
    }

    public void dup_x1()
    {
        super.dup_x1();

        producerStack.pop();
        producerStack.pop();
        producerStack.push(producerValue);
        producerStack.push(producerValue);
        producerStack.push(producerValue);

        actualProducerStack.dup_x1();
    }

    public void dup_x2()
    {
        super.dup_x2();

        producerStack.pop();
        producerStack.pop();
        producerStack.pop();
        producerStack.push(producerValue);
        producerStack.push(producerValue);
        producerStack.push(producerValue);
        producerStack.push(producerValue);

        actualProducerStack.dup_x2();
    }

    public void dup2()
    {
        super.dup2();

        producerStack.pop();
        producerStack.pop();
        producerStack.push(producerValue);
        producerStack.push(producerValue);
        producerStack.push(producerValue);
        producerStack.push(producerValue);

        actualProducerStack.dup2();
    }

    public void dup2_x1()
    {
        super.dup2_x1();

        producerStack.pop();
        producerStack.pop();
        producerStack.pop();
        producerStack.push(producerValue);
        producerStack.push(producerValue);
        producerStack.push(producerValue);
        producerStack.push(producerValue);
        producerStack.push(producerValue);

        actualProducerStack.dup2_x1();
    }

    public void dup2_x2()
    {
        super.dup2_x2();

        producerStack.pop();
        producerStack.pop();
        producerStack.pop();
        producerStack.pop();
        producerStack.push(producerValue);
        producerStack.push(producerValue);
        producerStack.push(producerValue);
        producerStack.push(producerValue);
        producerStack.push(producerValue);
        producerStack.push(producerValue);

        actualProducerStack.dup2_x2();
    }

    public void swap()
    {
        super.swap();

        producerStack.pop();
        producerStack.pop();
        producerStack.push(producerValue);
        producerStack.push(producerValue);

        actualProducerStack.swap();
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        if (object == null ||
            this.getClass() != object.getClass())
        {
            return false;
        }

        TracedStack other = (TracedStack)object;

        return super.equals(object) &&
               this.producerStack.equals(other.producerStack) &&
               this.actualProducerStack.equals(other.actualProducerStack);
    }


    public int hashCode()
    {
        return super.hashCode()         ^
               producerStack.hashCode() ^
               actualProducerStack.hashCode();
    }


    public String toString()
    {
        StringBuffer buffer = new StringBuffer();

        for (int index = 0; index < this.size(); index++)
        {
            Value value               = this.values[index];
            Value producerValue       = producerStack.getBottom(index);
            Value actualProducerValue = actualProducerStack.getBottom(index);
            buffer = buffer.append('[')
                           .append(producerValue == null ? "empty:" :
                                                           producerValue.equals(actualProducerValue) ? producerValue.toString() :
                                                                                                       producerValue.toString() + actualProducerValue.toString())
                           .append(value         == null ? "empty"  : value.toString())
                           .append(']');
        }

        return buffer.toString();
    }


    // Small utility methods.

    private void producerPush()
    {
        producerStack.push(producerValue);
        actualProducerStack.push(producerValue);
    }


    private void producerPop()
    {
        producerStack.pop();
        actualProducerStack.pop();
    }
}

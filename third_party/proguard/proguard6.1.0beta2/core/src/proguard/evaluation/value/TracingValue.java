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
 * This class represents a value that has been tagged with a sticky trace
 * value.
 *
 * @author Eric Lafortune
 */
public class TracingValue extends Value
{
    private Value traceValue;
    private Value value;


    /**
     * Creates a new TracingValue with the given trace value and value.
     */
    public TracingValue(Value traceValue, Value value)
    {

        this.traceValue = traceValue;
        this.value      = value;
    }


    /**
     * Returns the generalization of this TracingValue and the given other
     * TracingValue.
     */
    public final TracingValue generalize(TracingValue other)
    {
        return this.equals(other) ? this :
            new TracingValue(this.traceValue.generalize(other.traceValue),
                             this.value     .generalize(other.value));
    }


    // Implementations for Value.

    public Category1Value category1Value()
    {
        return value.category1Value();
    }

    public Category2Value category2Value()
    {
        return value.category2Value();
    }

    public IntegerValue integerValue()
    {
        return value.integerValue();
    }

    public LongValue longValue()
    {
        return value.longValue();
    }

    public FloatValue floatValue()
    {
        return value.floatValue();
    }

    public DoubleValue doubleValue()
    {
        return value.doubleValue();
    }

    public ReferenceValue referenceValue()
    {
        return value.referenceValue();
    }

    public final InstructionOffsetValue instructionOffsetValue()
    {
        return value.instructionOffsetValue();
    }

    public boolean isSpecific()
    {
        return value.isSpecific();
    }

    public boolean isParticular()
    {
        return value.isParticular();
    }

    public final Value generalize(Value other)
    {
        return
            other instanceof TracingValue ? generalize((TracingValue)other) :
            value.equals(other)           ? this :
                                            new TracingValue(traceValue,
                                                             value.generalize(other));
    }

    public boolean isCategory2()
    {
        return value.isCategory2();
    }

    public final int computationalType()
    {
        return value.computationalType();
    }

    public final String internalType()
    {
        return value.internalType();
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        if (object == null ||
            this.getClass() != object.getClass())
        {
            return false;
        }

        TracingValue other = (TracingValue)object;
        return
            this.traceValue.equals(other.traceValue) &&
            this.value     .equals(other.value);
    }


    public int hashCode()
    {
        return
            this.getClass().hashCode() ^
            traceValue.hashCode() ^
            value     .hashCode();
    }


    public String toString()
    {
        return 'P' + traceValue.toString() + value.toString();
    }
}
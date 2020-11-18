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

import proguard.classfile.Clazz;
import proguard.optimize.evaluation.ReferenceTracingValueFactory;

/**
 * This ReferenceValue represents a reference value that is tagged with a trace
 * value.
 *
 * @author Eric Lafortune
 */
public class TracedReferenceValue extends ReferenceValue
{
    private final ReferenceValue referenceValue;
    private final Value          traceValue;


    /**
     * Creates a new reference value with the given ID.
     */
    public TracedReferenceValue(ReferenceValue referenceValue,
                                Value          traceValue)
    {
        this.referenceValue = referenceValue;
        this.traceValue     = traceValue;
    }


    /**
     * Returns the reference value.
     */
    public ReferenceValue getReferenceValue()
    {
        return referenceValue;
    }


    /**
     * Returns the trace value.
     */
    public Value getTraceValue()
    {
        return traceValue;
    }


    // Implementations for ReferenceValue.

    public String getType()
    {
        return referenceValue.getType();
    }


    public Clazz getReferencedClass()
    {
        return referenceValue.getReferencedClass();
    }


    public boolean mayBeExtension()
    {
        return referenceValue.mayBeExtension();
    }


    public int isNull()
    {
        return referenceValue.isNull();
    }


    public int instanceOf(String otherType, Clazz otherReferencedClass)
    {
        return referenceValue.instanceOf(otherType, otherReferencedClass);
    }


    public ReferenceValue cast(String type, Clazz referencedClass, ValueFactory valueFactory, boolean alwaysCast)
    {
        // We're letting the value factory do the cast (either preserving the
        // trace value or setting a new one).
        return ((ReferenceTracingValueFactory)valueFactory).cast(this,
                                                                 type,
                                                                 referencedClass,
                                                                 alwaysCast);
    }


    public IntegerValue arrayLength(ValueFactory valueFactory)
    {
        return referenceValue.arrayLength(valueFactory);
    }


    public IntegerValue integerArrayLoad(IntegerValue indexValue, ValueFactory valueFactory)
    {
        return referenceValue.integerArrayLoad(indexValue, valueFactory);
    }


    public LongValue longArrayLoad(IntegerValue indexValue, ValueFactory valueFactory)
    {
        return referenceValue.longArrayLoad(indexValue, valueFactory);
    }


    public FloatValue floatArrayLoad(IntegerValue indexValue, ValueFactory valueFactory)
    {
        return referenceValue.floatArrayLoad(indexValue, valueFactory);
    }


    public DoubleValue doubleArrayLoad(IntegerValue indexValue, ValueFactory valueFactory)
    {
        return referenceValue.doubleArrayLoad(indexValue, valueFactory);
    }


    public ReferenceValue referenceArrayLoad(IntegerValue indexValue, ValueFactory valueFactory)
    {
        ReferenceValue value =
            referenceValue.referenceArrayLoad(indexValue, valueFactory);

        // We're keeping the existing trace value, if any, or attaching a new
        // one otherwise.
        return value instanceof TracedReferenceValue ?
            value :
            ((ReferenceTracingValueFactory)valueFactory).trace(value);
    }


    public void arrayStore(IntegerValue indexValue, Value value)
    {
        referenceValue.arrayStore(indexValue, value);
    }


    // Implementations of binary methods of ReferenceValue.

    public ReferenceValue generalize(ReferenceValue other)
    {
        return other.generalize(this);
    }

    public int equal(ReferenceValue other)
    {
        return other.equal(this);
    }


    // Implementations of binary ReferenceValue methods with
    // UnknownReferenceValue arguments.

    public ReferenceValue generalize(UnknownReferenceValue other)
    {
        return new TracedReferenceValue(referenceValue.generalize(other),
                                        traceValue);
    }

    public int equal(UnknownReferenceValue other)
    {
        return referenceValue.equal(other);
    }


    // Implementations of binary ReferenceValue methods with
    // TypedReferenceValue arguments.

    public ReferenceValue generalize(TypedReferenceValue other)
    {
        return new TracedReferenceValue(referenceValue.generalize(other),
                                        traceValue);
    }

    public int equal(TypedReferenceValue other)
    {
        return referenceValue.equal(other);
    }


    // Implementations of binary ReferenceValue methods with
    // IdentifiedReferenceValue arguments.

    public ReferenceValue generalize(IdentifiedReferenceValue other)
    {
        return new TracedReferenceValue(referenceValue.generalize(other),
                                        traceValue);
    }


    public int equal(IdentifiedReferenceValue other)
    {
        return referenceValue.equal(other);
    }


    // Implementations of binary ReferenceValue methods with
    // ArrayReferenceValue arguments.

    public ReferenceValue generalize(ArrayReferenceValue other)
    {
        return new TracedReferenceValue(referenceValue.generalize(other),
                                        traceValue);
    }


    public int equal(ArrayReferenceValue other)
    {
        return referenceValue.equal(other);
    }


    // Implementations of binary ReferenceValue methods with
    // IdentifiedArrayReferenceValue arguments.

    public ReferenceValue generalize(IdentifiedArrayReferenceValue other)
    {
        return new TracedReferenceValue(referenceValue.generalize(other),
                                        traceValue);
    }


    public int equal(IdentifiedArrayReferenceValue other)
    {
        return referenceValue.equal(other);
    }


    // Implementations of binary ReferenceValue methods with
    // DetailedArrayReferenceValue arguments.

    public ReferenceValue generalize(DetailedArrayReferenceValue other)
    {
        return new TracedReferenceValue(referenceValue.generalize(other),
                                        traceValue);
    }


    public int equal(DetailedArrayReferenceValue other)
    {
        return referenceValue.equal(other);
    }


    // Implementations of binary ReferenceValue methods with
    // TracedReferenceValue arguments.

    public ReferenceValue generalize(TracedReferenceValue other)
    {
        if (this.equals(other))
        {
            return this;
        }

        return new TracedReferenceValue(this.referenceValue.generalize(other.referenceValue),
                                        this.traceValue    .generalize(other.traceValue));
    }

    public int equal(TracedReferenceValue other)
    {
        return this.referenceValue.equal(other.referenceValue);
    }


    // Implementations for Value.

    public boolean isSpecific()
    {
        return referenceValue.isSpecific();
    }

    public boolean isParticular()
    {
        return referenceValue.isParticular();
    }

    public String internalType()
    {
        return referenceValue.internalType();
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        if (this == object)
        {
            return true;
        }

        if (!super.equals(object))
        {
            return false;
        }

        TracedReferenceValue other = (TracedReferenceValue)object;
        return this.referenceValue.equals(other.referenceValue) &&
               this.traceValue    .equals(other.traceValue);
    }


    public int hashCode()
    {
        return referenceValue.hashCode() ^
               traceValue.hashCode();
    }


    public String toString()
    {
        return traceValue.toString() + referenceValue.toString();
    }
}
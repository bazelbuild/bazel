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

import proguard.classfile.Clazz;

/**
 * This class represents a partially evaluated reference value.
 *
 * @author Eric Lafortune
 */
public abstract class ReferenceValue extends Category1Value
{
    /**
     * Returns the type.
     */
    public abstract String getType();
;

    /**
     * Returns the class that is referenced by the type.
     */
    public abstract Clazz getReferencedClass();


    // Basic unary methods.

    /**
     * Returns whether the type is <code>null</code>.
     */
    public abstract int isNull();


    /**
     * Returns whether the type is an instance of the given type.
     */
    public abstract int instanceOf(String otherType, Clazz otherReferencedClass);


    /**
     * Returns a generalization of this ReferenceValue that may be null,
     * depending on the flag.
     */
    public abstract ReferenceValue generalizeMayBeNull(boolean mayBeNull);


    /**
     * Returns the length of the array, assuming this type is an array.
     */
    public IntegerValue arrayLength(ValueFactory valueFactory)
    {
        return valueFactory.createIntegerValue();
    }


    /**
     * Returns the value of the array at the given index, assuming this type
     * is an integer array.
     */
    public IntegerValue integerArrayLoad(IntegerValue indexValue, ValueFactory valueFactory)
    {
        return valueFactory.createIntegerValue();
    }


    /**
     * Returns the value of the array at the given index, assuming this type
     * is an long array.
     */
    public LongValue longArrayLoad(IntegerValue indexValue, ValueFactory valueFactory)
    {
        return valueFactory.createLongValue();
    }


    /**
     * Returns the value of the array at the given index, assuming this type
     * is an float array.
     */
    public FloatValue floatArrayLoad(IntegerValue indexValue, ValueFactory valueFactory)
    {
        return valueFactory.createFloatValue();
    }


    /**
     * Returns the value of the array at the given index, assuming this type
     * is an double array.
     */
    public DoubleValue doubleArrayLoad(IntegerValue indexValue, ValueFactory valueFactory)
    {
        return valueFactory.createDoubleValue();
    }


    /**
     * Returns the value of the array at the given index, assuming this type
     * is a reference array.
     */
    public abstract ReferenceValue referenceArrayLoad(IntegerValue indexValue, ValueFactory valueFactory);


    /**
     * Stores the given value at the given index in the given array, assuming
     * this type is an array.
     */
    public void arrayStore(IntegerValue indexValue, Value value)
    {
    }


    // Basic binary methods.

    /**
     * Returns the generalization of this ReferenceValue and the given other
     * ReferenceValue.
     */
    public abstract ReferenceValue generalize(ReferenceValue other);


    /**
     * Returns whether this ReferenceValue is equal to the given other
     * ReferenceValue.
     * @return <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public abstract int equal(ReferenceValue other);


    // Derived unary methods.

    /**
     * Returns whether this ReferenceValue is not <code>null</code>.
     * @return <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public final int isNotNull()
    {
        return -isNull();
    }


    // Derived binary methods.

    /**
     * Returns whether this ReferenceValue and the given ReferenceValue are different.
     * @return <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public final int notEqual(ReferenceValue other)
    {
        return -equal(other);
    }


    // Similar binary methods, but this time with typed reference arguments.

    /**
     * Returns the generalization of this ReferenceValue and the given other
     * TypedReferenceValue.
     */
    public ReferenceValue generalize(TypedReferenceValue other)
    {
        return generalize((ReferenceValue)other);
    }


    /**
     * Returns whether this ReferenceValue is equal to the given other
     * TypedReferenceValue.
     * @return <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public int equal(TypedReferenceValue other)
    {
        return equal((ReferenceValue)other);
    }


    // Similar binary methods, but this time with identified reference
    // arguments.

    /**
     * Returns the generalization of this ReferenceValue and the given other
     * IdentifiedReferenceValue.
     */
    public ReferenceValue generalize(IdentifiedReferenceValue other)
    {
        return generalize((TypedReferenceValue)other);
    }


    /**
     * Returns whether this ReferenceValue is equal to the given other
     * IdentifiedReferenceValue.
     * @return <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public int equal(IdentifiedReferenceValue other)
    {
        return equal((TypedReferenceValue)other);
    }


    // Similar binary methods, but this time with array reference arguments.

    /**
     * Returns the generalization of this ReferenceValue and the given other
     * ArrayReferenceValue.
     */
    public ReferenceValue generalize(ArrayReferenceValue other)
    {
        return generalize((TypedReferenceValue)other);
    }


    /**
     * Returns whether this ReferenceValue is equal to the given other
     * ArrayReferenceValue.
     * @return <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public int equal(ArrayReferenceValue other)
    {
        return equal((TypedReferenceValue)other);
    }


    // Similar binary methods, but this time with identified array reference
    // arguments.

    /**
     * Returns the generalization of this ReferenceValue and the given other
     * IdentifiedArrayReferenceValue.
     */
    public ReferenceValue generalize(IdentifiedArrayReferenceValue other)
    {
        return generalize((ArrayReferenceValue)other);
    }


    /**
     * Returns whether this ReferenceValue is equal to the given other
     * IdentifiedArrayReferenceValue.
     * @return <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public int equal(IdentifiedArrayReferenceValue other)
    {
        return equal((ArrayReferenceValue)other);
    }


    // Similar binary methods, but this time with detailed array reference
    // arguments.

    /**
     * Returns the generalization of this ReferenceValue and the given other
     * DetailedArrayReferenceValue.
     */
    public ReferenceValue generalize(DetailedArrayReferenceValue other)
    {
        return generalize((IdentifiedArrayReferenceValue)other);
    }


    /**
     * Returns whether this ReferenceValue is equal to the given other
     * DetailedArrayReferenceValue.
     * @return <code>NEVER</code>, <code>MAYBE</code>, or <code>ALWAYS</code>.
     */
    public int equal(DetailedArrayReferenceValue other)
    {
        return equal((IdentifiedArrayReferenceValue)other);
    }


    // Implementations for Value.

    public final ReferenceValue referenceValue()
    {
        return this;
    }

    public final Value generalize(Value other)
    {
        return this.generalize(other.referenceValue());
    }

    public final int computationalType()
    {
        return TYPE_REFERENCE;
    }
}

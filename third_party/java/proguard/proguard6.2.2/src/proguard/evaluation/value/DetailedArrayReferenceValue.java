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
import proguard.classfile.util.ClassUtil;
import proguard.util.ArrayUtil;

/**
 * This IdentifiedArrayReferenceValue represents an identified array reference
 * value with its elements.
 *
 * @author Eric Lafortune
 */
class DetailedArrayReferenceValue extends IdentifiedArrayReferenceValue
{
    private static final int MAXIMUM_STORED_ARRAY_LENGTH = 32;


    private final Value[] values;


    /**
     * Creates a new array reference value with the given ID.
     */
    public DetailedArrayReferenceValue(String       type,
                                       Clazz        referencedClass,
                                       boolean      mayBeExtension,
                                       IntegerValue arrayLength,
                                       ValueFactory valuefactory,
                                       int          id)
    {
        super(type, referencedClass, mayBeExtension, arrayLength, valuefactory, id);

        // Is the array short enough to analyze?
        if (arrayLength.isParticular() &&
            arrayLength.value() >= 0   &&
            arrayLength.value() <= MAXIMUM_STORED_ARRAY_LENGTH)
        {
            // Initialize the values of the array.
            InitialValueFactory initialValueFactory =
                new InitialValueFactory(valuefactory);

            String elementType = ClassUtil.isInternalArrayType(type) ?
                type.substring(1) :
                type;

            this.values = new Value[arrayLength.value()];

            for (int index = 0; index < values.length; index++)
            {
                values[index] = initialValueFactory.createValue(elementType);
            }
        }
        else
        {
            // Just ignore the values of the array.
            this.values = null;
        }
    }


    // Implementations for ReferenceValue.

    public IntegerValue integerArrayLoad(IntegerValue indexValue, ValueFactory valueFactory)
    {
        Value value = arrayLoad(indexValue, valueFactory);
        return value != null ?
            value.integerValue() :
            super.integerArrayLoad(indexValue, valueFactory);
    }


    public LongValue longArrayLoad(IntegerValue indexValue, ValueFactory valueFactory)
    {
        Value value = arrayLoad(indexValue, valueFactory);
        return value != null ?
            value.longValue() :
            super.longArrayLoad(indexValue, valueFactory);
    }


    public FloatValue floatArrayLoad(IntegerValue indexValue, ValueFactory valueFactory)
    {
        Value value = arrayLoad(indexValue, valueFactory);
        return value != null ?
            value.floatValue() :
            super.floatArrayLoad(indexValue, valueFactory);
    }


    public DoubleValue doubleArrayLoad(IntegerValue indexValue, ValueFactory valueFactory)
    {
        Value value = arrayLoad(indexValue, valueFactory);
        return value != null ?
            value.doubleValue() :
            super.doubleArrayLoad(indexValue, valueFactory);
    }


    public ReferenceValue referenceArrayLoad(IntegerValue indexValue, ValueFactory valueFactory)
    {
        Value value = arrayLoad(indexValue, valueFactory);
        return value != null ?
            value.referenceValue() :
            super.referenceArrayLoad(indexValue, valueFactory);
    }


    /**
     * Returns the specified untyped value from the given array, or null if it
     * is unknown.
     */
    private Value arrayLoad(IntegerValue indexValue, ValueFactory valueFactory)
    {
        if (values != null &&
            indexValue.isParticular())
        {
            int index = indexValue.value();
            if (index >=0 &&
                index < values.length)
            {
                return values[index];
            }
        }

        return null;
    }


    public void arrayStore(IntegerValue indexValue, Value value)
    {
        if (values != null)
        {
            if (indexValue.isParticular())
            {
                int index = indexValue.value();
                if (index >=0 &&
                    index < values.length)
                {
                    values[index] = value;
                }
            }
            else
            {
                for (int index = 0; index < values.length; index++)
                {
                    values[index].generalize(value);
                }
            }
        }
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


//    // Implementations of binary ReferenceValue methods with
//    // UnknownReferenceValue arguments.
//
//    public ReferenceValue generalize(UnknownReferenceValue other)
//    {
//        return other;
//    }
//
//
//    public int equal(UnknownReferenceValue other)
//    {
//        return MAYBE;
//    }
//
//
//    // Implementations of binary ReferenceValue methods with
//    // TypedReferenceValue arguments.
//
//    public ReferenceValue generalize(TypedReferenceValue other)
//    {
//    }
//
//
//    public int equal(TypedReferenceValue other)
//    {
//    }
//
//
//    // Implementations of binary ReferenceValue methods with
//    // IdentifiedReferenceValue arguments.
//
//    public ReferenceValue generalize(IdentifiedReferenceValue other)
//    {
//        return generalize((TypedReferenceValue)other);
//    }
//
//
//    public int equal(IdentifiedReferenceValue other)
//    {
//        return equal((TypedReferenceValue)other);
//    }
//
//
//    // Implementations of binary ReferenceValue methods with
//    // ArrayReferenceValue arguments.
//
//    public ReferenceValue generalize(ArrayReferenceValue other)
//    {
//        return generalize((TypedReferenceValue)other);
//    }
//
//
//    public int equal(ArrayReferenceValue other)
//    {
//        return equal((TypedReferenceValue)other);
//    }
//
//
//    // Implementations of binary ReferenceValue methods with
//    // IdentifiedArrayReferenceValue arguments.
//
//    public ReferenceValue generalize(IdentifiedArrayReferenceValue other)
//    {
//        return generalize((ArrayReferenceValue)other);
//    }
//
//
//    public int equal(IdentifiedArrayReferenceValue other)
//    {
//        return equal((ArrayReferenceValue)other);
//    }
//
//
//    // Implementations of binary ReferenceValue methods with
//    // DetailedArrayReferenceValue arguments.
//
//    public ReferenceValue generalize(DetailedArrayReferenceValue other)
//    {
//        return generalize((IdentifiedArrayReferenceValue)other);
//    }
//
//
//    public int equal(DetailedArrayReferenceValue other)
//    {
//        return equal((IdentifiedArrayReferenceValue)other);
//    }
//
//
//    // Implementations of binary ReferenceValue methods with
//    // TracedReferenceValue arguments.
//
//    public ReferenceValue generalize(TracedReferenceValue other)
//    {
//        return other.generalize(this);
//    }
//
//
//    public int equal(TracedReferenceValue other)
//    {
//        return other.equal(this);
//    }


    // Implementations for Value.

    public boolean isParticular()
    {
        if (values == null)
        {
            return false;
        }

        for (int index = 0; index < values.length; index++)
        {
            if (!values[index].isParticular())
            {
                return false;
            }
        }

        return true;
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

        DetailedArrayReferenceValue other =
            (DetailedArrayReferenceValue)object;

        return ArrayUtil.equalOrNull(this.values, other.values);
    }


    public int hashCode()
    {
        return super.hashCode() ^
               ArrayUtil.hashCodeOrNull(values);
    }


    public String toString()
    {
        if (values == null)
        {
            return super.toString();
        }

        StringBuffer buffer = new StringBuffer(super.toString());

        buffer.append('{');
        for (int index = 0; index < values.length; index++)
        {
            buffer.append(values[index]);
            buffer.append(index < values.length-1 ? ',' : '}');
        }

        return buffer.toString();
    }
}
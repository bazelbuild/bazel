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

/**
 * This ReferenceValue represents a partially evaluated array. It has an array
 * length and possibly array values (up to a fixed maximum number). It is not
 * immutable.
 *
 * @author Eric Lafortune
 */
class ArrayReferenceValue extends TypedReferenceValue
{
    protected final IntegerValue arrayLength;


    /**
     * Creates a new ArrayReferenceValue.
     */
    public ArrayReferenceValue(String       type,
                               Clazz        referencedClass,
                               boolean      mayBeExtension,
                               IntegerValue arrayLength)
    {
        super(type, referencedClass, mayBeExtension, false);

        this.arrayLength = arrayLength;
    }


    // Implementations for ReferenceValue.

    public IntegerValue arrayLength(ValueFactory valueFactory)
    {
        return arrayLength;
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


    // Implementations of binary ReferenceValue methods with
    // ArrayReferenceValue arguments.

    public ReferenceValue generalize(ArrayReferenceValue other)
    {
        return
            this.equals(other)                            ? this :
            this.type != null            &&
            this.type.equals(other.type) &&
            this.referencedClass == other.referencedClass ? new ArrayReferenceValue(this.type,
                                                                                    this.referencedClass,
                                                                                    this.mayBeExtension || other.mayBeExtension,
                                                                                    this.arrayLength.generalize(other.arrayLength)) :
                                                            generalize((TypedReferenceValue)other);
    }


    public int equal(ArrayReferenceValue other)
    {
        if (this.arrayLength.equal(other.arrayLength) == NEVER)
        {
            return NEVER;
        }

        return equal((TypedReferenceValue)other);
    }


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

        ArrayReferenceValue other = (ArrayReferenceValue)object;
        return this.arrayLength.equals(other.arrayLength);
    }


    public int hashCode()
    {
        return super.hashCode() ^
               arrayLength.hashCode();
    }


    public String toString()
    {
        return super.toString() + '['+arrayLength+']';
    }
}

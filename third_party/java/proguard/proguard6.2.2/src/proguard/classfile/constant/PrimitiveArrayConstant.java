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

package proguard.classfile.constant;

import proguard.classfile.ClassConstants;
import proguard.classfile.Clazz;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.constant.visitor.PrimitiveArrayConstantElementVisitor;
import proguard.classfile.constant.visitor.PrimitiveArrayConstantVisitor;

/**
 * This unofficial Constant represents an array of primitives in the constant
 * pool. It is not supported by any Java specification and therefore only for
 * internal use.
 *
 * @author Eric Lafortune
 */
public class PrimitiveArrayConstant extends Constant
{
    public Object values;


    /**
     * Creates an uninitialized PrimitiveArrayConstant.
     */
    public PrimitiveArrayConstant()
    {
    }


    /**
     * Creates a new PrimitiveArrayConstant with the given array of values.
     */
    public PrimitiveArrayConstant(Object values)
    {
        this.values = values;
    }


    /**
     * Returns the type of the elements of the primitive array.
     */
    public char getPrimitiveType()
    {
        return values instanceof boolean[] ? ClassConstants.TYPE_BOOLEAN :
               values instanceof byte[]    ? ClassConstants.TYPE_BYTE    :
               values instanceof char[]    ? ClassConstants.TYPE_CHAR    :
               values instanceof short[]   ? ClassConstants.TYPE_SHORT   :
               values instanceof int[]     ? ClassConstants.TYPE_INT     :
               values instanceof float[]   ? ClassConstants.TYPE_FLOAT   :
               values instanceof long[]    ? ClassConstants.TYPE_LONG    :
               values instanceof double[]  ? ClassConstants.TYPE_DOUBLE  :
                                             0;
    }


    /**
     * Returns the length of the primitive array.
     */
    public int getLength()
    {
        return values instanceof boolean[] ? ((boolean[])values).length :
               values instanceof byte[]    ? ((byte[]   )values).length :
               values instanceof char[]    ? ((char[]   )values).length :
               values instanceof short[]   ? ((short[]  )values).length :
               values instanceof int[]     ? ((int[]    )values).length :
               values instanceof float[]   ? ((float[]  )values).length :
               values instanceof long[]    ? ((long[]   )values).length :
               values instanceof double[]  ? ((double[] )values).length :
                                             0;
    }


    /**
     * Returns the values.
     */
    public Object getValues()
    {
        return values;
    }


    /**
     * Applies the given PrimitiveArrayConstantVisitor to the primitive array.
     */
    public void primitiveArrayAccept(Clazz clazz, PrimitiveArrayConstantVisitor primitiveArrayConstantVisitor)
    {
        // The primitive arrays themselves don't accept visitors, so we have to
        // use instanceof tests.
        if (values instanceof boolean[])
        {
            primitiveArrayConstantVisitor.visitBooleanArrayConstant(clazz, this, (boolean[])values);
        }
        else if (values instanceof byte[])
        {
            primitiveArrayConstantVisitor.visitByteArrayConstant(clazz, this, (byte[])values);
        }
        else if (values instanceof char[])
        {
            primitiveArrayConstantVisitor.visitCharArrayConstant(clazz, this, (char[])values);
        }
        else if (values instanceof short[])
        {
            primitiveArrayConstantVisitor.visitShortArrayConstant(clazz, this, (short[])values);
        }
        else if (values instanceof int[])
        {
            primitiveArrayConstantVisitor.visitIntArrayConstant(clazz, this, (int[])values);
        }
        else if (values instanceof float[])
        {
            primitiveArrayConstantVisitor.visitFloatArrayConstant(clazz, this, (float[])values);
        }
        else if (values instanceof long[])
        {
            primitiveArrayConstantVisitor.visitLongArrayConstant(clazz, this, (long[])values);
        }
        else if (values instanceof double[])
        {
            primitiveArrayConstantVisitor.visitDoubleArrayConstant(clazz, this, (double[])values);
        }
    }


    /**
     * Applies the given PrimitiveArrayConstantElementVisitor to all elements
     * of the primitive array.
     */
    public void primitiveArrayElementsAccept(Clazz clazz, PrimitiveArrayConstantElementVisitor primitiveArrayConstantElementVisitor)
    {
        // The primitive arrays themselves don't accept visitors, so we have to
        // use instanceof tests.
        if (values instanceof boolean[])
        {
            boolean[] booleanValues = (boolean[])this.values;
            for (int index = 0; index < booleanValues.length; index++)
            {
                primitiveArrayConstantElementVisitor.visitBooleanArrayConstantElement(clazz, this, index, booleanValues[index]);
            }
        }
        else if (values instanceof byte[])
        {
            byte[] byteValues = (byte[])this.values;
            for (int index = 0; index < byteValues.length; index++)
            {
                primitiveArrayConstantElementVisitor.visitByteArrayConstantElement(clazz, this, index, byteValues[index]);
            }
        }
        else if (values instanceof char[])
        {
            char[] charValues = (char[])this.values;
            for (int index = 0; index < charValues.length; index++)
            {
                primitiveArrayConstantElementVisitor.visitCharArrayConstantElement(clazz, this, index, charValues[index]);
            }
        }
        else if (values instanceof short[])
        {
            short[] shortValues = (short[])this.values;
            for (int index = 0; index < shortValues.length; index++)
            {
                primitiveArrayConstantElementVisitor.visitShortArrayConstantElement(clazz, this, index, shortValues[index]);
            }
        }
        else if (values instanceof int[])
        {
            int[] intValues = (int[])this.values;
            for (int index = 0; index < intValues.length; index++)
            {
                primitiveArrayConstantElementVisitor.visitIntArrayConstantElement(clazz, this, index, intValues[index]);
            }
        }
        else if (values instanceof float[])
        {
            float[] floatValues = (float[])this.values;
            for (int index = 0; index < floatValues.length; index++)
            {
                primitiveArrayConstantElementVisitor.visitFloatArrayConstantElement(clazz, this, index, floatValues[index]);
            }
        }
        else if (values instanceof long[])
        {
            long[] longValues = (long[])this.values;
            for (int index = 0; index < longValues.length; index++)
            {
                primitiveArrayConstantElementVisitor.visitLongArrayConstantElement(clazz, this, index, longValues[index]);
            }
        }
        else if (values instanceof double[])
        {
            double[] doubleValues = (double[])this.values;
            for (int index = 0; index < doubleValues.length; index++)
            {
                primitiveArrayConstantElementVisitor.visitDoubleArrayConstantElement(clazz, this, index, doubleValues[index]);
            }
        }
    }


    // Implementations for Constant.

    public int getTag()
    {
        return ClassConstants.CONSTANT_PrimitiveArray;
    }

    public void accept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        constantVisitor.visitPrimitiveArrayConstant(clazz, this);
    }
}

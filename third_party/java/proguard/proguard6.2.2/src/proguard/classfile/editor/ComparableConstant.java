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
package proguard.classfile.editor;

import proguard.classfile.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.util.ArrayUtil;


/**
 * This class is a <code>Comparable</code> wrapper of <code>Constant</code>
 * objects. It can store an index, in order to identify the constant pool
 * entry after it has been sorted. The comparison is primarily based on the
 * types of the constant pool entries, and secondarily on the contents of
 * the constant pool entries.
 *
 * @author Eric Lafortune
 */
class      ComparableConstant
extends    SimplifiedVisitor
implements Comparable, ConstantVisitor
{
    private static final int[] PRIORITIES = new int[22];
    static
    {
        PRIORITIES[ClassConstants.CONSTANT_Integer]            =  0; // Possibly byte index (ldc).
        PRIORITIES[ClassConstants.CONSTANT_Float]              =  1;
        PRIORITIES[ClassConstants.CONSTANT_String]             =  2;
        PRIORITIES[ClassConstants.CONSTANT_Class]              =  3;
        PRIORITIES[ClassConstants.CONSTANT_Long]               =  4; // Always wide index (ldc2_w).
        PRIORITIES[ClassConstants.CONSTANT_Double]             =  5; // Always wide index (ldc2_w).
        PRIORITIES[ClassConstants.CONSTANT_Fieldref]           =  6; // Always wide index (getfield,...).
        PRIORITIES[ClassConstants.CONSTANT_Methodref]          =  7; // Always wide index (invokespecial,...).
        PRIORITIES[ClassConstants.CONSTANT_InterfaceMethodref] =  8; // Always wide index (invokeinterface).
        PRIORITIES[ClassConstants.CONSTANT_Dynamic]            =  9; // Always wide index (invokedynamic).
        PRIORITIES[ClassConstants.CONSTANT_InvokeDynamic]      = 10; // Always wide index (invokedynamic).
        PRIORITIES[ClassConstants.CONSTANT_MethodHandle]       = 11;
        PRIORITIES[ClassConstants.CONSTANT_NameAndType]        = 12;
        PRIORITIES[ClassConstants.CONSTANT_MethodType]         = 13;
        PRIORITIES[ClassConstants.CONSTANT_Module]             = 14;
        PRIORITIES[ClassConstants.CONSTANT_Package]            = 15;
        PRIORITIES[ClassConstants.CONSTANT_Utf8]               = 16;
        PRIORITIES[ClassConstants.CONSTANT_PrimitiveArray]     = 17;
    }

    private final Clazz    clazz;
    private final int      thisIndex;
    private final Constant thisConstant;

    private Constant otherConstant;
    private int      result;


    public ComparableConstant(Clazz clazz, int index, Constant constant)
    {
        this.clazz        = clazz;
        this.thisIndex    = index;
        this.thisConstant = constant;
    }


    public int getIndex()
    {
        return thisIndex;
    }


    public Constant getConstant()
    {
        return thisConstant;
    }


    // Implementations for Comparable.

    public int compareTo(Object other)
    {
        ComparableConstant otherComparableConstant = (ComparableConstant)other;

        otherConstant = otherComparableConstant.thisConstant;

        // Compare based on the original indices, if the actual constant pool
        // entries are the same.
        if (thisConstant == otherConstant)
        {
            int otherIndex = otherComparableConstant.thisIndex;

            return Integer.compare(thisIndex, otherIndex);
        }

        // Compare based on the tags, if they are different.
        int thisTag  = thisConstant.getTag();
        int otherTag = otherConstant.getTag();

        if (thisTag != otherTag)
        {
            return Integer.compare(PRIORITIES[thisTag], PRIORITIES[otherTag]);
        }

        // Otherwise compare based on the contents of the Constant objects.
        thisConstant.accept(clazz, this);

        return result;
    }


    // Implementations for ConstantVisitor.

    public void visitIntegerConstant(Clazz clazz, IntegerConstant integerConstant)
    {
        result = Integer.compare(integerConstant.getValue(),
                                 ((IntegerConstant)otherConstant).getValue());
    }

    public void visitLongConstant(Clazz clazz, LongConstant longConstant)
    {
        result = Long.compare(longConstant.getValue(),
                              ((LongConstant)otherConstant).getValue());
    }

    public void visitFloatConstant(Clazz clazz, FloatConstant floatConstant)
    {
        result = Float.compare(floatConstant.getValue(),
                               ((FloatConstant)otherConstant).getValue());
    }

    public void visitDoubleConstant(Clazz clazz, DoubleConstant doubleConstant)
    {
        result = Double.compare(doubleConstant.getValue(),
                                ((DoubleConstant)otherConstant).getValue());
    }

    public void visitPrimitiveArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant)
    {
        PrimitiveArrayConstant otherPrimitiveArrayConstant =
            (PrimitiveArrayConstant)otherConstant;

        char primitiveType      = primitiveArrayConstant.getPrimitiveType();
        char otherPrimitiveType = otherPrimitiveArrayConstant.getPrimitiveType();

        if (primitiveType != otherPrimitiveType)
        {
            result = Integer.compare(primitiveType, otherPrimitiveType);
        }
        else
        {
            Object values      = primitiveArrayConstant.getValues();
            Object otherValues = otherPrimitiveArrayConstant.getValues();

            result =
                values instanceof boolean[] ? ArrayUtil.compare((boolean[])values, ((boolean[])values).length, (boolean[])otherValues, ((boolean[])otherValues).length) :
                values instanceof byte[]    ? ArrayUtil.compare((byte[])   values, ((byte[])   values).length, (byte[])   otherValues, ((byte[])   otherValues).length) :
                values instanceof char[]    ? ArrayUtil.compare((char[])   values, ((char[])   values).length, (char[])   otherValues, ((char[])   otherValues).length) :
                values instanceof short[]   ? ArrayUtil.compare((short[])  values, ((short[])  values).length, (short[])  otherValues, ((short[])  otherValues).length) :
                values instanceof int[]     ? ArrayUtil.compare((int[])    values, ((int[])    values).length, (int[])    otherValues, ((int[])    otherValues).length) :
                values instanceof float[]   ? ArrayUtil.compare((float[])  values, ((float[])  values).length, (float[])  otherValues, ((float[])  otherValues).length) :
                values instanceof long[]    ? ArrayUtil.compare((long[])   values, ((long[])   values).length, (long[])   otherValues, ((long[])   otherValues).length) :
              /*values instanceof double[] */ ArrayUtil.compare((double[]) values, ((double[]) values).length, (double[]) otherValues, ((double[]) otherValues).length);
        }
    }

    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        result = stringConstant.getString(clazz).compareTo(((StringConstant)otherConstant).getString(clazz));
    }

    public void visitUtf8Constant(Clazz clazz, Utf8Constant utf8Constant)
    {
        result = utf8Constant.getString().compareTo(((Utf8Constant)otherConstant).getString());
    }

    public void visitDynamicConstant(Clazz clazz, DynamicConstant dynamicConstant)
    {
        DynamicConstant otherDynamicConstant = (DynamicConstant)otherConstant;

        int index      = dynamicConstant.getBootstrapMethodAttributeIndex();
        int otherIndex = otherDynamicConstant.getBootstrapMethodAttributeIndex();

        result = index < otherIndex ? -1 :
                 index > otherIndex ?  1 :
                     compare(dynamicConstant.getName(clazz),
                             dynamicConstant.getType(clazz),
                             otherDynamicConstant.getName(clazz),
                             otherDynamicConstant.getType(clazz));
    }

    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        InvokeDynamicConstant otherInvokeDynamicConstant = (InvokeDynamicConstant)otherConstant;

        int index      = invokeDynamicConstant.getBootstrapMethodAttributeIndex();
        int otherIndex = otherInvokeDynamicConstant.getBootstrapMethodAttributeIndex();

        result = index < otherIndex ? -1 :
                 index > otherIndex ?  1 :
                     compare(invokeDynamicConstant.getName(clazz),
                             invokeDynamicConstant.getType(clazz),
                             otherInvokeDynamicConstant.getName(clazz),
                             otherInvokeDynamicConstant.getType(clazz));
    }

    public void visitMethodHandleConstant(Clazz clazz, MethodHandleConstant methodHandleConstant)
    {
        MethodHandleConstant otherMethodHandleConstant = (MethodHandleConstant)otherConstant;

        int kind      = methodHandleConstant.getReferenceKind();
        int otherKind = otherMethodHandleConstant.getReferenceKind();

        result = kind < otherKind ? -1 :
                 kind > otherKind ?  1 :
                     compare(methodHandleConstant.getClassName(clazz),
                             methodHandleConstant.getName(clazz),
                             methodHandleConstant.getType(clazz),
                             otherMethodHandleConstant.getClassName(clazz),
                             otherMethodHandleConstant.getName(clazz),
                             otherMethodHandleConstant.getType(clazz));
    }

    public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
    {
        RefConstant otherRefConstant = (RefConstant)otherConstant;
        result = compare(refConstant.getClassName(clazz),
                         refConstant.getName(clazz),
                         refConstant.getType(clazz),
                         otherRefConstant.getClassName(clazz),
                         otherRefConstant.getName(clazz),
                         otherRefConstant.getType(clazz));
    }

    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        result = classConstant.getName(clazz).compareTo(((ClassConstant)otherConstant).getName(clazz));
    }

    public void visitMethodTypeConstant(Clazz clazz, MethodTypeConstant MethodTypeConstant)
    {
        MethodTypeConstant otherMethodTypeConstant = (MethodTypeConstant)otherConstant;
        result = MethodTypeConstant.getType(clazz)
                 .compareTo
                 (otherMethodTypeConstant.getType(clazz));
    }

    public void visitNameAndTypeConstant(Clazz clazz, NameAndTypeConstant nameAndTypeConstant)
    {
        NameAndTypeConstant otherNameAndTypeConstant = (NameAndTypeConstant)otherConstant;
        result = compare(nameAndTypeConstant.getName(clazz),
                         nameAndTypeConstant.getType(clazz),
                         otherNameAndTypeConstant.getName(clazz),
                         otherNameAndTypeConstant.getType(clazz));
    }


    public void visitModuleConstant(Clazz clazz, ModuleConstant moduleConstant)
    {
        result = moduleConstant.getName(clazz).compareTo(((ModuleConstant)otherConstant).getName(clazz));
    }


    public void visitPackageConstant(Clazz clazz, PackageConstant packageConstant)
    {
        result = packageConstant.getName(clazz).compareTo(((PackageConstant)otherConstant).getName(clazz));
    }

    // Implementations for Object.

    public boolean equals(Object other)
    {
        return other != null &&
               this.getClass().equals(other.getClass()) &&
               this.getConstant().getClass().equals(((ComparableConstant)other).getConstant().getClass()) &&
               this.compareTo(other) == 0;
    }


    public int hashCode()
    {
        return this.getClass().hashCode();
    }


    // Small utility methods.

    /**
     * Compares the given two pairs of strings.
     */
    private int compare(String string1a, String string1b,
                        String string2a, String string2b)
    {
        int comparison;
        return
            (comparison = string1a.compareTo(string2a)) != 0 ? comparison :
                          string1b.compareTo(string2b);
    }


    /**
     * Compares the given two triplets of strings.
     */
    private int compare(String string1a, String string1b, String string1c,
                        String string2a, String string2b, String string2c)
    {
        int comparison;
        return
            (comparison = string1a.compareTo(string2a)) != 0 ? comparison :
            (comparison = string1b.compareTo(string2b)) != 0 ? comparison :
                          string1c.compareTo(string2c);
    }
}

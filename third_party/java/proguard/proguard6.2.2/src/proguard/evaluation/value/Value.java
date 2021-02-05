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
 * This abstract class represents a partially evaluated value.
 *
 * @author Eric Lafortune
 */
public abstract class Value
{
    public static final int NEVER  = -1;
    public static final int MAYBE  = 0;
    public static final int ALWAYS = 1;

    public static final int TYPE_INTEGER            = 1;
    public static final int TYPE_LONG               = 2;
    public static final int TYPE_FLOAT              = 3;
    public static final int TYPE_DOUBLE             = 4;
    public static final int TYPE_REFERENCE          = 5;
    public static final int TYPE_INSTRUCTION_OFFSET = 6;
    public static final int TYPE_TOP                = 7;


    /**
     * Returns this Value as a Category1Value.
     */
    public Category1Value category1Value()
    {
        throw new IllegalArgumentException("Value \"" + this.toString() + "\" is not a Category 1 value [" + this.getClass().getName() + "]");
    }

    /**
     * Returns this Value as a Category2Value.
     */
    public Category2Value category2Value()
    {
        throw new IllegalArgumentException("Value \"" + this.toString() + "\" is not a Category 2 value [" + this.getClass().getName() + "]");
    }


    /**
     * Returns this Value as an IntegerValue.
     */
    public IntegerValue integerValue()
    {
        throw new IllegalArgumentException("Value \"" + this.toString() + "\" is not an integer value [" + this.getClass().getName() + "]");
    }

    /**
     * Returns this Value as a LongValue.
     */
    public LongValue longValue()
    {
        throw new IllegalArgumentException("Value \"" + this.toString() + "\" is not a long value [" + this.getClass().getName() + "]");
    }

    /**
     * Returns this Value as a FloatValue.
     */
    public FloatValue floatValue()
    {
        throw new IllegalArgumentException("Value \"" + this.toString() + "\" is not a float value [" + this.getClass().getName() + "]");
    }

    /**
     * Returns this Value as a DoubleValue.
     */
    public DoubleValue doubleValue()
    {
        throw new IllegalArgumentException("Value \"" + this.toString() + "\" is not a double value [" + this.getClass().getName() + "]");
    }

    /**
     * Returns this Value as a ReferenceValue.
     */
    public ReferenceValue referenceValue()
    {
        throw new IllegalArgumentException("Value \"" + this.toString() + "\" is not a reference value [" + this.getClass().getName() + "]");
    }

    /**
     * Returns this Value as an InstructionOffsetValue.
     */
    public InstructionOffsetValue instructionOffsetValue()
    {
        throw new IllegalArgumentException("Value \"" + this.toString() + "\" is not an instruction offset value [" + this.getClass().getName() + "]");
    }


    /**
     * Returns whether this Value represents a single specific (but possibly
     * unknown) value.
     */
    public boolean isSpecific()
    {
        return false;
    }


    /**
     * Returns whether this Value represents a single particular (known)
     * value.
     */
    public boolean isParticular()
    {
        return false;
    }


    /**
     * Returns the generalization of this Value and the given other Value.
     */
    public abstract Value generalize(Value other);


    /**
     * Returns whether the computational type of this Value is a category 2 type.
     * This means that it takes up the space of two category 1 types on the
     * stack, for instance.
     */
    public abstract boolean isCategory2();


    /**
     * Returns the computational type of this Value.
     * @return <code>TYPE_INTEGER</code>,
     *         <code>TYPE_LONG</code>,
     *         <code>TYPE_FLOAT</code>,
     *         <code>TYPE_DOUBLE</code>,
     *         <code>TYPE_REFERENCE</code>, or
     *         <code>TYPE_INSTRUCTION_OFFSET</code>.
     */
    public abstract int computationalType();


    /**
     * Returns the internal type of this Value.
     * @return <code>ClassConstants.TYPE_BOOLEAN</code>,
     *         <code>ClassConstants.TYPE_BYTE</code>,
     *         <code>ClassConstants.TYPE_CHAR</code>,
     *         <code>ClassConstants.TYPE_SHORT</code>,
     *         <code>ClassConstants.TYPE_INT</code>,
     *         <code>ClassConstants.TYPE_LONG</code>,
     *         <code>ClassConstants.TYPE_FLOAT</code>,
     *         <code>ClassConstants.TYPE_DOUBLE</code>,
     *         <code>ClassConstants.TYPE_CLASS_START ... ClassConstants.TYPE_CLASS_END</code>, or
     *         an array type containing any of these types (always as String).
     */
    public abstract String internalType();
}

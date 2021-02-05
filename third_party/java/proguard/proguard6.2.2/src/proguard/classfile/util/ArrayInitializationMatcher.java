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
package proguard.classfile.util;

import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.constant.Constant;
import proguard.classfile.instruction.*;
import proguard.evaluation.TracedStack;
import proguard.evaluation.value.*;
import proguard.optimize.evaluation.PartialEvaluator;
import proguard.optimize.peephole.InstructionSequenceReplacer;

/**
 * This class finds sequences of instructions that correspond to primitive
 * array initializations. Such initializations may be represented more
 * efficiently in other bytecode languages.
 *
 * @author Eric Lafortune
 */
public class ArrayInitializationMatcher
{
    private static final int X = InstructionSequenceReplacer.X;

    private final PartialEvaluator partialEvaluator;

    private int    arrayInitializationStart;
    private int    arrayInitializationEnd;
    private Object array;

    private final Constant[] CONSTANTS = new Constant[0];

    // Conversion with dex2jar might result in arrays
    // being pre-stored to a variable before initialization:
    //
    //   newarray
    //   astore X
    //   aload  X
    //   initialization start
    //
    private final Instruction[] ARRAY_PRESTORE_INSTRUCTIONS = new Instruction[]
        {
            new VariableInstruction(InstructionConstants.OP_ASTORE, X),
            new VariableInstruction(InstructionConstants.OP_ALOAD, X)
        };

    private final InstructionSequenceMatcher arrayPreStoreMatcher =
        new InstructionSequenceMatcher(CONSTANTS, ARRAY_PRESTORE_INSTRUCTIONS);


    /**
     * Creates a new ArrayInitializationMatcher.
     */
    public ArrayInitializationMatcher()
    {
        this(new PartialEvaluator());
    }


    /**
     * Creates a new ArrayInitializationMatcher that will use the given partial
     * evaluator. The evaluator must have been initialized before trying to
     * match any array initializations.
     * @param partialEvaluator the evaluator to be used for the analysis.
     */
    public ArrayInitializationMatcher(PartialEvaluator partialEvaluator)
    {
        this.partialEvaluator = partialEvaluator;
    }


    /**
     * Returns whether the code fragment starting at the specified newarray
     * instruction is followed by a static array initialization.
     * @param clazz               the class.
     * @param method              the method.
     * @param codeAttribute       the code attribute.
     * @param newArrayOffset      the offset of the newarray instruction.
     * @param newArrayInstruction the newarray instruction.
     * @return whether there is a static array initialization.
     */
    public boolean matchesArrayInitialization(Clazz             clazz,
                                              Method            method,
                                              CodeAttribute     codeAttribute,
                                              int               newArrayOffset,
                                              SimpleInstruction newArrayInstruction)
    {
        array = null;

        TracedStack stackBefore =
            partialEvaluator.getStackBefore(newArrayOffset);

        IntegerValue integerValue =
            stackBefore.getTop(0).integerValue();

        if (!integerValue.isParticular())
        {
            return false;
        }


        int arrayLength = integerValue.value();

        int newArrayType     = newArrayInstruction.constant;
        int arrayStoreOpcode = arrayStoreOpcode(newArrayType);

        byte[] code = codeAttribute.code;

        int         offset      = newArrayOffset;
        Instruction instruction = newArrayInstruction;

        int skipOffset = skipPreStoreInstructions(clazz, method, codeAttribute, offset + instruction.length(offset));
        if (skipOffset > 0)
        {
            offset         = skipOffset;
            newArrayOffset = offset;
            instruction    = InstructionFactory.create(code, offset);
        }

        // Remember the potential initialization start.
        int tmpInitializationStart = offset + instruction.length(offset);

        // Check if all the elements in the array are initialized.
        for (int index = 0; index < arrayLength; index++)
        {
            // Check if the array reference is pushed.
            instruction = InstructionFactory.create(code, offset += instruction.length(offset));

            if (instruction.stackPushCount(clazz) < 1 ||
                !partialEvaluator.getStackAfter(offset).getTopActualProducerValue(0).instructionOffsetValue().contains(newArrayOffset))
            {
                return false;
            }

            // Check that the array index is pushed.
            instruction = InstructionFactory.create(code, offset += instruction.length(offset));
            if (instruction.stackPushCount(clazz) != 1)
            {
                return false;
            }

            Value indexValue = partialEvaluator.getStackAfter(offset).getTop(0);
            if (indexValue.computationalType() != Value.TYPE_INTEGER ||
                !indexValue.integerValue().isParticular()            ||
                indexValue.integerValue().value() != index)
            {
                return false;
            }

            // Check if a particular value is pushed.
            instruction = InstructionFactory.create(code, offset += instruction.length(offset));
            if (instruction.stackPushCount(clazz) < 1 ||
                !partialEvaluator.getStackAfter(offset).getTop(0).isParticular())
            {
                return false;
            }

            Value elementValue = partialEvaluator.getStackAfter(offset).getTop(0);

            // Check if the value is stored in the array.
            instruction = InstructionFactory.create(code, offset += instruction.length(offset));
            if (instruction.opcode != arrayStoreOpcode)
            {
                return false;
            }

            if (index == 0)
            {
                array = newArray(newArrayType, arrayLength);
            }

            arrayStore(newArrayType, array, index, elementValue);
        }

        arrayInitializationStart = tmpInitializationStart;
        arrayInitializationEnd   = offset;

        return offset > newArrayOffset;
    }


    /**
     * Returns the offset to skip to after a new-array instruction.
     * <p>
     * This is a work-around for code converted by dex2jar. In some
     * cases, after an array has been created, it is immediately
     * stored into a variable and loaded again:
     * <pre>
     *   newarray
     *   astore X
     *   aload  X
     *   initialization
     * </pre>
     *
     * @param clazz            the class.
     * @param method           the method.
     * @param codeAttribute    the code attribute.
     * @param startOffset      the start offset.
     * @return the offset to skip to
     */
    private int skipPreStoreInstructions(Clazz         clazz,
                                         Method        method,
                                         CodeAttribute codeAttribute,
                                         int           startOffset)
    {
        final int instructionCount = arrayPreStoreMatcher.instructionCount();

        arrayPreStoreMatcher.reset();
        for (int count = 0, offset = startOffset;
             count < instructionCount && offset < codeAttribute.u4codeLength;
             count++)
        {
            Instruction instruction =
                InstructionFactory.create(codeAttribute.code, offset);

            instruction.accept(clazz, method, codeAttribute, offset,
                               arrayPreStoreMatcher);

            offset += instruction.length(offset);
        }

        if (arrayPreStoreMatcher.isMatching())
        {
            return arrayPreStoreMatcher.matchedInstructionOffset(instructionCount - 1);
        }

        return -1;
    }

    /**
     * Returns the first offset of the recent static array initialization, i.e. the first
     * initialization instruction after the newarray.
     * @see #matchesArrayInitialization
     */
    public int arrayInitializationStart()
    {
        return arrayInitializationStart;
    }

    /**
     * Returns the last offset of the recent static array initialization.
     * @see #matchesArrayInitialization
     */
    public int arrayInitializationEnd()
    {
        return arrayInitializationEnd;
    }

    /**
     * Returns the recent static array initialization.
     * @see #matchesArrayInitialization
     */
    public Object array()
    {
        return array;
    }


    private byte internalType(int newArrayType)
    {
        switch (newArrayType)
        {
            case InstructionConstants.ARRAY_T_BOOLEAN:
            case InstructionConstants.ARRAY_T_BYTE:
            case InstructionConstants.ARRAY_T_CHAR:
            case InstructionConstants.ARRAY_T_SHORT:
            case InstructionConstants.ARRAY_T_INT:     return Value.TYPE_INTEGER;
            case InstructionConstants.ARRAY_T_LONG:    return Value.TYPE_LONG;
            case InstructionConstants.ARRAY_T_FLOAT:   return Value.TYPE_FLOAT;
            case InstructionConstants.ARRAY_T_DOUBLE:  return Value.TYPE_DOUBLE;
            default:
                throw new IllegalArgumentException("Unexpected new array type ["+newArrayType+"]");
        }
    }


    private byte arrayStoreOpcode(int newArrayType)
    {
        switch (newArrayType)
        {
            case InstructionConstants.ARRAY_T_BOOLEAN:
            case InstructionConstants.ARRAY_T_BYTE:    return InstructionConstants.OP_BASTORE;
            case InstructionConstants.ARRAY_T_CHAR:    return InstructionConstants.OP_CASTORE;
            case InstructionConstants.ARRAY_T_SHORT:   return InstructionConstants.OP_SASTORE;
            case InstructionConstants.ARRAY_T_INT:     return InstructionConstants.OP_IASTORE;
            case InstructionConstants.ARRAY_T_LONG:    return InstructionConstants.OP_LASTORE;
            case InstructionConstants.ARRAY_T_FLOAT:   return InstructionConstants.OP_FASTORE;
            case InstructionConstants.ARRAY_T_DOUBLE:  return InstructionConstants.OP_DASTORE;
            default:
                throw new IllegalArgumentException("Unexpected new array type ["+newArrayType+"]");
        }
    }


    private Object newArray(int newArrayType, int arrayLength)
    {
        switch (newArrayType)
        {
            case InstructionConstants.ARRAY_T_BOOLEAN: return new boolean[arrayLength];
            case InstructionConstants.ARRAY_T_BYTE:    return new byte[arrayLength];
            case InstructionConstants.ARRAY_T_CHAR:    return new char[arrayLength];
            case InstructionConstants.ARRAY_T_SHORT:   return new short[arrayLength];
            case InstructionConstants.ARRAY_T_INT:     return new int[arrayLength];
            case InstructionConstants.ARRAY_T_LONG:    return new long[arrayLength];
            case InstructionConstants.ARRAY_T_FLOAT:   return new float[arrayLength];
            case InstructionConstants.ARRAY_T_DOUBLE:  return new double[arrayLength];
            default:
                throw new IllegalArgumentException("Unexpected new array type ["+newArrayType+"]");
        }
    }


    private void arrayStore(int newArrayType, Object array, int index, Value value)
    {
        switch (newArrayType)
        {
            case InstructionConstants.ARRAY_T_BOOLEAN: ((boolean[])array)[index] = 0 !=   value.integerValue().value(); break;
            case InstructionConstants.ARRAY_T_BYTE:    ((byte   [])array)[index] = (byte) value.integerValue().value(); break;
            case InstructionConstants.ARRAY_T_CHAR:    ((char   [])array)[index] = (char) value.integerValue().value(); break;
            case InstructionConstants.ARRAY_T_SHORT:   ((short  [])array)[index] = (short)value.integerValue().value(); break;
            case InstructionConstants.ARRAY_T_INT:     ((int    [])array)[index] =        value.integerValue().value(); break;
            case InstructionConstants.ARRAY_T_LONG:    ((long   [])array)[index] =        value.longValue().value();    break;
            case InstructionConstants.ARRAY_T_FLOAT:   ((float  [])array)[index] =        value.floatValue().value();   break;
            case InstructionConstants.ARRAY_T_DOUBLE:  ((double [])array)[index] =        value.doubleValue().value();  break;
            default:
                throw new IllegalArgumentException("Unexpected new array type ["+newArrayType+"]");
        }
    }
}
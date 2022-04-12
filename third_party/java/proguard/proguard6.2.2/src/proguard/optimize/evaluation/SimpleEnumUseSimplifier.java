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
package proguard.optimize.evaluation;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.editor.*;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.evaluation.value.*;
import proguard.optimize.info.SimpleEnumMarker;

/**
 * This AttributeVisitor simplifies the use of enums in the code attributes that
 * it visits.
 *
 * @see SimpleEnumMarker
 * @see MemberReferenceFixer
 * @author Eric Lafortune
 */
public class SimpleEnumUseSimplifier
extends      SimplifiedVisitor
implements   AttributeVisitor,
             InstructionVisitor,
             ConstantVisitor,
             ParameterVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("enum") != null;
    //*/


    private final InstructionVisitor extraInstructionVisitor;

    private final PartialEvaluator    partialEvaluator;
    private final CodeAttributeEditor codeAttributeEditor = new CodeAttributeEditor(true, true);
    private final ConstantVisitor     nullParameterFixer  = new ReferencedMemberVisitor(new AllParameterVisitor(false, this));

    // Fields acting as parameters and return values for the visitor methods.
    private Clazz         invocationClazz;
    private Method        invocationMethod;
    private CodeAttribute invocationCodeAttribute;
    private int           invocationOffset;
    private boolean       isSimpleEnum;


    /**
     * Creates a new SimpleEnumUseSimplifier.
     */
    public SimpleEnumUseSimplifier()
    {
        this(new PartialEvaluator(new TypedReferenceValueFactory()), null);
    }


    /**
     * Creates a new SimpleEnumDescriptorSimplifier.
     * @param partialEvaluator        the partial evaluator that will
     *                                execute the code and provide
     *                                information about the results.
     * @param extraInstructionVisitor an optional extra visitor for all
     *                                simplified instructions.
     */
    public SimpleEnumUseSimplifier(PartialEvaluator   partialEvaluator,
                                   InstructionVisitor extraInstructionVisitor)
    {
        this.partialEvaluator        = partialEvaluator;
        this.extraInstructionVisitor = extraInstructionVisitor;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        if (DEBUG)
        {
            System.out.println("SimpleEnumUseSimplifier: "+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz));
        }

        // Skip the non-static methods of simple enum classes.
        if (SimpleEnumMarker.isSimpleEnum(clazz) &&
            (method.getAccessFlags() & ClassConstants.ACC_STATIC) == 0)
        {
            return;
        }

        // Evaluate the method.
        partialEvaluator.visitCodeAttribute(clazz, method, codeAttribute);

        int codeLength = codeAttribute.u4codeLength;

        // Reset the code changes.
        codeAttributeEditor.reset(codeLength);

        // Replace any instructions that can be simplified.
        for (int offset = 0; offset < codeLength; offset++)
        {
            if (partialEvaluator.isTraced(offset))
            {
                Instruction instruction = InstructionFactory.create(codeAttribute.code,
                                                                    offset);

                instruction.accept(clazz, method, codeAttribute, offset, this);
            }
        }

        // Apply all accumulated changes to the code.
        codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
    }


    // Implementations for InstructionVisitor.

    public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
    {
        switch (simpleInstruction.opcode)
        {
            case InstructionConstants.OP_AALOAD:
            {
                if (isPushingSimpleEnum(offset))
                {
                    // Load a simple enum integer from an integer array.
                    replaceInstruction(clazz,
                                       offset,
                                       simpleInstruction,
                                       new SimpleInstruction(InstructionConstants.OP_IALOAD));
                }
                break;
            }
            case InstructionConstants.OP_AASTORE:
            {
                if (isPoppingSimpleEnumArray(offset, 2))
                {
                    // Store a simple enum integer in an integer array.
                    replaceInstruction(clazz,
                                       offset,
                                       simpleInstruction,
                                       new SimpleInstruction(InstructionConstants.OP_IASTORE));

                    // Replace any producers of null constants.
                    replaceNullStackEntryProducers(clazz, method, codeAttribute, offset);
                }
                break;
            }
            case InstructionConstants.OP_ARETURN:
            {
                if (isReturningSimpleEnum(clazz, method))
                {
                    // Return a simple enum integer instead of an enum.
                    replaceInstruction(clazz,
                                       offset,
                                       simpleInstruction,
                                       new SimpleInstruction(InstructionConstants.OP_IRETURN));

                    // Replace any producers of null constants.
                    replaceNullStackEntryProducers(clazz, method, codeAttribute, offset);
                }
                break;
            }
        }
    }


    public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
    {
        int variableIndex = variableInstruction.variableIndex;

        switch (variableInstruction.opcode)
        {
            case InstructionConstants.OP_ALOAD:
            case InstructionConstants.OP_ALOAD_0:
            case InstructionConstants.OP_ALOAD_1:
            case InstructionConstants.OP_ALOAD_2:
            case InstructionConstants.OP_ALOAD_3:
            {
                if (isPushingSimpleEnum(offset))
                {
                    // Load a simple enum integer instead of an enum.
                    replaceInstruction(clazz,
                                       offset,
                                       variableInstruction,
                                       new VariableInstruction(InstructionConstants.OP_ILOAD,
                                                               variableIndex));

                    // Replace any producers of null constants.
                    replaceNullVariableProducers(clazz,
                                                 method,
                                                 codeAttribute,
                                                 offset,
                                                 variableIndex);
                }
                break;
            }
            case InstructionConstants.OP_ASTORE:
            case InstructionConstants.OP_ASTORE_0:
            case InstructionConstants.OP_ASTORE_1:
            case InstructionConstants.OP_ASTORE_2:
            case InstructionConstants.OP_ASTORE_3:
            {
                if (!partialEvaluator.isSubroutineStart(offset) &&
                    isPoppingSimpleEnum(offset))
                {
                    // Store a simple enum integer instead of an enum.
                    replaceInstruction(clazz,
                                       offset,
                                       variableInstruction,
                                       new VariableInstruction(InstructionConstants.OP_ISTORE,
                                                               variableIndex));

                    // Replace any producers of null constants.
                    replaceNullStackEntryProducers(clazz, method, codeAttribute, offset);
                }
                break;
            }
        }
    }


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        switch (constantInstruction.opcode)
        {
            case InstructionConstants.OP_PUTSTATIC:
            case InstructionConstants.OP_PUTFIELD:
            {
                // Replace any producers of null constants.
                invocationClazz         = clazz;
                invocationMethod        = method;
                invocationCodeAttribute = codeAttribute;
                invocationOffset        = offset;
                clazz.constantPoolEntryAccept(constantInstruction.constantIndex,
                                              nullParameterFixer);
                break;
            }
            case InstructionConstants.OP_INVOKEVIRTUAL:
            {
                // Check if the instruction is calling a simple enum.
                String invokedMethodName =
                    clazz.getRefName(constantInstruction.constantIndex);
                String invokedMethodType =
                    clazz.getRefType(constantInstruction.constantIndex);
                int stackEntryIndex =
                    ClassUtil.internalMethodParameterSize(invokedMethodType);
                if (isPoppingSimpleEnum(offset, stackEntryIndex))
                {
                    replaceSupportedMethod(clazz,
                                           offset,
                                           constantInstruction,
                                           invokedMethodName,
                                           invokedMethodType);
                }

                // Fall through to check the parameters.
            }
            case InstructionConstants.OP_INVOKESPECIAL:
            case InstructionConstants.OP_INVOKESTATIC:
            case InstructionConstants.OP_INVOKEINTERFACE:
            {
                // Replace any producers of null constants.
                invocationClazz         = clazz;
                invocationMethod        = method;
                invocationCodeAttribute = codeAttribute;
                invocationOffset        = offset;
                clazz.constantPoolEntryAccept(constantInstruction.constantIndex,
                                              nullParameterFixer);
                break;
            }
            case InstructionConstants.OP_ANEWARRAY:
            {
                int constantIndex = constantInstruction.constantIndex;

                if (isReferencingSimpleEnum(clazz, constantIndex) &&
                    !ClassUtil.isInternalArrayType(clazz.getClassName(constantIndex)))
                {
                    // Create an integer array instead of an enum array.
                    replaceInstruction(clazz,
                                       offset,
                                       constantInstruction,
                                       new SimpleInstruction(InstructionConstants.OP_NEWARRAY,
                                                             InstructionConstants.ARRAY_T_INT));
                }
                break;
            }
            case InstructionConstants.OP_CHECKCAST:
            {
                if (isPoppingSimpleEnum(offset))
                {
                    // Enum classes can only be simple if the checkcast
                    // succeeds, so we can delete it.
                    deleteInstruction(clazz,
                                      offset,
                                      constantInstruction);

                    // Replace any producers of null constants.
                    replaceNullStackEntryProducers(clazz, method, codeAttribute, offset);
                }
                break;
            }
            case InstructionConstants.OP_INSTANCEOF:
            {
                if (isPoppingSimpleEnum(offset))
                {
                    // Enum classes can only be simple if the instanceof
                    // succeeds, so we can push a constant result.
                    replaceInstruction(clazz,
                                       offset,
                                       constantInstruction,
                                       new SimpleInstruction(InstructionConstants.OP_ICONST_1));

                    // Replace any producers of null constants.
                    replaceNullStackEntryProducers(clazz, method, codeAttribute, offset);
                }
                break;
            }
        }
    }


    public void visitBranchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction branchInstruction)
    {
        switch (branchInstruction.opcode)
        {
            case InstructionConstants.OP_IFACMPEQ:
            {
                if (isPoppingSimpleEnum(offset))
                {
                    // Compare simple enum integers instead of enums.
                    replaceInstruction(clazz,
                                       offset,
                                       branchInstruction,
                                       new BranchInstruction(InstructionConstants.OP_IFICMPEQ,
                                                             branchInstruction.branchOffset));

                    // Replace any producers of null constants.
                    replaceNullStackEntryProducers(clazz, method, codeAttribute, offset, 0);
                    replaceNullStackEntryProducers(clazz, method, codeAttribute, offset, 1);
                }
                break;
            }
            case InstructionConstants.OP_IFACMPNE:
            {
                if (isPoppingSimpleEnum(offset))
                {
                    // Compare simple enum integers instead of enums.
                    replaceInstruction(clazz,
                                       offset,
                                       branchInstruction,
                                       new BranchInstruction(InstructionConstants.OP_IFICMPNE,
                                                             branchInstruction.branchOffset));

                    // Replace any producers of null constants.
                    replaceNullStackEntryProducers(clazz, method, codeAttribute, offset, 0);
                    replaceNullStackEntryProducers(clazz, method, codeAttribute, offset, 1);
                }
                break;
            }
            case InstructionConstants.OP_IFNULL:
            {
                if (isPoppingSimpleEnum(offset))
                {
                    // Compare with 0 instead of null.
                    replaceInstruction(clazz,
                                       offset,
                                       branchInstruction,
                                       new BranchInstruction(
                                           InstructionConstants.OP_IFEQ,
                                           branchInstruction.branchOffset));
                }
                break;
            }
            case InstructionConstants.OP_IFNONNULL:
            {
                if (isPoppingSimpleEnum(offset))
                {
                    // Compare with 0 instead of null.
                    replaceInstruction(clazz,
                                       offset,
                                       branchInstruction,
                                       new BranchInstruction(InstructionConstants.OP_IFNE,
                                                             branchInstruction.branchOffset));
                }
                break;
            }
        }
    }


    public void visitAnySwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SwitchInstruction switchInstruction)
    {
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        // Does the constant refer to a simple enum type?
        isSimpleEnum = isSimpleEnum(stringConstant.referencedClass);
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        // Does the constant refer to a simple enum type?
        isSimpleEnum = isSimpleEnum(classConstant.referencedClass);
    }


    // Implementations for ParameterVisitor.

    public void visitParameter(Clazz clazz, Member member, int parameterIndex, int parameterCount, int parameterOffset, int parameterSize, String parameterType, Clazz referencedClass)
    {
        // Check if the parameter is passing a simple enum as a more general type.
        if (!ClassUtil.isInternalPrimitiveType(parameterType.charAt(0)) &&
            !ClassUtil.isInternalArrayType(parameterType)               &&
            isSimpleEnum(referencedClass))
        {
            // Replace any producers of null constants for this parameter.
            int stackEntryIndex = parameterSize - parameterOffset - 1;

            replaceNullStackEntryProducers(invocationClazz,
                                           invocationMethod,
                                           invocationCodeAttribute,
                                           invocationOffset,
                                           stackEntryIndex);
        }
    }


    // Small utility methods.

    /**
     * Returns whether the constant at the given offset is referencing a
     * simple enum class.
     */
    private boolean isReferencingSimpleEnum(Clazz clazz, int constantIndex)
    {
        isSimpleEnum = false;

        clazz.constantPoolEntryAccept(constantIndex, this);

        return isSimpleEnum;
    }


    /**
     * Returns whether the given method is returning a simple enum class.
     */
    private boolean isReturningSimpleEnum(Clazz clazz, Method method)
    {
        String descriptor = method.getDescriptor(clazz);
        String returnType = ClassUtil.internalMethodReturnType(descriptor);

        if (ClassUtil.isInternalClassType(returnType) &&
            !ClassUtil.isInternalArrayType(returnType))
        {
            Clazz[] referencedClasses =
                ((ProgramMethod)method).referencedClasses;

            if (referencedClasses != null)
            {
                Clazz returnedClass =
                    referencedClasses[referencedClasses.length - 1];

                return isSimpleEnum(returnedClass);
            }
        }

        return false;
    }


    /**
     * Returns whether the instruction at the given offset is pushing a simple
     * enum class.
     */
    private boolean isPushingSimpleEnum(int offset)
    {
        ReferenceValue referenceValue =
            partialEvaluator.getStackAfter(offset).getTop(0).referenceValue();

        Clazz referencedClass = referenceValue.getReferencedClass();

        return isSimpleEnum(referencedClass) &&
               !ClassUtil.isInternalArrayType(referenceValue.getType());
    }


    /**
     * Returns whether the instruction at the given offset is popping a simple
     * enum class.
     */
    private boolean isPoppingSimpleEnum(int offset)
    {
        return isPoppingSimpleEnum(offset, 0);
    }


    /**
     * Returns whether the instruction at the given offset is popping a simple
     * enum class.
     */
    private boolean isPoppingSimpleEnum(int offset, int stackEntryIndex)
    {
        ReferenceValue referenceValue =
            partialEvaluator.getStackBefore(offset).getTop(stackEntryIndex).referenceValue();

        return isSimpleEnum(referenceValue.getReferencedClass()) &&
               !ClassUtil.isInternalArrayType(referenceValue.getType());
    }


    /**
     * Returns whether the instruction at the given offset is popping a simple
     * enum type. This includes simple enum arrays.
     */
    private boolean isPoppingSimpleEnumType(int offset, int stackEntryIndex)
    {
        ReferenceValue referenceValue =
            partialEvaluator.getStackBefore(offset).getTop(stackEntryIndex).referenceValue();

        return isSimpleEnum(referenceValue.getReferencedClass());
    }


    /**
     * Returns whether the instruction at the given offset is popping a
     * one-dimensional simple enum array.
     */
    private boolean isPoppingSimpleEnumArray(int offset, int stackEntryIndex)
    {
        ReferenceValue referenceValue =
            partialEvaluator.getStackBefore(offset).getTop(stackEntryIndex).referenceValue();

        return isSimpleEnum(referenceValue.getReferencedClass()) &&
               ClassUtil.internalArrayTypeDimensionCount(referenceValue.getType()) == 1;
    }


    /**
     * Returns whether the given class is not null and a simple enum class.
     */
    private boolean isSimpleEnum(Clazz clazz)
    {
        return clazz != null &&
               SimpleEnumMarker.isSimpleEnum(clazz);
    }


    /**
     * Returns whether the specified enum method is supported for simple enums.
     */
    private void replaceSupportedMethod(Clazz       clazz,
                                        int         offset,
                                        Instruction instruction,
                                        String      name,
                                        String      type)
    {
        if (name.equals(ClassConstants.METHOD_NAME_ORDINAL) &&
            type.equals(ClassConstants.METHOD_TYPE_ORDINAL))
        {
            Instruction[] replacementInstructions = new Instruction[]
            {
                new SimpleInstruction(InstructionConstants.OP_ICONST_1),
                new SimpleInstruction(InstructionConstants.OP_ISUB),
            };

            replaceInstructions(clazz,
                                offset,
                                instruction,
                                replacementInstructions);
        }
    }


    /**
     * Replaces the instruction at the given offset by the given instructions.
     */
    private void replaceInstructions(Clazz         clazz,
                                     int           offset,
                                     Instruction   instruction,
                                     Instruction[] replacementInstructions)
    {
        if (DEBUG) System.out.println("  Replacing instruction "+instruction.toString(offset)+" -> "+replacementInstructions.length+" instructions");

        codeAttributeEditor.replaceInstruction(offset, replacementInstructions);

        // Visit the instruction, if required.
        if (extraInstructionVisitor != null)
        {
            // Note: we're not passing the right arguments for now, knowing that
            // they aren't used anyway.
            instruction.accept(clazz, null, null, offset, extraInstructionVisitor);
        }
    }


    /**
     * Replaces the instruction at the given offset by the given instruction,
     * popping any now unused stack entries.
     */
    private void replaceInstruction(Clazz       clazz,
                                    int         offset,
                                    Instruction instruction,
                                    Instruction replacementInstruction)
    {
        // Pop unneeded stack entries if necessary.
        int popCount =
            instruction.stackPopCount(clazz) -
            replacementInstruction.stackPopCount(clazz);

        insertPopInstructions(offset, popCount);

        if (DEBUG) System.out.println("  Replacing instruction "+instruction.toString(offset)+" -> "+replacementInstruction.toString()+(popCount == 0 ? "" : " ("+popCount+" pops)"));

        codeAttributeEditor.replaceInstruction(offset, replacementInstruction);

        // Visit the instruction, if required.
        if (extraInstructionVisitor != null)
        {
            // Note: we're not passing the right arguments for now, knowing that
            // they aren't used anyway.
            instruction.accept(clazz, null, null, offset, extraInstructionVisitor);
        }
    }


    /**
     * Deletes the instruction at the given offset, popping any now unused
     * stack entries.
     */
    private void deleteInstruction(Clazz       clazz,
                                    int         offset,
                                    Instruction instruction)
    {
        // Pop unneeded stack entries if necessary.
        //int popCount = instruction.stackPopCount(clazz);
        //
        //insertPopInstructions(offset, popCount);
        //
        //if (DEBUG) System.out.println("  Deleting instruction "+instruction.toString(offset)+(popCount == 0 ? "" : " ("+popCount+" pops)"));

        if (DEBUG) System.out.println("  Deleting instruction "+instruction.toString(offset));

        codeAttributeEditor.deleteInstruction(offset);

        // Visit the instruction, if required.
        if (extraInstructionVisitor != null)
        {
            // Note: we're not passing the right arguments for now, knowing that
            // they aren't used anyway.
            instruction.accept(clazz, null, null, offset, extraInstructionVisitor);
        }
    }


    /**
     * Pops the given number of stack entries before the instruction at the
     * given offset.
     */
    private void insertPopInstructions(int offset, int popCount)
    {
        switch (popCount)
        {
            case 0:
            {
                break;
            }
            case 1:
            {
                // Insert a single pop instruction.
                Instruction popInstruction =
                    new SimpleInstruction(InstructionConstants.OP_POP);

                codeAttributeEditor.insertBeforeInstruction(offset,
                                                            popInstruction);
                break;
            }
            case 2:
            {
                // Insert a single pop2 instruction.
                Instruction popInstruction =
                    new SimpleInstruction(InstructionConstants.OP_POP2);

                codeAttributeEditor.insertBeforeInstruction(offset,
                                                            popInstruction);
                break;
            }
            default:
            {
                // Insert the specified number of pop instructions.
                Instruction[] popInstructions =
                    new Instruction[popCount / 2 + popCount % 2];

                Instruction popInstruction =
                    new SimpleInstruction(InstructionConstants.OP_POP2);

                for (int index = 0; index < popCount / 2; index++)
                {
                      popInstructions[index] = popInstruction;
                }

                if (popCount % 2 == 1)
                {
                    popInstruction =
                        new SimpleInstruction(InstructionConstants.OP_POP);

                    popInstructions[popCount / 2] = popInstruction;
                }

                codeAttributeEditor.insertBeforeInstruction(offset,
                                                            popInstructions);
                break;
            }
        }
    }


    /**
     * Replaces aconst_null producers of the consumer of the top stack entry
     * at the given offset by iconst_0.
     */
    private void replaceNullStackEntryProducers(Clazz         clazz,
                                                Method        method,
                                                CodeAttribute codeAttribute,
                                                int           consumerOffset)
    {
        replaceNullStackEntryProducers(clazz, method, codeAttribute, consumerOffset, 0);
    }


    /**
     * Turn null reference producers of the specified stack entry into 0 int
     * producers. The partial evaluator generally can't identify them as
     * simple enums.
     */
    private void replaceNullStackEntryProducers(Clazz         clazz,
                                                Method        method,
                                                CodeAttribute codeAttribute,
                                                int           consumerOffset,
                                                int           stackEntryIndex)
    {
        InstructionOffsetValue producerOffsets =
            partialEvaluator.getStackBefore(consumerOffset).getTopActualProducerValue(stackEntryIndex).instructionOffsetValue();

        for (int index = 0; index < producerOffsets.instructionOffsetCount(); index++)
        {
            // Is the producer always pushing null?
            int producerOffset = producerOffsets.instructionOffset(index);
            if (producerOffset >= 0 &&
                partialEvaluator.getStackAfter(producerOffset).getTop(0).referenceValue().isNull() == Value.ALWAYS)
            {
                Instruction producerInstruction =
                    InstructionFactory.create(codeAttribute.code[producerOffset]);

                // Is it a simple case?
                switch (producerInstruction.opcode)
                {
                    case InstructionConstants.OP_ACONST_NULL:
                    case InstructionConstants.OP_ALOAD:
                    case InstructionConstants.OP_ALOAD_0:
                    case InstructionConstants.OP_ALOAD_1:
                    case InstructionConstants.OP_ALOAD_2:
                    case InstructionConstants.OP_ALOAD_3:
                    {
                        // Replace pushing null by pushing 0.
                        replaceInstruction(clazz,
                                           producerOffset,
                                           producerInstruction,
                                           new SimpleInstruction(InstructionConstants.OP_ICONST_0));
                        break;
                    }
                    default:
                    {
                        // Otherwise pop the null and then push 0.
                        replaceInstructions(clazz,
                                            producerOffset,
                                            producerInstruction,
                                            new Instruction[]
                                            {
                                                producerInstruction,
                                                new SimpleInstruction(InstructionConstants.OP_POP),
                                                new SimpleInstruction(InstructionConstants.OP_ICONST_0)
                                            });
                        break;
                    }
                }
            }
        }
    }


    /**
     * Turn null reference producers of the specified reference variable into
     * 0 int producers. The partial evaluator generally can't identify them
     * as simple enums.
     */
    private void replaceNullVariableProducers(Clazz         clazz,
                                              Method        method,
                                              CodeAttribute codeAttribute,
                                              int           consumerOffset,
                                              int           variableIndex)
    {
        InstructionOffsetValue producerOffsets =
            partialEvaluator.getVariablesBefore(consumerOffset).getProducerValue(variableIndex).instructionOffsetValue();

        for (int index = 0; index < producerOffsets.instructionOffsetCount(); index++)
        {
            if (!producerOffsets.isMethodParameter(index) &&
                !producerOffsets.isExceptionHandler(index))
            {
                // Is the producer always storing null?
                int producerOffset = producerOffsets.instructionOffset(index);
                if (partialEvaluator.getVariablesAfter(producerOffset).getValue(variableIndex).referenceValue().isNull() == Value.ALWAYS)
                {
                    // Replace storing the null reference value by storing an
                    // int value.
                    replaceInstruction(clazz,
                                       producerOffset,
                                       new VariableInstruction(InstructionConstants.OP_ASTORE, variableIndex),
                                       new VariableInstruction(InstructionConstants.OP_ISTORE, variableIndex));

                    // Replace pushing null by pushing 0.
                    replaceNullStackEntryProducers(clazz, method, codeAttribute, producerOffset);
                }
            }
        }
    }
}

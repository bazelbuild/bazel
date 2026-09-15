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
package proguard.preverify;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.preverification.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.editor.*;
import proguard.classfile.instruction.InstructionConstants;
import proguard.classfile.util.*;
import proguard.classfile.visitor.ClassPrinter;
import proguard.evaluation.*;
import proguard.evaluation.value.*;
import proguard.optimize.evaluation.*;

import java.util.*;

/**
 * This class can preverify methods in program class pools, according to a given
 * specification.
 *
 * @author Eric Lafortune
 */
public class CodePreverifier
extends      SimplifiedVisitor
implements   AttributeVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("cp") != null;
    //*/

    private static final int AT_METHOD_ENTRY = -1;


    private final boolean microEdition;

    private final ReferenceTracingValueFactory referenceTracingValueFactory = new ReferenceTracingValueFactory(new TypedReferenceValueFactory());
    private final PartialEvaluator             partialEvaluator             = new PartialEvaluator(referenceTracingValueFactory,
                                                                                                   new ReferenceTracingInvocationUnit(new BasicInvocationUnit(referenceTracingValueFactory)),
                                                                                                   true,
                                                                                                   referenceTracingValueFactory);
    private final InitializationFinder         initializationFinder         = new InitializationFinder(partialEvaluator, false);
    private final LivenessAnalyzer             livenessAnalyzer             = new LivenessAnalyzer(partialEvaluator, false, initializationFinder, false);
    private final CodeAttributeEditor          codeAttributeEditor          = new CodeAttributeEditor();


    /**
     * Creates a new CodePreverifier.
     */
    public CodePreverifier(boolean microEdition)
    {
        this.microEdition = microEdition;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // TODO: Remove this when the preverifier has stabilized.
        // Catch any unexpected exceptions from the actual visiting method.
        try
        {
            // Process the code.
            visitCodeAttribute0(clazz, method, codeAttribute);
        }
        catch (RuntimeException ex)
        {
            System.err.println("Unexpected error while preverifying:");
            System.err.println("  Class       = ["+clazz.getName()+"]");
            System.err.println("  Method      = ["+method.getName(clazz)+method.getDescriptor(clazz)+"]");
            System.err.println("  Exception   = ["+ex.getClass().getName()+"] ("+ex.getMessage()+")");

            throw ex;
        }
    }


    public void visitCodeAttribute0(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
//        DEBUG =
//            clazz.getName().equals("abc/Def") &&
//            method.getName(clazz).equals("abc");

        ProgramClass  programClass  = (ProgramClass)clazz;
        ProgramMethod programMethod = (ProgramMethod)method;

        int codeLength = codeAttribute.u4codeLength;

        // Evaluate the method.
        partialEvaluator.visitCodeAttribute(clazz, method, codeAttribute);
        initializationFinder.visitCodeAttribute(clazz, method, codeAttribute);
        livenessAnalyzer.visitCodeAttribute(clazz, method, codeAttribute);

        // We may have to remove unreachable code.
        codeAttributeEditor.reset(codeLength);

        // Collect the stack map frames.
        List stackMapFrameList = new ArrayList();

        for (int offset = 0; offset < codeLength; offset++)
        {
            // Only store frames at the beginning of code blocks.
            if (!partialEvaluator.isTraced(offset))
            {
                // Mark the unreachable instruction for deletion.
                codeAttributeEditor.deleteInstruction(offset);
            }
            else if (partialEvaluator.isBranchOrExceptionTarget(offset))
            {
                // Convert the variable values to types.
                VerificationType[] variableTypes =
                    correspondingVerificationTypes(programClass,
                                                   programMethod,
                                                   codeAttribute,
                                                   offset,
                                                   partialEvaluator.getVariablesBefore(offset));

                // Convert the stack values to types.
                VerificationType[] stackTypes =
                    correspondingVerificationTypes(programClass,
                                                   programMethod,
                                                   codeAttribute,
                                                   offset,
                                                   partialEvaluator.getStackBefore(offset));
                // Create and store a new frame.
                stackMapFrameList.add(new FullFrame(offset,
                                                    variableTypes,
                                                    stackTypes));
            }
        }

        // Compress the stack map frames if the target is not Java Micro Edition.
        if (!microEdition && !stackMapFrameList.isEmpty())
        {
            // Convert the initial variable values to types.
            VerificationType[] initialVariables =
                correspondingVerificationTypes(programClass,
                                               programMethod,
                                               codeAttribute,
                                               AT_METHOD_ENTRY,
                                               partialEvaluator.getVariablesBefore(0));

            // Special case: the <init> method.
            if (method.getName(programClass).equals(ClassConstants.METHOD_NAME_INIT))
            {
                initialVariables[0] = VerificationTypeFactory.createUninitializedThisType();
            }

            compressStackMapFrames(initialVariables,
                                   stackMapFrameList);
        }

        // Get the proper name for the attribute to be added/replaced/deleted.
        String stackMapAttributeName = microEdition ?
             ClassConstants.ATTR_StackMap :
             ClassConstants.ATTR_StackMapTable;

        int frameCount = stackMapFrameList.size();

        if (DEBUG)
        {
            Attribute originalStackMapAttribute = codeAttribute.getAttribute(clazz,
                                                                             stackMapAttributeName);

            if (originalStackMapAttribute != null)
            {
                int originalFrameCount = microEdition ?
                    ((StackMapAttribute)originalStackMapAttribute).u2stackMapFramesCount :
                    ((StackMapTableAttribute)originalStackMapAttribute).u2stackMapFramesCount;

                StackMapFrame[] originalFrames = microEdition ?
                    ((StackMapAttribute)originalStackMapAttribute).stackMapFrames :
                    ((StackMapTableAttribute)originalStackMapAttribute).stackMapFrames;

                if (frameCount != originalFrameCount ||
                    !Arrays.equals(stackMapFrameList.toArray(), originalFrames))
                {
                    System.out.println("Original preverification ["+clazz.getName()+"]:");
                    new ClassPrinter().visitProgramMethod(programClass, programMethod);
                }
            }
            else if (frameCount != 0)
            {
                System.out.println("Original preverification empty ["+clazz.getName()+"."+method.getName(clazz)+"]");
            }
        }

        if (frameCount == 0)
        {
            // Remove any stack map (table) attribute from the code attribute.
            new AttributesEditor(programClass, programMethod, codeAttribute, true).deleteAttribute(stackMapAttributeName);
        }
        else
        {
            Attribute stackMapAttribute;

            // Create the appropriate attribute.
            if (microEdition)
            {
                // Copy the frames into an array.
                FullFrame[] stackMapFrames = new FullFrame[frameCount];
                stackMapFrameList.toArray(stackMapFrames);

                // Put the frames into a stack map attribute.
                stackMapAttribute = new StackMapAttribute(stackMapFrames);
            }
            else
            {
                // Copy the frames into an array.
                StackMapFrame[] stackMapFrames = new StackMapFrame[frameCount];
                stackMapFrameList.toArray(stackMapFrames);

                // Put the frames into a stack map table attribute.
                stackMapAttribute = new StackMapTableAttribute(stackMapFrames);
            }

            // Fill out the name of the stack map attribute.
            stackMapAttribute.u2attributeNameIndex =
                new ConstantPoolEditor(programClass).addUtf8Constant(stackMapAttributeName);

            // Add the new stack map (table) attribute to the code attribute.
            new AttributesEditor(programClass, programMethod, codeAttribute, true).addAttribute(stackMapAttribute);

            if (DEBUG)
            {
                System.out.println("Preverifier ["+programClass.getName()+"."+programMethod.getName(programClass)+"]:");
                stackMapAttribute.accept(programClass, programMethod, codeAttribute, new ClassPrinter());
            }
        }

        // Apply code modifications, deleting unreachable code.
        codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
    }


    // Small utility methods.

    /**
     * Creates and returns the verification types corresponding to the given
     * variables. If necessary, class constants are added to the constant pool
     * of the given class.
     */
    private VerificationType[] correspondingVerificationTypes(ProgramClass    programClass,
                                                              ProgramMethod   programMethod,
                                                              CodeAttribute   codeAttribute,
                                                              int             offset,
                                                              TracedVariables variables)
    {
        int typeCount = 0;
        if (offset == AT_METHOD_ENTRY)
        {
            // Count the number of parameters, including any parameters
            // that are unused and overwritten right away.
            typeCount = ClassUtil.internalMethodParameterCount(
                programMethod.getDescriptor(programClass),
                programMethod.getAccessFlags());
        }
        else
        {
            typeCount = 0;

            int maximumVariablesSize = variables.size();
            int typeIndex = 0;

            // Count the the number of verification types, ignoring any nulls
            // at the end.
            for (int index = 0; index < maximumVariablesSize; index++)
            {
                Value value = variables.getValue(index);

                typeIndex++;

                // Remember the maximum live type (or uninitialized "this"
                // type) index. A dead uninitialized "this" is not possible in
                // plain Java code, but it is possible in optimized code and
                // in other languages like Kotlin (in exception handlers).
                // It has to be marked too ("flagThisUninit" in the JVM specs).
                if (value != null &&
                    (livenessAnalyzer.isAliveBefore(offset, index) ||
                     isUninitalizedThis(offset, index)))
                {
                    typeCount = typeIndex;

                    // Category 2 types that are alive are stored as single
                    // entries.
                    if (value.isCategory2())
                    {
                        index++;
                    }
                }
            }
        }

        // Create and fill out the verification types.
        VerificationType[] types = new VerificationType[typeCount];

        int typeIndex = 0;

        // Note the slightly different terminating condition, because the
        // types may have been truncated.
        for (int index = 0; typeIndex < typeCount; index++)
        {
            Value value         = variables.getValue(index);
            Value producerValue = variables.getProducerValue(index);

            // Fill out the type.
            VerificationType type;

            // Is the value not null and alive (or uninitialized "this")?
            if (value != null &&
                (offset == AT_METHOD_ENTRY ||
                 livenessAnalyzer.isAliveBefore(offset, index) ||
                 isUninitalizedThis(offset, index)))
            {
                type = correspondingVerificationType(programClass,
                                                     programMethod,
                                                     codeAttribute,
                                                     offset,
                                                     value,
                                                     producerValue);

                // Category 2 types that are alive are stored as single entries.
                if (value.isCategory2())
                {
                    index++;
                }
            }
            else
            {
                // A null value at method entry means that there was a branch to
                // offset 0 that has cleared the value. Then pick a dummy value so
                // it never matches the next frame at offset 0.
                type = offset == AT_METHOD_ENTRY ?
                    VerificationTypeFactory.createUninitializedThisType() :
                    VerificationTypeFactory.createTopType();
            }

            types[typeIndex++] = type;
        }

        return types;
    }


    /**
     * Creates and returns the verification types corresponding to the given
     * stack. If necessary, class constants are added to the constant pool
     * of the given class.
     */
    private VerificationType[] correspondingVerificationTypes(ProgramClass  programClass,
                                                              ProgramMethod programMethod,
                                                              CodeAttribute codeAttribute,
                                                              int           offset,
                                                              TracedStack   stack)
    {
        int maximumStackSize = stack.size();
        int typeCount = 0;

        // Count the the number of verification types.
        for (int index = 0; index < maximumStackSize; index++)
        {
            // We have to work down from the top of the stack.
            Value value = stack.getTop(index);

            typeCount++;

            // Category 2 types are stored as single entries.
            if (value.isCategory2())
            {
                index++;
            }
        }

        // Create and fill out the verification types.
        VerificationType[] types = new VerificationType[typeCount];

        int typeIndex = typeCount;

        for (int index = 0; index < maximumStackSize; index++)
        {
            // We have to work down from the top of the stack.
            Value value         = stack.getTop(index);
            Value producerValue = stack.getTopProducerValue(index);

            // Fill out the type.
            types[--typeIndex] =
                correspondingVerificationType(programClass,
                                              programMethod,
                                              codeAttribute,
                                              offset,
                                              value,
                                              producerValue);

            // Category 2 types are stored as single entries.
            if (value.isCategory2())
            {
                index++;
            }
        }

        return types;
    }


    /**
     * Creates and returns the verification type corresponding to the given
     * value. If necessary, a class constant is added to the constant pool of
     * the given class.
     */
    private VerificationType correspondingVerificationType(ProgramClass  programClass,
                                                           ProgramMethod programMethod,
                                                           CodeAttribute codeAttribute,
                                                           int           offset,
                                                           Value         value,
                                                           Value         producerValue)
    {
        if (value == null)
        {
            return VerificationTypeFactory.createTopType();
        }

        int type = value.computationalType();

        switch (type)
        {
            case Value.TYPE_INSTRUCTION_OFFSET:
            case Value.TYPE_INTEGER:   return VerificationTypeFactory.createIntegerType();
            case Value.TYPE_LONG:      return VerificationTypeFactory.createLongType();
            case Value.TYPE_FLOAT:     return VerificationTypeFactory.createFloatType();
            case Value.TYPE_DOUBLE:    return VerificationTypeFactory.createDoubleType();
            case Value.TYPE_TOP:       return VerificationTypeFactory.createTopType();

            case Value.TYPE_REFERENCE:
                // Is it a Null type?
                ReferenceValue referenceValue = value.referenceValue();
                if (referenceValue.isNull() == Value.ALWAYS)
                {
                    return VerificationTypeFactory.createNullType();
                }

                // Does the reference type have a single producer?
                if (offset != AT_METHOD_ENTRY)
                {
                    TracedReferenceValue tracedReferenceValue =
                        (TracedReferenceValue)referenceValue;

                    InstructionOffsetValue instructionOffsetValue =
                        tracedReferenceValue.getTraceValue().instructionOffsetValue();

                    if (instructionOffsetValue.instructionOffsetCount() == 1)
                    {
                        // Is it a method parameter?
                        if (instructionOffsetValue.isMethodParameter(0))
                        {
                            // Is the parameter an uninitialized "this"?
                            if (isUninitalizedThis(offset,
                                                   instructionOffsetValue.methodParameter(0)))
                            {
                                // It's an UninitializedThis type.
                                return VerificationTypeFactory.createUninitializedThisType();
                            }
                        }
                        // Is it a newly created instance?
                        else if (instructionOffsetValue.isNewinstance(0))
                        {
                            int producerOffset = instructionOffsetValue.instructionOffset(0);

                            // Is it still uninitialized?
                            if (!initializationFinder.isInitializedBefore(offset, instructionOffsetValue))
                            {
                                // It's an Uninitialized type.
                                return VerificationTypeFactory.createUninitializedType(producerOffset);
                            }
                        }
                    }
                }

                // It's an ordinary Object type.
                return VerificationTypeFactory.createObjectType(createClassConstant(programClass, referenceValue));
        }

        throw new IllegalArgumentException("Unknown computational type ["+type+"]");
    }


    /**
     * Finds or creates a class constant for the given reference value, and
     * returns its index in the constant pool.
     */
    private int createClassConstant(ProgramClass   programClass,
                                    ReferenceValue referenceValue)
    {
        return new ConstantPoolEditor(programClass).addClassConstant(referenceValue.getType(),
                                                                     referenceValue.getReferencedClass());
    }


    /**
     * Compresses the given list of full frames, for use in a stack map table.
     */
    private void compressStackMapFrames(VerificationType[] initialVariableTypes,
                                        List               stackMapFrameList)
    {
        int                previousVariablesCount = initialVariableTypes.length;
        VerificationType[] previousVariableTypes  = initialVariableTypes;

        int previousOffset = -1;

        for (int index = 0; index < stackMapFrameList.size(); index++)
        {
            FullFrame fullFrame = (FullFrame)stackMapFrameList.get(index);

            int                variablesCount = fullFrame.variablesCount;
            VerificationType[] variables      = fullFrame.variables;
            int                stackCount     = fullFrame.stackCount;
            VerificationType[] stack          = fullFrame.stack;

            // Start computing the compressed frame.
            // The default is the full frame.
            StackMapFrame compressedFrame = fullFrame;

            // Are all variables equal?
            if (variablesCount == previousVariablesCount &&
                equalVerificationTypes(variables, previousVariableTypes, variablesCount))
            {
                // Are the stacks equal?
                //if (stackCount == previousStackCount &&
                //    equalVerificationTypes(stack, previousStack, stackCount))
                //{
                //    // Remove the identical frame.
                //    stackMapFrameList.remove(index--);
                //
                //    // Move on to the next frame (at the same index).
                //    continue;
                //}
                // Is the new stack empty?
                //else
                if (stackCount == 0)
                {
                    compressedFrame = new SameZeroFrame();
                }
                // Does the new stack contain a single element?
                else if (stackCount == 1)
                {
                    compressedFrame = new SameOneFrame(stack[0]);
                }
            }
            // Is the stack empty?
            else if (stackCount == 0)
            {
                int additionalVariablesCount = variablesCount - previousVariablesCount;

                // Are the variables chopped?
                if (additionalVariablesCount < 0  &&
                    additionalVariablesCount > -4 &&
                    equalVerificationTypes(variables, previousVariableTypes, variablesCount))
                {
                    compressedFrame = new LessZeroFrame((byte)-additionalVariablesCount);
                }
                // Are the variables extended?
                else if (//previousVariablesCount   > 0 &&
                         additionalVariablesCount > 0 &&
                         additionalVariablesCount < 4 &&
                         equalVerificationTypes(variables, previousVariableTypes, previousVariablesCount))
                {
                    // Copy the additional variables into an array.
                    VerificationType[] additionalVariables = new VerificationType[additionalVariablesCount];
                    System.arraycopy(variables, variablesCount - additionalVariablesCount,
                                     additionalVariables, 0,
                                     additionalVariablesCount);

                    compressedFrame = new MoreZeroFrame(additionalVariables);
                }
            }

            // Compress the instruction offset.
            int offset = fullFrame.u2offsetDelta;
            compressedFrame.u2offsetDelta = offset - previousOffset - 1;
            previousOffset = offset;

            // Remember this frame.
            previousVariablesCount = fullFrame.variablesCount;
            previousVariableTypes  = fullFrame.variables;

            // Replace the full frame.
            stackMapFrameList.set(index, compressedFrame);
        }
    }


    /**
     * Returns whether the given arrays of verification types are equal, up to
     * the given length.
     */
    private boolean equalVerificationTypes(VerificationType[] types1,
                                           VerificationType[] types2,
                                           int                length)
    {
        if (length > 0 &&
            (types1.length < length ||
             types2.length < length))
        {
            return false;
        }

        for (int index = 0; index < length; index++)
        {
            if (!types1[index].equals(types2[index]))
            {
                return false;
            }
        }

        return true;
    }


    /**
     * Returns wheter the specified variable is an uninitialized "this" at the
     * given instruction offset.
     */
    private boolean isUninitalizedThis(int offset, int variableIndex)
    {
        return
            variableIndex == 0                   &&
            initializationFinder.isInitializer() &&
            offset <= initializationFinder.superInitializationOffset();
    }


    /**
     * Returns whether the given instruction opcode represents a dup or swap
     * instruction (dup, dup_x1, dup_x2, dup2, dup2_x1, dup2_x2, swap).
     */
    private boolean isDupOrSwap(int opcode)
    {
        return opcode >= InstructionConstants.OP_DUP &&
               opcode <= InstructionConstants.OP_SWAP;
    }
}

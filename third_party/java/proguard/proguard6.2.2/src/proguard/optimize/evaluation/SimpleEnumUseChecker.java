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
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.evaluation.*;
import proguard.evaluation.value.*;
import proguard.optimize.OptimizationInfoClassFilter;
import proguard.optimize.info.SimpleEnumMarker;

/**
 * This ClassVisitor marks enums that can't be simplified due to the way they
 * are used in the classes that it visits.
 *
 * @see SimpleEnumMarker
 * @author Eric Lafortune
 */
public class SimpleEnumUseChecker
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor,
             AttributeVisitor,
             BootstrapMethodInfoVisitor,
             ConstantVisitor,
             InstructionVisitor,
             ParameterVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("enum") != null;
    //*/

    private final PartialEvaluator       partialEvaluator;
    private final MemberVisitor          methodCodeChecker           = new AllAttributeVisitor(this);
    private final ConstantVisitor        invokedMethodChecker        = new ReferencedMemberVisitor(this);
    private final ConstantVisitor        parameterChecker            = new ReferencedMemberVisitor(new AllParameterVisitor(false, this));
    private final ClassVisitor           complexEnumMarker           = new OptimizationInfoClassFilter(new SimpleEnumMarker(false));
    private final ReferencedClassVisitor referencedComplexEnumMarker = new ReferencedClassVisitor(complexEnumMarker);


    // Fields acting as parameters and return values for the visitor methods.
    private int invocationOffset;


    /**
     * Creates a new SimpleEnumUseSimplifier.
     */
    public SimpleEnumUseChecker()
    {
        this(new PartialEvaluator(new TypedReferenceValueFactory()));
    }


    /**
     * Creates a new SimpleEnumUseChecker.
     * @param partialEvaluator the partial evaluator that will execute the code
     *                         and provide information about the results.
     */
    public SimpleEnumUseChecker(PartialEvaluator partialEvaluator)
    {
        this.partialEvaluator = partialEvaluator;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Unmark the simple enum classes in bootstrap methods attributes.
        programClass.attributesAccept(this);

        if ((programClass.getAccessFlags() & ClassConstants.ACC_ANNOTATION) != 0)
        {
            // Unmark the simple enum classes in annotations.
            programClass.methodsAccept(referencedComplexEnumMarker);
        }
        else
        {
            // Unmark the simple enum classes that are used in a complex way.
            programClass.methodsAccept(methodCodeChecker);
        }
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitBootstrapMethodsAttribute(Clazz clazz, BootstrapMethodsAttribute bootstrapMethodsAttribute)
    {
        // Unmark the simple enum classes in all bootstrap methods.
        bootstrapMethodsAttribute.bootstrapMethodEntriesAccept(clazz, this);
    }


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Evaluate the method.
        partialEvaluator.visitCodeAttribute(clazz, method, codeAttribute);

        int codeLength = codeAttribute.u4codeLength;

        // Check all traced instructions.
        for (int offset = 0; offset < codeLength; offset++)
        {
            if (partialEvaluator.isTraced(offset))
            {
                Instruction instruction = InstructionFactory.create(codeAttribute.code,
                                                                    offset);

                instruction.accept(clazz, method, codeAttribute, offset, this);

                // Check generalized stacks and variables at branch targets.
                if (partialEvaluator.isBranchOrExceptionTarget(offset))
                {
                    checkMixedStackEntriesBefore(offset);

                    checkMixedVariablesBefore(offset);
                }
            }
        }
    }


    // Implementations for BootstrapMethodInfoVisitor.

    public void visitBootstrapMethodInfo(Clazz clazz, BootstrapMethodInfo bootstrapMethodInfo)
    {
        // Unmark the simple enum classes referenced in the method handle.
        bootstrapMethodInfo.methodHandleAccept(clazz, this);

        // Unmark the simple enum classes referenced in the method arguments.
        bootstrapMethodInfo.methodArgumentsAccept(clazz, this);
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        // Unmark any simple enum class referenced in the string constant.
        stringConstant.referencedClassAccept(complexEnumMarker);
    }


    public void visitMethodHandleConstant(Clazz clazz, MethodHandleConstant methodHandleConstant)
    {
        // Unmark the simple enum classes referenced in the method handle
        // (through a reference constant).
        methodHandleConstant.referenceAccept(clazz, this);
    }


    public void visitMethodTypeConstant(Clazz clazz, MethodTypeConstant methodTypeConstant)
    {
        // Unmark the simple enum classes referenced in the method type constant.
        methodTypeConstant.referencedClassesAccept(referencedComplexEnumMarker);
    }


    public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
    {
        // Unmark the simple enum classes referenced in the reference.
        refConstant.referencedClassAccept(referencedComplexEnumMarker);
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        // Unmark any simple enum class referenced in the class constant.
        classConstant.referencedClassAccept(complexEnumMarker);
    }


    // Implementations for InstructionVisitor.

    public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
    {
        switch (simpleInstruction.opcode)
        {
            case InstructionConstants.OP_AASTORE:
            {
                // Check if the instruction is storing a simple enum in a
                // more general array.
                if (!isPoppingSimpleEnumType(offset, 2))
                {
                    if (DEBUG)
                    {
                        if (isPoppingSimpleEnumType(offset))
                        {
                            System.out.println("SimpleEnumUseChecker: ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"] stores enum ["+
                                               partialEvaluator.getStackBefore(offset).getTop(0).referenceValue().getType()+"] in more general array ["+
                                               partialEvaluator.getStackBefore(offset).getTop(2).referenceValue().getType()+"]");
                        }
                    }

                    markPoppedComplexEnumType(offset);
                }
                break;
            }
            case InstructionConstants.OP_ARETURN:
            {
                // Check if the instruction is returning a simple enum as a
                // more general type.
                if (!isReturningSimpleEnumType(clazz, method))
                {
                    if (DEBUG)
                    {
                        if (isPoppingSimpleEnumType(offset))
                        {
                            System.out.println("SimpleEnumUseChecker: ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"] returns enum [" +
                                               partialEvaluator.getStackBefore(offset).getTop(0).referenceValue().getType()+"] as more general type");
                        }
                    }

                    markPoppedComplexEnumType(offset);
                }
                break;
            }
            case InstructionConstants.OP_MONITORENTER:
            case InstructionConstants.OP_MONITOREXIT:
            {
                // Make sure the popped type is not a simple enum type.
                if (DEBUG)
                {
                    if (isPoppingSimpleEnumType(offset))
                    {
                        System.out.println("SimpleEnumUseChecker: ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"] uses enum ["+
                                           partialEvaluator.getStackBefore(offset).getTop(0).referenceValue().getType()+"] as monitor");
                    }
                }

                markPoppedComplexEnumType(offset);

                break;
            }
        }
    }


    public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
    {
    }


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        switch (constantInstruction.opcode)
        {
            case InstructionConstants.OP_PUTSTATIC:
            case InstructionConstants.OP_PUTFIELD:
            {
                // Check if the instruction is generalizing a simple enum to a
                // different type.
                invocationOffset = offset;
                clazz.constantPoolEntryAccept(constantInstruction.constantIndex,
                                              parameterChecker);
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
                if (isPoppingSimpleEnumType(offset, stackEntryIndex) &&
                    !isSupportedMethod(invokedMethodName,
                                       invokedMethodType))
                {
                    if (DEBUG)
                    {
                        System.out.println("SimpleEnumUseChecker: ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"] calls ["+partialEvaluator.getStackBefore(offset).getTop(stackEntryIndex).referenceValue().getType()+"."+invokedMethodName+"]");
                    }

                    markPoppedComplexEnumType(offset, stackEntryIndex);
                }

                // Check if any of the parameters is generalizing a simple
                // enum to a different type.
                invocationOffset = offset;
                clazz.constantPoolEntryAccept(constantInstruction.constantIndex,
                                              parameterChecker);
                break;
            }
            case InstructionConstants.OP_INVOKESPECIAL:
            case InstructionConstants.OP_INVOKESTATIC:
            case InstructionConstants.OP_INVOKEINTERFACE:
            {
                // Check if it is calling a method that we can't simplify.
                clazz.constantPoolEntryAccept(constantInstruction.constantIndex,
                                              invokedMethodChecker);

                // Check if any of the parameters is generalizing a simple
                // enum to a different type.
                invocationOffset = offset;
                clazz.constantPoolEntryAccept(constantInstruction.constantIndex,
                                              parameterChecker);
                break;
            }
            case InstructionConstants.OP_CHECKCAST:
            case InstructionConstants.OP_INSTANCEOF:
            {
                // Check if the instruction is popping a different type.
                if (!isPoppingExpectedType(offset,
                                           clazz,
                                           constantInstruction.constantIndex))
                {
                    if (DEBUG)
                    {
                        if (isPoppingSimpleEnumType(offset))
                        {
                            System.out.println("SimpleEnumUseChecker: ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"] is casting or checking ["+
                                               partialEvaluator.getStackBefore(offset).getTop(0).referenceValue().getType()+"] as ["+
                                               clazz.getClassName(constantInstruction.constantIndex)+"]");
                        }
                    }

                    // Make sure the popped type is not a simple enum type.
                    markPoppedComplexEnumType(offset);

                    // Make sure the checked type is not a simple enum type.
                    // Casts in values() and valueOf(String) are ok.
                    if (constantInstruction.opcode != InstructionConstants.OP_CHECKCAST ||
                        !isSimpleEnum(clazz)                                            ||
                        (method.getAccessFlags() & ClassConstants.ACC_STATIC) == 0      ||
                        !isMethodSkippedForCheckcast(method.getName(clazz),
                                                     method.getDescriptor(clazz)))
                    {
                        if (DEBUG)
                        {
                            if (isSimpleEnum(((ClassConstant)((ProgramClass)clazz).getConstant(constantInstruction.constantIndex)).referencedClass))
                            {
                                System.out.println("SimpleEnumUseChecker: ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"] is casting or checking ["+
                                                   partialEvaluator.getStackBefore(offset).getTop(0).referenceValue().getType()+"] as ["+
                                                   clazz.getClassName(constantInstruction.constantIndex)+"]");
                            }
                        }

                        markConstantComplexEnumType(clazz, constantInstruction.constantIndex);
                    }
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
            case InstructionConstants.OP_IFACMPNE:
            {
                // Check if the instruction is comparing different types.
                if (!isPoppingIdenticalTypes(offset, 0, 1))
                {
                    if (DEBUG)
                    {
                        if (isPoppingSimpleEnumType(offset, 0))
                        {
                            System.out.println("SimpleEnumUseChecker: ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"] compares ["+partialEvaluator.getStackBefore(offset).getTop(0).referenceValue().getType()+"] to plain type");
                        }

                        if (isPoppingSimpleEnumType(offset, 1))
                        {
                            System.out.println("SimpleEnumUseChecker: ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"] compares ["+partialEvaluator.getStackBefore(offset).getTop(1).referenceValue().getType()+"] to plain type");
                        }
                    }

                    // Make sure the first popped type is not a simple enum type.
                    markPoppedComplexEnumType(offset, 0);

                    // Make sure the second popped type is not a simple enum type.
                    markPoppedComplexEnumType(offset, 1);
                }
                break;
            }
        }
    }


    public void visitAnySwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SwitchInstruction switchInstruction)
    {
    }


    // Implementations for MemberVisitor.

    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod) {}


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        if (isSimpleEnum(programClass) &&
            isUnsupportedMethod(programMethod.getName(programClass),
                                programMethod.getDescriptor(programClass)))
        {
            if (DEBUG)
            {
                System.out.println("SimpleEnumUseChecker: invocation of ["+programClass.getName()+"."+programMethod.getName(programClass)+programMethod.getDescriptor(programClass)+"]");
            }

            complexEnumMarker.visitProgramClass(programClass);
        }
    }


    // Implementations for ParameterVisitor.

    public void visitParameter(Clazz clazz, Member member, int parameterIndex, int parameterCount, int parameterOffset, int parameterSize, String parameterType, Clazz referencedClass)
    {
        // Check if the parameter is passing a simple enum as a more general
        // type.
        int stackEntryIndex = parameterSize - parameterOffset - 1;
        if (ClassUtil.isInternalClassType(parameterType) &&
            !isPoppingExpectedType(invocationOffset, stackEntryIndex,
                                   ClassUtil.isInternalArrayType(parameterType) ?
                                       parameterType :
                                       ClassUtil.internalClassNameFromClassType(parameterType)))
        {
            if (DEBUG)
            {
                ReferenceValue poppedValue =
                    partialEvaluator.getStackBefore(invocationOffset).getTop(stackEntryIndex).referenceValue();
                if (isSimpleEnumType(poppedValue))
                {
                    System.out.println("SimpleEnumUseChecker: ["+poppedValue.getType()+"] "+
                                       (member instanceof Field ?
                                            ("is stored as more general type ["+parameterType+"] in field ["+clazz.getName()+"."+member.getName(clazz)+"]") :
                                            ("is passed as more general argument #"+parameterIndex+" ["+parameterType+"] to ["+clazz.getName()+"."+member.getName(clazz)+"]")));
                }
            }

            // Make sure the popped type is not a simple enum type.
            markPoppedComplexEnumType(invocationOffset, stackEntryIndex);
        }
    }


    // Small utility methods.

    /**
     * Returns whether the specified enum method is supported for simple enums.
     */
    private boolean isSupportedMethod(String name, String type)
    {
        return name.equals(ClassConstants.METHOD_NAME_ORDINAL) &&
               type.equals(ClassConstants.METHOD_TYPE_ORDINAL) ||

               name.equals(ClassConstants.METHOD_NAME_CLONE) &&
               type.equals(ClassConstants.METHOD_TYPE_CLONE);
    }


    /**
     * Returns whether the specified enum method is unsupported for simple enums.
     */
    private boolean isUnsupportedMethod(String name, String type)
    {
        return name.equals(ClassConstants.METHOD_NAME_VALUEOF);
    }


    /**
     * Returns whether the specified enum method shall be skipped when
     * analyzing checkcast instructions.
     */
    private boolean isMethodSkippedForCheckcast(String name, String type)
    {
        return name.equals(ClassConstants.METHOD_NAME_VALUEOF) ||
               name.equals(ClassConstants.METHOD_NAME_VALUES);
    }


    /**
     * Unmarks simple enum classes that are mixed with incompatible reference
     * types in the stack before the given instruction offset.
     */
    private void checkMixedStackEntriesBefore(int offset)
    {
        TracedStack stackBefore = partialEvaluator.getStackBefore(offset);

        // Check all stack entries.
        int stackSize = stackBefore.size();

        for (int stackEntryIndex = 0; stackEntryIndex < stackSize; stackEntryIndex++)
        {
            // Check reference entries.
            Value stackEntry = stackBefore.getBottom(stackEntryIndex);
            if (stackEntry.computationalType() == Value.TYPE_REFERENCE)
            {
                // Check reference entries with multiple producers.
                InstructionOffsetValue producerOffsets =
                    stackBefore.getBottomActualProducerValue(stackEntryIndex).instructionOffsetValue();

                int producerCount = producerOffsets.instructionOffsetCount();
                if (producerCount > 1)
                {
                    // Is the consumed stack entry not a simple enum?
                    ReferenceValue consumedStackEntry =
                        stackEntry.referenceValue();

                    if (!isSimpleEnumType(consumedStackEntry))
                    {
                        // Check all producers.
                        for (int producerIndex = 0; producerIndex < producerCount; producerIndex++)
                        {
                            if (!producerOffsets.isExceptionHandler(producerIndex))
                            {
                                int producerOffset =
                                    producerOffsets.instructionOffset(producerIndex);

                                if (DEBUG)
                                {
                                    ReferenceValue producedValue =
                                        partialEvaluator.getStackAfter(producerOffset).getTop(0).referenceValue();
                                    if (isSimpleEnumType(producedValue))
                                    {
                                        System.out.println("SimpleEnumUseChecker: ["+producedValue.getType()+"] mixed with general type on stack");
                                    }
                                }

                                // Make sure the produced stack entry isn't a
                                // simple enum either.
                                markPushedComplexEnumType(producerOffset);
                            }
                        }
                    }
                }
            }
        }
    }


    /**
     * Unmarks simple enum classes that are mixed with incompatible reference
     * types in the variables before the given instruction offset.
     */
    private void checkMixedVariablesBefore(int offset)
    {
        TracedVariables variablesBefore =
            partialEvaluator.getVariablesBefore(offset);

        // Check all variables.
        int variablesSize = variablesBefore.size();

        for (int variableIndex = 0; variableIndex < variablesSize; variableIndex++)
        {
            // Check reference variables.
            Value variable = variablesBefore.getValue(variableIndex);
            if (variable != null &&
                variable.computationalType() == Value.TYPE_REFERENCE)
            {
                // Check reference variables with multiple producers.
                InstructionOffsetValue producerOffsets =
                    variablesBefore.getProducerValue(variableIndex).instructionOffsetValue();

                int producerCount = producerOffsets.instructionOffsetCount();
                if (producerCount > 1)
                {
                    // Is the consumed variable not a simple enum?
                    ReferenceValue consumedVariable =
                        variable.referenceValue();

                    if (!isSimpleEnumType(consumedVariable))
                    {
                        // Check all producers.
                        for (int producerIndex = 0; producerIndex < producerCount; producerIndex++)
                        {
                            if (!producerOffsets.isMethodParameter(producerIndex))
                            {
                                int producerOffset =
                                    producerOffsets.instructionOffset(producerIndex);

                                if (DEBUG)
                                {
                                    ReferenceValue producedValue =
                                        partialEvaluator.getVariablesAfter(producerOffset).getValue(variableIndex).referenceValue();
                                    if (isSimpleEnumType(producedValue))
                                    {
                                        System.out.println("SimpleEnumUseChecker: ["+producedValue.getType()+"] mixed with general type in variables");
                                    }
                                }

                                // Make sure the stored variable entry isn't a
                                // simple enum either.
                                markStoredComplexEnumType(producerOffset, variableIndex);
                            }
                        }
                    }
                }
            }
        }
    }


    /**
     * Returns whether the instruction at the given offset is popping two
     * identical reference types.
     */
    private boolean isPoppingIdenticalTypes(int offset,
                                            int stackEntryIndex1,
                                            int stackEntryIndex2)
    {
        TracedStack stackBefore = partialEvaluator.getStackBefore(offset);

        String type1 =
            stackBefore.getTop(stackEntryIndex1).referenceValue().getType();
        String type2 =
            stackBefore.getTop(stackEntryIndex2).referenceValue().getType();

        return type1 == null ? type2 == null : type1.equals(type2);
    }


    /**
     * Returns whether the instruction at the given offset is popping exactly
     * the reference type of the specified class constant.
     */
    private boolean isPoppingExpectedType(int   offset,
                                          Clazz clazz,
                                          int   constantIndex)
    {
        return isPoppingExpectedType(offset, 0, clazz, constantIndex);
    }


    /**
     * Returns whether the instruction at the given offset is popping exactly
     * the reference type of the specified class constant.
     */
    private boolean isPoppingExpectedType(int   offset,
                                          int   stackEntryIndex,
                                          Clazz clazz,
                                          int   constantIndex)
    {
        return isPoppingExpectedType(offset,
                                     stackEntryIndex,
                                     clazz.getClassName(constantIndex));
    }


    /**
     * Returns whether the instruction at the given offset is popping exactly
     * the given reference type.
     */
    private boolean isPoppingExpectedType(int    offset,
                                          int    stackEntryIndex,
                                          String expectedType)
    {
        TracedStack stackBefore = partialEvaluator.getStackBefore(offset);

        String poppedType =
            stackBefore.getTop(stackEntryIndex).referenceValue().getType();

        return expectedType.equals(poppedType);
    }


    /**
     * Returns whether the given method is returning a simple enum type.
     * This includes simple enum arrays.
     */
    private boolean isReturningSimpleEnumType(Clazz clazz, Method method)
    {
        String descriptor = method.getDescriptor(clazz);
        String returnType = ClassUtil.internalMethodReturnType(descriptor);

        if (ClassUtil.isInternalClassType(returnType))
        {
            Clazz[] referencedClasses =
                ((ProgramMethod)method).referencedClasses;

            if (referencedClasses != null)
            {
                Clazz referencedClass =
                    referencedClasses[referencedClasses.length - 1];

                return isSimpleEnum(referencedClass);
            }
        }

        return false;
    }


    /**
     * Returns whether the instruction at the given offset is popping a type
     * with a simple enum class. This includes simple enum arrays.
     */
    private boolean isPoppingSimpleEnumType(int offset)
    {
        return isPoppingSimpleEnumType(offset, 0);
    }


    /**
     * Returns whether the instruction at the given offset is popping a type
     * with a simple enum class. This includes simple enum arrays.
     */
    private boolean isPoppingSimpleEnumType(int offset, int stackEntryIndex)
    {
        ReferenceValue referenceValue =
            partialEvaluator.getStackBefore(offset).getTop(stackEntryIndex).referenceValue();

        return isSimpleEnumType(referenceValue);
    }


    /**
     * Returns whether the given value is a simple enum type. This includes
     * simple enum arrays.
     */
    private boolean isSimpleEnumType(ReferenceValue referenceValue)
    {
        return isSimpleEnum(referenceValue.getReferencedClass());
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
     * Marks the enum class of the popped type as complex.
     */
    private void markConstantComplexEnumType(Clazz clazz, int constantIndex)
    {
        clazz.constantPoolEntryAccept(constantIndex,
                                      referencedComplexEnumMarker);
    }


    /**
     * Marks the enum class of the popped type as complex.
     */
    private void markPoppedComplexEnumType(int offset)
    {
        markPoppedComplexEnumType(offset, 0);
    }


    /**
     * Marks the enum class of the specified popped type as complex.
     */
    private void markPoppedComplexEnumType(int offset, int stackEntryIndex)
    {
        ReferenceValue referenceValue =
            partialEvaluator.getStackBefore(offset).getTop(stackEntryIndex).referenceValue();

        markComplexEnumType(referenceValue);
    }


    /**
     * Marks the enum class of the specified pushed type as complex.
     */
    private void markPushedComplexEnumType(int offset)
    {
        ReferenceValue referenceValue =
            partialEvaluator.getStackAfter(offset).getTop(0).referenceValue();

        markComplexEnumType(referenceValue);
    }


    /**
     * Marks the enum class of the specified stored type as complex.
     */
    private void markStoredComplexEnumType(int offset, int variableIndex)
    {
        ReferenceValue referenceValue =
            partialEvaluator.getVariablesAfter(offset).getValue(variableIndex).referenceValue();

        markComplexEnumType(referenceValue);
    }


    /**
     * Marks the enum class of the specified value as complex.
     */
    private void markComplexEnumType(ReferenceValue referenceValue)
    {
        Clazz clazz = referenceValue.getReferencedClass();
        if (clazz != null)
        {
            clazz.accept(complexEnumMarker);
        }
    }
}

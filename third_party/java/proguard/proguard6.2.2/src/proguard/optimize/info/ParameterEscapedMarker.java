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
package proguard.optimize.info;

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
import proguard.optimize.evaluation.*;

/**
 * This ClassPoolVisitor marks the reference parameters that have escaped or
 * that are escaping, outside or inside their methods.
 *
 * @see ReferenceEscapeChecker
 * @see ParameterEscapeMarker
 * @author Eric Lafortune
 */
public class ParameterEscapedMarker
extends      SimplifiedVisitor
implements   ClassPoolVisitor,
             ClassVisitor,
             MemberVisitor,
             AttributeVisitor,
             InstructionVisitor,
             ConstantVisitor
{
    /*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("pem") != null;
    //*/


    private final ClassVisitor                 parameterEscapedMarker =
        new AllMethodVisitor(
        new AllAttributeVisitor(this));
    private final ValueFactory                 valueFactory                = new BasicValueFactory();
    private final ReferenceTracingValueFactory tracingValueFactory         = new ReferenceTracingValueFactory(valueFactory);
    private final PartialEvaluator             partialEvaluator            =
        new PartialEvaluator(tracingValueFactory,
                             new ParameterTracingInvocationUnit(new BasicInvocationUnit(tracingValueFactory)),
                             true,
                             tracingValueFactory);
    private final ReferenceEscapeChecker       referenceEscapeChecker = new ReferenceEscapeChecker(partialEvaluator, false);

    // Parameters and values for visitor methods.
    private boolean newEscapes;
    private Method  referencingMethod;
    private int     referencingOffset;
    private int     referencingPopCount;


    /**
     * Creates a new ParameterModificationMarker.
     */
    public ParameterEscapedMarker()
    {
    }


    // Implementations for ClassPoolVisitor.

    public void visitClassPool(ClassPool classPool)
    {
        // Go over all classes and their methods, marking if parameters are
        // modified, until no new cases can be found.
        do
        {
            newEscapes = false;

            if (DEBUG)
            {
                System.out.println("ParameterEscapedMarker: new iteration");
            }

            // Go over all classes and their methods once.
            classPool.classesAccept(parameterEscapedMarker);
        }
        while (newEscapes);

        if (DEBUG)
        {
            classPool.classesAccept(new AllMethodVisitor(this));
        }
    }


    // Implementations for MemberVisitor.

    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        if (DEBUG)
        {
            System.out.println("ParameterEscapedMarker: [" + programClass.getName() + "." + programMethod.getName(programClass) + programMethod.getDescriptor(programClass) + "]");

            int parameterSize =
                ClassUtil.internalMethodParameterSize(programMethod.getDescriptor(programClass),
                                                      programMethod.getAccessFlags());

            for (int index = 0; index < parameterSize; index++)
            {
                System.out.println("  " +
                                   (hasParameterEscaped(programMethod, index) ? 'e' : '.') +
                                   " P" + index);
            }
        }
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Evaluate the code.
        partialEvaluator.visitCodeAttribute(clazz, method, codeAttribute);
        referenceEscapeChecker.visitCodeAttribute(clazz, method, codeAttribute);

        // Mark the parameters that are modified from the code.
        codeAttribute.instructionsAccept(clazz, method, partialEvaluator.tracedInstructionFilter(this));
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        switch (constantInstruction.opcode)
        {
            case InstructionConstants.OP_INVOKEVIRTUAL:
            case InstructionConstants.OP_INVOKESPECIAL:
            case InstructionConstants.OP_INVOKESTATIC:
            case InstructionConstants.OP_INVOKEINTERFACE:
                // Mark escaped reference parameters in the invoked method.
                referencingMethod   = method;
                referencingOffset   = offset;
                referencingPopCount = constantInstruction.stackPopCount(clazz);
                clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                break;
        }
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitAnyMethodrefConstant(Clazz clazz, RefConstant refConstant)
    {
        Method referencedMethod = (Method)refConstant.referencedMember;

        if (referencedMethod != null &&
            MethodOptimizationInfo.getMethodOptimizationInfo(referencedMethod) instanceof ProgramMethodOptimizationInfo)
        {
            // Mark reference parameters that are passed to the method.
            for (int parameterIndex = 0; parameterIndex < referencingPopCount; parameterIndex++)
            {
                int stackEntryIndex = referencingPopCount - parameterIndex - 1;

                TracedStack stackBefore = partialEvaluator.getStackBefore(referencingOffset);
                Value       stackEntry  = stackBefore.getTop(stackEntryIndex);

                if (stackEntry.computationalType() == Value.TYPE_REFERENCE)
                {
                    // Has the parameter escaped outside or inside the referencing
                    // method?
                    if (hasEscapedBefore(referencingOffset, stackEntryIndex))
                    {
                        markParameterEscaped(referencedMethod, parameterIndex);
                    }
                }
            }
        }
    }


    // Small utility methods.

    /**
     * Returns whether any of the producing reference values of the specified
     * stack entry before the given instruction offset are escaping or have
     * escaped.
     */
    private boolean hasEscapedBefore(int instructionOffset,
                                     int stackEntryIndex)
    {
        TracedStack stackBefore = partialEvaluator.getStackBefore(instructionOffset);
        Value       stackEntry  = stackBefore.getTop(stackEntryIndex);

        if (stackEntry.computationalType() == Value.TYPE_REFERENCE)
        {
            ReferenceValue referenceValue = stackEntry.referenceValue();

            // The null reference value may not have a trace value.
            if (referenceValue.isNull() != Value.ALWAYS &&
                hasEscaped(referenceValue))
            {
                return true;
            }
        }

        return false;
    }


    /**
     * Returns whether the producing reference value is escaping or has escaped.
     */
    private boolean hasEscaped(ReferenceValue referenceValue)
    {
        TracedReferenceValue   tracedReferenceValue   = (TracedReferenceValue)referenceValue;
        InstructionOffsetValue instructionOffsetValue = tracedReferenceValue.getTraceValue().instructionOffsetValue();

        int count = instructionOffsetValue.instructionOffsetCount();
        for (int index = 0; index < count; index++)
        {
            if (instructionOffsetValue.isMethodParameter(index) ?
                  hasParameterEscaped(referencingMethod, instructionOffsetValue.methodParameter(index)) :
                  referenceEscapeChecker.isInstanceEscaping(instructionOffsetValue.instructionOffset(index)))
            {
                return true;
            }
        }

        return false;
    }


    /**
     * Marks the given parameter as escaped from the given method.
     */
    private void markParameterEscaped(Method method, int parameterIndex)
    {
        ProgramMethodOptimizationInfo info = ProgramMethodOptimizationInfo.getProgramMethodOptimizationInfo(method);
        if (!info.hasParameterEscaped(parameterIndex))
        {
            info.setParameterEscaped(parameterIndex);

            newEscapes = true;
        }
    }


    /**
     * Marks the given parameters as escaped from the given method.
     */
    private void markEscapedParameters(Method method, long escapedParameters)
    {
        ProgramMethodOptimizationInfo info = ProgramMethodOptimizationInfo.getProgramMethodOptimizationInfo(method);
        if ((~info.getEscapedParameters() & escapedParameters) != 0)
        {
            info.updateEscapedParameters(escapedParameters);

            newEscapes = true;
        }
    }


    /**
     * Returns whether the given parameter is escaped from the given method.
     */
    public static boolean hasParameterEscaped(Method method, int parameterIndex)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).hasParameterEscaped(parameterIndex);
    }


    /**
     * Returns which parameters are escaped from the given method.
     */
    public static long getEscapedParameters(Method method)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).getEscapedParameters();
    }
}
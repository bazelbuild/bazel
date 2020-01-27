/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2017 Eric Lafortune @ GuardSquare
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
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.MemberVisitor;
import proguard.evaluation.value.Value;
import proguard.optimize.evaluation.PartialEvaluator;

/**
 * This MemberVisitor counts the parameters and marks the used parameters
 * of the methods that it visits. It also marks the 'this' parameters of
 * methods that have hierarchies.
 *
 * @author Eric Lafortune
 */
public class ParameterUsageMarker
extends      SimplifiedVisitor
implements   MemberVisitor,
             AttributeVisitor,
             InstructionVisitor
{
    private static final boolean DEBUG = false;


    private final boolean          markThisParameter;
    private final boolean          markAllParameters;
    private final PartialEvaluator partialEvaluator = new PartialEvaluator();


    /**
     * Creates a new ParameterUsageMarker.
     */
    public ParameterUsageMarker()
    {
        this(false, false);
    }


    /**
     * Creates a new ParameterUsageMarker that optionally marks all parameters.
     * @param markThisParameter specifies whether all 'this' parameters should
     *                          be marked as being used.
     * @param markAllParameters specifies whether all other parameters should
     *                          be marked as being used.
     */
    public ParameterUsageMarker(boolean markThisParameter,
                                boolean markAllParameters)
    {
        this.markThisParameter = markThisParameter;
        this.markAllParameters = markAllParameters;
    }


    // Implementations for MemberVisitor.

    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        int parameterSize =
            ClassUtil.internalMethodParameterSize(programMethod.getDescriptor(programClass),
                                                  programMethod.getAccessFlags());

        if (parameterSize > 0)
        {
            int accessFlags = programMethod.getAccessFlags();

            // Must we mark the 'this' parameter?
            if (markThisParameter &&
                (accessFlags & ClassConstants.ACC_STATIC) == 0)
            {
                // Mark the 'this' parameter.
                markParameterUsed(programMethod, 0);
            }

            // Must we mark all other parameters?
            if (markAllParameters)
            {
                // Mark all parameters, without the 'this' parameter.
                markUsedParameters(programMethod,
                                   (accessFlags & ClassConstants.ACC_STATIC) != 0 ?
                                       -1L : -2L);
            }

            // Is it a native method?
            if ((accessFlags & ClassConstants.ACC_NATIVE) != 0)
            {
                // Mark all parameters.
                markUsedParameters(programMethod, -1L);
            }

            // Is it an abstract method?
            else if ((accessFlags & ClassConstants.ACC_ABSTRACT) != 0)
            {
                // Mark the 'this' parameter.
                markParameterUsed(programMethod, 0);
            }

            // Is it a non-native, concrete method?
            else
            {
                // Is the method not static, but synchronized, or can it have
                // other implementations, or is it a class instance initializer?
                if ((accessFlags & ClassConstants.ACC_STATIC) == 0 &&
                    ((accessFlags & ClassConstants.ACC_SYNCHRONIZED) != 0 ||
                     programClass.mayHaveImplementations(programMethod)            ||
                     programMethod.getName(programClass).equals(ClassConstants.METHOD_NAME_INIT)))
                {
                    // Mark the 'this' parameter.
                    markParameterUsed(programMethod, 0);
                }

                // Mark the parameters that are used by the code.
                programMethod.attributesAccept(programClass, this);
            }

            if (DEBUG)
            {
                System.out.print("ParameterUsageMarker: ["+programClass.getName() +"."+programMethod.getName(programClass)+programMethod.getDescriptor(programClass)+"]: ");
                for (int index = 0; index < parameterSize; index++)
                {
                    System.out.print(isParameterUsed(programMethod, index) ? '+' : '-');
                }
                System.out.println();
            }

        }

        // Set the parameter size.
        setParameterSize(programMethod, parameterSize);
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        // Can the method have other implementations?
        if (libraryClass.mayHaveImplementations(libraryMethod))
        {
            // All implementations must keep all parameters of this method,
            // including the 'this' parameter.
            markUsedParameters(libraryMethod, -1L);
        }
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Evaluate the code.
        partialEvaluator.visitCodeAttribute(clazz, method, codeAttribute);

        // Mark the parameters that are used by the code.
        codeAttribute.instructionsAccept(clazz, method, this);
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
    {
        if (partialEvaluator.isTraced(offset) &&
            variableInstruction.isLoad())
        {
            int parameterIndex = variableInstruction.variableIndex;
            if (parameterIndex < codeAttribute.u2maxLocals)
            {
                Value producer =
                    partialEvaluator.getVariablesBefore(offset).getProducerValue(parameterIndex);
                if (producer != null &&
                    producer.instructionOffsetValue().contains(PartialEvaluator.AT_METHOD_ENTRY))
                {
                    // Mark the variable.
                    markParameterUsed(method, parameterIndex);

                    // Account for Category 2 instructions, which take up two entries.
                    if (variableInstruction.isCategory2())
                    {
                        markParameterUsed(method, parameterIndex + 1);
                    }
                }
            }
        }
    }


    // Small utility methods.

    /**
     * Sets the total size of the parameters.
     */
    private static void setParameterSize(Method method, int parameterSize)
    {
        MethodOptimizationInfo info = MethodOptimizationInfo.getMethodOptimizationInfo(method);
        if (info != null)
        {
            info.setParameterSize(parameterSize);
        }
    }


    /**
     * Returns the total size of the parameters.
     */
    public static int getParameterSize(Method method)
    {
        MethodOptimizationInfo info = MethodOptimizationInfo.getMethodOptimizationInfo(method);
        return info != null ? info.getParameterSize() : 0;
    }


    /**
     * Marks the given parameter as being used.
     */
    public static void markParameterUsed(Method method, int variableIndex)
    {
        MethodOptimizationInfo info = MethodOptimizationInfo.getMethodOptimizationInfo(method);
        if (info != null)
        {
            info.setParameterUsed(variableIndex);
        }
    }


    /**
     * Marks the given parameters as being used.
     */
    public static void markUsedParameters(Method method, long usedParameters)
    {
        MethodOptimizationInfo info = MethodOptimizationInfo.getMethodOptimizationInfo(method);
        if (info != null)
        {
            info.setUsedParameters(info.getUsedParameters() | usedParameters);
        }
    }


    /**
     * Returns whether the given parameter is being used.
     */
    public static boolean isParameterUsed(Method method, int variableIndex)
    {
        MethodOptimizationInfo info = MethodOptimizationInfo.getMethodOptimizationInfo(method);
        return info == null ||
               info.isParameterUsed(variableIndex);
    }


    /**
     * Returns which parameters are being used.
     */
    public static long getUsedParameters(Method method)
    {
        MethodOptimizationInfo info = MethodOptimizationInfo.getMethodOptimizationInfo(method);
        return info != null ? info.getUsedParameters() : -1L;
    }
}

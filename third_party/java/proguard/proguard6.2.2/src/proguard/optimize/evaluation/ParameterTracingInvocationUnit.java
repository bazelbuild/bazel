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
import proguard.classfile.constant.RefConstant;
import proguard.classfile.util.ClassUtil;
import proguard.evaluation.SimplifiedInvocationUnit;
import proguard.evaluation.value.*;
import proguard.optimize.info.ParameterEscapeMarker;

/**
 * This InvocationUnit tags reference values like
 * {@link ReferenceTracingInvocationUnit}, but adds detail to return values
 * from invoked methods.
 *
 * @see ReferenceTracingInvocationUnit
 * @see TracedReferenceValue
 * @see InstructionOffsetValue
 * @author Eric Lafortune
 */
public class ParameterTracingInvocationUnit
extends      ReferenceTracingInvocationUnit
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("ptiu") != null;
    //*/


    private Value[] parameters = new Value[256];


    /**
     * Creates a new ParameterTracingInvocationUnit that attaches trace
     * values specifying the parameter index to each parameter.
     * @param invocationUnit the invocation unit to which invocations will
     *                       be delegated.
     */
    public ParameterTracingInvocationUnit(SimplifiedInvocationUnit invocationUnit)
    {
        super(invocationUnit);
    }


    // Implementations for SimplifiedInvocationUnit.

    public void setMethodParameterValue(Clazz clazz, RefConstant refConstant, int parameterIndex, Value value)
    {
        super.setMethodParameterValue(clazz, refConstant, parameterIndex, value);

        parameters[parameterIndex] = value;
    }


    public Value getMethodReturnValue(Clazz clazz, RefConstant refConstant, String type)
    {
        Value returnValue =
            super.getMethodReturnValue(clazz, refConstant, type);

        // We only need to worry about reference values.
        if (returnValue.computationalType() != Value.TYPE_REFERENCE)
        {
            return returnValue;
        }

        Method referencedMethod = (Method)refConstant.referencedMember;
        if (referencedMethod != null)
        {
            // Start figuring out which trace value to attach to the return value.
            int offset =
                ((TracedReferenceValue)returnValue).getTraceValue().instructionOffsetValue().instructionOffset(0);

            // The trace value might be any external value or just a new instance.
            InstructionOffsetValue traceValue =
                ParameterEscapeMarker.returnsExternalValues(referencedMethod) ? new InstructionOffsetValue(offset | InstructionOffsetValue.FIELD_VALUE)  :
                ParameterEscapeMarker.returnsNewInstances(referencedMethod)   ? new InstructionOffsetValue(offset | InstructionOffsetValue.NEW_INSTANCE) :
                                                                                null;

            long returnedParameters =
                ParameterEscapeMarker.getReturnedParameters(referencedMethod);

            int parameterCount =
                ClassUtil.internalMethodParameterCount(refConstant.getType(clazz), isStatic);

            for (int parameterIndex = 0; parameterIndex < parameterCount; parameterIndex++)
            {
                if ((returnedParameters & (1L << parameterIndex)) != 0L)
                {
                    Value parameterValue = parameters[parameterIndex];
                    if (parameterValue instanceof TracedReferenceValue)
                    {
                        TracedReferenceValue tracedParameterValue =
                            (TracedReferenceValue)parameterValue;

                        if (mayReturnType(refConstant.referencedClass,
                                          referencedMethod,
                                          tracedParameterValue))
                        {
                            InstructionOffsetValue parameterTraceValue =
                                tracedParameterValue.getTraceValue().instructionOffsetValue();

                            traceValue = traceValue == null ?
                                parameterTraceValue :
                                traceValue.generalize(parameterTraceValue);
                        }
                    }
                }
            }

            if (DEBUG)
            {
                System.out.println("ParameterTracingInvocationUnit.getMethodReturnValue: calling ["+refConstant.getClassName(clazz)+"."+refConstant.getName(clazz)+refConstant.getType(clazz)+"] returns ["+traceValue+" "+returnValue+"]");
            }

            // Did we find more detailed information on the return value?
            // We should, unless the return value is always null.
            if (traceValue != null)
            {
                return trace(returnValue, traceValue);
            }
        }

        return returnValue;
    }


    // Small utility methods.

    /**
     * Returns whether the given method may return the given type of reference
     * value
     */
    private boolean mayReturnType(Clazz          clazz,
                                  Method         method,
                                  ReferenceValue referenceValue)
    {
        String returnType =
            ClassUtil.internalMethodReturnType(method.getDescriptor(clazz));

        Clazz[] referencedClasses = method instanceof ProgramMethod ?
            ((ProgramMethod)method).referencedClasses :
            ((LibraryMethod)method).referencedClasses;

        Clazz referencedClass =
            referencedClasses == null ||
            !ClassUtil.isInternalClassType(returnType) ? null :
                referencedClasses[referencedClasses.length - 1];

        return referenceValue.instanceOf(returnType,
                                         referencedClass) != Value.NEVER;
    }
}
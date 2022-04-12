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
import proguard.classfile.attribute.visitor.AttributeVisitor;
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
 * This MemberVisitor, AttributeVisitor, and InstructionVisitor marks the
 * reference parameters that are escaping, that are modified, or that are
 * returned.
 *
 * It also marks methods that may modify anything on the heap.
 *
 * The class must be called as a MemberVisitor on all members (to mark the
 * parameters of native methods, without code attributes), then as an
 * AttributeVisitor on their code attributes (so it can run its PartialEvaluator
 * and ReferenceEscapeChecker), and finally as an InstructionVisitor on its
 * instructions (to actually mark the parameters).
 *
 * @see SideEffectClassChecker
 * @see SideEffectClassMarker
 * @author Eric Lafortune
 */
public class ParameterEscapeMarker
extends      SimplifiedVisitor
implements   MemberVisitor,
             AttributeVisitor,
             InstructionVisitor,
             ConstantVisitor,
             ParameterVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("pem") != null;
    //*/


    private final MutableBoolean         repeatTrigger;
    private final PartialEvaluator       partialEvaluator;
    private final boolean                runPartialEvaluator;
    private final ReferenceEscapeChecker referenceEscapeChecker;
    private final boolean                runReferenceEscapeChecker;

    private final MemberVisitor parameterMarker = new AllParameterVisitor(true, this);

    // Parameters and values for visitor methods.
    private Method  referencingMethod;
    private int     referencingOffset;
    private int     referencingPopCount;
    private boolean isReturnValueEscaping;
    private boolean isReturnValueModified;


    /**
     * Creates a new ParameterEscapeMarker.
     */
    public ParameterEscapeMarker(MutableBoolean repeatTrigger)
    {
        this(repeatTrigger,
             new BasicValueFactory());
    }


    /**
     * Creates a new ParameterEscapeMarker.
     */
    public ParameterEscapeMarker(MutableBoolean repeatTrigger,
                                 ValueFactory   valueFactory)
    {
        this(repeatTrigger,
             valueFactory,
             new ReferenceTracingValueFactory(valueFactory));
    }


    /**
     * Creates a new ParameterEscapeMarker.
     */
    public ParameterEscapeMarker(MutableBoolean               repeatTrigger,
                                 ValueFactory                 valueFactory,
                                 ReferenceTracingValueFactory tracingValueFactory)
    {
        this(repeatTrigger,
             new PartialEvaluator(tracingValueFactory,
                                  new ParameterTracingInvocationUnit(new BasicInvocationUnit(tracingValueFactory)),
                                  true,
                                  tracingValueFactory),
             true);
    }


    /**
     * Creates a new ParameterEscapeMarker.
     */
    public ParameterEscapeMarker(MutableBoolean   repeatTrigger,
                                 PartialEvaluator partialEvaluator,
                                 boolean          runPartialEvaluator)
    {
        this(repeatTrigger,
             partialEvaluator,
             runPartialEvaluator,
             new ReferenceEscapeChecker(partialEvaluator, false),
             true);
    }


    /**
     * Creates a new ParameterEscapeMarker.
     */
    public ParameterEscapeMarker(MutableBoolean         repeatTrigger,
                                 PartialEvaluator       partialEvaluator,
                                 boolean                runPartialEvaluator,
                                 ReferenceEscapeChecker referenceEscapeChecker,
                                 boolean                runReferenceEscapeChecker)
    {
        this.repeatTrigger             = repeatTrigger;
        this.partialEvaluator          = partialEvaluator;
        this.runPartialEvaluator       = runPartialEvaluator;
        this.referenceEscapeChecker    = referenceEscapeChecker;
        this.runReferenceEscapeChecker = runReferenceEscapeChecker;
    }


    // Implementations for MemberVisitor.

    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        int accessFlags = programMethod.getAccessFlags();

        // Is it a native method?
        if ((accessFlags & ClassConstants.ACC_NATIVE) != 0)
        {
            // Mark all parameters.
            markModifiedParameters(programMethod, -1L);
            markEscapingParameters(programMethod, -1L);
            markReturnedParameters(programMethod, -1L);
            markAnythingModified(programMethod);
        }
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Evaluate the code.
        if (runPartialEvaluator)
        {
            partialEvaluator.visitCodeAttribute(clazz, method, codeAttribute);
        }

        if (runReferenceEscapeChecker)
        {
            referenceEscapeChecker.visitCodeAttribute(clazz, method, codeAttribute);
        }

        if (DEBUG)
        {
            // These results are not complete yet, since this class must still
            // be called as an InstructionVisitor.
            System.out.println("ParameterEscapeMarker: [" + clazz.getName() + "." + method.getName(clazz) + method.getDescriptor(clazz) + "]");

            int parameterCount =
                ClassUtil.internalMethodParameterCount(method.getDescriptor(clazz),
                                                       method.getAccessFlags());

            for (int index = 0; index < parameterCount; index++)
            {
                System.out.println("  " +
//                                   (hasParameterEscaped(method, index) ? 'e' : '.') +
                                   (isParameterEscaping(method, index) ? 'E' : '.') +
                                   (isParameterReturned(method, index) ? 'R' : '.') +
                                   (isParameterModified(method, index) ? 'M' : '.') +
                                   " P" + index);
            }

            System.out.println("  " +
                               (returnsExternalValues(method) ? 'X' : '.') +
                               "   Return value");
        }
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
    {
        switch (simpleInstruction.opcode)
        {
            case InstructionConstants.OP_AASTORE:
                // Mark array parameters whose element is modified.
                markModifiedParameters(method,
                                       offset,
                                       simpleInstruction.stackPopCount(clazz) - 1);

                // Mark reference values that are put in the array.
                markEscapingParameters(method, offset, 0);
                break;

            case InstructionConstants.OP_IASTORE:
            case InstructionConstants.OP_LASTORE:
            case InstructionConstants.OP_FASTORE:
            case InstructionConstants.OP_DASTORE:
            case InstructionConstants.OP_BASTORE:
            case InstructionConstants.OP_CASTORE:
            case InstructionConstants.OP_SASTORE:
                // Mark array parameters whose element is modified.
                markModifiedParameters(method,
                                       offset,
                                       simpleInstruction.stackPopCount(clazz) - 1);
                break;

            case InstructionConstants.OP_ARETURN:
                // Mark returned reference values.
                markReturnedParameters(clazz, method, offset, 0);
                break;

            case InstructionConstants.OP_ATHROW:
                // Mark the escaping reference values.
                markEscapingParameters(method, offset, 0);
                break;
        }
    }


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        switch (constantInstruction.opcode)
        {
            case InstructionConstants.OP_LDC:
            case InstructionConstants.OP_LDC_W:
            case InstructionConstants.OP_NEW:
            case InstructionConstants.OP_ANEWARRAY:
            case InstructionConstants.OP_MULTIANEWARRAY:
            case InstructionConstants.OP_GETSTATIC:
                // Mark possible modifications due to initializers.
                referencingMethod = method;
                referencingOffset = offset;
                clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                break;

            case InstructionConstants.OP_PUTSTATIC:
                // Mark some global modification.
                markAnythingModified(method);

                // Mark reference values that are put in the field.
                markEscapingParameters(method, offset, 0);
                break;

            case InstructionConstants.OP_GETFIELD:
                // Mark the owner of the field. The owner sort of escapes when
                // the field is retrieved. [DGD-1279] (test2181)
                markEscapingParameters(method, offset, 0);
                break;

            case InstructionConstants.OP_PUTFIELD:
                // Mark reference parameters whose field is modified.
                markModifiedParameters(method,
                                       offset,
                                       constantInstruction.stackPopCount(clazz) - 1);

                // Mark reference values that are put in the field.
                markEscapingParameters(method, offset, 0);
                break;

            case InstructionConstants.OP_INVOKEVIRTUAL:
            case InstructionConstants.OP_INVOKESPECIAL:
            case InstructionConstants.OP_INVOKESTATIC:
            case InstructionConstants.OP_INVOKEINTERFACE:
            case InstructionConstants.OP_INVOKEDYNAMIC:
                // Mark reference parameters that are modified as parameters
                // of the invoked method.
                // Mark reference values that are escaping as parameters
                // of the invoked method.
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


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        Clazz referencedClass = stringConstant.referencedClass;

        // If a static initializer may modify anything, so does the referencing
        // method.
        if (referencedClass == null ||
            SideEffectClassChecker.mayHaveSideEffects(clazz,
                                                      referencedClass))
        {
            markAnythingModified(referencingMethod);
        }
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        Clazz referencedClass = classConstant.referencedClass;

        // If a static initializer may modify anything, so does the referencing
        // method.
        if (referencedClass == null ||
            SideEffectClassChecker.mayHaveSideEffects(clazz,
                                                      referencedClass))
        {
            markAnythingModified(referencingMethod);
        }
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        markAnythingModified(referencingMethod);
    }


    public void visitFieldrefConstant(Clazz clazz, FieldrefConstant fieldrefConstant)
    {
        clazz.constantPoolEntryAccept(fieldrefConstant.u2classIndex, this);
    }


    public void visitAnyMethodrefConstant(Clazz clazz, RefConstant refConstant)
    {
        Method referencedMethod = (Method)refConstant.referencedMember;

        // If the referenced method or a static initializer may modify anything,
        // so does the referencing method.
        if (referencedMethod == null ||
            modifiesAnything(referencedMethod) ||
            SideEffectClassChecker.mayHaveSideEffects(clazz,
                                                      refConstant.referencedClass,
                                                      referencedMethod))
        {
            markAnythingModified(referencingMethod);
        }

        // Do we know the invoked method?
        if (referencedMethod == null)
        {
            // Mark all parameters of the invoking method that are passed to
            // the invoked method, since they may escape or or be modified
            // there.
            for (int parameterOffset = 0; parameterOffset < referencingPopCount; parameterOffset++)
            {
                int stackEntryIndex = referencingPopCount - parameterOffset - 1;

                markEscapingParameters(referencingMethod,
                                       referencingOffset,
                                       stackEntryIndex);

                markModifiedParameters(referencingMethod,
                                       referencingOffset,
                                       stackEntryIndex);
            }
        }
        else
        {
            // Remember whether the return value of the method is escaping or
            // modified later on.
            isReturnValueEscaping =
                referenceEscapeChecker.isInstanceEscaping(referencingOffset);

            isReturnValueModified =
                referenceEscapeChecker.isInstanceModified(referencingOffset);

            // Mark parameters of the invoking method that are passed to the
            // invoked method and escaping or modified there.
            refConstant.referencedMemberAccept(parameterMarker);
        }
    }


    // Implementations for ParameterVisitor.

    public void visitParameter(Clazz clazz, Member member, int parameterIndex, int parameterCount, int parameterOffset, int parameterSize, String parameterType, Clazz referencedClass)
    {
        if (!ClassUtil.isInternalPrimitiveType(parameterType.charAt(0)))
        {
            Method method = (Method)member;

            // Is the parameter escaping from the method,
            // or is it returned and then escaping?
            if (isParameterEscaping(method, parameterIndex) ||
                (isParameterReturned(method, parameterIndex) &&
                 isReturnValueEscaping))
            {
                markEscapingParameters(referencingMethod,
                                       referencingOffset,
                                       parameterSize - parameterOffset - 1);
            }

            // Is the parameter being modified in the method.
            // or is it returned and then modified?
            if (isParameterModified(method, parameterIndex) ||
                (isParameterReturned(method, parameterIndex) &&
                 isReturnValueModified))
            {
                markModifiedParameters(referencingMethod,
                                       referencingOffset,
                                       parameterSize - parameterOffset - 1);
            }
        }
    }


    // Small utility methods.

    /**
     * Marks the producing reference parameters (and the classes) of the
     * specified stack entry at the given instruction offset.
     */
    private void markEscapingParameters(Method method,
                                        int    consumerOffset,
                                        int    stackEntryIndex)
    {
        TracedStack stackBefore = partialEvaluator.getStackBefore(consumerOffset);
        Value       stackEntry  = stackBefore.getTop(stackEntryIndex);

        if (stackEntry.computationalType() == Value.TYPE_REFERENCE)
        {
            ReferenceValue referenceValue = stackEntry.referenceValue();

            // The null reference value may not have a trace value.
            if (referenceValue.isNull() != Value.ALWAYS)
            {
                markEscapingParameters(method, referenceValue);
            }
        }
    }


    /**
     * Marks the producing parameters (and the classes) of the given
     * reference value.
     */
    private void markEscapingParameters(Method         method,
                                        ReferenceValue referenceValue)
    {
        TracedReferenceValue   tracedReferenceValue = (TracedReferenceValue)referenceValue;
        InstructionOffsetValue producers            = tracedReferenceValue.getTraceValue().instructionOffsetValue();

        int producerCount = producers.instructionOffsetCount();
        for (int index = 0; index < producerCount; index++)
        {
            if (producers.isMethodParameter(index))
            {
                // We know exactly which parameter is escaping.
                markParameterEscaping(method, producers.methodParameter(index));
            }
        }
    }


    /**
     * Marks the given parameter as escaping from the given method.
     */
    private void markParameterEscaping(Method method, int parameterIndex)
    {
        MethodOptimizationInfo methodOptimizationInfo =
            MethodOptimizationInfo.getMethodOptimizationInfo(method);

        if (!methodOptimizationInfo.isParameterEscaping(parameterIndex) &&
            methodOptimizationInfo instanceof ProgramMethodOptimizationInfo)
        {
            ((ProgramMethodOptimizationInfo)methodOptimizationInfo).setParameterEscaping(parameterIndex);

            // Trigger the repeater if the setter has changed the value.
            if (methodOptimizationInfo.isParameterEscaping(parameterIndex))
            {
                repeatTrigger.set();
            }
        }
    }


    /**
     * Marks the given parameters as escaping from the given method.
     */
    private void markEscapingParameters(Method method, long escapingParameters)
    {
        MethodOptimizationInfo methodOptimizationInfo =
            MethodOptimizationInfo.getMethodOptimizationInfo(method);

        long oldEscapingParameters =
            methodOptimizationInfo.getEscapingParameters();

        if ((~oldEscapingParameters & escapingParameters) != 0 &&
            methodOptimizationInfo instanceof ProgramMethodOptimizationInfo)
        {
            ((ProgramMethodOptimizationInfo)methodOptimizationInfo).updateEscapingParameters(escapingParameters);

            // Trigger the repeater if the setter has changed the value.
            if (methodOptimizationInfo.getEscapingParameters() != oldEscapingParameters)
            {
                repeatTrigger.set();
            }
        }
    }


    /**
     * Returns whether the given parameter is escaping from the given method.
     */
    public static boolean isParameterEscaping(Method method, int parameterIndex)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).isParameterEscaping(parameterIndex);
    }


    /**
     * Returns which parameters are escaping from the given method.
     */
    public static long getEscapingParameters(Method method)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).getEscapingParameters();
    }


    /**
     * Marks the method and the returned reference parameters of the specified
     * stack entry at the given instruction offset.
     */
    private void markReturnedParameters(Clazz  clazz,
                                        Method method,
                                        int    returnOffset,
                                        int    stackEntryIndex)
    {
        TracedStack stackBefore = partialEvaluator.getStackBefore(returnOffset);
        Value       stackEntry  = stackBefore.getTop(stackEntryIndex);

        if (stackEntry.computationalType() == Value.TYPE_REFERENCE)
        {
            ReferenceValue referenceValue = stackEntry.referenceValue();

            // The null reference value may not have a trace value.
            if (referenceValue.isNull() != Value.ALWAYS &&
                mayReturnType(clazz, method, referenceValue))
            {
                markReturnedParameters(method, referenceValue);
            }
        }
    }


    /**
     * Marks the method and the producing parameters of the given reference
     * value.
     */
    private void markReturnedParameters(Method         method,
                                        ReferenceValue referenceValue)
    {
        TracedReferenceValue   tracedReferenceValue = (TracedReferenceValue)referenceValue;
        InstructionOffsetValue producers            = tracedReferenceValue.getTraceValue().instructionOffsetValue();

        int producerCount = producers.instructionOffsetCount();
        for (int index = 0; index < producerCount; index++)
        {
            if (producers.isMethodParameter(index))
            {
                // We know exactly which parameter is returned.
                markParameterReturned(method, producers.methodParameter(index));
            }
            else if (producers.isFieldValue(index))
            {
                markReturnsExternalValues(method);
            }
            else if (producers.isNewinstance(index) ||
                     producers.isExceptionHandler(index))
            {
                markReturnsNewInstances(method);
            }
        }
    }


    /**
     * Marks the given parameter as returned from the given method.
     */
    private void markParameterReturned(Method method, int parameterIndex)
    {
        MethodOptimizationInfo methodOptimizationInfo =
            MethodOptimizationInfo.getMethodOptimizationInfo(method);

        if (!methodOptimizationInfo.returnsParameter(parameterIndex) &&
            methodOptimizationInfo instanceof ProgramMethodOptimizationInfo)
        {
            ((ProgramMethodOptimizationInfo)methodOptimizationInfo).setParameterReturned(parameterIndex);

            // Trigger the repeater if the setter has changed the value.
            if (methodOptimizationInfo.returnsParameter(parameterIndex))
            {
                repeatTrigger.set();
            }
        }
    }


    /**
     * Marks the given parameters as returned from the given method.
     */
    private void markReturnedParameters(Method method, long returnedParameters)
    {
        MethodOptimizationInfo methodOptimizationInfo =
            MethodOptimizationInfo.getMethodOptimizationInfo(method);

        long oldReturnedParameters =
            methodOptimizationInfo.getReturnedParameters();

        if ((~oldReturnedParameters & returnedParameters) != 0 &&
            methodOptimizationInfo instanceof ProgramMethodOptimizationInfo)
        {
            ((ProgramMethodOptimizationInfo)methodOptimizationInfo).updateReturnedParameters(returnedParameters);

            // Trigger the repeater if the setter has changed the value.
            if (methodOptimizationInfo.getReturnedParameters() != oldReturnedParameters)
            {
                repeatTrigger.set();
            }
        }
    }


    /**
     * Returns whether the given parameter is returned from the given method.
     */
    public static boolean isParameterReturned(Method method, int parameterIndex)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).returnsParameter(parameterIndex);
    }


    /**
     * Returns which parameters are returned from the given method.
     */
    public static long getReturnedParameters(Method method)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).getReturnedParameters();
    }


    /**
     * Marks that the given method returns new instances (created inside the
     * method).
     */
    private void markReturnsNewInstances(Method method)
    {
        MethodOptimizationInfo methodOptimizationInfo =
            MethodOptimizationInfo.getMethodOptimizationInfo(method);

        if (!methodOptimizationInfo.returnsNewInstances() &&
            methodOptimizationInfo instanceof ProgramMethodOptimizationInfo)
        {
            ((ProgramMethodOptimizationInfo)methodOptimizationInfo).setReturnsNewInstances();

            // Trigger the repeater if the setter has changed the value.
            if (methodOptimizationInfo.returnsNewInstances())
            {
                repeatTrigger.set();
            }
        }
    }


    /**
     * Returns whether the given method returns new instances (created inside
     * the method).
     */
    public static boolean returnsNewInstances(Method method)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).returnsNewInstances();
    }


    /**
     * Marks that the given method returns external reference values (not
     * parameter or new instance).
     */
    private void markReturnsExternalValues(Method method)
    {
        MethodOptimizationInfo methodOptimizationInfo =
            MethodOptimizationInfo.getMethodOptimizationInfo(method);

        if (!methodOptimizationInfo.returnsExternalValues() &&
            methodOptimizationInfo instanceof ProgramMethodOptimizationInfo)
        {
            ((ProgramMethodOptimizationInfo)methodOptimizationInfo).setReturnsExternalValues();

            // Trigger the repeater if the setter has changed the value.
            if (methodOptimizationInfo.returnsExternalValues())
            {
                repeatTrigger.set();
            }
        }
    }


    /**
     * Returns whether the given method returns external reference values
     * (not parameter or new instance).
     */
    public static boolean returnsExternalValues(Method method)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).returnsExternalValues();
    }


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


    /**
     * Marks the producing reference parameters of the specified stack entry at
     * the given instruction offset.
     */
    private void markModifiedParameters(Method method,
                                        int    offset,
                                        int    stackEntryIndex)
    {
        TracedStack stackBefore = partialEvaluator.getStackBefore(offset);
        Value       stackEntry  = stackBefore.getTop(stackEntryIndex);

        if (stackEntry.computationalType() == Value.TYPE_REFERENCE)
        {
            ReferenceValue referenceValue = stackEntry.referenceValue();

            // The null reference value may not have a trace value.
            if (referenceValue.isNull() != Value.ALWAYS)
            {
                markModifiedParameters(method, referenceValue);
            }
        }
    }


    /**
     * Marks the producing parameters of the given reference value.
     */
    private void markModifiedParameters(Method         method,
                                        ReferenceValue referenceValue)
    {
        TracedReferenceValue   tracedReferenceValue = (TracedReferenceValue)referenceValue;
        InstructionOffsetValue producers            = tracedReferenceValue.getTraceValue().instructionOffsetValue();

        int producerCount = producers.instructionOffsetCount();
        for (int index = 0; index < producerCount; index++)
        {
            if (producers.isMethodParameter(index))
            {
                // We know exactly which parameter is being modified.
                markParameterModified(method, producers.methodParameter(index));
            }
            else if (!producers.isNewinstance(index) &&
                     !producers.isExceptionHandler(index))
            {
                // If some unknown instance is modified, any escaping parameters
                // may be modified.
                markModifiedParameters(method, getEscapingParameters(method));
                markAnythingModified(method);
            }
        }
    }


    /**
     * Marks the given parameter as modified by the given method.
     */
    private void markParameterModified(Method method, int parameterIndex)
    {
        MethodOptimizationInfo methodOptimizationInfo =
            MethodOptimizationInfo.getMethodOptimizationInfo(method);

        if (!methodOptimizationInfo.isParameterModified(parameterIndex) &&
            methodOptimizationInfo instanceof ProgramMethodOptimizationInfo)
        {
            ((ProgramMethodOptimizationInfo)methodOptimizationInfo).setParameterModified(parameterIndex);

            // Trigger the repeater if the setter has changed the value.
            if (methodOptimizationInfo.isParameterModified(parameterIndex))
            {
                repeatTrigger.set();
            }
        }
    }


    /**
     * Marks the given parameters as modified by the given method.
     */
    private void markModifiedParameters(Method method, long modifiedParameters)
    {
        MethodOptimizationInfo methodOptimizationInfo =
            MethodOptimizationInfo.getMethodOptimizationInfo(method);

        long oldModifiedParameters =
            methodOptimizationInfo.getModifiedParameters();

        if ((~oldModifiedParameters & modifiedParameters) != 0 &&
            methodOptimizationInfo instanceof ProgramMethodOptimizationInfo)
        {
            ((ProgramMethodOptimizationInfo)methodOptimizationInfo).updateModifiedParameters(modifiedParameters);

            // Trigger the repeater if the setter has changed the value.
            if (methodOptimizationInfo.getModifiedParameters() != oldModifiedParameters)
            {
                repeatTrigger.set();
            }
        }
    }


    /**
     * Returns whether the given parameter is modified by the given method.
     */
    public static boolean isParameterModified(Method method, int parameterIndex)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).isParameterModified(parameterIndex);
    }


    /**
     * Returns which parameters are modified by the given method.
     */
    public static long getModifiedParameters(Method method)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).getModifiedParameters();
    }


    /**
     * Marks that anything may be modified by the given method.
     */
    private void markAnythingModified(Method method)
    {
        MethodOptimizationInfo methodOptimizationInfo =
            MethodOptimizationInfo.getMethodOptimizationInfo(method);

        if (!methodOptimizationInfo.modifiesAnything() &&
            methodOptimizationInfo instanceof ProgramMethodOptimizationInfo)
        {
            ((ProgramMethodOptimizationInfo)methodOptimizationInfo).setModifiesAnything();

            // Trigger the repeater if the setter has changed the value.
            if (methodOptimizationInfo.modifiesAnything())
            {
                repeatTrigger.set();
            }
        }
    }


    /**
     * Returns whether anything may be modified by the given method. This takes
     * into account the side effects of static initializers, except the static
     * initializer of the invoked method (because it is better checked
     * explicitly as a function of the referencing class).
     *
     * @see SideEffectClassChecker#mayHaveSideEffects(Clazz, Clazz, Member)
     */
    public static boolean modifiesAnything(Method method)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).modifiesAnything();
    }
}
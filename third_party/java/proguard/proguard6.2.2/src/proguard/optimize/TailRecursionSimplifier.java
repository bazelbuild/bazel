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
package proguard.optimize;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.editor.CodeAttributeComposer;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;

/**
 * This MemberVisitor simplifies tail recursion calls in  all methods that it
 * visits.
 *
 * @author Eric Lafortune
 */
public class TailRecursionSimplifier
extends      SimplifiedVisitor
implements   AttributeVisitor,
             InstructionVisitor,
             ConstantVisitor,
             ExceptionInfoVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = true;
    //*/


    private final InstructionVisitor extraTailRecursionVisitor;


    private final CodeAttributeComposer codeAttributeComposer = new CodeAttributeComposer();
    private final MyRecursionChecker    recursionChecker      = new MyRecursionChecker();
    private final StackSizeComputer     stackSizeComputer     = new StackSizeComputer();

    private Method  targetMethod;
    private boolean inlinedAny;



    /**
     * Creates a new TailRecursionSimplifier.
     */
    public TailRecursionSimplifier()
    {
        this(null);
    }


    /**
     * Creates a new TailRecursionSimplifier with an extra visitor.
     * @param extraTailRecursionVisitor an optional extra visitor for all
     *                                  simplified tail recursions.
     */
    public TailRecursionSimplifier(InstructionVisitor extraTailRecursionVisitor)
    {
        this.extraTailRecursionVisitor = extraTailRecursionVisitor;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        int accessFlags = method.getAccessFlags();

        if (// Only check the method if it is private, static, or final.
            (accessFlags & (ClassConstants.ACC_PRIVATE |
                            ClassConstants.ACC_STATIC  |
                            ClassConstants.ACC_FINAL)) != 0 &&

            // Only check the method if it is not synchronized, etc.
            (accessFlags & (ClassConstants.ACC_SYNCHRONIZED |
                            ClassConstants.ACC_NATIVE       |
                            ClassConstants.ACC_ABSTRACT)) == 0)
        {
//            codeAttributeComposer.DEBUG = DEBUG =
//                clazz.getName().equals("abc/Def") &&
//                method.getName(clazz).equals("abc");

            targetMethod = method;
            inlinedAny   = false;
            codeAttributeComposer.reset();

            // The code may expand, due to expanding constant and variable
            // instructions.
            codeAttributeComposer.beginCodeFragment(codeAttribute.u4codeLength);

            // Copy the instructions.
            codeAttribute.instructionsAccept(clazz, method, this);

            // Update the code attribute if any code has been inlined.
            if (inlinedAny)
            {
                // Append a label just after the code.
                codeAttributeComposer.appendLabel(codeAttribute.u4codeLength);

                // Copy the exceptions.
                codeAttribute.exceptionsAccept(clazz, method, this);

                codeAttributeComposer.endCodeFragment();

                codeAttributeComposer.visitCodeAttribute(clazz, method, codeAttribute);
            }
        }
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
    {
        // Copy the instruction.
        codeAttributeComposer.appendInstruction(offset, instruction);
    }


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        // Is it a method invocation?
        switch (constantInstruction.opcode)
        {
            case InstructionConstants.OP_INVOKEVIRTUAL:
            case InstructionConstants.OP_INVOKESPECIAL:
            case InstructionConstants.OP_INVOKESTATIC:
            {
                // Is it a recursive call?
                clazz.constantPoolEntryAccept(constantInstruction.constantIndex, recursionChecker);

                if (recursionChecker.isRecursive())
                {
                    // Is the next instruction a return?
                    int nextOffset =
                        offset + constantInstruction.length(offset);

                    Instruction nextInstruction =
                        InstructionFactory.create(codeAttribute.code, nextOffset);

                    switch (nextInstruction.opcode)
                    {
                        case InstructionConstants.OP_IRETURN:
                        case InstructionConstants.OP_LRETURN:
                        case InstructionConstants.OP_FRETURN:
                        case InstructionConstants.OP_DRETURN:
                        case InstructionConstants.OP_ARETURN:
                        case InstructionConstants.OP_RETURN:
                        {
                            // Isn't the recursive call inside a try/catch block?
                            codeAttribute.exceptionsAccept(clazz, method, offset, recursionChecker);

                            if (recursionChecker.isRecursive())
                            {
                                // Is the stack empty after the return?
                                stackSizeComputer.visitCodeAttribute(clazz, method, codeAttribute);

                                if (stackSizeComputer.getStackSizeAfter(nextOffset) == 0)
                                {
                                    if (DEBUG)
                                    {
                                        System.out.println("TailRecursionSimplifier: ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"], inlining "+constantInstruction.toString(offset));
                                    }

                                    // Append a label.
                                    codeAttributeComposer.appendLabel(offset);

                                    storeParameters(clazz, method);

                                    // Branch back to the start of the method.
                                    int gotoOffset = offset + 1;
                                    codeAttributeComposer.appendInstruction(gotoOffset,
                                                                            new BranchInstruction(InstructionConstants.OP_GOTO, -gotoOffset));

                                    // The original return instruction will be
                                    // removed elsewhere, if possible.

                                    // Remember that the code has changed.
                                    inlinedAny = true;

                                    if (extraTailRecursionVisitor != null)
                                    {
                                        extraTailRecursionVisitor.visitConstantInstruction(clazz, method, codeAttribute, offset, constantInstruction);
                                    }

                                    // The invocation itself is no longer necessary.
                                    return;
                                }
                            }
                        }
                    }
                }

                break;
            }
        }

        // Copy the instruction.
        codeAttributeComposer.appendInstruction(offset, constantInstruction);
    }


    // Implementations for ExceptionInfoVisitor.

    public void visitExceptionInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, ExceptionInfo exceptionInfo)
    {
        codeAttributeComposer.appendException(new ExceptionInfo(exceptionInfo.u2startPC,
                                                                exceptionInfo.u2endPC,
                                                                exceptionInfo.u2handlerPC,
                                                                exceptionInfo.u2catchType));
    }


    /**
     * This ConstantVisitor and ExceptionInfoVisitor returns whether a method
     * invocation can be treated as tail-recursive.
     */
    private class MyRecursionChecker
    extends       SimplifiedVisitor
    implements    ConstantVisitor,
                  ExceptionInfoVisitor
    {
        private boolean recursive;


        /**
         * Returns whether the method invocation can be treated as
         * tail-recursive.
         */
        public boolean isRecursive()
        {
            return recursive;
        }

        // Implementations for ConstantVisitor.

        public void visitAnyMethodrefConstant(Clazz clazz, RefConstant methodrefConstant)
        {
            recursive = targetMethod.equals(methodrefConstant.referencedMember);
        }


        // Implementations for ExceptionInfoVisitor.

        public void visitExceptionInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, ExceptionInfo exceptionInfo)
        {
            recursive = false;
        }
    }


    // Small utility methods.

    /**
     * Appends instructions to pop the parameters for the given method, storing
     * them in new local variables.
     */
    private void storeParameters(Clazz clazz, Method method)
    {
        String descriptor = method.getDescriptor(clazz);

        boolean isStatic =
            (method.getAccessFlags() & ClassConstants.ACC_STATIC) != 0;

        // Count the number of parameters, taking into account their categories.
        int parameterSize   = ClassUtil.internalMethodParameterSize(descriptor);
        int parameterOffset = isStatic ? 0 : 1;

        // Store the parameter types.
        String[] parameterTypes = new String[parameterSize];

        InternalTypeEnumeration internalTypeEnumeration =
            new InternalTypeEnumeration(descriptor);

        for (int parameterIndex = 0; parameterIndex < parameterSize; parameterIndex++)
        {
            String parameterType = internalTypeEnumeration.nextType();
            parameterTypes[parameterIndex] = parameterType;
            if (ClassUtil.internalTypeSize(parameterType) == 2)
            {
                parameterIndex++;
            }
        }

        codeAttributeComposer.beginCodeFragment(parameterSize + 1);

        // Go over the parameter types backward, storing the stack entries
        // in their corresponding variables.
        for (int parameterIndex = parameterSize-1; parameterIndex >= 0; parameterIndex--)
        {
            String parameterType = parameterTypes[parameterIndex];
            if (parameterType != null)
            {
                byte opcode;
                switch (parameterType.charAt(0))
                {
                    case ClassConstants.TYPE_BOOLEAN:
                    case ClassConstants.TYPE_BYTE:
                    case ClassConstants.TYPE_CHAR:
                    case ClassConstants.TYPE_SHORT:
                    case ClassConstants.TYPE_INT:
                        opcode = InstructionConstants.OP_ISTORE;
                        break;

                    case ClassConstants.TYPE_LONG:
                        opcode = InstructionConstants.OP_LSTORE;
                        break;

                    case ClassConstants.TYPE_FLOAT:
                        opcode = InstructionConstants.OP_FSTORE;
                        break;

                    case ClassConstants.TYPE_DOUBLE:
                        opcode = InstructionConstants.OP_DSTORE;
                        break;

                    default:
                        opcode = InstructionConstants.OP_ASTORE;
                        break;
                }

                codeAttributeComposer.appendInstruction(parameterSize-parameterIndex-1,
                                                        new VariableInstruction(opcode, parameterOffset + parameterIndex));
            }
        }

        // Put the 'this' reference in variable 0.
        if (!isStatic)
        {
            codeAttributeComposer.appendInstruction(parameterSize,
                                                    new VariableInstruction(InstructionConstants.OP_ASTORE, 0));
        }

        codeAttributeComposer.endCodeFragment();
    }
}
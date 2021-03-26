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
package proguard.evaluation;

import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.evaluation.value.*;

/**
 * This InvocationUnit sets up the variables for entering a method,
 * and it updates the stack for the invocation of a class member,
 * using simple values.
 *
 * @author Eric Lafortune
 */
public abstract class SimplifiedInvocationUnit
extends               SimplifiedVisitor
implements            InvocationUnit,
                      ParameterVisitor,
                      ConstantVisitor
{
    private final MemberVisitor parameterInitializer = new AllParameterVisitor(true, this);

    // Fields acting as parameters between the visitor methods.
    private   Variables variables;
    protected boolean   isStatic;
    protected boolean   isLoad;
    protected Stack     stack;


    // Implementations for InvocationUnit.

    public void enterMethod(Clazz clazz, Method method, Variables variables)
    {
        // Count the number of parameters, taking into account their categories.
        int parameterSize =
            ClassUtil.internalMethodParameterSize(method.getDescriptor(clazz),
                                                  method.getAccessFlags());

        // Reuse the existing parameters object, ensuring the right size.
        variables.reset(parameterSize);

        // Initialize the parameters.
        this.variables = variables;
        method.accept(clazz, parameterInitializer);
        this.variables = null;
    }


    // Implementation for ParameterVisitor.

    public void visitParameter(Clazz clazz, Member member, int parameterIndex, int parameterCount, int parameterOffset, int parameterSize, String parameterType, Clazz referencedClass)
    {
        Method method = (Method)member;

        // Get the parameter value.
        Value value = getMethodParameterValue(clazz,
                                              method,
                                              parameterIndex,
                                              parameterType,
                                              referencedClass);

        // Store the value in the corresponding variable.
        variables.store(parameterOffset, value);
    }


    public void exitMethod(Clazz clazz, Method method, Value returnValue)
    {
        setMethodReturnValue(clazz, method, returnValue);
    }


    public void enterExceptionHandler(Clazz         clazz,
                                      Method        method,
                                      CodeAttribute codeAttribute,
                                      int           offset,
                                      int           catchType,
                                      Stack         stack)
    {
        ClassConstant exceptionClassConstant =
            (ClassConstant)((ProgramClass)clazz).getConstant(catchType);

        stack.push(getExceptionValue(clazz, exceptionClassConstant));
    }


    public void invokeMember(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction, Stack stack)
    {
        int constantIndex = constantInstruction.constantIndex;

        switch (constantInstruction.opcode)
        {
            case InstructionConstants.OP_GETSTATIC:
                isStatic = true;
                isLoad   = true;
                break;

            case InstructionConstants.OP_PUTSTATIC:
                isStatic = true;
                isLoad   = false;
                break;

            case InstructionConstants.OP_GETFIELD:
                isStatic = false;
                isLoad   = true;
                break;

            case InstructionConstants.OP_PUTFIELD:
                isStatic = false;
                isLoad   = false;
                break;

            case InstructionConstants.OP_INVOKESTATIC:
            case InstructionConstants.OP_INVOKEDYNAMIC:
                isStatic = true;
                break;

            case InstructionConstants.OP_INVOKEVIRTUAL:
            case InstructionConstants.OP_INVOKESPECIAL:
            case InstructionConstants.OP_INVOKEINTERFACE:
                isStatic = false;
                break;
        }

        // Pop the parameters and push the return value.
        this.stack = stack;
        clazz.constantPoolEntryAccept(constantIndex, this);
        this.stack = null;
    }


    // Implementations for ConstantVisitor.

    public void visitFieldrefConstant(Clazz clazz, FieldrefConstant fieldrefConstant)
    {
        // Pop the field value, if applicable.
        if (!isLoad)
        {
            setFieldValue(clazz, fieldrefConstant, stack.pop());
        }

        // Pop the reference value, if applicable.
        if (!isStatic)
        {
            setFieldClassValue(clazz, fieldrefConstant, stack.apop());
        }

        // Push the field value, if applicable.
        if (isLoad)
        {
            String type = fieldrefConstant.getType(clazz);

            stack.push(getFieldValue(clazz, fieldrefConstant, type));
        }
    }


    public void visitAnyMethodrefConstant(Clazz clazz, RefConstant methodrefConstant)
    {
        String type = methodrefConstant.getType(clazz);

        // Count the number of parameters.
        int parameterCount = ClassUtil.internalMethodParameterCount(type, isStatic);

        // Pop the parameters and the class reference, in reverse order.
        for (int parameterIndex = parameterCount-1; parameterIndex >= 0; parameterIndex--)
        {
            setMethodParameterValue(clazz, methodrefConstant, parameterIndex, stack.pop());
        }

        // Push the return value, if applicable.
        String returnType = ClassUtil.internalMethodReturnType(type);
        if (returnType.charAt(0) != ClassConstants.TYPE_VOID)
        {
            stack.push(getMethodReturnValue(clazz, methodrefConstant, returnType));
        }
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        String type = invokeDynamicConstant.getType(clazz);

        // Count the number of parameters.
        int parameterCount = ClassUtil.internalMethodParameterCount(type, isStatic);

        // Pop the parameters and the class reference, in reverse order.
        for (int parameterIndex = parameterCount-1; parameterIndex >= 0; parameterIndex--)
        {
            stack.pop();
        }

        // Push the return value, if applicable.
        String returnType = ClassUtil.internalMethodReturnType(type);
        if (returnType.charAt(0) != ClassConstants.TYPE_VOID)
        {
            stack.push(getMethodReturnValue(clazz, invokeDynamicConstant, returnType));
        }
    }


    /**
     * Returns the value of the specified exception.
     */
    public abstract Value getExceptionValue(Clazz         clazz,
                                            ClassConstant catchClassConstant);


    /**
     * Sets the class through which the specified field is accessed.
     */
    public abstract void setFieldClassValue(Clazz          clazz,
                                            RefConstant    refConstant,
                                            ReferenceValue value);


    /**
     * Returns the class though which the specified field is accessed.
     */
    public abstract Value getFieldClassValue(Clazz       clazz,
                                             RefConstant refConstant,
                                             String      type);


    /**
     * Sets the value of the specified field.
     */
    public abstract void setFieldValue(Clazz       clazz,
                                       RefConstant refConstant,
                                       Value       value);


    /**
     * Returns the value of the specified field.
     */
    public abstract Value getFieldValue(Clazz       clazz,
                                        RefConstant refConstant,
                                        String      type);


    /**
     * Sets the value of the specified method parameter.
     */
    public abstract void setMethodParameterValue(Clazz       clazz,
                                                 RefConstant refConstant,
                                                 int         parameterIndex,
                                                 Value       value);


    /**
     * Returns the value of the specified method parameter.
     */
    public abstract Value getMethodParameterValue(Clazz  clazz,
                                                  Method method,
                                                  int    parameterIndex,
                                                  String type,
                                                  Clazz  referencedClass);


    /**
     * Sets the return value of the specified method.
     */
    public abstract void setMethodReturnValue(Clazz  clazz,
                                              Method method,
                                              Value  value);


    /**
     * Returns the return value of the specified method.
     */
    public abstract Value getMethodReturnValue(Clazz       clazz,
                                               RefConstant refConstant,
                                               String      type);


    /**
     * Returns the return value of the specified method.
     */
    public abstract Value getMethodReturnValue(Clazz                 clazz,
                                               InvokeDynamicConstant invokeDynamicConstant,
                                               String                type);
}

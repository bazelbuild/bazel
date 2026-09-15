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
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;

/**
 * This InstructionVisitor marks the types of class accesses and class member
 * accesses of the methods whose instructions it visits.
 *
 * @author Eric Lafortune
 */
public class AccessMethodMarker
extends      SimplifiedVisitor
implements   InstructionVisitor,
             ConstantVisitor,
             ClassVisitor,
             MemberVisitor
{
    private Method invokingMethod;


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        invokingMethod = method;

        clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
       // Check the referenced class or class member, if any.
       stringConstant.referencedClassAccept(this);
       stringConstant.referencedMemberAccept(this);
    }


    public void visitDynamicConstant(Clazz clazz, DynamicConstant dynamicConstant)
    {
        // Check the bootstrap method.
        dynamicConstant.bootstrapMethodHandleAccept(clazz, this);
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        // Check the bootstrap method.
        invokeDynamicConstant.bootstrapMethodHandleAccept(clazz, this);
    }


    public void visitMethodHandleConstant(Clazz clazz, MethodHandleConstant methodHandleConstant)
    {
        // Check the method reference.
        clazz.constantPoolEntryAccept(methodHandleConstant.u2referenceIndex, this);
    }


    public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
    {
        // Check the referenced class.
        clazz.constantPoolEntryAccept(refConstant.u2classIndex, this);

        // Check the referenced class member itself.
        refConstant.referencedClassAccept(this);
        refConstant.referencedMemberAccept(this);
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        // Check the referenced class.
       classConstant.referencedClassAccept(this);
    }


    // Implementations for ClassVisitor.

    public void visitAnyClass(Clazz clazz)
    {
        int accessFlags = clazz.getAccessFlags();

        if ((accessFlags & ClassConstants.ACC_PUBLIC) == 0)
        {
            setAccessesPackageCode(invokingMethod);
        }
    }


    // Implementations for MemberVisitor.

    public void visitAnyMember(Clazz clazz, Member member)
    {
        int accessFlags = member.getAccessFlags();

        if      ((accessFlags & ClassConstants.ACC_PRIVATE)   != 0)
        {
            setAccessesPrivateCode(invokingMethod);
        }
        else if ((accessFlags & ClassConstants.ACC_PROTECTED) != 0)
        {
            setAccessesProtectedCode(invokingMethod);
        }
        else if ((accessFlags & ClassConstants.ACC_PUBLIC)    == 0)
        {
            setAccessesPackageCode(invokingMethod);
        }
    }


    // Small utility methods.

    private static void setAccessesPrivateCode(Method method)
    {
        ProgramMethodOptimizationInfo.getProgramMethodOptimizationInfo(method).setAccessesPrivateCode();
    }


    /**
     * Returns whether the given method accesses private class members.
     */
    public static boolean accessesPrivateCode(Method method)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).accessesPrivateCode();
    }


    private static void setAccessesPackageCode(Method method)
    {
        ProgramMethodOptimizationInfo.getProgramMethodOptimizationInfo(method).setAccessesPackageCode();
    }


    /**
     * Returns whether the given method accesses package visible classes or class
     * members.
     */
    public static boolean accessesPackageCode(Method method)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).accessesPackageCode();
    }


    private static void setAccessesProtectedCode(Method method)
    {
        ProgramMethodOptimizationInfo.getProgramMethodOptimizationInfo(method).setAccessesProtectedCode();
    }


    /**
     * Returns whether the given method accesses protected class members.
     */
    public static boolean accessesProtectedCode(Method method)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).accessesProtectedCode();
    }
}

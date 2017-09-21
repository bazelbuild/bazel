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
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;

/**
 * This ClassPoolVisitor marks all methods that have side effects.
 *
 * @see ReadWriteFieldMarker
 * @see NoSideEffectMethodMarker
 * @author Eric Lafortune
 */
public class SideEffectMethodMarker
extends      SimplifiedVisitor
implements   ClassPoolVisitor,
             ClassVisitor,
             MemberVisitor,
             AttributeVisitor
{
    // Reusable objects for checking whether instructions have side effects.
    private final SideEffectInstructionChecker sideEffectInstructionChecker            = new SideEffectInstructionChecker(false, true);
    private final SideEffectInstructionChecker initializerSideEffectInstructionChecker = new SideEffectInstructionChecker(false, false);

    // Parameters and values for visitor methods.
    private int     newSideEffectCount;
    private boolean hasSideEffects;


    // Implementations for ClassPoolVisitor.

    public void visitClassPool(ClassPool classPool)
    {
        // Go over all classes and their methods, marking if they have side
        // effects, until no new cases can be found.
        do
        {
            newSideEffectCount = 0;

            // Go over all classes and their methods once.
            classPool.classesAccept(this);
        }
        while (newSideEffectCount > 0);
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Go over all methods.
        programClass.methodsAccept(this);
    }


    // Implementations for MemberVisitor.

    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        if (!hasSideEffects(programMethod) &&
            !NoSideEffectMethodMarker.hasNoSideEffects(programMethod))
        {
            // Initialize the return value.
            hasSideEffects =
                (programMethod.getAccessFlags() &
                 (ClassConstants.ACC_NATIVE |
                  ClassConstants.ACC_SYNCHRONIZED)) != 0;

            // Look further if the method hasn't been marked yet.
            if (!hasSideEffects)
            {
                // Investigate the actual code.
                programMethod.attributesAccept(programClass, this);
            }

            // Mark the method depending on the return value.
            if (hasSideEffects)
            {
                markSideEffects(programMethod);

                newSideEffectCount++;
            }
        }
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Remember whether the code has any side effects.
        hasSideEffects = hasSideEffects(clazz, method, codeAttribute);
    }


    // Small utility methods.

    /**
     * Returns whether the given code has any side effects.
     */
    private boolean hasSideEffects(Clazz         clazz,
                                   Method        method,
                                   CodeAttribute codeAttribute)
    {
        byte[] code   = codeAttribute.code;
        int    length = codeAttribute.u4codeLength;

        SideEffectInstructionChecker checker =
            method.getName(clazz).equals(ClassConstants.METHOD_NAME_CLINIT) ?
                initializerSideEffectInstructionChecker :
                sideEffectInstructionChecker;

        // Go over all instructions.
        int offset = 0;
        do
        {
            // Get the current instruction.
            Instruction instruction = InstructionFactory.create(code, offset);

            // Check if it may be throwing exceptions.
            if (checker.hasSideEffects(clazz,
                                       method,
                                       codeAttribute,
                                       offset,
                                       instruction))
            {
                return true;
            }

            // Go to the next instruction.
            offset += instruction.length(offset);
        }
        while (offset < length);

        return false;
    }


    private static void markSideEffects(Method method)
    {
        MethodOptimizationInfo info = MethodOptimizationInfo.getMethodOptimizationInfo(method);
        if (info != null)
        {
            info.setSideEffects();
        }
    }


    public static boolean hasSideEffects(Method method)
    {
        MethodOptimizationInfo info = MethodOptimizationInfo.getMethodOptimizationInfo(method);
        return info == null ||
               info.hasSideEffects();
    }
}

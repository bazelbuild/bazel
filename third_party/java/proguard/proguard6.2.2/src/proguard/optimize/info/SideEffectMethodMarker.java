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
import proguard.classfile.instruction.Instruction;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;
import proguard.optimize.OptimizationInfoClassFilter;

/**
 * This MemberVisitor and InstructionVisitor marks all methods and classes
 * that have side effects.
 *
 * @see NoSideEffectMethodMarker
 * @author Eric Lafortune
 */
public class SideEffectMethodMarker
extends      SimplifiedVisitor
implements   MemberVisitor,
             InstructionVisitor
{
    private final MutableBoolean repeatTrigger;

    private final SideEffectInstructionChecker sideEffectInstructionChecker = new SideEffectInstructionChecker(false, true);
    private final ClassVisitor                 sideEffectClassMarker        = new OptimizationInfoClassFilter(
                                                                              new SideEffectClassMarker());



    /**
     * Creates a new SideEffectMethodMarker.
     */
    public SideEffectMethodMarker(MutableBoolean repeatTrigger)
    {
        this.repeatTrigger = repeatTrigger;
    }


    // Implementations for MemberVisitor.

    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        if ((programMethod.getAccessFlags() &
                 (ClassConstants.ACC_NATIVE |
                  ClassConstants.ACC_SYNCHRONIZED)) != 0)
        {
            markSideEffects(programClass, programMethod);
        }
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
    {
        // Check if it may be throwing exceptions.
        if (sideEffectInstructionChecker.hasSideEffects(clazz,
                                                        method,
                                                        codeAttribute,
                                                        offset,
                                                        instruction))
        {
            markSideEffects(clazz, method);
        }
    }


    // Small utility methods.

    private void markSideEffects(Clazz clazz, Method method)
    {
        MethodOptimizationInfo methodOptimizationInfo =
            MethodOptimizationInfo.getMethodOptimizationInfo(method);

        if (!methodOptimizationInfo.hasSideEffects()   &&
            methodOptimizationInfo instanceof ProgramMethodOptimizationInfo)
        {
            ((ProgramMethodOptimizationInfo)methodOptimizationInfo).setSideEffects();

            // Trigger the repeater if the setter has changed the value.
            if (methodOptimizationInfo.hasSideEffects())
            {
                repeatTrigger.set();

                // Also mark the class if the method is a static initializer.
                if (method.getName(clazz).equals(ClassConstants.METHOD_NAME_CLINIT))
                {
                    clazz.accept(sideEffectClassMarker);
                }
            }
        }
    }


    public static boolean hasSideEffects(Method method)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).hasSideEffects();
    }
}

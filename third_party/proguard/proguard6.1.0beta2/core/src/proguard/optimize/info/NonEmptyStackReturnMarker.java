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
import proguard.classfile.attribute.visitor.StackSizeComputer;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This InstructionVisitor marks all methods that return with a non-empty stack
 * (other than the return value).
 *
 * @author Eric Lafortune
 */
public class NonEmptyStackReturnMarker
extends      SimplifiedVisitor
implements   InstructionVisitor
{
    private final StackSizeComputer stackSizeComputer;


    /**
     * Creates a new NonEmptyStackReturnMarker
     * @param stackSizeComputer the stack size computer that can return the
     *                          stack sizes at the instructions that are
     *                          visited.
     */
    public NonEmptyStackReturnMarker(StackSizeComputer stackSizeComputer)
    {
        this.stackSizeComputer = stackSizeComputer;
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
    {
        switch (simpleInstruction.opcode)
        {
            case InstructionConstants.OP_LRETURN:
            case InstructionConstants.OP_DRETURN:
                markReturnWithNonEmptyStack(method, offset, 2);
                break;

            case InstructionConstants.OP_IRETURN:
            case InstructionConstants.OP_FRETURN:
            case InstructionConstants.OP_ARETURN:
                markReturnWithNonEmptyStack(method, offset, 1);
                break;

            case InstructionConstants.OP_RETURN:
                markReturnWithNonEmptyStack(method, offset, 0);
                break;
        }
    }


    // Small utility methods.

    /**
     * Marks the given method if the stack before the given instruction offset
     * has a size larger than the given size.
     */
    private void markReturnWithNonEmptyStack(Method method,
                                             int    offset,
                                             int    stackSize)
    {
        if (!stackSizeComputer.isReachable(offset) ||
            stackSizeComputer.getStackSizeBefore(offset) > stackSize)
        {
            setReturnsWithNonEmptyStack(method);
        }
    }


    private static void setReturnsWithNonEmptyStack(Method method)
    {
        ProgramMethodOptimizationInfo.getProgramMethodOptimizationInfo(method).setReturnsWithNonEmptyStack();
    }


    public static boolean returnsWithNonEmptyStack(Method method)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).returnsWithNonEmptyStack();
    }
}

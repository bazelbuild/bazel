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
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;

/**
 * This InstructionVisitor marks whether the methods whose instructions it
 * visits contain the invokedynamic instruction.
 *
 * @author Eric Lafortune
 */
public class DynamicInvocationMarker
extends      SimplifiedVisitor
implements   InstructionVisitor,
             ConstantVisitor,
             ClassVisitor,
             MemberVisitor
{
    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        if (constantInstruction.opcode == InstructionConstants.OP_INVOKEDYNAMIC)
        {
            setInvokesDynamically(method);
        }
    }


    // Small utility methods.

    private static void setInvokesDynamically(Method method)
    {
        ProgramMethodOptimizationInfo.getProgramMethodOptimizationInfo(method).setInvokesDynamically();
    }


    /**
     * Returns whether the given method calls the invokedynamic instruction.
     */
    public static boolean invokesDynamically(Method method)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).invokesDynamically();
    }
}

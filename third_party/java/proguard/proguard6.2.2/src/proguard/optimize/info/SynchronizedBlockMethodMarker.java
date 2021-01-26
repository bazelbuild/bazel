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
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This InstructionVisitor marks the existence of synchronized blocks
 * of the methods whose instructions it visits.
 *
 * @author Thomas Neidhart
 */
public class SynchronizedBlockMethodMarker
extends      SimplifiedVisitor
implements   InstructionVisitor
{
    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
    {
        if (simpleInstruction.opcode == InstructionConstants.OP_MONITORENTER ||
            simpleInstruction.opcode == InstructionConstants.OP_MONITOREXIT)
        {
            setHasSynchronizedBlock(method);
        }
    }

    // Small utility methods.

    private static void setHasSynchronizedBlock(Method method)
    {
        ProgramMethodOptimizationInfo.getProgramMethodOptimizationInfo(method).setHasSynchronizedBlock();
    }


    /**
     * Returns whether the given method accesses private class members.
     */
    public static boolean hasSynchronizedBlock(Method method)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).hasSynchronizedBlock();
    }

}

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
package proguard.classfile.instruction.visitor;

import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.instruction.*;
import proguard.util.ArrayUtil;


/**
 * This InstructionVisitor delegates all visits to each InstructionVisitor
 * in a given list.
 *
 * @author Eric Lafortune
 */
public class MultiInstructionVisitor implements InstructionVisitor
{
    private InstructionVisitor[] instructionVisitors;
    private int                  instructionVisitorCount;


    public MultiInstructionVisitor()
    {
        this.instructionVisitors = new InstructionVisitor[16];
    }


    public MultiInstructionVisitor(InstructionVisitor... instructionVisitors)
    {
        this.instructionVisitors     = instructionVisitors;
        this.instructionVisitorCount = instructionVisitors.length;
    }


    public void addInstructionVisitor(InstructionVisitor instructionVisitor)
    {
        instructionVisitors =
            ArrayUtil.add(instructionVisitors,
                          instructionVisitorCount++,
                          instructionVisitor);
    }


    // Implementations for InstructionVisitor.

    public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
    {
        for (int index = 0; index < instructionVisitorCount; index++)
        {
            instructionVisitors[index].visitSimpleInstruction(clazz, method, codeAttribute, offset, simpleInstruction);
        }
    }

    public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
    {
        for (int index = 0; index < instructionVisitorCount; index++)
        {
            instructionVisitors[index].visitVariableInstruction(clazz, method, codeAttribute, offset, variableInstruction);
        }
    }

    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        for (int index = 0; index < instructionVisitorCount; index++)
        {
            instructionVisitors[index].visitConstantInstruction(clazz, method, codeAttribute, offset, constantInstruction);
        }
    }

    public void visitBranchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction branchInstruction)
    {
        for (int index = 0; index < instructionVisitorCount; index++)
        {
            instructionVisitors[index].visitBranchInstruction(clazz, method, codeAttribute, offset, branchInstruction);
        }
    }

    public void visitTableSwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, TableSwitchInstruction tableSwitchInstruction)
    {
        for (int index = 0; index < instructionVisitorCount; index++)
        {
            instructionVisitors[index].visitTableSwitchInstruction(clazz, method, codeAttribute, offset, tableSwitchInstruction);
        }
    }

    public void visitLookUpSwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, LookUpSwitchInstruction lookUpSwitchInstruction)
    {
        for (int index = 0; index < instructionVisitorCount; index++)
        {
            instructionVisitors[index].visitLookUpSwitchInstruction(clazz, method, codeAttribute, offset, lookUpSwitchInstruction);
        }
    }
}

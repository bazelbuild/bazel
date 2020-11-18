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


/**
 * This interface specifies the methods for a visitor of
 * <code>Instruction</code> objects.
 *
 * @author Eric Lafortune
 */
public interface InstructionVisitor
{
    public void visitSimpleInstruction(      Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction       simpleInstruction);
    public void visitVariableInstruction(    Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction     variableInstruction);
    public void visitConstantInstruction(    Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction     constantInstruction);
    public void visitBranchInstruction(      Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction       branchInstruction);
    public void visitTableSwitchInstruction( Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, TableSwitchInstruction  tableSwitchInstruction);
    public void visitLookUpSwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, LookUpSwitchInstruction lookUpSwitchInstruction);
}

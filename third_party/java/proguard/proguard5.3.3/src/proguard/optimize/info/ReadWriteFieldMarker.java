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
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.MemberVisitor;

/**
 * This InstructionVisitor marks all fields that are write-only.
 *
 * @author Eric Lafortune
 */
public class ReadWriteFieldMarker
extends      SimplifiedVisitor
implements   InstructionVisitor,
             ConstantVisitor,
             MemberVisitor
{
    // Parameters for the visitor methods.
    private boolean reading = true;
    private boolean writing = true;


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        byte opcode = constantInstruction.opcode;

        // Check for instructions that involve fields.
        switch (opcode)
        {
            case InstructionConstants.OP_LDC:
            case InstructionConstants.OP_LDC_W:
                // Mark the field, if any, as being read from and written to.
                reading = true;
                writing = true;
                clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                break;

            case InstructionConstants.OP_GETSTATIC:
            case InstructionConstants.OP_GETFIELD:
                // Mark the field as being read from.
                reading = true;
                writing = false;
                clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                break;

            case InstructionConstants.OP_PUTSTATIC:
            case InstructionConstants.OP_PUTFIELD:
                // Mark the field as being written to.
                reading = false;
                writing = true;
                clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                break;
        }
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        // Mark the referenced field, if any.
        stringConstant.referencedMemberAccept(this);
    }


    public void visitFieldrefConstant(Clazz clazz, FieldrefConstant fieldrefConstant)
    {
        // Mark the referenced field.
        fieldrefConstant.referencedMemberAccept(this);
    }


    // Implementations for MemberVisitor.

    public void visitAnyMember(Clazz Clazz, Member member) {}


    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        // Mark the field if it is being read from.
        if (reading)
        {
            markAsRead(programField);
        }

        // Mark the field if it is being written to.
        if (writing)
        {
            markAsWritten(programField);
        }
    }


    // Small utility methods.

    private static void markAsRead(Field field)
    {
        FieldOptimizationInfo info = FieldOptimizationInfo.getFieldOptimizationInfo(field);
        if (info != null)
        {
            info.setRead();
        }
    }


    public static boolean isRead(Field field)
    {
        FieldOptimizationInfo info = FieldOptimizationInfo.getFieldOptimizationInfo(field);
        return info == null ||
               info.isRead();
    }


    private static void markAsWritten(Field field)
    {
        FieldOptimizationInfo info = FieldOptimizationInfo.getFieldOptimizationInfo(field);
        if (info != null)
        {
            info.setWritten();
        }
    }


    public static boolean isWritten(Field field)
    {
        FieldOptimizationInfo info = FieldOptimizationInfo.getFieldOptimizationInfo(field);
        return info == null ||
               info.isWritten();
    }
}

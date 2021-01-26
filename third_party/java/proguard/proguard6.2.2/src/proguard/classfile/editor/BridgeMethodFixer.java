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
package proguard.classfile.editor;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.constant.RefConstant;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.MemberVisitor;

/**
 * This MemberVisitor fixes all inappropriate bridge access flags of the
 * program methods that it visits, checking whether the methods to which they
 * bridge have the same name. Some compilers, like in Eclipse and in later
 * versions of JDK 1.6, complain if they can't find the method with the same
 * name.
 *
 * @author Eric Lafortune
 */
public class BridgeMethodFixer
extends      SimplifiedVisitor
implements   MemberVisitor,
             AttributeVisitor,
             InstructionVisitor,
             ConstantVisitor
{
    private static final boolean DEBUG = false;


    // Return values for the visitor methods.
    private String bridgedMethodName;


    // Implementations for MemberVisitor.

    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        if ((programMethod.getAccessFlags() & ClassConstants.ACC_BRIDGE) != 0)
        {
            programMethod.attributesAccept(programClass, this);
        }
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Go over the instructions of the bridge method.
        codeAttribute.instructionsAccept(clazz, method, this);
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        switch (constantInstruction.opcode)
        {
            case InstructionConstants.OP_INVOKEVIRTUAL:
            case InstructionConstants.OP_INVOKESPECIAL:
            case InstructionConstants.OP_INVOKESTATIC:
            case InstructionConstants.OP_INVOKEINTERFACE:
                // Get the name of the bridged method.
                clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);

                // Check if the name is different.
                if (!method.getName(clazz).equals(bridgedMethodName))
                {
                    if (DEBUG)
                    {
                        System.out.println("BridgeMethodFixer: ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"] does not bridge to ["+bridgedMethodName+"]");
                    }

                    // Clear the bridge flag.
                    ((ProgramMethod)method).u2accessFlags &= ~ClassConstants.ACC_BRIDGE;
                }
                break;
        }
    }


    // Implementations for ConstantVisitor.

    public void visitAnyMethodrefConstant(Clazz clazz, RefConstant refConstant)
    {
        bridgedMethodName = refConstant.getName(clazz);
    }
}
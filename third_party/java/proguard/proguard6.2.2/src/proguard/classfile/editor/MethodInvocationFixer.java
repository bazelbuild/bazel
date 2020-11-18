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
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;

/**
 * This AttributeVisitor fixes all inappropriate special/virtual/static/interface
 * invocations of the code attributes that it visits.
 *
 * @author Eric Lafortune
 */
public class MethodInvocationFixer
extends      SimplifiedVisitor
implements   AttributeVisitor,
             InstructionVisitor,
             ConstantVisitor
{
    private static final boolean DEBUG = false;


    private final CodeAttributeEditor codeAttributeEditor = new CodeAttributeEditor();
    private final NestHostFinder      nestHostFinder      = new NestHostFinder();

    // Return values for the visitor methods.
    private Clazz  referencedClass;
    private Clazz  referencedMethodClass;
    private Member referencedMethod;


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Reset the code attribute editor.
        codeAttributeEditor.reset(codeAttribute.u4codeLength);

        // Remap the variables of the instructions.
        codeAttribute.instructionsAccept(clazz, method, this);

        // Apply the code atribute editor.
        codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        int constantIndex = constantInstruction.constantIndex;

        // Get information on the called class and method, if present.
        referencedMethod = null;

        clazz.constantPoolEntryAccept(constantIndex, this);

        // Did we find the called class and method?
        if (referencedClass  != null &&
            referencedMethod != null)
        {
            // Do we need to update the opcode?
            byte opcode = constantInstruction.opcode;

            // Is the method static?
            if ((referencedMethod.getAccessFlags() & ClassConstants.ACC_STATIC) != 0)
            {
                // But is it not a static invocation?
                if (opcode != InstructionConstants.OP_INVOKESTATIC)
                {
                    // Replace the invocation by an invokestatic instruction.
                    Instruction replacementInstruction =
                        new ConstantInstruction(InstructionConstants.OP_INVOKESTATIC,
                                                constantIndex);

                    codeAttributeEditor.replaceInstruction(offset, replacementInstruction);

                    if (DEBUG)
                    {
                        debug(clazz, method, offset, constantInstruction, replacementInstruction);
                    }
                }
            }

            // Is the method private, or an instance initializer?
            else if ((referencedMethod.getAccessFlags() & ClassConstants.ACC_PRIVATE) != 0 ||
                     referencedMethod.getName(referencedMethodClass).equals(ClassConstants.METHOD_NAME_INIT))
            {
                // But is it not a special invocation?
                if (opcode != InstructionConstants.OP_INVOKESPECIAL &&
                    // Check if the two classes are in the same nest.
                    !nestHostFinder.inSameNest(clazz, referencedClass))
                {
                    // Replace the invocation by an invokespecial instruction.
                    Instruction replacementInstruction =
                        new ConstantInstruction(InstructionConstants.OP_INVOKESPECIAL,
                                                constantIndex);

                    codeAttributeEditor.replaceInstruction(offset, replacementInstruction);

                    if (DEBUG)
                    {
                        debug(clazz, method, offset, constantInstruction, replacementInstruction);
                    }
                }
            }

            // Is the method an interface method?
            else if ((referencedClass.getAccessFlags() & ClassConstants.ACC_INTERFACE) != 0)
            {
                int invokeinterfaceConstant =
                    (ClassUtil.internalMethodParameterSize(referencedMethod.getDescriptor(referencedMethodClass), false)) << 8;

                if (opcode == InstructionConstants.OP_INVOKESPECIAL &&
                    (referencedMethod.getAccessFlags() & ClassConstants.ACC_ABSTRACT) == 0)
                {
                    // Explicit calls to default interface methods *must* be preserved.
                }
                // But is it not an interface invocation, or is the parameter
                // size incorrect?
                else if (opcode != InstructionConstants.OP_INVOKEINTERFACE ||
                         constantInstruction.constant != invokeinterfaceConstant)
                {
                    // Fix the parameter size of the interface invocation.
                    Instruction replacementInstruction =
                        new ConstantInstruction(InstructionConstants.OP_INVOKEINTERFACE,
                                                constantIndex,
                                                invokeinterfaceConstant);

                    codeAttributeEditor.replaceInstruction(offset, replacementInstruction);

                    if (DEBUG)
                    {
                        debug(clazz, method, offset, constantInstruction, replacementInstruction);
                    }
                }
            }

            // The method is not static, private, an instance initializer, or
            // an interface method.
            else
            {
                // But is it not a virtual invocation?
                if (opcode != InstructionConstants.OP_INVOKEVIRTUAL &&
                    (// Replace any non-invokespecial.
                     opcode != InstructionConstants.OP_INVOKESPECIAL ||
                     // For invokespecial, replace invocations from the same
                     // class, and invocations to non-superclasses.
                     clazz.equals(referencedClass)                   ||
                     !clazz.extends_(referencedClass)))
                {
                    // Replace the invocation by an invokevirtual instruction.
                    Instruction replacementInstruction =
                        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL,
                                                constantIndex);

                    codeAttributeEditor.replaceInstruction(offset, replacementInstruction);

                    if (DEBUG)
                    {
                        debug(clazz, method, offset, constantInstruction, replacementInstruction);
                    }
                }
            }
        }
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitAnyMethodrefConstant(Clazz clazz, RefConstant refConstant)
    {
        // Remember the referenced class. Note that we're interested in the
        // class of the method reference, not in the class in which the
        // method was actually found, unless it is an array type.
        if (ClassUtil.isInternalArrayType(refConstant.getClassName(clazz)))
        {
            // For an array type, the class will be java.lang.Object.
            referencedClass = refConstant.referencedClass;
        }
        else
        {
            clazz.constantPoolEntryAccept(refConstant.u2classIndex, this);
        }

        // Remember the referenced method.
        referencedMethodClass = refConstant.referencedClass;
        referencedMethod      = refConstant.referencedMember;
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        // Remember the referenced class.
        referencedClass = classConstant.referencedClass;
    }


    // Small utility methods.

    private void debug(Clazz               clazz,
                       Method              method,
                       int                 offset,
                       ConstantInstruction constantInstruction,
                       Instruction         replacementInstruction)
    {
        System.out.println("MethodInvocationFixer ["+clazz.getName()+"."+
                           method.getName(clazz)+method.getDescriptor(clazz)+"] "+
                           constantInstruction.toString(offset)+" -> "+
                           replacementInstruction.toString(offset));
    }
}

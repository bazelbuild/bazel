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
package proguard.optimize;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.editor.CodeAttributeEditor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.MemberVisitor;

/**
 * This AttributeVisitor adds an additional integer parameter to the tweaked
 * initialization method invocations that it visits.
 */
public class DuplicateInitializerInvocationFixer
extends      SimplifiedVisitor
implements   AttributeVisitor,
             InstructionVisitor,
             ConstantVisitor,
             MemberVisitor
{
    private static final boolean DEBUG = false;

    private final InstructionVisitor extraAddedInstructionVisitor;

    private final CodeAttributeEditor codeAttributeEditor = new CodeAttributeEditor();

    private String descriptor;
    private int    descriptorLengthDelta;


    /**
     * Creates a new DuplicateInitializerInvocationFixer.
     */
    public DuplicateInitializerInvocationFixer()
    {
        this(null);
    }


    /**
     * Creates a new DuplicateInitializerInvocationFixer.
     * @param extraAddedInstructionVisitor an optional extra visitor for all
     *                                     added instructions.
     */
    public DuplicateInitializerInvocationFixer(InstructionVisitor extraAddedInstructionVisitor)
    {
        this.extraAddedInstructionVisitor = extraAddedInstructionVisitor;
    }


   // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {

        // Reset the code changes.
        codeAttributeEditor.reset(codeAttribute.u4codeLength);

        // Fix any duplicate constructor invocations.
        codeAttribute.instructionsAccept(clazz,
                                         method,
                                         this);

        // Apply all accumulated changes to the code.
        codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        if (constantInstruction.opcode == InstructionConstants.OP_INVOKESPECIAL)
        {
            descriptorLengthDelta = 0;
            clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);

            if (descriptorLengthDelta > 0)
            {
                Instruction extraInstruction =
                    new SimpleInstruction(descriptorLengthDelta == 1 ?
                                              InstructionConstants.OP_ICONST_0 :
                                              InstructionConstants.OP_ACONST_NULL);

                codeAttributeEditor.insertBeforeInstruction(offset,
                                                            extraInstruction);

                if (DEBUG)
                {
                    System.out.println("  ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"] Inserting "+extraInstruction.toString()+" before "+constantInstruction.toString(offset));
                }

                if (extraAddedInstructionVisitor != null)
                {
                    extraInstruction.accept(null, null, null, offset, extraAddedInstructionVisitor);
                }
            }
        }
    }


    // Implementations for ConstantVisitor.

    public void visitAnyMethodrefConstant(Clazz clazz, RefConstant refConstant)
    {
        // Check the referenced constructor descriptor.
        if (refConstant.getName(clazz).equals(ClassConstants.METHOD_NAME_INIT))
        {
            descriptor = refConstant.getType(clazz);

            refConstant.referencedMemberAccept(this);
        }
    }


    // Implementations for MemberVisitor.

    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod) {}


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        descriptorLengthDelta =
            programMethod.getDescriptor(programClass).length() - descriptor.length();

        if (DEBUG)
        {
            if (descriptorLengthDelta > 0)
            {
                System.out.println("DuplicateInitializerInvocationFixer:");
                System.out.println("  ["+programClass.getName()+"."+programMethod.getName(programClass)+programMethod.getDescriptor(programClass)+"] ("+ClassUtil.externalClassAccessFlags(programMethod.getAccessFlags())+") referenced by:");
            }
        }
    }
}

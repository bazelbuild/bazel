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
package proguard.optimize.evaluation;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.editor.*;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;
import proguard.optimize.info.SimpleEnumMarker;
import proguard.optimize.peephole.*;

/**
 * This ClassVisitor simplifies the classes that it visits to simple enums.
 *
 * @see SimpleEnumMarker
 * @see MemberReferenceFixer
 * @author Eric Lafortune
 */
public class SimpleEnumClassSimplifier
extends      SimplifiedVisitor
implements   ClassVisitor,
             AttributeVisitor,
             InstructionVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("enum") != null;
    //*/


    private static final int ENUM_CLASS_NAME          = InstructionSequenceReplacer.A;
    private static final int ENUM_TYPE_NAME           = InstructionSequenceReplacer.B;
    private static final int ENUM_CONSTANT_NAME       = InstructionSequenceReplacer.X;
    private static final int ENUM_CONSTANT_ORDINAL    = InstructionSequenceReplacer.Y;
    private static final int ENUM_CONSTANT_FIELD_NAME = InstructionSequenceReplacer.Z;

    private static final int STRING_ENUM_CONSTANT_NAME   = 0;

    private static final int METHOD_ENUM_INIT            = 1;
    private static final int FIELD_ENUM_CONSTANT         = 2;

    private static final int CLASS_ENUM                  = 3;

    private static final int NAME_AND_TYPE_ENUM_INIT     = 4;
    private static final int NAME_AND_TYPE_ENUM_CONSTANT = 5;

    private static final int UTF8_INIT                   = 6;
    private static final int UTF8_STRING_I               = 7;


    private static final Constant[] CONSTANTS = new Constant[]
    {
        new StringConstant(ENUM_CONSTANT_NAME, null, null),

        new MethodrefConstant(CLASS_ENUM, NAME_AND_TYPE_ENUM_INIT,     null, null),
        new FieldrefConstant( CLASS_ENUM, NAME_AND_TYPE_ENUM_CONSTANT, null, null),

        new ClassConstant(ENUM_CLASS_NAME,  null),

        new NameAndTypeConstant(UTF8_INIT, UTF8_STRING_I),
        new NameAndTypeConstant(ENUM_CONSTANT_FIELD_NAME, ENUM_TYPE_NAME),

        new Utf8Constant(ClassConstants.METHOD_NAME_INIT),
        new Utf8Constant(ClassConstants.METHOD_TYPE_INIT_ENUM),
    };

    private static final Instruction[][][] INSTRUCTION_SEQUENCES = new Instruction[][][]
    {
        {
            // Replace new Enum("name", constant)
            // by      constant + 1.
            {
                new ConstantInstruction(InstructionConstants.OP_NEW, CLASS_ENUM),
                new SimpleInstruction(InstructionConstants.OP_DUP),
                new ConstantInstruction(InstructionConstants.OP_LDC, STRING_ENUM_CONSTANT_NAME),
                new SimpleInstruction(InstructionConstants.OP_ICONST_0, ENUM_CONSTANT_ORDINAL),
                new ConstantInstruction(InstructionConstants.OP_INVOKESPECIAL, METHOD_ENUM_INIT),
            },
            {
                new SimpleInstruction(InstructionConstants.OP_SIPUSH, ENUM_CONSTANT_ORDINAL),
                new SimpleInstruction(InstructionConstants.OP_ICONST_1),
                new SimpleInstruction(InstructionConstants.OP_IADD),
            }
        },
        {
            // The name constants may have been encrypted.
            // Replace <init>(..., constant)
            // by      <init>(..., 0); pop; constant + 1.
            {
                new SimpleInstruction(InstructionConstants.OP_ICONST_0, ENUM_CONSTANT_ORDINAL),
                new ConstantInstruction(InstructionConstants.OP_INVOKESPECIAL, METHOD_ENUM_INIT),
            },
            {
                new SimpleInstruction(InstructionConstants.OP_ICONST_0),
                new ConstantInstruction(InstructionConstants.OP_INVOKESPECIAL, METHOD_ENUM_INIT),
                new SimpleInstruction(InstructionConstants.OP_POP),
                new SimpleInstruction(InstructionConstants.OP_SIPUSH, ENUM_CONSTANT_ORDINAL),
                new SimpleInstruction(InstructionConstants.OP_ICONST_1),
                new SimpleInstruction(InstructionConstants.OP_IADD),
            }
        },
    };

    private final CodeAttributeEditor codeAttributeEditor =
        new CodeAttributeEditor(true, true);

    private final InstructionSequencesReplacer instructionSequenceReplacer =
        new InstructionSequencesReplacer(CONSTANTS,
                                        INSTRUCTION_SEQUENCES,
                                        null,
                                        codeAttributeEditor);

    private final MemberVisitor initializerSimplifier = new AllAttributeVisitor(this);


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        if (DEBUG)
        {
            System.out.println("SimpleEnumClassSimplifier: ["+programClass.getName()+"]");
        }

        // Unmark the class as an enum.
        programClass.u2accessFlags &= ~ClassConstants.ACC_ENUM;

        // Remove the valueOf method, if present.
        Method valueOfMethod =
            programClass.findMethod(ClassConstants.METHOD_NAME_VALUEOF, null);
        if (valueOfMethod != null)
        {
            new ClassEditor(programClass).removeMethod(valueOfMethod);
        }

        // Simplify the static initializer.
        programClass.methodAccept(ClassConstants.METHOD_NAME_CLINIT,
                                  ClassConstants.METHOD_TYPE_CLINIT,
                                  initializerSimplifier);
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Set up the code attribute editor.
        codeAttributeEditor.reset(codeAttribute.u4codeLength);

        // Find the peephole changes.
        codeAttribute.instructionsAccept(clazz, method, instructionSequenceReplacer);

        // Apply the peephole changes.
        codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
    }
}

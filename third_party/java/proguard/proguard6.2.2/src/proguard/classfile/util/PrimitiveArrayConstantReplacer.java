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
package proguard.classfile.util;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.*;
import proguard.classfile.editor.*;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.visitor.ClassVisitor;

/**
 * This ClassVisitor replaces all PrimitiveArray constants by Java bytecode
 * compliant array store instructions.
 *
 * @see ArrayInitializationReplacer
 * @author Thomas Neidhart
 */
public class PrimitiveArrayConstantReplacer
extends      SimplifiedVisitor
implements   ClassVisitor,
             AttributeVisitor,
             InstructionVisitor,
             ConstantVisitor,
             PrimitiveArrayConstantElementVisitor
{
    private final CodeAttributeEditor  codeAttributeEditor  = new CodeAttributeEditor();
    private final ConstantPoolShrinker constantPoolShrinker = new ConstantPoolShrinker();

    // Fields acting as parameters and return values.

    private boolean                        classModified;
    private InstructionSequenceBuilder builder;


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        ConstantCounter counter = new ConstantCounter();
        programClass.constantPoolEntriesAccept(
            new ConstantTagFilter(ClassConstants.CONSTANT_PrimitiveArray,
            counter));

        // Replace PrimitiveArray constants if the class has any.
        if (counter.getCount() > 0)
        {
            classModified = false;

            programClass.methodsAccept(new AllAttributeVisitor(this));

            if (classModified)
            {
                // Remove the now unused PrimitiveArray constants.
                programClass.accept(constantPoolShrinker);
            }
        }
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        codeAttributeEditor.reset(codeAttribute.u4codeLength);

        codeAttribute.instructionsAccept(clazz, method, this);

        if (codeAttributeEditor.isModified())
        {
            codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);

            classModified = true;
        }
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        builder = null;

        clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);

        if (builder != null)
        {
            codeAttributeEditor.replaceInstruction(offset, builder.instructions());

            classModified = true;
        }
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitPrimitiveArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant)
    {
        char primitiveType = primitiveArrayConstant.getPrimitiveType();
        int  arrayLength   = primitiveArrayConstant.getLength();

        // Start composing a new array initialization sequence.
        builder = new InstructionSequenceBuilder((ProgramClass) clazz);

        // Push the primitive array length.
        builder.pushInt(primitiveArrayConstant.getLength());

        // Create the primitive array.
        builder.newarray(InstructionUtil.arrayTypeFromInternalType(primitiveType));

        // Fill out the primitive array elements.
        primitiveArrayConstant.primitiveArrayElementsAccept(clazz, this);
    }


    // Implementations for PrimitiveArrayConstantElementVisitor.

    public void visitBooleanArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, boolean value)
    {
        builder.dup()
               .pushInt(index)
               .iconst(value ? 1 : 0)
               .bastore();
    }


    public void visitByteArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, byte value)
    {
        builder.dup()
               .pushInt(index)
               .pushInt(value)
               .bastore();
    }


    public void visitCharArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, char value)
    {
        builder.dup()
               .pushInt(index)
               .pushInt(value)
               .castore();
    }


    public void visitShortArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, short value)
    {
        builder.dup()
               .pushInt(index)
               .pushInt(value)
               .sastore();
    }


    public void visitIntArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, int value)
    {
        builder.dup()
               .pushInt(index)
               .pushInt(value)
               .iastore();
    }


    public void visitFloatArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, float value)
    {
        builder.dup()
               .pushInt(index)
               .pushFloat(value)
               .fastore();
    }


    public void visitLongArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, long value)
    {
        builder.dup()
               .pushInt(index)
               .pushLong(value)
               .lastore();
    }


    public void visitDoubleArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, double value)
    {
        builder.dup()
               .pushInt(index)
               .pushDouble(value)
               .dastore();
    }
}

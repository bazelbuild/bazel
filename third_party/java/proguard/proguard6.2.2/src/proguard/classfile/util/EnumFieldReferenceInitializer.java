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
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.attribute.annotation.*;
import proguard.classfile.attribute.annotation.visitor.ElementValueVisitor;
import proguard.classfile.attribute.visitor.AllAttributeVisitor;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.*;
import proguard.classfile.visitor.MemberVisitor;

/**
 * This ElementValueVisitor initializes the field references of the
 * EnumConstantElementValue instances that it visits.
 *
 * @author Eric Lafortune
 */
public class EnumFieldReferenceInitializer
extends      SimplifiedVisitor
implements   ElementValueVisitor,
             InstructionVisitor,
             ConstantVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = true;
    //*/

    private MemberVisitor enumFieldFinder = new AllAttributeVisitor(
                                            new AllInstructionVisitor(this));

    // Fields acting as parameters and return values for the visitors.
    private String  enumTypeName;
    private String  enumConstantName;
    private boolean enumConstantNameFound;
    private Clazz   referencedEnumClass;
    private Field   referencedEnumField;


    // Implementations for ElementValueVisitor.

    public void visitAnyElementValue(Clazz clazz, Annotation annotation, ElementValue elementValue) {}


    public void visitEnumConstantElementValue(Clazz clazz, Annotation annotation, EnumConstantElementValue enumConstantElementValue)
    {

        if (enumConstantElementValue.referencedClasses != null    &&
            enumConstantElementValue.referencedClasses.length > 0)
        {
            referencedEnumClass = enumConstantElementValue.referencedClasses[0];
            if (referencedEnumClass != null)
            {
                // Try to find the enum field through the static enum
                // initialization code (at least for program classes).
                enumTypeName        = enumConstantElementValue.getTypeName(clazz);
                enumConstantName    = enumConstantElementValue.getConstantName(clazz);
                referencedEnumField = null;
                referencedEnumClass.methodAccept(ClassConstants.METHOD_NAME_CLINIT,
                                                 ClassConstants.METHOD_TYPE_CLINIT,
                                                 enumFieldFinder);

                // Otherwise try to find the enum field through its name.
                // The constant name could be different from the field name, if
                // the latter is already obfuscated.
                if (referencedEnumField == null)
                {
                    referencedEnumField =
                        referencedEnumClass.findField(enumConstantName,
                                                      enumTypeName);
                }

                if (DEBUG)
                {
                    System.out.println("EnumFieldReferenceInitializer: ["+referencedEnumClass.getName()+"."+enumConstantName+"] -> "+referencedEnumField);
                }

                enumConstantElementValue.referencedField = referencedEnumField;
            }
        }
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        switch (constantInstruction.opcode)
        {
            case InstructionConstants.OP_LDC:
            case InstructionConstants.OP_LDC_W:
            case InstructionConstants.OP_PUTSTATIC:
                clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                break;
        }
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        enumConstantNameFound =
            enumConstantName.equals(stringConstant.getString(clazz));
    }


    public void visitFieldrefConstant(Clazz clazz, FieldrefConstant fieldrefConstant)
    {
        if (enumConstantNameFound)
        {
            if (enumTypeName.equals(fieldrefConstant.getType(clazz)))
            {
                referencedEnumField = (Field)fieldrefConstant.referencedMember;
            }

            enumConstantNameFound = false;
        }
    }
}
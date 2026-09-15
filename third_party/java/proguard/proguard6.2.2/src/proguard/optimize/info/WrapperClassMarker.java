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
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.constant.*;
import proguard.classfile.instruction.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.evaluation.value.*;
import proguard.optimize.evaluation.StoringInvocationUnit;

/**
 * This ClassVisitor marks all program classes that are a simple wrapper for a
 * single non-null instance of another class.
 *
 * A wrapper class has
 * - exactly one non-static field, which references an object,
 * - exactly one initializer, with a single parameter that is never null,
 *   that initializes the field,
 * - no subclasses.
 *
 * @see StoringInvocationUnit
 * @author Eric Lafortune
 */
public class WrapperClassMarker
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor,
             AttributeVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    public  static       boolean DEBUG = System.getProperty("wcm")  != null;
    //*/


    private final Constant[] INITIALIZER_CONSTANTS = new Constant[]
    {
        new FieldrefConstant(InstructionSequenceMatcher.A,
                             InstructionSequenceMatcher.B, null, null),
    };

    // Instruction pattern:
    //   this.x = arg0;
    //   super.<init>;
    //   return;
    private final Instruction[] INITIALIZER_INSTRUCTIONS = new Instruction[]
    {
        new VariableInstruction(InstructionConstants.OP_ALOAD_0, 0),
        new VariableInstruction(InstructionConstants.OP_ALOAD_1, 1),
        new ConstantInstruction(InstructionConstants.OP_PUTFIELD, 0),
        new VariableInstruction(InstructionConstants.OP_ALOAD_0, 0),
        new ConstantInstruction(InstructionConstants.OP_INVOKESPECIAL, InstructionSequenceMatcher.X),
        new SimpleInstruction(InstructionConstants.OP_RETURN),
    };

    private final InstructionSequenceMatcher INITIALIZER_MATCHER = new InstructionSequenceMatcher(INITIALIZER_CONSTANTS, INITIALIZER_INSTRUCTIONS);

    // Fields acting as parameters and return values for the visitor methods.
    private Clazz wrappedClass;
    private int   wrapCounter;


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        if (programClass.subClasses == null ||
            programClass.subClasses.length == 0)
        {
            wrappedClass = null;

            // Can we find one non-static field with a class type?
            wrapCounter = 0;
            programClass.fieldsAccept(this);
            if (wrapCounter == 1)
            {
                // Can we find exactly one initializer that initializes this
                // field?
                wrapCounter = 0;
                programClass.methodsAccept(this);
                if (wrapCounter == 1)
                {
                    setWrappedClass(programClass, wrappedClass);
                }
            }
        }
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        // Is the field non-static and of a class type?
        if ((programField.getAccessFlags() & ClassConstants.ACC_STATIC) == 0 &&
            ClassUtil.isInternalClassType(programField.getDescriptor(programClass)) &&
            !ClassUtil.isInternalArrayType(programField.getDescriptor(programClass)))
        {
            wrappedClass = programField.referencedClass;
            if (wrappedClass != null)
            {
                wrapCounter++;
            }
            else
            {
                wrapCounter = Integer.MIN_VALUE;
            }
        }
        else
        {
            wrapCounter = Integer.MIN_VALUE;
        }
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        // Is the method an initializer?
        if (ClassUtil.isInitializer(programMethod.getName(programClass)))
        {
            // Does it have exactly one parameter?
            if (ClassUtil.internalMethodParameterCount(programMethod.getDescriptor(programClass)) == 1)
            {
                // Is the parameter a non-null reference?
                Value value =
                    StoringInvocationUnit.getMethodParameterValue(programMethod, 1);

                if (value != null                                     &&
                    value.computationalType() == Value.TYPE_REFERENCE &&
                    value.referenceValue().isNotNull() == Value.ALWAYS)
                {
                    // Does the method initialize the field?
                    programMethod.attributesAccept(programClass, this);
                }
                else
                {
                    wrapCounter = Integer.MIN_VALUE;
                }
            }
            else
            {
                wrapCounter = Integer.MIN_VALUE;
            }
        }
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute)  {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Is the initializer initializing the field?
        if (codeAttribute.u4codeLength == 10)
        {
            INITIALIZER_MATCHER.reset();
            codeAttribute.instructionsAccept(clazz, method, INITIALIZER_MATCHER);
            if (INITIALIZER_MATCHER.isMatching())
            {
                String initializerClassName = clazz.getName();
                String fieldClassName       = clazz.getClassName(INITIALIZER_MATCHER.matchedConstantIndex(InstructionSequenceMatcher.A));
                if (fieldClassName.equals(initializerClassName))
                {
                    wrapCounter++;
                }
                else
                {
                    wrapCounter = Integer.MIN_VALUE;
                }
            }
            else
            {
                wrapCounter = Integer.MIN_VALUE;
            }
        }
        else
        {
            wrapCounter = Integer.MIN_VALUE;
        }
    }


    // Small utility methods.

    private static void setWrappedClass(Clazz clazz, Clazz wrappedClass)
    {
        if (DEBUG)
        {
            System.out.println("WrapperClassMarker: ["+clazz.getName()+"] wraps ["+wrappedClass.getName()+"]");
        }

        ProgramClassOptimizationInfo.getProgramClassOptimizationInfo(clazz).setWrappedClass(wrappedClass);
    }


    public static Clazz getWrappedClass(Clazz clazz)
    {
        return ClassOptimizationInfo.getClassOptimizationInfo(clazz).getWrappedClass();
    }
}

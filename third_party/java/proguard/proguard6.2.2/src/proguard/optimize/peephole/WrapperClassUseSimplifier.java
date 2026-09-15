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

package proguard.optimize.peephole;

import proguard.classfile.*;
import proguard.classfile.attribute.Attribute;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.constant.ClassConstant;
import proguard.classfile.constant.Constant;
import proguard.classfile.constant.FieldrefConstant;
import proguard.classfile.constant.MethodrefConstant;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.editor.CodeAttributeEditor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.ClassVisitor;
import proguard.optimize.info.WrapperClassMarker;

/**
 * This AttributeVisitor simplifies the use of retargeted wrapper classes in
 * the code attributes that it visits. More specifically, it replaces
 *     "new Wrapper(targetClass)" by "targetClass", and
 *     "wrapper.field" by "wrapper".
 * You still need to retarget all class references, replacing references to
 * the wrapper class by references to the target class.
 *
 * @see WrapperClassMarker
 * @see ClassMerger
 * @see RetargetedClassFilter
 * @see TargetClassChanger
 * @author Eric Lafortune
 */
public class WrapperClassUseSimplifier
extends SimplifiedVisitor
implements AttributeVisitor,
        InstructionVisitor,
        ConstantVisitor,
        ClassVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("wc") != null;
    //*/


    private final InstructionVisitor extraInstructionVisitor;

    private final CodeAttributeEditor codeAttributeEditor = new CodeAttributeEditor(true, true);

    // Fields acting as parameters and return values for the visitor methods.
    private Clazz wrappedClass;
    private Instruction popInstruction;


    /**
     * Creates a new WrapperClassUseSimplifier.
     */
    public WrapperClassUseSimplifier()
    {
        this(null);
    }


    /**
     * Creates a new WrapperClassUseSimplifier.
     * @param extraInstructionVisitor an optional extra visitor for all
     *                                simplified instructions.
     */
    public WrapperClassUseSimplifier(InstructionVisitor extraInstructionVisitor)
    {
        this.extraInstructionVisitor = extraInstructionVisitor;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        if (DEBUG)
        {
            System.out.println("WrapperClassUseSimplifier: "+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz));
        }

        int codeLength = codeAttribute.u4codeLength;

        // Reset the code changes.
        codeAttributeEditor.reset(codeLength);

        // Edit the instructions.
        codeAttribute.instructionsAccept(clazz, method, this);

        // Apply all accumulated changes to the code.
        codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        switch (constantInstruction.opcode)
        {
            case InstructionConstants.OP_NEW:
            {
                // Is it instantiating a wrapper class?
                if (isReferencingWrapperClass(clazz, constantInstruction.constantIndex))
                {
                    // Is the next instruction a typical dup instruction?
                    int nextOffset = offset + constantInstruction.length(offset);
                    popInstruction = InstructionFactory.create(codeAttribute.code, nextOffset);
                    switch (popInstruction.canonicalOpcode())
                    {
                        case InstructionConstants.OP_DUP:
                        {
                            // Delete the new and dup instructions:
                            //     new Wrapper
                            //     dup
                            codeAttributeEditor.deleteInstruction(offset);
                            codeAttributeEditor.deleteInstruction(nextOffset);
                            popInstruction = null;
                            break;
                        }
                        case InstructionConstants.OP_ASTORE:
                        {
                            // Replace the new instance by a dummy null
                            // and remember to store the target instance:
                            //     new Wrapper -> aconst_null
                            //     astore x    -> remember
                            //     aload x
                            codeAttributeEditor.replaceInstruction(offset, new SimpleInstruction(InstructionConstants.OP_ACONST_NULL));
                            break;
                        }
                        default:
                        {
                            // Replace the new instance by a dummy null
                            // and remember to pop the target instance:
                            //     new Wrapper -> aconst_null
                            codeAttributeEditor.replaceInstruction(offset, new SimpleInstruction(InstructionConstants.OP_ACONST_NULL));
                            popInstruction = new SimpleInstruction(InstructionConstants.OP_POP);
                        }
                    }

                    if (extraInstructionVisitor != null)
                    {
                        extraInstructionVisitor.visitConstantInstruction(clazz, method, codeAttribute, offset, constantInstruction);
                    }
                }
                break;
            }
            case InstructionConstants.OP_INVOKESPECIAL:
            {
                // Is it initializing a wrapper class?
                if (isReferencingWrapperClass(clazz, constantInstruction.constantIndex))
                {
                    // Do we have a special pop instruction?
                    // TODO: May still fail with nested initializers.
                    if (popInstruction == null)
                    {
                        // Delete the initializer invocation (with the
                        // wrapper instance no longer on the stack):
                        //     Wrapper.<init>(target) -> target
                        codeAttributeEditor.deleteInstruction(offset);
                    }
                    else
                    {
                        // Delete the initializer invocation, and store
                        // the target instance again:
                        //     invokespecial Wrapper.<init>(target) -> astore x / pop
                        codeAttributeEditor.replaceInstruction(offset, new Instruction[]
                        {
                            popInstruction,
                            new SimpleInstruction(InstructionConstants.OP_POP),
                        });
                    }
                }
                break;
            }
            case InstructionConstants.OP_GETFIELD:
            {
                // Is it retrieving the field of the wrapper class?
                if (isReferencingWrapperClass(clazz, constantInstruction.constantIndex))
                {
                    // Delete the retrieval:
                    //     wrapper.field -> wrapper.
                    codeAttributeEditor.deleteInstruction(offset);
                }
                break;
            }
        }
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitFieldrefConstant(Clazz clazz, FieldrefConstant fieldrefConstant)
    {
        // Is the constant retrieving from a wrapper class?
        fieldrefConstant.referencedClassAccept(this);
    }


    public void visitMethodrefConstant(Clazz clazz, MethodrefConstant methodrefConstant)
    {
        if (methodrefConstant.getName(clazz).equals(ClassConstants.METHOD_NAME_INIT))
        {
            // Is the constant referring to a wrapper class?
            methodrefConstant.referencedClassAccept(this);
        }
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        // Is the constant referring to a wrapper class?
        classConstant.referencedClassAccept(this);
    }


    // Implementations for ClassVisitor.

    public void visitLibraryClass(LibraryClass libraryClass) {}


    public void visitProgramClass(ProgramClass programClass)
    {
        wrappedClass = ClassMerger.getTargetClass(programClass);
    }


    // Small utility methods.

    /**
     * Returns whether the constant at the given offset is referencing a
     * wrapper class (different from the given class) that is being retargeted.
     */
    private boolean isReferencingWrapperClass(Clazz clazz, int constantIndex)
    {
        wrappedClass = null;

        clazz.constantPoolEntryAccept(constantIndex, this);

        return wrappedClass != null;
    }
}

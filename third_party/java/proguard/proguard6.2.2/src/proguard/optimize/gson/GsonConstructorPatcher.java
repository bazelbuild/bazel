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
package proguard.optimize.gson;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.editor.*;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.MemberVisitor;
import proguard.evaluation.*;
import proguard.evaluation.value.*;
import proguard.optimize.evaluation.PartialEvaluator;

/**
 * Class visitor that patches the constructor of Gson so that the injected
 * optimized type adapter factory is registered at the right priority. It
 * also exposes the Excluder used by Gson to the outside if needed.
 *
 * @author Lars Vandenbergh
 */
public class GsonConstructorPatcher
extends      SimplifiedVisitor
implements   MemberVisitor,
             AttributeVisitor,
             InstructionVisitor
{
    private static final boolean DEBUG = false;

    private final CodeAttributeEditor codeAttributeEditor;
    private final TypedReferenceValueFactory valueFactory         =
        new TypedReferenceValueFactory();
    private final PartialEvaluator           partialEvaluator     =
        new PartialEvaluator(valueFactory,
                             new BasicInvocationUnit(new TypedReferenceValueFactory()),
                             true);
    private final AttributeVisitor           lazyPartialEvaluator =
        new AttributeNameFilter(ClassConstants.ATTR_Code,
                                new SingleTimeAttributeVisitor(
                                    partialEvaluator));
    private final static int THIS_PARAMETER       = 0;
    private final static int EXCLUDER_PARAMETER   = 1;
    private int              insertionOffset      = -1;
    private int              typeAdapterListLocal = -1;
    private boolean          addExcluder;


    /**
     * Constructs a new GsonConstructorPatcher.
     *
     * @param codeAttributeEditor the code attribute editor for editing the
     *                            code attribute of the Gson constructor.
     * @param addExcluder         determines whether or not to inject
     *                            code for exposing the Gson excluder.
     */
    public GsonConstructorPatcher(CodeAttributeEditor codeAttributeEditor,
                                  boolean             addExcluder)
    {
        this.codeAttributeEditor = codeAttributeEditor;
        this.addExcluder         = addExcluder;
    }


    // Implementations for MemberVisitor.

    @Override
    public void visitAnyMember(Clazz clazz, Member member) {}


    @Override
    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        // We make the assumption that there is one constructor with a List
        // of type adapter factories as one of its arguments. This has been
        // the case since Gson version 2.0 from 2011.
        String descriptor = programMethod.getDescriptor(programClass);
        if (descriptor.contains(ClassConstants.TYPE_JAVA_UTIL_LIST))
        {
            if(DEBUG)
            {
                System.out.println("GsonConstructorPatcher: patching " +
                                   programClass.getName() + " " +
                                   programMethod.getName(programClass) + " " +
                                   descriptor);
            }
            programMethod.attributesAccept(programClass, this);
        }
    }


    // Implementations for AttributeVisitor.

    @Override
    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    @Override
    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Search for insertion point and local that contains list of type
        // adapter factories.
        codeAttribute.instructionsAccept(clazz, method, this);

        if (insertionOffset != -1 && typeAdapterListLocal != -1)
        {
            // Set up the code attribute editor.
            codeAttributeEditor.reset(codeAttribute.u4codeLength);

            // Insert instructions for appending type adapter factory to the list.
            InstructionSequenceBuilder ____ = new InstructionSequenceBuilder((ProgramClass)clazz);
            ____.new_(ClassConstants.NAME_JAVA_UTIL_ARRAY_LIST)
                .dup()
                .aload(typeAdapterListLocal)
                .invokespecial(ClassConstants.NAME_JAVA_UTIL_ARRAY_LIST,
                               ClassConstants.METHOD_NAME_INIT,
                               ClassConstants.METHOD_TYPE_INIT_COLLECTION)
                .astore(typeAdapterListLocal)
                .aload(typeAdapterListLocal)
                .new_(OptimizedClassConstants.NAME_OPTIMIZED_TYPE_ADAPTER_FACTORY)
                .dup()
                .invokespecial(OptimizedClassConstants.NAME_OPTIMIZED_TYPE_ADAPTER_FACTORY,
                               ClassConstants.METHOD_NAME_INIT,
                               ClassConstants.METHOD_TYPE_INIT)
                .invokeinterface(ClassConstants.NAME_JAVA_UTIL_LIST,
                                 ClassConstants.METHOD_NAME_ADD,
                                 ClassConstants.METHOD_TYPE_ADD)
                .pop();

            // Insert instructions for assigning excluder to the artificial excluder field.
            if (addExcluder)
            {
                ____.aload(THIS_PARAMETER)
                    .aload(EXCLUDER_PARAMETER)
                    .putfield(GsonClassConstants.NAME_GSON,
                              OptimizedClassConstants.FIELD_NAME_EXCLUDER,
                              OptimizedClassConstants.FIELD_TYPE_EXCLUDER);
            }
            codeAttributeEditor.insertAfterInstruction(insertionOffset, ____.instructions());

            // Apply the insertion.
            codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
        }
    }


    // Implementations for InstructionVisitor

    @Override
    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
    {
        if (instruction.actualOpcode() == InstructionConstants.OP_INVOKEINTERFACE &&
            typeAdapterListLocal == -1)
        {
            ConstantInstruction constantInstruction = (ConstantInstruction)instruction;
            Constant constant = ((ProgramClass)clazz).constantPool[constantInstruction.constantIndex];
            if (constant instanceof InterfaceMethodrefConstant)
            {
                InterfaceMethodrefConstant interfaceMethodrefConstant = (InterfaceMethodrefConstant)constant;
                if (interfaceMethodrefConstant.getClassName(clazz).equals(ClassConstants.NAME_JAVA_UTIL_LIST) &&
                    interfaceMethodrefConstant.getName(clazz).equals(ClassConstants.METHOD_NAME_ADD_ALL)      &&
                    interfaceMethodrefConstant.getType(clazz).equals(ClassConstants.METHOD_TYPE_ADD_ALL))
                {
                    // We found an invocation to List.add(Object).
                    // Find out which instructions contributed to the top value
                    // on the stack and visit them to determine which local is
                    // passed as argument.
                    lazyPartialEvaluator.visitCodeAttribute(clazz,
                                                            method,
                                                            codeAttribute);
                    TracedStack stackBefore = partialEvaluator.getStackBefore(offset);
                    InstructionOffsetValue instructionOffsetValue = stackBefore.getTopProducerValue(0).instructionOffsetValue();
                    for (int instructionIndex = 0; instructionIndex < instructionOffsetValue.instructionOffsetCount(); instructionIndex++)
                    {
                        int instructionOffset = instructionOffsetValue.instructionOffset(instructionIndex);
                        codeAttribute.instructionAccept(clazz, method, instructionOffset, new LocalFinder());
                    }
                }
            }
        }
        else if (instruction.actualOpcode() == InstructionConstants.OP_INVOKESPECIAL &&
                 insertionOffset == -1)
        {
            ConstantInstruction constantInstruction = (ConstantInstruction)instruction;
            Constant constant = ((ProgramClass)clazz).constantPool[constantInstruction.constantIndex];
            if (constant instanceof MethodrefConstant)
            {
                MethodrefConstant methodrefConstant = (MethodrefConstant)constant;
                if (methodrefConstant.getClassName(clazz).equals(ClassConstants.NAME_JAVA_LANG_OBJECT) &&
                    methodrefConstant.getName(clazz).equals(ClassConstants.METHOD_NAME_INIT)           &&
                    methodrefConstant.getType(clazz).equals(ClassConstants.METHOD_TYPE_INIT))
                {
                    // We want to insert our patch after the call to Object.<init>.
                    insertionOffset = offset;
                }
            }
        }
    }


    private class LocalFinder
    extends       SimplifiedVisitor
    implements    InstructionVisitor
    {
        // Implementations for InstructionVisitor

        @Override
        public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
        {
            if (instruction.canonicalOpcode() == InstructionConstants.OP_ALOAD)
            {
                VariableInstruction variableInstruction = (VariableInstruction)instruction;
                typeAdapterListLocal = variableInstruction.variableIndex;
            }
        }
    }
}

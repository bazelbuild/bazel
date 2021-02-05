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
package proguard.backport;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.editor.*;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;

import java.util.*;

/**
 * This InstructionVisitor converts all indy String Concatenations in the visited
 * classes to StringBuilder-append chains.
 *
 * @author Tim Van Den Broecke
 */
public class StringConcatenationConverter
extends    SimplifiedVisitor
implements InstructionVisitor,

           // Implementation interfaces.
           AttributeVisitor,
           BootstrapMethodInfoVisitor,
           ConstantVisitor
{
    // Constants as per specification
    private static final char C_VARIABLE_ARGUMENT = '\u0001';
    private static final char C_CONSTANT_ARGUMENT = '\u0002';

    private final InstructionVisitor  extraInstructionVisitor;
    private final CodeAttributeEditor codeAttributeEditor;

    private InstructionSequenceBuilder appendChainComposer;
    private int                        estimatedStringLength;
    private int                        referencedBootstrapMethodIndex;
    private String                     concatenationRecipe;
    private int[]                      concatenationConstants;

    public StringConcatenationConverter(InstructionVisitor  extraInstructionVisitor,
                                        CodeAttributeEditor codeAttributeEditor)
    {
        this.extraInstructionVisitor = extraInstructionVisitor;
        this.codeAttributeEditor     = codeAttributeEditor;
    }


    // Implementations for InstructionVisitor.

    @Override
    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}

    @Override
    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        if (constantInstruction.opcode == InstructionConstants.OP_INVOKEDYNAMIC)
        {
            ProgramClass programClass = (ProgramClass) clazz;

            InvokeDynamicConstant invokeDynamicConstant =
                (InvokeDynamicConstant) programClass.getConstant(constantInstruction.constantIndex);

            // Remember the referenced bootstrap method index and extract the recipe from it.
            referencedBootstrapMethodIndex = invokeDynamicConstant.getBootstrapMethodAttributeIndex();
            concatenationRecipe            = null;
            concatenationConstants         = null;

            programClass.attributesAccept(this);

            if (concatenationRecipe != null)
            {
                //if (isMakeConcatWithConstants(invokeDynamicConstant.getName(programClass)))
                String descriptor = invokeDynamicConstant.getType(programClass);

                InstructionSequenceBuilder mainReplacementComposer = new InstructionSequenceBuilder(programClass);
                appendChainComposer                                = new InstructionSequenceBuilder(programClass);
                estimatedStringLength                              = 0;

                // Collect the argument types.
                InternalTypeEnumeration typeEnumeration = new InternalTypeEnumeration(descriptor);
                List<String>            types           = new ArrayList<String>();
                while (typeEnumeration.hasMoreTypes())
                {
                    types.add(typeEnumeration.nextType());
                }

                // Store the correct number of stack values in reverse
                // order in local variables
                int                  variableIndex = codeAttribute.u2maxLocals;
                ListIterator<String> typeIterator  = types.listIterator(types.size());
                while (typeIterator.hasPrevious())
                {
                    String type = typeIterator.previous();

                    mainReplacementComposer.store(variableIndex, type);
                    variableIndex += ClassUtil.internalTypeSize(type);
                }

                // Loop over the recipe.
                // Push the local variables one by one, insert
                // constants where necessary and create and append
                // instruction chain.
                typeIterator = types.listIterator();
                for (int argIndex = 0, constantCounter = 0; argIndex < concatenationRecipe.length(); argIndex++)
                {
                    switch (concatenationRecipe.charAt(argIndex))
                    {
                        case C_VARIABLE_ARGUMENT:
                            String type = typeIterator.next();
                            estimatedStringLength += typicalStringLengthFromType(type);

                            int variableSize = ClassUtil.internalTypeSize(type);
                            variableIndex -= variableSize;

                            appendChainComposer.load(variableIndex, type)
                                               .invokevirtual(ClassConstants.NAME_JAVA_LANG_STRING_BUILDER,
                                                              ClassConstants.METHOD_NAME_APPEND,
                                                              appendDescriptorFromInternalType(type));
                            break;

                        case C_CONSTANT_ARGUMENT:
                            int constantIndex = concatenationConstants[constantCounter++];

                            appendChainComposer.ldc_(constantIndex);
                            // Visit the constant to decide how it needs to be appended.
                            programClass.constantPoolEntryAccept(constantIndex, this);
                            break;

                        default:
                            // Find where the String stops and append it
                            int nextArgIndex = nextArgIndex(concatenationRecipe, argIndex);
                            estimatedStringLength += nextArgIndex - argIndex;
                            appendChainComposer.ldc(concatenationRecipe.substring(argIndex, nextArgIndex))
                                               .invokevirtual(ClassConstants.NAME_JAVA_LANG_STRING_BUILDER,
                                                              ClassConstants.METHOD_NAME_APPEND,
                                                              ClassConstants.METHOD_TYPE_STRING_STRING_BUILDER);

                            // Jump forward to the end of the String
                            argIndex = nextArgIndex - 1;
                            break;
                    }
                }

                // Create a StringBuilder with the estimated initial size
                mainReplacementComposer.new_(         ClassConstants.NAME_JAVA_LANG_STRING_BUILDER)
                                       .dup()
                                       .pushInt(      estimatedStringLength)
                                       .invokespecial(ClassConstants.NAME_JAVA_LANG_STRING_BUILDER,
                                                      ClassConstants.METHOD_NAME_INIT,
                                                      ClassConstants.METHOD_TYPE_INT_VOID);

                // Attach the 'append' instruction chain
                mainReplacementComposer.appendInstructions(appendChainComposer.instructions());

                // Finish with StringBuilder.toString()
                mainReplacementComposer.invokevirtual(ClassConstants.NAME_JAVA_LANG_STRING_BUILDER,
                                                      ClassConstants.METHOD_NAME_TOSTRING,
                                                      ClassConstants.METHOD_TYPE_TOSTRING);

                // Commit the code changes
                codeAttributeEditor.replaceInstruction(offset,
                                                       mainReplacementComposer.instructions());

                // Optionally let this instruction be visited some more
                if (extraInstructionVisitor != null)
                {
                    extraInstructionVisitor.visitConstantInstruction(clazz,
                                                                     method,
                                                                     codeAttribute,
                                                                     offset,
                                                                     constantInstruction);
                }
            }
        }
    }


    // Implementations for AttributeVisitor.

    @Override
    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}

    @Override
    public void visitBootstrapMethodsAttribute(Clazz                     clazz,
                                               BootstrapMethodsAttribute bootstrapMethodsAttribute)
    {
        bootstrapMethodsAttribute.bootstrapMethodEntryAccept(clazz, referencedBootstrapMethodIndex, this);
    }


    // Implementations for BootstrapMethodInfoVisitor.

    @Override
    public void visitBootstrapMethodInfo(Clazz               clazz,
                                         BootstrapMethodInfo bootstrapMethodInfo)
    {
        ProgramClass programClass = (ProgramClass) clazz;

        MethodHandleConstant bootstrapMethodHandle =
            (MethodHandleConstant) programClass.getConstant(bootstrapMethodInfo.u2methodHandleIndex);

        if (isStringConcatFactory(bootstrapMethodHandle.getClassName(clazz)))
        {
            concatenationRecipe    =
                ((StringConstant) programClass.getConstant(bootstrapMethodInfo.u2methodArguments[0])).getString(programClass);
            concatenationConstants = bootstrapMethodInfo.u2methodArgumentCount > 1 ?
                Arrays.copyOfRange(bootstrapMethodInfo.u2methodArguments, 1, bootstrapMethodInfo.u2methodArgumentCount) :
                new int[0];
        }
    }


    // Implementations for ConstantVisitor.

    @Override
    public void visitAnyConstant(Clazz clazz, Constant constant)
    {
        // append as Object by default. Override below if necessary.
        estimatedStringLength += 16;

        appendChainComposer.invokevirtual(ClassConstants.NAME_JAVA_LANG_STRING_BUILDER,
                                          ClassConstants.METHOD_NAME_APPEND,
                                          ClassConstants.METHOD_TYPE_OBJECT_STRING_BUILDER);
    }

    @Override
    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        estimatedStringLength += stringConstant.getString(clazz).length();

        appendChainComposer.invokevirtual(ClassConstants.NAME_JAVA_LANG_STRING_BUILDER,
                                          ClassConstants.METHOD_NAME_APPEND,
                                          ClassConstants.METHOD_TYPE_STRING_STRING_BUILDER);
    }


    // Small utility methods.

    private static boolean isStringConcatFactory(String className)
    {
        return ClassConstants.NAME_JAVA_LANG_INVOKE_STRING_CONCAT_FACTORY.equals(className);
    }

    private static boolean isMakeConcat(String methodName)
    {
        return ClassConstants.METHOD_NAME_MAKE_CONCAT.equals(methodName);
    }

    private static boolean isMakeConcatWithConstants(String methodName)
    {
        return ClassConstants.METHOD_NAME_MAKE_CONCAT_WITH_CONSTANTS.equals(methodName);
    }

    private static int typicalStringLengthFromType(String internalTypeName)
    {
        return internalTypeName.equals(String.valueOf(ClassConstants.TYPE_BOOLEAN)) ? ClassConstants.MAXIMUM_BOOLEAN_AS_STRING_LENGTH :
               internalTypeName.equals(String.valueOf(ClassConstants.TYPE_CHAR))    ? ClassConstants.MAXIMUM_CHAR_AS_STRING_LENGTH    :
               internalTypeName.equals(String.valueOf(ClassConstants.TYPE_BYTE))    ? ClassConstants.MAXIMUM_BYTE_AS_STRING_LENGTH    :
               internalTypeName.equals(String.valueOf(ClassConstants.TYPE_SHORT))   ? ClassConstants.MAXIMUM_SHORT_AS_STRING_LENGTH   :
               internalTypeName.equals(String.valueOf(ClassConstants.TYPE_INT))     ? ClassConstants.MAXIMUM_INT_AS_STRING_LENGTH     :
               internalTypeName.equals(String.valueOf(ClassConstants.TYPE_LONG))    ? ClassConstants.MAXIMUM_LONG_AS_STRING_LENGTH    :
               internalTypeName.equals(String.valueOf(ClassConstants.TYPE_FLOAT))   ? ClassConstants.MAXIMUM_FLOAT_AS_STRING_LENGTH   :
               internalTypeName.equals(String.valueOf(ClassConstants.TYPE_DOUBLE))  ? ClassConstants.MAXIMUM_DOUBLE_AS_STRING_LENGTH  :
                                                                                      ClassConstants.DEFAULT_STRINGBUILDER_INIT_SIZE  ;
    }

    private static String appendDescriptorFromInternalType(String internalTypeName)
    {
        return internalTypeName.equals(String.valueOf(ClassConstants.TYPE_BOOLEAN)) ? ClassConstants.METHOD_TYPE_BOOLEAN_STRING_BUILDER :
               internalTypeName.equals(String.valueOf(ClassConstants.TYPE_CHAR))    ? ClassConstants.METHOD_TYPE_CHAR_STRING_BUILDER    :
               internalTypeName.equals(String.valueOf(ClassConstants.TYPE_BYTE))  ||
               internalTypeName.equals(String.valueOf(ClassConstants.TYPE_SHORT)) ||
               internalTypeName.equals(String.valueOf(ClassConstants.TYPE_INT))     ? ClassConstants.METHOD_TYPE_INT_STRING_BUILDER     :
               internalTypeName.equals(String.valueOf(ClassConstants.TYPE_LONG))    ? ClassConstants.METHOD_TYPE_LONG_STRING_BUILDER    :
               internalTypeName.equals(String.valueOf(ClassConstants.TYPE_FLOAT))   ? ClassConstants.METHOD_TYPE_FLOAT_STRING_BUILDER   :
               internalTypeName.equals(String.valueOf(ClassConstants.TYPE_DOUBLE))  ? ClassConstants.METHOD_TYPE_DOUBLE_STRING_BUILDER  :
               internalTypeName.equals(ClassConstants.TYPE_JAVA_LANG_STRING)        ? ClassConstants.METHOD_TYPE_STRING_STRING_BUILDER  :
                                                                                      ClassConstants.METHOD_TYPE_OBJECT_STRING_BUILDER  ;
    }

    private static int nextArgIndex(String recipe, int fromIndex)
    {
        for(int i = fromIndex; i < recipe.length(); i++)
        {
            char c = recipe.charAt(i);
            if (c == C_VARIABLE_ARGUMENT ||
                c == C_CONSTANT_ARGUMENT)
            {
                return i;
            }
        }
        return recipe.length();
    }
}

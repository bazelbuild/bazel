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
import proguard.classfile.attribute.annotation.*;
import proguard.classfile.attribute.annotation.visitor.*;
import proguard.classfile.attribute.module.*;
import proguard.classfile.attribute.module.visitor.*;
import proguard.classfile.attribute.preverification.*;
import proguard.classfile.attribute.preverification.visitor.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;

import java.util.Arrays;

/**
 * This ClassVisitor remaps all possible references to constant pool entries
 * of the classes that it visits, based on a given index map. It is assumed that
 * the constant pool entries themselves have already been remapped.
 *
 * @author Eric Lafortune
 */
public class ConstantPoolRemapper
extends      SimplifiedVisitor
implements   ClassVisitor,
             ConstantVisitor,
             MemberVisitor,
             AttributeVisitor,
             BootstrapMethodInfoVisitor,
             InnerClassesInfoVisitor,
             ExceptionInfoVisitor,
             InstructionVisitor,
             StackMapFrameVisitor,
             VerificationTypeVisitor,
             ParameterInfoVisitor,
             LocalVariableInfoVisitor,
             LocalVariableTypeInfoVisitor,
             RequiresInfoVisitor,
             ExportsInfoVisitor,
             OpensInfoVisitor,
             ProvidesInfoVisitor,
             AnnotationVisitor,
             ElementValueVisitor
{
    private final CodeAttributeEditor codeAttributeEditor = new CodeAttributeEditor(false, true);

    private int[] constantIndexMap;


    /**
     * Sets the given mapping of old constant pool entry indexes to their new
     * indexes.
     */
    public void setConstantIndexMap(int[] constantIndexMap)
    {
        this.constantIndexMap = constantIndexMap;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Remap the local constant pool references.
        programClass.u2thisClass  = remapConstantIndex(programClass.u2thisClass);
        programClass.u2superClass = remapConstantIndex(programClass.u2superClass);

        remapConstantIndexArray(programClass.u2interfaces,
                                programClass.u2interfacesCount);

        // Remap the references of the constant pool entries themselves.
        programClass.constantPoolEntriesAccept(this);

        // Remap the references in all fields, methods, and attributes.
        programClass.fieldsAccept(this);
        programClass.methodsAccept(this);
        programClass.attributesAccept(this);
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
    }


    // Implementations for ConstantVisitor.

    public void visitIntegerConstant(Clazz clazz, IntegerConstant integerConstant)
    {
        // Nothing to do.
    }


    public void visitLongConstant(Clazz clazz, LongConstant longConstant)
    {
        // Nothing to do.
    }


    public void visitFloatConstant(Clazz clazz, FloatConstant floatConstant)
    {
        // Nothing to do.
    }


    public void visitDoubleConstant(Clazz clazz, DoubleConstant doubleConstant)
    {
        // Nothing to do.
    }


    public void visitPrimitiveArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant)
    {
        // Nothing to do.
    }


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        stringConstant.u2stringIndex =
            remapConstantIndex(stringConstant.u2stringIndex);
    }


    public void visitUtf8Constant(Clazz clazz, Utf8Constant utf8Constant)
    {
        // Nothing to do.
    }


    public void visitDynamicConstant(Clazz clazz, DynamicConstant dynamicConstant)
    {
        dynamicConstant.u2nameAndTypeIndex =
            remapConstantIndex(dynamicConstant.u2nameAndTypeIndex);
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        invokeDynamicConstant.u2nameAndTypeIndex =
            remapConstantIndex(invokeDynamicConstant.u2nameAndTypeIndex);
    }


    public void visitMethodHandleConstant(Clazz clazz, MethodHandleConstant methodHandleConstant)
    {
        methodHandleConstant.u2referenceIndex =
            remapConstantIndex(methodHandleConstant.u2referenceIndex);
    }

    public void visitModuleConstant(Clazz clazz, ModuleConstant moduleConstant)
    {
        moduleConstant.u2nameIndex =
            remapConstantIndex(moduleConstant.u2nameIndex);
    }

    public void visitPackageConstant(Clazz clazz, PackageConstant packageConstant)
    {
        packageConstant.u2nameIndex =
            remapConstantIndex(packageConstant.u2nameIndex);
    }


    public void visitFieldrefConstant(Clazz clazz, FieldrefConstant fieldrefConstant)
    {
        fieldrefConstant.u2classIndex =
            remapConstantIndex(fieldrefConstant.u2classIndex);
        fieldrefConstant.u2nameAndTypeIndex =
            remapConstantIndex(fieldrefConstant.u2nameAndTypeIndex);
    }


    public void visitInterfaceMethodrefConstant(Clazz clazz, InterfaceMethodrefConstant interfaceMethodrefConstant)
    {
        interfaceMethodrefConstant.u2classIndex =
            remapConstantIndex(interfaceMethodrefConstant.u2classIndex);
        interfaceMethodrefConstant.u2nameAndTypeIndex =
            remapConstantIndex(interfaceMethodrefConstant.u2nameAndTypeIndex);
    }


    public void visitMethodrefConstant(Clazz clazz, MethodrefConstant methodrefConstant)
    {
        methodrefConstant.u2classIndex =
            remapConstantIndex(methodrefConstant.u2classIndex);
        methodrefConstant.u2nameAndTypeIndex =
            remapConstantIndex(methodrefConstant.u2nameAndTypeIndex);
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        classConstant.u2nameIndex =
            remapConstantIndex(classConstant.u2nameIndex);
    }


    public void visitMethodTypeConstant(Clazz clazz, MethodTypeConstant methodTypeConstant)
    {
        methodTypeConstant.u2descriptorIndex =
            remapConstantIndex(methodTypeConstant.u2descriptorIndex);
    }


    public void visitNameAndTypeConstant(Clazz clazz, NameAndTypeConstant nameAndTypeConstant)
    {
        nameAndTypeConstant.u2nameIndex =
            remapConstantIndex(nameAndTypeConstant.u2nameIndex);
        nameAndTypeConstant.u2descriptorIndex =
            remapConstantIndex(nameAndTypeConstant.u2descriptorIndex);
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        visitMember(programClass, programField);
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        visitMember(programClass, programMethod);
    }


    private void visitMember(ProgramClass programClass, ProgramMember programMember)
    {
        // Remap the local constant pool references.
        programMember.u2nameIndex =
            remapConstantIndex(programMember.u2nameIndex);
        programMember.u2descriptorIndex =
            remapConstantIndex(programMember.u2descriptorIndex);

        // Remap the constant pool references of the remaining attributes.
        programMember.attributesAccept(programClass, this);
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
    {
        // Library classes are left unchanged.
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        // Library classes are left unchanged.
    }


    // Implementations for AttributeVisitor.

    public void visitUnknownAttribute(Clazz clazz, UnknownAttribute unknownAttribute)
    {
        unknownAttribute.u2attributeNameIndex =
            remapConstantIndex(unknownAttribute.u2attributeNameIndex);

        // There's not much else we can do with unknown attributes.
    }


    public void visitBootstrapMethodsAttribute(Clazz clazz, BootstrapMethodsAttribute bootstrapMethodsAttribute)
    {
        bootstrapMethodsAttribute.u2attributeNameIndex =
            remapConstantIndex(bootstrapMethodsAttribute.u2attributeNameIndex);

        // Remap the constant pool references of the bootstrap method entries.
        bootstrapMethodsAttribute.bootstrapMethodEntriesAccept(clazz, this);
    }


    public void visitSourceFileAttribute(Clazz clazz, SourceFileAttribute sourceFileAttribute)
    {
        sourceFileAttribute.u2attributeNameIndex =
            remapConstantIndex(sourceFileAttribute.u2attributeNameIndex);
        sourceFileAttribute.u2sourceFileIndex =
            remapConstantIndex(sourceFileAttribute.u2sourceFileIndex);
    }


    public void visitSourceDirAttribute(Clazz clazz, SourceDirAttribute sourceDirAttribute)
    {
        sourceDirAttribute.u2attributeNameIndex =
            remapConstantIndex(sourceDirAttribute.u2attributeNameIndex);
        sourceDirAttribute.u2sourceDirIndex       =
            remapConstantIndex(sourceDirAttribute.u2sourceDirIndex);
    }


    public void visitInnerClassesAttribute(Clazz clazz, InnerClassesAttribute innerClassesAttribute)
    {
        innerClassesAttribute.u2attributeNameIndex =
            remapConstantIndex(innerClassesAttribute.u2attributeNameIndex);

        // Remap the constant pool references of the inner classes.
        innerClassesAttribute.innerClassEntriesAccept(clazz, this);
    }


    public void visitEnclosingMethodAttribute(Clazz clazz, EnclosingMethodAttribute enclosingMethodAttribute)
    {
        enclosingMethodAttribute.u2attributeNameIndex =
            remapConstantIndex(enclosingMethodAttribute.u2attributeNameIndex);
        enclosingMethodAttribute.u2classIndex =
            remapConstantIndex(enclosingMethodAttribute.u2classIndex);
        enclosingMethodAttribute.u2nameAndTypeIndex =
            remapConstantIndex(enclosingMethodAttribute.u2nameAndTypeIndex);
    }


    public void visitNestHostAttribute(Clazz clazz, NestHostAttribute nestHostAttribute)
    {
        nestHostAttribute.u2attributeNameIndex =
            remapConstantIndex(nestHostAttribute.u2attributeNameIndex);
        nestHostAttribute.u2hostClassIndex =
            remapConstantIndex(nestHostAttribute.u2hostClassIndex);
    }


    public void visitNestMembersAttribute(Clazz clazz, NestMembersAttribute nestMembersAttribute)
    {
        nestMembersAttribute.u2attributeNameIndex =
            remapConstantIndex(nestMembersAttribute.u2attributeNameIndex);

        remapConstantIndexArray(nestMembersAttribute.u2classes,
                                nestMembersAttribute.u2classesCount);
    }


    public void visitModuleAttribute(Clazz clazz, ModuleAttribute moduleAttribute)
    {
        moduleAttribute.u2attributeNameIndex =
            remapConstantIndex(moduleAttribute.u2attributeNameIndex);
        moduleAttribute.u2moduleNameIndex =
            remapConstantIndex(moduleAttribute.u2moduleNameIndex);

        if (moduleAttribute.u2moduleVersionIndex != 0)
        {
            moduleAttribute.u2moduleVersionIndex =
                remapConstantIndex(moduleAttribute.u2moduleVersionIndex);
        }
        moduleAttribute.requiresAccept(clazz, this);
        moduleAttribute.exportsAccept(clazz, this);
        moduleAttribute.opensAccept(clazz, this);
        remapConstantIndexArray(moduleAttribute.u2uses,
                                moduleAttribute.u2usesCount);
        moduleAttribute.providesAccept(clazz, this);
    }


    public void visitModuleMainClassAttribute(Clazz clazz, ModuleMainClassAttribute moduleMainClassAttribute)
    {
        moduleMainClassAttribute.u2attributeNameIndex =
            remapConstantIndex(moduleMainClassAttribute.u2attributeNameIndex);
        moduleMainClassAttribute.u2mainClass =
            remapConstantIndex(moduleMainClassAttribute.u2mainClass);
    }


    public void visitModulePackagesAttribute(Clazz clazz, ModulePackagesAttribute modulePackagesAttribute)
    {
        modulePackagesAttribute.u2attributeNameIndex =
            remapConstantIndex(modulePackagesAttribute.u2attributeNameIndex);

        remapConstantIndexArray(modulePackagesAttribute.u2packages,
                                modulePackagesAttribute.u2packagesCount);
    }


    public void visitDeprecatedAttribute(Clazz clazz, DeprecatedAttribute deprecatedAttribute)
    {
        deprecatedAttribute.u2attributeNameIndex =
            remapConstantIndex(deprecatedAttribute.u2attributeNameIndex);
    }


    public void visitSyntheticAttribute(Clazz clazz, SyntheticAttribute syntheticAttribute)
    {
        syntheticAttribute.u2attributeNameIndex =
            remapConstantIndex(syntheticAttribute.u2attributeNameIndex);
    }


    public void visitSignatureAttribute(Clazz clazz, SignatureAttribute signatureAttribute)
    {
        signatureAttribute.u2attributeNameIndex =
            remapConstantIndex(signatureAttribute.u2attributeNameIndex);
        signatureAttribute.u2signatureIndex       =
            remapConstantIndex(signatureAttribute.u2signatureIndex);
    }


    public void visitConstantValueAttribute(Clazz clazz, Field field, ConstantValueAttribute constantValueAttribute)
    {
        constantValueAttribute.u2attributeNameIndex =
            remapConstantIndex(constantValueAttribute.u2attributeNameIndex);
        constantValueAttribute.u2constantValueIndex =
            remapConstantIndex(constantValueAttribute.u2constantValueIndex);
    }


    public void visitMethodParametersAttribute(Clazz clazz, Method method, MethodParametersAttribute methodParametersAttribute)
    {
        methodParametersAttribute.u2attributeNameIndex =
            remapConstantIndex(methodParametersAttribute.u2attributeNameIndex);

        // Remap the constant pool references of the parameter information.
        methodParametersAttribute.parametersAccept(clazz, method, this);
    }


    public void visitExceptionsAttribute(Clazz clazz, Method method, ExceptionsAttribute exceptionsAttribute)
    {
        exceptionsAttribute.u2attributeNameIndex =
            remapConstantIndex(exceptionsAttribute.u2attributeNameIndex);

        // Remap the constant pool references of the exceptions.
        remapConstantIndexArray(exceptionsAttribute.u2exceptionIndexTable,
                                exceptionsAttribute.u2exceptionIndexTableLength);
    }


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        codeAttribute.u2attributeNameIndex =
            remapConstantIndex(codeAttribute.u2attributeNameIndex);

        // Initially, the code attribute editor doesn't contain any changes.
        codeAttributeEditor.reset(codeAttribute.u4codeLength);

        // Remap the constant pool references of the instructions.
        codeAttribute.instructionsAccept(clazz, method, this);

        // Apply the code atribute editor. It will only contain any changes if
        // the code length is changing at any point.
        codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);

        // Remap the constant pool references of the exceptions and attributes.
        codeAttribute.exceptionsAccept(clazz, method, this);
        codeAttribute.attributesAccept(clazz, method, this);
    }


    public void visitStackMapAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapAttribute stackMapAttribute)
    {
        stackMapAttribute.u2attributeNameIndex =
            remapConstantIndex(stackMapAttribute.u2attributeNameIndex);

        // Remap the constant pool references of the stack map frames.
        stackMapAttribute.stackMapFramesAccept(clazz, method, codeAttribute, this);
    }


    public void visitStackMapTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapTableAttribute stackMapTableAttribute)
    {
        stackMapTableAttribute.u2attributeNameIndex =
            remapConstantIndex(stackMapTableAttribute.u2attributeNameIndex);

        // Remap the constant pool references of the stack map frames.
        stackMapTableAttribute.stackMapFramesAccept(clazz, method, codeAttribute, this);
    }


    public void visitLineNumberTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute lineNumberTableAttribute)
    {
        lineNumberTableAttribute.u2attributeNameIndex =
            remapConstantIndex(lineNumberTableAttribute.u2attributeNameIndex);
    }


    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        localVariableTableAttribute.u2attributeNameIndex =
            remapConstantIndex(localVariableTableAttribute.u2attributeNameIndex);

        // Remap the constant pool references of the local variables.
        localVariableTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        localVariableTypeTableAttribute.u2attributeNameIndex =
            remapConstantIndex(localVariableTypeTableAttribute.u2attributeNameIndex);

        // Remap the constant pool references of the local variables.
        localVariableTypeTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitAnyAnnotationsAttribute(Clazz clazz, AnnotationsAttribute annotationsAttribute)
    {
        annotationsAttribute.u2attributeNameIndex =
            remapConstantIndex(annotationsAttribute.u2attributeNameIndex);

        // Remap the constant pool references of the annotations.
        annotationsAttribute.annotationsAccept(clazz, this);
    }


    public void visitAnyParameterAnnotationsAttribute(Clazz clazz, Method method, ParameterAnnotationsAttribute parameterAnnotationsAttribute)
    {
        parameterAnnotationsAttribute.u2attributeNameIndex =
            remapConstantIndex(parameterAnnotationsAttribute.u2attributeNameIndex);

        // Remap the constant pool references of the annotations.
        parameterAnnotationsAttribute.annotationsAccept(clazz, method, this);
    }


    public void visitAnnotationDefaultAttribute(Clazz clazz, Method method, AnnotationDefaultAttribute annotationDefaultAttribute)
    {
        annotationDefaultAttribute.u2attributeNameIndex =
            remapConstantIndex(annotationDefaultAttribute.u2attributeNameIndex);

        // Remap the constant pool references of the annotations.
        annotationDefaultAttribute.defaultValueAccept(clazz, this);
    }


    // Implementations for BootstrapMethodInfoVisitor.

    public void visitBootstrapMethodInfo(Clazz clazz, BootstrapMethodInfo bootstrapMethodInfo)
    {
        bootstrapMethodInfo.u2methodHandleIndex =
            remapConstantIndex(bootstrapMethodInfo.u2methodHandleIndex);

        // Remap the constant pool references of the bootstrap methods..
        remapConstantIndexArray(bootstrapMethodInfo.u2methodArguments,
                                bootstrapMethodInfo.u2methodArgumentCount);
    }


    // Implementations for InnerClassesInfoVisitor.

    public void visitInnerClassesInfo(Clazz clazz, InnerClassesInfo innerClassesInfo)
    {
        if (innerClassesInfo.u2innerClassIndex != 0)
        {
            innerClassesInfo.u2innerClassIndex =
                remapConstantIndex(innerClassesInfo.u2innerClassIndex);
        }

        if (innerClassesInfo.u2outerClassIndex != 0)
        {
            innerClassesInfo.u2outerClassIndex =
                remapConstantIndex(innerClassesInfo.u2outerClassIndex);
        }

        if (innerClassesInfo.u2innerNameIndex != 0)
        {
            innerClassesInfo.u2innerNameIndex =
                remapConstantIndex(innerClassesInfo.u2innerNameIndex);
        }
    }


    // Implementations for ExceptionInfoVisitor.

    public void visitExceptionInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, ExceptionInfo exceptionInfo)
    {
        if (exceptionInfo.u2catchType != 0)
        {
            exceptionInfo.u2catchType =
                remapConstantIndex(exceptionInfo.u2catchType);
        }
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        // Is the new constant pool index different from the original one?
        int newConstantIndex = remapConstantIndex(constantInstruction.constantIndex);
        if (newConstantIndex != constantInstruction.constantIndex)
        {
            // Replace the instruction.
            Instruction replacementInstruction =
                new ConstantInstruction(constantInstruction.opcode,
                                        newConstantIndex,
                                        constantInstruction.constant);

            codeAttributeEditor.replaceInstruction(offset, replacementInstruction);
        }
    }


    // Implementations for StackMapFrameVisitor.

    public void visitAnyStackMapFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, StackMapFrame stackMapFrame) {}


    public void visitSameOneFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SameOneFrame sameOneFrame)
    {
        // Remap the constant pool references of the verification types.
        sameOneFrame.stackItemAccept(clazz, method, codeAttribute, offset, this);
    }


    public void visitMoreZeroFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, MoreZeroFrame moreZeroFrame)
    {
        // Remap the constant pool references of the verification types.
        moreZeroFrame.additionalVariablesAccept(clazz, method, codeAttribute, offset, this);
    }


    public void visitFullFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, FullFrame fullFrame)
    {
        // Remap the constant pool references of the verification types.
        fullFrame.variablesAccept(clazz, method, codeAttribute, offset, this);
        fullFrame.stackAccept(clazz, method, codeAttribute, offset, this);
    }


    // Implementations for VerificationTypeVisitor.

    public void visitAnyVerificationType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VerificationType verificationType) {}


    public void visitObjectType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ObjectType objectType)
    {
        objectType.u2classIndex =
            remapConstantIndex(objectType.u2classIndex);
    }


    // Implementations for ParameterInfoVisitor.

    public void visitParameterInfo(Clazz clazz, Method method, int parameterIndex, ParameterInfo parameterInfo)
    {
        parameterInfo.u2nameIndex =
            remapConstantIndex(parameterInfo.u2nameIndex);
    }


    // Implementations for LocalVariableInfoVisitor.

    public void visitLocalVariableInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableInfo localVariableInfo)
    {
        localVariableInfo.u2nameIndex =
            remapConstantIndex(localVariableInfo.u2nameIndex);
        localVariableInfo.u2descriptorIndex =
            remapConstantIndex(localVariableInfo.u2descriptorIndex);
    }


    // Implementations for LocalVariableTypeInfoVisitor.

    public void visitLocalVariableTypeInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeInfo localVariableTypeInfo)
    {
        localVariableTypeInfo.u2nameIndex =
            remapConstantIndex(localVariableTypeInfo.u2nameIndex);
        localVariableTypeInfo.u2signatureIndex =
            remapConstantIndex(localVariableTypeInfo.u2signatureIndex);
    }


    // Implementations for RequiresInfoVisitor.

    public void visitRequiresInfo(Clazz clazz, RequiresInfo requiresInfo)
    {
        requiresInfo.u2requiresIndex =
            remapConstantIndex(requiresInfo.u2requiresIndex);
        requiresInfo.u2requiresVersionIndex =
            remapConstantIndex(requiresInfo.u2requiresVersionIndex);
    }


    // Implementations for ExportsInfoVisitor.

    public void visitExportsInfo(Clazz clazz, ExportsInfo exportsInfo)
    {
        exportsInfo.u2exportsIndex =
            remapConstantIndex(exportsInfo.u2exportsIndex);
        remapConstantIndexArray(exportsInfo.u2exportsToIndex,
                                exportsInfo.u2exportsToCount);
    }


    // Implementations for OpensInfoVisitor.

    public void visitOpensInfo(Clazz clazz, OpensInfo opensInfo)
    {
        opensInfo.u2opensIndex =
            remapConstantIndex(opensInfo.u2opensIndex);
        remapConstantIndexArray(opensInfo.u2opensToIndex,
                                opensInfo.u2opensToCount);
    }


    // Implementations for ProvidesInfoVisitor.

    public void visitProvidesInfo(Clazz clazz, ProvidesInfo providesInfo)
    {
        providesInfo.u2providesIndex =
            remapConstantIndex(providesInfo.u2providesIndex);
        remapConstantIndexArray(providesInfo.u2providesWithIndex,
                                providesInfo.u2providesWithCount);
    }

    // Implementations for AnnotationVisitor.

    public void visitAnnotation(Clazz clazz, Annotation annotation)
    {
        annotation.u2typeIndex =
            remapConstantIndex(annotation.u2typeIndex);

        // Remap the constant pool references of the element values.
        annotation.elementValuesAccept(clazz, this);
    }


    // Implementations for ElementValueVisitor.

    public void visitConstantElementValue(Clazz clazz, Annotation annotation, ConstantElementValue constantElementValue)
    {
        constantElementValue.u2elementNameIndex =
            remapConstantIndex(constantElementValue.u2elementNameIndex);
        constantElementValue.u2constantValueIndex =
            remapConstantIndex(constantElementValue.u2constantValueIndex);
    }


    public void visitEnumConstantElementValue(Clazz clazz, Annotation annotation, EnumConstantElementValue enumConstantElementValue)
    {
        enumConstantElementValue.u2elementNameIndex =
            remapConstantIndex(enumConstantElementValue.u2elementNameIndex);
        enumConstantElementValue.u2typeNameIndex =
            remapConstantIndex(enumConstantElementValue.u2typeNameIndex);
        enumConstantElementValue.u2constantNameIndex =
            remapConstantIndex(enumConstantElementValue.u2constantNameIndex);
    }


    public void visitClassElementValue(Clazz clazz, Annotation annotation, ClassElementValue classElementValue)
    {
        classElementValue.u2elementNameIndex =
            remapConstantIndex(classElementValue.u2elementNameIndex);
        classElementValue.u2classInfoIndex =
            remapConstantIndex(classElementValue.u2classInfoIndex);
    }


    public void visitAnnotationElementValue(Clazz clazz, Annotation annotation, AnnotationElementValue annotationElementValue)
    {
        annotationElementValue.u2elementNameIndex =
            remapConstantIndex(annotationElementValue.u2elementNameIndex);

        // Remap the constant pool references of the annotation.
        annotationElementValue.annotationAccept(clazz, this);
    }


    public void visitArrayElementValue(Clazz clazz, Annotation annotation, ArrayElementValue arrayElementValue)
    {
        arrayElementValue.u2elementNameIndex =
            remapConstantIndex(arrayElementValue.u2elementNameIndex);

        // Remap the constant pool references of the element values.
        arrayElementValue.elementValuesAccept(clazz, annotation, this);
    }


    /**
     * Remaps all constant pool indices in the given array.
     */
    private void remapConstantIndexArray(int[] array, int length)
    {
        for (int index = 0; index < length; index++)
        {
            array[index] = remapConstantIndex(array[index]);
        }
    }


    // Small utility methods.

    /**
     * Returns the new constant pool index of the entry at the
     * given index.
     */
    private int remapConstantIndex(int constantIndex)
    {
        int remappedConstantIndex = constantIndexMap[constantIndex];
        if (remappedConstantIndex < 0)
        {
            throw new IllegalArgumentException("Can't remap constant index ["+constantIndex+"]");
        }

        return remappedConstantIndex;
    }
}

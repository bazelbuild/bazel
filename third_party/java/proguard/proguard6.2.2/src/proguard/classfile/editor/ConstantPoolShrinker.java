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
 * This ClassVisitor removes all unused entries from the constant pool.
 *
 * @author Eric Lafortune
 */
public class ConstantPoolShrinker
extends      SimplifiedVisitor
implements   ClassVisitor,

             // Implementation interfaces.
             MemberVisitor,
             ConstantVisitor,
             AttributeVisitor,
             BootstrapMethodInfoVisitor,
             InnerClassesInfoVisitor,
             ExceptionInfoVisitor,
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
             ElementValueVisitor,
             InstructionVisitor
{
    // A visitor info flag to indicate that the constant is being used.
    // Don't make a static instance, so we don't need to clear any old flags.
    private final Object USED = new Object();

    private       int[]                constantIndexMap     = new int[ClassConstants.TYPICAL_CONSTANT_POOL_SIZE];
    private final ConstantPoolRemapper constantPoolRemapper = new ConstantPoolRemapper();


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Mark this class's name.
        markConstant(programClass, programClass.u2thisClass);

        // Mark the superclass class constant.
        programClass.superClassConstantAccept(this);

        // Mark the interface class constants.
        programClass.interfaceConstantsAccept(this);

        // Mark the constants referenced by the class members.
        programClass.fieldsAccept(this);
        programClass.methodsAccept(this);

        // Mark the attributes.
        programClass.attributesAccept(this);

        // Shift the used constant pool entries together, filling out the
        // index map.
        int newConstantPoolCount =
            shrinkConstantPool(programClass.constantPool,
                               programClass.u2constantPoolCount);

        // Remap the references to the constant pool if it has shrunk.
        if (newConstantPoolCount < programClass.u2constantPoolCount)
        {
            programClass.u2constantPoolCount = newConstantPoolCount;

            // Remap all constant pool references.
            constantPoolRemapper.setConstantIndexMap(constantIndexMap);
            constantPoolRemapper.visitProgramClass(programClass);
        }
    }


    // Implementations for MemberVisitor.

    public void visitProgramMember(ProgramClass programClass, ProgramMember programMember)
    {
        // Mark the name and descriptor.
        markConstant(programClass, programMember.u2nameIndex);
        markConstant(programClass, programMember.u2descriptorIndex);

        // Mark the attributes.
        programMember.attributesAccept(programClass, this);
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant)
    {
        markAsUsed(constant);
    }


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        markAsUsed(stringConstant);

        markConstant(clazz, stringConstant.u2stringIndex);
    }


    public void visitDynamicConstant(Clazz clazz, DynamicConstant dynamicConstant)
    {
        markAsUsed(dynamicConstant);

        markConstant(clazz, dynamicConstant.u2nameAndTypeIndex);

        // Mark the bootstrap methods attribute.
        clazz.attributesAccept(this);
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        markAsUsed(invokeDynamicConstant);

        markConstant(clazz, invokeDynamicConstant.u2nameAndTypeIndex);

        // Mark the bootstrap methods attribute.
        clazz.attributesAccept(this);
    }


    public void visitMethodHandleConstant(Clazz clazz, MethodHandleConstant methodHandleConstant)
    {
        markAsUsed(methodHandleConstant);

        markConstant(clazz, methodHandleConstant.u2referenceIndex);
    }


    public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
    {
        markAsUsed(refConstant);

        markConstant(clazz, refConstant.u2classIndex);
        markConstant(clazz, refConstant.u2nameAndTypeIndex);
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        markAsUsed(classConstant);

        markConstant(clazz, classConstant.u2nameIndex);
    }


    public void visitMethodTypeConstant(Clazz clazz, MethodTypeConstant methodTypeConstant)
    {
        markAsUsed(methodTypeConstant);

        markConstant(clazz, methodTypeConstant.u2descriptorIndex);
    }


    public void visitNameAndTypeConstant(Clazz clazz, NameAndTypeConstant nameAndTypeConstant)
    {
        markAsUsed(nameAndTypeConstant);

        markConstant(clazz, nameAndTypeConstant.u2nameIndex);
        markConstant(clazz, nameAndTypeConstant.u2descriptorIndex);
    }


    public void visitModuleConstant(Clazz clazz, ModuleConstant moduleConstant)
    {
        markAsUsed(moduleConstant);

        markConstant(clazz, moduleConstant.u2nameIndex);
    }


    public void visitPackageConstant(Clazz clazz, PackageConstant packageConstant)
    {
        markAsUsed(packageConstant);

        markConstant(clazz, packageConstant.u2nameIndex);
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute)
    {
        markConstant(clazz, attribute.u2attributeNameIndex);
    }


    public void visitBootstrapMethodsAttribute(Clazz clazz, BootstrapMethodsAttribute bootstrapMethodsAttribute)
    {
        markConstant(clazz, bootstrapMethodsAttribute.u2attributeNameIndex);

        // Mark the bootstrap method entries.
        bootstrapMethodsAttribute.bootstrapMethodEntriesAccept(clazz, this);
    }


    public void visitSourceFileAttribute(Clazz clazz, SourceFileAttribute sourceFileAttribute)
    {
        markConstant(clazz, sourceFileAttribute.u2attributeNameIndex);
        markConstant(clazz, sourceFileAttribute.u2sourceFileIndex);
    }


    public void visitSourceDirAttribute(Clazz clazz, SourceDirAttribute sourceDirAttribute)
    {
        markConstant(clazz, sourceDirAttribute.u2attributeNameIndex);
        markConstant(clazz, sourceDirAttribute.u2sourceDirIndex);
    }


    public void visitInnerClassesAttribute(Clazz clazz, InnerClassesAttribute innerClassesAttribute)
    {
        markConstant(clazz, innerClassesAttribute.u2attributeNameIndex);

        // Mark the outer class entries.
        innerClassesAttribute.innerClassEntriesAccept(clazz, this);
    }


    public void visitEnclosingMethodAttribute(Clazz clazz, EnclosingMethodAttribute enclosingMethodAttribute)
    {
        markConstant(        clazz, enclosingMethodAttribute.u2attributeNameIndex);
        markConstant(        clazz, enclosingMethodAttribute.u2classIndex);
        markOptionalConstant(clazz, enclosingMethodAttribute.u2nameAndTypeIndex);
    }


    public void visitNestHostAttribute(Clazz clazz, NestHostAttribute nestHostAttribute)
    {
        markConstant(clazz, nestHostAttribute.u2attributeNameIndex);
        markConstant(clazz, nestHostAttribute.u2hostClassIndex);
    }


    public void visitNestMembersAttribute(Clazz clazz, NestMembersAttribute nestMembersAttribute)
    {
        markConstant(clazz, nestMembersAttribute.u2attributeNameIndex);

        // Mark the nest member class constants.
        nestMembersAttribute.memberClassConstantsAccept(clazz, this);
    }


    public void visitModuleAttribute(Clazz clazz, ModuleAttribute moduleAttribute)
    {
        markConstant(        clazz, moduleAttribute.u2attributeNameIndex);
        markConstant(        clazz, moduleAttribute.u2moduleNameIndex);
        markOptionalConstant(clazz, moduleAttribute.u2moduleVersionIndex);

        // Mark the constant pool entries referenced by the contained info.
        moduleAttribute.requiresAccept(clazz, this);
        moduleAttribute.exportsAccept(clazz, this);
        moduleAttribute.opensAccept(clazz, this);

        markConstants(clazz, moduleAttribute.u2uses, moduleAttribute.u2usesCount);

        // Mark the constant pool entries referenced by the provides info.
        moduleAttribute.providesAccept(clazz, this);
    }


    public void visitModuleMainClassAttribute(Clazz clazz, ModuleMainClassAttribute moduleMainClassAttribute)
    {
        markConstant(clazz, moduleMainClassAttribute.u2attributeNameIndex);
        markConstant(clazz, moduleMainClassAttribute.u2mainClass);
    }


    public void visitModulePackagesAttribute(Clazz clazz, ModulePackagesAttribute modulePackagesAttribute)
    {
        markConstant(clazz, modulePackagesAttribute.u2attributeNameIndex);

        // Mark the constant pool entries referenced by the packages info.
        modulePackagesAttribute.packagesAccept(clazz, this);
    }


    public void visitSignatureAttribute(Clazz clazz, SignatureAttribute signatureAttribute)
    {
        markConstant(clazz, signatureAttribute.u2attributeNameIndex);
        markConstant(clazz, signatureAttribute.u2signatureIndex);
    }


    public void visitConstantValueAttribute(Clazz clazz, Field field, ConstantValueAttribute constantValueAttribute)
    {
        markConstant(clazz, constantValueAttribute.u2attributeNameIndex);
        markConstant(clazz, constantValueAttribute.u2constantValueIndex);
    }


    public void visitMethodParametersAttribute(Clazz clazz, Method method, MethodParametersAttribute methodParametersAttribute)
    {
        markConstant(clazz, methodParametersAttribute.u2attributeNameIndex);

        // Mark the constant pool entries referenced by the parameter information.
        methodParametersAttribute.parametersAccept(clazz, method, this);
    }


    public void visitExceptionsAttribute(Clazz clazz, Method method, ExceptionsAttribute exceptionsAttribute)
    {
        markConstant(clazz, exceptionsAttribute.u2attributeNameIndex);

        // Mark the constant pool entries referenced by the exceptions.
        exceptionsAttribute.exceptionEntriesAccept((ProgramClass)clazz, this);
    }


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        markConstant(clazz, codeAttribute.u2attributeNameIndex);

        // Mark the constant pool entries referenced by the instructions,
        // by the exceptions, and by the attributes.
        codeAttribute.instructionsAccept(clazz, method, this);
        codeAttribute.exceptionsAccept(clazz, method, this);
        codeAttribute.attributesAccept(clazz, method, this);
    }


    public void visitStackMapAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapAttribute stackMapAttribute)
    {
        markConstant(clazz, stackMapAttribute.u2attributeNameIndex);

        // Mark the constant pool entries referenced by the stack map frames.
        stackMapAttribute.stackMapFramesAccept(clazz, method, codeAttribute, this);
    }


    public void visitStackMapTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapTableAttribute stackMapTableAttribute)
    {
        markConstant(clazz, stackMapTableAttribute.u2attributeNameIndex);

        // Mark the constant pool entries referenced by the stack map frames.
        stackMapTableAttribute.stackMapFramesAccept(clazz, method, codeAttribute, this);
    }


    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        markConstant(clazz, localVariableTableAttribute.u2attributeNameIndex);

        // Mark the constant pool entries referenced by the local variables.
        localVariableTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        markConstant(clazz, localVariableTypeTableAttribute.u2attributeNameIndex);

        // Mark the constant pool entries referenced by the local variable types.
        localVariableTypeTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitAnyAnnotationsAttribute(Clazz clazz, AnnotationsAttribute annotationsAttribute)
    {
        markConstant(clazz, annotationsAttribute.u2attributeNameIndex);

        // Mark the constant pool entries referenced by the annotations.
        annotationsAttribute.annotationsAccept(clazz, this);
    }


    public void visitAnyParameterAnnotationsAttribute(Clazz clazz, Method method, ParameterAnnotationsAttribute parameterAnnotationsAttribute)
    {
        markConstant(clazz, parameterAnnotationsAttribute.u2attributeNameIndex);

        // Mark the constant pool entries referenced by the annotations.
        parameterAnnotationsAttribute.annotationsAccept(clazz, method, this);
    }


    public void visitAnnotationDefaultAttribute(Clazz clazz, Method method, AnnotationDefaultAttribute annotationDefaultAttribute)
    {
        markConstant(clazz, annotationDefaultAttribute.u2attributeNameIndex);

        // Mark the constant pool entries referenced by the element value.
        annotationDefaultAttribute.defaultValueAccept(clazz, this);
    }


    // Implementations for BootstrapMethodInfoVisitor.

    public void visitBootstrapMethodInfo(Clazz clazz, BootstrapMethodInfo bootstrapMethodInfo)
    {
        markConstant(clazz, bootstrapMethodInfo.u2methodHandleIndex);

        // Mark the constant pool entries referenced by the arguments.
        bootstrapMethodInfo.methodArgumentsAccept(clazz, this);
    }


    // Implementations for InnerClassesInfoVisitor.

    public void visitInnerClassesInfo(Clazz clazz, InnerClassesInfo innerClassesInfo)
    {
        // Mark the constant pool entries referenced by the contained info.
        innerClassesInfo.innerClassConstantAccept(clazz, this);
        innerClassesInfo.outerClassConstantAccept(clazz, this);
        innerClassesInfo.innerNameConstantAccept(clazz, this);
    }


    // Implementations for ExceptionInfoVisitor.

    public void visitExceptionInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, ExceptionInfo exceptionInfo)
    {
        markOptionalConstant(clazz, exceptionInfo.u2catchType);
    }


    // Implementations for StackMapFrameVisitor.

    public void visitAnyStackMapFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, StackMapFrame stackMapFrame) {}


    public void visitSameOneFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SameOneFrame sameOneFrame)
    {
        // Mark the constant pool entries referenced by the verification types.
        sameOneFrame.stackItemAccept(clazz, method, codeAttribute, offset, this);
    }


    public void visitMoreZeroFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, MoreZeroFrame moreZeroFrame)
    {
        // Mark the constant pool entries referenced by the verification types.
        moreZeroFrame.additionalVariablesAccept(clazz, method, codeAttribute, offset, this);
    }


    public void visitFullFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, FullFrame fullFrame)
    {
        // Mark the constant pool entries referenced by the verification types.
        fullFrame.variablesAccept(clazz, method, codeAttribute, offset, this);
        fullFrame.stackAccept(clazz, method, codeAttribute, offset, this);
    }


    // Implementations for VerificationTypeVisitor.

    public void visitAnyVerificationType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VerificationType verificationType) {}


    public void visitObjectType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ObjectType objectType)
    {
        markConstant(clazz, objectType.u2classIndex);
    }


    // Implementations for ParameterInfoVisitor.

    public void visitParameterInfo(Clazz clazz, Method method, int parameterIndex, ParameterInfo parameterInfo)
    {
        markOptionalConstant(clazz, parameterInfo.u2nameIndex);
    }


    // Implementations for LocalVariableInfoVisitor.

    public void visitLocalVariableInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableInfo localVariableInfo)
    {
        markConstant(clazz, localVariableInfo.u2nameIndex);
        markConstant(clazz, localVariableInfo.u2descriptorIndex);
    }


    // Implementations for LocalVariableTypeInfoVisitor.

    public void visitLocalVariableTypeInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeInfo localVariableTypeInfo)
    {
        markConstant(clazz, localVariableTypeInfo.u2nameIndex);
        markConstant(clazz, localVariableTypeInfo.u2signatureIndex);
    }


    // Implementations for RequiresInfoVisitor.

    public void visitRequiresInfo(Clazz clazz, RequiresInfo requiresInfo)
    {
        markConstant(        clazz, requiresInfo.u2requiresIndex);
        markOptionalConstant(clazz, requiresInfo.u2requiresVersionIndex);
    }


    // Implementations for ExportsInfoVisitor.

    public void visitExportsInfo(Clazz clazz, ExportsInfo exportsInfo)
    {
        markConstant( clazz,  exportsInfo.u2exportsIndex);
        markConstants(clazz, exportsInfo.u2exportsToIndex, exportsInfo.u2exportsToCount);
    }


    // Implementations for OpensInfoVisitor.

    public void visitOpensInfo(Clazz clazz, OpensInfo opensInfo)
    {
        markConstant( clazz, opensInfo.u2opensIndex);
        markConstants(clazz, opensInfo.u2opensToIndex, opensInfo.u2opensToCount);
    }


    // Implementations for ProvidesInfoVisitor.

    public void visitProvidesInfo(Clazz clazz, ProvidesInfo providesInfo)
    {
        markConstant( clazz, providesInfo.u2providesIndex);
        markConstants(clazz, providesInfo.u2providesWithIndex, providesInfo.u2providesWithCount);
    }


    // Implementations for AnnotationVisitor.

    public void visitAnnotation(Clazz clazz, Annotation annotation)
    {
        markConstant(clazz, annotation.u2typeIndex);

        // Mark the constant pool entries referenced by the element values.
        annotation.elementValuesAccept(clazz, this);
    }


    // Implementations for ElementValueVisitor.

    public void visitConstantElementValue(Clazz clazz, Annotation annotation, ConstantElementValue constantElementValue)
    {
        markOptionalConstant(clazz, constantElementValue.u2elementNameIndex);
        markConstant(        clazz, constantElementValue.u2constantValueIndex);
    }


    public void visitEnumConstantElementValue(Clazz clazz, Annotation annotation, EnumConstantElementValue enumConstantElementValue)
    {
        markOptionalConstant(clazz, enumConstantElementValue.u2elementNameIndex);
        markConstant(        clazz, enumConstantElementValue.u2typeNameIndex);
        markConstant(        clazz, enumConstantElementValue.u2constantNameIndex);
    }


    public void visitClassElementValue(Clazz clazz, Annotation annotation, ClassElementValue classElementValue)
    {
        markOptionalConstant(clazz, classElementValue.u2elementNameIndex);
        markConstant(        clazz, classElementValue.u2classInfoIndex);
    }


    public void visitAnnotationElementValue(Clazz clazz, Annotation annotation, AnnotationElementValue annotationElementValue)
    {
        markOptionalConstant(clazz, annotationElementValue.u2elementNameIndex);

        // Mark the constant pool entries referenced by the annotation.
        annotationElementValue.annotationAccept(clazz, this);
    }


    public void visitArrayElementValue(Clazz clazz, Annotation annotation, ArrayElementValue arrayElementValue)
    {
        markOptionalConstant(clazz, arrayElementValue.u2elementNameIndex);

        // Mark the constant pool entries referenced by the element values.
        arrayElementValue.elementValuesAccept(clazz, annotation, this);
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        markConstant(clazz, constantInstruction.constantIndex);
    }


    // Small utility methods.

    /**
     * Marks the specified constant pool entries of the given class.
     * This includes visiting any referenced objects.
     */
    private void markConstants(Clazz clazz,
                               int[] constantIndices,
                               int   constantIndicesCount)
    {
        for (int index = 0; index < constantIndicesCount; index++)
        {
            markConstant(clazz, constantIndices[index]);
        }
    }


    /**
     * Marks the specified constant pool entry of the given class, if the index
     * is not 0. This includes visiting any referenced objects.
     */
    private void markOptionalConstant(Clazz clazz, int constantIndex)
    {
        if (constantIndex != 0)
        {
            markConstant(clazz, constantIndex);
        }
    }


    /**
     * Marks the specified constant pool entry of the given class.
     * This includes visiting any referenced objects.
     */
    private void markConstant(Clazz clazz, int constantIndex)
    {
        clazz.constantPoolEntryAccept(constantIndex, this);
    }


    /**
     * Marks the given visitor accepter as being used.
     */
    private void markAsUsed(Constant constant)
    {
        constant.setVisitorInfo(USED);
    }


    /**
     * Returns whether the given visitor accepter has been marked as being used.
     */
    private boolean isUsed(VisitorAccepter visitorAccepter)
    {
        return visitorAccepter.getVisitorInfo() == USED;
    }


    /**
     * Removes all constants that are not marked as being used from the given
     * constant pool.
     * @return the new number of entries.
     */
    private int shrinkConstantPool(Constant[] constantPool, int length)
    {
        // Create a new index map, if necessary.
        if (constantIndexMap.length < length)
        {
            constantIndexMap = new int[length];
        }

        int     counter = 1;
        boolean isUsed  = false;

        // Shift the used constant pool entries together.
        for (int index = 1; index < length; index++)
        {
            Constant constant = constantPool[index];

            // Is the constant being used? Don't update the flag if this is the
            // second half of a long entry.
            if (constant != null)
            {
                isUsed = isUsed(constant);
            }

            if (isUsed)
            {
                // Remember the new index.
                constantIndexMap[index] = counter;

                // Shift the constant pool entry.
                constantPool[counter++] = constant;
            }
            else
            {
                // Remember an invalid index.
                constantIndexMap[index] = -1;
            }
        }

        // Clear the remaining constant pool elements.
        Arrays.fill(constantPool, counter, length, null);

        return counter;
    }
}

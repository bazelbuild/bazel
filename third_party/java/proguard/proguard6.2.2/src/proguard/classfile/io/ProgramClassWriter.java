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
package proguard.classfile.io;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.annotation.*;
import proguard.classfile.attribute.annotation.target.*;
import proguard.classfile.attribute.annotation.target.visitor.*;
import proguard.classfile.attribute.annotation.visitor.*;
import proguard.classfile.attribute.module.*;
import proguard.classfile.attribute.module.visitor.*;
import proguard.classfile.attribute.preverification.*;
import proguard.classfile.attribute.preverification.visitor.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;

import java.io.*;

/**
 * This ClassVisitor writes out the ProgramClass objects that it visits to the
 * given DataOutput object.
 *
 * @author Eric Lafortune
 */
public class ProgramClassWriter
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor,
             ConstantVisitor,
             AttributeVisitor
{
    private RuntimeDataOutput dataOutput;

    private final ConstantBodyWriter         constantBodyWriter         = new ConstantBodyWriter();
    private final AttributeBodyWriter        attributeBodyWriter        = new AttributeBodyWriter();
    private final StackMapFrameBodyWriter    stackMapFrameBodyWriter    = new StackMapFrameBodyWriter();
    private final VerificationTypeBodyWriter verificationTypeBodyWriter = new VerificationTypeBodyWriter();
    private final ElementValueBodyWriter     elementValueBodyWriter     = new ElementValueBodyWriter();


    /**
     * Creates a new ProgramClassWriter for writing to the given DataOutput.
     */
    public ProgramClassWriter(DataOutput dataOutput)
    {
        this.dataOutput = new RuntimeDataOutput(dataOutput);
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Write the magic number.
        dataOutput.writeInt(ClassConstants.MAGIC);

        // Write the version numbers.
        dataOutput.writeShort(ClassUtil.internalMinorClassVersion(programClass.u4version));
        dataOutput.writeShort(ClassUtil.internalMajorClassVersion(programClass.u4version));

        // Write the constant pool.
        dataOutput.writeUnsignedShort(programClass.u2constantPoolCount);

        programClass.constantPoolEntriesAccept(this);

        // Write the general class information.
        // Ignore the higher bits outside the short range - these are for
        // internal purposes only.
        dataOutput.writeUnsignedShort(programClass.u2accessFlags & 0xffff);
        dataOutput.writeUnsignedShort(programClass.u2thisClass);
        dataOutput.writeUnsignedShort(programClass.u2superClass);

        // Write the interfaces.
        writeUnsignedShorts(programClass.u2interfaces,
                            programClass.u2interfacesCount);

        // Write the fields.
        dataOutput.writeUnsignedShort(programClass.u2fieldsCount);

        programClass.fieldsAccept(this);

        // Write the methods.
        dataOutput.writeUnsignedShort(programClass.u2methodsCount);

        programClass.methodsAccept(this);

        // Write the class attributes.
        dataOutput.writeUnsignedShort(programClass.u2attributesCount);

        programClass.attributesAccept(this);
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        // Write the general field information.
        // Ignore the higher bits outside the short range - these are for
        // internal purposes only.
        dataOutput.writeUnsignedShort(programField.u2accessFlags & 0xffff);
        dataOutput.writeUnsignedShort(programField.u2nameIndex);
        dataOutput.writeUnsignedShort(programField.u2descriptorIndex);

        // Write the field attributes.
        dataOutput.writeUnsignedShort(programField.u2attributesCount);

        programField.attributesAccept(programClass, this);
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        // Write the general method information.
        // Ignore the higher bits outside the short range - these are for
        // internal purposes only.
        dataOutput.writeUnsignedShort(programMethod.u2accessFlags & 0xffff);
        dataOutput.writeUnsignedShort(programMethod.u2nameIndex);
        dataOutput.writeUnsignedShort(programMethod.u2descriptorIndex);

        // Write the method attributes.
        dataOutput.writeUnsignedShort(programMethod.u2attributesCount);

        programMethod.attributesAccept(programClass, this);
    }


    public void visitLibraryMember(LibraryClass libraryClass, LibraryMember libraryMember)
    {
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant)
    {
        // Write the tag.
        dataOutput.writeByte(constant.getTag());

        // Write the actual body.
        constant.accept(clazz, constantBodyWriter);
    }


    private class ConstantBodyWriter
    extends       SimplifiedVisitor
    implements    ConstantVisitor,
                  PrimitiveArrayConstantElementVisitor
    {
        // Implementations for ConstantVisitor.

        public void visitIntegerConstant(Clazz clazz, IntegerConstant integerConstant)
        {
            dataOutput.writeInt(integerConstant.u4value);
        }


        public void visitLongConstant(Clazz clazz, LongConstant longConstant)
        {
            dataOutput.writeLong(longConstant.u8value);
        }


        public void visitFloatConstant(Clazz clazz, FloatConstant floatConstant)
        {
            dataOutput.writeFloat(floatConstant.f4value);
        }


        public void visitDoubleConstant(Clazz clazz, DoubleConstant doubleConstant)
        {
            dataOutput.writeDouble(doubleConstant.f8value);
        }


        public void visitPrimitiveArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant)
        {
            char u2primitiveType = primitiveArrayConstant.getPrimitiveType();
            int  u4Length   = primitiveArrayConstant.getLength();

            dataOutput.writeUnsignedShort(u2primitiveType);
            dataOutput.writeInt(u4Length);

            // Write the array values.
            primitiveArrayConstant.primitiveArrayElementsAccept(clazz, this);
        }


        public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
        {
            dataOutput.writeUnsignedShort(stringConstant.u2stringIndex);
        }


        public void visitUtf8Constant(Clazz clazz, Utf8Constant utf8Constant)
        {
            byte[] bytes = utf8Constant.getBytes();

            dataOutput.writeUnsignedShort(bytes.length);
            dataOutput.write(bytes);
        }


        public void visitDynamicConstant(Clazz clazz, DynamicConstant dynamicConstant)
        {
            dataOutput.writeUnsignedShort(dynamicConstant.u2bootstrapMethodAttributeIndex);
            dataOutput.writeUnsignedShort(dynamicConstant.u2nameAndTypeIndex);
        }


        public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
        {
            dataOutput.writeUnsignedShort(invokeDynamicConstant.u2bootstrapMethodAttributeIndex);
            dataOutput.writeUnsignedShort(invokeDynamicConstant.u2nameAndTypeIndex);
        }


        public void visitMethodHandleConstant(Clazz clazz, MethodHandleConstant methodHandleConstant)
        {
            dataOutput.writeByte(methodHandleConstant.u1referenceKind);
            dataOutput.writeUnsignedShort(methodHandleConstant.u2referenceIndex);
        }


        public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
        {
            dataOutput.writeUnsignedShort(refConstant.u2classIndex);
            dataOutput.writeUnsignedShort(refConstant.u2nameAndTypeIndex);
        }


        public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
        {
            dataOutput.writeUnsignedShort(classConstant.u2nameIndex);
        }


        public void visitMethodTypeConstant(Clazz clazz, MethodTypeConstant methodTypeConstant)
        {
            dataOutput.writeUnsignedShort(methodTypeConstant.u2descriptorIndex);
        }


        public void visitNameAndTypeConstant(Clazz clazz, NameAndTypeConstant nameAndTypeConstant)
        {
            dataOutput.writeUnsignedShort(nameAndTypeConstant.u2nameIndex);
            dataOutput.writeUnsignedShort(nameAndTypeConstant.u2descriptorIndex);
        }


        // Implementations for PrimitiveArrayConstantElementVisitor.

        public void visitBooleanArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, boolean value)
        {
            dataOutput.writeBoolean(value);
        }


        public void visitByteArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, byte value)
        {
            dataOutput.writeByte(value);
        }


        public void visitCharArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, char value)
        {
            dataOutput.writeChar(value);
        }


        public void visitShortArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, short value)
        {
            dataOutput.writeShort(value);
        }


        public void visitIntArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, int value)
        {
            dataOutput.writeInt(value);
        }


        public void visitFloatArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, float value)
        {
            dataOutput.writeFloat(value);
        }


        public void visitLongArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, long value)
        {
            dataOutput.writeLong(value);
        }


        public void visitDoubleArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, double value)
        {
            dataOutput.writeDouble(value);
        }


        public void visitModuleConstant(Clazz clazz, ModuleConstant moduleConstant)
        {
            dataOutput.writeUnsignedShort(moduleConstant.u2nameIndex);
        }


        public void visitPackageConstant(Clazz clazz, PackageConstant packageConstant)
        {
            dataOutput.writeUnsignedShort(packageConstant.u2nameIndex);
        }
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute)
    {
        // Write the attribute name index.
        dataOutput.writeUnsignedShort(attribute.u2attributeNameIndex);

        // We'll write the attribute body into an array first, so we can
        // automatically figure out its length.
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();

        // Temporarily replace the current data output.
        RuntimeDataOutput oldDataOutput = dataOutput;
        dataOutput = new RuntimeDataOutput(new DataOutputStream(byteArrayOutputStream));

        // Write the attribute body into the array. Note that the
        // accept method with two dummy null arguments never throws
        // an UnsupportedOperationException.
        attribute.accept(clazz, null, null, attributeBodyWriter);

        // Restore the original data output.
        dataOutput = oldDataOutput;

        // Write the attribute length and body.
        byte[] info = byteArrayOutputStream.toByteArray();

        dataOutput.writeInt(info.length);
        dataOutput.write(info);
    }


    private class AttributeBodyWriter
    extends       SimplifiedVisitor
    implements    AttributeVisitor,
                  BootstrapMethodInfoVisitor,
                  InnerClassesInfoVisitor,
                  ExceptionInfoVisitor,
                  StackMapFrameVisitor,
                  VerificationTypeVisitor,
                  LineNumberInfoVisitor,
                  ParameterInfoVisitor,
                  LocalVariableInfoVisitor,
                  LocalVariableTypeInfoVisitor,
                  RequiresInfoVisitor,
                  ExportsInfoVisitor,
                  OpensInfoVisitor,
                  ProvidesInfoVisitor,
                  AnnotationVisitor,
                  TypeAnnotationVisitor,
                  TargetInfoVisitor,
                  TypePathInfoVisitor,
                  LocalVariableTargetElementVisitor,
                  ElementValueVisitor
    {
        // Implementations for AttributeVisitor.

        public void visitUnknownAttribute(Clazz clazz, UnknownAttribute unknownAttribute)
        {
            // Write the unknown information.
            dataOutput.write(unknownAttribute.info);
        }


        public void visitBootstrapMethodsAttribute(Clazz clazz, BootstrapMethodsAttribute bootstrapMethodsAttribute)
        {
            // Write the bootstrap methods.
            dataOutput.writeUnsignedShort(bootstrapMethodsAttribute.u2bootstrapMethodsCount);

            bootstrapMethodsAttribute.bootstrapMethodEntriesAccept(clazz, this);
        }


        public void visitSourceFileAttribute(Clazz clazz, SourceFileAttribute sourceFileAttribute)
        {
            dataOutput.writeUnsignedShort(sourceFileAttribute.u2sourceFileIndex);
        }


        public void visitSourceDirAttribute(Clazz clazz, SourceDirAttribute sourceDirAttribute)
        {
            dataOutput.writeUnsignedShort(sourceDirAttribute.u2sourceDirIndex);
        }


        public void visitInnerClassesAttribute(Clazz clazz, InnerClassesAttribute innerClassesAttribute)
        {
            // Write the inner classes.
            dataOutput.writeUnsignedShort(innerClassesAttribute.u2classesCount);

            innerClassesAttribute.innerClassEntriesAccept(clazz, this);
        }


        public void visitEnclosingMethodAttribute(Clazz clazz, EnclosingMethodAttribute enclosingMethodAttribute)
        {
            dataOutput.writeUnsignedShort(enclosingMethodAttribute.u2classIndex);
            dataOutput.writeUnsignedShort(enclosingMethodAttribute.u2nameAndTypeIndex);
        }


        public void visitNestHostAttribute(Clazz clazz, NestHostAttribute nestHostAttribute)
        {
            dataOutput.writeUnsignedShort(nestHostAttribute.u2hostClassIndex);
        }


        public void visitNestMembersAttribute(Clazz clazz, NestMembersAttribute nestMembersAttribute)
        {
            // Write the nest host classes.
            writeUnsignedShorts(nestMembersAttribute.u2classes,
                                nestMembersAttribute.u2classesCount);
        }


        public void visitModuleAttribute(Clazz clazz, ModuleAttribute moduleAttribute)
        {
            dataOutput.writeUnsignedShort(moduleAttribute.u2moduleNameIndex);
            dataOutput.writeUnsignedShort(moduleAttribute.u2moduleFlags);
            dataOutput.writeUnsignedShort(moduleAttribute.u2moduleVersionIndex);

            // Write the requires.
            dataOutput.writeUnsignedShort(moduleAttribute.u2requiresCount);

            moduleAttribute.requiresAccept(clazz, this);

            // Write the exports.
            dataOutput.writeUnsignedShort(moduleAttribute.u2exportsCount);

            moduleAttribute.exportsAccept(clazz, this);

            // Write the opens.
            dataOutput.writeUnsignedShort(moduleAttribute.u2opensCount);

            moduleAttribute.opensAccept(clazz, this);

            // Write the uses.
            writeUnsignedShorts(moduleAttribute.u2uses, moduleAttribute.u2usesCount);

            // Write the provides.
            dataOutput.writeUnsignedShort(moduleAttribute.u2providesCount);

            moduleAttribute.providesAccept(clazz, this);
        }


        public void visitModuleMainClassAttribute(Clazz clazz, ModuleMainClassAttribute moduleMainClassAttribute)
        {
            dataOutput.writeUnsignedShort(moduleMainClassAttribute.u2mainClass);
        }


        public void visitModulePackagesAttribute(Clazz clazz, ModulePackagesAttribute modulePackagesAttribute)
        {
            // Write the packages.
            writeUnsignedShorts(modulePackagesAttribute.u2packages,
                                modulePackagesAttribute.u2packagesCount);
        }


        public void visitDeprecatedAttribute(Clazz clazz, DeprecatedAttribute deprecatedAttribute)
        {
            // This attribute does not contain any additional information.
        }


        public void visitSyntheticAttribute(Clazz clazz, SyntheticAttribute syntheticAttribute)
        {
            // This attribute does not contain any additional information.
        }


        public void visitSignatureAttribute(Clazz clazz, SignatureAttribute signatureAttribute)
        {
            dataOutput.writeUnsignedShort(signatureAttribute.u2signatureIndex);
        }


        public void visitConstantValueAttribute(Clazz clazz, Field field, ConstantValueAttribute constantValueAttribute)
        {
            dataOutput.writeUnsignedShort(constantValueAttribute.u2constantValueIndex);
        }


        public void visitMethodParametersAttribute(Clazz clazz, Method method, MethodParametersAttribute methodParametersAttribute)
        {
            // Write the parameter information.
            dataOutput.writeByte(methodParametersAttribute.u1parametersCount);

            methodParametersAttribute.parametersAccept(clazz, method, this);
        }


        public void visitExceptionsAttribute(Clazz clazz, Method method, ExceptionsAttribute exceptionsAttribute)
        {
            // Write the exceptions.
            writeUnsignedShorts(exceptionsAttribute.u2exceptionIndexTable,
                                exceptionsAttribute.u2exceptionIndexTableLength);
        }


        public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
        {
            // Write the stack size and local variable frame size.
            dataOutput.writeUnsignedShort(codeAttribute.u2maxStack);
            dataOutput.writeUnsignedShort(codeAttribute.u2maxLocals);

            // Write the byte code.
            dataOutput.writeInt(codeAttribute.u4codeLength);

            dataOutput.write(codeAttribute.code, 0, codeAttribute.u4codeLength);

            // Write the exceptions.
            dataOutput.writeUnsignedShort(codeAttribute.u2exceptionTableLength);

            codeAttribute.exceptionsAccept(clazz, method, this);

            // Write the code attributes.
            dataOutput.writeUnsignedShort(codeAttribute.u2attributesCount);

            codeAttribute.attributesAccept(clazz, method, ProgramClassWriter.this);
        }


        public void visitStackMapAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapAttribute stackMapAttribute)
        {
            // Write the stack map frames (only full frames, without tag).
            dataOutput.writeUnsignedShort(stackMapAttribute.u2stackMapFramesCount);

            stackMapAttribute.stackMapFramesAccept(clazz, method, codeAttribute, stackMapFrameBodyWriter);
        }


        public void visitStackMapTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapTableAttribute stackMapTableAttribute)
        {
            // Write the stack map frames.
            dataOutput.writeUnsignedShort(stackMapTableAttribute.u2stackMapFramesCount);

            stackMapTableAttribute.stackMapFramesAccept(clazz, method, codeAttribute, this);
        }


        public void visitLineNumberTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute lineNumberTableAttribute)
        {
            // Write the line numbers.
            dataOutput.writeUnsignedShort(lineNumberTableAttribute.u2lineNumberTableLength);

            lineNumberTableAttribute.lineNumbersAccept(clazz, method, codeAttribute, this);
        }


        public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
        {
            // Write the local variables.
            dataOutput.writeUnsignedShort(localVariableTableAttribute.u2localVariableTableLength);

            localVariableTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
        }


        public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
        {
            // Write the local variable types.
            dataOutput.writeUnsignedShort(localVariableTypeTableAttribute.u2localVariableTypeTableLength);

            localVariableTypeTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
        }


        public void visitRequiresInfo(Clazz clazz, RequiresInfo requiresInfo)
        {
            dataOutput.writeUnsignedShort(requiresInfo.u2requiresIndex);
            dataOutput.writeUnsignedShort(requiresInfo.u2requiresFlags);
            dataOutput.writeUnsignedShort(requiresInfo.u2requiresVersionIndex);
        }


        public void visitExportsInfo(Clazz clazz, ExportsInfo exportsInfo)
        {
            dataOutput.writeUnsignedShort(exportsInfo.u2exportsIndex);
            dataOutput.writeUnsignedShort(exportsInfo.u2exportsFlags);

            // Write tthe argets.
            writeUnsignedShorts(exportsInfo.u2exportsToIndex,
                                exportsInfo.u2exportsToCount);
        }


        public void visitOpensInfo(Clazz clazz, OpensInfo opensInfo)
        {
            dataOutput.writeUnsignedShort(opensInfo.u2opensIndex);
            dataOutput.writeUnsignedShort(opensInfo.u2opensFlags);

            // Write the targets.
            writeUnsignedShorts(opensInfo.u2opensToIndex,
                                opensInfo.u2opensToCount);
        }


        public void visitProvidesInfo(Clazz clazz, ProvidesInfo providesInfo)
        {
            dataOutput.writeUnsignedShort(providesInfo.u2providesIndex);

            // Write the provides.
            writeUnsignedShorts(providesInfo.u2providesWithIndex,
                                providesInfo.u2providesWithCount);
        }


        public void visitAnyAnnotationsAttribute(Clazz clazz, AnnotationsAttribute annotationsAttribute)
        {
            // Write the annotations.
            dataOutput.writeUnsignedShort(annotationsAttribute.u2annotationsCount);

            annotationsAttribute.annotationsAccept(clazz, this);
        }


        public void visitAnyParameterAnnotationsAttribute(Clazz clazz, Method method, ParameterAnnotationsAttribute parameterAnnotationsAttribute)
        {
            // Write the parameter annotations.
            dataOutput.writeByte(parameterAnnotationsAttribute.u1parametersCount);

            for (int parameterIndex = 0; parameterIndex < parameterAnnotationsAttribute.u1parametersCount; parameterIndex++)
            {
                // Write the parameter annotations of the given parameter.
                int          u2annotationsCount = parameterAnnotationsAttribute.u2parameterAnnotationsCount[parameterIndex];
                Annotation[] annotations        = parameterAnnotationsAttribute.parameterAnnotations[parameterIndex];

                dataOutput.writeUnsignedShort(u2annotationsCount);

                for (int index = 0; index < u2annotationsCount; index++)
                {
                    visitAnnotation(clazz, annotations[index]);
                }

            }
        }


        public void visitAnyTypeAnnotationsAttribute(Clazz clazz, TypeAnnotationsAttribute typeAnnotationsAttribute)
        {
            // Write the type annotations.
            dataOutput.writeUnsignedShort(typeAnnotationsAttribute.u2annotationsCount);

            typeAnnotationsAttribute.typeAnnotationsAccept(clazz, this);
        }


        public void visitAnnotationDefaultAttribute(Clazz clazz, Method method, AnnotationDefaultAttribute annotationDefaultAttribute)
        {
            // Write the default element value.
            annotationDefaultAttribute.defaultValue.accept(clazz, null, this);
        }


        // Implementations for BootstrapMethodInfoVisitor.

        public void visitBootstrapMethodInfo(Clazz clazz, BootstrapMethodInfo bootstrapMethodInfo)
        {
            dataOutput.writeUnsignedShort(bootstrapMethodInfo.u2methodHandleIndex);

            // Write the bootstrap method arguments.
            writeUnsignedShorts(bootstrapMethodInfo.u2methodArguments,
                                bootstrapMethodInfo.u2methodArgumentCount);
        }


        // Implementations for InnerClassesInfoVisitor.

        public void visitInnerClassesInfo(Clazz clazz, InnerClassesInfo innerClassesInfo)
        {
            dataOutput.writeUnsignedShort(innerClassesInfo.u2innerClassIndex);
            dataOutput.writeUnsignedShort(innerClassesInfo.u2outerClassIndex);
            dataOutput.writeUnsignedShort(innerClassesInfo.u2innerNameIndex);
            dataOutput.writeUnsignedShort(innerClassesInfo.u2innerClassAccessFlags);
        }


        // Implementations for ExceptionInfoVisitor.

        public void visitExceptionInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, ExceptionInfo exceptionInfo)
        {
            dataOutput.writeUnsignedShort(exceptionInfo.u2startPC);
            dataOutput.writeUnsignedShort(exceptionInfo.u2endPC);
            dataOutput.writeUnsignedShort(exceptionInfo.u2handlerPC);
            dataOutput.writeUnsignedShort(exceptionInfo.u2catchType);
        }


        // Implementations for StackMapFrameVisitor.

        public void visitAnyStackMapFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, StackMapFrame stackMapFrame)
        {
            // Write the stack map frame tag.
            dataOutput.writeByte(stackMapFrame.getTag());

            // Write the actual body.
            stackMapFrame.accept(clazz, method, codeAttribute, offset, stackMapFrameBodyWriter);
        }


        // Implementations for LineNumberInfoVisitor.

        public void visitLineNumberInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberInfo lineNumberInfo)
        {
            // Simply suppress line number overflows, typically caused by
            // inlined methods and their artificial line numbers.
            dataOutput.writeUnsignedShort(lineNumberInfo.u2startPC    > 0xffff ? 0 : lineNumberInfo.u2startPC);
            dataOutput.writeUnsignedShort(lineNumberInfo.u2lineNumber > 0xffff ? 0 : lineNumberInfo.u2lineNumber);
        }


        // Implementations for ParameterInfoVisitor.

        public void visitParameterInfo(Clazz clazz, Method method, int parameterIndex, ParameterInfo parameterInfo)
        {
            dataOutput.writeUnsignedShort(parameterInfo.u2nameIndex);
            dataOutput.writeUnsignedShort(parameterInfo.u2accessFlags);
        }


        // Implementations for LocalVariableInfoVisitor.

        public void visitLocalVariableInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableInfo localVariableInfo)
        {
            dataOutput.writeUnsignedShort(localVariableInfo.u2startPC);
            dataOutput.writeUnsignedShort(localVariableInfo.u2length);
            dataOutput.writeUnsignedShort(localVariableInfo.u2nameIndex);
            dataOutput.writeUnsignedShort(localVariableInfo.u2descriptorIndex);
            dataOutput.writeUnsignedShort(localVariableInfo.u2index);
        }


        // Implementations for LocalVariableTypeInfoVisitor.

        public void visitLocalVariableTypeInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeInfo localVariableTypeInfo)
        {
            dataOutput.writeUnsignedShort(localVariableTypeInfo.u2startPC);
            dataOutput.writeUnsignedShort(localVariableTypeInfo.u2length);
            dataOutput.writeUnsignedShort(localVariableTypeInfo.u2nameIndex);
            dataOutput.writeUnsignedShort(localVariableTypeInfo.u2signatureIndex);
            dataOutput.writeUnsignedShort(localVariableTypeInfo.u2index);
        }


        // Implementations for AnnotationVisitor.

        public void visitAnnotation(Clazz clazz, Annotation annotation)
        {
            // Write the annotation type.
            dataOutput.writeUnsignedShort(annotation.u2typeIndex);

            // Write the element value pairs.
            dataOutput.writeUnsignedShort(annotation.u2elementValuesCount);

            annotation.elementValuesAccept(clazz, this);
        }


        // Implementations for TypeAnnotationVisitor.

        public void visitTypeAnnotation(Clazz clazz, TypeAnnotation typeAnnotation)
        {
            // Write the target info.
            dataOutput.writeByte(typeAnnotation.targetInfo.u1targetType);

            typeAnnotation.targetInfoAccept(clazz, this);

            // Write the type path.
            dataOutput.writeByte(typeAnnotation.typePath.length);

            typeAnnotation.typePathInfosAccept(clazz, this);

            // Write the actual annotation.
            visitAnnotation(clazz, typeAnnotation);
        }


        // Implementations for TargetInfoVisitor.

        public void visitTypeParameterTargetInfo(Clazz clazz, TypeAnnotation typeAnnotation, TypeParameterTargetInfo typeParameterTargetInfo)
        {
            dataOutput.writeByte(typeParameterTargetInfo.u1typeParameterIndex);
        }


        public void visitSuperTypeTargetInfo(Clazz clazz, TypeAnnotation typeAnnotation, SuperTypeTargetInfo superTypeTargetInfo)
        {
            dataOutput.writeUnsignedShort(superTypeTargetInfo.u2superTypeIndex);
        }


        public void visitTypeParameterBoundTargetInfo(Clazz clazz, TypeAnnotation typeAnnotation, TypeParameterBoundTargetInfo typeParameterBoundTargetInfo)
        {
            dataOutput.writeByte(typeParameterBoundTargetInfo.u1typeParameterIndex);
            dataOutput.writeByte(typeParameterBoundTargetInfo.u1boundIndex);
        }


        public void visitEmptyTargetInfo(Clazz clazz, Member member, TypeAnnotation typeAnnotation, EmptyTargetInfo emptyTargetInfo)
        {
        }


        public void visitFormalParameterTargetInfo(Clazz clazz, Method method, TypeAnnotation typeAnnotation, FormalParameterTargetInfo formalParameterTargetInfo)
        {
            dataOutput.writeByte(formalParameterTargetInfo.u1formalParameterIndex);
        }


        public void visitThrowsTargetInfo(Clazz clazz, Method method, TypeAnnotation typeAnnotation, ThrowsTargetInfo throwsTargetInfo)
        {
            dataOutput.writeUnsignedShort(throwsTargetInfo.u2throwsTypeIndex);
        }


        public void visitLocalVariableTargetInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, LocalVariableTargetInfo localVariableTargetInfo)
        {
            // Write the local variable target elements.
            dataOutput.writeUnsignedShort(localVariableTargetInfo.u2tableLength);

            localVariableTargetInfo.targetElementsAccept(clazz, method, codeAttribute, typeAnnotation, this);
        }


        public void visitCatchTargetInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, CatchTargetInfo catchTargetInfo)
        {
            dataOutput.writeUnsignedShort(catchTargetInfo.u2exceptionTableIndex);
        }


        public void visitOffsetTargetInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, OffsetTargetInfo offsetTargetInfo)
        {
            dataOutput.writeUnsignedShort(offsetTargetInfo.u2offset);
        }


        public void visitTypeArgumentTargetInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, TypeArgumentTargetInfo typeArgumentTargetInfo)
        {
            dataOutput.writeUnsignedShort(typeArgumentTargetInfo.u2offset);
            dataOutput.writeByte(typeArgumentTargetInfo.u1typeArgumentIndex);
        }


        // Implementations for TypePathInfoVisitor.

        public void visitTypePathInfo(Clazz clazz, TypeAnnotation typeAnnotation, TypePathInfo typePathInfo)
        {
            dataOutput.writeByte(typePathInfo.u1typePathKind);
            dataOutput.writeByte(typePathInfo.u1typeArgumentIndex);
        }


        // Implementations for LocalVariableTargetElementVisitor.

        public void visitLocalVariableTargetElement(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, LocalVariableTargetInfo localVariableTargetInfo, LocalVariableTargetElement localVariableTargetElement)
        {
            dataOutput.writeUnsignedShort(localVariableTargetElement.u2startPC);
            dataOutput.writeUnsignedShort(localVariableTargetElement.u2length);
            dataOutput.writeUnsignedShort(localVariableTargetElement.u2index);
        }


        // Implementations for ElementValueVisitor.

        public void visitAnyElementValue(Clazz clazz, Annotation annotation, ElementValue elementValue)
        {
            // Write the element name index, if applicable.
            int u2elementNameIndex = elementValue.u2elementNameIndex;
            if (u2elementNameIndex != 0)
            {
                dataOutput.writeUnsignedShort(u2elementNameIndex);
            }

            // Write the tag.
            dataOutput.writeByte(elementValue.getTag());

            // Write the actual body.
            elementValue.accept(clazz, annotation, elementValueBodyWriter);
        }
    }


    private class StackMapFrameBodyWriter
    extends       SimplifiedVisitor
    implements    StackMapFrameVisitor,
                  VerificationTypeVisitor
    {
        public void visitSameZeroFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SameZeroFrame sameZeroFrame)
        {
            if (sameZeroFrame.getTag() == StackMapFrame.SAME_ZERO_FRAME_EXTENDED)
            {
                dataOutput.writeUnsignedShort(sameZeroFrame.u2offsetDelta);
            }
        }


        public void visitSameOneFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SameOneFrame sameOneFrame)
        {
            if (sameOneFrame.getTag() == StackMapFrame.SAME_ONE_FRAME_EXTENDED)
            {
                dataOutput.writeUnsignedShort(sameOneFrame.u2offsetDelta);
            }

            // Write the verification type of the stack entry.
            sameOneFrame.stackItemAccept(clazz, method, codeAttribute, offset, this);
        }


        public void visitLessZeroFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, LessZeroFrame lessZeroFrame)
        {
            dataOutput.writeUnsignedShort(lessZeroFrame.u2offsetDelta);
        }


        public void visitMoreZeroFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, MoreZeroFrame moreZeroFrame)
        {
            dataOutput.writeUnsignedShort(moreZeroFrame.u2offsetDelta);

            // Write the verification types of the additional local variables.
            moreZeroFrame.additionalVariablesAccept(clazz, method, codeAttribute, offset, this);
        }


        public void visitFullFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, FullFrame fullFrame)
        {
            dataOutput.writeUnsignedShort(fullFrame.u2offsetDelta);

            // Write the verification types of the local variables.
            dataOutput.writeUnsignedShort(fullFrame.variablesCount);
            fullFrame.variablesAccept(clazz, method, codeAttribute, offset, this);

            // Write the verification types of the stack entries.
            dataOutput.writeUnsignedShort(fullFrame.stackCount);
            fullFrame.stackAccept(clazz, method, codeAttribute, offset, this);
        }


        // Implementations for VerificationTypeVisitor.

        public void visitAnyVerificationType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VerificationType verificationType)
        {
            // Write the verification type tag.
            dataOutput.writeByte(verificationType.getTag());

            // Write the actual body.
            verificationType.accept(clazz, method, codeAttribute, offset, verificationTypeBodyWriter);
        }
    }


    private class VerificationTypeBodyWriter
    extends       SimplifiedVisitor
    implements    VerificationTypeVisitor
    {
        // Implementations for VerificationTypeVisitor.

        public void visitAnyVerificationType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VerificationType verificationType)
        {
            // Most verification types don't contain any additional information.
        }


        public void visitObjectType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ObjectType objectType)
        {
            dataOutput.writeUnsignedShort(objectType.u2classIndex);
        }


        public void visitUninitializedType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, UninitializedType uninitializedType)
        {
            dataOutput.writeUnsignedShort(uninitializedType.u2newInstructionOffset);
        }
    }


    private class ElementValueBodyWriter
    extends       SimplifiedVisitor
    implements    ElementValueVisitor
    {
        // Implementations for ElementValueVisitor.

        public void visitConstantElementValue(Clazz clazz, Annotation annotation, ConstantElementValue constantElementValue)
        {
            dataOutput.writeUnsignedShort(constantElementValue.u2constantValueIndex);
        }


        public void visitEnumConstantElementValue(Clazz clazz, Annotation annotation, EnumConstantElementValue enumConstantElementValue)
        {
            dataOutput.writeUnsignedShort(enumConstantElementValue.u2typeNameIndex);
            dataOutput.writeUnsignedShort(enumConstantElementValue.u2constantNameIndex);
        }


        public void visitClassElementValue(Clazz clazz, Annotation annotation, ClassElementValue classElementValue)
        {
            dataOutput.writeUnsignedShort(classElementValue.u2classInfoIndex);
        }


        public void visitAnnotationElementValue(Clazz clazz, Annotation annotation, AnnotationElementValue annotationElementValue)
        {
            // Write the annotation.
            attributeBodyWriter.visitAnnotation(clazz, annotationElementValue.annotationValue);
        }


        public void visitArrayElementValue(Clazz clazz, Annotation annotation, ArrayElementValue arrayElementValue)
        {
            // Write the element values.
            dataOutput.writeUnsignedShort(arrayElementValue.u2elementValuesCount);

            arrayElementValue.elementValuesAccept(clazz, annotation, attributeBodyWriter);
        }
    }


    // Small utility methods.

    /**
     * Writes the length and the contents of the given list of unsigned shorts.
     */
    private void writeUnsignedShorts(int[] array, int size)
    {
        dataOutput.writeUnsignedShort(size);

        for (int index = 0; index < size; index++)
        {
            dataOutput.writeUnsignedShort(array[index]);
        }
    }
}

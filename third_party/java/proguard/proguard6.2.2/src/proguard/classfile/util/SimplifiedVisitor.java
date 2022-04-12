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
import proguard.classfile.attribute.annotation.*;
import proguard.classfile.attribute.annotation.target.*;
import proguard.classfile.attribute.module.*;
import proguard.classfile.attribute.preverification.*;
import proguard.classfile.constant.*;
import proguard.classfile.instruction.*;

/**
 * This abstract utility class allows to implement various visitor interfaces
 * with simplified methods. The provided methods delegate to other versions
 * with fewer arguments or more general arguments.
 *
 * @author Eric Lafortune
 * @noinspection AbstractClassWithoutAbstractMethods
 */
public abstract class SimplifiedVisitor
{
    // Simplifications for ClassVisitor.

    /**
     * Visits any type of class member of the given class.
     */
    public void visitAnyClass(Clazz clazz)
    {
        throw new UnsupportedOperationException("Method must be overridden in ["+this.getClass().getName()+"] if ever called");
    }


    public void visitProgramClass(ProgramClass programClass)
    {
        visitAnyClass(programClass);
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        visitAnyClass(libraryClass);
    }


    // Simplifications for MemberVisitor.

    /**
     * Visits any type of class member of the given class.
     */
    public void visitAnyMember(Clazz clazz, Member member)
    {
        throw new UnsupportedOperationException("Method must be overridden in ["+this.getClass().getName()+"] if ever called");
    }


    /**
     * Visits any type of class member of the given program class.
     */
    public void visitProgramMember(ProgramClass programClass, ProgramMember programMember)
    {
        visitAnyMember(programClass, programMember);
    }


    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        visitProgramMember(programClass, programField);
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        visitProgramMember(programClass, programMethod);
    }


    /**
     * Visits any type of class member of the given library class.
     */
    public void visitLibraryMember(LibraryClass libraryClass, LibraryMember libraryMember)
    {
        visitAnyMember(libraryClass, libraryMember);
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
    {
        visitLibraryMember(libraryClass, libraryField);
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        visitLibraryMember(libraryClass, libraryMethod);
    }


    // Simplifications for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant)
    {
        throw new UnsupportedOperationException("Method must be overridden in ["+this.getClass().getName()+"] if ever called");
    }


    public void visitIntegerConstant(Clazz clazz, IntegerConstant integerConstant)
    {
        visitAnyConstant(clazz, integerConstant);
    }


    public void visitLongConstant(Clazz clazz, LongConstant longConstant)
    {
        visitAnyConstant(clazz, longConstant);
    }


    public void visitFloatConstant(Clazz clazz, FloatConstant floatConstant)
    {
        visitAnyConstant(clazz, floatConstant);
    }


    public void visitDoubleConstant(Clazz clazz, DoubleConstant doubleConstant)
    {
        visitAnyConstant(clazz, doubleConstant);
    }


    public void visitPrimitiveArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant)
    {
        visitAnyConstant(clazz, primitiveArrayConstant);
    }


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        visitAnyConstant(clazz, stringConstant);
    }


    public void visitUtf8Constant(Clazz clazz, Utf8Constant utf8Constant)
    {
        visitAnyConstant(clazz, utf8Constant);
    }


    public void visitDynamicConstant(Clazz clazz, DynamicConstant dynamicConstant)
    {
        visitAnyConstant(clazz, dynamicConstant);
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        visitAnyConstant(clazz, invokeDynamicConstant);
    }


    public void visitMethodHandleConstant(Clazz clazz, MethodHandleConstant methodHandleConstant)
    {
        visitAnyConstant(clazz, methodHandleConstant);
    }

    public void visitModuleConstant(Clazz clazz, ModuleConstant moduleConstant)
    {
        visitAnyConstant(clazz, moduleConstant);
    }

    public void visitPackageConstant(Clazz clazz, PackageConstant packageConstant)
    {
        visitAnyConstant(clazz, packageConstant);
    }


    /**
     * Visits any type of RefConstant of the given class.
     */
    public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
    {
        visitAnyConstant(clazz, refConstant);
    }


    public void visitFieldrefConstant(Clazz clazz, FieldrefConstant fieldrefConstant)
    {
        visitAnyRefConstant(clazz, fieldrefConstant);
    }


    /**
     * Visits any type of method RefConstant of the given class.
     */
    public void visitAnyMethodrefConstant(Clazz clazz, RefConstant refConstant)
    {
        visitAnyRefConstant(clazz, refConstant);
    }


    public void visitInterfaceMethodrefConstant(Clazz clazz, InterfaceMethodrefConstant interfaceMethodrefConstant)
    {
        visitAnyMethodrefConstant(clazz, interfaceMethodrefConstant);
    }


    public void visitMethodrefConstant(Clazz clazz, MethodrefConstant methodrefConstant)
    {
        visitAnyMethodrefConstant(clazz, methodrefConstant);
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        visitAnyConstant(clazz, classConstant);
    }


    public void visitMethodTypeConstant(Clazz clazz, MethodTypeConstant methodTypeConstant)
    {
        visitAnyConstant(clazz, methodTypeConstant);
    }


    public void visitNameAndTypeConstant(Clazz clazz, NameAndTypeConstant nameAndTypeConstant)
    {
        visitAnyConstant(clazz, nameAndTypeConstant);
    }


    // Simplifications for PrimitiveArrayConstantVisitor.

    public void visitAnyPrimitiveArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, Object values)
    {
        throw new UnsupportedOperationException("Method must be overridden in ["+this.getClass().getName()+"] if ever called");
    }


    public void visitBooleanArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, boolean[] values)
    {
        visitAnyPrimitiveArrayConstant(clazz, primitiveArrayConstant, values);
    }


    public void visitByteArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, byte[] values)
    {
        visitAnyPrimitiveArrayConstant(clazz, primitiveArrayConstant, values);
    }


    public void visitCharArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, char[] values)
    {
        visitAnyPrimitiveArrayConstant(clazz, primitiveArrayConstant, values);
    }


    public void visitShortArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, short[] values)
    {
        visitAnyPrimitiveArrayConstant(clazz, primitiveArrayConstant, values);
    }


    public void visitIntArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int[] values)
    {
        visitAnyPrimitiveArrayConstant(clazz, primitiveArrayConstant, values);
    }


    public void visitFloatArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, float[] values)
    {
        visitAnyPrimitiveArrayConstant(clazz, primitiveArrayConstant, values);
    }


    public void visitLongArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, long[] values)
    {
        visitAnyPrimitiveArrayConstant(clazz, primitiveArrayConstant, values);
    }


    public void visitDoubleArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, double[] values)
    {
        visitAnyPrimitiveArrayConstant(clazz, primitiveArrayConstant, values);
    }


    // Simplifications for PrimitiveArrayConstantElementVisitor.

    public void visitAnyPrimitiveArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index)
    {
        throw new UnsupportedOperationException("Method must be overridden in ["+this.getClass().getName()+"] if ever called");
    }


    public void visitBooleanArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, boolean value)
    {
        visitAnyPrimitiveArrayConstantElement(clazz, primitiveArrayConstant, index);
    }


    public void visitByteArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, byte value)
    {
        visitAnyPrimitiveArrayConstantElement(clazz, primitiveArrayConstant, index);
    }


    public void visitCharArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, char value)
    {
        visitAnyPrimitiveArrayConstantElement(clazz, primitiveArrayConstant, index);
    }


    public void visitShortArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, short value)
    {
        visitAnyPrimitiveArrayConstantElement(clazz, primitiveArrayConstant, index);
    }


    public void visitIntArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, int value)
    {
        visitAnyPrimitiveArrayConstantElement(clazz, primitiveArrayConstant, index);
    }


    public void visitFloatArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, float value)
    {
        visitAnyPrimitiveArrayConstantElement(clazz, primitiveArrayConstant, index);
    }


    public void visitLongArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, long value)
    {
        visitAnyPrimitiveArrayConstantElement(clazz, primitiveArrayConstant, index);
    }


    public void visitDoubleArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, double value)
    {
        visitAnyPrimitiveArrayConstantElement(clazz, primitiveArrayConstant, index);
    }


    // Simplifications for AttributeVisitor.

    /**
     * Visit any type of attribute.
     */
    public void visitAnyAttribute(Clazz clazz, Attribute attribute)
    {
        throw new UnsupportedOperationException("Method must be overridden in ["+this.getClass().getName()+"] if ever called");
    }


    public void visitUnknownAttribute(Clazz clazz, UnknownAttribute unknownAttribute)
    {
        visitAnyAttribute(clazz, unknownAttribute);
    }


    public void visitBootstrapMethodsAttribute(Clazz clazz, BootstrapMethodsAttribute bootstrapMethodsAttribute)
    {
        visitAnyAttribute(clazz, bootstrapMethodsAttribute);
    }


    public void visitSourceFileAttribute(Clazz clazz, SourceFileAttribute sourceFileAttribute)
    {
        visitAnyAttribute(clazz, sourceFileAttribute);
    }


    public void visitSourceDirAttribute(Clazz clazz, SourceDirAttribute sourceDirAttribute)
    {
        visitAnyAttribute(clazz, sourceDirAttribute);
    }


    public void visitInnerClassesAttribute(Clazz clazz, InnerClassesAttribute innerClassesAttribute)
    {
        visitAnyAttribute(clazz, innerClassesAttribute);
    }


    public void visitEnclosingMethodAttribute(Clazz clazz, EnclosingMethodAttribute enclosingMethodAttribute)
    {
        visitAnyAttribute(clazz, enclosingMethodAttribute);
    }


    public void visitNestHostAttribute(Clazz clazz, NestHostAttribute nestHostAttribute)
    {
        visitAnyAttribute(clazz, nestHostAttribute);
    }


    public void visitNestMembersAttribute(Clazz clazz, NestMembersAttribute nestMembersAttribute)
    {
        visitAnyAttribute(clazz, nestMembersAttribute);
    }


    public void visitModuleAttribute(Clazz clazz, ModuleAttribute moduleAttribute)
    {
        visitAnyAttribute(clazz, moduleAttribute);
    }


    public void visitModuleMainClassAttribute(Clazz clazz, ModuleMainClassAttribute moduleMainClassAttribute)
    {
        visitAnyAttribute(clazz, moduleMainClassAttribute);
    }


    public void visitModulePackagesAttribute(Clazz clazz, ModulePackagesAttribute modulePackagesAttribute)
    {
        visitAnyAttribute(clazz, modulePackagesAttribute);
    }


    public void visitDeprecatedAttribute(Clazz clazz, DeprecatedAttribute deprecatedAttribute)
    {
        visitAnyAttribute(clazz, deprecatedAttribute);
    }


    /**
     * Visits the given DeprecatedAttribute of any type of class member.
     */
    public void visitDeprecatedAttribute(Clazz clazz, Member member, DeprecatedAttribute deprecatedAttribute)
    {
        visitDeprecatedAttribute(clazz, deprecatedAttribute);
    }


    public void visitDeprecatedAttribute(Clazz clazz, Field field, DeprecatedAttribute deprecatedAttribute)
    {
        visitDeprecatedAttribute(clazz, (Member)field, deprecatedAttribute);
    }


    public void visitDeprecatedAttribute(Clazz clazz, Method method, DeprecatedAttribute deprecatedAttribute)
    {
        visitDeprecatedAttribute(clazz, (Member)method, deprecatedAttribute);
    }


    public void visitSyntheticAttribute(Clazz clazz, SyntheticAttribute syntheticAttribute)
    {
        visitAnyAttribute(clazz, syntheticAttribute);
    }


    /**
     * Visits the given SyntheticAttribute of any type of class member.
     */
    public void visitSyntheticAttribute(Clazz clazz, Member member, SyntheticAttribute syntheticAttribute)
    {
        visitSyntheticAttribute(clazz, syntheticAttribute);
    }


    public void visitSyntheticAttribute(Clazz clazz, Field field, SyntheticAttribute syntheticAttribute)
    {
        visitSyntheticAttribute(clazz, (Member)field, syntheticAttribute);
    }


    public void visitSyntheticAttribute(Clazz clazz, Method method, SyntheticAttribute syntheticAttribute)
    {
        visitSyntheticAttribute(clazz, (Member)method, syntheticAttribute);
    }


    public void visitSignatureAttribute(Clazz clazz, SignatureAttribute signatureAttribute)
    {
        visitAnyAttribute(clazz, signatureAttribute);
    }


    /**
     * Visits the given SignatureAttribute of any type of class member.
     */
    public void visitSignatureAttribute(Clazz clazz, Member member, SignatureAttribute signatureAttribute)
    {
        visitSignatureAttribute(clazz, signatureAttribute);
    }


    public void visitSignatureAttribute(Clazz clazz, Field field, SignatureAttribute signatureAttribute)
    {
        visitSignatureAttribute(clazz, (Member)field, signatureAttribute);
    }


    public void visitSignatureAttribute(Clazz clazz, Method method, SignatureAttribute signatureAttribute)
    {
        visitSignatureAttribute(clazz, (Member)method, signatureAttribute);
    }


    public void visitConstantValueAttribute(Clazz clazz, Field field, ConstantValueAttribute constantValueAttribute)
    {
        visitAnyAttribute(clazz, constantValueAttribute);
    }


    public void visitMethodParametersAttribute(Clazz clazz, Method method, MethodParametersAttribute methodParametersAttribute)
    {
        visitAnyAttribute(clazz, methodParametersAttribute);
    }


    public void visitExceptionsAttribute(Clazz clazz, Method method, ExceptionsAttribute exceptionsAttribute)
    {
        visitAnyAttribute(clazz, exceptionsAttribute);
    }


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        visitAnyAttribute(clazz, codeAttribute);
    }


    public void visitStackMapAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapAttribute stackMapAttribute)
    {
        visitAnyAttribute(clazz, stackMapAttribute);
    }


    public void visitStackMapTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapTableAttribute stackMapTableAttribute)
    {
        visitAnyAttribute(clazz, stackMapTableAttribute);
    }


    public void visitLineNumberTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute lineNumberTableAttribute)
    {
        visitAnyAttribute(clazz, lineNumberTableAttribute);
    }


    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        visitAnyAttribute(clazz, localVariableTableAttribute);
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        visitAnyAttribute(clazz, localVariableTypeTableAttribute);
    }


    /**
     * Visits any type of AnnotationsAttribute of a class.
     */
    public void visitAnyAnnotationsAttribute(Clazz clazz, AnnotationsAttribute annotationsAttribute)
    {
        visitAnyAttribute(clazz, annotationsAttribute);
    }


    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        visitAnyAnnotationsAttribute(clazz, runtimeVisibleAnnotationsAttribute);
    }


    /**
     * Visits the given RuntimeVisibleAnnotationsAttribute of any type of class member.
     */
    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, Member member, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        visitRuntimeVisibleAnnotationsAttribute(clazz, runtimeVisibleAnnotationsAttribute);
    }


    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, Field field, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        visitRuntimeVisibleAnnotationsAttribute(clazz, (Member)field, runtimeVisibleAnnotationsAttribute);
    }


    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        visitRuntimeVisibleAnnotationsAttribute(clazz, (Member)method, runtimeVisibleAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        visitAnyAnnotationsAttribute(clazz, runtimeInvisibleAnnotationsAttribute);
    }


    /**
     * Visits the given RuntimeInvisibleAnnotationsAttribute of any type of class member.
     */
    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, Member member, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        visitRuntimeInvisibleAnnotationsAttribute(clazz, runtimeInvisibleAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, Field field, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        visitRuntimeInvisibleAnnotationsAttribute(clazz, (Member)field, runtimeInvisibleAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        visitRuntimeInvisibleAnnotationsAttribute(clazz, (Member)method, runtimeInvisibleAnnotationsAttribute);
    }


    /**
     * Visits any type of ParameterAnnotationsAttribute.
     */
    public void visitAnyParameterAnnotationsAttribute(Clazz clazz, Method method, ParameterAnnotationsAttribute parameterAnnotationsAttribute)
    {
        visitAnyAttribute(clazz, parameterAnnotationsAttribute);
    }


    public void visitRuntimeVisibleParameterAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleParameterAnnotationsAttribute runtimeVisibleParameterAnnotationsAttribute)
    {
        visitAnyParameterAnnotationsAttribute(clazz, method, runtimeVisibleParameterAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleParameterAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleParameterAnnotationsAttribute runtimeInvisibleParameterAnnotationsAttribute)
    {
        visitAnyParameterAnnotationsAttribute(clazz, method, runtimeInvisibleParameterAnnotationsAttribute);
    }


    /**
     * Visits any type of TypeAnnotationsAttribute of a class.
     */
    public void visitAnyTypeAnnotationsAttribute(Clazz clazz, TypeAnnotationsAttribute typeAnnotationsAttribute)
    {
        visitAnyAnnotationsAttribute(clazz, typeAnnotationsAttribute);
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        visitAnyTypeAnnotationsAttribute(clazz, runtimeVisibleTypeAnnotationsAttribute);
    }


    /**
     * Visits the given RuntimeVisibleTypeAnnotationsAttribute of any type of class member.
     */
    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Member member, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        visitRuntimeVisibleTypeAnnotationsAttribute(clazz, runtimeVisibleTypeAnnotationsAttribute);
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Field field, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        visitRuntimeVisibleTypeAnnotationsAttribute(clazz, (Member)field, runtimeVisibleTypeAnnotationsAttribute);
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        visitRuntimeVisibleTypeAnnotationsAttribute(clazz, (Member)method, runtimeVisibleTypeAnnotationsAttribute);
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        visitRuntimeVisibleTypeAnnotationsAttribute(clazz, method, runtimeVisibleTypeAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        visitAnyTypeAnnotationsAttribute(clazz, runtimeInvisibleTypeAnnotationsAttribute);
    }


    /**
     * Visits the given RuntimeInvisibleTypeAnnotationsAttribute of any type of class member.
     */
    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Member member, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, runtimeInvisibleTypeAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Field field, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, (Member)field, runtimeInvisibleTypeAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, (Member)method, runtimeInvisibleTypeAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, method, runtimeInvisibleTypeAnnotationsAttribute);
    }


    public void visitAnnotationDefaultAttribute(Clazz clazz, Method method, AnnotationDefaultAttribute annotationDefaultAttribute)
    {
        visitAnyAttribute(clazz, annotationDefaultAttribute);
    }


    // Simplifications for InstructionVisitor.

    /**
     * Visits any type of Instruction.
     */
    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
    {
        throw new UnsupportedOperationException("Method must be overridden in ["+this.getClass().getName()+"] if ever called");
    }


    public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
    {
        visitAnyInstruction(clazz, method, codeAttribute, offset, simpleInstruction);
    }


    public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
    {
        visitAnyInstruction(clazz, method, codeAttribute, offset, variableInstruction);
    }


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        visitAnyInstruction(clazz, method, codeAttribute, offset, constantInstruction);
    }


    public void visitBranchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction branchInstruction)
    {
        visitAnyInstruction(clazz, method, codeAttribute, offset, branchInstruction);
    }


    /**
     * Visits either type of SwitchInstruction.
     */
    public void visitAnySwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SwitchInstruction switchInstruction)
    {
        visitAnyInstruction(clazz, method, codeAttribute, offset, switchInstruction);
    }


    public void visitTableSwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, TableSwitchInstruction tableSwitchInstruction)
    {
        visitAnySwitchInstruction(clazz, method, codeAttribute, offset, tableSwitchInstruction);
    }


    public void visitLookUpSwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, LookUpSwitchInstruction lookUpSwitchInstruction)
    {
        visitAnySwitchInstruction(clazz, method, codeAttribute, offset, lookUpSwitchInstruction);
    }


    // Simplifications for StackMapFrameVisitor.

    /**
     * Visits any type of VerificationType.
     */
    public void visitAnyStackMapFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, StackMapFrame stackMapFrame)
    {
        throw new UnsupportedOperationException("Method must be overridden in ["+this.getClass().getName()+"] if ever called");
    }


    public void visitSameZeroFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SameZeroFrame sameZeroFrame)
    {
        visitAnyStackMapFrame(clazz, method, codeAttribute, offset, sameZeroFrame);
    }


    public void visitSameOneFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SameOneFrame sameOneFrame)
    {
        visitAnyStackMapFrame(clazz, method, codeAttribute, offset, sameOneFrame);
    }


    public void visitLessZeroFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, LessZeroFrame lessZeroFrame)
    {
        visitAnyStackMapFrame(clazz, method, codeAttribute, offset, lessZeroFrame);
    }


    public void visitMoreZeroFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, MoreZeroFrame moreZeroFrame)
    {
        visitAnyStackMapFrame(clazz, method, codeAttribute, offset, moreZeroFrame);
    }


    public void visitFullFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, FullFrame fullFrame)
    {
        visitAnyStackMapFrame(clazz, method, codeAttribute, offset, fullFrame);
    }


    // Simplifications for VerificationTypeVisitor.

    /**
     * Visits any type of VerificationType.
     */
    public void visitAnyVerificationType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VerificationType verificationType)
    {
        throw new UnsupportedOperationException("Method must be overridden in ["+this.getClass().getName()+"] if ever called");
    }


    public void visitIntegerType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, IntegerType integerType)
    {
        visitAnyVerificationType(clazz, method, codeAttribute, offset, integerType);
    }


    public void visitFloatType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, FloatType floatType)
    {
        visitAnyVerificationType(clazz, method, codeAttribute, offset, floatType);
    }


    public void visitLongType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, LongType longType)
    {
        visitAnyVerificationType(clazz, method, codeAttribute, offset, longType);
    }


    public void visitDoubleType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, DoubleType doubleType)
    {
        visitAnyVerificationType(clazz, method, codeAttribute, offset, doubleType);
    }


    public void visitTopType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, TopType topType)
    {
        visitAnyVerificationType(clazz, method, codeAttribute, offset, topType);
    }


    public void visitObjectType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ObjectType objectType)
    {
        visitAnyVerificationType(clazz, method, codeAttribute, offset, objectType);
    }


    public void visitNullType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, NullType nullType)
    {
        visitAnyVerificationType(clazz, method, codeAttribute, offset, nullType);
    }


    public void visitUninitializedType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, UninitializedType uninitializedType)
    {
        visitAnyVerificationType(clazz, method, codeAttribute, offset, uninitializedType);
    }


    public void visitUninitializedThisType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, UninitializedThisType uninitializedThisType)
    {
        visitAnyVerificationType(clazz, method, codeAttribute, offset, uninitializedThisType);
    }


    public void visitStackIntegerType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, IntegerType integerType)
    {
        visitIntegerType(clazz, method, codeAttribute, offset, integerType);
    }


    public void visitStackFloatType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, FloatType floatType)
    {
        visitFloatType(clazz, method, codeAttribute, offset, floatType);
    }


    public void visitStackLongType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, LongType longType)
    {
        visitLongType(clazz, method, codeAttribute, offset, longType);
    }


    public void visitStackDoubleType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, DoubleType doubleType)
    {
        visitDoubleType(clazz, method, codeAttribute, offset, doubleType);
    }


    public void visitStackTopType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, TopType topType)
    {
        visitTopType(clazz, method, codeAttribute, offset, topType);
    }


    public void visitStackObjectType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, ObjectType objectType)
    {
        visitObjectType(clazz, method, codeAttribute, offset, objectType);
    }


    public void visitStackNullType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, NullType nullType)
    {
        visitNullType(clazz, method, codeAttribute, offset, nullType);
    }


    public void visitStackUninitializedType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, UninitializedType uninitializedType)
    {
        visitUninitializedType(clazz, method, codeAttribute, offset, uninitializedType);
    }


    public void visitStackUninitializedThisType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, UninitializedThisType uninitializedThisType)
    {
        visitUninitializedThisType(clazz, method, codeAttribute, offset, uninitializedThisType);
    }



    public void visitVariablesIntegerType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, IntegerType integerType)
    {
        visitIntegerType(clazz, method, codeAttribute, offset, integerType);
    }


    public void visitVariablesFloatType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, FloatType floatType)
    {
        visitFloatType(clazz, method, codeAttribute, offset, floatType);
    }


    public void visitVariablesLongType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, LongType longType)
    {
        visitLongType(clazz, method, codeAttribute, offset, longType);
    }


    public void visitVariablesDoubleType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, DoubleType doubleType)
    {
        visitDoubleType(clazz, method, codeAttribute, offset, doubleType);
    }


    public void visitVariablesTopType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, TopType topType)
    {
        visitTopType(clazz, method, codeAttribute, offset, topType);
    }


    public void visitVariablesObjectType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, ObjectType objectType)
    {
        visitObjectType(clazz, method, codeAttribute, offset, objectType);
    }


    public void visitVariablesNullType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, NullType nullType)
    {
        visitNullType(clazz, method, codeAttribute, offset, nullType);
    }


    public void visitVariablesUninitializedType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, UninitializedType uninitializedType)
    {
        visitUninitializedType(clazz, method, codeAttribute, offset, uninitializedType);
    }


    public void visitVariablesUninitializedThisType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, UninitializedThisType uninitializedThisType)
    {
        visitUninitializedThisType(clazz, method, codeAttribute, offset, uninitializedThisType);
    }


    // Simplifications for AnnotationVisitor.

    public void visitAnnotation(Clazz clazz, Annotation annotation)
    {
        throw new UnsupportedOperationException("Method must be overridden in ["+this.getClass().getName()+"] if ever called");
    }


    /**
     * Visits the given Annotation of any type of class member.
     */
    public void visitAnnotation(Clazz clazz, Member member, Annotation annotation)
    {
        visitAnnotation(clazz, annotation);
    }


    public void visitAnnotation(Clazz clazz, Field field, Annotation annotation)
    {
        visitAnnotation(clazz, (Member)field, annotation);
    }


    public void visitAnnotation(Clazz clazz, Method method, Annotation annotation)
    {
        visitAnnotation(clazz, (Member)method, annotation);
    }


    public void visitAnnotation(Clazz clazz, Method method, int parameterIndex, Annotation annotation)
    {
        visitAnnotation(clazz, method, annotation);
    }


    public void visitAnnotation(Clazz clazz, Method method, CodeAttribute codeAttribute, Annotation annotation)
    {
        visitAnnotation(clazz, method, annotation);
    }


    // Simplifications for TypeAnnotationVisitor.

    public void visitTypeAnnotation(Clazz clazz, TypeAnnotation typeAnnotation)
    {
        throw new UnsupportedOperationException("Method must be overridden in ["+this.getClass().getName()+"] if ever called");
    }


    /**
     * Visits the given TypeAnnotation of any type of class member.
     */
    public void visitTypeAnnotation(Clazz clazz, Member member, TypeAnnotation typeAnnotation)
    {
        visitTypeAnnotation(clazz, typeAnnotation);
    }


    public void visitTypeAnnotation(Clazz clazz, Field field, TypeAnnotation typeAnnotation)
    {
        visitTypeAnnotation(clazz, (Member)field, typeAnnotation);
    }


    public void visitTypeAnnotation(Clazz clazz, Method method, TypeAnnotation typeAnnotation)
    {
        visitTypeAnnotation(clazz, (Member)method, typeAnnotation);
    }


    public void visitTypeAnnotation(Clazz clazz, Method method, int parameterIndex, TypeAnnotation typeAnnotation)
    {
        visitTypeAnnotation(clazz, method, typeAnnotation);
    }


    public void visitTypeAnnotation(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation)
    {
        visitTypeAnnotation(clazz, method, typeAnnotation);
    }


    // Simplifications for TargetInfoVisitor.

    /**
     * Visits any type of TargetInfo.
     */
    public void visitAnyTargetInfo(Clazz clazz, TypeAnnotation typeAnnotation, TargetInfo targetInfo)
    {
        throw new UnsupportedOperationException("Method must be overridden in ["+this.getClass().getName()+"] if ever called");
    }


    public void visitTypeParameterTargetInfo(Clazz clazz, TypeAnnotation typeAnnotation, TypeParameterTargetInfo typeParameterTargetInfo)
    {
        visitAnyTargetInfo(clazz, typeAnnotation, typeParameterTargetInfo);
    }


    public void visitTypeParameterTargetInfo(Clazz clazz, Method method, TypeAnnotation typeAnnotation, TypeParameterTargetInfo typeParameterTargetInfo)
    {
        visitTypeParameterTargetInfo(clazz, typeAnnotation, typeParameterTargetInfo);
    }


    public void visitSuperTypeTargetInfo(Clazz clazz, TypeAnnotation typeAnnotation, SuperTypeTargetInfo superTypeTargetInfo)
    {
        visitAnyTargetInfo(clazz, typeAnnotation, superTypeTargetInfo);
    }


    public void visitTypeParameterBoundTargetInfo(Clazz clazz, TypeAnnotation typeAnnotation, TypeParameterBoundTargetInfo typeParameterBoundTargetInfo)
    {
        visitAnyTargetInfo(clazz, typeAnnotation, typeParameterBoundTargetInfo);
    }


    /**
     * Visits the given TypeParameterBoundTargetInfo of any type of class member.
     */
    public void visitTypeParameterBoundTargetInfo(Clazz clazz, Member member, TypeAnnotation typeAnnotation, TypeParameterBoundTargetInfo typeParameterBoundTargetInfo)
    {
        visitTypeParameterBoundTargetInfo(clazz, typeAnnotation, typeParameterBoundTargetInfo);
    }


    public void visitTypeParameterBoundTargetInfo(Clazz clazz, Field field, TypeAnnotation typeAnnotation, TypeParameterBoundTargetInfo typeParameterBoundTargetInfo)
    {
        visitTypeParameterBoundTargetInfo(clazz, (Member)field, typeAnnotation, typeParameterBoundTargetInfo);
    }


    public void visitTypeParameterBoundTargetInfo(Clazz clazz, Method method, TypeAnnotation typeAnnotation, TypeParameterBoundTargetInfo typeParameterBoundTargetInfo)
    {
        visitTypeParameterBoundTargetInfo(clazz, (Member)method, typeAnnotation, typeParameterBoundTargetInfo);
    }


    /**
     * Visits the given EmptyTargetInfo of any type of class member.
     */
    public void visitEmptyTargetInfo(Clazz clazz, Member member, TypeAnnotation typeAnnotation, EmptyTargetInfo emptyTargetInfo)
    {
        visitAnyTargetInfo(clazz, typeAnnotation, emptyTargetInfo);
    }


    public void visitEmptyTargetInfo(Clazz clazz, Field field, TypeAnnotation typeAnnotation, EmptyTargetInfo emptyTargetInfo)
    {
        visitEmptyTargetInfo(clazz, (Member)field, typeAnnotation, emptyTargetInfo);
    }


    public void visitEmptyTargetInfo(Clazz clazz, Method method, TypeAnnotation typeAnnotation, EmptyTargetInfo emptyTargetInfo)
    {
        visitEmptyTargetInfo(clazz, (Member)method, typeAnnotation, emptyTargetInfo);
    }


    public void visitFormalParameterTargetInfo(Clazz clazz, Method method, TypeAnnotation typeAnnotation, FormalParameterTargetInfo formalParameterTargetInfo)
    {
        visitAnyTargetInfo(clazz, typeAnnotation, formalParameterTargetInfo);
    }


    public void visitThrowsTargetInfo(Clazz clazz, Method method, TypeAnnotation typeAnnotation, ThrowsTargetInfo throwsTargetInfo)
    {
        visitAnyTargetInfo(clazz, typeAnnotation, throwsTargetInfo);
    }


    public void visitLocalVariableTargetInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, LocalVariableTargetInfo localVariableTargetInfo)
    {
        visitAnyTargetInfo(clazz, typeAnnotation, localVariableTargetInfo);
    }


    public void visitCatchTargetInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, CatchTargetInfo catchTargetInfo)
    {
        visitAnyTargetInfo(clazz, typeAnnotation, catchTargetInfo);
    }


    public void visitOffsetTargetInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, OffsetTargetInfo offsetTargetInfo)
    {
        visitAnyTargetInfo(clazz, typeAnnotation, offsetTargetInfo);
    }


    public void visitTypeArgumentTargetInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, TypeArgumentTargetInfo typeArgumentTargetInfo)
    {
        visitAnyTargetInfo(clazz, typeAnnotation, typeArgumentTargetInfo);
    }


    // Simplifications for TypePathInfoVisitor.

    public void visitTypePathInfo(Clazz clazz, TypeAnnotation typeAnnotation, TypePathInfo typePathInfo)
    {
        throw new UnsupportedOperationException("Method must be overridden in ["+this.getClass().getName()+"] if ever called");
    }


    /**
     * Visits the given TypePathInfo of any type of class member.
     */
    public void visitTypePathInfo(Clazz clazz, Member member, TypeAnnotation typeAnnotation, TypePathInfo typePathInfo)
    {
        visitTypePathInfo(clazz, typeAnnotation, typePathInfo);
    }


    public void visitTypePathInfo(Clazz clazz, Field field, TypeAnnotation typeAnnotation, TypePathInfo typePathInfo)
    {
        visitTypePathInfo(clazz, (Member)field, typeAnnotation, typePathInfo);
    }


    public void visitTypePathInfo(Clazz clazz, Method method, TypeAnnotation typeAnnotation, TypePathInfo typePathInfo)
    {
        visitTypePathInfo(clazz, (Member)method, typeAnnotation, typePathInfo);
    }


    public void visitTypePathInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, TypePathInfo typePathInfo)
    {
        visitTypePathInfo(clazz, method, typeAnnotation, typePathInfo);
    }


    // Simplifications for ElementValueVisitor.

    /**
     * Visits any type of ElementValue.
     */
    public void visitAnyElementValue(Clazz clazz, Annotation annotation, ElementValue elementValue)
    {
        throw new UnsupportedOperationException("Method must be overridden in ["+this.getClass().getName()+"] if ever called");
    }


    public void visitConstantElementValue(Clazz clazz, Annotation annotation, ConstantElementValue constantElementValue)
    {
        visitAnyElementValue(clazz, annotation, constantElementValue);
    }


    public void visitEnumConstantElementValue(Clazz clazz, Annotation annotation, EnumConstantElementValue enumConstantElementValue)
    {
        visitAnyElementValue(clazz, annotation, enumConstantElementValue);
    }


    public void visitClassElementValue(Clazz clazz, Annotation annotation, ClassElementValue classElementValue)
    {
        visitAnyElementValue(clazz, annotation, classElementValue);
    }


    public void visitAnnotationElementValue(Clazz clazz, Annotation annotation, AnnotationElementValue annotationElementValue)
    {
        visitAnyElementValue(clazz, annotation, annotationElementValue);
    }


    public void visitArrayElementValue(Clazz clazz, Annotation annotation, ArrayElementValue arrayElementValue)
    {
        visitAnyElementValue(clazz, annotation, arrayElementValue);
    }
}

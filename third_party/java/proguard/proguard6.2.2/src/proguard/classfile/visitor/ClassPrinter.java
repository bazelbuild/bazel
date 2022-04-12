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
package proguard.classfile.visitor;

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
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;

import java.io.PrintWriter;


/**
 * This <code>ClassVisitor</code> prints out the complete internal
 * structure of the classes it visits.
 *
 * @author Eric Lafortune
 */
public class ClassPrinter
extends      SimplifiedVisitor
implements   ClassVisitor,
             ConstantVisitor,
             MemberVisitor,
             AttributeVisitor,
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
             LocalVariableTargetElementVisitor,
             TypePathInfoVisitor,
             ElementValueVisitor,
             InstructionVisitor
{
    private static final String INDENTATION = "  ";

    private final PrintWriter pw;

    private int indentation;


    /**
     * Creates a new ClassPrinter that prints to the standard output.
     */
    public ClassPrinter()
    {
        // We're using the system's default character encoding for writing to
        // the standard output.
        this(new PrintWriter(System.out, true));
    }


    /**
     * Creates a new ClassPrinter that prints to the given writer.
     */
    public ClassPrinter(PrintWriter printWriter)
    {
        pw = printWriter;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        println("_____________________________________________________________________");
        println(visitorInfo(programClass) + " " +
                "Program class: " + programClass.getName());
        indent();
        println("Superclass:    " + programClass.getSuperName());
        println("Major version: 0x" + Integer.toHexString(ClassUtil.internalMajorClassVersion(programClass.u4version)));
        println("Minor version: 0x" + Integer.toHexString(ClassUtil.internalMinorClassVersion(programClass.u4version)));
        println("  = target " + ClassUtil.externalClassVersion(programClass.u4version));
        println("Access flags:  0x" + Integer.toHexString(programClass.u2accessFlags));
        println("  = " +
                ClassUtil.externalClassAccessFlags(programClass.u2accessFlags) +
                ((programClass.u2accessFlags & (ClassConstants.ACC_ENUM      |
                                                ClassConstants.ACC_INTERFACE |
                                                ClassConstants.ACC_MODULE)) == 0 ? "class " : "") +
                ClassUtil.externalClassName(programClass.getName()) +
                (programClass.u2superClass == 0 ? "" : " extends " +
                ClassUtil.externalClassName(programClass.getSuperName())));
        outdent();
        println();

        println("Interfaces (count = " + programClass.u2interfacesCount + "):");
        indent();
        programClass.interfaceConstantsAccept(this);
        outdent();
        println();

        println("Constant Pool (count = " + programClass.u2constantPoolCount + "):");
        indent();
        programClass.constantPoolEntriesAccept(this);
        outdent();
        println();

        println("Fields (count = " + programClass.u2fieldsCount + "):");
        indent();
        programClass.fieldsAccept(this);
        outdent();
        println();

        println("Methods (count = " + programClass.u2methodsCount + "):");
        indent();
        programClass.methodsAccept(this);
        outdent();
        println();

        println("Class file attributes (count = " + programClass.u2attributesCount + "):");
        indent();
        programClass.attributesAccept(this);
        outdent();
        println();
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        println("_____________________________________________________________________");
        println(visitorInfo(libraryClass) + " " +
                "Library class: " + libraryClass.getName());
        indent();
        println("Superclass:    " + libraryClass.getSuperName());
        println("Access flags:  0x" + Integer.toHexString(libraryClass.u2accessFlags));
        println("  = " +
                ClassUtil.externalClassAccessFlags(libraryClass.u2accessFlags) +
                ((libraryClass.u2accessFlags & (ClassConstants.ACC_ENUM      |
                                                ClassConstants.ACC_INTERFACE |
                                                ClassConstants.ACC_MODULE)) == 0 ? "class " : "") +
                ClassUtil.externalClassName(libraryClass.getName()) +
                (libraryClass.getSuperName() == null ? "" : " extends "  +
                ClassUtil.externalClassName(libraryClass.getSuperName())));
        outdent();
        println();

        println("Interfaces (count = " + libraryClass.interfaceClasses.length + "):");
        for (int index = 0; index < libraryClass.interfaceClasses.length; index++)
        {
            Clazz interfaceClass = libraryClass.interfaceClasses[index];
            if (interfaceClass != null)
            {
                println("  + " + interfaceClass.getName());
            }
        }

        println("Fields (count = " + libraryClass.fields.length + "):");
        libraryClass.fieldsAccept(this);

        println("Methods (count = " + libraryClass.methods.length + "):");
        libraryClass.methodsAccept(this);
    }


    // Implementations for ConstantVisitor.

    public void visitIntegerConstant(Clazz clazz, IntegerConstant integerConstant)
    {
        println(visitorInfo(integerConstant) + " Integer [" +
                integerConstant.getValue() + "]");
    }


    public void visitLongConstant(Clazz clazz, LongConstant longConstant)
    {
        println(visitorInfo(longConstant) + " Long [" +
                longConstant.getValue() + "]");
    }


    public void visitFloatConstant(Clazz clazz, FloatConstant floatConstant)
    {
        println(visitorInfo(floatConstant) + " Float [" +
                floatConstant.getValue() + "]");
    }


    public void visitDoubleConstant(Clazz clazz, DoubleConstant doubleConstant)
    {
        println(visitorInfo(doubleConstant) + " Double [" +
                doubleConstant.getValue() + "]");
    }


    public void visitPrimitiveArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant)
    {
        println(visitorInfo(primitiveArrayConstant) + " PrimitiveArray " +
                primitiveArrayConstant.getPrimitiveType() + "[" +
                primitiveArrayConstant.getLength() + "]");
    }


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        println(visitorInfo(stringConstant) + " String [" +
                stringConstant.getString(clazz) + "]");
    }


    public void visitUtf8Constant(Clazz clazz, Utf8Constant utf8Constant)
    {
        println(visitorInfo(utf8Constant) + " Utf8 [" +
                utf8Constant.getString() + "]");
    }


    public void visitDynamicConstant(Clazz clazz, DynamicConstant dynamicConstant)
    {
        println(visitorInfo(dynamicConstant) + " Dynamic [bootstrap method index = " + dynamicConstant.u2bootstrapMethodAttributeIndex + "]:");

        indent();
        clazz.constantPoolEntryAccept(dynamicConstant.u2nameAndTypeIndex, this);
        outdent();
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        println(visitorInfo(invokeDynamicConstant) + " InvokeDynamic [bootstrap method index = " + invokeDynamicConstant.u2bootstrapMethodAttributeIndex + "]:");

        indent();
        clazz.constantPoolEntryAccept(invokeDynamicConstant.u2nameAndTypeIndex, this);
        outdent();
    }


    public void visitMethodHandleConstant(Clazz clazz, MethodHandleConstant methodHandleConstant)
    {
        println(visitorInfo(methodHandleConstant) + " MethodHandle [kind = " + methodHandleConstant.u1referenceKind + "]:");

        indent();
        clazz.constantPoolEntryAccept(methodHandleConstant.u2referenceIndex, this);
        outdent();
    }

    public void visitModuleConstant(Clazz clazz, ModuleConstant moduleConstant)
    {
        println(visitorInfo(moduleConstant) + " Module [" +
                moduleConstant.getName(clazz) + "]");
    }


    public void visitPackageConstant(Clazz clazz, PackageConstant packageConstant)
    {
        println(visitorInfo(packageConstant) + " Package [" +
                packageConstant.getName(clazz) + "]");
    }


    public void visitFieldrefConstant(Clazz clazz, FieldrefConstant fieldrefConstant)
    {
        println(visitorInfo(fieldrefConstant) + " Fieldref [" +
                clazz.getClassName(fieldrefConstant.u2classIndex) + "." +
                clazz.getName(fieldrefConstant.u2nameAndTypeIndex) + " " +
                clazz.getType(fieldrefConstant.u2nameAndTypeIndex) + "]");
    }


    public void visitInterfaceMethodrefConstant(Clazz clazz, InterfaceMethodrefConstant interfaceMethodrefConstant)
    {
        println(visitorInfo(interfaceMethodrefConstant) + " InterfaceMethodref [" +
                clazz.getClassName(interfaceMethodrefConstant.u2classIndex)  + "." +
                clazz.getName(interfaceMethodrefConstant.u2nameAndTypeIndex) + " " +
                clazz.getType(interfaceMethodrefConstant.u2nameAndTypeIndex) + "]");
    }


    public void visitMethodrefConstant(Clazz clazz, MethodrefConstant methodrefConstant)
    {
        println(visitorInfo(methodrefConstant) + " Methodref [" +
                clazz.getClassName(methodrefConstant.u2classIndex)  + "." +
                clazz.getName(methodrefConstant.u2nameAndTypeIndex) + " " +
                clazz.getType(methodrefConstant.u2nameAndTypeIndex) + "]");
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        println(visitorInfo(classConstant) + " Class [" +
                classConstant.getName(clazz) + "]");
    }


    public void visitMethodTypeConstant(Clazz clazz, MethodTypeConstant methodTypeConstant)
    {
        println(visitorInfo(methodTypeConstant) + " MethodType [" +
                methodTypeConstant.getType(clazz) + "]");
    }


    public void visitNameAndTypeConstant(Clazz clazz, NameAndTypeConstant nameAndTypeConstant)
    {
        println(visitorInfo(nameAndTypeConstant) + " NameAndType [" +
                nameAndTypeConstant.getName(clazz) + " " +
                nameAndTypeConstant.getType(clazz) + "]");
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        println(visitorInfo(programField) + " " +
                "Field:        " +
                programField.getName(programClass) + " " +
                programField.getDescriptor(programClass));

        indent();
        println("Access flags: 0x" + Integer.toHexString(programField.u2accessFlags));
        println("  = " +
                ClassUtil.externalFullFieldDescription(programField.u2accessFlags,
                                                       programField.getName(programClass),
                                                       programField.getDescriptor(programClass)));

        visitMember(programClass, programField);
        outdent();
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        println(visitorInfo(programMethod) + " " +
                "Method:       " +
                programMethod.getName(programClass) +
                programMethod.getDescriptor(programClass));

        indent();
        println("Access flags: 0x" + Integer.toHexString(programMethod.u2accessFlags));
        println("  = " +
                ClassUtil.externalFullMethodDescription(programClass.getName(),
                                                        programMethod.u2accessFlags,
                                                        programMethod.getName(programClass),
                                                        programMethod.getDescriptor(programClass)));

        visitMember(programClass, programMethod);
        outdent();
    }


    private void visitMember(ProgramClass programClass, ProgramMember programMember)
    {
        if (programMember.u2attributesCount > 0)
        {
            println("Class member attributes (count = " + programMember.u2attributesCount + "):");
            programMember.attributesAccept(programClass, this);
        }
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
    {
        println(visitorInfo(libraryField) + " " +
                "Field:        " +
                libraryField.getName(libraryClass) + " " +
                libraryField.getDescriptor(libraryClass));

        indent();
        println("Access flags: 0x" + Integer.toHexString(libraryField.u2accessFlags));
        println("  = " +
                ClassUtil.externalFullFieldDescription(libraryField.u2accessFlags,
                                                       libraryField.getName(libraryClass),
                                                       libraryField.getDescriptor(libraryClass)));
        outdent();
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        println(visitorInfo(libraryMethod) + " " +
                "Method:       " +
                libraryMethod.getName(libraryClass) + " " +
                libraryMethod.getDescriptor(libraryClass));

        indent();
        println("Access flags: 0x" + Integer.toHexString(libraryMethod.u2accessFlags));
        println("  = " +
                ClassUtil.externalFullMethodDescription(libraryClass.getName(),
                                                        libraryMethod.u2accessFlags,
                                                        libraryMethod.getName(libraryClass),
                                                        libraryMethod.getDescriptor(libraryClass)));
        outdent();
    }


    // Implementations for AttributeVisitor.
    // Note that attributes are typically only referenced once, so we don't
    // test if they are marked already.

    public void visitUnknownAttribute(Clazz clazz, UnknownAttribute unknownAttribute)
    {
        println(visitorInfo(unknownAttribute) +
                " Unknown attribute (" + unknownAttribute.getAttributeName(clazz) + ")");
    }


    public void visitBootstrapMethodsAttribute(Clazz clazz, BootstrapMethodsAttribute bootstrapMethodsAttribute)
    {
        println(visitorInfo(bootstrapMethodsAttribute) +
                " Bootstrap methods attribute (count = " + bootstrapMethodsAttribute.u2bootstrapMethodsCount + "):");

        indent();
        bootstrapMethodsAttribute.bootstrapMethodEntriesAccept(clazz, this);
        outdent();
    }


    public void visitSourceFileAttribute(Clazz clazz, SourceFileAttribute sourceFileAttribute)
    {
        println(visitorInfo(sourceFileAttribute) +
                " Source file attribute:");

        indent();
        clazz.constantPoolEntryAccept(sourceFileAttribute.u2sourceFileIndex, this);
        outdent();
    }


    public void visitSourceDirAttribute(Clazz clazz, SourceDirAttribute sourceDirAttribute)
    {
        println(visitorInfo(sourceDirAttribute) +
                " Source dir attribute:");

        indent();
        clazz.constantPoolEntryAccept(sourceDirAttribute.u2sourceDirIndex, this);
        outdent();
    }


    public void visitInnerClassesAttribute(Clazz clazz, InnerClassesAttribute innerClassesAttribute)
    {
        println(visitorInfo(innerClassesAttribute) +
                " Inner classes attribute (count = " + innerClassesAttribute.u2classesCount + "):");

        indent();
        innerClassesAttribute.innerClassEntriesAccept(clazz, this);
        outdent();
    }


    public void visitEnclosingMethodAttribute(Clazz clazz, EnclosingMethodAttribute enclosingMethodAttribute)
    {
        println(visitorInfo(enclosingMethodAttribute) +
                " Enclosing method attribute:");

        indent();
        clazz.constantPoolEntryAccept(enclosingMethodAttribute.u2classIndex, this);

        if (enclosingMethodAttribute.u2nameAndTypeIndex != 0)
        {
            clazz.constantPoolEntryAccept(enclosingMethodAttribute.u2nameAndTypeIndex, this);
        }
        outdent();
    }


    public void visitNestHostAttribute(Clazz clazz, NestHostAttribute nestHostAttribute)
    {
        println(visitorInfo(nestHostAttribute) +
                " Nest host attribute:");

        indent();
        clazz.constantPoolEntryAccept(nestHostAttribute.u2hostClassIndex, this);
        outdent();
    }


    public void visitNestMembersAttribute(Clazz clazz, NestMembersAttribute nestMembersAttribute)
    {
        println(visitorInfo(nestMembersAttribute) +
                " Nest members attribute:");

        indent();
        nestMembersAttribute.memberClassConstantsAccept(clazz, this);
        outdent();
    }


    public void visitModuleAttribute(Clazz clazz, ModuleAttribute moduleAttribute)
    {
        println(visitorInfo(moduleAttribute) +
                " Module attribute:");

        indent();
        clazz.constantPoolEntryAccept(moduleAttribute.u2moduleNameIndex, this);
        println("Access flags:  0x" +
                Integer.toHexString(moduleAttribute.u2moduleFlags) +
                " = " +
                ClassUtil.externalModuleAccessFlags(moduleAttribute.u2moduleFlags));

        if (moduleAttribute.u2moduleVersionIndex != 0)
        {
            clazz.constantPoolEntryAccept(moduleAttribute.u2moduleVersionIndex, this);
        }
        println("Requires:");
        moduleAttribute.requiresAccept(clazz, this);
        println("Exports:");
        moduleAttribute.exportsAccept(clazz, this);
        println("Opens:");
        moduleAttribute.opensAccept(clazz, this);
        println("Uses services:");

        for (int index = 0; index < moduleAttribute.u2usesCount; index++)
        {
            clazz.constantPoolEntryAccept(moduleAttribute.u2uses[index], this);
        }

        println("Provides services:");
        moduleAttribute.providesAccept(clazz, this);
        outdent();
    }


    public void visitModuleMainClassAttribute(Clazz clazz, ModuleMainClassAttribute moduleMainClassAttribute)
    {
        println(visitorInfo(moduleMainClassAttribute) +
                " Module main class attribute:");

        indent();
        clazz.constantPoolEntryAccept(moduleMainClassAttribute.u2mainClass, this);
        outdent();
    }


    public void visitModulePackagesAttribute(Clazz clazz, ModulePackagesAttribute modulePackagesAttribute)
    {
        println(visitorInfo(modulePackagesAttribute) +
                " Module packages attribute (count = " + modulePackagesAttribute.u2packagesCount + "):");

        indent();
        modulePackagesAttribute.packagesAccept(clazz, this);
        outdent();
    }


    public void visitDeprecatedAttribute(Clazz clazz, DeprecatedAttribute deprecatedAttribute)
    {
        println(visitorInfo(deprecatedAttribute) +
                " Deprecated attribute");
    }


    public void visitSyntheticAttribute(Clazz clazz, SyntheticAttribute syntheticAttribute)
    {
        println(visitorInfo(syntheticAttribute) +
                " Synthetic attribute");
    }


    public void visitSignatureAttribute(Clazz clazz, SignatureAttribute signatureAttribute)
    {
        println(visitorInfo(signatureAttribute) +
                " Signature attribute:");

        indent();
        clazz.constantPoolEntryAccept(signatureAttribute.u2signatureIndex, this);
        outdent();
    }


    public void visitConstantValueAttribute(Clazz clazz, Field field, ConstantValueAttribute constantValueAttribute)
    {
        println(visitorInfo(constantValueAttribute) +
                " Constant value attribute:");

        clazz.constantPoolEntryAccept(constantValueAttribute.u2constantValueIndex, this);
    }


    public void visitMethodParametersAttribute(Clazz clazz, Method method, MethodParametersAttribute methodParametersAttribute)
    {
        println(visitorInfo(methodParametersAttribute) +
                " Method parameters attribute (count = " + methodParametersAttribute.u1parametersCount + "):");

        indent();
        methodParametersAttribute.parametersAccept(clazz, method, this);
        outdent();
    }


    public void visitExceptionsAttribute(Clazz clazz, Method method, ExceptionsAttribute exceptionsAttribute)
    {
        println(visitorInfo(exceptionsAttribute) +
                " Exceptions attribute (count = " + exceptionsAttribute.u2exceptionIndexTableLength + "):");

        indent();
        exceptionsAttribute.exceptionEntriesAccept(clazz, this);
        outdent();
    }


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        println(visitorInfo(codeAttribute) +
                " Code attribute instructions (code length = "+ codeAttribute.u4codeLength +
                ", locals = "+ codeAttribute.u2maxLocals +
                ", stack = "+ codeAttribute.u2maxStack + "):");

        indent();

        codeAttribute.instructionsAccept(clazz, method, this);

        println("Code attribute exceptions (count = " +
                codeAttribute.u2exceptionTableLength + "):");

        codeAttribute.exceptionsAccept(clazz, method, this);

        println("Code attribute attributes (attribute count = " +
                codeAttribute.u2attributesCount + "):");

        codeAttribute.attributesAccept(clazz, method, this);

        outdent();
    }


    public void visitStackMapAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapAttribute stackMapAttribute)
    {
        println(visitorInfo(codeAttribute) +
                " Stack map attribute (count = "+
                stackMapAttribute.u2stackMapFramesCount + "):");

        indent();
        stackMapAttribute.stackMapFramesAccept(clazz, method, codeAttribute, this);
        outdent();
    }


    public void visitStackMapTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapTableAttribute stackMapTableAttribute)
    {
        println(visitorInfo(codeAttribute) +
                " Stack map table attribute (count = "+
                stackMapTableAttribute.u2stackMapFramesCount + "):");

        indent();
        stackMapTableAttribute.stackMapFramesAccept(clazz, method, codeAttribute, this);
        outdent();
    }


    public void visitLineNumberTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute lineNumberTableAttribute)
    {
        println(visitorInfo(lineNumberTableAttribute) +
                " Line number table attribute (count = " +
                lineNumberTableAttribute.u2lineNumberTableLength + "):");

        indent();
        lineNumberTableAttribute.lineNumbersAccept(clazz, method, codeAttribute, this);
        outdent();
    }


    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        println(visitorInfo(localVariableTableAttribute) +
                " Local variable table attribute (count = " +
                localVariableTableAttribute.u2localVariableTableLength + "):");

        indent();
        localVariableTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
        outdent();
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        println(visitorInfo(localVariableTypeTableAttribute) +
                " Local variable type table attribute (count = "+
                localVariableTypeTableAttribute.u2localVariableTypeTableLength + "):");

        indent();
        localVariableTypeTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
        outdent();
    }


    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        println(visitorInfo(runtimeVisibleAnnotationsAttribute) +
                " Runtime visible annotations attribute:");

        indent();
        runtimeVisibleAnnotationsAttribute.annotationsAccept(clazz, this);
        outdent();
    }


    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        println(visitorInfo(runtimeInvisibleAnnotationsAttribute) +
                " Runtime invisible annotations attribute:");

        indent();
        runtimeInvisibleAnnotationsAttribute.annotationsAccept(clazz, this);
        outdent();
    }


    public void visitRuntimeVisibleParameterAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleParameterAnnotationsAttribute runtimeVisibleParameterAnnotationsAttribute)
    {
        println(visitorInfo(runtimeVisibleParameterAnnotationsAttribute) +
                " Runtime visible parameter annotations attribute (parameter count = " + runtimeVisibleParameterAnnotationsAttribute.u1parametersCount + "):");

        indent();
        runtimeVisibleParameterAnnotationsAttribute.annotationsAccept(clazz, method, this);
        outdent();
    }


    public void visitRuntimeInvisibleParameterAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleParameterAnnotationsAttribute runtimeInvisibleParameterAnnotationsAttribute)
    {
        println(visitorInfo(runtimeInvisibleParameterAnnotationsAttribute) +
                " Runtime invisible parameter annotations attribute (parameter count = " + runtimeInvisibleParameterAnnotationsAttribute.u1parametersCount + "):");

        indent();
        runtimeInvisibleParameterAnnotationsAttribute.annotationsAccept(clazz, method, this);
        outdent();
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        println(visitorInfo(runtimeVisibleTypeAnnotationsAttribute) +
                " Runtime visible type annotations attribute");

        indent();
        runtimeVisibleTypeAnnotationsAttribute.typeAnnotationsAccept(clazz, this);
        outdent();
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        println(visitorInfo(runtimeInvisibleTypeAnnotationsAttribute) +
                " Runtime invisible type annotations attribute");

        indent();
        runtimeInvisibleTypeAnnotationsAttribute.typeAnnotationsAccept(clazz, this);
        outdent();
    }


    public void visitAnnotationDefaultAttribute(Clazz clazz, Method method, AnnotationDefaultAttribute annotationDefaultAttribute)
    {
        println(visitorInfo(annotationDefaultAttribute) +
                " Annotation default attribute:");

        indent();
        annotationDefaultAttribute.defaultValueAccept(clazz, this);
        outdent();
    }


    // Implementations for BootstrapMethodInfoVisitor.

    public void visitBootstrapMethodInfo(Clazz clazz, BootstrapMethodInfo bootstrapMethodInfo)
    {
        println(visitorInfo(bootstrapMethodInfo) +
                " BootstrapMethodInfo (argument count = " +
                bootstrapMethodInfo.u2methodArgumentCount+ "):");

        indent();
        clazz.constantPoolEntryAccept(bootstrapMethodInfo.u2methodHandleIndex, this);
        bootstrapMethodInfo.methodArgumentsAccept(clazz, this);
        outdent();
    }


    // Implementations for InnerClassesInfoVisitor.

    public void visitInnerClassesInfo(Clazz clazz, InnerClassesInfo innerClassesInfo)
    {
        println(visitorInfo(innerClassesInfo) +
                " InnerClassesInfo:");

        indent();
        println("Access flags:  0x" + Integer.toHexString(innerClassesInfo.u2innerClassAccessFlags) + " = " +
                ClassUtil.externalClassAccessFlags(innerClassesInfo.u2innerClassAccessFlags));
        innerClassesInfo.innerClassConstantAccept(clazz, this);
        innerClassesInfo.outerClassConstantAccept(clazz, this);
        innerClassesInfo.innerNameConstantAccept(clazz, this);
        outdent();
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
    {
        println(instruction.toString(offset));
    }


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        println(constantInstruction.toString(offset));

        indent();
        clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
        outdent();
    }


    public void visitTableSwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, TableSwitchInstruction tableSwitchInstruction)
    {
        println(tableSwitchInstruction.toString(offset));

        indent();

        int[] jumpOffsets = tableSwitchInstruction.jumpOffsets;

        for (int index = 0; index < jumpOffsets.length; index++)
        {
            int jumpOffset = jumpOffsets[index];
            println(Integer.toString(tableSwitchInstruction.lowCase + index)  + ": offset = " + jumpOffset + ", target = " + (offset + jumpOffset));
        }

        int defaultOffset = tableSwitchInstruction.defaultOffset;
        println("default: offset = " + defaultOffset + ", target = "+ (offset + defaultOffset));

        outdent();
    }


    public void visitLookUpSwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, LookUpSwitchInstruction lookUpSwitchInstruction)
    {
        println(lookUpSwitchInstruction.toString(offset));

        indent();

        int[] cases       = lookUpSwitchInstruction.cases;
        int[] jumpOffsets = lookUpSwitchInstruction.jumpOffsets;

        for (int index = 0; index < jumpOffsets.length; index++)
        {
            int jumpOffset = jumpOffsets[index];
            println(Integer.toString(cases[index])  + ": offset = " + jumpOffset + ", target = " + (offset + jumpOffset));
        }

        int defaultOffset = lookUpSwitchInstruction.defaultOffset;
        println("default: offset = " + defaultOffset + ", target = "+ (offset + defaultOffset));

        outdent();
    }


    // Implementations for ExceptionInfoVisitor.

    public void visitExceptionInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, ExceptionInfo exceptionInfo)
    {
        println(visitorInfo(exceptionInfo) +
                " ExceptionInfo (" +
                exceptionInfo.u2startPC + " -> " +
                exceptionInfo.u2endPC + ": " +
                exceptionInfo.u2handlerPC + "):");

        if (exceptionInfo.u2catchType != 0)
        {
            clazz.constantPoolEntryAccept(exceptionInfo.u2catchType, this);
        }
    }


    // Implementations for StackMapFrameVisitor.

    public void visitSameZeroFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SameZeroFrame sameZeroFrame)
    {
        println(visitorInfo(sameZeroFrame) +
                " [" + offset  + "]" +
                " Var: ..., Stack: (empty)");
    }


    public void visitSameOneFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SameOneFrame sameOneFrame)
    {
        print(visitorInfo(sameOneFrame) +
              " [" + offset  + "]" +
              " Var: ..., Stack: ");

        sameOneFrame.stackItemAccept(clazz, method, codeAttribute, offset, this);

        println();
    }


    public void visitLessZeroFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, LessZeroFrame lessZeroFrame)
    {
        println(visitorInfo(lessZeroFrame) +
                " [" + offset  + "]" +
                " Var: -" + lessZeroFrame.choppedVariablesCount +
                ", Stack: (empty)");
    }


    public void visitMoreZeroFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, MoreZeroFrame moreZeroFrame)
    {
        print(visitorInfo(moreZeroFrame) +
              " [" + offset  + "]" +
              " Var: ...");

        moreZeroFrame.additionalVariablesAccept(clazz, method, codeAttribute, offset, this);

        pw.println(", Stack: (empty)");
    }


    public void visitFullFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, FullFrame fullFrame)
    {
        print(visitorInfo(fullFrame) +
              " [" + offset  + "]" +
              " Var: ");

        fullFrame.variablesAccept(clazz, method, codeAttribute, offset, this);

        pw.print(", Stack: ");

        fullFrame.stackAccept(clazz, method, codeAttribute, offset, this);

        println();
    }


    // Implementations for VerificationTypeVisitor.

    public void visitIntegerType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, IntegerType integerType)
    {
        pw.print("[i]");
    }


    public void visitFloatType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, FloatType floatType)
    {
        pw.print("[f]");
    }


    public void visitLongType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, LongType longType)
    {
        pw.print("[l]");
    }


    public void visitDoubleType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, DoubleType doubleType)
    {
        pw.print("[d]");
    }


    public void visitTopType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, TopType topType)
    {
        pw.print("[T]");
    }


    public void visitObjectType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ObjectType objectType)
    {
        pw.print("[a:" + clazz.getClassName(objectType.u2classIndex) + "]");
    }


    public void visitNullType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, NullType nullType)
    {
        pw.print("[n]");
    }


    public void visitUninitializedType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, UninitializedType uninitializedType)
    {
        pw.print("[u:" + uninitializedType.u2newInstructionOffset + "]");
    }


    public void visitUninitializedThisType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, UninitializedThisType uninitializedThisType)
    {
        pw.print("[u:this]");
    }


    // Implementations for LineNumberInfoVisitor.

    public void visitLineNumberInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberInfo lineNumberInfo)
    {
        println("[" + lineNumberInfo.u2startPC + "] -> line " +
                lineNumberInfo.u2lineNumber +
                (lineNumberInfo.getSource() == null ? "" : " [" + lineNumberInfo.getSource() + "]"));
    }


    // Implementations for ParameterInfoVisitor.

    public void visitParameterInfo(Clazz clazz, Method method, int parameterIndex, ParameterInfo parameterInfo)
    {
        println("p" + parameterIndex + ":  Access flags: 0x" + Integer.toHexString(parameterInfo.u2accessFlags) + " = " +
                ClassUtil.externalParameterAccessFlags(parameterInfo.u2accessFlags) + " [" +
                (parameterInfo.u2nameIndex == 0 ? "" : parameterInfo.getName(clazz)) + "]");
    }


    // Implementations for LocalVariableInfoVisitor.

    public void visitLocalVariableInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableInfo localVariableInfo)
    {
        println("v" + localVariableInfo.u2index + ": " +
                localVariableInfo.u2startPC + " -> " +
                (localVariableInfo.u2startPC + localVariableInfo.u2length) + " [" +
                localVariableInfo.getDescriptor(clazz) + " " +
                localVariableInfo.getName(clazz) + "]");
    }


    // Implementations for LocalVariableTypeInfoVisitor.

    public void visitLocalVariableTypeInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeInfo localVariableTypeInfo)
    {
        println("v" + localVariableTypeInfo.u2index + ": " +
                localVariableTypeInfo.u2startPC + " -> " +
                (localVariableTypeInfo.u2startPC + localVariableTypeInfo.u2length) + " [" +
                localVariableTypeInfo.getSignature(clazz) + " " +
                localVariableTypeInfo.getName(clazz) + "]");
    }


    // Implementations for RequiresInfoVisitor

    public void visitRequiresInfo(Clazz clazz, RequiresInfo requiresInfo)
    {
        println(visitorInfo(requiresInfo) +
                " RequiresInfo:");

        indent();
        clazz.constantPoolEntryAccept(requiresInfo.u2requiresIndex, this);
        println("Access flags:  0x" + Integer.toHexString(requiresInfo.u2requiresFlags) + " = " +
                ClassUtil.externalRequiresAccessFlags(requiresInfo.u2requiresFlags));
        clazz.constantPoolEntryAccept(requiresInfo.u2requiresVersionIndex, this);
        outdent();
    }


    // Implementations for ExportsInfoVisitor

    public void visitExportsInfo(Clazz clazz, ExportsInfo exportsInfo)
    {
        println(visitorInfo(exportsInfo) +
                " ExportsInfo (targets count = " +
                exportsInfo.u2exportsToCount + "):");

        indent();
        clazz.constantPoolEntryAccept(exportsInfo.u2exportsIndex, this);
        println("Access flags:  0x" + Integer.toHexString(exportsInfo.u2exportsFlags) + " = " +
                ClassUtil.externalExportsAccessFlags(exportsInfo.u2exportsFlags));

        for (int index = 0; index < exportsInfo.u2exportsToCount; index++)
        {
            clazz.constantPoolEntryAccept(exportsInfo.u2exportsToIndex[index], this);
        }

        outdent();
    }


    // Implementations for ExportsOpensVisitor

    public void visitOpensInfo(Clazz clazz, OpensInfo opensInfo)
    {
        println(visitorInfo(opensInfo) +
                " OpensInfo (targets count = " +
                opensInfo.u2opensToCount + "):");

        indent();
        clazz.constantPoolEntryAccept(opensInfo.u2opensIndex, this);
        println("Access flags:  0x" + Integer.toHexString(opensInfo.u2opensFlags) + " = " +
                ClassUtil.externalOpensAccessFlags(opensInfo.u2opensFlags));

        for (int index = 0; index < opensInfo.u2opensToCount; index++)
        {
            clazz.constantPoolEntryAccept(opensInfo.u2opensToIndex[index], this);
        }

        outdent();
    }


    // Implementations for ProvidesInfoVisitor

    public void visitProvidesInfo(Clazz clazz, ProvidesInfo providesInfo)
    {
        println(visitorInfo(providesInfo) +
                " ProvidesInfo (with count = " +
                providesInfo.u2providesWithCount + "):");

        indent();
        clazz.constantPoolEntryAccept(providesInfo.u2providesIndex, this);

        for (int index = 0; index < providesInfo.u2providesWithCount; index++)
        {
            clazz.constantPoolEntryAccept(providesInfo.u2providesWithIndex[index], this);
        }

        outdent();
    }


    // Implementations for AnnotationVisitor.

    public void visitAnnotation(Clazz clazz, Annotation annotation)
    {
        println(visitorInfo(annotation) +
                " Annotation [" + annotation.getType(clazz) + "]:");

        indent();
        annotation.elementValuesAccept(clazz, this);
        outdent();
    }


    public void visitAnnotation(Clazz clazz, Method method, int parameterIndex, Annotation annotation)
    {
        println(visitorInfo(annotation) +
                " Parameter #"+parameterIndex+", annotation [" + annotation.getType(clazz) + "]:");

        indent();
        annotation.elementValuesAccept(clazz, this);
        outdent();
    }


    // Implementations for TypeAnnotationVisitor.

    public void visitTypeAnnotation(Clazz clazz, TypeAnnotation typeAnnotation)
    {
        println(visitorInfo(typeAnnotation) +
                " Type annotation [" + typeAnnotation.getType(clazz) + "]:");

        indent();
        typeAnnotation.targetInfoAccept(clazz, this);

        println("Type path (count = " + typeAnnotation.typePath.length + "):");
        indent();
        typeAnnotation.typePathInfosAccept(clazz, this);
        outdent();

        typeAnnotation.elementValuesAccept(clazz, this);

        outdent();
    }


    // Implementations for TargetInfoVisitor.

    public void visitTypeParameterTargetInfo(Clazz clazz, TypeAnnotation typeAnnotation, TypeParameterTargetInfo typeParameterTargetInfo)
    {
        println("Target (type = 0x" + Integer.toHexString(typeParameterTargetInfo.u1targetType) + "): Parameter #" +
                typeParameterTargetInfo.u1typeParameterIndex);
    }


    public void visitSuperTypeTargetInfo(Clazz clazz, TypeAnnotation typeAnnotation, SuperTypeTargetInfo superTypeTargetInfo)
    {
        println("Target (type = 0x" + Integer.toHexString(superTypeTargetInfo.u1targetType) + "): " +
                (superTypeTargetInfo.u2superTypeIndex == SuperTypeTargetInfo.EXTENDS_INDEX ?
                     "super class" :
                     "interface #" + superTypeTargetInfo.u2superTypeIndex));
    }


    public void visitTypeParameterBoundTargetInfo(Clazz clazz, TypeAnnotation typeAnnotation, TypeParameterBoundTargetInfo typeParameterBoundTargetInfo)
    {
        println("Target (type = 0x" + Integer.toHexString(typeParameterBoundTargetInfo.u1targetType) + "): parameter #" +
                typeParameterBoundTargetInfo.u1typeParameterIndex + ", bound #" + typeParameterBoundTargetInfo.u1boundIndex);
    }


    public void visitEmptyTargetInfo(Clazz clazz, Member member, TypeAnnotation typeAnnotation, EmptyTargetInfo emptyTargetInfo)
    {
        println("Target (type = 0x" + Integer.toHexString(emptyTargetInfo.u1targetType) + ")");
    }


    public void visitFormalParameterTargetInfo(Clazz clazz, Method method, TypeAnnotation typeAnnotation, FormalParameterTargetInfo formalParameterTargetInfo)
    {
        println("Target (type = 0x" + Integer.toHexString(formalParameterTargetInfo.u1targetType) + "): formal parameter #" +
                formalParameterTargetInfo.u1formalParameterIndex);
    }


    public void visitThrowsTargetInfo(Clazz clazz, Method method, TypeAnnotation typeAnnotation, ThrowsTargetInfo throwsTargetInfo)
    {
        println("Target (type = 0x" + Integer.toHexString(throwsTargetInfo.u1targetType) + "): throws #" +
                throwsTargetInfo.u2throwsTypeIndex);
    }


    public void visitLocalVariableTargetInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, LocalVariableTargetInfo localVariableTargetInfo)
    {
        println("Target (type = 0x" + Integer.toHexString(localVariableTargetInfo.u1targetType) + "): local variables (count = " +
                localVariableTargetInfo.u2tableLength + ")");

        indent();
        localVariableTargetInfo.targetElementsAccept(clazz, method, codeAttribute, typeAnnotation, this);
        outdent();
    }


    public void visitCatchTargetInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, CatchTargetInfo catchTargetInfo)
    {
        println("Target (type = 0x" + Integer.toHexString(catchTargetInfo.u1targetType) + "): catch #" +
                catchTargetInfo.u2exceptionTableIndex);
    }


    public void visitOffsetTargetInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, OffsetTargetInfo offsetTargetInfo)
    {
        println("Target (type = 0x" + Integer.toHexString(offsetTargetInfo.u1targetType) + "): offset " +
                offsetTargetInfo.u2offset);
    }


    public void visitTypeArgumentTargetInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, TypeArgumentTargetInfo typeArgumentTargetInfo)
    {
        println("Target (type = 0x" + Integer.toHexString(typeArgumentTargetInfo.u1targetType) + "): offset " +
                typeArgumentTargetInfo.u2offset + ", type argument " +
                typeArgumentTargetInfo.u1typeArgumentIndex);
    }


    // Implementations for TypePathInfoVisitor.

    public void visitTypePathInfo(Clazz clazz, TypeAnnotation typeAnnotation, TypePathInfo typePathInfo)
    {
        println("kind = " +
                typePathInfo.u1typePathKind + ", argument index = " +
                typePathInfo.u1typeArgumentIndex);
    }


    // Implementations for LocalVariableTargetElementVisitor.

    public void visitLocalVariableTargetElement(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, LocalVariableTargetInfo localVariableTargetInfo, LocalVariableTargetElement localVariableTargetElement)
    {
        println("v" +
                localVariableTargetElement.u2index + ": " +
                localVariableTargetElement.u2startPC + " -> " +
                (localVariableTargetElement.u2startPC + localVariableTargetElement.u2length));
    }


    // Implementations for ElementValueVisitor.

    public void visitConstantElementValue(Clazz clazz, Annotation annotation, ConstantElementValue constantElementValue)
    {
        println(visitorInfo(constantElementValue) +
                " Constant element value [" +
                (constantElementValue.u2elementNameIndex == 0 ? "(default)" :
                constantElementValue.getMethodName(clazz)) + " '" +
                constantElementValue.u1tag + "']");

        indent();
        clazz.constantPoolEntryAccept(constantElementValue.u2constantValueIndex, this);
        outdent();
    }


    public void visitEnumConstantElementValue(Clazz clazz, Annotation annotation, EnumConstantElementValue enumConstantElementValue)
    {
        println(visitorInfo(enumConstantElementValue) +
                " Enum constant element value [" +
                (enumConstantElementValue.u2elementNameIndex == 0 ? "(default)" :
                enumConstantElementValue.getMethodName(clazz)) + ", " +
                enumConstantElementValue.getTypeName(clazz)  + ", " +
                enumConstantElementValue.getConstantName(clazz) + "]");
    }


    public void visitClassElementValue(Clazz clazz, Annotation annotation, ClassElementValue classElementValue)
    {
        println(visitorInfo(classElementValue) +
                " Class element value [" +
                (classElementValue.u2elementNameIndex == 0 ? "(default)" :
                classElementValue.getMethodName(clazz)) + ", " +
                classElementValue.getClassName(clazz) + "]");
    }


    public void visitAnnotationElementValue(Clazz clazz, Annotation annotation, AnnotationElementValue annotationElementValue)
    {
        println(visitorInfo(annotationElementValue) +
                " Annotation element value [" +
                (annotationElementValue.u2elementNameIndex == 0 ? "(default)" :
                annotationElementValue.getMethodName(clazz)) + "]:");

        indent();
        annotationElementValue.annotationAccept(clazz, this);
        outdent();
    }


    public void visitArrayElementValue(Clazz clazz, Annotation annotation, ArrayElementValue arrayElementValue)
    {
        println(visitorInfo(arrayElementValue) +
                " Array element value [" +
                (arrayElementValue.u2elementNameIndex == 0 ? "(default)" :
                arrayElementValue.getMethodName(clazz)) + "]:");

        indent();
        arrayElementValue.elementValuesAccept(clazz, annotation, this);
        outdent();
    }


    // Small utility methods.

    private void indent()
    {
        indentation++;
    }

    private void outdent()
    {
        indentation--;
    }

    private void println(String string)
    {
        print(string);
        println();

    }

    private void print(String string)
    {
        for (int index = 0; index < indentation; index++)
        {
            pw.print(INDENTATION);
        }

        pw.print(string);
    }

    private void println()
    {
        pw.println();
    }


    private String visitorInfo(VisitorAccepter visitorAccepter)
    {
        return visitorAccepter.getVisitorInfo() == null ? "-" : "+";
    }
}

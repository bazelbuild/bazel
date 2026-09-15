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
import proguard.classfile.attribute.annotation.visitor.*;
import proguard.classfile.attribute.preverification.*;
import proguard.classfile.attribute.preverification.visitor.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.Constant;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This <code>ClassVisitor</code> removes all visitor information of the
 * classes it visits.
 *
 * @author Eric Lafortune
 */
public class ClassCleaner
extends      SimplifiedVisitor
implements   ClassVisitor,
             ConstantVisitor,
             MemberVisitor,
             AttributeVisitor,
             BootstrapMethodInfoVisitor,
             ExceptionInfoVisitor,
             InnerClassesInfoVisitor,
             StackMapFrameVisitor,
             VerificationTypeVisitor,
             ParameterInfoVisitor,
             LocalVariableInfoVisitor,
             LocalVariableTypeInfoVisitor,
             AnnotationVisitor,
             TypeAnnotationVisitor,
             ElementValueVisitor
{
    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        clean(programClass);

        programClass.constantPoolEntriesAccept(this);

        programClass.fieldsAccept(this);
        programClass.methodsAccept(this);

        programClass.attributesAccept(this);
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        clean(libraryClass);

        libraryClass.fieldsAccept(this);
        libraryClass.methodsAccept(this);
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant)
    {
        clean(constant);
    }


    // Implementations for MemberVisitor.

    public void visitProgramMember(ProgramClass programClass, ProgramMember programMember)
    {
        clean(programMember);

        programMember.attributesAccept(programClass, this);
    }


    public void visitLibraryMember(LibraryClass libraryClass, LibraryMember libraryMember)
    {
        clean(libraryMember);
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute)
    {
        clean(attribute);
    }


    public void visitBootstrapMethodsAttribute(Clazz clazz, BootstrapMethodsAttribute bootstrapMethodsAttribute)
    {
        clean(bootstrapMethodsAttribute);

        bootstrapMethodsAttribute.bootstrapMethodEntriesAccept(clazz, this);
    }


    public void visitInnerClassesAttribute(Clazz clazz, InnerClassesAttribute innerClassesAttribute)
    {
        clean(innerClassesAttribute);

        innerClassesAttribute.innerClassEntriesAccept(clazz, this);
    }


    public void visitMethodParametersAttribute(Clazz clazz, Method method, MethodParametersAttribute methodParametersAttribute)
    {
        clean(methodParametersAttribute);

        methodParametersAttribute.parametersAccept(clazz, method, this);
    }


    public void visitExceptionsAttribute(Clazz clazz, Method method, ExceptionsAttribute exceptionsAttribute)
    {
        clean(exceptionsAttribute);

        exceptionsAttribute.exceptionEntriesAccept((ProgramClass)clazz, this);
    }


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        clean(codeAttribute);

        codeAttribute.exceptionsAccept(clazz, method, this);
        codeAttribute.attributesAccept(clazz, method, this);
    }


    public void visitStackMapAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapAttribute stackMapAttribute)
    {
        clean(stackMapAttribute);

        stackMapAttribute.stackMapFramesAccept(clazz, method, codeAttribute, this);
    }


    public void visitStackMapTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapTableAttribute stackMapTableAttribute)
    {
        clean(stackMapTableAttribute);

        stackMapTableAttribute.stackMapFramesAccept(clazz, method, codeAttribute, this);
    }


    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        clean(localVariableTableAttribute);

        localVariableTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        clean(localVariableTypeTableAttribute);

        localVariableTypeTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitAnyAnnotationsAttribute(Clazz clazz, AnnotationsAttribute annotationsAttribute)
    {
        clean(annotationsAttribute);

        annotationsAttribute.annotationsAccept(clazz, this);
    }


    public void visitAnyParameterAnnotationsAttribute(Clazz clazz, Method method, ParameterAnnotationsAttribute parameterAnnotationsAttribute)
    {
        clean(parameterAnnotationsAttribute);

        parameterAnnotationsAttribute.annotationsAccept(clazz, method, this);
    }


    public void visitAnyTypeAnnotationsAttribute(Clazz clazz, TypeAnnotationsAttribute typeAnnotationsAttribute)
    {
        clean(typeAnnotationsAttribute);

        typeAnnotationsAttribute.typeAnnotationsAccept(clazz, this);
    }


    public void visitAnnotationDefaultAttribute(Clazz clazz, Method method, AnnotationDefaultAttribute annotationDefaultAttribute)
    {
        clean(annotationDefaultAttribute);

        annotationDefaultAttribute.defaultValueAccept(clazz, this);
    }


    // Implementations for BootstrapMethodInfoVisitor.

    public void visitBootstrapMethodInfo(Clazz clazz, BootstrapMethodInfo bootstrapMethodInfo)
    {
        clean(bootstrapMethodInfo);
    }


    // Implementations for InnerClassesInfoVisitor.

    public void visitInnerClassesInfo(Clazz clazz, InnerClassesInfo innerClassesInfo)
    {
        clean(innerClassesInfo);
    }


    // Implementations for ExceptionInfoVisitor.

    public void visitExceptionInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, ExceptionInfo exceptionInfo)
    {
        clean(exceptionInfo);
    }


    // Implementations for StackMapFrameVisitor.

    public void visitSameZeroFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SameZeroFrame sameZeroFrame)
    {
        clean(sameZeroFrame);
    }


    public void visitSameOneFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SameOneFrame sameOneFrame)
    {
        clean(sameOneFrame);

        sameOneFrame.stackItemAccept(clazz, method, codeAttribute, offset, this);
    }


    public void visitLessZeroFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, LessZeroFrame lessZeroFrame)
    {
        clean(lessZeroFrame);
    }


    public void visitMoreZeroFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, MoreZeroFrame moreZeroFrame)
    {
        clean(moreZeroFrame);

        moreZeroFrame.additionalVariablesAccept(clazz, method, codeAttribute, offset, this);
    }


    public void visitFullFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, FullFrame fullFrame)
    {
        clean(fullFrame);

        fullFrame.variablesAccept(clazz, method, codeAttribute, offset, this);
        fullFrame.stackAccept(clazz, method, codeAttribute, offset, this);
    }


    // Implementations for VerificationTypeVisitor.

    public void visitAnyVerificationType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VerificationType verificationType)
    {
        clean(verificationType);
    }


    // Implementations for ParameterInfoVisitor.

    public void visitParameterInfo(Clazz clazz, Method method, int parameterIndex, ParameterInfo parameterInfo)
    {
        clean(parameterInfo);
    }


    // Implementations for LocalVariableInfoVisitor.

    public void visitLocalVariableInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableInfo localVariableInfo)
    {
        clean(localVariableInfo);
    }


    // Implementations for LocalVariableTypeInfoVisitor.

    public void visitLocalVariableTypeInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeInfo localVariableTypeInfo)
    {
        clean(localVariableTypeInfo);
    }


    // Implementations for AnnotationVisitor.

    public void visitAnnotation(Clazz clazz, Annotation annotation)
    {
        clean(annotation);

        annotation.elementValuesAccept(clazz, this);
    }


    // Implementations for TypeAnnotationVisitor.

    public void visitTypeAnnotation(Clazz clazz, TypeAnnotation typeAnnotation)
    {
        clean(typeAnnotation);

        //typeAnnotation.targetInfoAccept(clazz, this);
        //typeAnnotation.typePathInfosAccept(clazz, this);
        typeAnnotation.elementValuesAccept(clazz, this);
    }


    // Implementations for ElementValueVisitor.

    public void visitAnyElementValue(Clazz clazz, Annotation annotation, ElementValue elementValue)
    {
        clean(elementValue);
    }


    public void visitAnnotationElementValue(Clazz clazz, Annotation annotation, AnnotationElementValue annotationElementValue)
    {
        clean(annotationElementValue);

        annotationElementValue.annotationAccept(clazz, this);
    }


    public void visitArrayElementValue(Clazz clazz, Annotation annotation, ArrayElementValue arrayElementValue)
    {
        clean(arrayElementValue);
    }


    // Small utility methods.

    private void clean(VisitorAccepter visitorAccepter)
    {
        visitorAccepter.setVisitorInfo(null);
    }
}

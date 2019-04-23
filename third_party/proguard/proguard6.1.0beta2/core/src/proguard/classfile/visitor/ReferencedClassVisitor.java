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
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This ClassVisitor, MemberVisitor, ConstantVisitor, AttributeVisitor, etc.
 * lets a given ClassVisitor visit all the referenced classes of the elements
 * that it visits. Only downstream elements are considered (in order to avoid
 * loops and repeated visits).
 *
 * @author Eric Lafortune
 */
public class ReferencedClassVisitor
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor,
             ConstantVisitor,
             AttributeVisitor,
             LocalVariableInfoVisitor,
             LocalVariableTypeInfoVisitor,
             AnnotationVisitor,
             ElementValueVisitor
{
    protected final ClassVisitor classVisitor;


    public ReferencedClassVisitor(ClassVisitor classVisitor)
    {
        this.classVisitor = classVisitor;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Visit the constant pool entries.
        programClass.constantPoolEntriesAccept(this);

        // Visit the fields and methods.
        programClass.fieldsAccept(this);
        programClass.methodsAccept(this);

        // Visit the attributes.
        programClass.attributesAccept(this);
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        // Visit the superclass and interfaces.
        libraryClass.superClassAccept(classVisitor);
        libraryClass.interfacesAccept(classVisitor);

        // Visit the fields and methods.
        libraryClass.fieldsAccept(this);
        libraryClass.methodsAccept(this);
    }


    // Implementations for MemberVisitor.

    public void visitProgramMember(ProgramClass programClass, ProgramMember programMember)
    {
        // Let the visitor visit the classes referenced in the descriptor string.
        programMember.referencedClassesAccept(classVisitor);

        // Visit the attributes.
        programMember.attributesAccept(programClass, this);
    }


    public void visitLibraryMember(LibraryClass programClass, LibraryMember libraryMember)
    {
        // Let the visitor visit the classes referenced in the descriptor string.
        libraryMember.referencedClassesAccept(classVisitor);
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        // Let the visitor visit the class referenced in the string constant.
        stringConstant.referencedClassAccept(classVisitor);
    }


    public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
    {
        // Let the visitor visit the class referenced in the reference constant.
        refConstant.referencedClassAccept(classVisitor);
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        // Let the visitor visit the class referenced in the reference constant.
        invokeDynamicConstant.referencedClassesAccept(classVisitor);
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        // Let the visitor visit the class referenced in the class constant.
        classConstant.referencedClassAccept(classVisitor);
    }


    public void visitMethodTypeConstant(Clazz clazz, MethodTypeConstant methodTypeConstant)
    {
        // Let the visitor visit the classes referenced in the method type constant.
        methodTypeConstant.referencedClassesAccept(classVisitor);
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitEnclosingMethodAttribute(Clazz clazz, EnclosingMethodAttribute enclosingMethodAttribute)
    {
        // Let the visitor visit the class of the enclosing method.
        enclosingMethodAttribute.referencedClassAccept(classVisitor);
    }


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Visit the attributes of the code attribute.
        codeAttribute.attributesAccept(clazz, method, this);
    }


    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        // Visit the local variables.
        localVariableTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        // Visit the local variable types.
        localVariableTypeTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitSignatureAttribute(Clazz clazz, SignatureAttribute signatureAttribute)
    {
        // Let the visitor visit the classes referenced in the signature string.
        signatureAttribute.referencedClassesAccept(classVisitor);
    }


    public void visitAnyAnnotationsAttribute(Clazz clazz, AnnotationsAttribute annotationsAttribute)
    {
        // Visit the annotations.
        annotationsAttribute.annotationsAccept(clazz, this);
    }


    public void visitAnyParameterAnnotationsAttribute(Clazz clazz, Method method, ParameterAnnotationsAttribute parameterAnnotationsAttribute)
    {
        // Visit the parameter annotations.
        parameterAnnotationsAttribute.annotationsAccept(clazz, method, this);
    }


    public void visitAnnotationDefaultAttribute(Clazz clazz, Method method, AnnotationDefaultAttribute annotationDefaultAttribute)
    {
        // Visit the default element value.
        annotationDefaultAttribute.defaultValueAccept(clazz, this);
    }


    // Implementations for LocalVariableInfoVisitor.

    public void visitLocalVariableInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableInfo localVariableInfo)
    {
        // Let the visitor visit the class referenced in the local variable.
        localVariableInfo.referencedClassAccept(classVisitor);
    }


    // Implementations for LocalVariableTypeInfoVisitor.

    public void visitLocalVariableTypeInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeInfo localVariableTypeInfo)
    {
        // Let the visitor visit the classes referenced in the local variable type.
        localVariableTypeInfo.referencedClassesAccept(classVisitor);
    }


    // Implementations for AnnotationVisitor.

    public void visitAnnotation(Clazz clazz, Annotation annotation)
    {
        // Let the visitor visit the classes referenced in the annotation.
        annotation.referencedClassesAccept(classVisitor);

        // Visit the element values.
        annotation.elementValuesAccept(clazz, this);
    }


    // Implementations for ElementValueVisitor.

    public void visitAnyElementValue(Clazz clazz, Annotation annotation, ElementValue elementValue) {}


    public void visitEnumConstantElementValue(Clazz clazz, Annotation annotation, EnumConstantElementValue enumConstantElementValue)
    {
        // Let the visitor visit the classes referenced in the constant element value.
        enumConstantElementValue.referencedClassesAccept(classVisitor);
    }


    public void visitClassElementValue(Clazz clazz, Annotation annotation, ClassElementValue classElementValue)
    {
        // Let the visitor visit the classes referenced in the class element value.
        classElementValue.referencedClassesAccept(classVisitor);
    }


    public void visitAnnotationElementValue(Clazz clazz, Annotation annotation, AnnotationElementValue annotationElementValue)
    {
        // Visit the contained annotation.
        annotationElementValue.annotationAccept(clazz, this);
    }


    public void visitArrayElementValue(Clazz clazz, Annotation annotation, ArrayElementValue arrayElementValue)
    {
        // Visit the element values.
        arrayElementValue.elementValuesAccept(clazz, annotation, this);
    }
}

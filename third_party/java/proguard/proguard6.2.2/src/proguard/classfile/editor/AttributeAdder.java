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
import proguard.classfile.attribute.preverification.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.ReferencedClassVisitor;

import java.util.Arrays;

/**
 * This AttributeVisitor adds all attributes that it visits to the given
 * target class, class member, or attribute.
 *
 * @author Eric Lafortune
 */
public class AttributeAdder
extends      SimplifiedVisitor
implements   AttributeVisitor
{
    private static final byte[]          EMPTY_BYTES       = new byte[0];
    private static final int[]           EMPTY_INTS        = new int[0];
    private static final Attribute[]     EMPTY_ATTRIBUTES  = new Attribute[0];
    private static final ExceptionInfo[] EMPTY_EXCEPTIONS  = new ExceptionInfo[0];
    private static final Annotation[]    EMPTY_ANNOTATIONS = new Annotation[0];


    private final ProgramClass  targetClass;
    private final ProgramMember targetMember;
    private final CodeAttribute targetCodeAttribute;
    private final boolean       replaceAttributes;

    private final ConstantAdder    constantAdder;
    private final AttributesEditor attributesEditor;


    /**
     * Creates a new AttributeAdder that will copy attributes into the given
     * target class.
     */
    public AttributeAdder(ProgramClass targetClass,
                          boolean      replaceAttributes)
    {
        this(targetClass, null, null, replaceAttributes);
    }


    /**
     * Creates a new AttributeAdder that will copy attributes into the given
     * target class member.
     */
    public AttributeAdder(ProgramClass  targetClass,
                          ProgramMember targetMember,
                          boolean       replaceAttributes)
    {
        this(targetClass, targetMember, null, replaceAttributes);
    }


    /**
     * Creates a new AttributeAdder that will copy attributes into the given
     * target attribute.
     */
    public AttributeAdder(ProgramClass  targetClass,
                          ProgramMember targetMember,
                          CodeAttribute targetCodeAttribute,
                          boolean       replaceAttributes)
    {
        this.targetClass         = targetClass;
        this.targetMember        = targetMember;
        this.targetCodeAttribute = targetCodeAttribute;
        this.replaceAttributes   = replaceAttributes;

        constantAdder    = new ConstantAdder(targetClass);
        attributesEditor = new AttributesEditor(targetClass,
                                                targetMember,
                                                targetCodeAttribute,
                                                replaceAttributes);
    }


    // Implementations for AttributeVisitor.

    public void visitUnknownAttribute(Clazz clazz, UnknownAttribute unknownAttribute)
    {
        // Create a copy of the attribute.
        UnknownAttribute newUnknownAttribute =
            new UnknownAttribute(constantAdder.addConstant(clazz, unknownAttribute.u2attributeNameIndex),
                                 unknownAttribute.u4attributeLength,
                                 unknownAttribute.info);

        // Add it to the target class.
        attributesEditor.addAttribute(newUnknownAttribute);
    }


    public void visitSourceFileAttribute(Clazz clazz, SourceFileAttribute sourceFileAttribute)
    {
        // Create a copy of the attribute.
        SourceFileAttribute newSourceFileAttribute =
            new SourceFileAttribute(constantAdder.addConstant(clazz, sourceFileAttribute.u2attributeNameIndex),
                                    constantAdder.addConstant(clazz, sourceFileAttribute.u2sourceFileIndex));

        // Add it to the target class.
        attributesEditor.addAttribute(newSourceFileAttribute);
    }


    public void visitSourceDirAttribute(Clazz clazz, SourceDirAttribute sourceDirAttribute)
    {
        // Create a copy of the attribute.
        SourceDirAttribute newSourceDirAttribute =
            new SourceDirAttribute(constantAdder.addConstant(clazz, sourceDirAttribute.u2attributeNameIndex),
                                   constantAdder.addConstant(clazz, sourceDirAttribute.u2sourceDirIndex));

        // Add it to the target class.
        attributesEditor.addAttribute(newSourceDirAttribute);
    }


    public void visitInnerClassesAttribute(Clazz clazz, InnerClassesAttribute innerClassesAttribute)
    {
        // Create a copy of the attribute.
        InnerClassesAttribute newInnerClassesAttribute =
            new InnerClassesAttribute(constantAdder.addConstant(clazz, innerClassesAttribute.u2attributeNameIndex),
                                      0,
                                      null);

        // Add it to the target class.
        attributesEditor.addAttribute(newInnerClassesAttribute);
    }


    public void visitEnclosingMethodAttribute(Clazz clazz, EnclosingMethodAttribute enclosingMethodAttribute)
    {
        // Create a copy of the attribute.
        EnclosingMethodAttribute newEnclosingMethodAttribute =
            new EnclosingMethodAttribute(constantAdder.addConstant(clazz, enclosingMethodAttribute.u2attributeNameIndex),
                                         constantAdder.addConstant(clazz, enclosingMethodAttribute.u2classIndex),
                                         enclosingMethodAttribute.u2nameAndTypeIndex == 0 ? 0 :
                                         constantAdder.addConstant(clazz, enclosingMethodAttribute.u2nameAndTypeIndex));

        newEnclosingMethodAttribute.referencedClass  = enclosingMethodAttribute.referencedClass;
        newEnclosingMethodAttribute.referencedMethod = enclosingMethodAttribute.referencedMethod;

        // Add it to the target class.
        attributesEditor.addAttribute(newEnclosingMethodAttribute);
    }


    public void visitNestHostAttribute(Clazz clazz, NestHostAttribute nestHostAttribute)
    {
        // Create a copy of the attribute.
        NestHostAttribute newNestHostAttribute =
            new NestHostAttribute(constantAdder.addConstant(clazz, nestHostAttribute.u2attributeNameIndex),
                                  constantAdder.addConstant(clazz, nestHostAttribute.u2hostClassIndex));

        // Add it to the target class.
        attributesEditor.addAttribute(newNestHostAttribute);
    }


    public void visitNestMembersAttribute(Clazz clazz, NestMembersAttribute nestMembersAttribute)
    {
        // Create a copy of the attribute.
        NestMembersAttribute newNestMembersAttribute =
            new NestMembersAttribute(constantAdder.addConstant(clazz, nestMembersAttribute.u2attributeNameIndex),
                                     0,
                                     nestMembersAttribute.u2classesCount > 0 ?
                                         new int[nestMembersAttribute.u2classesCount] :
                                         EMPTY_INTS);

        // Add the nest members.
        nestMembersAttribute.memberClassConstantsAccept(targetClass,
                                                        new NestMemberAdder(targetClass,
                                                                            newNestMembersAttribute));

        // Add it to the target class.
        attributesEditor.addAttribute(newNestMembersAttribute);
    }


    public void visitDeprecatedAttribute(Clazz clazz, DeprecatedAttribute deprecatedAttribute)
    {
        // Create a copy of the attribute.
        DeprecatedAttribute newDeprecatedAttribute =
            new DeprecatedAttribute(constantAdder.addConstant(clazz, deprecatedAttribute.u2attributeNameIndex));

        // Add it to the target.
        attributesEditor.addAttribute(newDeprecatedAttribute);
    }


    public void visitSyntheticAttribute(Clazz clazz, SyntheticAttribute syntheticAttribute)
    {
        // Create a copy of the attribute.
        SyntheticAttribute newSyntheticAttribute =
            new SyntheticAttribute(constantAdder.addConstant(clazz, syntheticAttribute.u2attributeNameIndex));

        // Add it to the target.
        attributesEditor.addAttribute(newSyntheticAttribute);
    }


    public void visitSignatureAttribute(Clazz clazz, SignatureAttribute signatureAttribute)
    {
        // Create a copy of the attribute.
        SignatureAttribute newSignatureAttribute =
            new SignatureAttribute(constantAdder.addConstant(clazz, signatureAttribute.u2attributeNameIndex),
                                   constantAdder.addConstant(clazz, signatureAttribute.u2signatureIndex));

        newSignatureAttribute.referencedClasses = signatureAttribute.referencedClasses;

        // Add it to the target.
        attributesEditor.addAttribute(newSignatureAttribute);
    }


    public void visitConstantValueAttribute(Clazz clazz, Field field, ConstantValueAttribute constantValueAttribute)
    {
        // Create a copy of the attribute.
        ConstantValueAttribute newConstantValueAttribute =
            new ConstantValueAttribute(constantAdder.addConstant(clazz, constantValueAttribute.u2attributeNameIndex),
                                       constantAdder.addConstant(clazz, constantValueAttribute.u2constantValueIndex));

        // Add it to the target field.
        attributesEditor.addAttribute(newConstantValueAttribute);
    }


    public void visitMethodParametersAttribute(Clazz clazz, Method method, MethodParametersAttribute methodParametersAttribute)
    {
        // Create a new local variable table attribute.
        MethodParametersAttribute newMethodParametersAttribute =
            new MethodParametersAttribute(constantAdder.addConstant(clazz, methodParametersAttribute.u2attributeNameIndex),
                                          methodParametersAttribute.u1parametersCount,
                                          new ParameterInfo[methodParametersAttribute.u1parametersCount]);

        // Add the local variables.
        methodParametersAttribute.parametersAccept(clazz,
                                                   method,
                                                   new ParameterInfoAdder(targetClass, newMethodParametersAttribute));

        // Add it to the target.
        attributesEditor.addAttribute(newMethodParametersAttribute);
    }


    public void visitExceptionsAttribute(Clazz clazz, Method method, ExceptionsAttribute exceptionsAttribute)
    {
        // Create a new exceptions attribute.
        ExceptionsAttribute newExceptionsAttribute =
            new ExceptionsAttribute(constantAdder.addConstant(clazz, exceptionsAttribute.u2attributeNameIndex),
                                    0,
                                    exceptionsAttribute.u2exceptionIndexTableLength > 0 ?
                                        new int[exceptionsAttribute.u2exceptionIndexTableLength] :
                                        EMPTY_INTS);

        // Add the exceptions.
        exceptionsAttribute.exceptionEntriesAccept((ProgramClass)clazz,
                                                   new ExceptionAdder(targetClass,
                                                                      newExceptionsAttribute));

        // Add it to the target method.
        attributesEditor.addAttribute(newExceptionsAttribute);
    }


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Create a new code attribute.
        CodeAttribute newCodeAttribute =
            new CodeAttribute(constantAdder.addConstant(clazz, codeAttribute.u2attributeNameIndex),
                              codeAttribute.u2maxStack,
                              codeAttribute.u2maxLocals,
                              0,
                              EMPTY_BYTES,
                              0,
                              codeAttribute.u2exceptionTableLength > 0 ?
                                  new ExceptionInfo[codeAttribute.u2exceptionTableLength] :
                                  EMPTY_EXCEPTIONS,
                              0,
                              codeAttribute.u2attributesCount > 0 ?
                                  new Attribute[codeAttribute.u2attributesCount] :
                                  EMPTY_ATTRIBUTES);

        CodeAttributeComposer codeAttributeComposer = new CodeAttributeComposer();

        codeAttributeComposer.beginCodeFragment(codeAttribute.u4codeLength + 32);

        // Add the instructions.
        codeAttribute.instructionsAccept(clazz,
                                         method,
                                         new InstructionAdder(targetClass,
                                                              codeAttributeComposer));

        // Append a label just after the code.
        codeAttributeComposer.appendLabel(codeAttribute.u4codeLength);

        // Add the exceptions.
        codeAttribute.exceptionsAccept(clazz,
                                       method,
                                       new ExceptionInfoAdder(targetClass,
                                                              codeAttributeComposer));

        // Add a line number if there wasn't a line number table before,
        // so we keep track of the source.
        if (codeAttribute.getAttribute(clazz, ClassConstants.ATTR_LineNumberTable) == null)
        {
            String source =
                clazz.getName()             + '.' +
                method.getName(clazz)       +
                method.getDescriptor(clazz) +
                ":0:0";

            codeAttributeComposer.insertLineNumber(
                new ExtendedLineNumberInfo(0, 0, source));
        }

        codeAttributeComposer.endCodeFragment();

        // Add the attributes.
        codeAttribute.attributesAccept(clazz,
                                       method,
                                       new AttributeAdder(targetClass,
                                                          targetMember,
                                                          newCodeAttribute,
                                                          replaceAttributes));

        // Apply these changes to the new code attribute.
        codeAttributeComposer.visitCodeAttribute(targetClass,
                                                 (Method)targetMember,
                                                 newCodeAttribute);

        // Add the completed code attribute to the target method.
        attributesEditor.addAttribute(newCodeAttribute);
    }


    public void visitStackMapAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapAttribute stackMapAttribute)
    {
        // TODO: Implement method.
    }


    public void visitStackMapTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapTableAttribute stackMapTableAttribute)
    {
        // TODO: Implement method.
    }


    public void visitLineNumberTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute lineNumberTableAttribute)
    {
        // Create a new line number table attribute.
        LineNumberTableAttribute newLineNumberTableAttribute =
            new LineNumberTableAttribute(constantAdder.addConstant(clazz, lineNumberTableAttribute.u2attributeNameIndex),
                                         0,
                                         new LineNumberInfo[lineNumberTableAttribute.u2lineNumberTableLength]);

        // Add the line numbers.
        lineNumberTableAttribute.accept(clazz,
                                        method,
                                        codeAttribute,
                                        new LineNumberInfoAdder(newLineNumberTableAttribute));

        // Add it to the target.
        attributesEditor.addAttribute(newLineNumberTableAttribute);
    }


    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        // Create a new local variable table attribute.
        LocalVariableTableAttribute newLocalVariableTableAttribute =
            new LocalVariableTableAttribute(constantAdder.addConstant(clazz, localVariableTableAttribute.u2attributeNameIndex),
                                            0,
                                            new LocalVariableInfo[localVariableTableAttribute.u2localVariableTableLength]);

        // Add the local variables.
        localVariableTableAttribute.localVariablesAccept(clazz,
                                                         method,
                                                         codeAttribute,
                                                         new LocalVariableInfoAdder(targetClass, newLocalVariableTableAttribute));

        // Add it to the target.
        attributesEditor.addAttribute(newLocalVariableTableAttribute);
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        // Create a new local variable type table attribute.
        LocalVariableTypeTableAttribute newLocalVariableTypeTableAttribute =
            new LocalVariableTypeTableAttribute(constantAdder.addConstant(clazz, localVariableTypeTableAttribute.u2attributeNameIndex),
                                            0,
                                            new LocalVariableTypeInfo[localVariableTypeTableAttribute.u2localVariableTypeTableLength]);

        // Add the local variable types.
        localVariableTypeTableAttribute.localVariablesAccept(clazz,
                                                             method,
                                                             codeAttribute,
                                                             new LocalVariableTypeInfoAdder(targetClass, newLocalVariableTypeTableAttribute));

        // Add it to the target.
        attributesEditor.addAttribute(newLocalVariableTypeTableAttribute);
    }


    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        // Create a new annotations attribute.
        RuntimeVisibleAnnotationsAttribute newAnnotationsAttribute =
            new RuntimeVisibleAnnotationsAttribute(constantAdder.addConstant(clazz, runtimeVisibleAnnotationsAttribute.u2attributeNameIndex),
                                                   0,
                                                   new Annotation[runtimeVisibleAnnotationsAttribute.u2annotationsCount]);

        // Add the annotations.
        runtimeVisibleAnnotationsAttribute.annotationsAccept(clazz,
                                                             new AnnotationAdder(targetClass,
                                                                                 newAnnotationsAttribute));

        // Add it to the target.
        attributesEditor.addAttribute(newAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        // Create a new annotations attribute.
        RuntimeInvisibleAnnotationsAttribute newAnnotationsAttribute =
            new RuntimeInvisibleAnnotationsAttribute(constantAdder.addConstant(clazz, runtimeInvisibleAnnotationsAttribute.u2attributeNameIndex),
                                                     0,
                                                     new Annotation[runtimeInvisibleAnnotationsAttribute.u2annotationsCount]);

        // Add the annotations.
        runtimeInvisibleAnnotationsAttribute.annotationsAccept(clazz,
                                                               new AnnotationAdder(targetClass,
                                                                                   newAnnotationsAttribute));

        // Add it to the target.
        attributesEditor.addAttribute(newAnnotationsAttribute);
    }


    public void visitRuntimeVisibleParameterAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleParameterAnnotationsAttribute runtimeVisibleParameterAnnotationsAttribute)
    {
        // Create a new annotations attribute.
        Annotation[][] parameterAnnotations =
            new Annotation[runtimeVisibleParameterAnnotationsAttribute.u1parametersCount][];

        Arrays.fill(parameterAnnotations, EMPTY_ANNOTATIONS);

        RuntimeVisibleParameterAnnotationsAttribute newParameterAnnotationsAttribute =
            new RuntimeVisibleParameterAnnotationsAttribute(constantAdder.addConstant(clazz, runtimeVisibleParameterAnnotationsAttribute.u2attributeNameIndex),
                                                            0,
                                                            new int[runtimeVisibleParameterAnnotationsAttribute.u1parametersCount],
                                                            parameterAnnotations);

        // Add the annotations.
        runtimeVisibleParameterAnnotationsAttribute.annotationsAccept(clazz,
                                                                      method,
                                                                      new AnnotationAdder(targetClass,
                                                                                          newParameterAnnotationsAttribute));

        // Add it to the target.
        attributesEditor.addAttribute(newParameterAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleParameterAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleParameterAnnotationsAttribute runtimeInvisibleParameterAnnotationsAttribute)
    {
        // Create a new annotations attribute.
        Annotation[][] parameterAnnotations =
            new Annotation[runtimeInvisibleParameterAnnotationsAttribute.u1parametersCount][];

        Arrays.fill(parameterAnnotations, EMPTY_ANNOTATIONS);

        RuntimeInvisibleParameterAnnotationsAttribute newParameterAnnotationsAttribute =
            new RuntimeInvisibleParameterAnnotationsAttribute(constantAdder.addConstant(clazz, runtimeInvisibleParameterAnnotationsAttribute.u2attributeNameIndex),
                                                              0,
                                                              new int[runtimeInvisibleParameterAnnotationsAttribute.u1parametersCount],
                                                              parameterAnnotations);

        // Add the annotations.
        runtimeInvisibleParameterAnnotationsAttribute.annotationsAccept(clazz,
                                                                        method,
                                                                        new AnnotationAdder(targetClass,
                                                                                            newParameterAnnotationsAttribute));

        // Add it to the target.
        attributesEditor.addAttribute(newParameterAnnotationsAttribute);
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        // Create a new type annotations attribute.
        RuntimeVisibleTypeAnnotationsAttribute newTypeAnnotationsAttribute =
            new RuntimeVisibleTypeAnnotationsAttribute(constantAdder.addConstant(clazz, runtimeVisibleTypeAnnotationsAttribute.u2attributeNameIndex),
                                                       0,
                                                       new TypeAnnotation[runtimeVisibleTypeAnnotationsAttribute.u2annotationsCount]);

        // Add the annotations.
        runtimeVisibleTypeAnnotationsAttribute.typeAnnotationsAccept(clazz,
                                                                     new TypeAnnotationAdder(targetClass,
                                                                                             newTypeAnnotationsAttribute));

        // Add it to the target.
        attributesEditor.addAttribute(newTypeAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        // Create a new type annotations attribute.
        RuntimeInvisibleTypeAnnotationsAttribute newTypeAnnotationsAttribute =
            new RuntimeInvisibleTypeAnnotationsAttribute(constantAdder.addConstant(clazz, runtimeInvisibleTypeAnnotationsAttribute.u2attributeNameIndex),
                                                         0,
                                                         new TypeAnnotation[runtimeInvisibleTypeAnnotationsAttribute.u2annotationsCount]);

        // Add the annotations.
        runtimeInvisibleTypeAnnotationsAttribute.typeAnnotationsAccept(clazz,
                                                                       new TypeAnnotationAdder(targetClass,
                                                                                               newTypeAnnotationsAttribute));

        // Add it to the target.
        attributesEditor.addAttribute(newTypeAnnotationsAttribute);
    }


    public void visitAnnotationDefaultAttribute(Clazz clazz, Method method, AnnotationDefaultAttribute annotationDefaultAttribute)
    {
        // Create a new annotation default attribute.
        AnnotationDefaultAttribute newAnnotationDefaultAttribute =
            new AnnotationDefaultAttribute(constantAdder.addConstant(clazz, annotationDefaultAttribute.u2attributeNameIndex),
                                           null);

        // Add the annotations.
        annotationDefaultAttribute.defaultValueAccept(clazz,
                                                      new ElementValueAdder(targetClass,
                                                                            newAnnotationDefaultAttribute,
                                                                            false));

        // Add it to the target.
        attributesEditor.addAttribute(newAnnotationDefaultAttribute);
    }
}

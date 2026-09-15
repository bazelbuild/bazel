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
import proguard.classfile.attribute.preverification.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;

import java.util.Arrays;


/**
 * This ClassVisitor removes UTF-8 constant pool entries that are not used.
 *
 * @author Eric Lafortune
 */
public class Utf8Shrinker
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor,
             ConstantVisitor,
             AttributeVisitor,
             InnerClassesInfoVisitor,
             ParameterInfoVisitor,
             LocalVariableInfoVisitor,
             LocalVariableTypeInfoVisitor,
             AnnotationVisitor,
             ElementValueVisitor
{
    // A visitor info flag to indicate the UTF-8 constant pool entry is being used.
    private static final Object USED = new Object();

    private       int[]                constantIndexMap     = new int[ClassConstants.TYPICAL_CONSTANT_POOL_SIZE];
    private final ConstantPoolRemapper constantPoolRemapper = new ConstantPoolRemapper();


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Mark the UTF-8 entries referenced by the other constant pool entries.
        programClass.constantPoolEntriesAccept(this);

        // Mark the UTF-8 entries referenced by the fields and methods.
        programClass.fieldsAccept(this);
        programClass.methodsAccept(this);

        // Mark the UTF-8 entries referenced by the attributes.
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
        // Mark the name and descriptor UTF-8 entries.
        markCpUtf8Entry(programClass, programMember.u2nameIndex);
        markCpUtf8Entry(programClass, programMember.u2descriptorIndex);

        // Mark the UTF-8 entries referenced by the attributes.
        programMember.attributesAccept(programClass, this);
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        markCpUtf8Entry(clazz, stringConstant.u2stringIndex);
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        markCpUtf8Entry(clazz, classConstant.u2nameIndex);
    }


    public void visitNameAndTypeConstant(Clazz clazz, NameAndTypeConstant nameAndTypeConstant)
    {
        markCpUtf8Entry(clazz, nameAndTypeConstant.u2nameIndex);
        markCpUtf8Entry(clazz, nameAndTypeConstant.u2descriptorIndex);
    }


    // Implementations for AttributeVisitor.

    public void visitUnknownAttribute(Clazz clazz, UnknownAttribute unknownAttribute)
    {
        // This is the best we can do for unknown attributes.
        markCpUtf8Entry(clazz, unknownAttribute.u2attributeNameIndex);
    }


    public void visitSourceFileAttribute(Clazz clazz, SourceFileAttribute sourceFileAttribute)
    {
        markCpUtf8Entry(clazz, sourceFileAttribute.u2attributeNameIndex);

        markCpUtf8Entry(clazz, sourceFileAttribute.u2sourceFileIndex);
    }


    public void visitSourceDirAttribute(Clazz clazz, SourceDirAttribute sourceDirAttribute)
    {
        markCpUtf8Entry(clazz, sourceDirAttribute.u2attributeNameIndex);

        markCpUtf8Entry(clazz, sourceDirAttribute.u2sourceDirIndex);
    }


    public void visitInnerClassesAttribute(Clazz clazz, InnerClassesAttribute innerClassesAttribute)
    {
        markCpUtf8Entry(clazz, innerClassesAttribute.u2attributeNameIndex);

        // Mark the UTF-8 entries referenced by the inner classes.
        innerClassesAttribute.innerClassEntriesAccept(clazz, this);
    }


    public void visitEnclosingMethodAttribute(Clazz clazz, EnclosingMethodAttribute enclosingMethodAttribute)
    {
        markCpUtf8Entry(clazz, enclosingMethodAttribute.u2attributeNameIndex);

        // These entries have already been marked in the constant pool.
        //clazz.constantPoolEntryAccept(this, enclosingMethodAttribute.u2classIndex);
        //clazz.constantPoolEntryAccept(this, enclosingMethodAttribute.u2nameAndTypeIndex);
    }


    public void visitNestHostAttribute(Clazz clazz, NestHostAttribute nestHostAttribute)
    {
        markCpUtf8Entry(clazz, nestHostAttribute.u2attributeNameIndex);
    }


    public void visitNestMembersAttribute(Clazz clazz, NestMembersAttribute nestMembersAttribute)
    {
        markCpUtf8Entry(clazz, nestMembersAttribute.u2attributeNameIndex);
    }


    public void visitModuleAttribute(Clazz clazz, ModuleAttribute moduleAttribute)
    {
        markCpUtf8Entry(clazz, moduleAttribute.u2attributeNameIndex);
    }


    public void visitModuleMainClassAttribute(Clazz clazz, ModuleMainClassAttribute moduleMainClassAttribute)
    {
        markCpUtf8Entry(clazz, moduleMainClassAttribute.u2attributeNameIndex);
    }


    public void visitModulePackagesAttribute(Clazz clazz, ModulePackagesAttribute modulePackagesAttribute)
    {
        markCpUtf8Entry(clazz, modulePackagesAttribute.u2attributeNameIndex);
    }


    public void visitDeprecatedAttribute(Clazz clazz, DeprecatedAttribute deprecatedAttribute)
    {
        markCpUtf8Entry(clazz, deprecatedAttribute.u2attributeNameIndex);
    }


    public void visitSyntheticAttribute(Clazz clazz, SyntheticAttribute syntheticAttribute)
    {
        markCpUtf8Entry(clazz, syntheticAttribute.u2attributeNameIndex);
    }


    public void visitSignatureAttribute(Clazz clazz, SignatureAttribute signatureAttribute)
    {
        markCpUtf8Entry(clazz, signatureAttribute.u2attributeNameIndex);

        markCpUtf8Entry(clazz, signatureAttribute.u2signatureIndex);
    }


    public void visitConstantValueAttribute(Clazz clazz, Field field, ConstantValueAttribute constantValueAttribute)
    {
        markCpUtf8Entry(clazz, constantValueAttribute.u2attributeNameIndex);
    }


    public void visitMethodParametersAttribute(Clazz clazz, Method method, MethodParametersAttribute methodParametersAttribute)
    {
        markCpUtf8Entry(clazz, methodParametersAttribute.u2attributeNameIndex);

        // Mark the UTF-8 entries referenced by the parameter information.
        methodParametersAttribute.parametersAccept(clazz, method, this);
    }


    public void visitExceptionsAttribute(Clazz clazz, Method method, ExceptionsAttribute exceptionsAttribute)
    {
        markCpUtf8Entry(clazz, exceptionsAttribute.u2attributeNameIndex);
    }


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        markCpUtf8Entry(clazz, codeAttribute.u2attributeNameIndex);

        // Mark the UTF-8 entries referenced by the attributes.
        codeAttribute.attributesAccept(clazz, method, this);
    }


    public void visitStackMapAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapAttribute stackMapAttribute)
    {
        markCpUtf8Entry(clazz, stackMapAttribute.u2attributeNameIndex);
    }


    public void visitStackMapTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapTableAttribute stackMapTableAttribute)
    {
        markCpUtf8Entry(clazz, stackMapTableAttribute.u2attributeNameIndex);
    }


    public void visitLineNumberTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute lineNumberTableAttribute)
    {
        markCpUtf8Entry(clazz, lineNumberTableAttribute.u2attributeNameIndex);
    }


    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        markCpUtf8Entry(clazz, localVariableTableAttribute.u2attributeNameIndex);

        // Mark the UTF-8 entries referenced by the local variables.
        localVariableTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        markCpUtf8Entry(clazz, localVariableTypeTableAttribute.u2attributeNameIndex);

        // Mark the UTF-8 entries referenced by the local variable types.
        localVariableTypeTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitAnyAnnotationsAttribute(Clazz clazz, AnnotationsAttribute annotationsAttribute)
    {
        markCpUtf8Entry(clazz, annotationsAttribute.u2attributeNameIndex);

        // Mark the UTF-8 entries referenced by the annotations.
        annotationsAttribute.annotationsAccept(clazz, this);
    }


    public void visitAnyParameterAnnotationsAttribute(Clazz clazz, Method method, ParameterAnnotationsAttribute parameterAnnotationsAttribute)
    {
        markCpUtf8Entry(clazz, parameterAnnotationsAttribute.u2attributeNameIndex);

        // Mark the UTF-8 entries referenced by the annotations.
        parameterAnnotationsAttribute.annotationsAccept(clazz, method, this);
    }


    public void visitAnnotationDefaultAttribute(Clazz clazz, Method method, AnnotationDefaultAttribute annotationDefaultAttribute)
    {
        markCpUtf8Entry(clazz, annotationDefaultAttribute.u2attributeNameIndex);

        // Mark the UTF-8 entries referenced by the element value.
        annotationDefaultAttribute.defaultValueAccept(clazz, this);
    }


    // Implementations for InnerClassesInfoVisitor.

    public void visitInnerClassesInfo(Clazz clazz, InnerClassesInfo innerClassesInfo)
    {
        if (innerClassesInfo.u2innerNameIndex != 0)
        {
            markCpUtf8Entry(clazz, innerClassesInfo.u2innerNameIndex);
        }
    }


    // Implementations for ParameterInfoVisitor.

    public void visitParameterInfo(Clazz clazz, Method method, int parameterIndex, ParameterInfo parameterInfo)
    {
        if (parameterInfo.u2nameIndex != 0)
        {
            markCpUtf8Entry(clazz, parameterInfo.u2nameIndex);
        }
    }


    // Implementations for LocalVariableInfoVisitor.

    public void visitLocalVariableInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableInfo localVariableInfo)
    {
        markCpUtf8Entry(clazz, localVariableInfo.u2nameIndex);
        markCpUtf8Entry(clazz, localVariableInfo.u2descriptorIndex);
    }


    // Implementations for LocalVariableTypeInfoVisitor.

    public void visitLocalVariableTypeInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeInfo localVariableTypeInfo)
    {
        markCpUtf8Entry(clazz, localVariableTypeInfo.u2nameIndex);
        markCpUtf8Entry(clazz, localVariableTypeInfo.u2signatureIndex);
    }


    // Implementations for AnnotationVisitor.

    public void visitAnnotation(Clazz clazz, Annotation annotation)
    {
        markCpUtf8Entry(clazz, annotation.u2typeIndex);

        // Mark the UTF-8 entries referenced by the element values.
        annotation.elementValuesAccept(clazz, this);
    }


    // Implementations for ElementValueVisitor.

    public void visitConstantElementValue(Clazz clazz, Annotation annotation, ConstantElementValue constantElementValue)
    {
        if (constantElementValue.u2elementNameIndex != 0)
        {
            markCpUtf8Entry(clazz, constantElementValue.u2elementNameIndex);
        }

        // Only the string constant element value refers to a UTF-8 entry.
        if (constantElementValue.u1tag == ClassConstants.ELEMENT_VALUE_STRING_CONSTANT)
        {
            markCpUtf8Entry(clazz, constantElementValue.u2constantValueIndex);
        }
    }


    public void visitEnumConstantElementValue(Clazz clazz, Annotation annotation, EnumConstantElementValue enumConstantElementValue)
    {
        if (enumConstantElementValue.u2elementNameIndex != 0)
        {
            markCpUtf8Entry(clazz, enumConstantElementValue.u2elementNameIndex);
        }

        markCpUtf8Entry(clazz, enumConstantElementValue.u2typeNameIndex);
        markCpUtf8Entry(clazz, enumConstantElementValue.u2constantNameIndex);
    }


    public void visitClassElementValue(Clazz clazz, Annotation annotation, ClassElementValue classElementValue)
    {
        if (classElementValue.u2elementNameIndex != 0)
        {
            markCpUtf8Entry(clazz, classElementValue.u2elementNameIndex);
        }

        markCpUtf8Entry(clazz, classElementValue.u2classInfoIndex);
    }


    public void visitAnnotationElementValue(Clazz clazz, Annotation annotation, AnnotationElementValue annotationElementValue)
    {
        if (annotationElementValue.u2elementNameIndex != 0)
        {
            markCpUtf8Entry(clazz, annotationElementValue.u2elementNameIndex);
        }

        // Mark the UTF-8 entries referenced by the annotation.
        annotationElementValue.annotationAccept(clazz, this);
    }


    public void visitArrayElementValue(Clazz clazz, Annotation annotation, ArrayElementValue arrayElementValue)
    {
        if (arrayElementValue.u2elementNameIndex != 0)
        {
            markCpUtf8Entry(clazz, arrayElementValue.u2elementNameIndex);
        }

        // Mark the UTF-8 entries referenced by the element values.
        arrayElementValue.elementValuesAccept(clazz, annotation, this);
    }


    // Small utility methods.

    /**
     * Marks the given UTF-8 constant pool entry of the given class.
     */
    private void markCpUtf8Entry(Clazz clazz, int index)
    {
         markAsUsed((Utf8Constant)((ProgramClass)clazz).getConstant(index));
    }


    /**
     * Marks the given VisitorAccepter as being used.
     * In this context, the VisitorAccepter will be a Utf8Constant object.
     */
    private void markAsUsed(VisitorAccepter visitorAccepter)
    {
        visitorAccepter.setVisitorInfo(USED);
    }


    /**
     * Returns whether the given VisitorAccepter has been marked as being used.
     * In this context, the VisitorAccepter will be a Utf8Constant object.
     */
    private boolean isUsed(VisitorAccepter visitorAccepter)
    {
        return visitorAccepter.getVisitorInfo() == USED;
    }


    /**
     * Removes all UTF-8 entries that are not marked as being used
     * from the given constant pool.
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
                isUsed = constant.getTag() != ClassConstants.CONSTANT_Utf8 ||
                         isUsed(constant);
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

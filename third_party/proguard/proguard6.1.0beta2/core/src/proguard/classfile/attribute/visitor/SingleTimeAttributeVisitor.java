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
package proguard.classfile.attribute.visitor;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.annotation.*;
import proguard.classfile.attribute.module.*;
import proguard.classfile.attribute.preverification.*;

/**
 * This AttributeVisitor delegates all visits to a given AttributeVisitor,
 * although only once to the same attribute in a row.
 *
 * It can for example be used to lazily apply a visitor in a place where it
 * would be called multiple times.
 *
 * @author Eric Lafortune
 */
public class SingleTimeAttributeVisitor
implements   AttributeVisitor
{
    private final AttributeVisitor attributeVisitor;

    private Attribute lastVisitedAttribute;


    public SingleTimeAttributeVisitor(AttributeVisitor attributeVisitor)
    {
        this.attributeVisitor = attributeVisitor;
    }


    // Implementations for AttributeVisitor.

    @Override
    public void visitUnknownAttribute(Clazz clazz, UnknownAttribute unknownAttribute)
    {
        if (!unknownAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitUnknownAttribute(clazz, unknownAttribute);

            lastVisitedAttribute = unknownAttribute;
        }
    }

    @Override
    public void visitBootstrapMethodsAttribute(Clazz clazz, BootstrapMethodsAttribute bootstrapMethodsAttribute)
    {
        if (!bootstrapMethodsAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitBootstrapMethodsAttribute(clazz, bootstrapMethodsAttribute);

            lastVisitedAttribute = bootstrapMethodsAttribute;
        }
    }

    @Override
    public void visitSourceFileAttribute(Clazz clazz, SourceFileAttribute sourceFileAttribute)
    {
        if (!sourceFileAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitSourceFileAttribute(clazz, sourceFileAttribute);

            lastVisitedAttribute = sourceFileAttribute;
        }
    }

    @Override
    public void visitSourceDirAttribute(Clazz clazz, SourceDirAttribute sourceDirAttribute)
    {
        if (!sourceDirAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitSourceDirAttribute(clazz, sourceDirAttribute);

            lastVisitedAttribute = sourceDirAttribute;
        }
    }

    @Override
    public void visitInnerClassesAttribute(Clazz clazz, InnerClassesAttribute innerClassesAttribute)
    {
        if (!innerClassesAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitInnerClassesAttribute(clazz, innerClassesAttribute);

            lastVisitedAttribute = innerClassesAttribute;
        }
    }

    @Override
    public void visitEnclosingMethodAttribute(Clazz clazz, EnclosingMethodAttribute enclosingMethodAttribute)
    {
        if (!enclosingMethodAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitEnclosingMethodAttribute(clazz, enclosingMethodAttribute);

            lastVisitedAttribute = enclosingMethodAttribute;
        }
    }

    @Override
    public void visitNestHostAttribute(Clazz clazz, NestHostAttribute nestHostAttribute)
    {
        if (!nestHostAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitNestHostAttribute(clazz, nestHostAttribute);

            lastVisitedAttribute = nestHostAttribute;
        }
    }

    @Override
    public void visitNestMembersAttribute(Clazz clazz, NestMembersAttribute nestMembersAttribute)
    {
        if (!nestMembersAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitNestMembersAttribute(clazz, nestMembersAttribute);

            lastVisitedAttribute = nestMembersAttribute;
        }
    }

    @Override
    public void visitModuleAttribute(Clazz clazz, ModuleAttribute moduleAttribute)
    {
        if (!moduleAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitModuleAttribute(clazz, moduleAttribute);

            lastVisitedAttribute = moduleAttribute;
        }
    }

    @Override
    public void visitModuleMainClassAttribute(Clazz clazz, ModuleMainClassAttribute moduleMainClassAttribute)
    {
        if (!moduleMainClassAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitModuleMainClassAttribute(clazz, moduleMainClassAttribute);

            lastVisitedAttribute = moduleMainClassAttribute;
        }
    }

    @Override
    public void visitModulePackagesAttribute(Clazz clazz, ModulePackagesAttribute modulePackagesAttribute)
    {
        if (!modulePackagesAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitModulePackagesAttribute(clazz, modulePackagesAttribute);

            lastVisitedAttribute = modulePackagesAttribute;
        }
    }

    @Override
    public void visitDeprecatedAttribute(Clazz clazz, DeprecatedAttribute deprecatedAttribute)
    {
        if (!deprecatedAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitDeprecatedAttribute(clazz, deprecatedAttribute);

            lastVisitedAttribute = deprecatedAttribute;
        }
    }

    @Override
    public void visitDeprecatedAttribute(Clazz clazz, Field field, DeprecatedAttribute deprecatedAttribute)
    {
        if (!deprecatedAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitDeprecatedAttribute(clazz, field, deprecatedAttribute);

            lastVisitedAttribute = deprecatedAttribute;
        }
    }

    @Override
    public void visitDeprecatedAttribute(Clazz clazz, Method method, DeprecatedAttribute deprecatedAttribute)
    {
        if (!deprecatedAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitDeprecatedAttribute(clazz, method, deprecatedAttribute);

            lastVisitedAttribute = deprecatedAttribute;
        }
    }

    @Override
    public void visitSyntheticAttribute(Clazz clazz, SyntheticAttribute syntheticAttribute)
    {
        if (!syntheticAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitSyntheticAttribute(clazz, syntheticAttribute);

            lastVisitedAttribute = syntheticAttribute;
        }
    }

    @Override
    public void visitSyntheticAttribute(Clazz clazz, Field field, SyntheticAttribute syntheticAttribute)
    {
        if (!syntheticAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitSyntheticAttribute(clazz, field, syntheticAttribute);

            lastVisitedAttribute = syntheticAttribute;
        }
    }

    @Override
    public void visitSyntheticAttribute(Clazz clazz, Method method, SyntheticAttribute syntheticAttribute)
    {
        if (!syntheticAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitSyntheticAttribute(clazz, method, syntheticAttribute);

            lastVisitedAttribute = syntheticAttribute;
        }
    }

    @Override
    public void visitSignatureAttribute(Clazz clazz, SignatureAttribute signatureAttribute)
    {
        if (!signatureAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitSignatureAttribute(clazz, signatureAttribute);

            lastVisitedAttribute = signatureAttribute;
        }
    }

    @Override
    public void visitSignatureAttribute(Clazz clazz, Field field, SignatureAttribute signatureAttribute)
    {
        if (!signatureAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitSignatureAttribute(clazz, field, signatureAttribute);

            lastVisitedAttribute = signatureAttribute;
        }
    }

    @Override
    public void visitSignatureAttribute(Clazz clazz, Method method, SignatureAttribute signatureAttribute)
    {
        if (!signatureAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitSignatureAttribute(clazz, method, signatureAttribute);

            lastVisitedAttribute = signatureAttribute;
        }
    }

    @Override
    public void visitConstantValueAttribute(Clazz clazz, Field field, ConstantValueAttribute constantValueAttribute)
    {
        if (!constantValueAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitConstantValueAttribute(clazz, field, constantValueAttribute);

            lastVisitedAttribute = constantValueAttribute;
        }
    }

    @Override
    public void visitMethodParametersAttribute(Clazz clazz, Method method, MethodParametersAttribute methodParametersAttribute)
    {
        if (!methodParametersAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitMethodParametersAttribute(clazz, method, methodParametersAttribute);

            lastVisitedAttribute = methodParametersAttribute;
        }
    }

    @Override
    public void visitExceptionsAttribute(Clazz clazz, Method method, ExceptionsAttribute exceptionsAttribute)
    {
        if (!exceptionsAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitExceptionsAttribute(clazz, method, exceptionsAttribute);

            lastVisitedAttribute = exceptionsAttribute;
        }
    }

    @Override
    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        if (!codeAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitCodeAttribute(clazz, method, codeAttribute);

            lastVisitedAttribute = codeAttribute;
        }
    }

    @Override
    public void visitStackMapAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapAttribute stackMapAttribute)
    {
        if (!stackMapAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitStackMapAttribute(clazz, method, codeAttribute, stackMapAttribute);

            lastVisitedAttribute = stackMapAttribute;
        }
    }

    @Override
    public void visitStackMapTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapTableAttribute stackMapTableAttribute)
    {
        if (!stackMapTableAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitStackMapTableAttribute(clazz, method, codeAttribute, stackMapTableAttribute);

            lastVisitedAttribute = stackMapTableAttribute;
        }
    }

    @Override
    public void visitLineNumberTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute lineNumberTableAttribute)
    {
        if (!lineNumberTableAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitLineNumberTableAttribute(clazz, method, codeAttribute, lineNumberTableAttribute);

            lastVisitedAttribute = lineNumberTableAttribute;
        }
    }

    @Override
    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        if (!localVariableTableAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitLocalVariableTableAttribute(clazz, method, codeAttribute, localVariableTableAttribute);

            lastVisitedAttribute = localVariableTableAttribute;
        }
    }

    @Override
    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        if (!localVariableTypeTableAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitLocalVariableTypeTableAttribute(clazz, method, codeAttribute, localVariableTypeTableAttribute);

            lastVisitedAttribute = localVariableTypeTableAttribute;
        }
    }

    @Override
    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        if (!runtimeVisibleAnnotationsAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitRuntimeVisibleAnnotationsAttribute(clazz, runtimeVisibleAnnotationsAttribute);

            lastVisitedAttribute = runtimeVisibleAnnotationsAttribute;
        }
    }

    @Override
    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, Field field, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        if (!runtimeVisibleAnnotationsAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitRuntimeVisibleAnnotationsAttribute(clazz, field, runtimeVisibleAnnotationsAttribute);

            lastVisitedAttribute = runtimeVisibleAnnotationsAttribute;
        }
    }

    @Override
    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        if (!runtimeVisibleAnnotationsAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitRuntimeVisibleAnnotationsAttribute(clazz, method, runtimeVisibleAnnotationsAttribute);

            lastVisitedAttribute = runtimeVisibleAnnotationsAttribute;
        }
    }

    @Override
    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        if (!runtimeInvisibleAnnotationsAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitRuntimeInvisibleAnnotationsAttribute(clazz, runtimeInvisibleAnnotationsAttribute);

            lastVisitedAttribute = runtimeInvisibleAnnotationsAttribute;
        }
    }

    @Override
    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, Field field, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        if (!runtimeInvisibleAnnotationsAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitRuntimeInvisibleAnnotationsAttribute(clazz, field, runtimeInvisibleAnnotationsAttribute);

            lastVisitedAttribute = runtimeInvisibleAnnotationsAttribute;
        }
    }

    @Override
    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        if (!runtimeInvisibleAnnotationsAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitRuntimeInvisibleAnnotationsAttribute(clazz, method, runtimeInvisibleAnnotationsAttribute);

            lastVisitedAttribute = runtimeInvisibleAnnotationsAttribute;
        }
    }

    @Override
    public void visitRuntimeVisibleParameterAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleParameterAnnotationsAttribute runtimeVisibleParameterAnnotationsAttribute)
    {
        if (!runtimeVisibleParameterAnnotationsAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitRuntimeVisibleParameterAnnotationsAttribute(clazz, method, runtimeVisibleParameterAnnotationsAttribute);

            lastVisitedAttribute = runtimeVisibleParameterAnnotationsAttribute;
        }
    }

    @Override
    public void visitRuntimeInvisibleParameterAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleParameterAnnotationsAttribute runtimeInvisibleParameterAnnotationsAttribute)
    {
        if (!runtimeInvisibleParameterAnnotationsAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitRuntimeInvisibleParameterAnnotationsAttribute(clazz, method, runtimeInvisibleParameterAnnotationsAttribute);

            lastVisitedAttribute = runtimeInvisibleParameterAnnotationsAttribute;
        }
    }

    @Override
    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        if (!runtimeVisibleTypeAnnotationsAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitRuntimeVisibleTypeAnnotationsAttribute(clazz, runtimeVisibleTypeAnnotationsAttribute);

            lastVisitedAttribute = runtimeVisibleTypeAnnotationsAttribute;
        }
    }

    @Override
    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Field field, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        if (!runtimeVisibleTypeAnnotationsAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitRuntimeVisibleTypeAnnotationsAttribute(clazz, field, runtimeVisibleTypeAnnotationsAttribute);

            lastVisitedAttribute = runtimeVisibleTypeAnnotationsAttribute;
        }
    }

    @Override
    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        if (!runtimeVisibleTypeAnnotationsAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitRuntimeVisibleTypeAnnotationsAttribute(clazz, method, runtimeVisibleTypeAnnotationsAttribute);

            lastVisitedAttribute = runtimeVisibleTypeAnnotationsAttribute;
        }
    }

    @Override
    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        if (!runtimeVisibleTypeAnnotationsAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitRuntimeVisibleTypeAnnotationsAttribute(clazz, method, codeAttribute, runtimeVisibleTypeAnnotationsAttribute);

            lastVisitedAttribute = runtimeVisibleTypeAnnotationsAttribute;
        }
    }

    @Override
    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        if (!runtimeInvisibleTypeAnnotationsAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, runtimeInvisibleTypeAnnotationsAttribute);

            lastVisitedAttribute = runtimeInvisibleTypeAnnotationsAttribute;
        }
    }

    @Override
    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Field field, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        if (!runtimeInvisibleTypeAnnotationsAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, field, runtimeInvisibleTypeAnnotationsAttribute);

            lastVisitedAttribute = runtimeInvisibleTypeAnnotationsAttribute;
        }
    }

    @Override
    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        if (!runtimeInvisibleTypeAnnotationsAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, method, runtimeInvisibleTypeAnnotationsAttribute);

            lastVisitedAttribute = runtimeInvisibleTypeAnnotationsAttribute;
        }
    }

    @Override
    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        if (!runtimeInvisibleTypeAnnotationsAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, method, codeAttribute, runtimeInvisibleTypeAnnotationsAttribute);

            lastVisitedAttribute = runtimeInvisibleTypeAnnotationsAttribute;
        }
    }

    @Override
    public void visitAnnotationDefaultAttribute(Clazz clazz, Method method, AnnotationDefaultAttribute annotationDefaultAttribute)
    {
        if (!annotationDefaultAttribute.equals(lastVisitedAttribute))
        {
            attributeVisitor.visitAnnotationDefaultAttribute(clazz, method, annotationDefaultAttribute);

            lastVisitedAttribute = annotationDefaultAttribute;
        }
    }
}

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
import proguard.obfuscate.AttributeShrinker;

/**
 * This AttributeVisitor delegates its visits to one of two other
 * AttributeVisitor instances, depending on whether the visited attribute
 * is strictly required or not.
 *
 * Stack map attributes and stack map table attributes are treated as optional.
 *
 * @see AttributeShrinker
 *
 * @author Eric Lafortune
 */
public class RequiredAttributeFilter
implements   AttributeVisitor
{
    private final AttributeVisitor requiredAttributeVisitor;
    private final AttributeVisitor optionalAttributeVisitor;


    /**
     * Creates a new RequiredAttributeFilter for visiting required attributes.
     * @param requiredAttributeVisitor   the visitor that will visit required
     *                                   attributes.
     */
    public RequiredAttributeFilter(AttributeVisitor requiredAttributeVisitor)
    {
        this(requiredAttributeVisitor, null);
    }


    /**
     * Creates a new RequiredAttributeFilter for visiting required and
     * optional attributes.
     * @param requiredAttributeVisitor the visitor that will visit required
     *                                 attributes.
     * @param optionalAttributeVisitor the visitor that will visit optional
     *                                 attributes.
     */
    public RequiredAttributeFilter(AttributeVisitor requiredAttributeVisitor,
                                   AttributeVisitor optionalAttributeVisitor)
    {
        this.requiredAttributeVisitor = requiredAttributeVisitor;
        this.optionalAttributeVisitor = optionalAttributeVisitor;
    }


    // Implementations for AttributeVisitor.

    public void visitUnknownAttribute(Clazz clazz, UnknownAttribute unknownAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitUnknownAttribute(clazz, unknownAttribute);
        }
    }


    public void visitBootstrapMethodsAttribute(Clazz clazz, BootstrapMethodsAttribute bootstrapMethodsAttribute)
    {
        if (requiredAttributeVisitor != null)
        {
            requiredAttributeVisitor.visitBootstrapMethodsAttribute(clazz, bootstrapMethodsAttribute);
        }
    }


    public void visitSourceFileAttribute(Clazz clazz, SourceFileAttribute sourceFileAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitSourceFileAttribute(clazz, sourceFileAttribute);
        }
    }


    public void visitSourceDirAttribute(Clazz clazz, SourceDirAttribute sourceDirAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitSourceDirAttribute(clazz, sourceDirAttribute);
        }
    }


    public void visitInnerClassesAttribute(Clazz clazz, InnerClassesAttribute innerClassesAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitInnerClassesAttribute(clazz, innerClassesAttribute);
        }
    }


    public void visitEnclosingMethodAttribute(Clazz clazz, EnclosingMethodAttribute enclosingMethodAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitEnclosingMethodAttribute(clazz, enclosingMethodAttribute);
        }
    }


    public void visitNestHostAttribute(Clazz clazz, NestHostAttribute nestHostAttribute)
    {
        if (requiredAttributeVisitor != null)
        {
            requiredAttributeVisitor.visitNestHostAttribute(clazz, nestHostAttribute);
        }
    }


    public void visitNestMembersAttribute(Clazz clazz, NestMembersAttribute nestMembersAttribute)
    {
        if (requiredAttributeVisitor != null)
        {
            requiredAttributeVisitor.visitNestMembersAttribute(clazz, nestMembersAttribute);
        }
    }


    public void visitModuleAttribute(Clazz clazz, ModuleAttribute moduleAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitModuleAttribute(clazz, moduleAttribute);
        }
    }


    public void visitModuleMainClassAttribute(Clazz clazz, ModuleMainClassAttribute moduleMainClassAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitModuleMainClassAttribute(clazz, moduleMainClassAttribute);
        }
    }


    public void visitModulePackagesAttribute(Clazz clazz, ModulePackagesAttribute modulePackagesAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitModulePackagesAttribute(clazz, modulePackagesAttribute);
        }
    }


    public void visitDeprecatedAttribute(Clazz clazz, DeprecatedAttribute deprecatedAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitDeprecatedAttribute(clazz, deprecatedAttribute);
        }
    }


    public void visitDeprecatedAttribute(Clazz clazz, Field field, DeprecatedAttribute deprecatedAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitDeprecatedAttribute(clazz, field, deprecatedAttribute);
        }
    }


    public void visitDeprecatedAttribute(Clazz clazz, Method method, DeprecatedAttribute deprecatedAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitDeprecatedAttribute(clazz, method, deprecatedAttribute);
        }
    }


    public void visitSyntheticAttribute(Clazz clazz, SyntheticAttribute syntheticAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitSyntheticAttribute(clazz, syntheticAttribute);
        }
    }


    public void visitSyntheticAttribute(Clazz clazz, Field field, SyntheticAttribute syntheticAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitSyntheticAttribute(clazz, field, syntheticAttribute);
        }
    }


    public void visitSyntheticAttribute(Clazz clazz, Method method, SyntheticAttribute syntheticAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitSyntheticAttribute(clazz, method, syntheticAttribute);
        }
    }


    public void visitSignatureAttribute(Clazz clazz, SignatureAttribute signatureAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitSignatureAttribute(clazz, signatureAttribute);
        }
    }


    public void visitSignatureAttribute(Clazz clazz, Field field, SignatureAttribute signatureAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitSignatureAttribute(clazz, field, signatureAttribute);
        }
    }


    public void visitSignatureAttribute(Clazz clazz, Method method, SignatureAttribute signatureAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitSignatureAttribute(clazz, method, signatureAttribute);
        }
    }


    public void visitConstantValueAttribute(Clazz clazz, Field field, ConstantValueAttribute constantValueAttribute)
    {
        if (requiredAttributeVisitor != null)
        {
            requiredAttributeVisitor.visitConstantValueAttribute(clazz, field, constantValueAttribute);
        }
    }


    public void visitMethodParametersAttribute(Clazz clazz, Method method, MethodParametersAttribute exceptionsAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitMethodParametersAttribute(clazz, method, exceptionsAttribute);
        }
    }


    public void visitExceptionsAttribute(Clazz clazz, Method method, ExceptionsAttribute exceptionsAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitExceptionsAttribute(clazz, method, exceptionsAttribute);
        }
    }


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        if (requiredAttributeVisitor != null)
        {
            requiredAttributeVisitor.visitCodeAttribute(clazz, method, codeAttribute);
        }
    }


    public void visitStackMapAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapAttribute stackMapAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitStackMapAttribute(clazz, method, codeAttribute, stackMapAttribute);
        }
    }


    public void visitStackMapTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapTableAttribute stackMapTableAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitStackMapTableAttribute(clazz, method, codeAttribute, stackMapTableAttribute);
        }
    }


    public void visitLineNumberTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute lineNumberTableAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitLineNumberTableAttribute(clazz, method, codeAttribute, lineNumberTableAttribute);
        }
    }


    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitLocalVariableTableAttribute(clazz, method, codeAttribute, localVariableTableAttribute);
        }
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitLocalVariableTypeTableAttribute(clazz, method, codeAttribute, localVariableTypeTableAttribute);
        }
    }


    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitRuntimeVisibleAnnotationsAttribute(clazz, runtimeVisibleAnnotationsAttribute);
        }
    }


    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, Field field, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitRuntimeVisibleAnnotationsAttribute(clazz, field, runtimeVisibleAnnotationsAttribute);
        }
    }


    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitRuntimeVisibleAnnotationsAttribute(clazz, method, runtimeVisibleAnnotationsAttribute);
        }
    }


    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitRuntimeInvisibleAnnotationsAttribute(clazz, runtimeInvisibleAnnotationsAttribute);
        }
    }


    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, Field field, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitRuntimeInvisibleAnnotationsAttribute(clazz, field, runtimeInvisibleAnnotationsAttribute);
        }
    }


    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitRuntimeInvisibleAnnotationsAttribute(clazz, method, runtimeInvisibleAnnotationsAttribute);
        }
    }


    public void visitRuntimeVisibleParameterAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleParameterAnnotationsAttribute runtimeVisibleParameterAnnotationsAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitRuntimeVisibleParameterAnnotationsAttribute(clazz, method, runtimeVisibleParameterAnnotationsAttribute);
        }
    }


    public void visitRuntimeInvisibleParameterAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleParameterAnnotationsAttribute runtimeInvisibleParameterAnnotationsAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitRuntimeInvisibleParameterAnnotationsAttribute(clazz, method, runtimeInvisibleParameterAnnotationsAttribute);
        }
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitRuntimeVisibleTypeAnnotationsAttribute(clazz, runtimeVisibleTypeAnnotationsAttribute);
        }
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Field field, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitRuntimeVisibleTypeAnnotationsAttribute(clazz, field, runtimeVisibleTypeAnnotationsAttribute);
        }
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitRuntimeVisibleTypeAnnotationsAttribute(clazz, method, runtimeVisibleTypeAnnotationsAttribute);
        }
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitRuntimeVisibleTypeAnnotationsAttribute(clazz, method, codeAttribute, runtimeVisibleTypeAnnotationsAttribute);
        }
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, runtimeInvisibleTypeAnnotationsAttribute);
        }
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Field field, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, field, runtimeInvisibleTypeAnnotationsAttribute);
        }
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, method, runtimeInvisibleTypeAnnotationsAttribute);
        }
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, method, codeAttribute, runtimeInvisibleTypeAnnotationsAttribute);
        }
    }


    public void visitAnnotationDefaultAttribute(Clazz clazz, Method method, AnnotationDefaultAttribute annotationDefaultAttribute)
    {
        if (optionalAttributeVisitor != null)
        {
            optionalAttributeVisitor.visitAnnotationDefaultAttribute(clazz, method, annotationDefaultAttribute);
        }
    }
}

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
import proguard.util.ArrayUtil;

/**
 * This AttributeVisitor delegates all visits to each AttributeVisitor
 * in a given list.
 *
 * @author Eric Lafortune
 */
public class MultiAttributeVisitor implements AttributeVisitor
{
    private AttributeVisitor[] attributeVisitors;
    private int                attributeVisitorCount;


    public MultiAttributeVisitor()
    {
        this.attributeVisitors = new AttributeVisitor[16];
    }


    public MultiAttributeVisitor(AttributeVisitor... attributeVisitors)
    {
        this.attributeVisitors     = attributeVisitors;
        this.attributeVisitorCount = attributeVisitors.length;
    }


    public void addAttributeVisitor(AttributeVisitor attributeVisitor)
    {
        attributeVisitors =
            ArrayUtil.add(attributeVisitors,
                          attributeVisitorCount++,
                          attributeVisitor);
    }


    // Implementations for AttributeVisitor.


    public void visitUnknownAttribute(Clazz clazz, UnknownAttribute unknownAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitUnknownAttribute(clazz, unknownAttribute);
        }
    }


    public void visitBootstrapMethodsAttribute(Clazz clazz, BootstrapMethodsAttribute bootstrapMethodsAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitBootstrapMethodsAttribute(clazz, bootstrapMethodsAttribute);
        }
    }


    public void visitSourceFileAttribute(Clazz clazz, SourceFileAttribute sourceFileAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitSourceFileAttribute(clazz, sourceFileAttribute);
        }
    }


    public void visitSourceDirAttribute(Clazz clazz, SourceDirAttribute sourceDirAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitSourceDirAttribute(clazz, sourceDirAttribute);
        }
    }


    public void visitInnerClassesAttribute(Clazz clazz, InnerClassesAttribute innerClassesAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitInnerClassesAttribute(clazz, innerClassesAttribute);
        }
    }


    public void visitEnclosingMethodAttribute(Clazz clazz, EnclosingMethodAttribute enclosingMethodAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitEnclosingMethodAttribute(clazz, enclosingMethodAttribute);
        }
    }


    public void visitNestHostAttribute(Clazz clazz, NestHostAttribute nestHostAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitNestHostAttribute(clazz, nestHostAttribute);
        }
    }


    public void visitNestMembersAttribute(Clazz clazz, NestMembersAttribute nestMembersAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitNestMembersAttribute(clazz, nestMembersAttribute);
        }
    }


    public void visitModuleAttribute(Clazz clazz, ModuleAttribute moduleAttribute)
    {
        for (int index = 0; index < attributeVisitors.length; index++)
        {
            attributeVisitors[index].visitModuleAttribute(clazz, moduleAttribute);
        }
    }


    public void visitModuleMainClassAttribute(Clazz clazz, ModuleMainClassAttribute moduleMainClassAttribute)
    {
        for (int index = 0; index < attributeVisitors.length; index++)
        {
            attributeVisitors[index].visitModuleMainClassAttribute(clazz, moduleMainClassAttribute);
        }
    }


    public void visitModulePackagesAttribute(Clazz clazz, ModulePackagesAttribute modulePackagesAttribute)
    {
        for (int index = 0; index < attributeVisitors.length; index++)
        {
            attributeVisitors[index].visitModulePackagesAttribute(clazz, modulePackagesAttribute);
        }
    }


    public void visitDeprecatedAttribute(Clazz clazz, DeprecatedAttribute deprecatedAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitDeprecatedAttribute(clazz, deprecatedAttribute);
        }
    }


    public void visitSyntheticAttribute(Clazz clazz, SyntheticAttribute syntheticAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitSyntheticAttribute(clazz, syntheticAttribute);
        }
    }


    public void visitSignatureAttribute(Clazz clazz, SignatureAttribute syntheticAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitSignatureAttribute(clazz, syntheticAttribute);
        }
    }


    public void visitDeprecatedAttribute(Clazz clazz, Field field, DeprecatedAttribute deprecatedAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitDeprecatedAttribute(clazz, field, deprecatedAttribute);
        }
    }


    public void visitSyntheticAttribute(Clazz clazz, Field field, SyntheticAttribute syntheticAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitSyntheticAttribute(clazz, field, syntheticAttribute);
        }
    }


    public void visitSignatureAttribute(Clazz clazz, Field field, SignatureAttribute syntheticAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitSignatureAttribute(clazz, field, syntheticAttribute);
        }
    }


    public void visitDeprecatedAttribute(Clazz clazz, Method method, DeprecatedAttribute deprecatedAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitDeprecatedAttribute(clazz, method, deprecatedAttribute);
        }
    }


    public void visitSyntheticAttribute(Clazz clazz, Method method, SyntheticAttribute syntheticAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitSyntheticAttribute(clazz, method, syntheticAttribute);
        }
    }


    public void visitSignatureAttribute(Clazz clazz, Method method, SignatureAttribute syntheticAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitSignatureAttribute(clazz, method, syntheticAttribute);
        }
    }


    public void visitConstantValueAttribute(Clazz clazz, Field field, ConstantValueAttribute constantValueAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitConstantValueAttribute(clazz, field, constantValueAttribute);
        }
    }


    public void visitMethodParametersAttribute(Clazz clazz, Method method, MethodParametersAttribute methodParametersAttribute)
    {
        for (int index = 0; index < attributeVisitors.length; index++)
        {
            attributeVisitors[index].visitMethodParametersAttribute(clazz, method, methodParametersAttribute);
        }
    }


    public void visitExceptionsAttribute(Clazz clazz, Method method, ExceptionsAttribute exceptionsAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitExceptionsAttribute(clazz, method, exceptionsAttribute);
        }
    }


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitCodeAttribute(clazz, method, codeAttribute);
        }
    }


    public void visitStackMapAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapAttribute stackMapAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitStackMapAttribute(clazz, method, codeAttribute, stackMapAttribute);
        }
    }


    public void visitStackMapTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapTableAttribute stackMapTableAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitStackMapTableAttribute(clazz, method, codeAttribute, stackMapTableAttribute);
        }
    }


    public void visitLineNumberTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute lineNumberTableAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitLineNumberTableAttribute(clazz, method, codeAttribute, lineNumberTableAttribute);
        }
    }


    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitLocalVariableTableAttribute(clazz, method, codeAttribute, localVariableTableAttribute);
        }
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitLocalVariableTypeTableAttribute(clazz, method, codeAttribute, localVariableTypeTableAttribute);
        }
    }


    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitRuntimeVisibleAnnotationsAttribute(clazz, runtimeVisibleAnnotationsAttribute);
        }
    }


    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitRuntimeInvisibleAnnotationsAttribute(clazz, runtimeInvisibleAnnotationsAttribute);
        }
    }


    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, Field field, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitRuntimeVisibleAnnotationsAttribute(clazz, field, runtimeVisibleAnnotationsAttribute);
        }
    }


    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, Field field, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitRuntimeInvisibleAnnotationsAttribute(clazz, field, runtimeInvisibleAnnotationsAttribute);
        }
    }


    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitRuntimeVisibleAnnotationsAttribute(clazz, method, runtimeVisibleAnnotationsAttribute);
        }
    }


    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitRuntimeInvisibleAnnotationsAttribute(clazz, method, runtimeInvisibleAnnotationsAttribute);
        }
    }


    public void visitRuntimeVisibleParameterAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleParameterAnnotationsAttribute runtimeVisibleParameterAnnotationsAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitRuntimeVisibleParameterAnnotationsAttribute(clazz, method, runtimeVisibleParameterAnnotationsAttribute);
        }
    }


    public void visitRuntimeInvisibleParameterAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleParameterAnnotationsAttribute runtimeInvisibleParameterAnnotationsAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitRuntimeInvisibleParameterAnnotationsAttribute(clazz, method, runtimeInvisibleParameterAnnotationsAttribute);
        }
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        for (int index = 0; index < attributeVisitors.length; index++)
        {
            attributeVisitors[index].visitRuntimeVisibleTypeAnnotationsAttribute(clazz, runtimeVisibleTypeAnnotationsAttribute);
        }
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Field field, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        for (int index = 0; index < attributeVisitors.length; index++)
        {
            attributeVisitors[index].visitRuntimeVisibleTypeAnnotationsAttribute(clazz, field, runtimeVisibleTypeAnnotationsAttribute);
        }
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        for (int index = 0; index < attributeVisitors.length; index++)
        {
            attributeVisitors[index].visitRuntimeVisibleTypeAnnotationsAttribute(clazz, method, runtimeVisibleTypeAnnotationsAttribute);
        }
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        for (int index = 0; index < attributeVisitors.length; index++)
        {
            attributeVisitors[index].visitRuntimeVisibleTypeAnnotationsAttribute(clazz, method, codeAttribute, runtimeVisibleTypeAnnotationsAttribute);
        }
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        for (int index = 0; index < attributeVisitors.length; index++)
        {
            attributeVisitors[index].visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, runtimeInvisibleTypeAnnotationsAttribute);
        }
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Field field, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        for (int index = 0; index < attributeVisitors.length; index++)
        {
            attributeVisitors[index].visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, field, runtimeInvisibleTypeAnnotationsAttribute);
        }
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        for (int index = 0; index < attributeVisitors.length; index++)
        {
            attributeVisitors[index].visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, method, runtimeInvisibleTypeAnnotationsAttribute);
        }
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        for (int index = 0; index < attributeVisitors.length; index++)
        {
            attributeVisitors[index].visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, method, codeAttribute, runtimeInvisibleTypeAnnotationsAttribute);
        }
    }


    public void visitAnnotationDefaultAttribute(Clazz clazz, Method method, AnnotationDefaultAttribute annotationDefaultAttribute)
    {
        for (int index = 0; index < attributeVisitorCount; index++)
        {
            attributeVisitors[index].visitAnnotationDefaultAttribute(clazz, method, annotationDefaultAttribute);
        }
    }
}

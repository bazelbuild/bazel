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
 * This interface specifies the methods for a visitor of <code>Attribute</code>
 * objects.
 *
 * @author Eric Lafortune
 */
public interface AttributeVisitor
{
    // Attributes that are attached to classes.

    public void visitUnknownAttribute(               Clazz clazz,                UnknownAttribute           unknownAttribute);
    public void visitBootstrapMethodsAttribute(      Clazz clazz,                BootstrapMethodsAttribute  bootstrapMethodsAttribute);
    public void visitSourceFileAttribute(            Clazz clazz,                SourceFileAttribute        sourceFileAttribute);
    public void visitSourceDirAttribute(             Clazz clazz,                SourceDirAttribute         sourceDirAttribute);
    public void visitInnerClassesAttribute(          Clazz clazz,                InnerClassesAttribute      innerClassesAttribute);
    public void visitEnclosingMethodAttribute(       Clazz clazz,                EnclosingMethodAttribute   enclosingMethodAttribute);
    public void visitNestHostAttribute(              Clazz clazz,                NestHostAttribute          nestHostAttribute);
    public void visitNestMembersAttribute(           Clazz clazz,                NestMembersAttribute       nestMembersAttribute);
    public void visitModuleAttribute(                Clazz clazz,                ModuleAttribute            moduleAttribute);
    public void visitModuleMainClassAttribute(       Clazz clazz,                ModuleMainClassAttribute   moduleMainClassAttribute);
    public void visitModulePackagesAttribute(        Clazz clazz,                ModulePackagesAttribute    modulePackagesAttribute);
    public void visitDeprecatedAttribute(            Clazz clazz,                DeprecatedAttribute deprecatedAttribute);
    public void visitDeprecatedAttribute(            Clazz clazz, Field  field,  DeprecatedAttribute deprecatedAttribute);
    public void visitDeprecatedAttribute(            Clazz clazz, Method method, DeprecatedAttribute deprecatedAttribute);

    public void visitSyntheticAttribute(             Clazz clazz,                SyntheticAttribute  syntheticAttribute);
    public void visitSyntheticAttribute(             Clazz clazz, Field  field,  SyntheticAttribute  syntheticAttribute);
    public void visitSyntheticAttribute(             Clazz clazz, Method method, SyntheticAttribute  syntheticAttribute);

    public void visitSignatureAttribute(             Clazz clazz,                SignatureAttribute  signatureAttribute);
    public void visitSignatureAttribute(             Clazz clazz, Field  field,  SignatureAttribute  signatureAttribute);
    public void visitSignatureAttribute(             Clazz clazz, Method method, SignatureAttribute  signatureAttribute);

    // Attributes that are attached to fields.

    public void visitConstantValueAttribute(         Clazz clazz, Field  field,  ConstantValueAttribute constantValueAttribute);

    // Attributes that are attached to methods.

    public void visitMethodParametersAttribute(      Clazz clazz, Method method, MethodParametersAttribute methodParametersAttribute);
    public void visitExceptionsAttribute(            Clazz clazz, Method method, ExceptionsAttribute       exceptionsAttribute);
    public void visitCodeAttribute(                  Clazz clazz, Method method, CodeAttribute             codeAttribute);

    // Attributes that are attached to code attributes.

    public void visitStackMapAttribute(              Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapAttribute               stackMapAttribute);
    public void visitStackMapTableAttribute(         Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapTableAttribute          stackMapTableAttribute);
    public void visitLineNumberTableAttribute(       Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute        lineNumberTableAttribute);
    public void visitLocalVariableTableAttribute(    Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute     localVariableTableAttribute);
    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute);

    // Annotation attributes.

    public void visitRuntimeVisibleAnnotationsAttribute(           Clazz clazz,                RuntimeVisibleAnnotationsAttribute   runtimeVisibleAnnotationsAttribute);
    public void visitRuntimeVisibleAnnotationsAttribute(           Clazz clazz, Field  field,  RuntimeVisibleAnnotationsAttribute   runtimeVisibleAnnotationsAttribute);
    public void visitRuntimeVisibleAnnotationsAttribute(           Clazz clazz, Method method, RuntimeVisibleAnnotationsAttribute   runtimeVisibleAnnotationsAttribute);

    public void visitRuntimeInvisibleAnnotationsAttribute(         Clazz clazz,                RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute);
    public void visitRuntimeInvisibleAnnotationsAttribute(         Clazz clazz, Field  field,  RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute);
    public void visitRuntimeInvisibleAnnotationsAttribute(         Clazz clazz, Method method, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute);

    public void visitRuntimeVisibleParameterAnnotationsAttribute(  Clazz clazz, Method method, RuntimeVisibleParameterAnnotationsAttribute   runtimeVisibleParameterAnnotationsAttribute);
    public void visitRuntimeInvisibleParameterAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleParameterAnnotationsAttribute runtimeInvisibleParameterAnnotationsAttribute);

    public void visitRuntimeVisibleTypeAnnotationsAttribute(       Clazz clazz,                                             RuntimeVisibleTypeAnnotationsAttribute   runtimeVisibleTypeAnnotationsAttribute);
    public void visitRuntimeVisibleTypeAnnotationsAttribute(       Clazz clazz, Field  field,                               RuntimeVisibleTypeAnnotationsAttribute   runtimeVisibleTypeAnnotationsAttribute);
    public void visitRuntimeVisibleTypeAnnotationsAttribute(       Clazz clazz, Method method,                              RuntimeVisibleTypeAnnotationsAttribute   runtimeVisibleTypeAnnotationsAttribute);
    public void visitRuntimeVisibleTypeAnnotationsAttribute(       Clazz clazz, Method method, CodeAttribute codeAttribute, RuntimeVisibleTypeAnnotationsAttribute   runtimeVisibleTypeAnnotationsAttribute);

    public void visitRuntimeInvisibleTypeAnnotationsAttribute(     Clazz clazz,                                             RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute);
    public void visitRuntimeInvisibleTypeAnnotationsAttribute(     Clazz clazz, Field  field,                               RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute);
    public void visitRuntimeInvisibleTypeAnnotationsAttribute(     Clazz clazz, Method method,                              RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute);
    public void visitRuntimeInvisibleTypeAnnotationsAttribute(     Clazz clazz, Method method, CodeAttribute codeAttribute, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute);

    public void visitAnnotationDefaultAttribute(                   Clazz clazz, Method method, AnnotationDefaultAttribute annotationDefaultAttribute);
}
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
package proguard.optimize;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.annotation.*;
import proguard.classfile.attribute.module.*;
import proguard.classfile.attribute.preverification.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.util.ClassUtil;

/**
 * This AttributeVisitor delegates its call to another AttributeVisitor, and
 * prints out the code if the other visitor has changed it.
 *
 * @author Eric Lafortune
 */
public class ChangedCodePrinter
implements   AttributeVisitor
{
    private final AttributeVisitor attributeVisitor;


    public ChangedCodePrinter(AttributeVisitor attributeVisitor)
    {
        this.attributeVisitor = attributeVisitor;
    }


    // Implementations for AttributeVisitor.

    public void visitUnknownAttribute(Clazz clazz, UnknownAttribute unknownAttribute)
    {
        attributeVisitor.visitUnknownAttribute(clazz, unknownAttribute);
    }


    public void visitBootstrapMethodsAttribute(Clazz clazz, BootstrapMethodsAttribute bootstrapMethodsAttribute)
    {
        attributeVisitor.visitBootstrapMethodsAttribute(clazz, bootstrapMethodsAttribute);
    }


    public void visitSourceFileAttribute(Clazz clazz, SourceFileAttribute sourceFileAttribute)
    {
        attributeVisitor.visitSourceFileAttribute(clazz, sourceFileAttribute);
    }


    public void visitSourceDirAttribute(Clazz clazz, SourceDirAttribute sourceDirAttribute)
    {
        attributeVisitor.visitSourceDirAttribute(clazz, sourceDirAttribute);
    }


    public void visitInnerClassesAttribute(Clazz clazz, InnerClassesAttribute innerClassesAttribute)
    {
        attributeVisitor.visitInnerClassesAttribute(clazz, innerClassesAttribute);
    }


    public void visitEnclosingMethodAttribute(Clazz clazz, EnclosingMethodAttribute enclosingMethodAttribute)
    {
        attributeVisitor.visitEnclosingMethodAttribute(clazz, enclosingMethodAttribute);
    }


    public void visitNestHostAttribute(Clazz clazz, NestHostAttribute nestHostAttribute)
    {
        attributeVisitor.visitNestHostAttribute(clazz, nestHostAttribute);
    }


    public void visitNestMembersAttribute(Clazz clazz, NestMembersAttribute nestMembersAttribute)
    {
        attributeVisitor.visitNestMembersAttribute(clazz, nestMembersAttribute);
    }


    public void visitModuleAttribute(Clazz clazz, ModuleAttribute moduleAttribute)
    {
        attributeVisitor.visitModuleAttribute(clazz, moduleAttribute);
    }


    public void visitModuleMainClassAttribute(Clazz clazz, ModuleMainClassAttribute moduleMainClassAttribute)
    {
        attributeVisitor.visitModuleMainClassAttribute(clazz, moduleMainClassAttribute);
    }


    public void visitModulePackagesAttribute(Clazz clazz, ModulePackagesAttribute modulePackagesAttribute)
    {
        attributeVisitor.visitModulePackagesAttribute(clazz, modulePackagesAttribute);
    }


    public void visitDeprecatedAttribute(Clazz clazz, DeprecatedAttribute deprecatedAttribute)
    {
        attributeVisitor.visitDeprecatedAttribute(clazz, deprecatedAttribute);
    }


    public void visitSyntheticAttribute(Clazz clazz, SyntheticAttribute syntheticAttribute)
    {
        attributeVisitor.visitSyntheticAttribute(clazz, syntheticAttribute);
    }


    public void visitSignatureAttribute(Clazz clazz, SignatureAttribute syntheticAttribute)
    {
        attributeVisitor.visitSignatureAttribute(clazz, syntheticAttribute);
    }


    public void visitDeprecatedAttribute(Clazz clazz, Field field, DeprecatedAttribute deprecatedAttribute)
    {
        attributeVisitor.visitDeprecatedAttribute(clazz, field, deprecatedAttribute);
    }


    public void visitSyntheticAttribute(Clazz clazz, Field field, SyntheticAttribute syntheticAttribute)
    {
        attributeVisitor.visitSyntheticAttribute(clazz, field, syntheticAttribute);
    }


    public void visitSignatureAttribute(Clazz clazz, Field field, SignatureAttribute syntheticAttribute)
    {
        attributeVisitor.visitSignatureAttribute(clazz, field, syntheticAttribute);
    }


    public void visitDeprecatedAttribute(Clazz clazz, Method method, DeprecatedAttribute deprecatedAttribute)
    {
        attributeVisitor.visitDeprecatedAttribute(clazz, method, deprecatedAttribute);
    }


    public void visitSyntheticAttribute(Clazz clazz, Method method, SyntheticAttribute syntheticAttribute)
    {
        attributeVisitor.visitSyntheticAttribute(clazz, method, syntheticAttribute);
    }


    public void visitSignatureAttribute(Clazz clazz, Method method, SignatureAttribute syntheticAttribute)
    {
        attributeVisitor.visitSignatureAttribute(clazz, method, syntheticAttribute);
    }


    public void visitConstantValueAttribute(Clazz clazz, Field field, ConstantValueAttribute constantValueAttribute)
    {
        attributeVisitor.visitConstantValueAttribute(clazz, field, constantValueAttribute);
    }


    public void visitMethodParametersAttribute(Clazz clazz, Method method, MethodParametersAttribute exceptionsAttribute)
    {
        attributeVisitor.visitMethodParametersAttribute(clazz, method, exceptionsAttribute);
    }


    public void visitExceptionsAttribute(Clazz clazz, Method method, ExceptionsAttribute exceptionsAttribute)
    {
        attributeVisitor.visitExceptionsAttribute(clazz, method, exceptionsAttribute);
    }


    public void visitStackMapAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapAttribute stackMapAttribute)
    {
        attributeVisitor.visitStackMapAttribute(clazz, method, codeAttribute, stackMapAttribute);
    }


    public void visitStackMapTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapTableAttribute stackMapTableAttribute)
    {
        attributeVisitor.visitStackMapTableAttribute(clazz, method, codeAttribute, stackMapTableAttribute);
    }


    public void visitLineNumberTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute lineNumberTableAttribute)
    {
        attributeVisitor.visitLineNumberTableAttribute(clazz, method, codeAttribute, lineNumberTableAttribute);
    }


    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        attributeVisitor.visitLocalVariableTableAttribute(clazz, method, codeAttribute, localVariableTableAttribute);
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        attributeVisitor.visitLocalVariableTypeTableAttribute(clazz, method, codeAttribute, localVariableTypeTableAttribute);
    }


    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        attributeVisitor.visitRuntimeVisibleAnnotationsAttribute(clazz, runtimeVisibleAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        attributeVisitor.visitRuntimeInvisibleAnnotationsAttribute(clazz, runtimeInvisibleAnnotationsAttribute);
    }


    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, Field field, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        attributeVisitor.visitRuntimeVisibleAnnotationsAttribute(clazz, field, runtimeVisibleAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, Field field, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        attributeVisitor.visitRuntimeInvisibleAnnotationsAttribute(clazz, field, runtimeInvisibleAnnotationsAttribute);
    }


    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        attributeVisitor.visitRuntimeVisibleAnnotationsAttribute(clazz, method, runtimeVisibleAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        attributeVisitor.visitRuntimeInvisibleAnnotationsAttribute(clazz, method, runtimeInvisibleAnnotationsAttribute);
    }


    public void visitRuntimeVisibleParameterAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleParameterAnnotationsAttribute runtimeVisibleParameterAnnotationsAttribute)
    {
        attributeVisitor.visitRuntimeVisibleParameterAnnotationsAttribute(clazz, method, runtimeVisibleParameterAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleParameterAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleParameterAnnotationsAttribute runtimeInvisibleParameterAnnotationsAttribute)
    {
        attributeVisitor.visitRuntimeInvisibleParameterAnnotationsAttribute(clazz, method, runtimeInvisibleParameterAnnotationsAttribute);
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        attributeVisitor.visitRuntimeVisibleTypeAnnotationsAttribute(clazz, runtimeVisibleTypeAnnotationsAttribute);
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Field field, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        attributeVisitor.visitRuntimeVisibleTypeAnnotationsAttribute(clazz, field, runtimeVisibleTypeAnnotationsAttribute);
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        attributeVisitor.visitRuntimeVisibleTypeAnnotationsAttribute(clazz, method, runtimeVisibleTypeAnnotationsAttribute);
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        attributeVisitor.visitRuntimeVisibleTypeAnnotationsAttribute(clazz, method, codeAttribute, runtimeVisibleTypeAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        attributeVisitor.visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, runtimeInvisibleTypeAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Field field, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        attributeVisitor.visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, field, runtimeInvisibleTypeAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        attributeVisitor.visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, method, runtimeInvisibleTypeAnnotationsAttribute);
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        attributeVisitor.visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, method, codeAttribute, runtimeInvisibleTypeAnnotationsAttribute);
    }


    public void visitAnnotationDefaultAttribute(Clazz clazz, Method method, AnnotationDefaultAttribute annotationDefaultAttribute)
    {
        attributeVisitor.visitAnnotationDefaultAttribute(clazz, method, annotationDefaultAttribute);
    }


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        byte[] code    = codeAttribute.code;
        byte[] oldCode = new byte[code.length];

        // Copy the current code.
        System.arraycopy(code, 0, oldCode, 0, codeAttribute.u4codeLength);

        // Delegate to the real visitor.
        attributeVisitor.visitCodeAttribute(clazz, method, codeAttribute);

        // Check if the code has changed.
        if (codeHasChanged(codeAttribute, oldCode))
        {
            printChangedCode(clazz, method, codeAttribute, oldCode);
        }
    }


    // Small utility methods.

    private boolean codeHasChanged(CodeAttribute codeAttribute, byte[] oldCode)
    {
        if (oldCode.length != codeAttribute.u4codeLength)
        {
            return true;
        }

        for (int index = 0; index < codeAttribute.u4codeLength; index++)
        {
            if (oldCode[index] != codeAttribute.code[index])
            {
                return true;
            }
        }

        return false;
    }


    private void printChangedCode(Clazz         clazz,
                                  Method        method,
                                  CodeAttribute codeAttribute,
                                  byte[]        oldCode)
    {
        System.out.println("Class "+ClassUtil.externalClassName(clazz.getName()));
        System.out.println("Method "+ClassUtil.externalFullMethodDescription(clazz.getName(),
                                                                             0,
                                                                             method.getName(clazz),
                                                                             method.getDescriptor(clazz)));

        for (int index = 0; index < codeAttribute.u4codeLength; index++)
        {
            System.out.println(
                (oldCode[index] == codeAttribute.code[index]? "  -- ":"  => ")+
                index+": "+
                Integer.toHexString(0x100|oldCode[index]           &0xff).substring(1)+" "+
                Integer.toHexString(0x100|codeAttribute.code[index]&0xff).substring(1));
        }
    }
}

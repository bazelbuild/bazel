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
import proguard.classfile.visitor.SimpleClassPrinter;

/**
 * This AttributeVisitor delegates to a given AttributeVisitor, timing the
 * invocations and printing out warnings when the timings exceed a given
 * threshold.
 *
 * @author Eric Lafortune
 */
public class DebugAttributeVisitor
implements   AttributeVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    public  static       boolean DEBUG = System.getProperty("dav") != null;
    //*/


    private final String           message;
    private final long             maximumTime;
    private final AttributeVisitor attributeVisitor;


    /**
     * Creates a new DebugAttributeVisitor.
     * @param attributeVisitor the AttributeVisitor to which visits will be
     *                         delegated.
     */
    public DebugAttributeVisitor(AttributeVisitor attributeVisitor)
    {
        this(attributeVisitor.getClass().getName(),
             attributeVisitor);
    }


    /**
     * Creates a new DebugAttributeVisitor.
     * @param message          the message to be printed when the maximum
     *                         invocation time is exceeded.
     * @param attributeVisitor the AttributeVisitor to which visits will be
     *                         delegated.
     */
    public DebugAttributeVisitor(String           message,
                                 AttributeVisitor attributeVisitor)
    {
        this(message,
             10000L,
             attributeVisitor);
    }


    /**
     * Creates a new DebugAttributeVisitor.
     * @param message          the message to be printed when the maximum
     *                         invocation time is exceeded.
     * @param maximumTime      the maximum invocation time.
     * @param attributeVisitor the AttributeVisitor to which visits will be
     *                         delegated.
     */
    public DebugAttributeVisitor(String           message,
                                 long             maximumTime,
                                 AttributeVisitor attributeVisitor)
    {
        String debugTime = System.getProperty("debug.time");

        this.message          = message;
        this.maximumTime      = debugTime != null ? Long.valueOf(debugTime) : maximumTime;
        this.attributeVisitor = attributeVisitor;
    }


    public void visitLibraryClass(LibraryClass libraryClass) {}


    // Implementations for AttributeVisitor.

    public void visitUnknownAttribute(Clazz clazz, UnknownAttribute unknownAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitUnknownAttribute(clazz, unknownAttribute);

        checkTime(clazz, unknownAttribute, startTime);
    }

    public void visitBootstrapMethodsAttribute(Clazz clazz, BootstrapMethodsAttribute bootstrapMethodsAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitBootstrapMethodsAttribute(clazz, bootstrapMethodsAttribute);

        checkTime(clazz, bootstrapMethodsAttribute, startTime);
    }

    public void visitSourceFileAttribute(Clazz clazz, SourceFileAttribute sourceFileAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitSourceFileAttribute(clazz, sourceFileAttribute);

        checkTime(clazz, sourceFileAttribute, startTime);
    }

    public void visitSourceDirAttribute(Clazz clazz, SourceDirAttribute sourceDirAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitSourceDirAttribute(clazz, sourceDirAttribute);

        checkTime(clazz, sourceDirAttribute, startTime);
    }

    public void visitInnerClassesAttribute(Clazz clazz, InnerClassesAttribute innerClassesAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitInnerClassesAttribute(clazz, innerClassesAttribute);

        checkTime(clazz, innerClassesAttribute, startTime);
    }

    public void visitEnclosingMethodAttribute(Clazz clazz, EnclosingMethodAttribute enclosingMethodAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitEnclosingMethodAttribute(clazz, enclosingMethodAttribute);

        checkTime(clazz, enclosingMethodAttribute, startTime);
    }


    public void visitNestHostAttribute(Clazz clazz, NestHostAttribute nestHostAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitNestHostAttribute(clazz, nestHostAttribute);

        checkTime(clazz, nestHostAttribute, startTime);
    }


    public void visitNestMembersAttribute(Clazz clazz, NestMembersAttribute nestMembersAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitNestMembersAttribute(clazz, nestMembersAttribute);

        checkTime(clazz, nestMembersAttribute, startTime);
    }


    public void visitModuleAttribute(Clazz clazz, ModuleAttribute moduleAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitModuleAttribute(clazz, moduleAttribute);

        checkTime(clazz, moduleAttribute, startTime);
    }


    public void visitModuleMainClassAttribute(Clazz clazz, ModuleMainClassAttribute moduleMainClassAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitModuleMainClassAttribute(clazz, moduleMainClassAttribute);

        checkTime(clazz, moduleMainClassAttribute, startTime);
    }


    public void visitModulePackagesAttribute(Clazz clazz, ModulePackagesAttribute modulePackagesAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitModulePackagesAttribute(clazz, modulePackagesAttribute);

        checkTime(clazz, modulePackagesAttribute, startTime);
    }


    public void visitDeprecatedAttribute(Clazz clazz, DeprecatedAttribute deprecatedAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitDeprecatedAttribute(clazz, deprecatedAttribute);

        checkTime(clazz, deprecatedAttribute, startTime);
    }

    public void visitDeprecatedAttribute(Clazz clazz, Field field, DeprecatedAttribute deprecatedAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitDeprecatedAttribute(clazz, field, deprecatedAttribute);

        checkTime(clazz, field, deprecatedAttribute, startTime);
    }

    public void visitDeprecatedAttribute(Clazz clazz, Method method, DeprecatedAttribute deprecatedAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitDeprecatedAttribute(clazz, method, deprecatedAttribute);

        checkTime(clazz, method, deprecatedAttribute, startTime);
    }

    public void visitSyntheticAttribute(Clazz clazz, SyntheticAttribute syntheticAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitSyntheticAttribute(clazz, syntheticAttribute);

        checkTime(clazz, syntheticAttribute, startTime);
    }

    public void visitSyntheticAttribute(Clazz clazz, Field field, SyntheticAttribute syntheticAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitSyntheticAttribute(clazz, field, syntheticAttribute);

        checkTime(clazz, field, syntheticAttribute, startTime);
    }

    public void visitSyntheticAttribute(Clazz clazz, Method method, SyntheticAttribute syntheticAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitSyntheticAttribute(clazz, method, syntheticAttribute);

        checkTime(clazz, method, syntheticAttribute, startTime);
    }

    public void visitSignatureAttribute(Clazz clazz, SignatureAttribute signatureAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitSignatureAttribute(clazz, signatureAttribute);

        checkTime(clazz, signatureAttribute, startTime);
    }

    public void visitSignatureAttribute(Clazz clazz, Field field, SignatureAttribute signatureAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitSignatureAttribute(clazz, field, signatureAttribute);

        checkTime(clazz, field, signatureAttribute, startTime);
    }

    public void visitSignatureAttribute(Clazz clazz, Method method, SignatureAttribute signatureAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitSignatureAttribute(clazz, method, signatureAttribute);

        checkTime(clazz, method, signatureAttribute, startTime);
    }

    public void visitConstantValueAttribute(Clazz clazz, Field field, ConstantValueAttribute constantValueAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitConstantValueAttribute(clazz, field, constantValueAttribute);

        checkTime(clazz, field, constantValueAttribute, startTime);
    }

    public void visitMethodParametersAttribute(Clazz clazz, Method method, MethodParametersAttribute methodParametersAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitMethodParametersAttribute(clazz, method, methodParametersAttribute);

        checkTime(clazz, method, methodParametersAttribute, startTime);
    }

    public void visitExceptionsAttribute(Clazz clazz, Method method, ExceptionsAttribute exceptionsAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitExceptionsAttribute(clazz, method, exceptionsAttribute);

        checkTime(clazz, method, exceptionsAttribute, startTime);
    }

    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitCodeAttribute(clazz, method, codeAttribute);

        checkTime(clazz, method, codeAttribute, startTime);
    }

    public void visitStackMapAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapAttribute stackMapAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitStackMapAttribute(clazz, method, codeAttribute, stackMapAttribute);

        checkTime(clazz, method, codeAttribute, startTime);
    }

    public void visitStackMapTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapTableAttribute stackMapTableAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitStackMapTableAttribute(clazz, method, codeAttribute, stackMapTableAttribute);

        checkTime(clazz, method, codeAttribute, startTime);
    }

    public void visitLineNumberTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute lineNumberTableAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitLineNumberTableAttribute(clazz, method, codeAttribute, lineNumberTableAttribute);

        checkTime(clazz, method, codeAttribute, startTime);
    }

    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitLocalVariableTableAttribute(clazz, method, codeAttribute, localVariableTableAttribute);

        checkTime(clazz, method, codeAttribute, startTime);
    }

    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitLocalVariableTypeTableAttribute(clazz, method, codeAttribute, localVariableTypeTableAttribute);

        checkTime(clazz, method, codeAttribute, startTime);
    }

    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitRuntimeVisibleAnnotationsAttribute(clazz, runtimeVisibleAnnotationsAttribute);

        checkTime(clazz, runtimeVisibleAnnotationsAttribute, startTime);
    }

    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, Field field, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitRuntimeVisibleAnnotationsAttribute(clazz, field, runtimeVisibleAnnotationsAttribute);

        checkTime(clazz, field, runtimeVisibleAnnotationsAttribute, startTime);
    }

    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitRuntimeVisibleAnnotationsAttribute(clazz, method, runtimeVisibleAnnotationsAttribute);

        checkTime(clazz, method, runtimeVisibleAnnotationsAttribute, startTime);
    }

    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitRuntimeInvisibleAnnotationsAttribute(clazz, runtimeInvisibleAnnotationsAttribute);

        checkTime(clazz, runtimeInvisibleAnnotationsAttribute, startTime);
    }

    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, Field field, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitRuntimeInvisibleAnnotationsAttribute(clazz, field, runtimeInvisibleAnnotationsAttribute);

        checkTime(clazz, field, runtimeInvisibleAnnotationsAttribute, startTime);
    }

    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitRuntimeInvisibleAnnotationsAttribute(clazz, method, runtimeInvisibleAnnotationsAttribute);

        checkTime(clazz, method, runtimeInvisibleAnnotationsAttribute, startTime);
    }

    public void visitRuntimeVisibleParameterAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleParameterAnnotationsAttribute runtimeVisibleParameterAnnotationsAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitRuntimeVisibleParameterAnnotationsAttribute(clazz, method, runtimeVisibleParameterAnnotationsAttribute);

        checkTime(clazz, method, runtimeVisibleParameterAnnotationsAttribute, startTime);
    }

    public void visitRuntimeInvisibleParameterAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleParameterAnnotationsAttribute runtimeInvisibleParameterAnnotationsAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitRuntimeInvisibleParameterAnnotationsAttribute(clazz, method, runtimeInvisibleParameterAnnotationsAttribute);

        checkTime(clazz, method, runtimeInvisibleParameterAnnotationsAttribute, startTime);
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitRuntimeVisibleTypeAnnotationsAttribute(clazz, runtimeVisibleTypeAnnotationsAttribute);

        checkTime(clazz, runtimeVisibleTypeAnnotationsAttribute, startTime);
    }

    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Field field, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitRuntimeVisibleTypeAnnotationsAttribute(clazz, field, runtimeVisibleTypeAnnotationsAttribute);

        checkTime(clazz, field, runtimeVisibleTypeAnnotationsAttribute, startTime);
    }

    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitRuntimeVisibleTypeAnnotationsAttribute(clazz, method, runtimeVisibleTypeAnnotationsAttribute);

        checkTime(clazz, method, runtimeVisibleTypeAnnotationsAttribute, startTime);
    }

    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitRuntimeVisibleTypeAnnotationsAttribute(clazz, method, codeAttribute, runtimeVisibleTypeAnnotationsAttribute);

        checkTime(clazz, method, runtimeVisibleTypeAnnotationsAttribute, startTime);
    }

    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, runtimeInvisibleTypeAnnotationsAttribute);

        checkTime(clazz, runtimeInvisibleTypeAnnotationsAttribute, startTime);
    }

    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Field field, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, field, runtimeInvisibleTypeAnnotationsAttribute);

        checkTime(clazz, field, runtimeInvisibleTypeAnnotationsAttribute, startTime);
    }

    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, method, runtimeInvisibleTypeAnnotationsAttribute);

        checkTime(clazz, method, runtimeInvisibleTypeAnnotationsAttribute, startTime);
    }

    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitRuntimeInvisibleTypeAnnotationsAttribute(clazz, method, codeAttribute, runtimeInvisibleTypeAnnotationsAttribute);

        checkTime(clazz, method, runtimeInvisibleTypeAnnotationsAttribute, startTime);
    }


    public void visitAnnotationDefaultAttribute(Clazz clazz, Method method, AnnotationDefaultAttribute annotationDefaultAttribute)
    {
        long startTime = startTime();

        attributeVisitor.visitAnnotationDefaultAttribute(clazz, method, annotationDefaultAttribute);

        checkTime(clazz, method, annotationDefaultAttribute, startTime);
    }


    // Small utility methods.

    /**
     * Returns the current time.
     */
    private long startTime()
    {
        return DEBUG ? System.currentTimeMillis() : 0L;
    }


    /**
     * Checks the time after having visited the given attribute, and prints
     * out a message if necessary.
     */
    private void checkTime(Clazz     clazz,
                           Attribute attribute,
                           long      startTime)
    {
        if (DEBUG)
        {
            long endTime = startTime();

            long deltaTime = endTime - startTime;

            if (deltaTime > maximumTime)
            {
                System.err.println("=== " + message + " took "+((double)deltaTime/1000.)+" seconds ===");
                //attribute.accept(clazz, new ClassPrinter());
                System.err.println();
            }
        }
    }


    /**
     * Checks the time after having visited the given attribute, and prints
     * out a message if necessary.
     */
    private void checkTime(Clazz     clazz,
                           Field     field,
                           Attribute attribute,
                           long      startTime)
    {
        if (DEBUG)
        {
            long endTime = startTime();

            long deltaTime = endTime - startTime;

            if (deltaTime > maximumTime)
            {
                System.err.println("=== " + message + " took "+((double)deltaTime/1000.)+" seconds ===");
                field.accept(clazz, new SimpleClassPrinter(true));
                System.err.println();
            }
        }
    }


    /**
     * Checks the time after having visited the given attribute, and prints
     * out a message if necessary.
     */
    private void checkTime(Clazz     clazz,
                           Method    method,
                           Attribute attribute,
                           long      startTime)
    {
        if (DEBUG)
        {
            long endTime = startTime();

            long deltaTime = endTime - startTime;

            if (deltaTime > maximumTime)
            {
                System.err.println("=== " + message + " took "+((double)deltaTime/1000.)+" seconds ===");
                method.accept(clazz, new SimpleClassPrinter(true));
                System.err.println();
            }
        }
    }
}

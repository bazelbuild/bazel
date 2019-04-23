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
package proguard.backport;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.ClassVisitor;
import proguard.util.ArrayUtil;

import java.util.*;

/**
 * This ClassVisitor collects all lambda expressions that are defined in
 * a visited class.
 *
 * @author Thomas Neidhart
 */
public class LambdaExpressionCollector
extends    SimplifiedVisitor
implements ClassVisitor,

           // Implementation interfaces.
           ConstantVisitor,
           AttributeVisitor,
           BootstrapMethodInfoVisitor
{
    private final Map<Integer, LambdaExpression> lambdaExpressions;

    private InvokeDynamicConstant referencedInvokeDynamicConstant;
    private int                   referencedBootstrapMethodIndex;
    private Clazz                 referencedInvokedClass;
    private Method                referencedInvokedMethod;


    public LambdaExpressionCollector(Map<Integer, LambdaExpression> lambdaExpressions)
    {
        this.lambdaExpressions = lambdaExpressions;
    }


    // Implementations for ClassVisitor.

    @Override
    public void visitLibraryClass(LibraryClass libraryClass) {}


    @Override
    public void visitProgramClass(ProgramClass programClass)
    {
        // Visit any InvokeDynamic constant.
        programClass.constantPoolEntriesAccept(
            new ConstantTagFilter(ClassConstants.CONSTANT_InvokeDynamic,
            this));
    }


    // Implementations for ConstantVisitor.

    @Override
    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    @Override
    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        referencedInvokeDynamicConstant = invokeDynamicConstant;
        referencedBootstrapMethodIndex  = invokeDynamicConstant.getBootstrapMethodAttributeIndex();
        clazz.attributesAccept(this);
    }


    @Override
    public void visitAnyMethodrefConstant(Clazz clazz, RefConstant refConstant)
    {
        referencedInvokedClass  = refConstant.referencedClass;
        referencedInvokedMethod = (Method) refConstant.referencedMember;
    }


    // Implementations for AttributeVisitor.

    @Override
    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    @Override
    public void visitBootstrapMethodsAttribute(Clazz                     clazz,
                                               BootstrapMethodsAttribute bootstrapMethodsAttribute)
    {
        bootstrapMethodsAttribute.bootstrapMethodEntryAccept(clazz, referencedBootstrapMethodIndex, this);
    }


    // Implementations for BootstrapMethodInfoVisitor.

    @Override
    public void visitBootstrapMethodInfo(Clazz clazz, BootstrapMethodInfo bootstrapMethodInfo)
    {
        ProgramClass programClass = (ProgramClass) clazz;

        MethodHandleConstant bootstrapMethodHandle =
            (MethodHandleConstant) programClass.getConstant(bootstrapMethodInfo.u2methodHandleIndex);

        if (isLambdaMetaFactory(bootstrapMethodHandle.getClassName(clazz)))
        {
            String factoryMethodDescriptor =
                referencedInvokeDynamicConstant.getType(clazz);

            String interfaceClassName =
                ClassUtil.internalClassNameFromClassType(ClassUtil.internalMethodReturnType(factoryMethodDescriptor));

            // Find the actual method that is being invoked.
            MethodHandleConstant invokedMethodHandle =
                (MethodHandleConstant) programClass.getConstant(bootstrapMethodInfo.u2methodArguments[1]);

            referencedInvokedClass  = null;
            referencedInvokedMethod = null;
            clazz.constantPoolEntryAccept(invokedMethodHandle.u2referenceIndex, this);

            // Collect all the useful information.
            LambdaExpression lambdaExpression =
                new LambdaExpression(programClass,
                                     referencedBootstrapMethodIndex,
                                     bootstrapMethodInfo,
                                     factoryMethodDescriptor,
                                     new String[] { interfaceClassName },
                                     new String[0],
                                     referencedInvokeDynamicConstant.getName(clazz),
                                     getMethodTypeConstant(programClass, bootstrapMethodInfo.u2methodArguments[0]).getType(clazz),
                                     invokedMethodHandle.getReferenceKind(),
                                     invokedMethodHandle.getClassName(clazz),
                                     invokedMethodHandle.getName(clazz),
                                     invokedMethodHandle.getType(clazz),
                                     referencedInvokedClass,
                                     referencedInvokedMethod);

            if (isAlternateFactoryMethod(bootstrapMethodHandle.getName(clazz)))
            {
                int flags =
                    getIntegerConstant(programClass,
                                       bootstrapMethodInfo.u2methodArguments[3]);

                // For the alternate metafactory, the optional arguments start
                // at index 4.
                int argumentIndex = 4;

                if ((flags & ClassConstants.FLAG_MARKERS) != 0)
                {
                    int markerInterfaceCount =
                        getIntegerConstant(programClass,
                                           bootstrapMethodInfo.u2methodArguments[argumentIndex++]);

                    for (int i = 0; i < markerInterfaceCount; i++)
                    {
                        String interfaceName =
                            programClass.getClassName(bootstrapMethodInfo.u2methodArguments[argumentIndex++]);

                        lambdaExpression.interfaces =
                            ArrayUtil.add(lambdaExpression.interfaces,
                                          lambdaExpression.interfaces.length,
                                          interfaceName);
                    }
                }

                if ((flags & ClassConstants.FLAG_BRIDGES) != 0)
                {
                    int bridgeMethodCount =
                        getIntegerConstant(programClass,
                                           bootstrapMethodInfo.u2methodArguments[argumentIndex++]);

                    for (int i = 0; i < bridgeMethodCount; i++)
                    {
                        MethodTypeConstant methodTypeConstant =
                            getMethodTypeConstant(programClass,
                                                  bootstrapMethodInfo.u2methodArguments[argumentIndex++]);

                        lambdaExpression.bridgeMethodDescriptors =
                            ArrayUtil.add(lambdaExpression.bridgeMethodDescriptors,
                                          lambdaExpression.bridgeMethodDescriptors.length,
                                          methodTypeConstant.getType(programClass));
                    }
                }

                if ((flags & ClassConstants.FLAG_SERIALIZABLE) != 0)
                {
                    lambdaExpression.interfaces =
                        ArrayUtil.add(lambdaExpression.interfaces,
                                      lambdaExpression.interfaces.length,
                                      ClassConstants.NAME_JAVA_IO_SERIALIZABLE);
                }
            }

            lambdaExpressions.put(referencedBootstrapMethodIndex, lambdaExpression);
        }
    }

    // Small utility methods

    private static final String NAME_JAVA_LANG_INVOKE_LAMBDA_METAFACTORY = "java/lang/invoke/LambdaMetafactory";
    private static final String LAMBDA_ALTERNATE_METAFACTORY_METHOD      = "altMetafactory";

    private static boolean isLambdaMetaFactory(String className)
    {
        return NAME_JAVA_LANG_INVOKE_LAMBDA_METAFACTORY.equals(className);
    }

    private static boolean isAlternateFactoryMethod(String methodName)
    {
        return LAMBDA_ALTERNATE_METAFACTORY_METHOD.equals(methodName);
    }

    private static int getIntegerConstant(ProgramClass programClass, int constantIndex)
    {
        return ((IntegerConstant) programClass.getConstant(constantIndex)).getValue();
    }


    private static MethodTypeConstant getMethodTypeConstant(ProgramClass programClass, int constantIndex)
    {
        return (MethodTypeConstant) programClass.getConstant(constantIndex);
    }
}
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
import proguard.classfile.attribute.BootstrapMethodInfo;
import proguard.classfile.util.*;

/**
 * A small helper class that captures useful information
 * about a lambda expression as encountered in a class file.
 *
 * @author Thomas Neidhart
 */
public class LambdaExpression
{
    // The referenced class of the lambda expression.
    public ProgramClass referencedClass;

    // The referenced bootstrap method index.
    public int                 bootstrapMethodIndex;
    // The referenced bootstrap method info.
    public BootstrapMethodInfo bootstrapMethodInfo;

    // The lambda factory method type.
    public String factoryMethodDescriptor;

    // The implemented interfaces of the Lambda expression.
    public String[] interfaces;

    // The additional bridge method descriptors to be added.
    public String[] bridgeMethodDescriptors;

    // The name and descriptor of the implemented interface method.
    public String interfaceMethod;
    public String interfaceMethodDescriptor;

    // Information regarding the invoked method.
    public int    invokedReferenceKind;
    public String invokedClassName;
    public String invokedMethodName;
    public String invokedMethodDesc;

    public Clazz  referencedInvokedClass;
    public Method referencedInvokedMethod;

    // The created lambda class.
    public ProgramClass lambdaClass;


    /**
     * Creates a new initialized LambdaExpression (except for the lambdaClass).
     */
    public LambdaExpression(ProgramClass        referencedClass,
                            int                 bootstrapMethodIndex,
                            BootstrapMethodInfo bootstrapMethodInfo,
                            String              factoryMethodDescriptor,
                            String[]            interfaces,
                            String[]            bridgeMethodDescriptors,
                            String              interfaceMethod,
                            String              interfaceMethodDescriptor,
                            int                 invokedReferenceKind,
                            String              invokedClassName,
                            String              invokedMethodName,
                            String              invokedMethodDesc,
                            Clazz               referencedInvokedClass,
                            Method              referencedInvokedMethod)
    {
        this.referencedClass           = referencedClass;
        this.bootstrapMethodIndex      = bootstrapMethodIndex;
        this.bootstrapMethodInfo       = bootstrapMethodInfo;
        this.factoryMethodDescriptor   = factoryMethodDescriptor;
        this.interfaces                = interfaces;
        this.bridgeMethodDescriptors   = bridgeMethodDescriptors;
        this.interfaceMethod           = interfaceMethod;
        this.interfaceMethodDescriptor = interfaceMethodDescriptor;
        this.invokedReferenceKind      = invokedReferenceKind;
        this.invokedClassName          = invokedClassName;
        this.invokedMethodName         = invokedMethodName;
        this.invokedMethodDesc         = invokedMethodDesc;
        this.referencedInvokedClass    = referencedInvokedClass;
        this.referencedInvokedMethod   = referencedInvokedMethod;
    }


    /**
     * Returns the class name of the converted anonymous class.
     */
    public String getLambdaClassName()
    {
        return String.format("%s$$Lambda$%d",
                             referencedClass.getName(),
                             bootstrapMethodIndex);
    }


    public String getConstructorDescriptor()
    {
        if (isStateless())
        {
            return ClassConstants.METHOD_TYPE_INIT;
        }
        else
        {
            int endIndex = factoryMethodDescriptor.indexOf(ClassConstants.METHOD_ARGUMENTS_CLOSE);

            return factoryMethodDescriptor.substring(0, endIndex + 1) + ClassConstants.TYPE_VOID;
        }
    }


    /**
     * Returns whether the lambda expression is serializable.
     */
    public boolean isSerializable()
    {
        for (String interfaceName : interfaces)
        {
            if (ClassConstants.NAME_JAVA_IO_SERIALIZABLE.equals(interfaceName))
            {
                return true;
            }
        }
        return false;
    }


    /**
     * Returns whether the lambda expression is actually a method reference.
     */
    public boolean isMethodReference()
    {
        return !isLambdaMethod(invokedMethodName);
    }


    /**
     * Returns whether the lambda expression is stateless.
     */
    public boolean isStateless()
    {
        // The lambda expression is stateless if the factory method does
        // not have arguments.
        return
            ClassUtil.internalMethodParameterCount(factoryMethodDescriptor) == 0;
    }


    /**
     * Returns whether the invoked method is a static interface method.
     */
    public boolean invokesStaticInterfaceMethod()
    {
        // We assume unknown classes are not interfaces.
        return invokedReferenceKind == ClassConstants.REF_invokeStatic &&
               referencedInvokedClass != null                          &&
               (referencedInvokedClass.getAccessFlags() & ClassConstants.ACC_INTERFACE) != 0;
    }


    /**
     * Returns whether the invoked method is a non-static, private synthetic
     * method in an interface.
     */
     boolean referencesPrivateSyntheticInterfaceMethod()
     {
         return (referencedInvokedClass .getAccessFlags() &  ClassConstants.ACC_INTERFACE)  != 0 &&
                (referencedInvokedMethod.getAccessFlags() & (ClassConstants.ACC_PRIVATE |
                                                             ClassConstants.ACC_SYNTHETIC)) != 0 ;
     }


    /**
     * Returns whether an accessor method is needed to access
     * the invoked method from the lambda class.
     */
    public boolean needsAccessorMethod()
    {
        // We assume unknown classes don't need an accessor method.
        return referencedInvokedClass != null &&
               new MemberFinder().findMethod(lambdaClass,
                                             referencedInvokedClass,
                                             invokedMethodName,
                                             invokedMethodDesc) == null;
    }


    /**
     * Returns whether the lambda expression is a method reference
     * to a private constructor.
     */
    public boolean referencesPrivateConstructor()
    {
        return invokedReferenceKind == ClassConstants.REF_newInvokeSpecial &&
               ClassConstants.METHOD_NAME_INIT.equals(invokedMethodName)   &&
               (referencedInvokedMethod.getAccessFlags() & ClassConstants.ACC_PRIVATE) != 0;
    }


    // Small Utility methods.

    private static final String LAMBDA_METHOD_PREFIX = "lambda$";

    private static boolean isLambdaMethod(String methodName)
    {
        return methodName.startsWith(LAMBDA_METHOD_PREFIX);
    }
}



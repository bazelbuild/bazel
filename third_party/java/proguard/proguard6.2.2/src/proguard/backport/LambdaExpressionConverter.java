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
import proguard.classfile.editor.*;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.util.MultiValueMap;

import java.util.*;

/**
 * This ClassVisitor converts all lambda expressions in the visited
 * classes to anonymous inner classes.
 *
 * @author Thomas Neidhart
 */
public class LambdaExpressionConverter
extends    SimplifiedVisitor
implements ClassVisitor,

           // Implementation interfaces.
           MemberVisitor,
           AttributeVisitor,
           InstructionVisitor
{
    private static final String LAMBDA_SINGLETON_FIELD_NAME = "INSTANCE";

    private final ClassPool                      programClassPool;
    private final ClassPool                      libraryClassPool;
    private final MultiValueMap<String, String>  injectedClassNameMap;
    private final ClassVisitor                   extraClassVisitor;

    private final Map<Integer, LambdaExpression> lambdaExpressionMap;
    private final CodeAttributeEditor            codeAttributeEditor;
    private final MemberRemover                  memberRemover;


    public LambdaExpressionConverter(ClassPool                     programClassPool,
                                     ClassPool                     libraryClassPool,
                                     MultiValueMap<String, String> injectedClassNameMap,
                                     ClassVisitor                  extraClassVisitor)
    {
        this.programClassPool     = programClassPool;
        this.libraryClassPool     = libraryClassPool;
        this.injectedClassNameMap = injectedClassNameMap;
        this.extraClassVisitor    = extraClassVisitor;

        this.lambdaExpressionMap  = new HashMap<Integer, LambdaExpression>();
        this.codeAttributeEditor  = new CodeAttributeEditor(true, true);
        this.memberRemover        = new MemberRemover();
    }


    // Implementations for ClassVisitor.

    @Override
    public void visitLibraryClass(LibraryClass libraryClass) {}


    @Override
    public void visitProgramClass(ProgramClass programClass)
    {
        lambdaExpressionMap.clear();
        programClass.accept(new LambdaExpressionCollector(lambdaExpressionMap));

        for (LambdaExpression lambdaExpression : lambdaExpressionMap.values())
        {
            ProgramClass lambdaClass = createLambdaClass(lambdaExpression);

            // Add the converted lambda class to the program class pool
            // and the injected class name map.
            programClassPool.addClass(lambdaClass);
            injectedClassNameMap.put(programClass.getName(), lambdaClass.getName());

            if (extraClassVisitor != null)
            {
                extraClassVisitor.visitProgramClass(lambdaClass);
            }
        }

        if (!lambdaExpressionMap.isEmpty())
        {
            // Replace all InvokeDynamic instructions.
            programClass.accept(
                new AllMethodVisitor(
                new AllAttributeVisitor(
                this)));

            // Initialize the hierarchy and references of all lambda classes.
            for (LambdaExpression lambdaExpression : lambdaExpressionMap.values())
            {
                lambdaExpression.lambdaClass.accept(
                    new MultiClassVisitor(
                        new ClassSuperHierarchyInitializer(programClassPool, libraryClassPool),
                        new ClassSubHierarchyInitializer(),
                        new ClassReferenceInitializer(programClassPool, libraryClassPool)
                    ));
            }

            // Remove deserialization hooks as they are no longer needed.
            programClass.methodsAccept(this);
            memberRemover.visitProgramClass(programClass);
        }
    }


    // Implementations for AttributeVisitor.

    @Override
    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    @Override
    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        codeAttributeEditor.reset(codeAttribute.u4codeLength);

        codeAttribute.instructionsAccept(clazz, method, this);

        if (codeAttributeEditor.isModified())
        {
            codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
        }
    }


    // Implementations for InstructionVisitor.

    @Override
    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    @Override
    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        if (constantInstruction.opcode == InstructionConstants.OP_INVOKEDYNAMIC)
        {
            ProgramClass programClass = (ProgramClass) clazz;

            InvokeDynamicConstant invokeDynamicConstant =
                (InvokeDynamicConstant) programClass.getConstant(constantInstruction.constantIndex);

            int bootstrapMethodIndex = invokeDynamicConstant.getBootstrapMethodAttributeIndex();
            if (lambdaExpressionMap.containsKey(bootstrapMethodIndex))
            {
                LambdaExpression lambdaExpression = lambdaExpressionMap.get(bootstrapMethodIndex);
                String lambdaClassName = lambdaExpression.getLambdaClassName();

                InstructionSequenceBuilder builder =
                    new InstructionSequenceBuilder(programClass);

                if (lambdaExpression.isStateless())
                {
                    builder.getstatic(lambdaClassName,
                                      LAMBDA_SINGLETON_FIELD_NAME,
                                      ClassUtil.internalTypeFromClassName(lambdaClassName));
                }
                else
                {
                    int maxLocals = codeAttribute.u2maxLocals;

                    String methodDescriptor =
                        lambdaExpression.getConstructorDescriptor();
                    int parameterSize =
                        ClassUtil.internalMethodParameterSize(methodDescriptor);

                    // TODO: the special optimization in case there is only 1
                    //       parameter has been disabled as the used stack
                    //       manipulation instructions might confuse the class
                    //       converter (testcase 1988).
                    if (parameterSize == 1 && false)
                    {
                        // If only 1 parameter is captured by the lambda expression,
                        // and it is a Category 1 value, we can avoid storing the
                        // current stack to variables.
                        builder.new_(lambdaClassName)
                               .dup_x1()
                               .swap()
                               .invokespecial(lambdaClassName,
                                              ClassConstants.METHOD_NAME_INIT,
                                              methodDescriptor);
                    }
                    else
                    {
                        // More than 1 (or a Category 2) parameter is captured
                        // by the lambda expression. We need to store the current
                        // call stack to variables, create the lambda instance and
                        // load the call stack again from the temporary variables.

                        // Collect the argument types.
                        InternalTypeEnumeration typeEnumeration =
                            new InternalTypeEnumeration(methodDescriptor);
                        List<String> types = new ArrayList<String>();
                        while(typeEnumeration.hasMoreTypes())
                        {
                            types.add(typeEnumeration.nextType());
                        }

                        // Store the current call stack in reverse order
                        // into temporary variables.
                        int variableIndex = maxLocals;
                        ListIterator<String> typeIterator =
                            types.listIterator(types.size());
                        while (typeIterator.hasPrevious())
                        {
                            String type = typeIterator.previous();

                            builder.store(variableIndex, type);
                            variableIndex += ClassUtil.internalTypeSize(type);
                        }

                        // Create the lambda instance.
                        builder.new_(lambdaClassName);
                        builder.dup();

                        // Reconstruct the call stack by loading it from
                        // the temporary variables.
                        typeIterator = types.listIterator();
                        while (typeIterator.hasNext())
                        {
                            String type = typeIterator.next();

                            int variableSize = ClassUtil.internalTypeSize(type);
                            variableIndex -= variableSize;
                            builder.load(variableIndex, type);
                        }

                        builder.invokespecial(lambdaClassName,
                                              ClassConstants.METHOD_NAME_INIT,
                                              methodDescriptor);
                    }
                }

                codeAttributeEditor.replaceInstruction(offset,
                                                       builder.instructions());
            }
        }
    }


    // Implementations for MemberVisitor.

    @Override
    public void visitAnyMember(Clazz clazz, Member member) {}


    @Override
    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        if (isDeserializationHook(programClass, programMethod))
        {
            memberRemover.visitProgramMethod(programClass, programMethod);
        }
    }


    // Small utility methods.

    private static final String METHOD_NAME_DESERIALIZE_LAMBDA = "$deserializeLambda$";
    private static final String METHOD_TYPE_DESERIALIZE_LAMBDA = "(Ljava/lang/invoke/SerializedLambda;)Ljava/lang/Object;";

    private static boolean isDeserializationHook(Clazz clazz, Method method)
    {
        return method.getName(clazz)      .equals(METHOD_NAME_DESERIALIZE_LAMBDA) &&
               method.getDescriptor(clazz).equals(METHOD_TYPE_DESERIALIZE_LAMBDA) &&
               hasFlag(method, ClassConstants.ACC_PRIVATE |
                               ClassConstants.ACC_STATIC  |
                               ClassConstants.ACC_SYNTHETIC);
    }

    private static boolean hasFlag(Member member, int flag)
    {
        return (member.getAccessFlags() & flag) == flag;
    }

    private ProgramClass createLambdaClass(LambdaExpression lambdaExpression)
    {
        String lambdaClassName = lambdaExpression.getLambdaClassName();

        ProgramClass lambdaClass =
            new ProgramClass(ClassConstants.CLASS_VERSION_1_5,
                             1,
                             new Constant[10],
                             0,
                             0,
                             0);

        ConstantPoolEditor constantPoolEditor = new ConstantPoolEditor(lambdaClass);

        lambdaClass.u2thisClass =
            constantPoolEditor.addClassConstant(lambdaClassName, lambdaClass);
        lambdaClass.u2superClass =
            constantPoolEditor.addClassConstant(ClassConstants.NAME_JAVA_LANG_OBJECT,
                                                null);

        ClassEditor classEditor = new ClassEditor(lambdaClass);

        String[] interfaces = lambdaExpression.interfaces;
        for (String interfaceName : interfaces)
        {
            classEditor.addInterface(constantPoolEditor.addClassConstant(interfaceName,
                                                                         null));
        }

        // Store the created lambda class in the LambdaExpression
        // data structure for later use.
        lambdaExpression.lambdaClass = lambdaClass;

        // [DGD-968] When a lambda expression is called from a `default`
        // interface method, ensure that it is stateless and visible to the
        // lambda class instead of generating an accessor method. The method
        // will be properly backported by the {@link StaticInterfaceMethodConverter}.
        if (lambdaExpression.referencesPrivateSyntheticInterfaceMethod())
        {
            fixInterfaceLambdaMethod(lambdaExpression.referencedClass,
                                     (ProgramMethod) lambdaExpression.referencedInvokedMethod,
                                     lambdaExpression);
        }
        else if (lambdaExpression.referencesPrivateConstructor() ||
                 lambdaExpression.needsAccessorMethod())
        {
            // In case the invoked method can not be accessed directly
            // by the lambda class, add a synthetic accessor method.

            addAccessorMethod(lambdaExpression.referencedClass,
                              lambdaExpression);
        }

        if (lambdaExpression.isStateless())
        {
            completeStatelessLambdaClass(lambdaClass, lambdaExpression);
        }
        else
        {
            completeCapturingLambdaClass(lambdaClass, lambdaExpression);
        }

        if (lambdaExpression.bridgeMethodDescriptors.length > 0)
        {
            addBridgeMethods(lambdaClass, lambdaExpression);
        }

        return lambdaClass;
    }


    private void fixInterfaceLambdaMethod(ProgramClass     programClass,
                                          ProgramMethod    programMethod,
                                          LambdaExpression lambdaExpression)
    {
        // Change the access flags to package private to make the method
        // accessible from the lambda class.
        programMethod.accept(programClass, new MemberAccessSetter(0));

        // If the method is not yet static, make it static
        // by updating its access flags / descriptor.
        if ((programMethod.getAccessFlags() & (ClassConstants.ACC_STATIC)) == 0)
        {
            programMethod.accept(programClass,
                new MemberAccessFlagSetter(ClassConstants.ACC_STATIC));

            String newDescriptor =
                prependParameterToMethodDescriptor(lambdaExpression.invokedMethodDesc,
                                                   ClassUtil.internalTypeFromClassType(programClass.getName()));

            programMethod.u2descriptorIndex =
                (new ConstantPoolEditor(programClass).addUtf8Constant(newDescriptor));

            // Update the lambda expression accordingly.
            lambdaExpression.invokedMethodDesc    = newDescriptor;
            lambdaExpression.invokedReferenceKind = ClassConstants.REF_invokeStatic;
        }
    }


    private void addAccessorMethod(ProgramClass     programClass,
                                   LambdaExpression lambdaExpression)
    {
        SimplifiedClassEditor classEditor = new SimplifiedClassEditor(programClass);

        String className = programClass.getName();

        // Create accessor method.
        String shortClassName =
            ClassUtil.externalShortClassName(
            ClassUtil.externalClassName(className));

        String accessorMethodName =
            String.format("accessor$%s$lambda%d",
                          shortClassName,
                          lambdaExpression.bootstrapMethodIndex);

        String accessorMethodDescriptor =
            lambdaExpression.invokedMethodDesc;
        int accessFlags =
            lambdaExpression.referencedInvokedMethod.getAccessFlags();

        // Method reference to a constructor.
        if (lambdaExpression.invokedReferenceKind == ClassConstants.REF_newInvokeSpecial)
        {
            // Replace the return type of the accessor method -> change to created type.

            // Collect first all parameters.
            List<String> invokedParameterTypes = new ArrayList<String>();
            int methodParameterSize =
                ClassUtil.internalMethodParameterSize(accessorMethodDescriptor);
            for (int i = 0; i < methodParameterSize; i++)
            {
                String invokedParameterType =
                    ClassUtil.internalMethodParameterType(accessorMethodDescriptor, i);
                invokedParameterTypes.add(invokedParameterType);
            }

            String invokedClassType =
                ClassUtil.internalTypeFromClassName(lambdaExpression.invokedClassName);

            // Build new method descriptor with the updated return type.
            accessorMethodDescriptor =
                ClassUtil.internalMethodDescriptorFromInternalTypes(invokedClassType,
                                                                    invokedParameterTypes);
        }
        else if ((accessFlags & ClassConstants.ACC_STATIC) == 0)
        {
            accessorMethodDescriptor =
                prependParameterToMethodDescriptor(accessorMethodDescriptor,
                                                   ClassUtil.internalTypeFromClassType(className));
        }

        CompactCodeAttributeComposer composer =
            classEditor.addMethod(ClassConstants.ACC_STATIC |
                                  ClassConstants.ACC_SYNTHETIC,
                                  accessorMethodName,
                                  accessorMethodDescriptor,
                                  50);

        // If the lambda expression is a method reference to a constructor,
        // we need to create the object first.
        if (lambdaExpression.invokedReferenceKind == ClassConstants.REF_newInvokeSpecial)
        {
            composer.new_(lambdaExpression.invokedClassName)
                .dup();
        }

        // Load the parameters next.
        InternalTypeEnumeration typeEnumeration =
            new InternalTypeEnumeration(accessorMethodDescriptor);

        completeInterfaceMethod(lambdaExpression,
                                composer,
                                0,
                                typeEnumeration,
                                null);

        classEditor.finishEditing();

        // Update the lambda expression to point to the created
        // accessor method instead.
        lambdaExpression.invokedClassName     = programClass.getName();
        lambdaExpression.invokedMethodName    = accessorMethodName;
        lambdaExpression.invokedMethodDesc    = accessorMethodDescriptor;
        lambdaExpression.invokedReferenceKind = ClassConstants.REF_invokeStatic;

        lambdaExpression.referencedInvokedClass  = programClass;
        lambdaExpression.referencedInvokedMethod = programClass.findMethod(accessorMethodName,
                                                                           accessorMethodDescriptor);
    }


    private void completeStatelessLambdaClass(ProgramClass     lambdaClass,
                                              LambdaExpression lambdaExpression)
    {
        String lambdaClassType = ClassUtil.internalTypeFromClassName(lambdaClass.getName());

        SimplifiedClassEditor classEditor = new SimplifiedClassEditor(lambdaClass);

        // Add singleton field
        classEditor.addField(ClassConstants.ACC_PUBLIC |
                             ClassConstants.ACC_STATIC |
                             ClassConstants.ACC_FINAL,
                             LAMBDA_SINGLETON_FIELD_NAME,
                             lambdaClassType);

        // Add the constructor.
        CompactCodeAttributeComposer composer =
            classEditor.addMethod(ClassConstants.ACC_PUBLIC,
                                  ClassConstants.METHOD_NAME_INIT,
                                  ClassConstants.METHOD_TYPE_INIT,
                                  10);

        composer.aload_0()
                .invokespecial(ClassConstants.NAME_JAVA_LANG_OBJECT,
                               ClassConstants.METHOD_NAME_INIT,
                               ClassConstants.METHOD_TYPE_INIT)
                .return_();

        // Add static initializer.
        composer =
            classEditor.addMethod(ClassConstants.ACC_STATIC,
                                  ClassConstants.METHOD_NAME_CLINIT,
                                  ClassConstants.METHOD_TYPE_CLINIT,
                                  30);

        composer.new_(lambdaClass.getName())
                .dup()
                .invokespecial(lambdaClass.getName(),
                               ClassConstants.METHOD_NAME_INIT,
                               ClassConstants.METHOD_TYPE_INIT)
                .putstatic(lambdaClass.getName(),
                           LAMBDA_SINGLETON_FIELD_NAME,
                           lambdaClassType)
                .return_();

        // If the lambda expression is serializable, create a readResolve method
        // to return the singleton field.
        if (lambdaExpression.isSerializable())
        {
            composer =
                classEditor.addMethod(ClassConstants.ACC_PRIVATE,
                                      ClassConstants.METHOD_NAME_READ_RESOLVE,
                                      ClassConstants.METHOD_TYPE_READ_RESOLVE,
                                      10);

            composer.getstatic(lambdaClass.getName(),
                               LAMBDA_SINGLETON_FIELD_NAME,
                               lambdaClassType)
                    .areturn();
        }

        // Add the interface method.
        composer =
            classEditor.addMethod(ClassConstants.ACC_PUBLIC,
                                  lambdaExpression.interfaceMethod,
                                  lambdaExpression.interfaceMethodDescriptor,
                                  50);

        if (lambdaExpression.invokedReferenceKind == ClassConstants.REF_newInvokeSpecial)
        {
            InternalTypeEnumeration typeEnumeration =
                new InternalTypeEnumeration(lambdaExpression.interfaceMethodDescriptor);

            InternalTypeEnumeration invokedTypeEnumeration =
                new InternalTypeEnumeration(lambdaExpression.invokedMethodDesc);

            composer.new_(lambdaExpression.invokedClassName)
                    .dup();

            // Convert the remaining parameters if they are present.
            completeInterfaceMethod(lambdaExpression,
                                    composer,
                                    1,
                                    typeEnumeration,
                                    invokedTypeEnumeration);
        }
        else
        {
            InternalTypeEnumeration typeEnumeration =
                new InternalTypeEnumeration(lambdaExpression.interfaceMethodDescriptor);

            InternalTypeEnumeration invokedTypeEnumeration =
                new InternalTypeEnumeration(lambdaExpression.invokedMethodDesc);

            boolean isInvokeVirtualOrInterface =
                lambdaExpression.invokedReferenceKind == ClassConstants.REF_invokeVirtual ||
                lambdaExpression.invokedReferenceKind == ClassConstants.REF_invokeInterface;

            int paramIndex = 1;

            // If we invoke a method on an object, we need to cast it to the invoked type.
            if (isInvokeVirtualOrInterface)
            {
                String type = typeEnumeration.nextType();
                String invokedType =
                    ClassUtil.internalTypeFromClassName(lambdaExpression.invokedClassName);

                composer.load(paramIndex, type);
                paramIndex += ClassUtil.internalTypeSize(type);

                convertToTargetType(type, invokedType, composer);
            }

            // Convert the remaining parameters if they are present.
            completeInterfaceMethod(lambdaExpression,
                                    composer,
                                    paramIndex,
                                    typeEnumeration,
                                    invokedTypeEnumeration);
        }

        classEditor.finishEditing();
    }


    private void completeCapturingLambdaClass(ProgramClass     lambdaClass,
                                              LambdaExpression lambdaExpression)
    {
        SimplifiedClassEditor classEditor = new SimplifiedClassEditor(lambdaClass);

        // Create constructor.
        String ctorDescriptor = lambdaExpression.getConstructorDescriptor();
        CompactCodeAttributeComposer composer =
            classEditor.addMethod(ClassConstants.ACC_PUBLIC,
                                  ClassConstants.METHOD_NAME_INIT,
                                  ctorDescriptor,
                                  50);

        composer.aload_0()
                .invokespecial(ClassConstants.NAME_JAVA_LANG_OBJECT,
                               ClassConstants.METHOD_NAME_INIT,
                               ClassConstants.METHOD_TYPE_INIT);

        InternalTypeEnumeration typeEnumeration =
            new InternalTypeEnumeration(ctorDescriptor);

        int argIndex    = 0;
        int variableIndex = 1;
        while (typeEnumeration.hasMoreTypes())
        {
            String type      = typeEnumeration.nextType();
            String fieldName = "arg$" + argIndex++;

            classEditor.addField(ClassConstants.ACC_PRIVATE | ClassConstants.ACC_FINAL,
                                 fieldName,
                                 type);

            composer.aload_0();
            composer.load(variableIndex, type);
            composer.putfield(lambdaClass.getName(), fieldName, type);

            variableIndex += ClassUtil.internalTypeSize(type);
        }

        composer.return_();

        // Create interface method.
        composer =
            classEditor.addMethod(ClassConstants.ACC_PUBLIC,
                                  lambdaExpression.interfaceMethod,
                                  lambdaExpression.interfaceMethodDescriptor,
                                  50);

        // Load the instance fields first.
        typeEnumeration =
            new InternalTypeEnumeration(ctorDescriptor);

        InternalTypeEnumeration invokedTypeEnumeration =
            new InternalTypeEnumeration(lambdaExpression.invokedMethodDesc);

        boolean isInvokeVirtualOrInterface =
            lambdaExpression.invokedReferenceKind == ClassConstants.REF_invokeVirtual ||
            lambdaExpression.invokedReferenceKind == ClassConstants.REF_invokeInterface;

        argIndex = 0;
        while (typeEnumeration.hasMoreTypes())
        {
            String type        = typeEnumeration.nextType();
            String invokedType = isInvokeVirtualOrInterface && argIndex == 0 ?
                null : invokedTypeEnumeration.nextType();

            String fieldName = "arg$" + argIndex++;

            composer.aload_0()
                    .getfield(lambdaClass.getName(), fieldName, type);

            if (invokedType != null)
            {
                convertToTargetType(type, invokedType, composer);
            }
        }

        // And then the method parameters.
        typeEnumeration =
            new InternalTypeEnumeration(lambdaExpression.interfaceMethodDescriptor);

        completeInterfaceMethod(lambdaExpression,
                                composer,
                                1,
                                typeEnumeration,
                                invokedTypeEnumeration);

        classEditor.finishEditing();
    }


    private void completeInterfaceMethod(LambdaExpression             lambdaExpression,
                                         CompactCodeAttributeComposer composer,
                                         int                          parameterIndex,
                                         InternalTypeEnumeration      typeEnumeration,
                                         InternalTypeEnumeration      invokedTypeEnumeration)
    {
        while (typeEnumeration.hasMoreTypes())
        {
            String type        = typeEnumeration.nextType();
            String invokedType =
                invokedTypeEnumeration != null ?
                    invokedTypeEnumeration.nextType() :
                    null;

            composer.load(parameterIndex, type);
            parameterIndex += ClassUtil.internalTypeSize(type);

            if (invokedType != null)
            {
                convertToTargetType(type, invokedType, composer);
            }
        }

        switch (lambdaExpression.invokedReferenceKind)
        {
            case ClassConstants.REF_invokeStatic:
                if (lambdaExpression.invokesStaticInterfaceMethod())
                {
                    composer.invokestaticinterface(lambdaExpression.invokedClassName,
                                                   lambdaExpression.invokedMethodName,
                                                   lambdaExpression.invokedMethodDesc,
                                                   lambdaExpression.referencedInvokedClass,
                                                   lambdaExpression.referencedInvokedMethod);
                }
                else
                {
                    composer.invokestatic(lambdaExpression.invokedClassName,
                                          lambdaExpression.invokedMethodName,
                                          lambdaExpression.invokedMethodDesc,
                                          lambdaExpression.referencedInvokedClass,
                                          lambdaExpression.referencedInvokedMethod);
                }
                break;

            case ClassConstants.REF_invokeVirtual:
                composer.invokevirtual(lambdaExpression.invokedClassName,
                                       lambdaExpression.invokedMethodName,
                                       lambdaExpression.invokedMethodDesc,
                                       lambdaExpression.referencedInvokedClass,
                                       lambdaExpression.referencedInvokedMethod);
                break;

            case ClassConstants.REF_invokeInterface:
                composer.invokeinterface(lambdaExpression.invokedClassName,
                                         lambdaExpression.invokedMethodName,
                                         lambdaExpression.invokedMethodDesc,
                                         lambdaExpression.referencedInvokedClass,
                                         lambdaExpression.referencedInvokedMethod);
                break;

            case ClassConstants.REF_newInvokeSpecial:
            case ClassConstants.REF_invokeSpecial:
                composer.invokespecial(lambdaExpression.invokedClassName,
                                       lambdaExpression.invokedMethodName,
                                       lambdaExpression.invokedMethodDesc,
                                       lambdaExpression.referencedInvokedClass,
                                       lambdaExpression.referencedInvokedMethod);
                break;
        }

        String methodReturnType = typeEnumeration.returnType();

        if (invokedTypeEnumeration != null)
        {
            convertToTargetType(invokedTypeEnumeration.returnType(),
                                methodReturnType,
                                composer);
        }

        composer.return_(methodReturnType);
    }


    private void addBridgeMethods(ProgramClass lambdaClass, LambdaExpression lambdaExpression)
    {
        SimplifiedClassEditor classEditor = new SimplifiedClassEditor(lambdaClass);

        String methodName = lambdaExpression.interfaceMethod;
        for (String bridgeMethodDescriptor : lambdaExpression.bridgeMethodDescriptors)
        {
            Method method = lambdaClass.findMethod(methodName, bridgeMethodDescriptor);
            if (method != null)
            {
                continue;
            }

            CompactCodeAttributeComposer composer =
                classEditor.addMethod(ClassConstants.ACC_PUBLIC    |
                                      ClassConstants.ACC_SYNTHETIC |
                                      ClassConstants.ACC_BRIDGE,
                                      methodName,
                                      bridgeMethodDescriptor,
                                      50);

            composer.aload_0();

            InternalTypeEnumeration interfaceTypeEnumeration =
                new InternalTypeEnumeration(lambdaExpression.interfaceMethodDescriptor);

            InternalTypeEnumeration bridgeTypeEnumeration =
                new InternalTypeEnumeration(bridgeMethodDescriptor);
            int variableIndex = 1;
            while (bridgeTypeEnumeration.hasMoreTypes())
            {
                String type = bridgeTypeEnumeration.nextType();
                String interfaceType = interfaceTypeEnumeration.nextType();

                composer.load(variableIndex, type);
                variableIndex += ClassUtil.internalTypeSize(type);

                convertToTargetType(type, interfaceType, composer);
            }

            composer.invokevirtual(lambdaClass.getName(),
                                   lambdaExpression.interfaceMethod,
                                   lambdaExpression.interfaceMethodDescriptor);

            String methodReturnType = bridgeTypeEnumeration.returnType();

            convertToTargetType(interfaceTypeEnumeration.returnType(),
                                methodReturnType,
                                composer);

            composer.return_(methodReturnType);
        }

        classEditor.finishEditing();
    }


    private static String prependParameterToMethodDescriptor(String methodDescriptor,
                                                             String type)
    {
        StringBuilder methodDescBuilder = new StringBuilder();

        methodDescBuilder.append('(');
        methodDescBuilder.append(type);

        InternalTypeEnumeration typeEnumeration =
            new InternalTypeEnumeration(methodDescriptor);

        while (typeEnumeration.hasMoreTypes())
        {
            methodDescBuilder.append(typeEnumeration.nextType());
        }

        methodDescBuilder.append(')');
        methodDescBuilder.append(typeEnumeration.returnType());
        return methodDescBuilder.toString();
    }


    /**
     * Adds the required instructions to the provided CodeAttributeComposer
     * to convert the current value on the stack to the given targetType.
     */
    private static void convertToTargetType(String                       sourceType,
                                            String                       targetType,
                                            CompactCodeAttributeComposer composer)
    {
        if (ClassUtil.isInternalPrimitiveType(sourceType) &&
            !ClassUtil.isInternalPrimitiveType(targetType))
        {
            // Perform auto-boxing.
            switch (sourceType.charAt(0))
            {
                case ClassConstants.TYPE_INT:
                    composer.invokestatic(ClassConstants.NAME_JAVA_LANG_INTEGER,
                                          "valueOf",
                                          "(I)Ljava/lang/Integer;");
                    break;

                case ClassConstants.TYPE_BYTE:
                    composer.invokestatic(ClassConstants.NAME_JAVA_LANG_BYTE,
                                          "valueOf",
                                          "(B)Ljava/lang/Byte;");
                    break;

                case ClassConstants.TYPE_CHAR:
                    composer.invokestatic(ClassConstants.NAME_JAVA_LANG_CHARACTER,
                                          "valueOf",
                                          "(C)Ljava/lang/Character;");
                    break;

                case ClassConstants.TYPE_SHORT:
                    composer.invokestatic(ClassConstants.NAME_JAVA_LANG_SHORT,
                                          "valueOf",
                                          "(S)Ljava/lang/Short;");
                    break;

                case ClassConstants.TYPE_BOOLEAN:
                    composer.invokestatic(ClassConstants.NAME_JAVA_LANG_BOOLEAN,
                                          "valueOf",
                                          "(Z)Ljava/lang/Boolean;");
                    break;

                case ClassConstants.TYPE_LONG:
                    composer.invokestatic(ClassConstants.NAME_JAVA_LANG_LONG,
                                          "valueOf",
                                          "(J)Ljava/lang/Long;");
                    break;

                case ClassConstants.TYPE_FLOAT:
                    composer.invokestatic(ClassConstants.NAME_JAVA_LANG_FLOAT,
                                          "valueOf",
                                          "(F)Ljava/lang/Float;");
                    break;

                case ClassConstants.TYPE_DOUBLE:
                    composer.invokestatic(ClassConstants.NAME_JAVA_LANG_DOUBLE,
                                          "valueOf",
                                          "(D)Ljava/lang/Double;");
                    break;
            }
        }
        else if (!ClassUtil.isInternalPrimitiveType(sourceType) &&
                 ClassUtil.isInternalPrimitiveType(targetType))
        {
            boolean castRequired = sourceType.equals(ClassConstants.TYPE_JAVA_LANG_OBJECT);

            // Perform auto-unboxing.
            switch (targetType.charAt(0))
            {
                case ClassConstants.TYPE_INT:
                    if (castRequired)
                    {
                        composer.checkcast("java/lang/Number");
                    }
                    composer.invokevirtual("java/lang/Number",
                                           "intValue",
                                           "()I");
                    break;

                case ClassConstants.TYPE_BYTE:
                    if (castRequired)
                    {
                        composer.checkcast(ClassConstants.NAME_JAVA_LANG_BYTE);
                    }
                    composer.invokevirtual(ClassConstants.NAME_JAVA_LANG_BYTE,
                                           "byteValue",
                                           "()B");
                    break;

                case ClassConstants.TYPE_CHAR:
                    if (castRequired)
                    {
                        composer.checkcast(ClassConstants.NAME_JAVA_LANG_CHARACTER);
                    }
                    composer.invokevirtual(ClassConstants.NAME_JAVA_LANG_CHARACTER,
                                           "charValue",
                                           "()C");
                    break;

                case ClassConstants.TYPE_SHORT:
                    if (castRequired)
                    {
                        composer.checkcast(ClassConstants.NAME_JAVA_LANG_SHORT);
                    }
                    composer.invokevirtual(ClassConstants.NAME_JAVA_LANG_SHORT,
                                           "shortValue",
                                           "()S");
                    break;

                case ClassConstants.TYPE_BOOLEAN:
                    if (castRequired)
                    {
                        composer.checkcast(ClassConstants.NAME_JAVA_LANG_BOOLEAN);
                    }
                    composer.invokevirtual(ClassConstants.NAME_JAVA_LANG_BOOLEAN,
                                           "booleanValue",
                                           "()Z");
                    break;

                case ClassConstants.TYPE_LONG:
                    if (castRequired)
                    {
                        composer.checkcast("java/lang/Number");
                    }
                    composer.invokevirtual("java/lang/Number",
                                           "longValue",
                                           "()J");
                    break;

                case ClassConstants.TYPE_FLOAT:
                    if (castRequired)
                    {
                        composer.checkcast("java/lang/Number");
                    }
                    composer.invokevirtual("java/lang/Number",
                                           "floatValue",
                                           "()F");
                    break;

                case ClassConstants.TYPE_DOUBLE:
                    if (castRequired)
                    {
                        composer.checkcast("java/lang/Number");
                    }
                    composer.invokevirtual("java/lang/Number",
                                           "doubleValue",
                                           "()D");
                    break;
            }
        }
        else if (ClassUtil.isInternalClassType(sourceType)   &&
                 (ClassUtil.isInternalClassType(targetType) ||
                  ClassUtil.isInternalArrayType(targetType)) &&
                 !sourceType.equals(targetType)              &&
                 // No need to cast to java/lang/Object.
                 !ClassConstants.TYPE_JAVA_LANG_OBJECT.equals(targetType))
        {
            // Cast to target type.
            composer.checkcast(ClassUtil.internalClassTypeFromType(targetType));
        }
    }
}
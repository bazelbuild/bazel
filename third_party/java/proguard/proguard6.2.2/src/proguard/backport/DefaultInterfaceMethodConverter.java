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
import proguard.classfile.editor.*;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.util.StringTransformer;

import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;


/**
 * This ClassVisitor moves all default interface methods in the visited
 * interfaces to concrete implementations.
 *
 * @author Thomas Neidhart
 */
public class DefaultInterfaceMethodConverter
extends    SimplifiedVisitor
implements ClassVisitor,

           // Implementation interfaces.
           AttributeVisitor
{
    private final ClassVisitor  modifiedClassVisitor;
    private final MemberVisitor extraMemberVisitor;

    // Fields acting as parameters and return values for the visitor methods.

    private final Set<Clazz> implClasses       = new LinkedHashSet<Clazz>();
    private boolean          hasDefaultMethods;


    public DefaultInterfaceMethodConverter(ClassVisitor  modifiedClassVisitor,
                                           MemberVisitor extraMemberVisitor)
    {
        this.modifiedClassVisitor = modifiedClassVisitor;
        this.extraMemberVisitor   = extraMemberVisitor;
    }


    // Implementations for ClassVisitor.

    @Override
    public void visitLibraryClass(LibraryClass libraryClass) {}


    @Override
    public void visitProgramClass(ProgramClass programClass)
    {
        hasDefaultMethods = false;
        implClasses.clear();

        // Collect all implementations of the interface.
        programClass.hierarchyAccept(false, false, false, true,
            new ProgramClassFilter(
            // Ignore other interfaces that extend this one.
            new ClassAccessFilter(0, ClassConstants.ACC_INTERFACE,
            new ClassCollector(implClasses))));

        programClass.accept(
            new AllMethodVisitor(
            new MemberAccessFilter(0, ClassConstants.ACC_STATIC,
            new AllAttributeVisitor(this))));

        if (hasDefaultMethods)
        {
            // Shrink the constant pool of unused constants.
            programClass.accept(new ConstantPoolShrinker());
        }
    }


    // Implementations for AttributeVisitor.

    @Override
    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    @Override
    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        hasDefaultMethods = true;

        ProgramClass interfaceClass = (ProgramClass) clazz;
        ProgramMethod defaultMethod = (ProgramMethod) method;

        for (Clazz implClass : implClasses)
        {
            ProgramClass targetClass = (ProgramClass) implClass;

            // Add the default method to the implementing class
            // if necessary.
            if (!hasInheritedMethod(targetClass,
                                    defaultMethod.getName(interfaceClass),
                                    defaultMethod.getDescriptor(interfaceClass)))
            {
                defaultMethod.accept(interfaceClass,
                    new MemberAdder(targetClass));

                targetClass.accept(modifiedClassVisitor);
            }

            // Add the default method as a different method and adapt
            // super invocations to it, if necessary.
            if (callsDefaultMethodUsingSuper(targetClass,
                                             interfaceClass,
                                             defaultMethod))
            {
                replaceDefaultMethodInvocation(targetClass,
                                               interfaceClass,
                                               defaultMethod);

                targetClass.accept(modifiedClassVisitor);
            }
        }

        // Remove the code attribute from the method and
        // add make it abstract.
        defaultMethod.accept(interfaceClass,
            new MultiMemberVisitor(
                new NamedAttributeDeleter(ClassConstants.ATTR_Code),

                new MemberAccessFlagSetter(ClassConstants.ACC_ABSTRACT)
            ));

        // Call extra visitor for each visited default method.
        if (extraMemberVisitor != null)
        {
            defaultMethod.accept(interfaceClass, extraMemberVisitor);
        }
    }


    // Small utility methods.

    private boolean hasInheritedMethod(Clazz  clazz,
                                       String methodName,
                                       String methodDescriptor)
    {
        MemberCounter counter = new MemberCounter();

        clazz.hierarchyAccept(true, true, false, false,
            new NamedMethodVisitor(methodName, methodDescriptor,
            counter));

        return counter.getCount() > 0;
    }


    /**
     * Returns true if any method of the given class
     * calls Interface.super.defaultMethod(...).
     */
    private boolean callsDefaultMethodUsingSuper(Clazz  clazz,
                                                 Clazz  interfaceClass,
                                                 Method defaultMethod)
    {
        final AtomicBoolean foundInvocation = new AtomicBoolean(false);

        clazz.accept(
            new AllMethodVisitor(
            new AllAttributeVisitor(
            new AllInstructionVisitor(
            new SuperInvocationInstructionMatcher(interfaceClass,
                                                  defaultMethod)
            {
                @Override
                public void superInvocation(Clazz               clazz,
                                            Method              method,
                                            CodeAttribute       codeAttribute,
                                            int                 offset,
                                            CodeAttributeEditor codeAttributeEditor)
                {
                    foundInvocation.set(true);
                }
            }))));

        return foundInvocation.get();
    }


    /**
     * Replaces any super calls to the given default interface method
     * in the target class. The default method is copied to the target
     * class and the invoke is updated accordingly.
     */
    private void replaceDefaultMethodInvocation(ProgramClass  targetClass,
                                                ProgramClass  interfaceClass,
                                                ProgramMethod interfaceMethod)
    {
        // Copy the interface method to the target class, with an updated name.
        StringTransformer memberRenamer = new StringTransformer()
        {
            public String transform(String string)
            {
                return "default$" + string;
            }
        };

        interfaceMethod.accept(interfaceClass,
            new MemberAdder(targetClass, memberRenamer, null));

        String targetMethodName =
            memberRenamer.transform(interfaceMethod.getName(interfaceClass));

        // Update invocations of the method inside the target class.
        String descriptor   = interfaceMethod.getDescriptor(interfaceClass);
        Method targetMethod = targetClass.findMethod(targetMethodName, descriptor);

        ConstantPoolEditor constantPoolEditor = new ConstantPoolEditor(targetClass);
        final int          constantIndex      = constantPoolEditor.addMethodrefConstant(targetClass, targetMethod);

        targetClass.accept(
            new AllMethodVisitor(
            new AllAttributeVisitor(
            new SuperInvocationInstructionMatcher(interfaceClass,
                                                  interfaceMethod)
            {
                @Override
                public void superInvocation(Clazz               clazz,
                                            Method              method,
                                            CodeAttribute       codeAttribute,
                                            int                 offset,
                                            CodeAttributeEditor codeAttributeEditor)
                {
                    Instruction instruction =
                        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL,
                                                constantIndex);

                    codeAttributeEditor.replaceInstruction(offset, instruction);
                }
            })));
    }


    /**
     * This InstructionVisitor will call the {@code superInvocation(...)} method
     * for any encountered INVOKESPECIAL instruction whose associated
     * constant is an InterfaceMethodRefConstant and matches the given
     * referenced class and method.
     */
    private static class SuperInvocationInstructionMatcher
    extends              SimplifiedVisitor
    implements           AttributeVisitor,
                         InstructionVisitor,
                         ConstantVisitor
    {
        private final Clazz               referencedClass;
        private final Method              referencedMethod;
        private final CodeAttributeEditor codeAttributeEditor = new CodeAttributeEditor();

        private boolean matchingInvocation;


        public SuperInvocationInstructionMatcher(Clazz  referencedClass,
                                                 Method referencedMethod)
        {
            this.referencedClass  = referencedClass;
            this.referencedMethod = referencedMethod;
        }


        // Implementations for AttributeVisitor.

        public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


        public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
        {
            // Set up the code attribute editor.
            codeAttributeEditor.reset(codeAttribute.u4codeLength);

            // Find the peephole optimizations.
            codeAttribute.instructionsAccept(clazz, method, this);

            // Apply the peephole optimizations.
            codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
        }


        // Implementations for InstructionVisitor.

        @Override
        public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


        @Override
        public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
        {
            switch (constantInstruction.opcode)
            {
                case InstructionConstants.OP_INVOKESPECIAL:
                    matchingInvocation = false;
                    clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);

                    if (matchingInvocation)
                    {
                        superInvocation(clazz, method, codeAttribute, offset, codeAttributeEditor);
                    }
                    break;
            }
        }


        // Implementations for ConstantVisitor.

        @Override
        public void visitAnyConstant(Clazz clazz, Constant constant) {}


        @Override
        public void visitInterfaceMethodrefConstant(Clazz clazz, InterfaceMethodrefConstant interfaceMethodrefConstant)
        {
            if (interfaceMethodrefConstant.referencedClass  == referencedClass &&
                interfaceMethodrefConstant.referencedMember == referencedMethod)
            {
                matchingInvocation = true;
            }
        }


        /**
         * The callback method which will be called for each detected super invocation
         * of the specified interface method.
         */
        public void superInvocation(Clazz               clazz,
                                    Method              method,
                                    CodeAttribute       codeAttribute,
                                    int                 offset,
                                    CodeAttributeEditor codeAttributeEditor) {}
    }
}



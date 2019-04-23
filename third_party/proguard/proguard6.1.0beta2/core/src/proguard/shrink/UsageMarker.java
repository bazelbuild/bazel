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
package proguard.shrink;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.annotation.*;
import proguard.classfile.attribute.module.*;
import proguard.classfile.attribute.module.visitor.*;
import proguard.classfile.attribute.preverification.*;
import proguard.classfile.attribute.preverification.visitor.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.*;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;

/**
 * This ClassVisitor and MemberVisitor recursively marks all classes and class
 * elements that are being used.
 *
 * @see ClassShrinker
 *
 * @author Eric Lafortune
 */
class      UsageMarker
extends    SimplifiedVisitor
implements ClassVisitor,
           MemberVisitor,
           ConstantVisitor,
           AttributeVisitor,
           InnerClassesInfoVisitor,
           ExceptionInfoVisitor,
           StackMapFrameVisitor,
           VerificationTypeVisitor,
           ParameterInfoVisitor,
           LocalVariableInfoVisitor,
           LocalVariableTypeInfoVisitor,
           RequiresInfoVisitor,
           ExportsInfoVisitor,
           OpensInfoVisitor,
           ProvidesInfoVisitor,
//         AnnotationVisitor,
//         ElementValueVisitor,
           InstructionVisitor
{
    // A visitor info flag to indicate the ProgramMember object is being used,
    // if its Clazz can be determined as being used as well.
    private static final Object POSSIBLY_USED = new Object();
    // A visitor info flag to indicate the visitor accepter is being used.
    private static final Object USED          = new Object();


    private final MyInterfaceUsageMarker          interfaceUsageMarker           = new MyInterfaceUsageMarker();
    private final MyDefaultMethodUsageMarker      defaultMethodUsageMarker       = new MyDefaultMethodUsageMarker();
    private final MyPossiblyUsedMemberUsageMarker possiblyUsedMemberUsageMarker  = new MyPossiblyUsedMemberUsageMarker();
    private final MemberVisitor                   nonEmptyMethodUsageMarker      = new AllAttributeVisitor(
                                                                                   new MyNonEmptyMethodUsageMarker());
    private final ConstantVisitor                 parameterlessConstructorMarker = new ConstantTagFilter(new int[] { ClassConstants.CONSTANT_String, ClassConstants.CONSTANT_Class },
                                                                                   new ReferencedClassVisitor(
                                                                                   new NamedMethodVisitor(ClassConstants.METHOD_NAME_INIT,
                                                                                                          ClassConstants.METHOD_TYPE_INIT,
                                                                                                          this)));

    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        if (shouldBeMarkedAsUsed(programClass))
        {
            // Mark this class.
            markAsUsed(programClass);

            markProgramClassBody(programClass);
        }
    }


    protected void markProgramClassBody(ProgramClass programClass)
    {
        // Mark this class's name.
        markConstant(programClass, programClass.u2thisClass);

        // Mark the superclass.
        markOptionalConstant(programClass, programClass.u2superClass);

        // Give the interfaces preliminary marks.
        programClass.hierarchyAccept(false, false, true, false,
                                     interfaceUsageMarker);

        // Explicitly mark the <clinit> method, if it's not empty.
        programClass.methodAccept(ClassConstants.METHOD_NAME_CLINIT,
                                  ClassConstants.METHOD_TYPE_CLINIT,
                                  nonEmptyMethodUsageMarker);

        // Process all class members that have already been marked as possibly used.
        programClass.fieldsAccept(possiblyUsedMemberUsageMarker);
        programClass.methodsAccept(possiblyUsedMemberUsageMarker);

        // Mark the attributes.
        programClass.attributesAccept(this);
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        if (shouldBeMarkedAsUsed(libraryClass))
        {
            markAsUsed(libraryClass);

            // We're not going to analyze all library code. We're assuming that
            // if this class is being used, all of its methods will be used as
            // well. We'll mark them as such (here and in all subclasses).

            // Mark the superclass.
            Clazz superClass = libraryClass.superClass;
            if (superClass != null)
            {
                superClass.accept(this);
            }

            // Mark the interfaces.
            Clazz[] interfaceClasses = libraryClass.interfaceClasses;
            if (interfaceClasses != null)
            {
                for (int index = 0; index < interfaceClasses.length; index++)
                {
                    if (interfaceClasses[index] != null)
                    {
                        interfaceClasses[index].accept(this);
                    }
                }
            }

            // Mark all methods.
            libraryClass.methodsAccept(this);
        }
    }


    /**
     * This ClassVisitor marks ProgramClass objects as possibly used,
     * and it visits LibraryClass objects with its outer UsageMarker.
     */
    private class MyInterfaceUsageMarker
    implements    ClassVisitor
    {
        public void visitProgramClass(ProgramClass programClass)
        {
            if (shouldBeMarkedAsPossiblyUsed(programClass))
            {
                // We can't process the interface yet, because it might not
                // be required. Give it a preliminary mark.
                markAsPossiblyUsed(programClass);
            }
        }

        public void visitLibraryClass(LibraryClass libraryClass)
        {
            // Make sure all library interface methods are marked.
            UsageMarker.this.visitLibraryClass(libraryClass);
        }
    }


    /**
     * This MemberVisitor marks ProgramMethod objects of default
     * implementations that may be present in interface classes.
     */
    private class MyDefaultMethodUsageMarker
    extends       SimplifiedVisitor
    implements    MemberVisitor
    {
        // Implementations for MemberVisitor.

        public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
        {
            if (shouldBeMarkedAsUsed(programMethod))
            {
                markAsUsed(programMethod);

                // Mark the method body.
                markProgramMethodBody(programClass, programMethod);

                // Note that, if the method has been marked as possibly used,
                // the method hierarchy has already been marked (cfr. below).
            }
        }
    }


    /**
     * This MemberVisitor marks ProgramField and ProgramMethod objects that
     * have already been marked as possibly used.
     */
    private class MyPossiblyUsedMemberUsageMarker
    extends       SimplifiedVisitor
    implements    MemberVisitor
    {
        // Implementations for MemberVisitor.

        public void visitProgramField(ProgramClass programClass, ProgramField programField)
        {
            // Has the method already been referenced?
            if (isPossiblyUsed(programField))
            {
                markAsUsed(programField);

                // Mark the name and descriptor.
                markConstant(programClass, programField.u2nameIndex);
                markConstant(programClass, programField.u2descriptorIndex);

                // Mark the attributes.
                programField.attributesAccept(programClass, UsageMarker.this);

                // Mark the classes referenced in the descriptor string.
                programField.referencedClassesAccept(UsageMarker.this);
            }
        }


        public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
        {
            // Has the method already been referenced?
            if (isPossiblyUsed(programMethod))
            {
                markAsUsed(programMethod);

                // Mark the method body.
                markProgramMethodBody(programClass, programMethod);

                // Note that, if the method has been marked as possibly used,
                // the method hierarchy has already been marked (cfr. below).
            }
        }
    }


    /**
     * This AttributeVisitor marks ProgramMethod objects of non-empty methods.
     */
    private class MyNonEmptyMethodUsageMarker
    extends       SimplifiedVisitor
    implements    AttributeVisitor
    {
        // Implementations for AttributeVisitor.

        public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


        public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
        {
            if (codeAttribute.u4codeLength > 1)
            {
                method.accept(clazz, UsageMarker.this);
            }
        }
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        if (shouldBeMarkedAsUsed(programField))
        {
            // Is the field's class used?
            if (isUsed(programClass))
            {
                markAsUsed(programField);

                // Mark the field body.
                markProgramFieldBody(programClass, programField);
            }

            // Hasn't the field been marked as possibly being used yet?
            else if (shouldBeMarkedAsPossiblyUsed(programField))
            {
                // We can't process the field yet, because the class isn't
                // marked as being used (yet). Give it a preliminary mark.
                markAsPossiblyUsed(programField);
            }
        }
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        if (shouldBeMarkedAsUsed(programMethod))
        {
            // Is the method's class used?
            if (isUsed(programClass))
            {
                markAsUsed(programMethod);

                // Mark the method body.
                markProgramMethodBody(programClass, programMethod);

                // Mark the method hierarchy.
                markMethodHierarchy(programClass, programMethod);
            }

            // Hasn't the method been marked as possibly being used yet?
            else if (shouldBeMarkedAsPossiblyUsed(programMethod))
            {
                // We can't process the method yet, because the class isn't
                // marked as being used (yet). Give it a preliminary mark.
                markAsPossiblyUsed(programMethod);

                // Mark the method hierarchy.
                markMethodHierarchy(programClass, programMethod);
            }
        }
    }


    public void visitLibraryField(LibraryClass programClass, LibraryField programField) {}


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        if (shouldBeMarkedAsUsed(libraryMethod))
        {
            markAsUsed(libraryMethod);

            // Mark the method hierarchy.
            markMethodHierarchy(libraryClass, libraryMethod);
        }
    }


    protected void markProgramFieldBody(ProgramClass programClass, ProgramField programField)
    {
        // Mark the name and descriptor.
        markConstant(programClass, programField.u2nameIndex);
        markConstant(programClass, programField.u2descriptorIndex);

        // Mark the attributes.
        programField.attributesAccept(programClass, this);

        // Mark the classes referenced in the descriptor string.
        programField.referencedClassesAccept(this);
    }


    protected void markProgramMethodBody(ProgramClass programClass, ProgramMethod programMethod)
    {
        // Mark the name and descriptor.
        markConstant(programClass, programMethod.u2nameIndex);
        markConstant(programClass, programMethod.u2descriptorIndex);

        // Mark the attributes.
        programMethod.attributesAccept(programClass, this);

        // Mark the classes referenced in the descriptor string.
        programMethod.referencedClassesAccept(this);
    }


    /**
     * Marks the hierarchy of implementing or overriding methods corresponding
     * to the given method, if any.
     */
    protected void markMethodHierarchy(Clazz clazz, Method method)
    {
        // Only visit the hierarchy if the method is not private, static, or
        // an initializer.
        int accessFlags = method.getAccessFlags();
        if ((accessFlags &
             (ClassConstants.ACC_PRIVATE |
              ClassConstants.ACC_STATIC)) == 0 &&
            !ClassUtil.isInitializer(method.getName(clazz)))
        {
            // We can skip private and static methods in the hierarchy, and
            // also abstract methods, unless they might widen a current
            // non-public access.
            int requiredUnsetAccessFlags =
                ClassConstants.ACC_PRIVATE |
                ClassConstants.ACC_STATIC  |
                ((accessFlags & ClassConstants.ACC_PUBLIC) == 0 ? 0 :
                     ClassConstants.ACC_ABSTRACT);

            // Mark default implementations in interfaces down the hierarchy,
            // if this is an interface itself.
            // TODO: This may be premature if there aren't any concrete implementing classes.
            clazz.accept(new ClassAccessFilter(ClassConstants.ACC_INTERFACE, 0,
                         new ClassHierarchyTraveler(false, false, false, true,
                         new ProgramClassFilter(
                         new ClassAccessFilter(ClassConstants.ACC_INTERFACE, 0,
                         new NamedMethodVisitor(method.getName(clazz),
                                                method.getDescriptor(clazz),
                         new MemberAccessFilter(0, requiredUnsetAccessFlags,
                         defaultMethodUsageMarker)))))));

            // Mark other implementations.
            clazz.accept(new ConcreteClassDownTraveler(
                         new ClassHierarchyTraveler(true, true, false, true,
                         new NamedMethodVisitor(method.getName(clazz),
                                                method.getDescriptor(clazz),
                         new MemberAccessFilter(0, requiredUnsetAccessFlags,
                         this)))));
        }
    }


    // Implementations for ConstantVisitor.

    public void visitIntegerConstant(Clazz clazz, IntegerConstant integerConstant)
    {
        if (shouldBeMarkedAsUsed(integerConstant))
        {
            markAsUsed(integerConstant);
        }
    }


    public void visitLongConstant(Clazz clazz, LongConstant longConstant)
    {
        if (shouldBeMarkedAsUsed(longConstant))
        {
            markAsUsed(longConstant);
        }
    }


    public void visitFloatConstant(Clazz clazz, FloatConstant floatConstant)
    {
        if (shouldBeMarkedAsUsed(floatConstant))
        {
            markAsUsed(floatConstant);
        }
    }


    public void visitDoubleConstant(Clazz clazz, DoubleConstant doubleConstant)
    {
        if (shouldBeMarkedAsUsed(doubleConstant))
        {
            markAsUsed(doubleConstant);
        }
    }


    public void visitPrimitiveArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant)
    {
        if (shouldBeMarkedAsUsed(primitiveArrayConstant))
        {
            markAsUsed(primitiveArrayConstant);
        }
    }


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        if (shouldBeMarkedAsUsed(stringConstant))
        {
            markAsUsed(stringConstant);

            markConstant(clazz, stringConstant.u2stringIndex);

            // Mark the referenced class and class member, if any.
            stringConstant.referencedClassAccept(this);
            stringConstant.referencedMemberAccept(this);
        }
    }


    public void visitUtf8Constant(Clazz clazz, Utf8Constant utf8Constant)
    {
        if (shouldBeMarkedAsUsed(utf8Constant))
        {
            markAsUsed(utf8Constant);
        }
    }


    public void visitDynamicConstant(Clazz clazz, DynamicConstant dynamicConstant)
    {
        if (shouldBeMarkedAsUsed(dynamicConstant))
        {
            markAsUsed(dynamicConstant);

            markConstant(clazz, dynamicConstant.u2nameAndTypeIndex);

            // Mark the referenced descriptor classes.
            dynamicConstant.referencedClassesAccept(this);

            // Mark the bootstrap methods attribute.
            clazz.attributesAccept(new MyBootStrapMethodUsageMarker(dynamicConstant.u2bootstrapMethodAttributeIndex));
        }
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        if (shouldBeMarkedAsUsed(invokeDynamicConstant))
        {
            markAsUsed(invokeDynamicConstant);

            markConstant(clazz, invokeDynamicConstant.u2nameAndTypeIndex);

            // Mark the referenced descriptor classes.
            invokeDynamicConstant.referencedClassesAccept(this);

            // Mark the bootstrap methods attribute.
            clazz.attributesAccept(new MyBootStrapMethodUsageMarker(invokeDynamicConstant.u2bootstrapMethodAttributeIndex));
        }
    }


    public void visitMethodHandleConstant(Clazz clazz, MethodHandleConstant methodHandleConstant)
    {
        if (shouldBeMarkedAsUsed(methodHandleConstant))
        {
            markAsUsed(methodHandleConstant);

            markConstant(clazz, methodHandleConstant.u2referenceIndex);
        }
    }


    public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
    {
        if (shouldBeMarkedAsUsed(refConstant))
        {
            markAsUsed(refConstant);

            markConstant(clazz, refConstant.u2classIndex);
            markConstant(clazz, refConstant.u2nameAndTypeIndex);

            // When compiled with "-target 1.2" or higher, the class or
            // interface actually containing the referenced class member may
            // be higher up the hierarchy. Make sure it's marked, in case it
            // isn't used elsewhere.
            refConstant.referencedClassAccept(this);

            // Mark the referenced class member itself.
            refConstant.referencedMemberAccept(this);
        }
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        if (shouldBeMarkedAsUsed(classConstant))
        {
            markAsUsed(classConstant);

            markConstant(clazz, classConstant.u2nameIndex);

            // Mark the referenced class itself.
            classConstant.referencedClassAccept(this);
        }
    }


    public void visitMethodTypeConstant(Clazz clazz, MethodTypeConstant methodTypeConstant)
    {
        if (shouldBeMarkedAsUsed(methodTypeConstant))
        {
            markAsUsed(methodTypeConstant);

            markConstant(clazz, methodTypeConstant.u2descriptorIndex);

            // Mark the referenced descriptor classes.
            methodTypeConstant.referencedClassesAccept(this);
        }
    }


    public void visitNameAndTypeConstant(Clazz clazz, NameAndTypeConstant nameAndTypeConstant)
    {
        if (shouldBeMarkedAsUsed(nameAndTypeConstant))
        {
            markAsUsed(nameAndTypeConstant);

            markConstant(clazz, nameAndTypeConstant.u2nameIndex);
            markConstant(clazz, nameAndTypeConstant.u2descriptorIndex);
        }
    }


    public void visitModuleConstant(Clazz clazz, ModuleConstant moduleConstant)
    {
        if (shouldBeMarkedAsUsed(moduleConstant))
        {
            markAsUsed(moduleConstant);

            markConstant(clazz, moduleConstant.u2nameIndex);
        }
    }


    public void visitPackageConstant(Clazz clazz, PackageConstant packageConstant)
    {
        if (shouldBeMarkedAsUsed(packageConstant))
        {
            markAsUsed(packageConstant);

            markConstant(clazz, packageConstant.u2nameIndex);
        }
    }


    /**
     * This AttributeVisitor marks the bootstrap methods attributes, their
     * method entries, their method handles, and their arguments.
     */
    private class MyBootStrapMethodUsageMarker
    extends       SimplifiedVisitor
    implements    AttributeVisitor,
                  BootstrapMethodInfoVisitor
    {
        private int bootstrapMethodIndex;


        private MyBootStrapMethodUsageMarker(int bootstrapMethodIndex)
        {
            this.bootstrapMethodIndex = bootstrapMethodIndex;
        }


        // Implementations for AttributeVisitor.

        public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


        public void visitBootstrapMethodsAttribute(Clazz clazz, BootstrapMethodsAttribute bootstrapMethodsAttribute)
        {
            if (shouldBeMarkedAsUsed(bootstrapMethodsAttribute))
            {
                markAsUsed(bootstrapMethodsAttribute);

                markConstant(clazz, bootstrapMethodsAttribute.u2attributeNameIndex);
            }

            bootstrapMethodsAttribute.bootstrapMethodEntryAccept(clazz,
                                                                 bootstrapMethodIndex,
                                                                 this);
        }


        // Implementations for BootstrapMethodInfoVisitor.

        public void visitBootstrapMethodInfo(Clazz clazz, BootstrapMethodInfo bootstrapMethodInfo)
        {
            markAsUsed(bootstrapMethodInfo);

            markConstant(clazz, bootstrapMethodInfo.u2methodHandleIndex);

            // Mark the constant pool entries referenced by the arguments.
            bootstrapMethodInfo.methodArgumentsAccept(clazz, UsageMarker.this);
        }
    }


    // Implementations for AttributeVisitor.
    // Note that attributes are typically only referenced once, so we don't
    // test if they have been marked already.

    public void visitUnknownAttribute(Clazz clazz, UnknownAttribute unknownAttribute)
    {
        // This is the best we can do for unknown attributes.
        markAsUsed(unknownAttribute);

        markConstant(clazz, unknownAttribute.u2attributeNameIndex);
    }


    public void visitBootstrapMethodsAttribute(Clazz clazz, BootstrapMethodsAttribute bootstrapMethodsAttribute)
    {
        // Don't mark the attribute and its name here. We may mark it in
        // MyBootStrapMethodsAttributeUsageMarker.
    }


    public void visitSourceFileAttribute(Clazz clazz, SourceFileAttribute sourceFileAttribute)
    {
        markAsUsed(sourceFileAttribute);

        markConstant(clazz, sourceFileAttribute.u2attributeNameIndex);
        markConstant(clazz, sourceFileAttribute.u2sourceFileIndex);
    }


    public void visitSourceDirAttribute(Clazz clazz, SourceDirAttribute sourceDirAttribute)
    {
        markAsUsed(sourceDirAttribute);

        markConstant(clazz, sourceDirAttribute.u2attributeNameIndex);
        markConstant(clazz, sourceDirAttribute.u2sourceDirIndex);
    }


    public void visitInnerClassesAttribute(Clazz clazz, InnerClassesAttribute innerClassesAttribute)
    {
        // Don't mark the attribute and its name yet. We may mark it later, in
        // InnerUsageMarker.
        //markAsUsed(innerClassesAttribute);

        //markConstant(clazz, innerClassesAttribute.u2attrNameIndex);

        // Do mark the outer class entries.
        innerClassesAttribute.innerClassEntriesAccept(clazz, this);
    }


    public void visitEnclosingMethodAttribute(Clazz clazz, EnclosingMethodAttribute enclosingMethodAttribute)
    {
        markAsUsed(enclosingMethodAttribute);

        markConstant(        clazz, enclosingMethodAttribute.u2attributeNameIndex);
        markConstant(        clazz, enclosingMethodAttribute.u2classIndex);
        markOptionalConstant(clazz, enclosingMethodAttribute.u2nameAndTypeIndex);
    }


    public void visitNestHostAttribute(Clazz clazz, NestHostAttribute nestHostAttribute)
    {
        // Don't mark the attribute and its contents yet. We may mark it later,
        // in NestUsageMarker.
        //markAsUsed(nestHostAttribute);

        //markConstant(clazz, nestHostAttribute.u2attributeNameIndex);
        //markConstant(clazz, nestHostAttribute.u2hostClassIndex);
    }


    public void visitNestMembersAttribute(Clazz clazz, NestMembersAttribute nestMembersAttribute)
    {
        // Don't mark the attribute and its contents yet. We may mark it later,
        // in NestUsageMarker.
        //markAsUsed(nestMembersAttribute);

        //markConstant(clazz, nestMembersAttribute.u2attributeNameIndex);

        // Mark the nest member entries.
        //nestMembersAttribute.memberClassConstantsAccept(clazz, this);
    }


    public void visitModuleAttribute(Clazz clazz, ModuleAttribute moduleAttribute)
    {
        markAsUsed(moduleAttribute);

        markConstant(        clazz, moduleAttribute.u2attributeNameIndex);
        markConstant(        clazz, moduleAttribute.u2moduleNameIndex);
        markOptionalConstant(clazz, moduleAttribute.u2moduleVersionIndex);

        // Mark the constant pool entries referenced by the contained info.
        moduleAttribute.requiresAccept(clazz, this);
        moduleAttribute.exportsAccept(clazz, this);
        moduleAttribute.opensAccept(clazz, this);

        markConstants(clazz, moduleAttribute.u2uses, moduleAttribute.u2usesCount);

        // Mark the constant pool entries referenced by the provides info.
        moduleAttribute.providesAccept(clazz, this);
    }


    public void visitModuleMainClassAttribute(Clazz clazz, ModuleMainClassAttribute moduleMainClassAttribute)
    {
        markAsUsed(moduleMainClassAttribute);

        markConstant(clazz, moduleMainClassAttribute.u2attributeNameIndex);
        markConstant(clazz, moduleMainClassAttribute.u2mainClass);
    }


    public void visitModulePackagesAttribute(Clazz clazz, ModulePackagesAttribute modulePackagesAttribute)
    {
        markAsUsed(modulePackagesAttribute);

        markConstant(clazz, modulePackagesAttribute.u2attributeNameIndex);

        // Mark the constant pool entries referenced by the packages info.
        modulePackagesAttribute.packagesAccept(clazz, this);
    }


    public void visitDeprecatedAttribute(Clazz clazz, DeprecatedAttribute deprecatedAttribute)
    {
        markAsUsed(deprecatedAttribute);

        markConstant(clazz, deprecatedAttribute.u2attributeNameIndex);
    }


    public void visitSyntheticAttribute(Clazz clazz, SyntheticAttribute syntheticAttribute)
    {
        markAsUsed(syntheticAttribute);

        markConstant(clazz, syntheticAttribute.u2attributeNameIndex);
    }


    public void visitSignatureAttribute(Clazz clazz, SignatureAttribute signatureAttribute)
    {
        markAsUsed(signatureAttribute);

        markConstant(clazz, signatureAttribute.u2attributeNameIndex);
        markConstant(clazz, signatureAttribute.u2signatureIndex);

        // Don't mark the referenced classes. We'll clean them up in
        // ClassShrinker, if they appear unused.
        //// Mark the classes referenced in the descriptor string.
        //signatureAttribute.referencedClassesAccept(this);
    }


    public void visitConstantValueAttribute(Clazz clazz, Field field, ConstantValueAttribute constantValueAttribute)
    {
        markAsUsed(constantValueAttribute);

        markConstant(clazz, constantValueAttribute.u2attributeNameIndex);
        markConstant(clazz, constantValueAttribute.u2constantValueIndex);
    }


    public void visitMethodParametersAttribute(Clazz clazz, Method method, MethodParametersAttribute methodParametersAttribute)
    {
        markAsUsed(methodParametersAttribute);

        markConstant(clazz, methodParametersAttribute.u2attributeNameIndex);

        // Mark the constant pool entries referenced by the parameter information.
        methodParametersAttribute.parametersAccept(clazz, method, this);
    }


    public void visitExceptionsAttribute(Clazz clazz, Method method, ExceptionsAttribute exceptionsAttribute)
    {
        markAsUsed(exceptionsAttribute);

        markConstant(clazz, exceptionsAttribute.u2attributeNameIndex);

        // Mark the constant pool entries referenced by the exceptions.
        exceptionsAttribute.exceptionEntriesAccept((ProgramClass)clazz, this);
    }


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        markAsUsed(codeAttribute);

        markConstant(clazz, codeAttribute.u2attributeNameIndex);

        // Mark the constant pool entries referenced by the instructions,
        // by the exceptions, and by the attributes.
        codeAttribute.instructionsAccept(clazz, method, this);
        codeAttribute.exceptionsAccept(clazz, method, this);
        codeAttribute.attributesAccept(clazz, method, this);
    }


    public void visitStackMapAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapAttribute stackMapAttribute)
    {
        markAsUsed(stackMapAttribute);

        markConstant(clazz, stackMapAttribute.u2attributeNameIndex);

        // Mark the constant pool entries referenced by the stack map frames.
        stackMapAttribute.stackMapFramesAccept(clazz, method, codeAttribute, this);
    }


    public void visitStackMapTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapTableAttribute stackMapTableAttribute)
    {
        markAsUsed(stackMapTableAttribute);

        markConstant(clazz, stackMapTableAttribute.u2attributeNameIndex);

        // Mark the constant pool entries referenced by the stack map frames.
        stackMapTableAttribute.stackMapFramesAccept(clazz, method, codeAttribute, this);
    }


    public void visitLineNumberTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute lineNumberTableAttribute)
    {
        markAsUsed(lineNumberTableAttribute);

        markConstant(clazz, lineNumberTableAttribute.u2attributeNameIndex);
    }


    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        // Don't mark the attribute and its contents yet. We may mark them later,
        // in LocalVariableTypeUsageMarker.
        //markAsUsed(localVariableTableAttribute);
        //
        //markConstant(clazz, localVariableTableAttribute.u2attributeNameIndex);
        //
        //// Mark the constant pool entries referenced by the local variables.
        //localVariableTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        // Don't mark the attribute and its contents yet. We may mark them later,
        // in LocalVariableTypeUsageMarker.
        //markAsUsed(localVariableTypeTableAttribute);
        //
        //markConstant(clazz, localVariableTypeTableAttribute.u2attributeNameIndex);
        //
        //// Mark the constant pool entries referenced by the local variable types.
        //localVariableTypeTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitAnyAnnotationsAttribute(Clazz clazz, AnnotationsAttribute annotationsAttribute)
    {
        // Don't mark the attribute and its contents yet. We may mark them later,
        // in AnnotationUsageMarker.
        //markAsUsed(annotationsAttribute);
        //
        //markConstant(clazz, annotationsAttribute.u2attributeNameIndex);
        //
        //// Mark the constant pool entries referenced by the annotations.
        //annotationsAttribute.annotationsAccept(clazz, this);
    }


    public void visitAnyParameterAnnotationsAttribute(Clazz clazz, Method method, ParameterAnnotationsAttribute parameterAnnotationsAttribute)
    {
        // Don't mark the attribute and its contents yet. We may mark them later,
        // in AnnotationUsageMarker.
        //markAsUsed(parameterAnnotationsAttribute);
        //
        //markConstant(clazz, parameterAnnotationsAttribute.u2attributeNameIndex);
        //
        //// Mark the constant pool entries referenced by the annotations.
        //parameterAnnotationsAttribute.annotationsAccept(clazz, method, this);
    }


    public void visitAnnotationDefaultAttribute(Clazz clazz, Method method, AnnotationDefaultAttribute annotationDefaultAttribute)
    {
        // Don't mark the attribute and its contents yet. We may mark them later,
        // in AnnotationUsageMarker.
        //markAsUsed(annotationDefaultAttribute);
        //
        //markConstant(clazz, annotationDefaultAttribute.u2attributeNameIndex);
        //
        //// Mark the constant pool entries referenced by the element value.
        //annotationDefaultAttribute.defaultValueAccept(clazz, this);
    }


    // Implementations for ExceptionInfoVisitor.

    public void visitExceptionInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, ExceptionInfo exceptionInfo)
    {
        markAsUsed(exceptionInfo);

        markOptionalConstant(clazz, exceptionInfo.u2catchType);
    }


    // Implementations for InnerClassesInfoVisitor.

    public void visitInnerClassesInfo(Clazz clazz, InnerClassesInfo innerClassesInfo)
    {
        // At this point, we only mark outer classes of this class.
        // Inner class can be marked later, by InnerUsageMarker.
        if (innerClassesInfo.u2innerClassIndex != 0 &&
            clazz.getName().equals(clazz.getClassName(innerClassesInfo.u2innerClassIndex)))
        {
            markAsUsed(innerClassesInfo);

            // Mark the constant pool entries referenced by the contained info.
            innerClassesInfo.innerClassConstantAccept(clazz, this);
            innerClassesInfo.outerClassConstantAccept(clazz, this);
            innerClassesInfo.innerNameConstantAccept(clazz, this);
        }
    }


    // Implementations for StackMapFrameVisitor.

    public void visitAnyStackMapFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, StackMapFrame stackMapFrame) {}


    public void visitSameOneFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SameOneFrame sameOneFrame)
    {
        // Mark the constant pool entries referenced by the verification types.
        sameOneFrame.stackItemAccept(clazz, method, codeAttribute, offset, this);
    }


    public void visitMoreZeroFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, MoreZeroFrame moreZeroFrame)
    {
        // Mark the constant pool entries referenced by the verification types.
        moreZeroFrame.additionalVariablesAccept(clazz, method, codeAttribute, offset, this);
    }


    public void visitFullFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, FullFrame fullFrame)
    {
        // Mark the constant pool entries referenced by the verification types.
        fullFrame.variablesAccept(clazz, method, codeAttribute, offset, this);
        fullFrame.stackAccept(clazz, method, codeAttribute, offset, this);
    }


    // Implementations for VerificationTypeVisitor.

    public void visitAnyVerificationType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VerificationType verificationType) {}


    public void visitObjectType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ObjectType objectType)
    {
        markConstant(clazz, objectType.u2classIndex);
    }


    // Implementations for ParameterInfoVisitor.

    public void visitParameterInfo(Clazz clazz, Method method, int parameterIndex, ParameterInfo parameterInfo)
    {
        parameterInfo.nameConstantAccept (clazz, this);
    }


    // Implementations for LocalVariableInfoVisitor.

    public void visitLocalVariableInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableInfo localVariableInfo)
    {
        markConstant(clazz, localVariableInfo.u2nameIndex);
        markConstant(clazz, localVariableInfo.u2descriptorIndex);
    }


    // Implementations for LocalVariableTypeInfoVisitor.

    public void visitLocalVariableTypeInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeInfo localVariableTypeInfo)
    {
        markConstant(clazz, localVariableTypeInfo.u2nameIndex);
        markConstant(clazz, localVariableTypeInfo.u2signatureIndex);
    }


    // Implementations for RequiresInfoVisitor.

    public void visitRequiresInfo(Clazz clazz, RequiresInfo requiresInfo)
    {
        markConstant(        clazz, requiresInfo.u2requiresIndex);
        markOptionalConstant(clazz, requiresInfo.u2requiresVersionIndex);
    }


    // Implementations for ExportsInfoVisitor.

    public void visitExportsInfo(Clazz clazz, ExportsInfo exportsInfo)
    {
        markConstant( clazz,  exportsInfo.u2exportsIndex);
        markConstants(clazz, exportsInfo.u2exportsToIndex, exportsInfo.u2exportsToCount);
    }


    // Implementations for OpensInfoVisitor.

    public void visitOpensInfo(Clazz clazz, OpensInfo opensInfo)
    {
        markConstant( clazz, opensInfo.u2opensIndex);
        markConstants(clazz, opensInfo.u2opensToIndex, opensInfo.u2opensToCount);
    }


    // Implementations for ProvidesInfoVisitor.

    public void visitProvidesInfo(Clazz clazz, ProvidesInfo providesInfo)
    {
        markConstant( clazz, providesInfo.u2providesIndex);
        markConstants(clazz, providesInfo.u2providesWithIndex, providesInfo.u2providesWithCount);
    }

//    // Implementations for AnnotationVisitor.
//
//    public void visitAnnotation(Clazz clazz, Annotation annotation)
//    {
//        markConstant(clazz, annotation.u2typeIndex);
//
//        // Mark the constant pool entries referenced by the element values.
//        annotation.elementValuesAccept(clazz, this);
//    }
//
//
//    // Implementations for ElementValueVisitor.
//
//    public void visitConstantElementValue(Clazz clazz, Annotation annotation, ConstantElementValue constantElementValue)
//    {
//        markOptionalConstant(clazz, constantElementValue.u2elementNameIndex);
//        markConstant(        clazz, constantElementValue.u2constantValueIndex);
//    }
//
//
//    public void visitEnumConstantElementValue(Clazz clazz, Annotation annotation, EnumConstantElementValue enumConstantElementValue)
//    {
//        markOptionalConstant(clazz, enumConstantElementValue.u2elementNameIndex);
//        markConstant(        clazz, enumConstantElementValue.u2typeNameIndex);
//        markConstant(        clazz, enumConstantElementValue.u2constantNameIndex);
//    }
//
//
//    public void visitClassElementValue(Clazz clazz, Annotation annotation, ClassElementValue classElementValue)
//    {
//        markOptionalConstant(clazz, classElementValue.u2elementNameIndex);
//
//        // Mark the referenced class constant pool entry.
//        markConstant(clazz, classElementValue.u2classInfoIndex);
//    }
//
//
//    public void visitAnnotationElementValue(Clazz clazz, Annotation annotation, AnnotationElementValue annotationElementValue)
//    {
//        markOptionalConstant(clazz, annotationElementValue.u2elementNameIndex);
//
//        // Mark the constant pool entries referenced by the annotation.
//        annotationElementValue.annotationAccept(clazz, this);
//    }
//
//
//    public void visitArrayElementValue(Clazz clazz, Annotation annotation, ArrayElementValue arrayElementValue)
//    {
//        markOptionalConstant(clazz, arrayElementValue.u2elementNameIndex);
//
//        // Mark the constant pool entries referenced by the element values.
//        arrayElementValue.elementValuesAccept(clazz, annotation, this);
//    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        markConstant(clazz, constantInstruction.constantIndex);

        // Also mark the parameterless constructor of the class, in case the
        // string constant or class constant is being used in a Class.forName
        // or a .class construct.
        clazz.constantPoolEntryAccept(constantInstruction.constantIndex,
                                      parameterlessConstructorMarker);
    }


    // Small utility methods.

    /**
     * Marks the given visitor accepter as being used.
     */
    protected void markAsUsed(VisitorAccepter visitorAccepter)
    {
        visitorAccepter.setVisitorInfo(USED);
    }


    /**
     * Returns whether the given visitor accepter should still be marked as
     * being used.
     */
    protected boolean shouldBeMarkedAsUsed(VisitorAccepter visitorAccepter)
    {
        return visitorAccepter.getVisitorInfo() != USED;
    }


    /**
     * Returns whether the given visitor accepter has been marked as being used.
     */
    protected boolean isUsed(VisitorAccepter visitorAccepter)
    {
        return visitorAccepter.getVisitorInfo() == USED;
    }


    /**
     * Marks the given visitor accepter as possibly being used.
     */
    protected void markAsPossiblyUsed(VisitorAccepter visitorAccepter)
    {
        visitorAccepter.setVisitorInfo(POSSIBLY_USED);
    }


    /**
     * Returns whether the given visitor accepter should still be marked as
     * possibly being used.
     */
    protected boolean shouldBeMarkedAsPossiblyUsed(VisitorAccepter visitorAccepter)
    {
        return visitorAccepter.getVisitorInfo() != USED &&
               visitorAccepter.getVisitorInfo() != POSSIBLY_USED;
    }


    /**
     * Returns whether the given visitor accepter has been marked as possibly
     * being used.
     */
    protected boolean isPossiblyUsed(VisitorAccepter visitorAccepter)
    {
        return visitorAccepter.getVisitorInfo() == POSSIBLY_USED;
    }


    /**
     * Clears any usage marks from the given visitor accepter.
     */
    protected void markAsUnused(VisitorAccepter visitorAccepter)
    {
        visitorAccepter.setVisitorInfo(null);
    }


    /**
     * Marks the specified constant pool entries of the given class.
     * This includes visiting any referenced objects.
     */
    private void markConstants(Clazz clazz,
                               int[] constantIndices,
                               int   constantIndicesCount)
    {
        for (int index = 0; index < constantIndicesCount; index++)
        {
            markConstant(clazz, constantIndices[index]);
        }
    }


    /**
     * Marks the specified constant pool entry of the given class, if the index
     * is not 0. This includes visiting any referenced objects.
     */
    private void markOptionalConstant(Clazz clazz, int constantIndex)
    {
        if (constantIndex != 0)
        {
            markConstant(clazz, constantIndex);
        }
    }


    /**
     * Marks the specified constant pool entry of the given class.
     * This includes visiting any referenced objects.
     */
    private void markConstant(Clazz clazz, int constantIndex)
    {
        clazz.constantPoolEntryAccept(constantIndex, this);
    }
}

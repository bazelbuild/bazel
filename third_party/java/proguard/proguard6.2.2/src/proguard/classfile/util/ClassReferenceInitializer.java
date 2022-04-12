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
package proguard.classfile.util;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.annotation.*;
import proguard.classfile.attribute.annotation.visitor.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.visitor.*;


/**
 * This ClassVisitor initializes the references of all classes that
 * it visits.
 * <p>
 * All class constant pool entries get direct references to the corresponding
 * classes. These references make it more convenient to travel up and across
 * the class hierarchy.
 * <p>
 * All field and method reference constant pool entries get direct references
 * to the corresponding classes, fields, and methods.
 * <p>
 * All name and type constant pool entries get a list of direct references to
 * the classes listed in the type.
 * <p>
 * This visitor optionally prints warnings if some items can't be found.
 * <p>
 * The class hierarchy must be initialized before using this visitor.
 *
 * @author Eric Lafortune
 */
public class ClassReferenceInitializer
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor,
             ConstantVisitor,
             AttributeVisitor,
             LocalVariableInfoVisitor,
             LocalVariableTypeInfoVisitor,
             AnnotationVisitor,
             ElementValueVisitor
{
    private final ClassPool      programClassPool;
    private final ClassPool      libraryClassPool;
    private final WarningPrinter missingClassWarningPrinter;
    private final WarningPrinter missingProgramMemberWarningPrinter;
    private final WarningPrinter missingLibraryMemberWarningPrinter;
    private final WarningPrinter dependencyWarningPrinter;

    private final MemberFinder memberFinder = new MemberFinder();


    /**
     * Creates a new ClassReferenceInitializer that initializes the references
     * of all visited class files.
     */
    public ClassReferenceInitializer(ClassPool programClassPool,
                                     ClassPool libraryClassPool)
    {
        this(programClassPool, libraryClassPool, null, null, null, null);
    }


    /**
     * Creates a new ClassReferenceInitializer that initializes the references
     * of all visited class files, optionally printing warnings if some classes
     * or class members can't be found or if they are in the program class pool.
     */
    public ClassReferenceInitializer(ClassPool      programClassPool,
                                     ClassPool      libraryClassPool,
                                     WarningPrinter missingClassWarningPrinter,
                                     WarningPrinter missingProgramMemberWarningPrinter,
                                     WarningPrinter missingLibraryMemberWarningPrinter,
                                     WarningPrinter dependencyWarningPrinter)
    {
        this.programClassPool                   = programClassPool;
        this.libraryClassPool                   = libraryClassPool;
        this.missingClassWarningPrinter         = missingClassWarningPrinter;
        this.missingProgramMemberWarningPrinter = missingProgramMemberWarningPrinter;
        this.missingLibraryMemberWarningPrinter = missingLibraryMemberWarningPrinter;
        this.dependencyWarningPrinter           = dependencyWarningPrinter;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Initialize the constant pool entries.
        programClass.constantPoolEntriesAccept(this);

        // Initialize all fields and methods.
        programClass.fieldsAccept(this);
        programClass.methodsAccept(this);

        // Initialize the attributes.
        programClass.attributesAccept(this);
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        // Initialize all fields and methods.
        libraryClass.fieldsAccept(this);
        libraryClass.methodsAccept(this);
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        programField.referencedClass =
            findReferencedClass(programClass,
                                programField.getDescriptor(programClass));

        // Initialize the attributes.
        programField.attributesAccept(programClass, this);
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        programMethod.referencedClasses =
            findReferencedClasses(programClass,
                                  programMethod.getDescriptor(programClass));

        // Initialize the attributes.
        programMethod.attributesAccept(programClass, this);
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
    {
        libraryField.referencedClass =
            findReferencedClass(libraryClass,
                                libraryField.getDescriptor(libraryClass));
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        libraryMethod.referencedClasses =
            findReferencedClasses(libraryClass,
                                  libraryMethod.getDescriptor(libraryClass));
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        // Fill out the String class.
        stringConstant.javaLangStringClass =
            findClass(clazz, ClassConstants.NAME_JAVA_LANG_STRING);
    }


    public void visitDynamicConstant(Clazz clazz, DynamicConstant dynamicConstant)
    {
        dynamicConstant.referencedClasses =
            findReferencedClasses(clazz,
                                  dynamicConstant.getType(clazz));
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        invokeDynamicConstant.referencedClasses =
            findReferencedClasses(clazz,
                                  invokeDynamicConstant.getType(clazz));
    }


    public void visitMethodHandleConstant(Clazz clazz, MethodHandleConstant methodHandleConstant)
    {
        // Fill out the MethodHandle class.
        methodHandleConstant.javaLangInvokeMethodHandleClass =
            findClass(clazz, ClassConstants.NAME_JAVA_LANG_INVOKE_METHOD_HANDLE);
    }


    public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
    {
        String className = refConstant.getClassName(clazz);

        // Methods for array types should be found in the Object class.
        if (ClassUtil.isInternalArrayType(className))
        {
            className = ClassConstants.NAME_JAVA_LANG_OBJECT;
        }

        // See if we can find the referenced class.
        // Unresolved references are assumed to refer to library classes
        // that will not change anyway.
        Clazz referencedClass = findClass(clazz, className);

        if (referencedClass != null)
        {
            String name = refConstant.getName(clazz);
            String type = refConstant.getType(clazz);

            boolean isFieldRef = refConstant.getTag() == ClassConstants.CONSTANT_Fieldref;

            // See if we can find the referenced class member somewhere in the
            // hierarchy.
            refConstant.referencedMember = memberFinder.findMember(clazz,
                                                                   referencedClass,
                                                                   name,
                                                                   type,
                                                                   isFieldRef);
            refConstant.referencedClass  = memberFinder.correspondingClass();

            if (refConstant.referencedMember == null)
            {
                // We haven't found the class member anywhere in the hierarchy.
                boolean isProgramClass = referencedClass instanceof ProgramClass;

                WarningPrinter missingMemberWarningPrinter = isProgramClass ?
                    missingProgramMemberWarningPrinter :
                    missingLibraryMemberWarningPrinter;

                if (missingMemberWarningPrinter != null)
                {
                    missingMemberWarningPrinter.print(clazz.getName(),
                                                      className,
                                                      "Warning: " +
                                                      ClassUtil.externalClassName(clazz.getName()) +
                                                      ": can't find referenced " +
                                                      (isFieldRef ?
                                                          "field '"  + ClassUtil.externalFullFieldDescription(0, name, type) :
                                                          "method '" + ClassUtil.externalFullMethodDescription(className, 0, name, type)) +
                                                      "' in " +
                                                      (isProgramClass ?
                                                          "program" :
                                                          "library") +
                                                      " class " +
                                                      ClassUtil.externalClassName(className));
                }
            }
        }
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        // Fill out the referenced class.
        classConstant.referencedClass =
            findClass(clazz, ClassUtil.internalClassNameFromClassType(classConstant.getName(clazz)));

        // Fill out the Class class.
        classConstant.javaLangClassClass =
            findClass(clazz, ClassConstants.NAME_JAVA_LANG_CLASS);
    }


    public void visitMethodTypeConstant(Clazz clazz, MethodTypeConstant methodTypeConstant)
    {
        // Fill out the MethodType class.
        methodTypeConstant.javaLangInvokeMethodTypeClass =
            findClass(clazz, ClassConstants.NAME_JAVA_LANG_INVOKE_METHOD_TYPE);

        methodTypeConstant.referencedClasses =
            findReferencedClasses(clazz,
                                  methodTypeConstant.getType(clazz));
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitEnclosingMethodAttribute(Clazz clazz, EnclosingMethodAttribute enclosingMethodAttribute)
    {
        String enclosingClassName = enclosingMethodAttribute.getClassName(clazz);

        // See if we can find the referenced class.
        enclosingMethodAttribute.referencedClass =
            findClass(clazz, enclosingClassName);

        if (enclosingMethodAttribute.referencedClass != null)
        {
            // Is there an enclosing method? Otherwise it's just initialization
            // code outside of the constructors.
            if (enclosingMethodAttribute.u2nameAndTypeIndex != 0)
            {
                String name = enclosingMethodAttribute.getName(clazz);
                String type = enclosingMethodAttribute.getType(clazz);

                // See if we can find the method in the referenced class.
                enclosingMethodAttribute.referencedMethod =
                    enclosingMethodAttribute.referencedClass.findMethod(name, type);

                if (enclosingMethodAttribute.referencedMethod == null &&
                    missingProgramMemberWarningPrinter != null)
                {
                    // We couldn't find the enclosing method.
                    String className = clazz.getName();

                    missingProgramMemberWarningPrinter.print(className,
                                                             enclosingClassName,
                                                             "Warning: " +
                                                             ClassUtil.externalClassName(className) +
                                                             ": can't find enclosing method '" +
                                                             ClassUtil.externalFullMethodDescription(enclosingClassName, 0, name, type) +
                                                             "' in program class " +
                                                             ClassUtil.externalClassName(enclosingClassName));
                }
            }
        }
    }


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Initialize the nested attributes.
        codeAttribute.attributesAccept(clazz, method, this);
    }


    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        // Initialize the local variables.
        localVariableTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        // Initialize the local variable types.
        localVariableTypeTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitSignatureAttribute(Clazz clazz, SignatureAttribute signatureAttribute)
    {
        signatureAttribute.referencedClasses =
            findReferencedClasses(clazz,
                                  signatureAttribute.getSignature(clazz));
    }


    public void visitAnyAnnotationsAttribute(Clazz clazz, AnnotationsAttribute annotationsAttribute)
    {
        // Initialize the annotations.
        annotationsAttribute.annotationsAccept(clazz, this);
    }


    public void visitAnyParameterAnnotationsAttribute(Clazz clazz, Method method, ParameterAnnotationsAttribute parameterAnnotationsAttribute)
    {
        // Initialize the annotations.
        parameterAnnotationsAttribute.annotationsAccept(clazz, method, this);
    }


    public void visitAnnotationDefaultAttribute(Clazz clazz, Method method, AnnotationDefaultAttribute annotationDefaultAttribute)
    {
        // Initialize the annotation.
        annotationDefaultAttribute.defaultValueAccept(clazz, this);
    }


    // Implementations for LocalVariableInfoVisitor.

    public void visitLocalVariableInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableInfo localVariableInfo)
    {
        localVariableInfo.referencedClass =
            findReferencedClass(clazz,
                                localVariableInfo.getDescriptor(clazz));
    }


    // Implementations for LocalVariableTypeInfoVisitor.

    public void visitLocalVariableTypeInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeInfo localVariableTypeInfo)
    {
        localVariableTypeInfo.referencedClasses =
            findReferencedClasses(clazz,
                                  localVariableTypeInfo.getSignature(clazz));
    }


    // Implementations for AnnotationVisitor.

    public void visitAnnotation(Clazz clazz, Annotation annotation)
    {
        annotation.referencedClasses =
            findReferencedClasses(clazz,
                                  annotation.getType(clazz));

        // Initialize the element values.
        annotation.elementValuesAccept(clazz, this);
    }


    // Implementations for ElementValueVisitor.

    public void visitConstantElementValue(Clazz clazz, Annotation annotation, ConstantElementValue constantElementValue)
    {
        initializeElementValue(clazz, annotation, constantElementValue);
    }


    public void visitEnumConstantElementValue(Clazz clazz, Annotation annotation, EnumConstantElementValue enumConstantElementValue)
    {
        initializeElementValue(clazz, annotation, enumConstantElementValue);

        enumConstantElementValue.referencedClasses =
            findReferencedClasses(clazz,
                                  enumConstantElementValue.getTypeName(clazz));
    }


    public void visitClassElementValue(Clazz clazz, Annotation annotation, ClassElementValue classElementValue)
    {
        initializeElementValue(clazz, annotation, classElementValue);

        classElementValue.referencedClasses =
            findReferencedClasses(clazz,
                                  classElementValue.getClassName(clazz));
    }


    public void visitAnnotationElementValue(Clazz clazz, Annotation annotation, AnnotationElementValue annotationElementValue)
    {
        initializeElementValue(clazz, annotation, annotationElementValue);

        // Initialize the annotation.
        annotationElementValue.annotationAccept(clazz, this);
    }


    public void visitArrayElementValue(Clazz clazz, Annotation annotation, ArrayElementValue arrayElementValue)
    {
        initializeElementValue(clazz, annotation, arrayElementValue);

        // Initialize the element values.
        arrayElementValue.elementValuesAccept(clazz, annotation, this);
    }


    /**
     * Initializes the referenced method of an element value, if any.
     */
    private void initializeElementValue(Clazz clazz, Annotation annotation, ElementValue elementValue)
    {
        // See if we have a referenced class.
        if (annotation                      != null &&
            annotation.referencedClasses    != null &&
            elementValue.u2elementNameIndex != 0)
        {
            // See if we can find the method in the referenced class
            // (ignoring the descriptor).
            String name = elementValue.getMethodName(clazz);

            Clazz referencedClass = annotation.referencedClasses[0];
            elementValue.referencedClass  = referencedClass;
            elementValue.referencedMethod = referencedClass.findMethod(name, null);
        }
    }


    // Small utility methods.

    /**
     * Returns the single class referenced by the given descriptor, or
     * <code>null</code> if there isn't any useful reference.
     */
    private Clazz findReferencedClass(Clazz  referencingClass,
                                      String descriptor)
    {
        DescriptorClassEnumeration enumeration =
            new DescriptorClassEnumeration(descriptor);

        enumeration.nextFluff();

        if (enumeration.hasMoreClassNames())
        {
            return findClass(referencingClass, enumeration.nextClassName());
        }

        return null;
    }


    /**
     * Returns an array of classes referenced by the given descriptor, or
     * <code>null</code> if there aren't any useful references.
     */
    private Clazz[] findReferencedClasses(Clazz  referencingClass,
                                          String descriptor)
    {
        DescriptorClassEnumeration enumeration =
            new DescriptorClassEnumeration(descriptor);

        int classCount = enumeration.classCount();
        if (classCount > 0)
        {
            Clazz[] referencedClasses = new Clazz[classCount];

            boolean foundReferencedClasses = false;

            for (int index = 0; index < classCount; index++)
            {
                String fluff = enumeration.nextFluff();
                String name  = enumeration.nextClassName();

                Clazz referencedClass = findClass(referencingClass, name);

                if (referencedClass != null)
                {
                    referencedClasses[index] = referencedClass;
                    foundReferencedClasses = true;
                }
            }

            if (foundReferencedClasses)
            {
                return referencedClasses;
            }
        }

        return null;
    }


    /**
     * Returns the class with the given name, either for the program class pool
     * or from the library class pool, or <code>null</code> if it can't be found.
     */
    private Clazz findClass(Clazz referencingClass, String name)
    {
        // Is it an array type?
        if (ClassUtil.isInternalArrayType(name))
        {
            // Ignore any primitive array types.
            if (!ClassUtil.isInternalClassType(name))
            {
                return null;
            }

            // Strip the array part.
            name = ClassUtil.internalClassNameFromClassType(name);
        }

        // First look for the class in the program class pool.
        Clazz clazz = programClassPool.getClass(name);

        // Otherwise look for the class in the library class pool.
        if (clazz == null)
        {
            clazz = libraryClassPool.getClass(name);

            if (clazz == null &&
                missingClassWarningPrinter != null)
            {
                // We didn't find the superclass or interface. Print a warning.
                String referencingClassName = referencingClass.getName();

                missingClassWarningPrinter.print(referencingClassName,
                                                 name,
                                                 "Warning: " +
                                                 ClassUtil.externalClassName(referencingClassName) +
                                                 ": can't find referenced class " +
                                                 ClassUtil.externalClassName(name));
            }
        }
        else if (dependencyWarningPrinter != null)
        {
            // The superclass or interface was found in the program class pool.
            // Print a warning.
            String referencingClassName = referencingClass.getName();

            dependencyWarningPrinter.print(referencingClassName,
                                           name,
                                           "Warning: library class " +
                                           ClassUtil.externalClassName(referencingClassName) +
                                           " depends on program class " +
                                           ClassUtil.externalClassName(name));
        }

        return clazz;
    }
}

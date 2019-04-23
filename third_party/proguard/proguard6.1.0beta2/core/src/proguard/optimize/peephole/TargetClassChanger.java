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
package proguard.optimize.peephole;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.annotation.*;
import proguard.classfile.attribute.annotation.visitor.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.editor.*;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;

/**
 * This ClassVisitor replaces references to classes and class members if the
 * classes have targets that are intended to replace them.
 *
 * @see VerticalClassMerger
 * @see ClassReferenceFixer
 * @see MemberReferenceFixer
 * @author Eric Lafortune
 */
public class TargetClassChanger
extends      SimplifiedVisitor
implements   ClassVisitor,
             ConstantVisitor,
             MemberVisitor,
             AttributeVisitor,
             LocalVariableInfoVisitor,
             LocalVariableTypeInfoVisitor,
             AnnotationVisitor,
             ElementValueVisitor
{
    private static final boolean DEBUG = false;


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // We're only making changes locally in the class.
        // Not all other classes may have been retargeted yet.

        // Change the references of the constant pool.
        programClass.constantPoolEntriesAccept(this);

        // Change the references of the class members.
        programClass.fieldsAccept(this);
        programClass.methodsAccept(this);

        // Change the references of the attributes.
        programClass.attributesAccept(this);

        // Remove duplicate interfaces and interface classes that have ended
        // up pointing to the class itself.
        boolean[] delete = null;
        for (int index = 0; index < programClass.u2interfacesCount; index++)
        {
            Clazz interfaceClass = programClass.getInterface(index);
            if (interfaceClass != null &&
                (programClass.equals(interfaceClass) ||
                 containsInterfaceClass(programClass,
                                        index,
                                        interfaceClass)))
            {
                // Lazily create the array.
                if (delete == null)
                {
                    delete = new boolean[programClass.u2interfacesCount];
                }

                delete[index] = true;
            }
        }

        if (delete != null)
        {
            new InterfaceDeleter(delete).visitProgramClass(programClass);
        }

        // Is the class being retargeted?
        Clazz targetClass = ClassMerger.getTargetClass(programClass);
        if (targetClass != null)
        {
            // We're not changing anything special in the superclass and
            // interface hierarchy of the retargeted class. The shrinking
            // step will remove the class for us.

            // Restore the class name. We have to add a new class entry
            // to avoid an existing entry with the same name being reused. The
            // names have to be fixed later, based on their referenced classes.
            programClass.u2thisClass =
                addNewClassConstant(programClass,
                                    programClass.getName(),
                                    programClass);

            // This class will no longer have any subclasses, because their
            // subclasses and interfaces will be retargeted.
            programClass.subClasses = null;
        }
        else
        {
            // This class has become the subclass of its possibly new
            // superclass and of any new interfaces.
            ConstantVisitor subclassAdder =
                new ReferencedClassVisitor(
                new SubclassFilter(programClass,
                new SubclassAdder(programClass)));

            programClass.superClassConstantAccept(subclassAdder);
            programClass.interfaceConstantsAccept(subclassAdder);

            // TODO: Maybe restore private method references.
        }
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        // Change the references of the class members.
        libraryClass.fieldsAccept(this);
        libraryClass.methodsAccept(this);
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        // Change the referenced class.
        programField.referencedClass =
            updateReferencedClass(programField.referencedClass);

        // Change the references of the attributes.
        programField.attributesAccept(programClass, this);
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        // Change the referenced classes.
        updateReferencedClasses(programMethod.referencedClasses);

        // Change the references of the attributes.
        programMethod.attributesAccept(programClass, this);
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
    {
        // Change the referenced class.
        libraryField.referencedClass =
            updateReferencedClass(libraryField.referencedClass);
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        // Change the referenced classes.
        updateReferencedClasses(libraryMethod.referencedClasses);
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        // Does the string refer to a class, due to a Class.forName construct?
        Clazz referencedClass    = stringConstant.referencedClass;
        Clazz newReferencedClass = updateReferencedClass(referencedClass);
        if (referencedClass != newReferencedClass)
        {
            // Change the referenced class.
            stringConstant.referencedClass = newReferencedClass;

            // Change the referenced class member, if applicable.
            stringConstant.referencedMember =
                updateReferencedMember(stringConstant.referencedMember,
                                       stringConstant.getString(clazz),
                                       null,
                                       newReferencedClass);
        }
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        // Change the referenced classes.
        updateReferencedClasses(invokeDynamicConstant.referencedClasses);
    }


    public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
    {
        Clazz referencedClass    = refConstant.referencedClass;
        Clazz newReferencedClass = updateReferencedClass(referencedClass);
        if (referencedClass != newReferencedClass)
        {
            if (DEBUG)
            {
                System.out.println("TargetClassChanger:");
                System.out.println("  ["+clazz.getName()+"] changing reference from ["+refConstant.referencedClass+"."+refConstant.referencedMember.getName(refConstant.referencedClass)+refConstant.referencedMember.getDescriptor(refConstant.referencedClass)+"]");
            }

            // Change the referenced class.
            refConstant.referencedClass  = newReferencedClass;

            // Change the referenced class member.
            refConstant.referencedMember =
                updateReferencedMember(refConstant.referencedMember,
                                       refConstant.getName(clazz),
                                       refConstant.getType(clazz),
                                       newReferencedClass);

            if (DEBUG)
            {
                System.out.println("  ["+clazz.getName()+"]                    to   ["+refConstant.referencedClass+"."+refConstant.referencedMember.getName(refConstant.referencedClass)+refConstant.referencedMember.getDescriptor(refConstant.referencedClass)+"]");
            }
        }
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        // Change the referenced class.
        classConstant.referencedClass =
            updateReferencedClass(classConstant.referencedClass);
    }


    public void visitMethodTypeConstant(Clazz clazz, MethodTypeConstant methodTypeConstant)
    {
        updateReferencedClasses(methodTypeConstant.referencedClasses);
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Change the references of the attributes.
        codeAttribute.attributesAccept(clazz, method, this);
    }


    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        // Change the references of the local variables.
        localVariableTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        // Change the references of the local variables.
        localVariableTypeTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitSignatureAttribute(Clazz clazz, SignatureAttribute signatureAttribute)
    {
        // Change the referenced classes.
        updateReferencedClasses(signatureAttribute.referencedClasses);
    }


    public void visitAnyAnnotationsAttribute(Clazz clazz, AnnotationsAttribute annotationsAttribute)
    {
        // Change the references of the annotations.
        annotationsAttribute.annotationsAccept(clazz, this);
    }


    public void visitAnyParameterAnnotationsAttribute(Clazz clazz, Method method, ParameterAnnotationsAttribute parameterAnnotationsAttribute)
    {
        // Change the references of the annotations.
        parameterAnnotationsAttribute.annotationsAccept(clazz, method, this);
    }


    public void visitAnnotationDefaultAttribute(Clazz clazz, Method method, AnnotationDefaultAttribute annotationDefaultAttribute)
    {
        // Change the references of the annotation.
        annotationDefaultAttribute.defaultValueAccept(clazz, this);
    }


    // Implementations for LocalVariableInfoVisitor.

    public void visitLocalVariableInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableInfo localVariableInfo)
    {
        // Change the referenced class.
        localVariableInfo.referencedClass =
            updateReferencedClass(localVariableInfo.referencedClass);
    }


    // Implementations for LocalVariableTypeInfoVisitor.

    public void visitLocalVariableTypeInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeInfo localVariableTypeInfo)
    {
        // Change the referenced classes.
        updateReferencedClasses(localVariableTypeInfo.referencedClasses);
    }


    // Implementations for AnnotationVisitor.

    public void visitAnnotation(Clazz clazz, Annotation annotation)
    {
        // Change the referenced classes.
        updateReferencedClasses(annotation.referencedClasses);

        // Change the references of the element values.
        annotation.elementValuesAccept(clazz, this);
    }


    // Implementations for ElementValueVisitor.

    public void visitAnyElementValue(Clazz clazz, Annotation annotation, ElementValue elementValue)
    {
        Clazz referencedClass    = elementValue.referencedClass;
        Clazz newReferencedClass = updateReferencedClass(referencedClass);
        if (referencedClass != newReferencedClass)
        {
            // Change the referenced annotation class.
            elementValue.referencedClass  = newReferencedClass;

            // Change the referenced method.
            elementValue.referencedMethod =
                (Method)updateReferencedMember(elementValue.referencedMethod,
                                               elementValue.getMethodName(clazz),
                                               null,
                                               newReferencedClass);
        }
    }


    public void visitConstantElementValue(Clazz clazz, Annotation annotation, ConstantElementValue constantElementValue)
    {
        // Change the referenced annotation class and method.
        visitAnyElementValue(clazz, annotation, constantElementValue);
    }


    public void visitEnumConstantElementValue(Clazz clazz, Annotation annotation, EnumConstantElementValue enumConstantElementValue)
    {
        // Change the referenced annotation class and method.
        visitAnyElementValue(clazz, annotation, enumConstantElementValue);

        // Change the referenced classes.
        updateReferencedClasses(enumConstantElementValue.referencedClasses);
    }


    public void visitClassElementValue(Clazz clazz, Annotation annotation, ClassElementValue classElementValue)
    {
        // Change the referenced annotation class and method.
        visitAnyElementValue(clazz, annotation, classElementValue);

        // Change the referenced classes.
        updateReferencedClasses(classElementValue.referencedClasses);
    }


    public void visitAnnotationElementValue(Clazz clazz, Annotation annotation, AnnotationElementValue annotationElementValue)
    {
        // Change the referenced annotation class and method.
        visitAnyElementValue(clazz, annotation, annotationElementValue);

        // Change the references of the annotation.
        annotationElementValue.annotationAccept(clazz, this);
    }


    public void visitArrayElementValue(Clazz clazz, Annotation annotation, ArrayElementValue arrayElementValue)
    {
        // Change the referenced annotation class and method.
        visitAnyElementValue(clazz, annotation, arrayElementValue);

        // Change the references of the element values.
        arrayElementValue.elementValuesAccept(clazz, annotation, this);
    }


    // Small utility methods.

     /**
     * Returns whether the given class contains the given interface
     * class in its first given number of interfaces.
     */
    private boolean containsInterfaceClass(Clazz clazz,
                                           int   interfaceCount,
                                           Clazz interfaceClass)
    {
        for (int index = 0; index < interfaceCount; index++)
        {
            if (interfaceClass.equals(clazz.getInterface(index)))
            {
                return true;
            }
        }

        return false;
    }


   /**
     * Updates the retargeted classes in the given array of classes.
     */
    private void updateReferencedClasses(Clazz[] referencedClasses)
    {
        if (referencedClasses == null)
        {
            return;
        }

        for (int index = 0; index < referencedClasses.length; index++)
        {
            referencedClasses[index] =
                updateReferencedClass(referencedClasses[index]);
        }
    }


    /**
     * Returns the retargeted class of the given class.
     */
    private Clazz updateReferencedClass(Clazz referencedClass)
    {
        if (referencedClass == null)
        {
            return null;
        }

        Clazz targetClazz = ClassMerger.getTargetClass(referencedClass);
        return targetClazz != null ?
            targetClazz :
            referencedClass;
    }


    /**
     * Returns the retargeted class member of the given class member.
     */
    private Member updateReferencedMember(Member referencedMember,
                                          String name,
                                          String type,
                                          Clazz  newReferencedClass)
    {
        if (referencedMember == null)
        {
            return null;
        }

        return referencedMember instanceof Field ?
            (Member)newReferencedClass.findField(name, type) :
            (Member)newReferencedClass.findMethod(name, type);
    }


    /**
     * Explicitly adds a new class constant for the given class in the given
     * program class.
     */
    private int addNewClassConstant(ProgramClass programClass,
                                    String       className,
                                    Clazz        referencedClass)
    {
        ConstantPoolEditor constantPoolEditor =
            new ConstantPoolEditor(programClass);

        int nameIndex =
            constantPoolEditor.addUtf8Constant(className);

        int classConstantIndex =
            constantPoolEditor.addConstant(new ClassConstant(nameIndex,
                                                             referencedClass));
        return classConstantIndex;
    }
}
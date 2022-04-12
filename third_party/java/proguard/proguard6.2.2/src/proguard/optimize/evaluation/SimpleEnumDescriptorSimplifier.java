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
package proguard.optimize.evaluation;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.editor.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.optimize.KeepMarker;
import proguard.optimize.info.*;

/**
 * This ClassVisitor simplifies the descriptors that contain simple enums in
 * the program classes that it visits.
 *
 * @see SimpleEnumMarker
 * @see MemberReferenceFixer
 * @author Eric Lafortune
 */
public class SimpleEnumDescriptorSimplifier
extends      SimplifiedVisitor
implements   ClassVisitor,
             ConstantVisitor,
             MemberVisitor,
             AttributeVisitor,
             LocalVariableInfoVisitor,
             LocalVariableTypeInfoVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("enum") != null;
    //*/

    private static final boolean DEBUG_EXTRA = false;


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        if (DEBUG)
        {
            System.out.println("SimpleEnumDescriptorSimplifier: "+programClass.getName());
        }

        // Simplify the class members.
        programClass.fieldsAccept(this);
        programClass.methodsAccept(this);

        // Simplify the attributes.
        //programClass.attributesAccept(this);

        // Simplify the simple enum array constants.
        programClass.constantPoolEntriesAccept(this);
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        // Does the constant refer to a simple enum type?
        Clazz referencedClass = stringConstant.referencedClass;
        if (isSimpleEnum(referencedClass))
        {
            // Is it an array type?
            String name = stringConstant.getString(clazz);
            if (ClassUtil.isInternalArrayType(name))
            {
                // Update the type.
                ConstantPoolEditor constantPoolEditor =
                    new ConstantPoolEditor((ProgramClass)clazz);

                String newName = simplifyDescriptor(name, referencedClass);

                stringConstant.u2stringIndex =
                    constantPoolEditor.addUtf8Constant(newName);

                // Clear the referenced class.
                stringConstant.referencedClass = null;
            }
        }
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        // Update the descriptor if it has any simple enum classes.
        String descriptor    = invokeDynamicConstant.getType(clazz);
        String newDescriptor = simplifyDescriptor(descriptor, invokeDynamicConstant.referencedClasses);

        if (!descriptor.equals(newDescriptor))
        {
            // Update the descriptor.
            ConstantPoolEditor constantPoolEditor =
                new ConstantPoolEditor((ProgramClass)clazz);

            invokeDynamicConstant.u2nameAndTypeIndex =
                constantPoolEditor.addNameAndTypeConstant(invokeDynamicConstant.getName(clazz),
                                                          newDescriptor);

            // Update the referenced classes.
            invokeDynamicConstant.referencedClasses =
                simplifyReferencedClasses(descriptor, invokeDynamicConstant.referencedClasses);
        }
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        // Does the constant refer to a simple enum type?
        Clazz referencedClass = classConstant.referencedClass;
        if (isSimpleEnum(referencedClass))
        {
            // Is it an array type?
            String name = classConstant.getName(clazz);
            if (ClassUtil.isInternalArrayType(name))
            {
                // Update the type.
                ConstantPoolEditor constantPoolEditor =
                    new ConstantPoolEditor((ProgramClass)clazz);

                String newName = simplifyDescriptor(name, referencedClass);

                classConstant.u2nameIndex =
                    constantPoolEditor.addUtf8Constant(newName);

                // Clear the referenced class.
                classConstant.referencedClass = null;
            }
        }
    }


    public void visitMethodTypeConstant(Clazz clazz, MethodTypeConstant methodTypeConstant)
    {
        // Update the descriptor if it has any simple enum classes.
        String descriptor    = methodTypeConstant.getType(clazz);
        String newDescriptor = simplifyDescriptor(descriptor, methodTypeConstant.referencedClasses);

        if (!descriptor.equals(newDescriptor))
        {
            // Update the descriptor.
            ConstantPoolEditor constantPoolEditor =
                new ConstantPoolEditor((ProgramClass)clazz);

            methodTypeConstant.u2descriptorIndex =
                constantPoolEditor.addUtf8Constant(newDescriptor);

            // Update the referenced classes.
            methodTypeConstant.referencedClasses =
                simplifyReferencedClasses(descriptor, methodTypeConstant.referencedClasses);
        }
    }



    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        // Update the descriptor if it has a simple enum class.
        String descriptor    = programField.getDescriptor(programClass);
        String newDescriptor = simplifyDescriptor(descriptor, programField.referencedClass);

        if (!descriptor.equals(newDescriptor))
        {
            String name    = programField.getName(programClass);
            String newName = name + ClassConstants.SPECIAL_MEMBER_SEPARATOR + Long.toHexString(Math.abs((descriptor).hashCode()));

            if (DEBUG)
            {
                System.out.println("SimpleEnumDescriptorSimplifier: ["+programClass.getName()+"."+name+" "+descriptor + "] -> ["+newName+" "+newDescriptor+"]");
            }

            ConstantPoolEditor constantPoolEditor =
                new ConstantPoolEditor(programClass);

            // Update the name.
            programField.u2nameIndex =
                constantPoolEditor.addUtf8Constant(newName);

            // Update the descriptor itself.
            programField.u2descriptorIndex =
                constantPoolEditor.addUtf8Constant(newDescriptor);

            // Clear the referenced class.
            programField.referencedClass = null;

            // Clear the enum flag.
            programField.u2accessFlags &= ~ClassConstants.ACC_ENUM;

            // Clear the field value.
            if (!KeepMarker.isKept(programField))
            {
                ProgramFieldOptimizationInfo.getProgramFieldOptimizationInfo(programField).resetValue(programClass, programField);
            }

            // Simplify the signature.
            programField.attributesAccept(programClass, this);
        }
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
//        // Skip the valueOf method.
//        if (programMethod.getName(programClass).equals(ClassConstants.METHOD_NAME_VALUEOF))
//        {
//            return;
//        }

        // Simplify the code, the signature, and the parameter annotations,
        // before simplifying the descriptor.
        programMethod.attributesAccept(programClass, this);

        // Update the descriptor if it has any simple enum classes.
        String descriptor    = programMethod.getDescriptor(programClass);
        String newDescriptor = simplifyDescriptor(descriptor, programMethod.referencedClasses);

        if (!descriptor.equals(newDescriptor))
        {
            String name    = programMethod.getName(programClass);
            String newName = name;

            // Append a code, if the method isn't a class instance initializer.
            if (!name.equals(ClassConstants.METHOD_NAME_INIT))
            {
                newName += ClassConstants.SPECIAL_MEMBER_SEPARATOR + Long.toHexString(Math.abs((descriptor).hashCode()));
            }

            if (DEBUG)
            {
                System.out.println("SimpleEnumDescriptorSimplifier: ["+programClass.getName()+"."+name+descriptor+"] -> ["+newName+newDescriptor+"]");
            }

            ConstantPoolEditor constantPoolEditor =
                new ConstantPoolEditor(programClass);

            // Update the name, if necessary.
            if (!newName.equals(name))
            {
                programMethod.u2nameIndex =
                    constantPoolEditor.addUtf8Constant(newName);
            }

            // Update the descriptor itself.
            programMethod.u2descriptorIndex =
                constantPoolEditor.addUtf8Constant(newDescriptor);

            // Update the referenced classes.
            programMethod.referencedClasses =
                simplifyReferencedClasses(descriptor, programMethod.referencedClasses);
        }
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Simplify the local variable descriptors.
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
        // Compute the new signature.
        String signature    = signatureAttribute.getSignature(clazz);
        String newSignature = simplifyDescriptor(signature,
                                                 signatureAttribute.referencedClasses);

        if (!signature.equals(newSignature))
        {
            // Update the signature.
            signatureAttribute.u2signatureIndex =
                new ConstantPoolEditor((ProgramClass)clazz).addUtf8Constant(newSignature);

            // Update the referenced classes.
            signatureAttribute.referencedClasses =
                simplifyReferencedClasses(signature, signatureAttribute.referencedClasses);
        }
    }


   // Implementations for LocalVariableInfoVisitor.

    public void visitLocalVariableInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableInfo localVariableInfo)
    {
        // Update the descriptor if it has a simple enum class.
        String descriptor    = localVariableInfo.getDescriptor(clazz);
        String newDescriptor = simplifyDescriptor(descriptor, localVariableInfo.referencedClass);

        if (!descriptor.equals(newDescriptor))
        {
            // Update the descriptor.
            localVariableInfo.u2descriptorIndex =
                new ConstantPoolEditor((ProgramClass)clazz).addUtf8Constant(newDescriptor);

            // Clear the referenced class.
            localVariableInfo.referencedClass = null;
        }
    }


    // Implementations for LocalVariableTypeInfoVisitor.

    public void visitLocalVariableTypeInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeInfo localVariableTypeInfo)
    {
        // We're only looking at the base type for now.
        if (localVariableTypeInfo.referencedClasses != null &&
            localVariableTypeInfo.referencedClasses.length > 0)
        {
            // Update the signature if it has any simple enum classes.
            String signature    = localVariableTypeInfo.getSignature(clazz);
            String newSignature = simplifyDescriptor(signature,
                                                     localVariableTypeInfo.referencedClasses[0]);

            if (!signature.equals(newSignature))
            {
                // Update the signature.
                localVariableTypeInfo.u2signatureIndex =
                    new ConstantPoolEditor((ProgramClass)clazz).addUtf8Constant(newSignature);

                // Clear the referenced class.
                localVariableTypeInfo.referencedClasses[0] = null;
            }
        }
    }


    // Small utility methods.

    /**
     * Returns the descriptor with simplified enum type.
     */
    private String simplifyDescriptor(String descriptor,
                                      Clazz  referencedClass)
    {
        return isSimpleEnum(referencedClass) ?
                   descriptor.substring(0, ClassUtil.internalArrayTypeDimensionCount(descriptor)) + ClassConstants.TYPE_INT :
                   descriptor;
    }


    /**
     * Returns the descriptor with simplified enum types.
     */
    private String simplifyDescriptor(String  descriptor,
                                      Clazz[] referencedClasses)
    {
        if (referencedClasses != null)
        {
            if (DEBUG_EXTRA)
            {
                System.out.println("  Before: ["+descriptor+"]");
            }

            InternalTypeEnumeration typeEnumeration =
                new InternalTypeEnumeration(descriptor);

            int referencedClassIndex = 0;

            StringBuffer newDescriptorBuffer =
                new StringBuffer(descriptor.length());

            // Go over the formal type parameters.
            if (typeEnumeration.hasFormalTypeParameters())
            {
                // Consider the classes referenced by this formal type
                // parameter.
                String type = typeEnumeration.formalTypeParameters();

                DescriptorClassEnumeration classEnumeration =
                    new DescriptorClassEnumeration(type);

                newDescriptorBuffer.append(classEnumeration.nextFluff());

                // Replace any simple enum types.
                while (classEnumeration.hasMoreClassNames())
                {
                    // Get the class.
                    String className =
                        classEnumeration.nextClassName();

                    Clazz referencedClass =
                        referencedClasses[referencedClassIndex++];

                    // Is this class a simple enum type?
                    if (isSimpleEnum(referencedClass))
                    {
                        // Let's replace it by java.lang.Integer.
                        className = ClassConstants.NAME_JAVA_LANG_INTEGER;
                    }

                    newDescriptorBuffer.append(className);
                    newDescriptorBuffer.append(classEnumeration.nextFluff());
                }
            }

            if (typeEnumeration.isMethodSignature())
            {
                newDescriptorBuffer.append(ClassConstants.METHOD_ARGUMENTS_OPEN);
            }

            // Go over the main types (class types or parameter types).
            while (typeEnumeration.hasMoreTypes())
            {
                // Consider the classes referenced by this parameter type.
                String type = typeEnumeration.nextType();

                DescriptorClassEnumeration classEnumeration =
                    new DescriptorClassEnumeration(type);

                String firstFluff = classEnumeration.nextFluff();

                if (classEnumeration.hasMoreClassNames())
                {
                    // Get the first class.
                    String firstClassName =
                        classEnumeration.nextClassName();

                    Clazz firstReferencedClass =
                        referencedClasses[referencedClassIndex++];

                    // Is the first class a simple enum type?
                    if (isSimpleEnum(firstReferencedClass))
                    {
                        // Replace it by a primitive int, with any array
                        // prefix.
                        newDescriptorBuffer.append(type.substring(0, ClassUtil.internalArrayTypeDimensionCount(type)));
                        newDescriptorBuffer.append(ClassConstants.TYPE_INT);

                        // Skip any other classes of this type.
                        classEnumeration.nextFluff();
                        while (classEnumeration.hasMoreClassNames())
                        {
                            classEnumeration.nextClassName();
                            classEnumeration.nextFluff();

                            referencedClassIndex++;
                        }
                    }
                    else
                    {
                        newDescriptorBuffer.append(firstFluff);
                        newDescriptorBuffer.append(firstClassName);
                        newDescriptorBuffer.append(classEnumeration.nextFluff());

                        // Replace any other simple enum types.
                        while (classEnumeration.hasMoreClassNames())
                        {
                            // Get the class.
                            String className =
                                classEnumeration.nextClassName();

                            Clazz referencedClass =
                                referencedClasses[referencedClassIndex++];

                            // Is this class a simple enum type?
                            if (isSimpleEnum(referencedClass))
                            {
                                // Let's replace it by java.lang.Integer.
                                className = ClassConstants.NAME_JAVA_LANG_INTEGER;
                            }

                            newDescriptorBuffer.append(className);
                            newDescriptorBuffer.append(classEnumeration.nextFluff());
                        }
                    }
                }
                else
                {
                    newDescriptorBuffer.append(firstFluff);
                }
            }

            if (typeEnumeration.isMethodSignature())
            {
                newDescriptorBuffer.append(ClassConstants.METHOD_ARGUMENTS_CLOSE);

                // Consider the classes referenced by the return type.
                String type = typeEnumeration.returnType();

                DescriptorClassEnumeration classEnumeration =
                    new DescriptorClassEnumeration(type);

                String firstFluff = classEnumeration.nextFluff();

                if (classEnumeration.hasMoreClassNames())
                {
                    // Get the first class.
                    String firstClassName =
                        classEnumeration.nextClassName();

                    Clazz firstReferencedClass =
                        referencedClasses[referencedClassIndex++];

                    // Is the first class a simple enum type?
                    if (isSimpleEnum(firstReferencedClass))
                    {
                        // Replace it by a primitive int, with any array
                        // prefix.
                        newDescriptorBuffer.append(type.substring(0, ClassUtil.internalArrayTypeDimensionCount(type)));
                        newDescriptorBuffer.append(ClassConstants.TYPE_INT);
                    }
                    else
                    {
                        newDescriptorBuffer.append(firstFluff);
                        newDescriptorBuffer.append(firstClassName);
                        newDescriptorBuffer.append(classEnumeration.nextFluff());

                        // Replace any other simple enum types.
                        while (classEnumeration.hasMoreClassNames())
                        {
                            // Get the class.
                            String className =
                                classEnumeration.nextClassName();

                            Clazz referencedClass =
                                referencedClasses[referencedClassIndex++];

                            // Is this class a simple enum type?
                            if (isSimpleEnum(referencedClass))
                            {
                                // Let's replace it by java.lang.Integer.
                                className = ClassConstants.NAME_JAVA_LANG_INTEGER;
                            }

                            newDescriptorBuffer.append(className);
                            newDescriptorBuffer.append(classEnumeration.nextFluff());
                        }
                    }
                }
                else
                {
                    newDescriptorBuffer.append(firstFluff);
                }
            }

            descriptor = newDescriptorBuffer.toString();

            if (DEBUG_EXTRA)
            {
                System.out.println("  After:  ["+descriptor+"]");
            }
        }

        return descriptor;
    }


    /**
     * Returns the simplified and shrunk array of referenced classes for the
     * given descriptor.
     */
    private Clazz[] simplifyReferencedClasses(String  descriptor,
                                              Clazz[] referencedClasses)
    {
        if (referencedClasses != null)
        {
            if (DEBUG_EXTRA)
            {
                System.out.print("  Referenced before:");
                for (int index = 0; index < referencedClasses.length; index++)
                {
                    System.out.print(" ["+(referencedClasses[index] == null ? null : referencedClasses[index].getName())+"]");
                }
                System.out.println();
            }

            InternalTypeEnumeration typeEnumeration =
                new InternalTypeEnumeration(descriptor);

            int referencedClassIndex    = 0;
            int newReferencedClassIndex = 0;

            // Go over the formal type parameters.
            if (typeEnumeration.hasFormalTypeParameters())
            {
                // Consider the classes referenced by this formal type
                // parameter.
                String type = typeEnumeration.formalTypeParameters();

                DescriptorClassEnumeration classEnumeration =
                    new DescriptorClassEnumeration(type);

                classEnumeration.nextFluff();

                // Replace any simple enum types.
                while (classEnumeration.hasMoreClassNames())
                {
                    // Get the class.
                    classEnumeration.nextClassName();
                    classEnumeration.nextFluff();

                    Clazz referencedClass =
                        referencedClasses[referencedClassIndex++];

                    // Clear the referenced class if it is a simple
                    // enum type (now java.lang.Integer).
                    referencedClasses[newReferencedClassIndex++] =
                        isSimpleEnum(referencedClass) ? null : referencedClass;
                }
            }

            // Go over the main types (class types or parameter types).
            while (typeEnumeration.hasMoreTypes())
            {
                // Consider the classes referenced by this parameter type.
                String type = typeEnumeration.nextType();

                DescriptorClassEnumeration classEnumeration =
                    new DescriptorClassEnumeration(type);

                classEnumeration.nextFluff();

                if (classEnumeration.hasMoreClassNames())
                {
                    // Get the first class.
                    classEnumeration.nextClassName();
                    classEnumeration.nextFluff();

                    Clazz firstReferencedClass =
                        referencedClasses[referencedClassIndex++];

                    // Is the first class a simple enum type?
                    if (isSimpleEnum(firstReferencedClass))
                    {
                        // Replace it by a primitive int.

                        // Skip any other classes of this type.
                        classEnumeration.nextFluff();
                        while (classEnumeration.hasMoreClassNames())
                        {
                            classEnumeration.nextClassName();
                            classEnumeration.nextFluff();

                            referencedClassIndex++;
                        }
                    }
                    else
                    {
                        referencedClasses[newReferencedClassIndex++] =
                            firstReferencedClass;

                        // Replace any other simple enum types.
                        while (classEnumeration.hasMoreClassNames())
                        {
                            // Get the class.
                            classEnumeration.nextClassName();
                            classEnumeration.nextFluff();

                            Clazz referencedClass =
                                referencedClasses[referencedClassIndex++];

                            // Clear the referenced class if it is a simple
                            // enum type (now java.lang.Integer).
                            referencedClasses[newReferencedClassIndex++] =
                                isSimpleEnum(referencedClass) ? null : referencedClass;
                        }
                    }
                }
            }

            if (typeEnumeration.isMethodSignature())
            {
                // Consider the classes referenced by the return type.
                String type = typeEnumeration.returnType();

                DescriptorClassEnumeration classEnumeration =
                    new DescriptorClassEnumeration(type);

                classEnumeration.nextFluff();

                if (classEnumeration.hasMoreClassNames())
                {
                    // Get the first class.
                    classEnumeration.nextClassName();
                    classEnumeration.nextFluff();

                    Clazz firstReferencedClass =
                        referencedClasses[referencedClassIndex++];

                    // Is the first class a simple enum type?
                    if (isSimpleEnum(firstReferencedClass))
                    {
                        // Replace it by a primitive int.
                        // Clear all remaining referenced classes.
                    }
                    else
                    {
                        referencedClasses[newReferencedClassIndex++] =
                            firstReferencedClass;

                        // Replace any other simple enum types.
                        while (classEnumeration.hasMoreClassNames())
                        {
                            // Get the class.
                            classEnumeration.nextClassName();
                            classEnumeration.nextFluff();

                            Clazz referencedClass =
                                referencedClasses[referencedClassIndex++];

                            // Clear the referenced class if it is a simple
                            // enum type (now java.lang.Integer).
                            referencedClasses[newReferencedClassIndex++] =
                                isSimpleEnum(referencedClass) ? null : referencedClass;
                        }
                    }
                }
            }

            // Shrink the array to the proper size.
            if (newReferencedClassIndex == 0)
            {
                referencedClasses = null;
            }
            else if (newReferencedClassIndex < referencedClassIndex)
            {
                Clazz[] newReferencedClasses = new Clazz[newReferencedClassIndex];
                System.arraycopy(referencedClasses, 0,
                                 newReferencedClasses, 0,
                                 newReferencedClassIndex);

                referencedClasses = newReferencedClasses;

                if (DEBUG_EXTRA)
                {
                    System.out.print("  Referenced after: ");
                    for (int index = 0; index < referencedClasses.length; index++)
                    {
                        System.out.print(" ["+(referencedClasses[index] == null ? null : referencedClasses[index].getName())+"]");
                    }
                    System.out.println();
                }
            }
        }

        return referencedClasses;
    }


    /**
     * Returns whether the given class is not null and a simple enum class.
     */
    private boolean isSimpleEnum(Clazz clazz)
    {
        return clazz != null &&
               SimpleEnumMarker.isSimpleEnum(clazz);
    }
}

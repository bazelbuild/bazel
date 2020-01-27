/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2017 Eric Lafortune @ GuardSquare
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
package proguard.classfile.editor;

import proguard.classfile.*;
import proguard.classfile.constant.*;

/**
 * This class can add constant pool entries to a given class.
 *
 * @author Eric Lafortune
 */
public class ConstantPoolEditor
{
    private static final boolean DEBUG = false;

    private ProgramClass targetClass;


    /**
     * Creates a new ConstantPoolEditor that will edit constants in the given
     * target class.
     */
    public ConstantPoolEditor(ProgramClass targetClass)
    {
        this.targetClass = targetClass;
    }


    /**
     * Finds or creates a IntegerConstant constant pool entry with the given
     * value.
     * @return the constant pool index of the Utf8Constant.
     */
    public int addIntegerConstant(int value)
    {
        int        constantPoolCount = targetClass.u2constantPoolCount;
        Constant[] constantPool      = targetClass.constantPool;

        // Check if the entry already exists.
        for (int index = 1; index < constantPoolCount; index++)
        {
            Constant constant = constantPool[index];

            if (constant != null &&
                constant.getTag() == ClassConstants.CONSTANT_Integer)
            {
                IntegerConstant integerConstant = (IntegerConstant)constant;
                if (integerConstant.getValue() == value)
                {
                    return index;
                }
            }
        }

        return addConstant(new IntegerConstant(value));
    }


    /**
     * Finds or creates a LongConstant constant pool entry with the given value.
     * @return the constant pool index of the LongConstant.
     */
    public int addLongConstant(long value)
    {
        int        constantPoolCount = targetClass.u2constantPoolCount;
        Constant[] constantPool      = targetClass.constantPool;

        // Check if the entry already exists.
        for (int index = 1; index < constantPoolCount; index++)
        {
            Constant constant = constantPool[index];

            if (constant != null &&
                constant.getTag() == ClassConstants.CONSTANT_Long)
            {
                LongConstant longConstant = (LongConstant)constant;
                if (longConstant.getValue() == value)
                {
                    return index;
                }
            }
        }

        return addConstant(new LongConstant(value));
    }


    /**
     * Finds or creates a FloatConstant constant pool entry with the given
     * value.
     * @return the constant pool index of the FloatConstant.
     */
    public int addFloatConstant(float value)
    {
        int        constantPoolCount = targetClass.u2constantPoolCount;
        Constant[] constantPool      = targetClass.constantPool;

        // Check if the entry already exists.
        for (int index = 1; index < constantPoolCount; index++)
        {
            Constant constant = constantPool[index];

            if (constant != null &&
                constant.getTag() == ClassConstants.CONSTANT_Float)
            {
                FloatConstant floatConstant = (FloatConstant)constant;
                if (floatConstant.getValue() == value)
                {
                    return index;
                }
            }
        }

        return addConstant(new FloatConstant(value));
    }


    /**
     * Finds or creates a DoubleConstant constant pool entry with the given
     * value.
     * @return the constant pool index of the DoubleConstant.
     */
    public int addDoubleConstant(double value)
    {
        int        constantPoolCount = targetClass.u2constantPoolCount;
        Constant[] constantPool      = targetClass.constantPool;

        // Check if the entry already exists.
        for (int index = 1; index < constantPoolCount; index++)
        {
            Constant constant = constantPool[index];

            if (constant != null &&
                constant.getTag() == ClassConstants.CONSTANT_Double)
            {
                DoubleConstant doubleConstant = (DoubleConstant)constant;
                if (doubleConstant.getValue() == value)
                {
                    return index;
                }
            }
        }

        return addConstant(new DoubleConstant(value));
    }


    /**
     * Finds or creates a StringConstant constant pool entry with the given
     * value.
     * @return the constant pool index of the StringConstant.
     */
    public int addStringConstant(String string,
                                 Clazz  referencedClass,
                                 Member referencedMember)
    {
        int        constantPoolCount = targetClass.u2constantPoolCount;
        Constant[] constantPool      = targetClass.constantPool;

        // Check if the entry already exists.
        for (int index = 1; index < constantPoolCount; index++)
        {
            Constant constant = constantPool[index];

            if (constant != null &&
                constant.getTag() == ClassConstants.CONSTANT_String)
            {
                StringConstant stringConstant = (StringConstant)constant;
                if (stringConstant.getString(targetClass).equals(string))
                {
                    return index;
                }
            }
        }

        return addConstant(new StringConstant(addUtf8Constant(string),
                                              referencedClass,
                                              referencedMember));
    }


    /**
     * Finds or creates a InvokeDynamicConstant constant pool entry with the
     * given bootstrap method constant pool entry index, method name, and
     * descriptor.
     * @return the constant pool index of the InvokeDynamicConstant.
     */
    public int addInvokeDynamicConstant(int     bootstrapMethodIndex,
                                        String  name,
                                        String  descriptor,
                                        Clazz[] referencedClasses)
    {
        return addInvokeDynamicConstant(bootstrapMethodIndex,
                                        addNameAndTypeConstant(name, descriptor),
                                        referencedClasses);
    }


    /**
     * Finds or creates a InvokeDynamicConstant constant pool entry with the given
     * class constant pool entry index and name and type constant pool entry
     * index.
     * @return the constant pool index of the InvokeDynamicConstant.
     */
    public int addInvokeDynamicConstant(int     bootstrapMethodIndex,
                                        int     nameAndTypeIndex,
                                        Clazz[] referencedClasses)
    {
        int        constantPoolCount = targetClass.u2constantPoolCount;
        Constant[] constantPool      = targetClass.constantPool;

        // Check if the entry already exists.
        for (int index = 1; index < constantPoolCount; index++)
        {
            Constant constant = constantPool[index];

            if (constant != null &&
                constant.getTag() == ClassConstants.CONSTANT_InvokeDynamic)
            {
                InvokeDynamicConstant invokeDynamicConstant = (InvokeDynamicConstant)constant;
                if (invokeDynamicConstant.u2bootstrapMethodAttributeIndex == bootstrapMethodIndex &&
                    invokeDynamicConstant.u2nameAndTypeIndex     == nameAndTypeIndex)
                {
                    return index;
                }
            }
        }

        return addConstant(new InvokeDynamicConstant(bootstrapMethodIndex,
                                                     nameAndTypeIndex,
                                                     referencedClasses));
    }


    /**
     * Finds or creates a MethodHandleConstant constant pool entry of the
     * specified kind and with the given field ref, interface method ref,
     * or method ref constant pool entry index.
     * @return the constant pool index of the MethodHandleConstant.
     */
    public int addMethodHandleConstant(int referenceKind,
                                       int referenceIndex)
    {
        int        constantPoolCount = targetClass.u2constantPoolCount;
        Constant[] constantPool      = targetClass.constantPool;

        // Check if the entry already exists.
        for (int index = 1; index < constantPoolCount; index++)
        {
            Constant constant = constantPool[index];

            if (constant != null &&
                constant.getTag() == ClassConstants.CONSTANT_MethodHandle)
            {
                MethodHandleConstant methodHandleConstant = (MethodHandleConstant)constant;
                if (methodHandleConstant.u1referenceKind  == referenceKind &&
                    methodHandleConstant.u2referenceIndex == referenceIndex)
                {
                    return index;
                }
            }
        }

        return addConstant(new MethodHandleConstant(referenceKind,
                                                    referenceIndex));
    }


    /**
     * Finds or creates a FieldrefConstant constant pool entry for the given
     * class and field.
     * @return the constant pool index of the FieldrefConstant.
     */
    public int addFieldrefConstant(Clazz  referencedClass,
                                   Member referencedMember)
    {
        return addFieldrefConstant(referencedClass.getName(),
                                   referencedMember.getName(referencedClass),
                                   referencedMember.getDescriptor(referencedClass),
                                   referencedClass,
                                   referencedMember);
    }


    /**
     * Finds or creates a FieldrefConstant constant pool entry with the given
     * class name, field name, and descriptor.
     * @return the constant pool index of the FieldrefConstant.
     */
    public int addFieldrefConstant(String className,
                                   String name,
                                   String descriptor,
                                   Clazz  referencedClass,
                                   Member referencedMember)
    {
        return addFieldrefConstant(className,
                                   addNameAndTypeConstant(name, descriptor),
                                   referencedClass,
                                   referencedMember);
    }


    /**
     * Finds or creates a FieldrefConstant constant pool entry with the given
     * class name, field name, and descriptor.
     * @return the constant pool index of the FieldrefConstant.
     */
    public int addFieldrefConstant(String className,
                                   int    nameAndTypeIndex,
                                   Clazz  referencedClass,
                                   Member referencedMember)
    {
        return addFieldrefConstant(addClassConstant(className, referencedClass),
                                   nameAndTypeIndex,
                                   referencedClass,
                                   referencedMember);
    }


    /**
     * Finds or creates a FieldrefConstant constant pool entry with the given
     * class constant pool entry index, field name, and descriptor.
     * @return the constant pool index of the FieldrefConstant.
     */
    public int addFieldrefConstant(int    classIndex,
                                   String name,
                                   String descriptor,
                                   Clazz  referencedClass,
                                   Member referencedMember)
    {
        return addFieldrefConstant(classIndex,
                                   addNameAndTypeConstant(name, descriptor),
                                   referencedClass,
                                   referencedMember);
    }


    /**
     * Finds or creates a FieldrefConstant constant pool entry with the given
     * class constant pool entry index and name and type constant pool entry
     * index.
     * @return the constant pool index of the FieldrefConstant.
     */
    public int addFieldrefConstant(int    classIndex,
                                   int    nameAndTypeIndex,
                                   Clazz  referencedClass,
                                   Member referencedMember)
    {
        int        constantPoolCount = targetClass.u2constantPoolCount;
        Constant[] constantPool      = targetClass.constantPool;

        // Check if the entry already exists.
        for (int index = 1; index < constantPoolCount; index++)
        {
            Constant constant = constantPool[index];

            if (constant != null &&
                constant.getTag() == ClassConstants.CONSTANT_Fieldref)
            {
                FieldrefConstant fieldrefConstant = (FieldrefConstant)constant;
                if (fieldrefConstant.u2classIndex         == classIndex &&
                    fieldrefConstant.u2nameAndTypeIndex   == nameAndTypeIndex)
                {
                    return index;
                }
            }
        }

        return addConstant(new FieldrefConstant(classIndex,
                                                nameAndTypeIndex,
                                                referencedClass,
                                                referencedMember));
    }


    /**
     * Finds or creates a InterfaceMethodrefConstant constant pool entry with the
     * given class name, method name, and descriptor.
     * @return the constant pool index of the InterfaceMethodrefConstant.
     */
    public int addInterfaceMethodrefConstant(String className,
                                             String name,
                                             String descriptor,
                                             Clazz  referencedClass,
                                             Member referencedMember)
    {
        return addInterfaceMethodrefConstant(className,
                                             addNameAndTypeConstant(name, descriptor),
                                             referencedClass,
                                             referencedMember);
    }


    /**
     * Finds or creates a InterfaceMethodrefConstant constant pool entry with the
     * given class name, method name, and descriptor.
     * @return the constant pool index of the InterfaceMethodrefConstant.
     */
    public int addInterfaceMethodrefConstant(String className,
                                             int    nameAndTypeIndex,
                                             Clazz  referencedClass,
                                             Member referencedMember)
    {
        return addInterfaceMethodrefConstant(addClassConstant(className, referencedClass),
                                             nameAndTypeIndex,
                                             referencedClass,
                                             referencedMember);
    }


    /**
     * Finds or creates a InterfaceMethodrefConstant constant pool entry for the
     * given class and method.
     * @return the constant pool index of the InterfaceMethodrefConstant.
     */
    public int addInterfaceMethodrefConstant(Clazz  referencedClass,
                                             Member referencedMember)
    {
        return addInterfaceMethodrefConstant(referencedClass.getName(),
                                             referencedMember.getName(referencedClass),
                                             referencedMember.getDescriptor(referencedClass),
                                             referencedClass,
                                             referencedMember);
    }


    /**
     * Finds or creates a InterfaceMethodrefConstant constant pool entry with the
     * given class constant pool entry index, method name, and descriptor.
     * @return the constant pool index of the InterfaceMethodrefConstant.
     */
    public int addInterfaceMethodrefConstant(int    classIndex,
                                             String name,
                                             String descriptor,
                                             Clazz  referencedClass,
                                             Member referencedMember)
    {
        return addInterfaceMethodrefConstant(classIndex,
                                             addNameAndTypeConstant(name, descriptor),
                                             referencedClass,
                                             referencedMember);
    }


    /**
     * Finds or creates a InterfaceMethodrefConstant constant pool entry with the
     * given class constant pool entry index and name and type constant pool
     * entry index.
     * @return the constant pool index of the InterfaceMethodrefConstant.
     */
    public int addInterfaceMethodrefConstant(int    classIndex,
                                             int    nameAndTypeIndex,
                                             Clazz  referencedClass,
                                             Member referencedMember)
    {
        int        constantPoolCount = targetClass.u2constantPoolCount;
        Constant[] constantPool      = targetClass.constantPool;

        // Check if the entry already exists.
        for (int index = 1; index < constantPoolCount; index++)
        {
            Constant constant = constantPool[index];

            if (constant != null &&
                            constant.getTag() == ClassConstants.CONSTANT_InterfaceMethodref)
            {
                InterfaceMethodrefConstant methodrefConstant = (InterfaceMethodrefConstant)constant;
                if (methodrefConstant.u2classIndex       == classIndex &&
                    methodrefConstant.u2nameAndTypeIndex == nameAndTypeIndex)
                {
                    return index;
                }
            }
        }

        return addConstant(new InterfaceMethodrefConstant(classIndex,
                                                          nameAndTypeIndex,
                                                          referencedClass,
                                                          referencedMember));
    }


    /**
     * Finds or creates a MethodrefConstant constant pool entry for the given
     * class and method.
     * @return the constant pool index of the MethodrefConstant.
     */
    public int addMethodrefConstant(Clazz  referencedClass,
                                    Member referencedMember)
    {
        return addMethodrefConstant(referencedClass.getName(),
                                    referencedMember.getName(referencedClass),
                                    referencedMember.getDescriptor(referencedClass),
                                    referencedClass,
                                    referencedMember);
    }


    /**
     * Finds or creates a MethodrefConstant constant pool entry with the given
     * class name, method name, and descriptor.
     * @return the constant pool index of the MethodrefConstant.
     */
    public int addMethodrefConstant(String className,
                                    String name,
                                    String descriptor,
                                    Clazz  referencedClass,
                                    Member referencedMember)
    {
        return addMethodrefConstant(className,
                                    addNameAndTypeConstant(name, descriptor),
                                    referencedClass,
                                    referencedMember);
    }


    /**
     * Finds or creates a MethodrefConstant constant pool entry with the given
     * class name, method name, and descriptor.
     * @return the constant pool index of the MethodrefConstant.
     */
    public int addMethodrefConstant(String className,
                                    int    nameAndTypeIndex,
                                    Clazz  referencedClass,
                                    Member referencedMember)
    {
        return addMethodrefConstant(addClassConstant(className, referencedClass),
                                    nameAndTypeIndex,
                                    referencedClass,
                                    referencedMember);
    }


    /**
     * Finds or creates a MethodrefConstant constant pool entry with the given
     * class constant pool entry index, method name, and descriptor.
     * @return the constant pool index of the MethodrefConstant.
     */
    public int addMethodrefConstant(int    classIndex,
                                    String name,
                                    String descriptor,
                                    Clazz  referencedClass,
                                    Member referencedMember)
    {
        return addMethodrefConstant(classIndex,
                                    addNameAndTypeConstant(name, descriptor),
                                    referencedClass,
                                    referencedMember);
    }


    /**
     * Finds or creates a MethodrefConstant constant pool entry with the given
     * class constant pool entry index and name and type constant pool entry
     * index.
     * @return the constant pool index of the MethodrefConstant.
     */
    public int addMethodrefConstant(int    classIndex,
                                    int    nameAndTypeIndex,
                                    Clazz  referencedClass,
                                    Member referencedMember)
    {
        int        constantPoolCount = targetClass.u2constantPoolCount;
        Constant[] constantPool      = targetClass.constantPool;

        // Check if the entry already exists.
        for (int index = 1; index < constantPoolCount; index++)
        {
            Constant constant = constantPool[index];

            if (constant != null &&
                constant.getTag() == ClassConstants.CONSTANT_Methodref)
            {
                MethodrefConstant methodrefConstant = (MethodrefConstant)constant;
                if (methodrefConstant.u2classIndex       == classIndex &&
                    methodrefConstant.u2nameAndTypeIndex == nameAndTypeIndex)
                {
                    return index;
                }
            }
        }

        return addConstant(new MethodrefConstant(classIndex,
                                                 nameAndTypeIndex,
                                                 referencedClass,
                                                 referencedMember));
    }


    /**
     * Finds or creates a ClassConstant constant pool entry for the given class.
     * @return the constant pool index of the ClassConstant.
     */
    public int addClassConstant(Clazz referencedClass)
    {
        return addClassConstant(referencedClass.getName(),
                                referencedClass);
    }


    /**
     * Finds or creates a ClassConstant constant pool entry with the given name.
     * @return the constant pool index of the ClassConstant.
     */
    public int addClassConstant(String name,
                                Clazz  referencedClass)
    {
        int        constantPoolCount = targetClass.u2constantPoolCount;
        Constant[] constantPool      = targetClass.constantPool;

        // Check if the entry already exists.
        for (int index = 1; index < constantPoolCount; index++)
        {
            Constant constant = constantPool[index];

            if (constant != null &&
                constant.getTag() == ClassConstants.CONSTANT_Class)
            {
                ClassConstant classConstant = (ClassConstant)constant;
                if (classConstant.getName(targetClass).equals(name))
                {
                    return index;
                }
            }
        }

        int nameIndex = addUtf8Constant(name);

        return addConstant(new ClassConstant(nameIndex, referencedClass));
    }


    /**
     * Finds or creates a MethodTypeConstant constant pool entry with the given
     * type.
     * @return the constant pool index of the MethodTypeConstant.
     */
    public int addMethodTypeConstant(String  type,
                                     Clazz[] referencedClasses)
    {
        int        constantPoolCount = targetClass.u2constantPoolCount;
        Constant[] constantPool      = targetClass.constantPool;

        // Check if the entry already exists.
        for (int index = 1; index < constantPoolCount; index++)
        {
            Constant constant = constantPool[index];

            if (constant != null &&
                constant.getTag() == ClassConstants.CONSTANT_MethodType)
            {
                MethodTypeConstant methodTypeConstant = (MethodTypeConstant)constant;
                if (methodTypeConstant.getType(targetClass).equals(type))
                {
                    return index;
                }
            }
        }

        return addConstant(new MethodTypeConstant(addUtf8Constant(type),
                                                  referencedClasses));
    }


    /**
     * Finds or creates a NameAndTypeConstant constant pool entry with the given
     * name and type.
     * @return the constant pool index of the NameAndTypeConstant.
     */
    public int addNameAndTypeConstant(String name,
                                      String type)
    {
        int        constantPoolCount = targetClass.u2constantPoolCount;
        Constant[] constantPool      = targetClass.constantPool;

        // Check if the entry already exists.
        for (int index = 1; index < constantPoolCount; index++)
        {
            Constant constant = constantPool[index];

            if (constant != null &&
                constant.getTag() == ClassConstants.CONSTANT_NameAndType)
            {
                NameAndTypeConstant nameAndTypeConstant = (NameAndTypeConstant)constant;
                if (nameAndTypeConstant.getName(targetClass).equals(name) &&
                    nameAndTypeConstant.getType(targetClass).equals(type))
                {
                    return index;
                }
            }
        }

        return addConstant(new NameAndTypeConstant(addUtf8Constant(name),
                                                   addUtf8Constant(type)));
    }


    /**
     * Finds or creates a Utf8Constant constant pool entry for the given string.
     * @return the constant pool index of the Utf8Constant.
     */
    public int addUtf8Constant(String string)
    {
        int        constantPoolCount = targetClass.u2constantPoolCount;
        Constant[] constantPool      = targetClass.constantPool;

        // Check if the entry already exists.
        for (int index = 1; index < constantPoolCount; index++)
        {
            Constant constant = constantPool[index];

            if (constant != null &&
                constant.getTag() == ClassConstants.CONSTANT_Utf8)
            {
                Utf8Constant utf8Constant = (Utf8Constant)constant;
                if (utf8Constant.getString().equals(string))
                {
                    return index;
                }
            }
        }

        return addConstant(new Utf8Constant(string));
    }


    /**
     * Adds a given constant pool entry to the end of the constant pool/
     * @return the constant pool index for the added entry.
     */
    public int addConstant(Constant constant)
    {
        int        constantPoolCount = targetClass.u2constantPoolCount;
        Constant[] constantPool      = targetClass.constantPool;

        // Make sure there is enough space for another constant pool entry.
        if (constantPool.length < constantPoolCount+2)
        {
            targetClass.constantPool = new Constant[constantPoolCount+2];
            System.arraycopy(constantPool, 0,
                             targetClass.constantPool, 0,
                             constantPoolCount);
            constantPool = targetClass.constantPool;
        }

        if (DEBUG)
        {
            System.out.println(targetClass.getName()+": adding ["+constant.getClass().getName()+"] at index "+targetClass.u2constantPoolCount);
        }

        // Create a new Utf8Constant for the given string.
        constantPool[targetClass.u2constantPoolCount++] = constant;

        // Long constants and double constants take up two entries in the
        // constant pool.
        int tag = constant.getTag();
        if (tag == ClassConstants.CONSTANT_Long ||
            tag == ClassConstants.CONSTANT_Double)
        {
            constantPool[targetClass.u2constantPoolCount++] = null;
        }

        return constantPoolCount;
    }
}

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

import java.util.List;

/**
 * Utility methods for converting between internal and external representations
 * of names and descriptions.
 *
 * @author Eric Lafortune
 */
public class ClassUtil
{
    private static final String EMPTY_STRING = "";


    /**
     * Checks whether the given class magic number is correct.
     * @param magicNumber the magic number.
     * @throws UnsupportedOperationException when the magic number is incorrect.
     */
    public static void checkMagicNumber(int magicNumber) throws UnsupportedOperationException
    {
        if (magicNumber != ClassConstants.MAGIC)
        {
            throw new UnsupportedOperationException("Invalid magic number ["+Integer.toHexString(magicNumber)+"] in class");
        }
    }


    /**
     * Returns the combined class version number.
     * @param majorVersion the major part of the class version number.
     * @param minorVersion the minor part of the class version number.
     * @return the combined class version number.
     */
    public static int internalClassVersion(int majorVersion, int minorVersion)
    {
        return (majorVersion << 16) | minorVersion;
    }


    /**
     * Returns the major part of the given class version number.
     * @param internalClassVersion the combined class version number.
     * @return the major part of the class version number.
     */
    public static int internalMajorClassVersion(int internalClassVersion)
    {
        return internalClassVersion >>> 16;
    }


    /**
     * Returns the internal class version number.
     * @param internalClassVersion the external class version number.
     * @return the internal class version number.
     */
    public static int internalMinorClassVersion(int internalClassVersion)
    {
        return internalClassVersion & 0xffff;
    }


    /**
     * Returns the internal class version number.
     * @param externalClassVersion the external class version number.
     * @return the internal class version number.
     */
    public static int internalClassVersion(String externalClassVersion)
    {
        return
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_1_0) ||
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_1_1) ? ClassConstants.CLASS_VERSION_1_0 :
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_1_2) ? ClassConstants.CLASS_VERSION_1_2 :
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_1_3) ? ClassConstants.CLASS_VERSION_1_3 :
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_1_4) ? ClassConstants.CLASS_VERSION_1_4 :
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_1_5_ALIAS) ||
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_1_5) ? ClassConstants.CLASS_VERSION_1_5 :
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_1_6_ALIAS) ||
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_1_6) ? ClassConstants.CLASS_VERSION_1_6 :
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_1_7_ALIAS) ||
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_1_7) ? ClassConstants.CLASS_VERSION_1_7 :
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_1_8_ALIAS) ||
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_1_8) ? ClassConstants.CLASS_VERSION_1_8 :
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_1_9_ALIAS) ||
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_1_9) ? ClassConstants.CLASS_VERSION_1_9 :
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_10)  ? ClassConstants.CLASS_VERSION_10  :
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_11)  ? ClassConstants.CLASS_VERSION_11  :
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_12)  ? ClassConstants.CLASS_VERSION_12  :
            externalClassVersion.equals(JavaConstants.CLASS_VERSION_13)  ? ClassConstants.CLASS_VERSION_13  :
                                                                           0;
    }


    /**
     * Returns the minor part of the given class version number.
     * @param internalClassVersion the combined class version number.
     * @return the minor part of the class version number.
     */
    public static String externalClassVersion(int internalClassVersion)
    {
        switch (internalClassVersion)
        {
            case ClassConstants.CLASS_VERSION_1_0: return JavaConstants.CLASS_VERSION_1_0;
            case ClassConstants.CLASS_VERSION_1_2: return JavaConstants.CLASS_VERSION_1_2;
            case ClassConstants.CLASS_VERSION_1_3: return JavaConstants.CLASS_VERSION_1_3;
            case ClassConstants.CLASS_VERSION_1_4: return JavaConstants.CLASS_VERSION_1_4;
            case ClassConstants.CLASS_VERSION_1_5: return JavaConstants.CLASS_VERSION_1_5;
            case ClassConstants.CLASS_VERSION_1_6: return JavaConstants.CLASS_VERSION_1_6;
            case ClassConstants.CLASS_VERSION_1_7: return JavaConstants.CLASS_VERSION_1_7;
            case ClassConstants.CLASS_VERSION_1_8: return JavaConstants.CLASS_VERSION_1_8;
            case ClassConstants.CLASS_VERSION_1_9: return JavaConstants.CLASS_VERSION_1_9;
            case ClassConstants.CLASS_VERSION_10:  return JavaConstants.CLASS_VERSION_10;
            case ClassConstants.CLASS_VERSION_11:  return JavaConstants.CLASS_VERSION_11;
            case ClassConstants.CLASS_VERSION_12:  return JavaConstants.CLASS_VERSION_12;
            case ClassConstants.CLASS_VERSION_13:  return JavaConstants.CLASS_VERSION_13;
            default:                               return null;
        }
    }


    /**
     * Checks whether the given class version number is supported.
     * @param internalClassVersion the combined class version number.
     * @throws UnsupportedOperationException when the version is not supported.
     */
    public static void checkVersionNumbers(int internalClassVersion) throws UnsupportedOperationException
    {
        if (internalClassVersion < ClassConstants.CLASS_VERSION_1_0 ||
            internalClassVersion > ClassConstants.CLASS_VERSION_13)
        {
            throw new UnsupportedOperationException("Unsupported version number ["+
                                                    internalMajorClassVersion(internalClassVersion)+"."+
                                                    internalMinorClassVersion(internalClassVersion)+"] (maximum "+
                                                    ClassConstants.CLASS_VERSION_13_MAJOR+"."+
                                                    ClassConstants.CLASS_VERSION_13_MINOR+", Java "+
                                                    JavaConstants.CLASS_VERSION_13+")");
        }
    }


    /**
     * Converts an external class name into an internal class name.
     * @param externalClassName the external class name,
     *                          e.g. "<code>java.lang.Object</code>"
     * @return the internal class name,
     *                          e.g. "<code>java/lang/Object</code>".
     */
    public static String internalClassName(String externalClassName)
    {
        return externalClassName.replace(JavaConstants.PACKAGE_SEPARATOR,
                                         ClassConstants.PACKAGE_SEPARATOR);
    }


    /**
     * Converts an internal class description into an external class description.
     * @param accessFlags       the access flags of the class.
     * @param internalClassName the internal class name,
     *                          e.g. "<code>java/lang/Object</code>".
     * @return the external class description,
     *                          e.g. "<code>public java.lang.Object</code>".
     */
    public static String externalFullClassDescription(int    accessFlags,
                                                      String internalClassName)
    {
        return externalClassAccessFlags(accessFlags) +
               externalClassName(internalClassName);
    }


    /**
     * Converts an internal class name into an external class name.
     * @param internalClassName the internal class name,
     *                          e.g. "<code>java/lang/Object</code>".
     * @return the external class name,
     *                          e.g. "<code>java.lang.Object</code>".
     */
    public static String externalClassName(String internalClassName)
    {
        return //internalClassName.startsWith(ClassConstants.PACKAGE_JAVA_LANG) &&
               //internalClassName.indexOf(ClassConstants.PACKAGE_SEPARATOR, ClassConstants.PACKAGE_JAVA_LANG.length() + 1) < 0 ?
               //internalClassName.substring(ClassConstants.PACKAGE_JAVA_LANG.length()) :
               internalClassName.replace(ClassConstants.PACKAGE_SEPARATOR,
                                         JavaConstants.PACKAGE_SEPARATOR);
    }


    /**
     * Returns the external base type of an external array type, dropping any
     * array brackets.
     * @param externalArrayType the external array type,
     *                          e.g. "<code>java.lang.Object[][]</code>"
     * @return the external base type,
     *                          e.g. "<code>java.lang.Object</code>".
     */
    public static String externalBaseType(String externalArrayType)
    {
        int index = externalArrayType.indexOf(JavaConstants.TYPE_ARRAY);
        return index >= 0 ?
            externalArrayType.substring(0, index) :
            externalArrayType;
    }


    /**
     * Returns the external short class name of an external class name, dropping
     * the package specification.
     * @param externalClassName the external class name,
     *                          e.g. "<code>java.lang.Object</code>"
     * @return the external short class name,
     *                          e.g. "<code>Object</code>".
     */
    public static String externalShortClassName(String externalClassName)
    {
        int index = externalClassName.lastIndexOf(JavaConstants.PACKAGE_SEPARATOR);
        return externalClassName.substring(index+1);
    }


    /**
     * Returns whether the given internal type is an array type.
     * @param internalType the internal type,
     *                     e.g. "<code>[[Ljava/lang/Object;</code>".
     * @return <code>true</code> if the given type is an array type,
     *         <code>false</code> otherwise.
     */
    public static boolean isInternalArrayType(String internalType)
    {
        return internalType.length() > 1 &&
               internalType.charAt(0) == ClassConstants.TYPE_ARRAY;
    }


    /**
     * Returns the number of dimensions of the given internal type.
     * @param internalType the internal type,
     *                     e.g. "<code>[[Ljava/lang/Object;</code>".
     * @return the number of dimensions, e.g. 2.
     */
    public static int internalArrayTypeDimensionCount(String internalType)
    {
        int dimensions = 0;
        while (internalType.charAt(dimensions) == ClassConstants.TYPE_ARRAY)
        {
            dimensions++;
        }

        return dimensions;
    }


    /**
     * Returns whether the given internal class name is one of the interfaces
     * that is implemented by all array types. These class names are
     * "<code>java/lang/Object</code>", "<code>java/lang/Cloneable</code>", and
     * "<code>java/io/Serializable</code>"
     * @param internalClassName the internal class name,
     *                          e.g. "<code>java/lang/Object</code>".
     * @return <code>true</code> if the given type is an array interface name,
     *         <code>false</code> otherwise.
     */
    public static boolean isInternalArrayInterfaceName(String internalClassName)
    {
        return ClassConstants.NAME_JAVA_LANG_OBJECT.equals(internalClassName)    ||
               ClassConstants.NAME_JAVA_LANG_CLONEABLE.equals(internalClassName) ||
               ClassConstants.NAME_JAVA_IO_SERIALIZABLE.equals(internalClassName);
    }


    /**
     * Returns whether the given internal type is a plain primitive type
     * (not void).
     * @param internalType the internal type,
     *                     e.g. "<code>I</code>".
     * @return <code>true</code> if the given type is a class type,
     *         <code>false</code> otherwise.
     */
    public static boolean isInternalPrimitiveType(char internalType)
    {
        return internalType == ClassConstants.TYPE_BOOLEAN ||
               internalType == ClassConstants.TYPE_BYTE    ||
               internalType == ClassConstants.TYPE_CHAR    ||
               internalType == ClassConstants.TYPE_SHORT   ||
               internalType == ClassConstants.TYPE_INT     ||
               internalType == ClassConstants.TYPE_FLOAT   ||
               internalType == ClassConstants.TYPE_LONG    ||
               internalType == ClassConstants.TYPE_DOUBLE;
    }


    /**
     * Returns whether the given internal type is a plain primitive type
     * (not void).
     * @param internalType the internal type,
     *                     e.g. "<code>I</code>".
     * @return <code>true</code> if the given type is a class type,
     *         <code>false</code> otherwise.
     */
    public static boolean isInternalPrimitiveType(String internalType)
    {
        return  isInternalPrimitiveType(internalType.charAt(0));
    }


    /**
     * Returns whether the given internal type is a primitive Category 2 type.
     * @param internalType the internal type,
     *                     e.g. "<code>L</code>".
     * @return <code>true</code> if the given type is a Category 2 type,
     *         <code>false</code> otherwise.
     */
    public static boolean isInternalCategory2Type(String internalType)
    {
        return internalType.length() == 1 &&
               (internalType.charAt(0) == ClassConstants.TYPE_LONG ||
                internalType.charAt(0) == ClassConstants.TYPE_DOUBLE);
    }


    /**
     * Returns whether the given internal type is a plain class type
     * (including an array type of a plain class type).
     * @param internalType the internal type,
     *                     e.g. "<code>Ljava/lang/Object;</code>".
     * @return <code>true</code> if the given type is a class type,
     *         <code>false</code> otherwise.
     */
    public static boolean isInternalClassType(String internalType)
    {
        int length = internalType.length();
        return length > 1 &&
//             internalType.charAt(0)        == ClassConstants.TYPE_CLASS_START &&
               internalType.charAt(length-1) == ClassConstants.TYPE_CLASS_END;
    }


    /**
     * Returns the internal type of a given class name.
     * @param internalClassName the internal class name,
     *                          e.g. "<code>java/lang/Object</code>".
     * @return the internal type,
     *                          e.g. "<code>Ljava/lang/Object;</code>".
     */
    public static String internalTypeFromClassName(String internalClassName)
    {
        return internalArrayTypeFromClassName(internalClassName, 0);
    }


    /**
     * Returns the internal array type of a given class name with a given number
     * of dimensions. If the number of dimensions is 0, the class name itself is
     * returned.
     * @param internalClassName the internal class name,
     *                          e.g. "<code>java/lang/Object</code>".
     * @param dimensionCount    the number of array dimensions.
     * @return the internal array type of the array elements,
     *                          e.g. "<code>Ljava/lang/Object;</code>".
     */
    public static String internalArrayTypeFromClassName(String internalClassName,
                                                        int    dimensionCount)
    {
        StringBuffer buffer = new StringBuffer(internalClassName.length() + dimensionCount + 2);

        for (int dimension = 0; dimension < dimensionCount; dimension++)
        {
            buffer.append(ClassConstants.TYPE_ARRAY);
        }

        return buffer.append(ClassConstants.TYPE_CLASS_START)
                     .append(internalClassName)
                     .append(ClassConstants.TYPE_CLASS_END)
                     .toString();
    }


    /**
     * Returns the internal array type of a given type, with a given number of
     * additional dimensions.
     * @param internalType the internal class name,
     *                          e.g. "<code>[Ljava/lang/Object;</code>".
     * @param dimensionDelta    the number of additional array dimensions,
     *                          e.g. 1.
     * @return the internal array type of the array elements,
     *                          e.g. "<code>[[Ljava/lang/Object;</code>".
     */
    public static String internalArrayTypeFromType(String internalType,
                                                   int    dimensionDelta)
    {
        StringBuffer buffer = new StringBuffer(internalType.length() + dimensionDelta);

        for (int dimension = 0; dimension < dimensionDelta; dimension++)
        {
            buffer.append(ClassConstants.TYPE_ARRAY);
        }

        return buffer.append(internalType).toString();
    }


    /**
     * Returns the internal element type of a given internal array type.
     * @param internalArrayType the internal array type,
     *                          e.g. "<code>[[Ljava/lang/Object;</code>" or
     *                               "<code>[I</code>".
     * @return the internal type of the array elements,
     *                          e.g. "<code>Ljava/lang/Object;</code>" or
     *                               "<code>I</code>".
     */
    public static String internalTypeFromArrayType(String internalArrayType)
    {
        int index = internalArrayType.lastIndexOf(ClassConstants.TYPE_ARRAY);
        return internalArrayType.substring(index + 1);
    }


    /**
     * Returns the internal class type (class name or array type) of a given
     * internal type (including an array type). This is the type that can be
     * stored in a class constant.
     * @param internalType the internal class type,
     *                          e.g. "<code>[I</code>",
     *                               "<code>[Ljava/lang/Object;</code>", or
     *                               "<code>Ljava/lang/Object;</code>".
     * @return the internal class name,
     *                          e.g. "<code>[I</code>",
     *                               "<code>[Ljava/lang/Object;</code>", or
     *                               "<code>java/lang/Object</code>".
     */
    public static String internalClassTypeFromType(String internalType)
    {
        return isInternalArrayType(internalType) ?
            internalType :
            internalClassNameFromClassType(internalType);
    }


    /**
     * Returns the internal type of of a given class type (class name or array
     * type). This is the type that can be stored in a class constant.
     * @param internalType the internal class type,
     *                          e.g. "<code>[I</code>",
     *                               "<code>[Ljava/lang/Object;</code>", or
     *                               "<code>java/lang/Object</code>".
     * @return the internal class name,
     *                          e.g. "<code>[I</code>",
     *                               "<code>[Ljava/lang/Object;</code>", or
     *                               "<code>Ljava/lang/Object;</code>".
     */
    public static String internalTypeFromClassType(String internalType)
    {
        return isInternalArrayType(internalType) ?
            internalType :
            internalTypeFromClassName(internalType);
    }


    /**
     * Returns the internal class name of a given internal class type
     * (including an array type). Types involving primitive types are returned
     * unchanged.
     * @param internalClassType the internal class type,
     *                          e.g. "<code>[Ljava/lang/Object;</code>",
     *                               "<code>Ljava/lang/Object;</code>", or
     *                               "<code>java/lang/Object</code>".
     * @return the internal class name,
     *                          e.g. "<code>java/lang/Object</code>".
     */
    public static String internalClassNameFromClassType(String internalClassType)
    {
        return isInternalClassType(internalClassType) ?
            internalClassType.substring(internalClassType.indexOf(ClassConstants.TYPE_CLASS_START)+1,
                                        internalClassType.length()-1) :
            internalClassType;
    }


    /**
     * Returns the internal class name of any given internal descriptor type,
     * disregarding array prefixes.
     * @param internalClassType the internal class type,
     *                          e.g. "<code>Ljava/lang/Object;</code>" or
     *                               "<code>[[I</code>".
     * @return the internal class name,
     *                          e.g. "<code>java/lang/Object</code>" or
     *                               <code>null</code>.
     */
    public static String internalClassNameFromType(String internalClassType)
    {
        if (!isInternalClassType(internalClassType))
        {
            return null;
        }

        // Is it an array type?
        if (isInternalArrayType(internalClassType))
        {
            internalClassType = internalTypeFromArrayType(internalClassType);
        }

        return internalClassNameFromClassType(internalClassType);
    }


    /**
     * Returns the internal numeric (or void or array) class name corresponding
     * to the given internal primitive type.
     * @param internalPrimitiveType the internal class type,
     *                          e.g. "<code>I</code>" or
     *                               "<code>V</code>".
     * @return the internal class name,
     *                          e.g. "<code>java/lang/Integer</code>" or
     *                               <code>java/lang/Void</code>.
     */
    public static String internalNumericClassNameFromPrimitiveType(char internalPrimitiveType)
    {
        switch (internalPrimitiveType)
        {
            case ClassConstants.TYPE_VOID:    return ClassConstants.NAME_JAVA_LANG_VOID;
            case ClassConstants.TYPE_BOOLEAN: return ClassConstants.NAME_JAVA_LANG_BOOLEAN;
            case ClassConstants.TYPE_BYTE:    return ClassConstants.NAME_JAVA_LANG_BYTE;
            case ClassConstants.TYPE_CHAR:    return ClassConstants.NAME_JAVA_LANG_CHARACTER;
            case ClassConstants.TYPE_SHORT:   return ClassConstants.NAME_JAVA_LANG_SHORT;
            case ClassConstants.TYPE_INT:     return ClassConstants.NAME_JAVA_LANG_INTEGER;
            case ClassConstants.TYPE_LONG:    return ClassConstants.NAME_JAVA_LANG_LONG;
            case ClassConstants.TYPE_FLOAT:   return ClassConstants.NAME_JAVA_LANG_FLOAT;
            case ClassConstants.TYPE_DOUBLE:  return ClassConstants.NAME_JAVA_LANG_DOUBLE;
            case ClassConstants.TYPE_ARRAY:   return ClassConstants.NAME_JAVA_LANG_REFLECT_ARRAY;
            default:
                throw new IllegalArgumentException("Unexpected primitive type ["+internalPrimitiveType+"]");
        }
    }


    /**
     * Returns the internal numeric (or void or array) class name corresponding
     * to the given internal primitive type.
     * @param internalPrimitiveClassName the internal class name,
     *                          e.g. "<code>java/lang/Integer</code>" or
     *                               <code>java/lang/Void</code>.
     * @return the internal class type,
     *                          e.g. "<code>I</code>" or
     *                               "<code>V</code>".
     */
    public static char internalPrimitiveTypeFromNumericClassName(String internalPrimitiveClassName)
    {
        if (internalPrimitiveClassName.equals(ClassConstants.NAME_JAVA_LANG_VOID))          return ClassConstants.TYPE_VOID;
        if (internalPrimitiveClassName.equals(ClassConstants.NAME_JAVA_LANG_BOOLEAN))       return ClassConstants.TYPE_BOOLEAN;
        if (internalPrimitiveClassName.equals(ClassConstants.NAME_JAVA_LANG_BYTE))          return ClassConstants.TYPE_BYTE;
        if (internalPrimitiveClassName.equals(ClassConstants.NAME_JAVA_LANG_CHARACTER))     return ClassConstants.TYPE_CHAR;
        if (internalPrimitiveClassName.equals(ClassConstants.NAME_JAVA_LANG_SHORT))         return ClassConstants.TYPE_SHORT;
        if (internalPrimitiveClassName.equals(ClassConstants.NAME_JAVA_LANG_INTEGER))       return ClassConstants.TYPE_INT;
        if (internalPrimitiveClassName.equals(ClassConstants.NAME_JAVA_LANG_LONG))          return ClassConstants.TYPE_LONG;
        if (internalPrimitiveClassName.equals(ClassConstants.NAME_JAVA_LANG_FLOAT))         return ClassConstants.TYPE_FLOAT;
        if (internalPrimitiveClassName.equals(ClassConstants.NAME_JAVA_LANG_DOUBLE))        return ClassConstants.TYPE_DOUBLE;
        if (internalPrimitiveClassName.equals(ClassConstants.NAME_JAVA_LANG_REFLECT_ARRAY)) return ClassConstants.TYPE_ARRAY;

        throw new IllegalArgumentException("Unexpected primitive class name ["+internalPrimitiveClassName+"]");
    }


    /**
     * Returns whether the given method name refers to a class initializer or
     * an instance initializer.
     * @param internalMethodName the internal method name,
     *                           e.g. "<code>&ltclinit&gt;</code>".
     * @return whether the method name refers to an initializer,
     *                           e.g. <code>true</code>.
     */
    public static boolean isInitializer(String internalMethodName)
    {
        return internalMethodName.equals(ClassConstants.METHOD_NAME_CLINIT) ||
               internalMethodName.equals(ClassConstants.METHOD_NAME_INIT);
    }


    /**
     * Returns the internal type of the given internal method descriptor.
     * @param internalMethodDescriptor the internal method descriptor,
     *                                 e.g. "<code>(II)Z</code>".
     * @return the internal return type,
     *                                 e.g. "<code>Z</code>".
     */
    public static String internalMethodReturnType(String internalMethodDescriptor)
    {
        int index = internalMethodDescriptor.indexOf(ClassConstants.METHOD_ARGUMENTS_CLOSE);
        return internalMethodDescriptor.substring(index + 1);
    }


    /**
     * Returns the number of parameters of the given internal method descriptor.
     * @param internalMethodDescriptor the internal method descriptor,
     *                                 e.g. "<code>(ID)Z</code>".
     * @return the number of parameters,
     *                                 e.g. 2.
     */
    public static int internalMethodParameterCount(String internalMethodDescriptor)
    {
        return internalMethodParameterCount(internalMethodDescriptor, true);
    }


    /**
     * Returns the number of parameters of the given internal method descriptor.
     * @param internalMethodDescriptor the internal method descriptor,
     *                                 e.g. "<code>(ID)Z</code>".
     * @param accessFlags              the access flags of the method,
     *                                 e.g. 0.
     * @return the number of parameters,
     *                                 e.g. 3.
     */
    public static int internalMethodParameterCount(String internalMethodDescriptor,
                                                   int    accessFlags)
    {
        return internalMethodParameterCount(internalMethodDescriptor,
                                            (accessFlags & ClassConstants.ACC_STATIC) != 0);
    }


    /**
     * Returns the number of parameters of the given internal method descriptor.
     * @param internalMethodDescriptor the internal method descriptor,
     *                                 e.g. "<code>(ID)Z</code>".
     * @param isStatic                 specifies whether the method is static,
     *                                 e.g. false.
     * @return the number of parameters,
     *                                 e.g. 3.
     */
    public static int internalMethodParameterCount(String  internalMethodDescriptor,
                                                   boolean isStatic)
    {
        int counter = isStatic ? 0 : 1;
        int index   = 1;

        while (true)
        {
            char c = internalMethodDescriptor.charAt(index++);
            switch (c)
            {
                case ClassConstants.TYPE_ARRAY:
                {
                    // Just ignore all array characters.
                    break;
                }
                case ClassConstants.TYPE_CLASS_START:
                {
                    counter++;

                    // Skip the class name.
                    index = internalMethodDescriptor.indexOf(ClassConstants.TYPE_CLASS_END, index) + 1;
                    break;
                }
                default:
                {
                    counter++;
                    break;
                }
                case ClassConstants.METHOD_ARGUMENTS_CLOSE:
                {
                    return counter;
                }
            }
        }
    }


    /**
     * Returns the size taken up on the stack by the parameters of the given
     * internal method descriptor. This accounts for long and double parameters
     * taking up two entries.
     * @param internalMethodDescriptor the internal method descriptor,
     *                                 e.g. "<code>(ID)Z</code>".
     * @return the size taken up on the stack,
     *                                 e.g. 3.
     */
    public static int internalMethodParameterSize(String internalMethodDescriptor)
    {
        return internalMethodParameterSize(internalMethodDescriptor, true);
    }


    /**
     * Returns the size taken up on the stack by the parameters of the given
     * internal method descriptor. This accounts for long and double parameters
     * taking up two entries, and a non-static method taking up an additional
     * entry.
     * @param internalMethodDescriptor the internal method descriptor,
     *                                 e.g. "<code>(ID)Z</code>".
     * @param accessFlags              the access flags of the method,
     *                                 e.g. 0.
     * @return the size taken up on the stack,
     *                                 e.g. 4.
     */
    public static int internalMethodParameterSize(String internalMethodDescriptor,
                                                  int    accessFlags)
    {
        return internalMethodParameterSize(internalMethodDescriptor,
                                           (accessFlags & ClassConstants.ACC_STATIC) != 0);
    }


    /**
     * Returns the size taken up on the stack by the parameters of the given
     * internal method descriptor. This accounts for long and double parameters
     * taking up two spaces, and a non-static method taking up an additional
     * entry.
     * @param internalMethodDescriptor the internal method descriptor,
     *                                 e.g. "<code>(ID)Z</code>".
     * @param isStatic                 specifies whether the method is static,
     *                                 e.g. false.
     * @return the size taken up on the stack,
     *                                 e.g. 4.
     */
    public static int internalMethodParameterSize(String  internalMethodDescriptor,
                                                  boolean isStatic)
    {
        int size  = isStatic ? 0 : 1;
        int index = 1;

        while (true)
        {
            char c = internalMethodDescriptor.charAt(index++);
            switch (c)
            {
                case ClassConstants.TYPE_LONG:
                case ClassConstants.TYPE_DOUBLE:
                {
                    size += 2;
                    break;
                }
                case ClassConstants.TYPE_CLASS_START:
                {
                    size++;

                    // Skip the class name.
                    index = internalMethodDescriptor.indexOf(ClassConstants.TYPE_CLASS_END, index) + 1;
                    break;
                }
                case ClassConstants.TYPE_ARRAY:
                {
                    size++;

                    // Skip all array characters.
                    while ((c = internalMethodDescriptor.charAt(index++)) == ClassConstants.TYPE_ARRAY) {}

                    if (c == ClassConstants.TYPE_CLASS_START)
                    {
                        // Skip the class type.
                        index = internalMethodDescriptor.indexOf(ClassConstants.TYPE_CLASS_END, index) + 1;
                    }
                    break;
                }
                default:
                {
                    size++;
                    break;
                }
                case ClassConstants.METHOD_ARGUMENTS_CLOSE:
                {
                    return size;
                }
            }
        }
    }


    /**
     * Returns the parameter number in the given internal method descriptor,
     * corresponding to the given variable index. This accounts for long and
     * double parameters taking up two spaces, and a non-static method taking
     * up an additional entry. The method returns 0 if the index corresponds
     * to the 'this' parameter and -1 if the index does not correspond to a
     * parameter.
     * @param internalMethodDescriptor the internal method descriptor,
     *                                 e.g. "<code>(IDI)Z</code>".
     * @param accessFlags              the access flags of the method,
     *                                 e.g. 0.
     * @param variableIndex            the variable index of the parameter,
     *                                 e.g. 4.
     * @return the parameter number in the descriptor,
     *                                 e.g. 3.
     */
    public static int internalMethodParameterNumber(String internalMethodDescriptor,
                                                    int    accessFlags,
                                                    int    variableIndex)
    {
        return internalMethodParameterNumber(internalMethodDescriptor,
                                             (accessFlags & ClassConstants.ACC_STATIC) != 0,
                                             variableIndex);
    }


    /**
     * Returns the parameter number in the given internal method descriptor,
     * corresponding to the given variable index. This accounts for long and
     * double parameters taking up two spaces, and a non-static method taking
     * up an additional entry. The method returns 0 if the index corresponds
     * to the 'this' parameter and -1 if the index does not correspond to a
     * parameter.
     * @param internalMethodDescriptor the internal method descriptor,
     *                                 e.g. "<code>(IDI)Z</code>".
     * @param isStatic                 specifies whether the method is static,
     *                                 e.g. false.
     * @param variableIndex            the variable index of the parameter,
     *                                 e.g. 4.
     * @return the parameter number in the descriptor,
     *                                 e.g. 3.
     */
    public static int internalMethodParameterNumber(String  internalMethodDescriptor,
                                                    boolean isStatic,
                                                    int     variableIndex)
    {
        int parameterIndex  = 0;
        int parameterNumber = 0;

        // Is it a non-static method?
        if (!isStatic)
        {
            if (variableIndex == 0)
            {
                return 0;
            }

            variableIndex--;
            parameterNumber++;
        }

        // Loop over all variables until we've found the right index.
        InternalTypeEnumeration internalTypeEnumeration =
            new InternalTypeEnumeration(internalMethodDescriptor);

        while (internalTypeEnumeration.hasMoreTypes())
        {
            if (variableIndex == parameterIndex)
            {
                return parameterNumber;
            }

            String internalType = internalTypeEnumeration.nextType();

            parameterIndex += internalTypeSize(internalType);
            parameterNumber++;
        }

        return -1;
    }


    /**
     * Returns the variable index corresponding to the given parameter number
     * in the given internal method descriptor. This accounts for long and
     * double parameters taking up two spaces, and a non-static method taking
     * up an additional entry. The method returns 0 if the number corresponds
     * to the 'this' parameter and -1 if the number does not correspond to a
     * parameter.
     * @param internalMethodDescriptor the internal method descriptor,
     *                                 e.g. "<code>(IDI)Z</code>".
     * @param accessFlags              the access flags of the method,
     *                                 e.g. 0.
     * @param parameterNumber          the parameter number,
     *                                 e.g. 3.
     * @return the corresponding variable index,
     *                                 e.g. 4.
     */
    public static int internalMethodVariableIndex(String internalMethodDescriptor,
                                                  int    accessFlags,
                                                  int    parameterNumber)
    {
        return internalMethodVariableIndex(internalMethodDescriptor,
                                           (accessFlags & ClassConstants.ACC_STATIC) != 0,
                                           parameterNumber);
    }


    /**
     * Returns the parameter index in the given internal method descriptor,
     * corresponding to the given variable number. This accounts for long and
     * double parameters taking up two spaces, and a non-static method taking
     * up an additional entry. The method returns 0 if the number corresponds
     * to the 'this' parameter and -1 if the number does not correspond to a
     * parameter.
     * @param internalMethodDescriptor the internal method descriptor,
     *                                 e.g. "<code>(IDI)Z</code>".
     * @param isStatic                 specifies whether the method is static,
     *                                 e.g. false.
     * @param parameterNumber          the parameter number,
     *                                 e.g. 3.
     * @return the corresponding variable index,
     *                                 e.g. 4.
     */
    public static int internalMethodVariableIndex(String  internalMethodDescriptor,
                                                  boolean isStatic,
                                                  int     parameterNumber)
    {
        int variableNumber = 0;
        int variableIndex  = isStatic ? 0 : 1;

        // Loop over the given number of parameters.
        InternalTypeEnumeration internalTypeEnumeration =
            new InternalTypeEnumeration(internalMethodDescriptor);

        for (int counter = 0; counter < parameterNumber; counter++)
        {
            String internalType = internalTypeEnumeration.nextType();

            variableIndex += internalTypeSize(internalType);
        }

        return variableIndex;
    }


    /**
     * Returns the internal type of the parameter in the given method descriptor,
     * at the given index.
     *
     * @param internalMethodDescriptor the internal method descriptor
     *                                 e.g. "<code>(IDI)Z</code>".
     * @param parameterIndex           the parameter index, e.g. 1.
     * @return the parameter's type, e.g. "<code>D</code>".
     */
    public static String internalMethodParameterType(String  internalMethodDescriptor,
                                                     int     parameterIndex)
    {
        InternalTypeEnumeration typeEnum = new InternalTypeEnumeration(internalMethodDescriptor);
        String                  type     = null;
        for (int i = 0; i <= parameterIndex; i++)
        {
            type = typeEnum.nextType();
        }
        return type;
    }


    /**
     * Returns the size taken up on the stack by the given internal type.
     * The size is 1, except for long and double types, for which it is 2,
     * and for the void type, for which 0 is returned.
     * @param internalType the internal type,
     *                     e.g. "<code>I</code>".
     * @return the size taken up on the stack,
     *                     e.g. 1.
     */
    public static int internalTypeSize(String internalType)
    {
        if (internalType.length() == 1)
        {
            char internalPrimitiveType = internalType.charAt(0);
            if      (internalPrimitiveType == ClassConstants.TYPE_LONG ||
                     internalPrimitiveType == ClassConstants.TYPE_DOUBLE)
            {
                return 2;
            }
            else if (internalPrimitiveType == ClassConstants.TYPE_VOID)
            {
                return 0;
            }
        }

        return 1;
    }


    /**
     * Converts an external type into an internal type.
     * @param externalType the external type,
     *                     e.g. "<code>java.lang.Object[][]</code>" or
     *                          "<code>int[]</code>".
     * @return the internal type,
     *                     e.g. "<code>[[Ljava/lang/Object;</code>" or
     *                          "<code>[I</code>".
     */
    public static String internalType(String externalType)
    {
        // Strip the array part, if any.
        int dimensionCount = externalArrayTypeDimensionCount(externalType);
        if (dimensionCount > 0)
        {
            externalType = externalType.substring(0, externalType.length() - dimensionCount * JavaConstants.TYPE_ARRAY.length());
        }

        // Analyze the actual type part.
        char internalTypeChar =
            externalType.equals(JavaConstants.TYPE_VOID   ) ? ClassConstants.TYPE_VOID    :
            externalType.equals(JavaConstants.TYPE_BOOLEAN) ? ClassConstants.TYPE_BOOLEAN :
            externalType.equals(JavaConstants.TYPE_BYTE   ) ? ClassConstants.TYPE_BYTE    :
            externalType.equals(JavaConstants.TYPE_CHAR   ) ? ClassConstants.TYPE_CHAR    :
            externalType.equals(JavaConstants.TYPE_SHORT  ) ? ClassConstants.TYPE_SHORT   :
            externalType.equals(JavaConstants.TYPE_INT    ) ? ClassConstants.TYPE_INT     :
            externalType.equals(JavaConstants.TYPE_FLOAT  ) ? ClassConstants.TYPE_FLOAT   :
            externalType.equals(JavaConstants.TYPE_LONG   ) ? ClassConstants.TYPE_LONG    :
            externalType.equals(JavaConstants.TYPE_DOUBLE ) ? ClassConstants.TYPE_DOUBLE  :
            externalType.equals("%"                       ) ? '%'                         :
                                                              (char)0;

        String internalType =
            internalTypeChar != 0 ? String.valueOf(internalTypeChar) :
                                    ClassConstants.TYPE_CLASS_START +
                                    internalClassName(externalType) +
                                    ClassConstants.TYPE_CLASS_END;

        // Prepend the array part, if any.
        for (int count = 0; count < dimensionCount; count++)
        {
            internalType = ClassConstants.TYPE_ARRAY + internalType;
        }

        return internalType;
    }


    /**
     * Returns the number of dimensions of the given external type.
     * @param externalType the external type,
     *                     e.g. "<code>[[Ljava/lang/Object;</code>".
     * @return the number of dimensions, e.g. 2.
     */
    public static int externalArrayTypeDimensionCount(String externalType)
    {
        int dimensions = 0;
        int length = JavaConstants.TYPE_ARRAY.length();
        int offset = externalType.length() - length;
        while (externalType.regionMatches(offset,
                                          JavaConstants.TYPE_ARRAY,
                                          0,
                                          length))
        {
            dimensions++;
            offset -= length;
        }

        return dimensions;
    }


    /**
     * Converts an internal type into an external type.
     * @param internalType the internal type,
     *                     e.g. "<code>Ljava/lang/Object;</code>" or
     *                          "<code>[[Ljava/lang/Object;</code>" or
     *                          "<code>[I</code>".
     * @return the external type,
     *                     e.g. "<code>java.lang.Object</code>" or
     *                          "<code>java.lang.Object[][]</code>" or
     *                          "<code>int[]</code>".
     */
    public static String externalType(String internalType)
    {
        // Strip the array part, if any.
        int dimensionCount = internalArrayTypeDimensionCount(internalType);
        if (dimensionCount > 0)
        {
            internalType = internalType.substring(dimensionCount);
        }

        // Analyze the actual type part.
        char internalTypeChar = internalType.charAt(0);

        String externalType =
            internalTypeChar == ClassConstants.TYPE_VOID        ? JavaConstants.TYPE_VOID    :
            internalTypeChar == ClassConstants.TYPE_BOOLEAN     ? JavaConstants.TYPE_BOOLEAN :
            internalTypeChar == ClassConstants.TYPE_BYTE        ? JavaConstants.TYPE_BYTE    :
            internalTypeChar == ClassConstants.TYPE_CHAR        ? JavaConstants.TYPE_CHAR    :
            internalTypeChar == ClassConstants.TYPE_SHORT       ? JavaConstants.TYPE_SHORT   :
            internalTypeChar == ClassConstants.TYPE_INT         ? JavaConstants.TYPE_INT     :
            internalTypeChar == ClassConstants.TYPE_FLOAT       ? JavaConstants.TYPE_FLOAT   :
            internalTypeChar == ClassConstants.TYPE_LONG        ? JavaConstants.TYPE_LONG    :
            internalTypeChar == ClassConstants.TYPE_DOUBLE      ? JavaConstants.TYPE_DOUBLE  :
            internalTypeChar == '%'                             ? "%"                        :
            internalTypeChar == ClassConstants.TYPE_CLASS_START ? externalClassName(internalType.substring(1, internalType.indexOf(ClassConstants.TYPE_CLASS_END))) :
                                                                  null;

        if (externalType == null)
        {
            throw new IllegalArgumentException("Unknown type ["+internalType+"]");
        }

        // Append the array part, if any.
        for (int count = 0; count < dimensionCount; count++)
        {
            externalType += JavaConstants.TYPE_ARRAY;
        }

        return externalType;
    }


    /**
     * Converts an internal type into an external type, as expected by
     * Class.forName.
     * @param internalType the internal type,
     *                     e.g. "<code>Ljava/lang/Object;</code>" or
     *                          "<code>[[Ljava/lang/Object;</code>" or
     *                          "<code>[I</code>".
     * @return the external type,
     *                     e.g. "<code>java.lang.Object</code>" or
     *                          "<code>[[Ljava.lang.Object;</code>" or
     *                          "<code>[I</code>".
     */
    public static String externalClassForNameType(String internalType)
    {
        return isInternalArrayType(internalType) ?
            externalClassName(internalType) :
            externalClassName(internalClassNameFromClassType(internalType));
    }


    /**
     * Returns whether the given internal descriptor String represents a method
     * descriptor.
     * @param internalDescriptor the internal descriptor String,
     *                           e.g. "<code>(II)Z</code>".
     * @return <code>true</code> if the given String is a method descriptor,
     *         <code>false</code> otherwise.
     */
    public static boolean isInternalMethodDescriptor(String internalDescriptor)
    {
        return internalDescriptor.charAt(0) == ClassConstants.METHOD_ARGUMENTS_OPEN;
    }


    /**
     * Returns whether the given member String represents an external method
     * name with arguments.
     * @param externalMemberNameAndArguments the external member String,
     *                                       e.g. "<code>myField</code>" or
     *                                       e.g. "<code>myMethod(int,int)</code>".
     * @return <code>true</code> if the given String refers to a method,
     *         <code>false</code> otherwise.
     */
    public static boolean isExternalMethodNameAndArguments(String externalMemberNameAndArguments)
    {
        return externalMemberNameAndArguments.indexOf(JavaConstants.METHOD_ARGUMENTS_OPEN) > 0;
    }


    /**
     * Returns the name part of the given external method name and arguments.
     * @param externalMethodNameAndArguments the external method name and arguments,
     *                                       e.g. "<code>myMethod(int,int)</code>".
     * @return the name part of the String, e.g. "<code>myMethod</code>".
     */
    public static String externalMethodName(String externalMethodNameAndArguments)
    {
        ExternalTypeEnumeration externalTypeEnumeration =
            new ExternalTypeEnumeration(externalMethodNameAndArguments);

        return externalTypeEnumeration.methodName();
    }


    /**
     * Converts the given external method return type and name and arguments to
     * an internal method descriptor.
     * @param externalReturnType             the external method return type,
     *                                       e.g. "<code>boolean</code>".
     * @param externalMethodNameAndArguments the external method name and arguments,
     *                                       e.g. "<code>myMethod(int,int)</code>".
     * @return the internal method descriptor,
     *                                       e.g. "<code>(II)Z</code>".
     */
    public static String internalMethodDescriptor(String externalReturnType,
                                                  String externalMethodNameAndArguments)
    {
        StringBuffer internalMethodDescriptor = new StringBuffer();
        internalMethodDescriptor.append(ClassConstants.METHOD_ARGUMENTS_OPEN);

        ExternalTypeEnumeration externalTypeEnumeration =
            new ExternalTypeEnumeration(externalMethodNameAndArguments);

        while (externalTypeEnumeration.hasMoreTypes())
        {
            internalMethodDescriptor.append(internalType(externalTypeEnumeration.nextType()));
        }

        internalMethodDescriptor.append(ClassConstants.METHOD_ARGUMENTS_CLOSE);
        internalMethodDescriptor.append(internalType(externalReturnType));

        return internalMethodDescriptor.toString();
    }


    /**
     * Converts the given external method return type and List of arguments to
     * an internal method descriptor.
     * @param externalReturnType the external method return type,
     *                                       e.g. "<code>boolean</code>".
     * @param externalArguments the external method arguments,
     *                                       e.g. <code>{ "int", "int" }</code>.
     * @return the internal method descriptor,
     *                                       e.g. "<code>(II)Z</code>".
     */
    public static String internalMethodDescriptor(String externalReturnType,
                                                  List   externalArguments)
    {
        StringBuffer internalMethodDescriptor = new StringBuffer();
        internalMethodDescriptor.append(ClassConstants.METHOD_ARGUMENTS_OPEN);

        for (int index = 0; index < externalArguments.size(); index++)
        {
            internalMethodDescriptor.append(internalType((String)externalArguments.get(index)));
        }

        internalMethodDescriptor.append(ClassConstants.METHOD_ARGUMENTS_CLOSE);
        internalMethodDescriptor.append(internalType(externalReturnType));

        return internalMethodDescriptor.toString();
    }


    /**
     * Converts the given internal method return type and List of arguments to
     * an internal method descriptor.
     *
     * @param internalReturnType the external method return type,
     *                           e.g. "<code>Z</code>".
     * @param internalArguments  the external method arguments,
     *                           e.g. <code>{ "I", "I" }</code>.
     * @return the internal method descriptor, e.g. "<code>(II)Z</code>".
     */
    public static String internalMethodDescriptorFromInternalTypes(String       internalReturnType,
                                                                   List<String> internalArguments)
    {
        StringBuilder internalMethodDescriptor = new StringBuilder();
        internalMethodDescriptor.append(ClassConstants.METHOD_ARGUMENTS_OPEN);

        for (String argument : internalArguments)
        {
            internalMethodDescriptor.append(argument);
        }

        internalMethodDescriptor.append(ClassConstants.METHOD_ARGUMENTS_CLOSE);
        internalMethodDescriptor.append(internalReturnType);

        return internalMethodDescriptor.toString();
    }


    /**
     * Converts an internal field description into an external full field description.
     * @param accessFlags             the access flags of the field.
     * @param fieldName               the field name,
     *                                e.g. "<code>myField</code>".
     * @param internalFieldDescriptor the internal field descriptor,
     *                                e.g. "<code>Z</code>".
     * @return the external full field description,
     *                                e.g. "<code>public boolean myField</code>".
     */
    public static String externalFullFieldDescription(int    accessFlags,
                                                      String fieldName,
                                                      String internalFieldDescriptor)
    {
        return externalFieldAccessFlags(accessFlags) +
               externalType(internalFieldDescriptor) +
               ' ' +
               fieldName;
    }


    /**
     * Converts an internal method description into an external full method description.
     * @param internalClassName        the internal name of the class of the method,
     *                                 e.g. "<code>mypackage/MyClass</code>".
     * @param accessFlags              the access flags of the method.
     * @param internalMethodName       the internal method name,
     *                                 e.g. "<code>myMethod</code>" or
     *                                      "<code>&lt;init&gt;</code>".
     * @param internalMethodDescriptor the internal method descriptor,
     *                                 e.g. "<code>(II)Z</code>".
     * @return the external full method description,
     *                                 e.g. "<code>public boolean myMethod(int,int)</code>" or
     *                                      "<code>public MyClass(int,int)</code>".
     */
    public static String externalFullMethodDescription(String internalClassName,
                                                       int    accessFlags,
                                                       String internalMethodName,
                                                       String internalMethodDescriptor)
    {
        return externalMethodAccessFlags(accessFlags) +
               externalMethodReturnTypeAndName(internalClassName,
                                               internalMethodName,
                                               internalMethodDescriptor) +
               JavaConstants.METHOD_ARGUMENTS_OPEN +
               externalMethodArguments(internalMethodDescriptor) +
               JavaConstants.METHOD_ARGUMENTS_CLOSE;
    }


    /**
     * Converts internal class access flags into an external access description.
     * @param accessFlags the class access flags.
     * @return the external class access description,
     *         e.g. "<code>public final </code>".
     */
    public static String externalClassAccessFlags(int accessFlags)
    {
        return externalClassAccessFlags(accessFlags, "");
    }


    /**
     * Converts internal class access flags into an external access description.
     * @param accessFlags the class access flags.
     * @param prefix      a prefix that is added to each access modifier.
     * @return the external class access description,
     *         e.g. "<code>public final </code>".
     */
    public static String externalClassAccessFlags(int accessFlags, String prefix)
    {
        if (accessFlags == 0)
        {
            return EMPTY_STRING;
        }

        StringBuffer string = new StringBuffer(50);

        if ((accessFlags & ClassConstants.ACC_PUBLIC) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_PUBLIC).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_PRIVATE) != 0)
        {
            // Only in InnerClasses attributes.
            string.append(prefix).append(JavaConstants.ACC_PRIVATE).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_PROTECTED) != 0)
        {
            // Only in InnerClasses attributes.
            string.append(prefix).append(JavaConstants.ACC_PROTECTED).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_STATIC) != 0)
        {
            // Only in InnerClasses attributes.
            string.append(prefix).append(JavaConstants.ACC_STATIC).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_FINAL) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_FINAL).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_ANNOTATION) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_ANNOTATION);
        }
        if ((accessFlags & ClassConstants.ACC_INTERFACE) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_INTERFACE).append(' ');
        }
        else if ((accessFlags & ClassConstants.ACC_ENUM) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_ENUM).append(' ');
        }
        else if ((accessFlags & ClassConstants.ACC_ABSTRACT) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_ABSTRACT).append(' ');
        }
        else if ((accessFlags & ClassConstants.ACC_SYNTHETIC) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_SYNTHETIC).append(' ');
        }
        else if ((accessFlags & ClassConstants.ACC_MODULE) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_MODULE).append(' ');
        }

        return string.toString();
    }


    /**
     * Converts internal field access flags into an external access description.
     * @param accessFlags the field access flags.
     * @return the external field access description,
     *         e.g. "<code>public volatile </code>".
     */
    public static String externalFieldAccessFlags(int accessFlags)
    {
        return externalFieldAccessFlags(accessFlags, "");
    }


    /**
     * Converts internal field access flags into an external access description.
     * @param accessFlags the field access flags.
     * @param prefix      a prefix that is added to each access modifier.
     * @return the external field access description,
     *         e.g. "<code>public volatile </code>".
     */
    public static String externalFieldAccessFlags(int accessFlags, String prefix)
    {
        if (accessFlags == 0)
        {
            return EMPTY_STRING;
        }

        StringBuffer string = new StringBuffer(50);

        if ((accessFlags & ClassConstants.ACC_PUBLIC) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_PUBLIC).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_PRIVATE) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_PRIVATE).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_PROTECTED) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_PROTECTED).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_STATIC) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_STATIC).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_FINAL) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_FINAL).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_VOLATILE) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_VOLATILE).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_TRANSIENT) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_TRANSIENT).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_SYNTHETIC) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_SYNTHETIC).append(' ');
        }

        return string.toString();
    }


    /**
     * Converts internal method access flags into an external access description.
     * @param accessFlags the method access flags.
     * @return the external method access description,
     *                    e.g. "<code>public synchronized </code>".
     */
    public static String externalMethodAccessFlags(int accessFlags)
    {
        return externalMethodAccessFlags(accessFlags, "");
    }


    /**
     * Converts internal method access flags into an external access description.
     * @param accessFlags the method access flags.
     * @param prefix      a prefix that is added to each access modifier.
     * @return the external method access description,
     *                    e.g. "public synchronized ".
     */
    public static String externalMethodAccessFlags(int accessFlags, String prefix)
    {
        if (accessFlags == 0)
        {
            return EMPTY_STRING;
        }

        StringBuffer string = new StringBuffer(50);

        if ((accessFlags & ClassConstants.ACC_PUBLIC) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_PUBLIC).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_PRIVATE) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_PRIVATE).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_PROTECTED) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_PROTECTED).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_STATIC) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_STATIC).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_FINAL) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_FINAL).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_SYNCHRONIZED) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_SYNCHRONIZED).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_BRIDGE) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_BRIDGE).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_VARARGS) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_VARARGS).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_NATIVE) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_NATIVE).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_ABSTRACT) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_ABSTRACT).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_STRICT) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_STRICT).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_SYNTHETIC) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_SYNTHETIC).append(' ');
        }

        return string.toString();
    }


    /**
     * Converts internal method parameter access flags into an external access
     * description.
     * @param accessFlags the method parameter access flags.
     * @return the external method parameter access description,
     *                    e.g. "<code>final mandated </code>".
     */
    public static String externalParameterAccessFlags(int accessFlags)
    {
        return externalParameterAccessFlags(accessFlags, "");
    }


    /**
     * Converts internal method parameter access flags into an external access
     * description.
     * @param accessFlags the method parameter access flags.
     * @param prefix      a prefix that is added to each access modifier.
     * @return the external method parameter access description,
     *                    e.g. "final mandated ".
     */
    public static String externalParameterAccessFlags(int accessFlags, String prefix)
    {
        if (accessFlags == 0)
        {
            return EMPTY_STRING;
        }

        StringBuffer string = new StringBuffer(50);

        if ((accessFlags & ClassConstants.ACC_FINAL) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_FINAL).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_SYNTHETIC) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_SYNTHETIC).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_MANDATED) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_MANDATED).append(' ');
        }

        return string.toString();
    }


    /**
     * Converts an internal method descriptor into an external method return type.
     * @param internalMethodDescriptor the internal method descriptor,
     *                                 e.g. "<code>(II)Z</code>".
     * @return the external method return type,
     *                                 e.g. "<code>boolean</code>".
     */
    public static String externalMethodReturnType(String internalMethodDescriptor)
    {
        return externalType(internalMethodReturnType(internalMethodDescriptor));
    }


    /**
     * Converts internal module access flags into an external access
     * description.
     * @param accessFlags the module access flags.
     * @return the external module access description,
     *                    e.g. "<code>open mandated </code>".
     */
    public static String externalModuleAccessFlags(int accessFlags)
    {
        return externalModuleAccessFlags(accessFlags, "");
    }


    /**
     * Converts internal module access flags into an external access
     * description.
     * @param accessFlags the module access flags.
     * @param prefix      a prefix that is added to each access modifier.
     * @return the external module access description,
     *                    e.g. "<code>final mandated </code>".
     */
    public static String externalModuleAccessFlags(int accessFlags, String prefix)
    {
        if (accessFlags == 0)
        {
            return EMPTY_STRING;
        }

        StringBuffer string = new StringBuffer(50);

        if ((accessFlags & ClassConstants.ACC_OPEN) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_OPEN).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_SYNTHETIC) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_SYNTHETIC).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_MANDATED) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_MANDATED).append(' ');
        }

        return string.toString();
    }


    /**
     * Converts internal module requires access flags into an external access
     * description.
     * @param accessFlags the module requires access flags.
     * @return the external module requires access description,
     *                    e.g. "<code>static mandated </code>".
     */
    public static String externalRequiresAccessFlags(int accessFlags)
    {
        return externalRequiresAccessFlags(accessFlags, "");
    }


    /**
     * Converts internal module requires access flags into an external access
     * description.
     * @param accessFlags the module requires access flags.
     * @param prefix      a prefix that is added to each access modifier.
     * @return the external module requires access description,
     *                    e.g. "<code>static mandated </code>".
     */
    public static String externalRequiresAccessFlags(int accessFlags, String prefix)
    {
        if (accessFlags == 0)
        {
            return EMPTY_STRING;
        }

        StringBuffer string = new StringBuffer(50);

        if ((accessFlags & ClassConstants.ACC_TRANSITIVE) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_TRANSITIVE).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_STATIC_PHASE) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_STATIC).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_SYNTHETIC) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_SYNTHETIC).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_MANDATED) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_MANDATED).append(' ');
        }

        return string.toString();
    }


    /**
     * Converts internal module exports access flags into an external access
     * description.
     * @param accessFlags the module exports access flags.
     * @return the external module exports access description,
     *                    e.g. "<code>synthetic mandated </code>".
     */
    public static String externalExportsAccessFlags(int accessFlags)
    {
        return externalExportsAccessFlags(accessFlags, "");
    }


    /**
     * Converts internal module exports access flags into an external access
     * description.
     * @param accessFlags the module exports access flags.
     * @param prefix      a prefix that is added to each access modifier.
     * @return the external module exports access description,
     *                    e.g. "<code>static mandated </code>".
     */
    public static String externalExportsAccessFlags(int accessFlags, String prefix)
    {
        if (accessFlags == 0)
        {
            return EMPTY_STRING;
        }

        StringBuffer string = new StringBuffer(50);

        if ((accessFlags & ClassConstants.ACC_SYNTHETIC) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_SYNTHETIC).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_MANDATED) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_MANDATED).append(' ');
        }

        return string.toString();
    }


    /**
     * Converts internal module opens access flags into an external access
     * description.
     * @param accessFlags the module opens access flags.
     * @return the external module opens access description,
     *                    e.g. "<code>synthetic mandated </code>".
     */
    public static String externalOpensAccessFlags(int accessFlags)
    {
        return externalOpensAccessFlags(accessFlags, "");
    }


    /**
     * Converts internal module opens access flags into an external access
     * description.
     * @param accessFlags the module opens access flags.
     * @param prefix      a prefix that is added to each access modifier.
     * @return the external module opens access description,
     *                    e.g. "<code>static mandated </code>".
     */
    public static String externalOpensAccessFlags(int accessFlags, String prefix)
    {
        if (accessFlags == 0)
        {
            return EMPTY_STRING;
        }

        StringBuffer string = new StringBuffer(50);

        if ((accessFlags & ClassConstants.ACC_SYNTHETIC) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_SYNTHETIC).append(' ');
        }
        if ((accessFlags & ClassConstants.ACC_MANDATED) != 0)
        {
            string.append(prefix).append(JavaConstants.ACC_MANDATED).append(' ');
        }

        return string.toString();
    }


    /**
     * Converts an internal class name, method name, and method descriptor to
     * an external method return type and name.
     * @param internalClassName        the internal name of the class of the method,
     *                                 e.g. "<code>mypackage/MyClass</code>".
     * @param internalMethodName       the internal method name,
     *                                 e.g. "<code>myMethod</code>" or
     *                                      "<code>&lt;init&gt;</code>".
     * @param internalMethodDescriptor the internal method descriptor,
     *                                 e.g. "<code>(II)Z</code>".
     * @return the external method return type and name,
     *                                 e.g. "<code>boolean myMethod</code>" or
     *                                      "<code>MyClass</code>".
     */
    private static String externalMethodReturnTypeAndName(String internalClassName,
                                                          String internalMethodName,
                                                          String internalMethodDescriptor)
    {
        return internalMethodName.equals(ClassConstants.METHOD_NAME_INIT) ?
            externalShortClassName(externalClassName(internalClassName)) :
            (externalMethodReturnType(internalMethodDescriptor) +
             ' ' +
             internalMethodName);
    }


    /**
     * Converts an internal method descriptor into an external method argument
     * description.
     * @param internalMethodDescriptor the internal method descriptor,
     *                                 e.g. "<code>(II)Z</code>".
     * @return the external method argument description,
     *                                 e.g. "<code>int,int</code>".
     */
    public static String externalMethodArguments(String internalMethodDescriptor)
    {
        StringBuffer externalMethodNameAndArguments = new StringBuffer();

        InternalTypeEnumeration internalTypeEnumeration =
            new InternalTypeEnumeration(internalMethodDescriptor);

        while (internalTypeEnumeration.hasMoreTypes())
        {
            externalMethodNameAndArguments.append(externalType(internalTypeEnumeration.nextType()));
            if (internalTypeEnumeration.hasMoreTypes())
            {
                externalMethodNameAndArguments.append(JavaConstants.METHOD_ARGUMENTS_SEPARATOR);
            }
        }

        return externalMethodNameAndArguments.toString();
    }


    /**
     * Returns the internal package name of the given internal class name.
     * @param internalClassName the internal class name,
     *                          e.g. "<code>java/lang/Object</code>".
     * @return the internal package name,
     *                          e.g. "<code>java/lang</code>".
     */
    public static String internalPackageName(String internalClassName)
    {
        String internalPackagePrefix = internalPackagePrefix(internalClassName);
        int length = internalPackagePrefix.length();
        return length > 0 ?
            internalPackagePrefix.substring(0, length - 1) :
            "";
    }


    /**
     * Returns the internal package prefix of the given internal class name.
     * @param internalClassName the internal class name,
     *                          e.g. "<code>java/lang/Object</code>".
     * @return the internal package prefix,
     *                          e.g. "<code>java/lang/</code>".
     */
    public static String internalPackagePrefix(String internalClassName)
    {
        return internalClassName.substring(0, internalClassName.lastIndexOf(ClassConstants.PACKAGE_SEPARATOR,
                                                                            internalClassName.length() - 2) + 1);
    }


    /**
     * Returns the external package name of the given external class name.
     * @param externalClassName the external class name,
     *                          e.g. "<code>java.lang.Object</code>".
     * @return the external package name,
     *                          e.g. "<code>java.lang</code>".
     */
    public static String externalPackageName(String externalClassName)
    {
        String externalPackagePrefix = externalPackagePrefix(externalClassName);
        int length = externalPackagePrefix.length();
        return length > 0 ?
            externalPackagePrefix.substring(0, length - 1) :
            "";
    }


    /**
     * Returns the external package prefix of the given external class name.
     * @param externalClassName the external class name,
     *                          e.g. "<code>java.lang.Object</code>".
     * @return the external package prefix,
     *                          e.g. "<code>java.lang.</code>".
     */
    public static String externalPackagePrefix(String externalClassName)
    {
        return externalClassName.substring(0, externalClassName.lastIndexOf(JavaConstants.PACKAGE_SEPARATOR,
                                                                            externalClassName.length() - 2) + 1);
    }
}

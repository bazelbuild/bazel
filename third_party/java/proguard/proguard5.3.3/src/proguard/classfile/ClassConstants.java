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
package proguard.classfile;

/**
 * Constants used in representing a Java class file (*.class).
 *
 * @author Eric Lafortune
 */
public class ClassConstants
{
    public static final String CLASS_FILE_EXTENSION = ".class";

    public static final int MAGIC = 0xCAFEBABE;

    public static final int CLASS_VERSION_1_0_MAJOR = 45;
    public static final int CLASS_VERSION_1_0_MINOR = 3;
    public static final int CLASS_VERSION_1_2_MAJOR = 46;
    public static final int CLASS_VERSION_1_2_MINOR = 0;
    public static final int CLASS_VERSION_1_3_MAJOR = 47;
    public static final int CLASS_VERSION_1_3_MINOR = 0;
    public static final int CLASS_VERSION_1_4_MAJOR = 48;
    public static final int CLASS_VERSION_1_4_MINOR = 0;
    public static final int CLASS_VERSION_1_5_MAJOR = 49;
    public static final int CLASS_VERSION_1_5_MINOR = 0;
    public static final int CLASS_VERSION_1_6_MAJOR = 50;
    public static final int CLASS_VERSION_1_6_MINOR = 0;
    public static final int CLASS_VERSION_1_7_MAJOR = 51;
    public static final int CLASS_VERSION_1_7_MINOR = 0;
    public static final int CLASS_VERSION_1_8_MAJOR = 52;
    public static final int CLASS_VERSION_1_8_MINOR = 0;

    public static final int CLASS_VERSION_1_0 = (CLASS_VERSION_1_0_MAJOR << 16) | CLASS_VERSION_1_0_MINOR;
    public static final int CLASS_VERSION_1_2 = (CLASS_VERSION_1_2_MAJOR << 16) | CLASS_VERSION_1_2_MINOR;
    public static final int CLASS_VERSION_1_3 = (CLASS_VERSION_1_3_MAJOR << 16) | CLASS_VERSION_1_3_MINOR;
    public static final int CLASS_VERSION_1_4 = (CLASS_VERSION_1_4_MAJOR << 16) | CLASS_VERSION_1_4_MINOR;
    public static final int CLASS_VERSION_1_5 = (CLASS_VERSION_1_5_MAJOR << 16) | CLASS_VERSION_1_5_MINOR;
    public static final int CLASS_VERSION_1_6 = (CLASS_VERSION_1_6_MAJOR << 16) | CLASS_VERSION_1_6_MINOR;
    public static final int CLASS_VERSION_1_7 = (CLASS_VERSION_1_7_MAJOR << 16) | CLASS_VERSION_1_7_MINOR;
    public static final int CLASS_VERSION_1_8 = (CLASS_VERSION_1_8_MAJOR << 16) | CLASS_VERSION_1_8_MINOR;

    public static final int ACC_PUBLIC       = 0x0001;
    public static final int ACC_PRIVATE      = 0x0002;
    public static final int ACC_PROTECTED    = 0x0004;
    public static final int ACC_STATIC       = 0x0008;
    public static final int ACC_FINAL        = 0x0010;
    public static final int ACC_SUPER        = 0x0020;
    public static final int ACC_SYNCHRONIZED = 0x0020;
    public static final int ACC_VOLATILE     = 0x0040;
    public static final int ACC_TRANSIENT    = 0x0080;
    public static final int ACC_BRIDGE       = 0x0040;
    public static final int ACC_VARARGS      = 0x0080;
    public static final int ACC_NATIVE       = 0x0100;
    public static final int ACC_INTERFACE    = 0x0200;
    public static final int ACC_ABSTRACT     = 0x0400;
    public static final int ACC_STRICT       = 0x0800;
    public static final int ACC_SYNTHETIC    = 0x1000;
    public static final int ACC_ANNOTATTION  = 0x2000;
    public static final int ACC_ENUM         = 0x4000;
    public static final int ACC_MANDATED     = 0x8000;

    public static final int VALID_ACC_CLASS     = ACC_PUBLIC       |
                                                  ACC_FINAL        |
                                                  ACC_SUPER        |
                                                  ACC_INTERFACE    |
                                                  ACC_ABSTRACT     |
                                                  ACC_SYNTHETIC    |
                                                  ACC_ANNOTATTION  |
                                                  ACC_ENUM;
    public static final int VALID_ACC_FIELD     = ACC_PUBLIC       |
                                                  ACC_PRIVATE      |
                                                  ACC_PROTECTED    |
                                                  ACC_STATIC       |
                                                  ACC_FINAL        |
                                                  ACC_VOLATILE     |
                                                  ACC_TRANSIENT    |
                                                  ACC_SYNTHETIC    |
                                                  ACC_ENUM;
    public static final int VALID_ACC_METHOD    = ACC_PUBLIC       |
                                                  ACC_PRIVATE      |
                                                  ACC_PROTECTED    |
                                                  ACC_STATIC       |
                                                  ACC_FINAL        |
                                                  ACC_SYNCHRONIZED |
                                                  ACC_BRIDGE       |
                                                  ACC_VARARGS      |
                                                  ACC_NATIVE       |
                                                  ACC_ABSTRACT     |
                                                  ACC_STRICT       |
                                                  ACC_SYNTHETIC;
    public static final int VALID_ACC_PARAMETER = ACC_FINAL        |
                                                  ACC_SYNTHETIC    |
                                                  ACC_MANDATED;

    public static final int CONSTANT_Utf8               = 1;
    public static final int CONSTANT_Integer            = 3;
    public static final int CONSTANT_Float              = 4;
    public static final int CONSTANT_Long               = 5;
    public static final int CONSTANT_Double             = 6;
    public static final int CONSTANT_Class              = 7;
    public static final int CONSTANT_String             = 8;
    public static final int CONSTANT_Fieldref           = 9;
    public static final int CONSTANT_Methodref          = 10;
    public static final int CONSTANT_InterfaceMethodref = 11;
    public static final int CONSTANT_NameAndType        = 12;
    public static final int CONSTANT_MethodHandle       = 15;
    public static final int CONSTANT_MethodType         = 16;
    public static final int CONSTANT_InvokeDynamic      = 18;

    public static final int REF_getField         = 1;
    public static final int REF_getStatic        = 2;
    public static final int REF_putField         = 3;
    public static final int REF_putStatic        = 4;
    public static final int REF_invokeVirtual    = 5;
    public static final int REF_invokeStatic     = 6;
    public static final int REF_invokeSpecial    = 7;
    public static final int REF_newInvokeSpecial = 8;
    public static final int REF_invokeInterface  = 9;

    public static final String ATTR_BootstrapMethods                     = "BootstrapMethods";
    public static final String ATTR_SourceFile                           = "SourceFile";
    public static final String ATTR_SourceDir                            = "SourceDir";
    public static final String ATTR_InnerClasses                         = "InnerClasses";
    public static final String ATTR_EnclosingMethod                      = "EnclosingMethod";
    public static final String ATTR_Deprecated                           = "Deprecated";
    public static final String ATTR_Synthetic                            = "Synthetic";
    public static final String ATTR_Signature                            = "Signature";
    public static final String ATTR_ConstantValue                        = "ConstantValue";
    public static final String ATTR_MethodParameters                     = "MethodParameters";
    public static final String ATTR_Exceptions                           = "Exceptions";
    public static final String ATTR_Code                                 = "Code";
    public static final String ATTR_StackMap                             = "StackMap";
    public static final String ATTR_StackMapTable                        = "StackMapTable";
    public static final String ATTR_LineNumberTable                      = "LineNumberTable";
    public static final String ATTR_LocalVariableTable                   = "LocalVariableTable";
    public static final String ATTR_LocalVariableTypeTable               = "LocalVariableTypeTable";
    public static final String ATTR_RuntimeVisibleAnnotations            = "RuntimeVisibleAnnotations";
    public static final String ATTR_RuntimeInvisibleAnnotations          = "RuntimeInvisibleAnnotations";
    public static final String ATTR_RuntimeVisibleParameterAnnotations   = "RuntimeVisibleParameterAnnotations";
    public static final String ATTR_RuntimeInvisibleParameterAnnotations = "RuntimeInvisibleParameterAnnotations";
    public static final String ATTR_RuntimeVisibleTypeAnnotations        = "RuntimeVisibleTypeAnnotations";
    public static final String ATTR_RuntimeInvisibleTypeAnnotations      = "RuntimeInvisibleTypeAnnotations";
    public static final String ATTR_AnnotationDefault                    = "AnnotationDefault";

    public static final int ANNOTATION_TARGET_ParameterGenericClass             = 0x00;
    public static final int ANNOTATION_TARGET_ParameterGenericMethod            = 0x01;
    public static final int ANNOTATION_TARGET_Extends                           = 0x10;
    public static final int ANNOTATION_TARGET_BoundGenericClass                 = 0x11;
    public static final int ANNOTATION_TARGET_BoundGenericMethod                = 0x12;
    public static final int ANNOTATION_TARGET_Field                             = 0x13;
    public static final int ANNOTATION_TARGET_Return                            = 0x14;
    public static final int ANNOTATION_TARGET_Receiver                          = 0x15;
    public static final int ANNOTATION_TARGET_Parameter                         = 0x16;
    public static final int ANNOTATION_TARGET_Throws                            = 0x17;
    public static final int ANNOTATION_TARGET_LocalVariable                     = 0x40;
    public static final int ANNOTATION_TARGET_ResourceVariable                  = 0x41;
    public static final int ANNOTATION_TARGET_Catch                             = 0x42;
    public static final int ANNOTATION_TARGET_InstanceOf                        = 0x43;
    public static final int ANNOTATION_TARGET_New                               = 0x44;
    public static final int ANNOTATION_TARGET_MethodReferenceNew                = 0x45;
    public static final int ANNOTATION_TARGET_MethodReference                   = 0x46;
    public static final int ANNOTATION_TARGET_Cast                              = 0x47;
    public static final int ANNOTATION_TARGET_ArgumentGenericMethodNew          = 0x48;
    public static final int ANNOTATION_TARGET_ArgumentGenericMethod             = 0x49;
    public static final int ANNOTATION_TARGET_ArgumentGenericMethodReferenceNew = 0x4a;
    public static final int ANNOTATION_TARGET_ArgumentGenericMethodReference    = 0x4b;

    public static final char ELEMENT_VALUE_STRING_CONSTANT = 's';
    public static final char ELEMENT_VALUE_ENUM_CONSTANT   = 'e';
    public static final char ELEMENT_VALUE_CLASS           = 'c';
    public static final char ELEMENT_VALUE_ANNOTATION      = '@';
    public static final char ELEMENT_VALUE_ARRAY           = '[';

    public static final char PACKAGE_SEPARATOR        = '/';
    public static final char INNER_CLASS_SEPARATOR    = '$';
    public static final char SPECIAL_CLASS_CHARACTER  = '-';
    public static final char SPECIAL_MEMBER_SEPARATOR = '$';

    public static final char METHOD_ARGUMENTS_OPEN  = '(';
    public static final char METHOD_ARGUMENTS_CLOSE = ')';

    public static final String PACKAGE_JAVA_LANG                        = "java/lang/";
    public static final String NAME_JAVA_LANG_OBJECT                    = "java/lang/Object";
    public static final String TYPE_JAVA_LANG_OBJECT                    = "Ljava/lang/Object;";
    public static final String NAME_JAVA_LANG_CLONEABLE                 = "java/lang/Cloneable";
    public static final String NAME_JAVA_LANG_THROWABLE                 = "java/lang/Throwable";
    public static final String NAME_JAVA_LANG_CLASS                     = "java/lang/Class";
    public static final String NAME_JAVA_LANG_STRING                    = "java/lang/String";
    public static final String NAME_JAVA_LANG_STRING_BUFFER             = "java/lang/StringBuffer";
    public static final String NAME_JAVA_LANG_STRING_BUILDER            = "java/lang/StringBuilder";
    public static final String NAME_JAVA_LANG_INVOKE_METHOD_HANDLE      = "java/lang/invoke/MethodHandle";
    public static final String NAME_JAVA_LANG_INVOKE_METHOD_TYPE        = "java/lang/invoke/MethodType";
    public static final String NAME_JAVA_LANG_VOID                      = "java/lang/Void";
    public static final String NAME_JAVA_LANG_BOOLEAN                   = "java/lang/Boolean";
    public static final String NAME_JAVA_LANG_BYTE                      = "java/lang/Byte";
    public static final String NAME_JAVA_LANG_SHORT                     = "java/lang/Short";
    public static final String NAME_JAVA_LANG_CHARACTER                 = "java/lang/Character";
    public static final String NAME_JAVA_LANG_INTEGER                   = "java/lang/Integer";
    public static final String NAME_JAVA_LANG_LONG                      = "java/lang/Long";
    public static final String NAME_JAVA_LANG_FLOAT                     = "java/lang/Float";
    public static final String NAME_JAVA_LANG_DOUBLE                    = "java/lang/Double";
    public static final String NAME_JAVA_LANG_MATH                      = "java/lang/Math";
    public static final String NAME_JAVA_LANG_SYSTEM                    = "java/lang/System";
    public static final String NAME_JAVA_LANG_RUNTIME                   = "java/lang/Runtime";
    public static final String NAME_JAVA_LANG_REFLECT_ARRAY             = "java/lang/reflect/Array";
    public static final String NAME_JAVA_LANG_REFLECT_FIELD             = "java/lang/reflect/Field";
    public static final String NAME_JAVA_LANG_REFLECT_METHOD            = "java/lang/reflect/Method";
    public static final String NAME_JAVA_LANG_REFLECT_CONSTRUCTOR       = "java/lang/reflect/Constructor";
    public static final String NAME_JAVA_LANG_REFLECT_ACCESSIBLE_OBJECT = "java/lang/reflect/AccessibleObject";
    public static final String NAME_JAVA_IO_SERIALIZABLE                = "java/io/Serializable";

    public static final String NAME_JAVA_UTIL_CONCURRENT_ATOMIC_ATOMIC_INTEGER_FIELD_UPDATER   = "java/util/concurrent/atomic/AtomicIntegerFieldUpdater";
    public static final String NAME_JAVA_UTIL_CONCURRENT_ATOMIC_ATOMIC_LONG_FIELD_UPDATER      = "java/util/concurrent/atomic/AtomicLongFieldUpdater";
    public static final String NAME_JAVA_UTIL_CONCURRENT_ATOMIC_ATOMIC_REFERENCE_FIELD_UPDATER = "java/util/concurrent/atomic/AtomicReferenceFieldUpdater";

    public static final String METHOD_NAME_INIT   = "<init>";
    public static final String METHOD_TYPE_INIT   = "()V";
    public static final String METHOD_NAME_CLINIT = "<clinit>";
    public static final String METHOD_TYPE_CLINIT = "()V";

    public static final String METHOD_NAME_CLASS_FOR_NAME                      = "forName";
    public static final String METHOD_TYPE_CLASS_FOR_NAME                      = "(Ljava/lang/String;)Ljava/lang/Class;";
    public static final String METHOD_NAME_CLASS_GET_COMPONENT_TYPE            = "getComponentType";
    public static final String METHOD_TYPE_CLASS_GET_COMPONENT_TYPE            = "()Ljava/lang/Class;";
    public static final String METHOD_NAME_CLASS_GET_FIELD                     = "getField";
    public static final String METHOD_TYPE_CLASS_GET_FIELD                     = "(Ljava/lang/String;)Ljava/lang/reflect/Field;";
    public static final String METHOD_NAME_CLASS_GET_DECLARED_FIELD            = "getDeclaredField";
    public static final String METHOD_TYPE_CLASS_GET_DECLARED_FIELD            = "(Ljava/lang/String;)Ljava/lang/reflect/Field;";
    public static final String CONSTRUCTOR_NAME_CLASS_GET_CONSTRUCTOR          = "getConstructor";
    public static final String CONSTRUCTOR_TYPE_CLASS_GET_CONSTRUCTOR          = "([Ljava/lang/Class;)Ljava/lang/reflect/Constructor;";
    public static final String CONSTRUCTOR_NAME_CLASS_GET_DECLARED_CONSTRUCTOR = "getDeclaredConstructor";
    public static final String CONSTRUCTOR_TYPE_CLASS_GET_DECLARED_CONSTRUCTOR = "([Ljava/lang/Class;)Ljava/lang/reflect/Constructor;";
    public static final String METHOD_NAME_CLASS_GET_METHOD                    = "getMethod";
    public static final String METHOD_TYPE_CLASS_GET_METHOD                    = "(Ljava/lang/String;[Ljava/lang/Class;)Ljava/lang/reflect/Method;";
    public static final String METHOD_NAME_CLASS_GET_DECLARED_METHOD           = "getDeclaredMethod";
    public static final String METHOD_TYPE_CLASS_GET_DECLARED_METHOD           = "(Ljava/lang/String;[Ljava/lang/Class;)Ljava/lang/reflect/Method;";
    public static final String METHOD_NAME_CLASS_GET_DECLARING_CLASS           = "getDeclaringClass";
    public static final String METHOD_NAME_CLASS_GET_ENCLOSING_CLASS           = "getEnclosingClass";
    public static final String METHOD_NAME_CLASS_GET_ENCLOSING_CONSTRUCTOR     = "getEnclosingConstructor";
    public static final String METHOD_NAME_CLASS_GET_ENCLOSING_METHOD          = "getEnclosingMethod";
    public static final String METHOD_NAME_GET_ANNOTATION                      = "getAnnotation";
    public static final String METHOD_NAME_GET_ANNOTATIONS                     = "getAnnotations";
    public static final String METHOD_NAME_GET_DECLARED_ANNOTATIONS            = "getDeclaredAnnotations";
    public static final String METHOD_NAME_GET_PARAMETER_ANNOTATIONS           = "getParameterAnnotations";
    public static final String METHOD_NAME_GET_TYPE_PREFIX                     = "getType";
    public static final String METHOD_NAME_GET_GENERIC_PREFIX                  = "getGeneric";
    public static final String METHOD_NAME_NEW_UPDATER                         = "newUpdater";
    public static final String METHOD_TYPE_NEW_INTEGER_UPDATER                 = "(Ljava/lang/Class;Ljava/lang/String;)Ljava/util/concurrent/atomic/AtomicIntegerFieldUpdater;";
    public static final String METHOD_TYPE_NEW_LONG_UPDATER                    = "(Ljava/lang/Class;Ljava/lang/String;)Ljava/util/concurrent/atomic/AtomicLongFieldUpdater;";
    public static final String METHOD_TYPE_NEW_REFERENCE_UPDATER               = "(Ljava/lang/Class;Ljava/lang/Class;Ljava/lang/String;)Ljava/util/concurrent/atomic/AtomicReferenceFieldUpdater;";

    public static final String METHOD_NAME_DOT_CLASS_JAVAC = "class$";
    public static final String METHOD_TYPE_DOT_CLASS_JAVAC = "(Ljava/lang/String;)Ljava/lang/Class;";
    public static final String METHOD_NAME_DOT_CLASS_JIKES = "class";
    public static final String METHOD_TYPE_DOT_CLASS_JIKES = "(Ljava/lang/String;Z)Ljava/lang/Class;";

    public static final String METHOD_TYPE_INIT_ENUM = "(Ljava/lang/String;I)V";

    public static final String METHOD_NAME_NEW_INSTANCE = "newInstance";
    public static final String METHOD_TYPE_NEW_INSTANCE = "()Ljava/lang/Object;";

    public static final String METHOD_NAME_EQUALS                 = "equals";
    public static final String METHOD_TYPE_EQUALS                 = "(Ljava/lang/Object;)Z";
    public static final String METHOD_NAME_LENGTH                 = "length";
    public static final String METHOD_NAME_VALUEOF                = "valueOf";
    public static final String METHOD_TYPE_VALUEOF_BOOLEAN        = "(Z)Ljava/lang/String;";
    public static final String METHOD_TYPE_VALUEOF_CHAR           = "(C)Ljava/lang/String;";
    public static final String METHOD_TYPE_VALUEOF_INT            = "(I)Ljava/lang/String;";
    public static final String METHOD_TYPE_VALUEOF_LONG           = "(J)Ljava/lang/String;";
    public static final String METHOD_TYPE_VALUEOF_FLOAT          = "(F)Ljava/lang/String;";
    public static final String METHOD_TYPE_VALUEOF_DOUBLE         = "(D)Ljava/lang/String;";
    public static final String METHOD_TYPE_VALUEOF_OBJECT         = "(Ljava/lang/Object;)Ljava/lang/String;";

    public static final String METHOD_TYPE_LENGTH                 = "()I";
    public static final String METHOD_NAME_APPEND                 = "append";
    public static final String METHOD_TYPE_STRING_VOID            = "(Ljava/lang/String;)V";
    public static final String METHOD_TYPE_BOOLEAN_STRING_BUFFER  = "(Z)Ljava/lang/StringBuffer;";
    public static final String METHOD_TYPE_CHAR_STRING_BUFFER     = "(C)Ljava/lang/StringBuffer;";
    public static final String METHOD_TYPE_INT_STRING_BUFFER      = "(I)Ljava/lang/StringBuffer;";
    public static final String METHOD_TYPE_LONG_STRING_BUFFER     = "(J)Ljava/lang/StringBuffer;";
    public static final String METHOD_TYPE_FLOAT_STRING_BUFFER    = "(F)Ljava/lang/StringBuffer;";
    public static final String METHOD_TYPE_DOUBLE_STRING_BUFFER   = "(D)Ljava/lang/StringBuffer;";
    public static final String METHOD_TYPE_STRING_STRING_BUFFER   = "(Ljava/lang/String;)Ljava/lang/StringBuffer;";
    public static final String METHOD_TYPE_OBJECT_STRING_BUFFER   = "(Ljava/lang/Object;)Ljava/lang/StringBuffer;";
    public static final String METHOD_TYPE_BOOLEAN_STRING_BUILDER = "(Z)Ljava/lang/StringBuilder;";
    public static final String METHOD_TYPE_CHAR_STRING_BUILDER    = "(C)Ljava/lang/StringBuilder;";
    public static final String METHOD_TYPE_INT_STRING_BUILDER     = "(I)Ljava/lang/StringBuilder;";
    public static final String METHOD_TYPE_LONG_STRING_BUILDER    = "(J)Ljava/lang/StringBuilder;";
    public static final String METHOD_TYPE_FLOAT_STRING_BUILDER   = "(F)Ljava/lang/StringBuilder;";
    public static final String METHOD_TYPE_DOUBLE_STRING_BUILDER  = "(D)Ljava/lang/StringBuilder;";
    public static final String METHOD_TYPE_STRING_STRING_BUILDER  = "(Ljava/lang/String;)Ljava/lang/StringBuilder;";
    public static final String METHOD_TYPE_OBJECT_STRING_BUILDER  = "(Ljava/lang/Object;)Ljava/lang/StringBuilder;";
    public static final String METHOD_NAME_TOSTRING               = "toString";
    public static final String METHOD_TYPE_TOSTRING               = "()Ljava/lang/String;";
    public static final String METHOD_NAME_CLONE                  = "clone";
    public static final String METHOD_TYPE_CLONE                  = "()Ljava/lang/Object;";

    public static final String METHOD_NAME_VALUES                 = "values";
    public static final String METHOD_NAME_ORDINAL                = "ordinal";
    public static final String METHOD_TYPE_ORDINAL                = "()I";

    public static final char TYPE_VOID                   = 'V';
    public static final char TYPE_BOOLEAN                = 'Z';
    public static final char TYPE_BYTE                   = 'B';
    public static final char TYPE_CHAR                   = 'C';
    public static final char TYPE_SHORT                  = 'S';
    public static final char TYPE_INT                    = 'I';
    public static final char TYPE_LONG                   = 'J';
    public static final char TYPE_FLOAT                  = 'F';
    public static final char TYPE_DOUBLE                 = 'D';
    public static final char TYPE_CLASS_START            = 'L';
    public static final char TYPE_CLASS_END              = ';';
    public static final char TYPE_ARRAY                  = '[';
    public static final char TYPE_GENERIC_VARIABLE_START = 'T';
    public static final char TYPE_GENERIC_START          = '<';
    public static final char TYPE_GENERIC_BOUND          = ':';
    public static final char TYPE_GENERIC_END            = '>';

    public static final int TYPICAL_CONSTANT_POOL_SIZE       = 256;
    public static final int TYPICAL_FIELD_COUNT              = 64;
    public static final int TYPICAL_METHOD_COUNT             = 64;
    public static final int TYPICAL_CODE_LENGTH              = 1024;
    public static final int TYPICAL_LINE_NUMBER_TABLE_LENGTH = 1024;
    public static final int TYPICAL_EXCEPTION_TABLE_LENGTH   = 16;
    public static final int TYPICAL_VARIABLES_SIZE           = 64;
    public static final int TYPICAL_STACK_SIZE               = 16;
}

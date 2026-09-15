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
package proguard.classfile;

/**
 * Constants used in representing a Java source file (*.java).
 *
 * @author Eric Lafortune
 */
public interface JavaConstants
{
    public static final String JAVA_FILE_EXTENSION = ".java";

    public static final String CLASS_VERSION_1_0       = "1.0";
    public static final String CLASS_VERSION_1_1       = "1.1";
    public static final String CLASS_VERSION_1_2       = "1.2";
    public static final String CLASS_VERSION_1_3       = "1.3";
    public static final String CLASS_VERSION_1_4       = "1.4";
    public static final String CLASS_VERSION_1_5       = "1.5";
    public static final String CLASS_VERSION_1_6       = "1.6";
    public static final String CLASS_VERSION_1_7       = "1.7";
    public static final String CLASS_VERSION_1_8       = "1.8";
    public static final String CLASS_VERSION_1_9       = "1.9";
    public static final String CLASS_VERSION_1_5_ALIAS = "5";
    public static final String CLASS_VERSION_1_6_ALIAS = "6";
    public static final String CLASS_VERSION_1_7_ALIAS = "7";
    public static final String CLASS_VERSION_1_8_ALIAS = "8";
    public static final String CLASS_VERSION_1_9_ALIAS = "9";
    public static final String CLASS_VERSION_10        = "10";
    public static final String CLASS_VERSION_11        = "11";
    public static final String CLASS_VERSION_12        = "12";
    public static final String CLASS_VERSION_13        = "13";

    public static final String ACC_PUBLIC       = "public";
    public static final String ACC_PRIVATE      = "private";
    public static final String ACC_PROTECTED    = "protected";
    public static final String ACC_STATIC       = "static";
    public static final String ACC_FINAL        = "final";
//  public static final String ACC_SUPER        = "super";
    public static final String ACC_SYNCHRONIZED = "synchronized";
    public static final String ACC_VOLATILE     = "volatile";
    public static final String ACC_TRANSIENT    = "transient";
    public static final String ACC_BRIDGE       = "bridge";
    public static final String ACC_VARARGS      = "varargs";
    public static final String ACC_NATIVE       = "native";
    public static final String ACC_INTERFACE    = "interface";
    public static final String ACC_ABSTRACT     = "abstract";
    public static final String ACC_STRICT       = "strictfp";
    public static final String ACC_SYNTHETIC    = "synthetic";
    public static final String ACC_ANNOTATION   = "@";
    public static final String ACC_ENUM         = "enum";
    public static final String ACC_MANDATED     = "mandated";
//  public static final String ACC_CONSTRUCTOR  = "constructor";
    public static final String ACC_MODULE       = "module";
    public static final String ACC_OPEN         = "open";
    public static final String ACC_TRANSITIVE   = "transitive";
//  public static final String ACC_STATIC_PHASE = "static";

    public static final char PACKAGE_SEPARATOR     = '.';
    public static final char INNER_CLASS_SEPARATOR = '.';
    public static final char SPECIAL_CLASS_CHARACTER        = '-';
    public static final char SPECIAL_MEMBER_SEPARATOR       = '$';

    public static final char METHOD_ARGUMENTS_OPEN      = '(';
    public static final char METHOD_ARGUMENTS_CLOSE     = ')';
    public static final char METHOD_ARGUMENTS_SEPARATOR = ',';

    public static final String TYPE_JAVA_LANG_OBJECT = "java.lang.Object";
    public static final String PACKAGE_JAVA_LANG     = "java.lang.";

    public static final String TYPE_VOID    = "void";
    public static final String TYPE_BOOLEAN = "boolean";
    public static final String TYPE_BYTE    = "byte";
    public static final String TYPE_CHAR    = "char";
    public static final String TYPE_SHORT   = "short";
    public static final String TYPE_INT     = "int";
    public static final String TYPE_FLOAT   = "float";
    public static final String TYPE_LONG    = "long";
    public static final String TYPE_DOUBLE  = "double";
    public static final String TYPE_ARRAY   = "[]";
}

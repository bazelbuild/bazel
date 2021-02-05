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
package proguard;

/**
 * This class provides constants for parsing and writing ProGuard configurations.
 *
 * @author Eric Lafortune
 */
class ConfigurationConstants
{
    public static final String OPTION_PREFIX            = "-";
    public static final String AT_DIRECTIVE             = "@";
    public static final String INCLUDE_DIRECTIVE        = "-include";
    public static final String BASE_DIRECTORY_DIRECTIVE = "-basedirectory";

    public static final String INJARS_OPTION       = "-injars";
    public static final String OUTJARS_OPTION      = "-outjars";
    public static final String LIBRARYJARS_OPTION  = "-libraryjars";
    public static final String RESOURCEJARS_OPTION = "-resourcejars";

    public static final String IF_OPTION                             = "-if";
    public static final String KEEP_OPTION                           = "-keep";
    public static final String KEEP_CLASS_MEMBERS_OPTION             = "-keepclassmembers";
    public static final String KEEP_CLASSES_WITH_MEMBERS_OPTION      = "-keepclasseswithmembers";
    public static final String KEEP_NAMES_OPTION                     = "-keepnames";
    public static final String KEEP_CLASS_MEMBER_NAMES_OPTION        = "-keepclassmembernames";
    public static final String KEEP_CLASSES_WITH_MEMBER_NAMES_OPTION = "-keepclasseswithmembernames";
    public static final String INCLUDE_DESCRIPTOR_CLASSES_SUBOPTION  = "includedescriptorclasses";
    public static final String INCLUDE_CODE_SUBOPTION                = "includecode";
    public static final String ALLOW_SHRINKING_SUBOPTION             = "allowshrinking";
    public static final String ALLOW_OPTIMIZATION_SUBOPTION          = "allowoptimization";
    public static final String ALLOW_OBFUSCATION_SUBOPTION           = "allowobfuscation";
    public static final String PRINT_SEEDS_OPTION                    = "-printseeds";

    public static final String DONT_SHRINK_OPTION         = "-dontshrink";
    public static final String PRINT_USAGE_OPTION         = "-printusage";
    public static final String WHY_ARE_YOU_KEEPING_OPTION = "-whyareyoukeeping";

    public static final String DONT_OPTIMIZE_OPTION                    = "-dontoptimize";
    public static final String OPTIMIZATIONS                           = "-optimizations";
    public static final String OPTIMIZATION_PASSES                     = "-optimizationpasses";
    public static final String ASSUME_NO_SIDE_EFFECTS_OPTION           = "-assumenosideeffects";
    public static final String ASSUME_NO_EXTERNAL_SIDE_EFFECTS_OPTION  = "-assumenoexternalsideeffects";
    public static final String ASSUME_NO_ESCAPING_PARAMETERS_OPTION    = "-assumenoescapingparameters";
    public static final String ASSUME_NO_EXTERNAL_RETURN_VALUES_OPTION = "-assumenoexternalreturnvalues";
    public static final String ASSUME_VALUES_OPTION                    = "-assumevalues";
    public static final String ALLOW_ACCESS_MODIFICATION_OPTION        = "-allowaccessmodification";
    public static final String MERGE_INTERFACES_AGGRESSIVELY_OPTION    = "-mergeinterfacesaggressively";

    public static final String DONT_OBFUSCATE_OPTION                  = "-dontobfuscate";
    public static final String PRINT_MAPPING_OPTION                   = "-printmapping";
    public static final String APPLY_MAPPING_OPTION                   = "-applymapping";
    public static final String OBFUSCATION_DICTIONARY_OPTION          = "-obfuscationdictionary";
    public static final String CLASS_OBFUSCATION_DICTIONARY_OPTION    = "-classobfuscationdictionary";
    public static final String PACKAGE_OBFUSCATION_DICTIONARY_OPTION  = "-packageobfuscationdictionary";
    public static final String OVERLOAD_AGGRESSIVELY_OPTION           = "-overloadaggressively";
    public static final String USE_UNIQUE_CLASS_MEMBER_NAMES_OPTION   = "-useuniqueclassmembernames";
    public static final String DONT_USE_MIXED_CASE_CLASS_NAMES_OPTION = "-dontusemixedcaseclassnames";
    public static final String KEEP_PACKAGE_NAMES_OPTION              = "-keeppackagenames";
    public static final String FLATTEN_PACKAGE_HIERARCHY_OPTION       = "-flattenpackagehierarchy";
    public static final String REPACKAGE_CLASSES_OPTION               = "-repackageclasses";
    public static final String DEFAULT_PACKAGE_OPTION                 = "-defaultpackage";
    public static final String KEEP_ATTRIBUTES_OPTION                 = "-keepattributes";
    public static final String KEEP_PARAMETER_NAMES_OPTION            = "-keepparameternames";
    public static final String RENAME_SOURCE_FILE_ATTRIBUTE_OPTION    = "-renamesourcefileattribute";
    public static final String ADAPT_CLASS_STRINGS_OPTION             = "-adaptclassstrings";
    public static final String ADAPT_RESOURCE_FILE_NAMES_OPTION       = "-adaptresourcefilenames";
    public static final String ADAPT_RESOURCE_FILE_CONTENTS_OPTION    = "-adaptresourcefilecontents";

    public static final String DONT_PREVERIFY_OPTION = "-dontpreverify";
    public static final String MICRO_EDITION_OPTION  = "-microedition";
    public static final String ANDROID_OPTION        = "-android";

    public static final String VERBOSE_OPTION                                    = "-verbose";
    public static final String DONT_NOTE_OPTION                                  = "-dontnote";
    public static final String DONT_WARN_OPTION                                  = "-dontwarn";
    public static final String IGNORE_WARNINGS_OPTION                            = "-ignorewarnings";
    public static final String PRINT_CONFIGURATION_OPTION                        = "-printconfiguration";
    public static final String DUMP_OPTION                                       = "-dump";
    public static final String ADD_CONFIGURATION_DEBUGGING_OPTION                = "-addconfigurationdebugging";
    public static final String SKIP_NON_PUBLIC_LIBRARY_CLASSES_OPTION            = "-skipnonpubliclibraryclasses";
    public static final String DONT_SKIP_NON_PUBLIC_LIBRARY_CLASSES_OPTION       = "-dontskipnonpubliclibraryclasses";
    public static final String DONT_SKIP_NON_PUBLIC_LIBRARY_CLASS_MEMBERS_OPTION = "-dontskipnonpubliclibraryclassmembers";
    public static final String TARGET_OPTION                                     = "-target";
    public static final String KEEP_DIRECTORIES_OPTION                           = "-keepdirectories";
    public static final String FORCE_PROCESSING_OPTION                           = "-forceprocessing";


    public static final String ANY_FILE_KEYWORD            = "**";

    public static final String ANY_ATTRIBUTE_KEYWORD       = "*";
    public static final String ATTRIBUTE_SEPARATOR_KEYWORD = ",";

    public static final String JAR_SEPARATOR_KEYWORD   = System.getProperty("path.separator");

    public static final char OPEN_SYSTEM_PROPERTY  = '<';
    public static final char CLOSE_SYSTEM_PROPERTY = '>';

    public static final String ANNOTATION_KEYWORD         = "@";
    public static final String NEGATOR_KEYWORD            = "!";
    public static final String CLASS_KEYWORD              = "class";
    public static final String ANY_CLASS_KEYWORD          = "*";
    public static final String ANY_TYPE_KEYWORD           = "***";
    public static final String IMPLEMENTS_KEYWORD         = "implements";
    public static final String EXTENDS_KEYWORD            = "extends";
    public static final String OPEN_KEYWORD               = "{";
    public static final String ANY_CLASS_MEMBER_KEYWORD   = "*";
    public static final String ANY_FIELD_KEYWORD          = "<fields>";
    public static final String ANY_METHOD_KEYWORD         = "<methods>";
    public static final String OPEN_ARGUMENTS_KEYWORD     = "(";
    public static final String ARGUMENT_SEPARATOR_KEYWORD = ",";
    public static final String ANY_ARGUMENTS_KEYWORD      = "...";
    public static final String CLOSE_ARGUMENTS_KEYWORD    = ")";
    public static final String EQUAL_KEYWORD              = "=";
    public static final String RETURN_KEYWORD             = "return";
    public static final String FALSE_KEYWORD              = "false";
    public static final String TRUE_KEYWORD               = "true";
    public static final String RANGE_KEYWORD              = "..";
    public static final String SEPARATOR_KEYWORD          = ";";
    public static final String CLOSE_KEYWORD              = "}";
}

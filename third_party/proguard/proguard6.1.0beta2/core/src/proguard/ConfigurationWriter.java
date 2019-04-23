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

import proguard.classfile.ClassConstants;
import proguard.classfile.util.ClassUtil;
import proguard.util.*;

import java.io.*;
import java.net.*;
import java.util.*;

/**
 * This class writes ProGuard configurations to a file.
 *
 * @author Eric Lafortune
 */
public class ConfigurationWriter
{
    private static final String[] KEEP_OPTIONS = new String[]
    {
        ConfigurationConstants.KEEP_OPTION,
        ConfigurationConstants.KEEP_CLASS_MEMBERS_OPTION,
        ConfigurationConstants.KEEP_CLASSES_WITH_MEMBERS_OPTION
    };


    private final PrintWriter writer;
    private       File        baseDir;


    /**
     * Creates a new ConfigurationWriter for the given file name.
     */
    public ConfigurationWriter(File configurationFile) throws IOException
    {
        this(PrintWriterUtil.createPrintWriterOut(configurationFile));

        baseDir = configurationFile.getParentFile();
    }


    /**
     * Creates a new ConfigurationWriter for the given PrintWriter.
     */
    public ConfigurationWriter(PrintWriter writer) throws IOException
    {
        this.writer = writer;
    }


    /**
     * Closes this ConfigurationWriter.
     */
    public void close() throws IOException
    {
        writer.close();
    }


    /**
     * Writes the given configuration.
     * @param configuration the configuration that is to be written out.
     * @throws IOException if an IO error occurs while writing the configuration.
     */
    public void write(Configuration configuration) throws IOException
    {
        // Write the program class path (input and output entries).
        writeJarOptions(ConfigurationConstants.INJARS_OPTION,
                        ConfigurationConstants.OUTJARS_OPTION,
                        configuration.programJars);
        writer.println();

        // Write the library class path (output entries only).
        writeJarOptions(ConfigurationConstants.LIBRARYJARS_OPTION,
                        ConfigurationConstants.LIBRARYJARS_OPTION,
                        configuration.libraryJars);
        writer.println();

        // Write the other options.
        writeOption(ConfigurationConstants.SKIP_NON_PUBLIC_LIBRARY_CLASSES_OPTION,            configuration.skipNonPublicLibraryClasses);
        writeOption(ConfigurationConstants.DONT_SKIP_NON_PUBLIC_LIBRARY_CLASS_MEMBERS_OPTION, !configuration.skipNonPublicLibraryClassMembers);
        writeOption(ConfigurationConstants.KEEP_DIRECTORIES_OPTION,                           configuration.keepDirectories);
        writeOption(ConfigurationConstants.TARGET_OPTION,                                     ClassUtil.externalClassVersion(configuration.targetClassVersion));
        writeOption(ConfigurationConstants.FORCE_PROCESSING_OPTION,                           configuration.lastModified == Long.MAX_VALUE);

        writeOption(ConfigurationConstants.DONT_SHRINK_OPTION, !configuration.shrink);
        writeOption(ConfigurationConstants.PRINT_USAGE_OPTION, configuration.printUsage);

        writeOption(ConfigurationConstants.DONT_OPTIMIZE_OPTION,                 !configuration.optimize);
        writeOption(ConfigurationConstants.OPTIMIZATIONS,                        configuration.optimizations);
        writeOption(ConfigurationConstants.OPTIMIZATION_PASSES,                  configuration.optimizationPasses);
        writeOption(ConfigurationConstants.ALLOW_ACCESS_MODIFICATION_OPTION,     configuration.allowAccessModification);
        writeOption(ConfigurationConstants.MERGE_INTERFACES_AGGRESSIVELY_OPTION, configuration.mergeInterfacesAggressively);

        writeOption(ConfigurationConstants.DONT_OBFUSCATE_OPTION,                            !configuration.obfuscate);
        writeOption(ConfigurationConstants.PRINT_MAPPING_OPTION,                             configuration.printMapping);
        writeOption(ConfigurationConstants.APPLY_MAPPING_OPTION,                             configuration.applyMapping);
        writeOption(ConfigurationConstants.OBFUSCATION_DICTIONARY_OPTION,                    configuration.obfuscationDictionary);
        writeOption(ConfigurationConstants.CLASS_OBFUSCATION_DICTIONARY_OPTION,              configuration.classObfuscationDictionary);
        writeOption(ConfigurationConstants.PACKAGE_OBFUSCATION_DICTIONARY_OPTION,            configuration.packageObfuscationDictionary);
        writeOption(ConfigurationConstants.OVERLOAD_AGGRESSIVELY_OPTION,                     configuration.overloadAggressively);
        writeOption(ConfigurationConstants.USE_UNIQUE_CLASS_MEMBER_NAMES_OPTION,             configuration.useUniqueClassMemberNames);
        writeOption(ConfigurationConstants.DONT_USE_MIXED_CASE_CLASS_NAMES_OPTION,           !configuration.useMixedCaseClassNames);
        writeOption(ConfigurationConstants.KEEP_PACKAGE_NAMES_OPTION,                        configuration.keepPackageNames, true);
        writeOption(ConfigurationConstants.FLATTEN_PACKAGE_HIERARCHY_OPTION,                 configuration.flattenPackageHierarchy, true);
        writeOption(ConfigurationConstants.REPACKAGE_CLASSES_OPTION,                         configuration.repackageClasses, true);
        writeOption(ConfigurationConstants.KEEP_ATTRIBUTES_OPTION,                           configuration.keepAttributes);
        writeOption(ConfigurationConstants.KEEP_PARAMETER_NAMES_OPTION,                      configuration.keepParameterNames);
        writeOption(ConfigurationConstants.RENAME_SOURCE_FILE_ATTRIBUTE_OPTION,              configuration.newSourceFileAttribute);
        writeOption(ConfigurationConstants.ADAPT_CLASS_STRINGS_OPTION,                       configuration.adaptClassStrings, true);
        writeOption(ConfigurationConstants.ADAPT_RESOURCE_FILE_NAMES_OPTION,                 configuration.adaptResourceFileNames);
        writeOption(ConfigurationConstants.ADAPT_RESOURCE_FILE_CONTENTS_OPTION,              configuration.adaptResourceFileContents);

        writeOption(ConfigurationConstants.DONT_PREVERIFY_OPTION,     !configuration.preverify);
        writeOption(ConfigurationConstants.MICRO_EDITION_OPTION,      configuration.microEdition);
        writeOption(ConfigurationConstants.ANDROID_OPTION,            configuration.android);

        writeOption(ConfigurationConstants.VERBOSE_OPTION,                     configuration.verbose);
        writeOption(ConfigurationConstants.DONT_NOTE_OPTION,                   configuration.note, true);
        writeOption(ConfigurationConstants.DONT_WARN_OPTION,                   configuration.warn, true);
        writeOption(ConfigurationConstants.IGNORE_WARNINGS_OPTION,             configuration.ignoreWarnings);
        writeOption(ConfigurationConstants.PRINT_CONFIGURATION_OPTION,         configuration.printConfiguration);
        writeOption(ConfigurationConstants.DUMP_OPTION,                        configuration.dump);
        writeOption(ConfigurationConstants.ADD_CONFIGURATION_DEBUGGING_OPTION, configuration.addConfigurationDebugging);

        writeOption(ConfigurationConstants.PRINT_SEEDS_OPTION, configuration.printSeeds);
        writer.println();

        // Write the "why are you keeping" options.
        writeOptions(ConfigurationConstants.WHY_ARE_YOU_KEEPING_OPTION, configuration.whyAreYouKeeping);
        writer.println();

        // Write the keep options.
        writeOptions(KEEP_OPTIONS, configuration.keep);

        // Write the "no side effect methods" options.
        writeOptions(ConfigurationConstants.ASSUME_NO_SIDE_EFFECTS_OPTION,           configuration.assumeNoSideEffects);
        writeOptions(ConfigurationConstants.ASSUME_NO_EXTERNAL_SIDE_EFFECTS_OPTION,  configuration.assumeNoExternalSideEffects);
        writeOptions(ConfigurationConstants.ASSUME_NO_ESCAPING_PARAMETERS_OPTION,    configuration.assumeNoEscapingParameters);
        writeOptions(ConfigurationConstants.ASSUME_NO_EXTERNAL_RETURN_VALUES_OPTION, configuration.assumeNoExternalReturnValues);
        writeOptions(ConfigurationConstants.ASSUME_VALUES_OPTION,                    configuration.assumeValues);


        if (writer.checkError())
        {
            throw new IOException("Can't write configuration");
        }
    }


    private void writeJarOptions(String    inputEntryOptionName,
                                 String    outputEntryOptionName,
                                 ClassPath classPath)
    {
        if (classPath != null)
        {
            for (int index = 0; index < classPath.size(); index++)
            {
                ClassPathEntry entry = classPath.get(index);
                String optionName = entry.isOutput() ?
                     outputEntryOptionName :
                     inputEntryOptionName;

                writer.print(optionName);
                writer.print(' ');
                writer.print(relativeFileName(entry.getFile()));

                // Append the filters, if any.
                boolean filtered = false;

                // For backward compatibility, the aar and apk filters come
                // first.
                filtered = writeFilter(filtered, entry.getAarFilter());
                filtered = writeFilter(filtered, entry.getApkFilter());
                filtered = writeFilter(filtered, entry.getZipFilter());
                filtered = writeFilter(filtered, entry.getJmodFilter());
                filtered = writeFilter(filtered, entry.getEarFilter());
                filtered = writeFilter(filtered, entry.getWarFilter());
                filtered = writeFilter(filtered, entry.getJarFilter());
                filtered = writeFilter(filtered, entry.getFilter());

                if (filtered)
                {
                    writer.print(ConfigurationConstants.CLOSE_ARGUMENTS_KEYWORD);
                }

                writer.println();
            }
        }
    }


    private boolean writeFilter(boolean filtered, List filter)
    {
        if (filtered)
        {
            writer.print(ConfigurationConstants.SEPARATOR_KEYWORD);
        }

        if (filter != null)
        {
            if (!filtered)
            {
                writer.print(ConfigurationConstants.OPEN_ARGUMENTS_KEYWORD);
            }

            writer.print(ListUtil.commaSeparatedString(filter, true));

            filtered = true;
        }

        return filtered;
    }


    private void writeOption(String optionName, boolean flag)
    {
        if (flag)
        {
            writer.println(optionName);
        }
    }


    private void writeOption(String optionName, int argument)
    {
        if (argument != 1)
        {
            writer.print(optionName);
            writer.print(' ');
            writer.println(argument);
        }
    }


    private void writeOption(String optionName, List arguments)
    {
        writeOption(optionName, arguments, false);
    }


    private void writeOption(String  optionName,
                             List    arguments,
                             boolean replaceInternalClassNames)
    {
        if (arguments != null)
        {
            if (arguments.isEmpty())
            {
                writer.println(optionName);
            }
            else
            {
                if (replaceInternalClassNames)
                {
                    arguments = externalClassNames(arguments);
                }

                writer.print(optionName);
                writer.print(' ');
                writer.println(ListUtil.commaSeparatedString(arguments, true));
            }
        }
    }


    private void writeOption(String optionName, String arguments)
    {
        writeOption(optionName, arguments, false);
    }


    private void writeOption(String  optionName,
                             String  arguments,
                             boolean replaceInternalClassNames)
    {
        if (arguments != null)
        {
            if (replaceInternalClassNames)
            {
                arguments = ClassUtil.externalClassName(arguments);
            }

            writer.print(optionName);
            writer.print(' ');
            writer.println(quotedString(arguments));
        }
    }


    private void writeOption(String optionName, URL url)
    {
        if (url != null)
        {
            if (url.getPath().length() > 0)
            {
                String fileName = url.toExternalForm();
                if (url.getProtocol().equals("file"))
                {
                    try
                    {
                        fileName = relativeFileName(new File(url.toURI()));
                    }
                    catch (URISyntaxException ignore) {}
                }
                else
                {
                }

                writer.print(optionName);
                writer.print(' ');
                writer.println(fileName);
            }
            else
            {
                writer.println(optionName);
            }
        }
    }


    private void writeOption(String optionName, File file)
    {
        if (file != null)
        {
            if (file.getPath().length() > 0)
            {
                writer.print(optionName);
                writer.print(' ');
                writer.println(relativeFileName(file));
            }
            else
            {
                writer.println(optionName);
            }
        }
    }


    private void writeOptions(String[] optionNames,
                              List     keepClassSpecifications)
    {
        if (keepClassSpecifications != null)
        {
            for (int index = 0; index < keepClassSpecifications.size(); index++)
            {
                writeOption(optionNames, (KeepClassSpecification)keepClassSpecifications.get(index));
            }
        }
    }


    private void writeOption(String[]               optionNames,
                             KeepClassSpecification keepClassSpecification)
    {
        if (keepClassSpecification.condition != null)
        {
            writeOption(ConfigurationConstants.IF_OPTION, keepClassSpecification.condition);
        }

        // Compose the option name.
        String optionName = optionNames[keepClassSpecification.markConditionally ? 2 :
                                        keepClassSpecification.markClasses       ? 0 :
                                                                                   1];

        if (keepClassSpecification.markDescriptorClasses)
        {
            optionName += ConfigurationConstants.ARGUMENT_SEPARATOR_KEYWORD +
                          ConfigurationConstants.INCLUDE_DESCRIPTOR_CLASSES_SUBOPTION;
        }

        if (keepClassSpecification.markCodeAttributes)
        {
            optionName += ConfigurationConstants.ARGUMENT_SEPARATOR_KEYWORD +
                          ConfigurationConstants.INCLUDE_CODE_SUBOPTION;
        }

        if (keepClassSpecification.allowShrinking)
        {
            optionName += ConfigurationConstants.ARGUMENT_SEPARATOR_KEYWORD +
                          ConfigurationConstants.ALLOW_SHRINKING_SUBOPTION;
        }

        if (keepClassSpecification.allowOptimization)
        {
            optionName += ConfigurationConstants.ARGUMENT_SEPARATOR_KEYWORD +
                          ConfigurationConstants.ALLOW_OPTIMIZATION_SUBOPTION;
        }

        if (keepClassSpecification.allowObfuscation)
        {
            optionName += ConfigurationConstants.ARGUMENT_SEPARATOR_KEYWORD +
                          ConfigurationConstants.ALLOW_OBFUSCATION_SUBOPTION;
        }

        // Write out the option with the proper class specification.
        writeOption(optionName, keepClassSpecification);
    }


    private void writeOptions(String optionName,
                              List   classSpecifications)
    {
        if (classSpecifications != null)
        {
            for (int index = 0; index < classSpecifications.size(); index++)
            {
                writeOption(optionName, (ClassSpecification)classSpecifications.get(index));
            }
        }
    }


    private void writeOption(String             optionName,
                             ClassSpecification classSpecification)
    {
        writer.println();

        // Write out the comments for this option.
        writeComments(classSpecification.comments);

        writer.print(optionName);
        writer.print(' ');

        // Write out the required annotation, if any.
        if (classSpecification.annotationType != null)
        {
            writer.print(ConfigurationConstants.ANNOTATION_KEYWORD);
            writer.print(ClassUtil.externalType(classSpecification.annotationType));
            writer.print(' ');
        }

        // Write out the class access flags.
        writer.print(ClassUtil.externalClassAccessFlags(classSpecification.requiredUnsetAccessFlags,
                                                        ConfigurationConstants.NEGATOR_KEYWORD));

        writer.print(ClassUtil.externalClassAccessFlags(classSpecification.requiredSetAccessFlags));

        // Write out the class keyword, if we didn't write the interface
        // keyword earlier.
        if (((classSpecification.requiredSetAccessFlags |
              classSpecification.requiredUnsetAccessFlags) &
             (ClassConstants.ACC_INTERFACE |
              ClassConstants.ACC_ENUM      |
              ClassConstants.ACC_MODULE)) == 0)
        {
            writer.print(ConfigurationConstants.CLASS_KEYWORD);
        }

        writer.print(' ');

        // Write out the class name.
        writer.print(classSpecification.className != null ?
            ClassUtil.externalClassName(classSpecification.className) :
            ConfigurationConstants.ANY_CLASS_KEYWORD);

        // Write out the extends template, if any.
        if (classSpecification.extendsAnnotationType != null ||
            classSpecification.extendsClassName      != null)
        {
            writer.print(' ');
            writer.print(ConfigurationConstants.EXTENDS_KEYWORD);
            writer.print(' ');

            // Write out the required extends annotation, if any.
            if (classSpecification.extendsAnnotationType != null)
            {
                writer.print(ConfigurationConstants.ANNOTATION_KEYWORD);
                writer.print(ClassUtil.externalType(classSpecification.extendsAnnotationType));
                writer.print(' ');
            }

            // Write out the extended class name.
            writer.print(classSpecification.extendsClassName != null ?
                ClassUtil.externalClassName(classSpecification.extendsClassName) :
                ConfigurationConstants.ANY_CLASS_KEYWORD);
        }

        // Write out the keep field and keep method options, if any.
        if (classSpecification.fieldSpecifications  != null ||
            classSpecification.methodSpecifications != null)
        {
            writer.print(' ');
            writer.println(ConfigurationConstants.OPEN_KEYWORD);

            writeFieldSpecification( classSpecification.fieldSpecifications);
            writeMethodSpecification(classSpecification.methodSpecifications);

            writer.println(ConfigurationConstants.CLOSE_KEYWORD);
        }
        else
        {
            writer.println();
        }
    }



    private void writeComments(String comments)
    {
        if (comments != null)
        {
            int index = 0;
            while (index < comments.length())
            {
                int breakIndex = comments.indexOf('\n', index);
                if (breakIndex < 0)
                {
                    breakIndex = comments.length();
                }

                writer.print('#');

                if (comments.charAt(index) != ' ')
                {
                    writer.print(' ');
                }

                writer.println(comments.substring(index, breakIndex));

                index = breakIndex + 1;
            }
        }
    }


    private void writeFieldSpecification(List memberSpecifications)
    {
        if (memberSpecifications != null)
        {
            for (int index = 0; index < memberSpecifications.size(); index++)
            {
                MemberSpecification memberSpecification =
                    (MemberSpecification)memberSpecifications.get(index);

                writer.print("    ");

                // Write out the required annotation, if any.
                if (memberSpecification.annotationType != null)
                {
                    writer.print(ConfigurationConstants.ANNOTATION_KEYWORD);
                    writer.println(ClassUtil.externalType(memberSpecification.annotationType));
                    writer.print("    ");
                }

                // Write out the field access flags.
                writer.print(ClassUtil.externalFieldAccessFlags(memberSpecification.requiredUnsetAccessFlags,
                                                                ConfigurationConstants.NEGATOR_KEYWORD));

                writer.print(ClassUtil.externalFieldAccessFlags(memberSpecification.requiredSetAccessFlags));

                // Write out the field name and descriptor.
                String name       = memberSpecification.name;
                String descriptor = memberSpecification.descriptor;

                writer.print(descriptor == null ? name == null ?
                    ConfigurationConstants.ANY_FIELD_KEYWORD             :
                    ConfigurationConstants.ANY_TYPE_KEYWORD + ' ' + name :
                    ClassUtil.externalFullFieldDescription(0,
                                                           name == null ? ConfigurationConstants.ANY_CLASS_MEMBER_KEYWORD : name,
                                                           descriptor));

                writeValueAssignment(ConfigurationConstants.EQUAL_KEYWORD,
                                     memberSpecification);

                writer.println(ConfigurationConstants.SEPARATOR_KEYWORD);
            }
        }
    }


    private void writeMethodSpecification(List memberSpecifications)
    {
        if (memberSpecifications != null)
        {
            for (int index = 0; index < memberSpecifications.size(); index++)
            {
                MemberSpecification memberSpecification =
                    (MemberSpecification)memberSpecifications.get(index);

                writer.print("    ");

                // Write out the required annotation, if any.
                if (memberSpecification.annotationType != null)
                {
                    writer.print(ConfigurationConstants.ANNOTATION_KEYWORD);
                    writer.println(ClassUtil.externalType(memberSpecification.annotationType));
                    writer.print("    ");
                }

                // Write out the method access flags.
                writer.print(ClassUtil.externalMethodAccessFlags(memberSpecification.requiredUnsetAccessFlags,
                                                                 ConfigurationConstants.NEGATOR_KEYWORD));

                writer.print(ClassUtil.externalMethodAccessFlags(memberSpecification.requiredSetAccessFlags));

                // Write out the method name and descriptor.
                String name       = memberSpecification.name;
                String descriptor = memberSpecification.descriptor;

                writer.print(descriptor == null ? name == null ?
                    ConfigurationConstants.ANY_METHOD_KEYWORD :
                    ConfigurationConstants.ANY_TYPE_KEYWORD + ' ' + name + ConfigurationConstants.OPEN_ARGUMENTS_KEYWORD + ConfigurationConstants.ANY_ARGUMENTS_KEYWORD + ConfigurationConstants.CLOSE_ARGUMENTS_KEYWORD :
                    ClassUtil.externalFullMethodDescription(ClassConstants.METHOD_NAME_INIT,
                                                            0,
                                                            name == null ? ConfigurationConstants.ANY_CLASS_MEMBER_KEYWORD : name,
                                                            descriptor));

                writeValueAssignment(ConfigurationConstants.RETURN_KEYWORD,
                                     memberSpecification);

                writer.println(ConfigurationConstants.SEPARATOR_KEYWORD);
            }
        }
    }


    private void writeValueAssignment(String              assignmentKeyword,
                                      MemberSpecification memberSpecification)
    {
        if (memberSpecification instanceof MemberValueSpecification)
        {
            MemberValueSpecification memberValueSpecification =
                (MemberValueSpecification)memberSpecification;

            Number[] values = memberValueSpecification.values;
            if (values != null)
            {
                writer.print(' ');
                writer.print(assignmentKeyword);
                writer.print(' ');

                // Write the first value.
                // Is it a boolean?
                String descriptor = memberSpecification.descriptor;
                if (descriptor != null &&
                    ClassUtil.internalMethodReturnType(descriptor).equals("" + ClassConstants.TYPE_BOOLEAN))
                {
                    // It's a boolean (represented as an integer).
                    writer.print(values[0].intValue() != 0);
                }
                else
                {
                    // It's a number.
                    writer.print(values[0]);
                }

                // Write the second value of the range, if any.
                if (values.length > 1)
                {
                    writer.print(ConfigurationConstants.RANGE_KEYWORD);
                    writer.print(values[1]);
                }
            }
        }
    }


    /**
     * Returns a list with external versions of the given list of internal
     * class names.
     */
    private List externalClassNames(List internalClassNames)
    {
        List externalClassNames = new ArrayList(internalClassNames.size());

        for (int index = 0; index < internalClassNames.size(); index++)
        {
            externalClassNames.add(ClassUtil.externalClassName((String)internalClassNames.get(index)));
        }

        return externalClassNames;
    }


    /**
     * Returns a relative file name of the given file, if possible.
     * The file name is also quoted, if necessary.
     */
    private String relativeFileName(File file)
    {
        String fileName = file.getAbsolutePath();

        // See if we can convert the file name into a relative file name.
        if (baseDir != null)
        {
            String baseDirName = baseDir.getAbsolutePath() + File.separator;
            if (fileName.startsWith(baseDirName))
            {
                fileName = fileName.substring(baseDirName.length());
            }
        }

        return quotedString(fileName);
    }


    /**
     * Returns a quoted version of the given string, if necessary.
     */
    private String quotedString(String string)
    {
        return string.length()     == 0 ||
               string.indexOf(' ') >= 0 ||
               string.indexOf('@') >= 0 ||
               string.indexOf('{') >= 0 ||
               string.indexOf('}') >= 0 ||
               string.indexOf('(') >= 0 ||
               string.indexOf(')') >= 0 ||
               string.indexOf(':') >= 0 ||
               string.indexOf(';') >= 0 ||
               string.indexOf(',') >= 0  ? ("'" + string + "'") :
                                           (      string      );
    }


    /**
     * A main method for testing configuration writing.
     */
    public static void main(String[] args)
    {
        try
        {
            ConfigurationWriter writer = new ConfigurationWriter(new File(args[0]));

            writer.write(new Configuration());
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
    }
}

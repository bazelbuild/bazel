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
package proguard;

import proguard.classfile.*;
import proguard.classfile.util.ClassUtil;
import proguard.util.ListUtil;

import java.io.*;
import java.net.URL;
import java.util.*;


/**
 * This class parses ProGuard configurations. Configurations can be read from an
 * array of arguments or from a configuration file or URL. External references
 * in file names ('<...>') can be resolved against a given set of properties.
 *
 * @author Eric Lafortune
 */
public class ConfigurationParser
{
    private final WordReader reader;
    private final Properties properties;

    private String     nextWord;
    private String     lastComments;


    /**
     * Creates a new ConfigurationParser for the given String arguments and
     * the given Properties.
     */
    public ConfigurationParser(String[]   args,
                               Properties properties) throws IOException
    {
        this(args, null, properties);
    }


    /**
     * Creates a new ConfigurationParser for the given String arguments,
     * with the given base directory and the given Properties.
     */
    public ConfigurationParser(String[]   args,
                               File       baseDir,
                               Properties properties) throws IOException
    {
        this(new ArgumentWordReader(args, baseDir), properties);
    }


    /**
     * Creates a new ConfigurationParser for the given lines,
     * with the given base directory and the given Properties.
     */
    public ConfigurationParser(String     lines,
                               String     description,
                               File       baseDir,
                               Properties properties) throws IOException
    {
        this(new LineWordReader(new LineNumberReader(new StringReader(lines)),
                                description,
                                baseDir),
             properties);
    }


    /**
     * Creates a new ConfigurationParser for the given file, with the system
     * Properties.
     * @deprecated Temporary code for backward compatibility in Obclipse.
     */
    public ConfigurationParser(File file) throws IOException
    {
        this(file, System.getProperties());
    }


    /**
     * Creates a new ConfigurationParser for the given file and the given
     * Properties.
     */
    public ConfigurationParser(File       file,
                               Properties properties) throws IOException
    {
        this(new FileWordReader(file), properties);
    }


    /**
     * Creates a new ConfigurationParser for the given URL and the given
     * Properties.
     */
    public ConfigurationParser(URL        url,
                               Properties properties) throws IOException
    {
        this(new FileWordReader(url), properties);
    }


    /**
     * Creates a new ConfigurationParser for the given word reader and the
     * given Properties.
     */
    public ConfigurationParser(WordReader reader,
                               Properties properties) throws IOException
    {
        this.reader     = reader;
        this.properties = properties;

        readNextWord();
    }


    /**
     * Parses and returns the configuration.
     * @param configuration the configuration that is updated as a side-effect.
     * @throws ParseException if the any of the configuration settings contains
     *                        a syntax error.
     * @throws IOException if an IO error occurs while reading a configuration.
     */
    public void parse(Configuration configuration)
    throws ParseException, IOException
    {
        while (nextWord != null)
        {
            lastComments = reader.lastComments();

            // First include directives.
            if      (ConfigurationConstants.AT_DIRECTIVE                                     .startsWith(nextWord) ||
                     ConfigurationConstants.INCLUDE_DIRECTIVE                                .startsWith(nextWord)) configuration.lastModified                     = parseIncludeArgument(configuration.lastModified);
            else if (ConfigurationConstants.BASE_DIRECTORY_DIRECTIVE                         .startsWith(nextWord)) parseBaseDirectoryArgument();

            // Then configuration options with or without arguments.
            else if (ConfigurationConstants.INJARS_OPTION                                    .startsWith(nextWord)) configuration.programJars                      = parseClassPathArgument(configuration.programJars, false);
            else if (ConfigurationConstants.OUTJARS_OPTION                                   .startsWith(nextWord)) configuration.programJars                      = parseClassPathArgument(configuration.programJars, true);
            else if (ConfigurationConstants.LIBRARYJARS_OPTION                               .startsWith(nextWord)) configuration.libraryJars                      = parseClassPathArgument(configuration.libraryJars, false);
            else if (ConfigurationConstants.RESOURCEJARS_OPTION                              .startsWith(nextWord)) throw new ParseException("The '-resourcejars' option is no longer supported. Please use the '-injars' option for all input");
            else if (ConfigurationConstants.SKIP_NON_PUBLIC_LIBRARY_CLASSES_OPTION           .startsWith(nextWord)) configuration.skipNonPublicLibraryClasses      = parseNoArgument(true);
            else if (ConfigurationConstants.DONT_SKIP_NON_PUBLIC_LIBRARY_CLASSES_OPTION      .startsWith(nextWord)) configuration.skipNonPublicLibraryClasses      = parseNoArgument(false);
            else if (ConfigurationConstants.DONT_SKIP_NON_PUBLIC_LIBRARY_CLASS_MEMBERS_OPTION.startsWith(nextWord)) configuration.skipNonPublicLibraryClassMembers = parseNoArgument(false);
            else if (ConfigurationConstants.TARGET_OPTION                                    .startsWith(nextWord)) configuration.targetClassVersion               = parseClassVersion();
            else if (ConfigurationConstants.FORCE_PROCESSING_OPTION                          .startsWith(nextWord)) configuration.lastModified                     = parseNoArgument(Long.MAX_VALUE);

            else if (ConfigurationConstants.KEEP_OPTION                                      .startsWith(nextWord)) configuration.keep                             = parseKeepClassSpecificationArguments(configuration.keep, true,  false, false);
            else if (ConfigurationConstants.KEEP_CLASS_MEMBERS_OPTION                        .startsWith(nextWord)) configuration.keep                             = parseKeepClassSpecificationArguments(configuration.keep, false, false, false);
            else if (ConfigurationConstants.KEEP_CLASSES_WITH_MEMBERS_OPTION                 .startsWith(nextWord)) configuration.keep                             = parseKeepClassSpecificationArguments(configuration.keep, false, true,  false);
            else if (ConfigurationConstants.KEEP_NAMES_OPTION                                .startsWith(nextWord)) configuration.keep                             = parseKeepClassSpecificationArguments(configuration.keep, true,  false, true);
            else if (ConfigurationConstants.KEEP_CLASS_MEMBER_NAMES_OPTION                   .startsWith(nextWord)) configuration.keep                             = parseKeepClassSpecificationArguments(configuration.keep, false, false, true);
            else if (ConfigurationConstants.KEEP_CLASSES_WITH_MEMBER_NAMES_OPTION            .startsWith(nextWord)) configuration.keep                             = parseKeepClassSpecificationArguments(configuration.keep, false, true,  true);
            else if (ConfigurationConstants.PRINT_SEEDS_OPTION                               .startsWith(nextWord)) configuration.printSeeds                       = parseOptionalFile();

            // After '-keep'.
            else if (ConfigurationConstants.KEEP_DIRECTORIES_OPTION                          .startsWith(nextWord)) configuration.keepDirectories                  = parseCommaSeparatedList("directory name", true, true, false, true, false, true, false, false, configuration.keepDirectories);

            else if (ConfigurationConstants.DONT_SHRINK_OPTION                               .startsWith(nextWord)) configuration.shrink                           = parseNoArgument(false);
            else if (ConfigurationConstants.PRINT_USAGE_OPTION                               .startsWith(nextWord)) configuration.printUsage                       = parseOptionalFile();
            else if (ConfigurationConstants.WHY_ARE_YOU_KEEPING_OPTION                       .startsWith(nextWord)) configuration.whyAreYouKeeping                 = parseClassSpecificationArguments(configuration.whyAreYouKeeping);

            else if (ConfigurationConstants.DONT_OPTIMIZE_OPTION                             .startsWith(nextWord)) configuration.optimize                         = parseNoArgument(false);
            else if (ConfigurationConstants.OPTIMIZATION_PASSES                              .startsWith(nextWord)) configuration.optimizationPasses               = parseIntegerArgument();
            else if (ConfigurationConstants.OPTIMIZATIONS                                    .startsWith(nextWord)) configuration.optimizations                    = parseCommaSeparatedList("optimization name", true, false, false, false, false, false, false, false, configuration.optimizations);
            else if (ConfigurationConstants.ASSUME_NO_SIDE_EFFECTS_OPTION                    .startsWith(nextWord)) configuration.assumeNoSideEffects              = parseClassSpecificationArguments(configuration.assumeNoSideEffects);
            else if (ConfigurationConstants.ALLOW_ACCESS_MODIFICATION_OPTION                 .startsWith(nextWord)) configuration.allowAccessModification          = parseNoArgument(true);
            else if (ConfigurationConstants.MERGE_INTERFACES_AGGRESSIVELY_OPTION             .startsWith(nextWord)) configuration.mergeInterfacesAggressively      = parseNoArgument(true);

            else if (ConfigurationConstants.DONT_OBFUSCATE_OPTION                            .startsWith(nextWord)) configuration.obfuscate                        = parseNoArgument(false);
            else if (ConfigurationConstants.PRINT_MAPPING_OPTION                             .startsWith(nextWord)) configuration.printMapping                     = parseOptionalFile();
            else if (ConfigurationConstants.APPLY_MAPPING_OPTION                             .startsWith(nextWord)) configuration.applyMapping                     = parseFile();
            else if (ConfigurationConstants.OBFUSCATION_DICTIONARY_OPTION                    .startsWith(nextWord)) configuration.obfuscationDictionary            = parseFile();
            else if (ConfigurationConstants.CLASS_OBFUSCATION_DICTIONARY_OPTION              .startsWith(nextWord)) configuration.classObfuscationDictionary       = parseFile();
            else if (ConfigurationConstants.PACKAGE_OBFUSCATION_DICTIONARY_OPTION            .startsWith(nextWord)) configuration.packageObfuscationDictionary     = parseFile();
            else if (ConfigurationConstants.OVERLOAD_AGGRESSIVELY_OPTION                     .startsWith(nextWord)) configuration.overloadAggressively             = parseNoArgument(true);
            else if (ConfigurationConstants.USE_UNIQUE_CLASS_MEMBER_NAMES_OPTION             .startsWith(nextWord)) configuration.useUniqueClassMemberNames        = parseNoArgument(true);
            else if (ConfigurationConstants.DONT_USE_MIXED_CASE_CLASS_NAMES_OPTION           .startsWith(nextWord)) configuration.useMixedCaseClassNames           = parseNoArgument(false);
            else if (ConfigurationConstants.KEEP_PACKAGE_NAMES_OPTION                        .startsWith(nextWord)) configuration.keepPackageNames                 = parseCommaSeparatedList("package name", true, true, false, false, true, false, true, false, configuration.keepPackageNames);
            else if (ConfigurationConstants.FLATTEN_PACKAGE_HIERARCHY_OPTION                 .startsWith(nextWord)) configuration.flattenPackageHierarchy          = ClassUtil.internalClassName(parseOptionalArgument());
            else if (ConfigurationConstants.REPACKAGE_CLASSES_OPTION                         .startsWith(nextWord)) configuration.repackageClasses                 = ClassUtil.internalClassName(parseOptionalArgument());
            else if (ConfigurationConstants.DEFAULT_PACKAGE_OPTION                           .startsWith(nextWord)) configuration.repackageClasses                 = ClassUtil.internalClassName(parseOptionalArgument());
            else if (ConfigurationConstants.KEEP_ATTRIBUTES_OPTION                           .startsWith(nextWord)) configuration.keepAttributes                   = parseCommaSeparatedList("attribute name", true, true, false, false, true, false, false, false, configuration.keepAttributes);
            else if (ConfigurationConstants.KEEP_PARAMETER_NAMES_OPTION                      .startsWith(nextWord)) configuration.keepParameterNames               = parseNoArgument(true);
            else if (ConfigurationConstants.RENAME_SOURCE_FILE_ATTRIBUTE_OPTION              .startsWith(nextWord)) configuration.newSourceFileAttribute           = parseOptionalArgument();
            else if (ConfigurationConstants.ADAPT_CLASS_STRINGS_OPTION                       .startsWith(nextWord)) configuration.adaptClassStrings                = parseCommaSeparatedList("class name", true, true, false, false, true, false, true, false, configuration.adaptClassStrings);
            else if (ConfigurationConstants.ADAPT_RESOURCE_FILE_NAMES_OPTION                 .startsWith(nextWord)) configuration.adaptResourceFileNames           = parseCommaSeparatedList("resource file name", true, true, false, true, false, false, false, false, configuration.adaptResourceFileNames);
            else if (ConfigurationConstants.ADAPT_RESOURCE_FILE_CONTENTS_OPTION              .startsWith(nextWord)) configuration.adaptResourceFileContents        = parseCommaSeparatedList("resource file name", true, true, false, true, false, false, false, false, configuration.adaptResourceFileContents);

            else if (ConfigurationConstants.DONT_PREVERIFY_OPTION                            .startsWith(nextWord)) configuration.preverify                        = parseNoArgument(false);
            else if (ConfigurationConstants.MICRO_EDITION_OPTION                             .startsWith(nextWord)) configuration.microEdition                     = parseNoArgument(true);

            else if (ConfigurationConstants.VERBOSE_OPTION                                   .startsWith(nextWord)) configuration.verbose                          = parseNoArgument(true);
            else if (ConfigurationConstants.DONT_NOTE_OPTION                                 .startsWith(nextWord)) configuration.note                             = parseCommaSeparatedList("class name", true, true, false, false, true, false, true, false, configuration.note);
            else if (ConfigurationConstants.DONT_WARN_OPTION                                 .startsWith(nextWord)) configuration.warn                             = parseCommaSeparatedList("class name", true, true, false, false, true, false, true, false, configuration.warn);
            else if (ConfigurationConstants.IGNORE_WARNINGS_OPTION                           .startsWith(nextWord)) configuration.ignoreWarnings                   = parseNoArgument(true);
            else if (ConfigurationConstants.PRINT_CONFIGURATION_OPTION                       .startsWith(nextWord)) configuration.printConfiguration               = parseOptionalFile();
            else if (ConfigurationConstants.DUMP_OPTION                                      .startsWith(nextWord)) configuration.dump                             = parseOptionalFile();
            else
            {
                throw new ParseException("Unknown option " + reader.locationDescription());
            }
        }
    }



    /**
     * Closes the configuration.
     * @throws IOException if an IO error occurs while closing the configuration.
     */
    public void close() throws IOException
    {
        if (reader != null)
        {
            reader.close();
        }
    }


    private long parseIncludeArgument(long lastModified) throws ParseException, IOException
    {
        // Read the configuration file name.
        readNextWord("configuration file name", true, false);

        File file = file(nextWord);
        reader.includeWordReader(new FileWordReader(file));

        readNextWord();

        return Math.max(lastModified, file.lastModified());
    }


    private void parseBaseDirectoryArgument() throws ParseException, IOException
    {
        // Read the base directory name.
        readNextWord("base directory name", true, false);

        reader.setBaseDir(file(nextWord));

        readNextWord();
    }


    private ClassPath parseClassPathArgument(ClassPath classPath,
                                             boolean   isOutput)
    throws ParseException, IOException
    {
        // Create a new List if necessary.
        if (classPath == null)
        {
            classPath = new ClassPath();
        }

        while (true)
        {
            // Read the next jar name.
            readNextWord("jar or directory name", true, false);

            // Create a new class path entry.
            ClassPathEntry entry = new ClassPathEntry(file(nextWord), isOutput);

            // Read the opening parenthesis or the separator, if any.
            readNextWord();

            // Read the optional filters.
            if (!configurationEnd() &&
                ConfigurationConstants.OPEN_ARGUMENTS_KEYWORD.equals(nextWord))
            {
                // Read all filters in an array.
                List[] filters = new List[7];

                int counter = 0;
                do
                {
                    // Read the filter.
                    filters[counter++] =
                        parseCommaSeparatedList("filter", true, true, true, true, false, true, false, false, null);
                }
                while (counter < filters.length &&
                       ConfigurationConstants.SEPARATOR_KEYWORD.equals(nextWord));

                // Make sure there is a closing parenthesis.
                if (!ConfigurationConstants.CLOSE_ARGUMENTS_KEYWORD.equals(nextWord))
                {
                    throw new ParseException("Expecting separating '" + ConfigurationConstants.ARGUMENT_SEPARATOR_KEYWORD +
                                             "' or '" + ConfigurationConstants.SEPARATOR_KEYWORD +
                                             "', or closing '" + ConfigurationConstants.CLOSE_ARGUMENTS_KEYWORD +
                                             "' before " + reader.locationDescription());
                }

                // Set all filters from the array on the entry.
                entry.setFilter(filters[--counter]);
                if (counter > 0)
                {
                    entry.setJarFilter(filters[--counter]);
                    if (counter > 0)
                    {
                        entry.setWarFilter(filters[--counter]);
                        if (counter > 0)
                        {
                            entry.setEarFilter(filters[--counter]);
                            if (counter > 0)
                            {
                                entry.setZipFilter(filters[--counter]);
                                if (counter > 0)
                                {
                                    // For backward compatibility, the apk
                                    // filter comes second in the list.
                                    entry.setApkFilter(filters[--counter]);
                                    if (counter > 0)
                                    {
                                        // For backward compatibility, the aar
                                        // filter comes first in the list.
                                        entry.setAarFilter(filters[--counter]);
                                    }
                                }
                            }
                        }
                    }
                }

                // Read the separator, if any.
                readNextWord();
            }

            // Add the entry to the list.
            classPath.add(entry);

            if (configurationEnd())
            {
                return classPath;
            }

            if (!nextWord.equals(ConfigurationConstants.JAR_SEPARATOR_KEYWORD))
            {
                throw new ParseException("Expecting class path separator '" + ConfigurationConstants.JAR_SEPARATOR_KEYWORD +
                                         "' before " + reader.locationDescription());
            }
        }
    }


    private int parseClassVersion()
    throws ParseException, IOException
    {
        // Read the obligatory target.
        readNextWord("java version");

        int classVersion = ClassUtil.internalClassVersion(nextWord);
        if (classVersion == 0)
        {
            throw new ParseException("Unsupported java version " + reader.locationDescription());
        }

        readNextWord();

        return classVersion;
    }


    private int parseIntegerArgument()
    throws ParseException, IOException
    {
        try
        {
            // Read the obligatory integer.
            readNextWord("integer");

            int integer = Integer.parseInt(nextWord);

            readNextWord();

            return integer;
        }
        catch (NumberFormatException e)
        {
            throw new ParseException("Expecting integer argument instead of '" + nextWord +
                                     "' before " + reader.locationDescription());
        }
    }


    private File parseFile()
    throws ParseException, IOException
    {
        // Read the obligatory file name.
        readNextWord("file name", true, false);

        // Make sure the file is properly resolved.
        File file = file(nextWord);

        readNextWord();

        return file;
    }


    private File parseOptionalFile()
    throws ParseException, IOException
    {
        // Read the optional file name.
        readNextWord(true);

        // Didn't the user specify a file name?
        if (configurationEnd())
        {
            return Configuration.STD_OUT;
        }

        // Make sure the file is properly resolved.
        File file = file(nextWord);

        readNextWord();

        return file;
    }


    private String parseOptionalArgument() throws IOException
    {
        // Read the optional argument.
        readNextWord();

        // Didn't the user specify an argument?
        if (configurationEnd())
        {
            return "";
        }

        String argument = nextWord;

        readNextWord();

        return argument;
    }


    private boolean parseNoArgument(boolean value) throws IOException
    {
        readNextWord();

        return value;
    }


    private long parseNoArgument(long value) throws IOException
    {
        readNextWord();

        return value;
    }


    private List parseKeepClassSpecificationArguments(List    keepClassSpecifications,
                                                      boolean markClasses,
                                                      boolean markConditionally,
                                                      boolean allowShrinking)
    throws ParseException, IOException
    {
        // Create a new List if necessary.
        if (keepClassSpecifications == null)
        {
            keepClassSpecifications = new ArrayList();
        }

        boolean markDescriptorClasses = false;
        //boolean allowShrinking        = false;
        boolean allowOptimization     = false;
        boolean allowObfuscation      = false;

        // Read the keep modifiers.
        while (true)
        {
            readNextWord("keyword '" + ConfigurationConstants.CLASS_KEYWORD +
                         "', '"      + JavaConstants.ACC_INTERFACE +
                         "', or '"   + JavaConstants.ACC_ENUM + "'",
                         false, true);

            if (!ConfigurationConstants.ARGUMENT_SEPARATOR_KEYWORD.equals(nextWord))
            {
                // Not a comma. Stop parsing the keep modifiers.
                break;
            }

            readNextWord("keyword '" + ConfigurationConstants.ALLOW_SHRINKING_SUBOPTION +
                         "', '"      + ConfigurationConstants.ALLOW_OPTIMIZATION_SUBOPTION +
                         "', or '"   + ConfigurationConstants.ALLOW_OBFUSCATION_SUBOPTION + "'");

            if      (ConfigurationConstants.INCLUDE_DESCRIPTOR_CLASSES_SUBOPTION.startsWith(nextWord))
            {
                markDescriptorClasses = true;
            }
            else if (ConfigurationConstants.ALLOW_SHRINKING_SUBOPTION           .startsWith(nextWord))
            {
                allowShrinking        = true;
            }
            else if (ConfigurationConstants.ALLOW_OPTIMIZATION_SUBOPTION        .startsWith(nextWord))
            {
                allowOptimization     = true;
            }
            else if (ConfigurationConstants.ALLOW_OBFUSCATION_SUBOPTION         .startsWith(nextWord))
            {
                allowObfuscation      = true;
            }
            else
            {
                throw new ParseException("Expecting keyword '" + ConfigurationConstants.INCLUDE_DESCRIPTOR_CLASSES_SUBOPTION +
                                         "', '"                + ConfigurationConstants.ALLOW_SHRINKING_SUBOPTION +
                                         "', '"                + ConfigurationConstants.ALLOW_OPTIMIZATION_SUBOPTION +
                                         "', or '"             + ConfigurationConstants.ALLOW_OBFUSCATION_SUBOPTION +
                                         "' before " + reader.locationDescription());
            }
        }

        // Read the class configuration.
        ClassSpecification classSpecification =
            parseClassSpecificationArguments();

        // Create and add the keep configuration.
        keepClassSpecifications.add(new KeepClassSpecification(markClasses,
                                                               markConditionally,
                                                               markDescriptorClasses,
                                                               allowShrinking,
                                                               allowOptimization,
                                                               allowObfuscation,
                                                               classSpecification));
        return keepClassSpecifications;
    }


    private List parseClassSpecificationArguments(List classSpecifications)
    throws ParseException, IOException
    {
        // Create a new List if necessary.
        if (classSpecifications == null)
        {
            classSpecifications = new ArrayList();
        }

        // Read and add the class configuration.
        readNextWord("keyword '" + ConfigurationConstants.CLASS_KEYWORD +
                     "', '"      + JavaConstants.ACC_INTERFACE +
                     "', or '"   + JavaConstants.ACC_ENUM + "'",
                     false, true);

        classSpecifications.add(parseClassSpecificationArguments());

        return classSpecifications;
    }


    /**
     * Parses and returns a class specification.
     * @throws ParseException if the class specification contains a syntax error.
     * @throws IOException    if an IO error occurs while reading the class
     *                        specification.
     */
    public ClassSpecification parseClassSpecificationArguments()
    throws ParseException, IOException
    {
        // Clear the annotation type.
        String annotationType = null;

        // Clear the class access modifiers.
        int requiredSetClassAccessFlags   = 0;
        int requiredUnsetClassAccessFlags = 0;

        // Parse the class annotations and access modifiers until the class keyword.
        while (!ConfigurationConstants.CLASS_KEYWORD.equals(nextWord))
        {
            // Strip the negating sign, if any.
            boolean negated =
                nextWord.startsWith(ConfigurationConstants.NEGATOR_KEYWORD);

            String strippedWord = negated ?
                nextWord.substring(1) :
                nextWord;

            // Parse the class access modifiers.
            int accessFlag =
                strippedWord.equals(JavaConstants.ACC_PUBLIC)     ? ClassConstants.ACC_PUBLIC      :
                strippedWord.equals(JavaConstants.ACC_FINAL)      ? ClassConstants.ACC_FINAL       :
                strippedWord.equals(JavaConstants.ACC_INTERFACE)  ? ClassConstants.ACC_INTERFACE   :
                strippedWord.equals(JavaConstants.ACC_ABSTRACT)   ? ClassConstants.ACC_ABSTRACT    :
                strippedWord.equals(JavaConstants.ACC_SYNTHETIC)  ? ClassConstants.ACC_SYNTHETIC   :
                strippedWord.equals(JavaConstants.ACC_ANNOTATION) ? ClassConstants.ACC_ANNOTATTION :
                strippedWord.equals(JavaConstants.ACC_ENUM)       ? ClassConstants.ACC_ENUM        :
                                                                    unknownAccessFlag();

            // Is it an annotation modifier?
            if (accessFlag == ClassConstants.ACC_ANNOTATTION)
            {
                // Already read the next word.
                readNextWord("annotation type or keyword '" + JavaConstants.ACC_INTERFACE + "'",
                             false, false);

                // Is the next word actually an annotation type?
                if (!nextWord.equals(JavaConstants.ACC_INTERFACE) &&
                    !nextWord.equals(JavaConstants.ACC_ENUM)      &&
                    !nextWord.equals(ConfigurationConstants.CLASS_KEYWORD))
                {
                    // Parse the annotation type.
                    annotationType =
                        ListUtil.commaSeparatedString(
                        parseCommaSeparatedList("annotation type",
                                                false, false, false, false, true, false, false, true, null), false);

                    // Continue parsing the access modifier that we just read
                    // in the next cycle.
                    continue;
                }

                // Otherwise just handle the annotation modifier.
            }

            if (!negated)
            {
                requiredSetClassAccessFlags   |= accessFlag;
            }
            else
            {
                requiredUnsetClassAccessFlags |= accessFlag;
            }

            if ((requiredSetClassAccessFlags &
                 requiredUnsetClassAccessFlags) != 0)
            {
                throw new ParseException("Conflicting class access modifiers for '" + strippedWord +
                                         "' before " + reader.locationDescription());
            }

            if (strippedWord.equals(JavaConstants.ACC_INTERFACE) ||
                strippedWord.equals(JavaConstants.ACC_ENUM)      ||
                strippedWord.equals(ConfigurationConstants.CLASS_KEYWORD))
            {
                // The interface or enum keyword. Stop parsing the class flags.
                break;
            }

            // Should we read the next word?
            if (accessFlag != ClassConstants.ACC_ANNOTATTION)
            {
                readNextWord("keyword '" + ConfigurationConstants.CLASS_KEYWORD +
                             "', '"      + JavaConstants.ACC_INTERFACE +
                             "', or '"   + JavaConstants.ACC_ENUM + "'",
                             false, true);
            }
        }

       // Parse the class name part.
        String externalClassName =
            ListUtil.commaSeparatedString(
            parseCommaSeparatedList("class name or interface name",
                                    true, false, false, false, true, false, false, false, null), false);

        // For backward compatibility, allow a single "*" wildcard to match any
        // class.
        String className = ConfigurationConstants.ANY_CLASS_KEYWORD.equals(externalClassName) ?
            null :
            ClassUtil.internalClassName(externalClassName);

        // Clear the annotation type and the class name of the extends part.
        String extendsAnnotationType = null;
        String extendsClassName      = null;

        if (!configurationEnd())
        {
            // Parse 'implements ...' or 'extends ...' part, if any.
            if (ConfigurationConstants.IMPLEMENTS_KEYWORD.equals(nextWord) ||
                ConfigurationConstants.EXTENDS_KEYWORD.equals(nextWord))
            {
                readNextWord("class name or interface name", false, true);

                // Parse the annotation type, if any.
                if (ConfigurationConstants.ANNOTATION_KEYWORD.equals(nextWord))
                {
                    extendsAnnotationType =
                        ListUtil.commaSeparatedString(
                        parseCommaSeparatedList("annotation type",
                                                true, false, false, false, true, false, false, true, null), false);
                }

                String externalExtendsClassName =
                    ListUtil.commaSeparatedString(
                    parseCommaSeparatedList("class name or interface name",
                                            false, false, false, false, true, false, false, false, null), false);

                extendsClassName = ConfigurationConstants.ANY_CLASS_KEYWORD.equals(externalExtendsClassName) ?
                    null :
                    ClassUtil.internalClassName(externalExtendsClassName);
            }
        }

        // Create the basic class specification.
        ClassSpecification classSpecification =
            new ClassSpecification(lastComments,
                                   requiredSetClassAccessFlags,
                                   requiredUnsetClassAccessFlags,
                                   annotationType,
                                   className,
                                   extendsAnnotationType,
                                   extendsClassName);


        // Now add any class members to this class specification.
        if (!configurationEnd())
        {
            // Check the class member opening part.
            if (!ConfigurationConstants.OPEN_KEYWORD.equals(nextWord))
            {
                throw new ParseException("Expecting opening '" + ConfigurationConstants.OPEN_KEYWORD +
                                         "' at " + reader.locationDescription());
            }

            // Parse all class members.
            while (true)
            {
                readNextWord("class member description" +
                             " or closing '" + ConfigurationConstants.CLOSE_KEYWORD + "'",
                             false, true);

                if (nextWord.equals(ConfigurationConstants.CLOSE_KEYWORD))
                {
                    // The closing brace. Stop parsing the class members.
                    readNextWord();

                    break;
                }

                parseMemberSpecificationArguments(externalClassName,
                                                  classSpecification);
            }
        }

        return classSpecification;
    }


    private void parseMemberSpecificationArguments(String             externalClassName,
                                                   ClassSpecification classSpecification)
    throws ParseException, IOException
    {
        // Clear the annotation name.
        String annotationType = null;

        // Parse the class member access modifiers, if any.
        int requiredSetMemberAccessFlags   = 0;
        int requiredUnsetMemberAccessFlags = 0;

        while (!configurationEnd(true))
        {
            // Parse the annotation type, if any.
            if (ConfigurationConstants.ANNOTATION_KEYWORD.equals(nextWord))
            {
                annotationType =
                    ListUtil.commaSeparatedString(
                    parseCommaSeparatedList("annotation type",
                                            true, false, false, false, true, false, false, true, null), false);
                continue;
            }

            String strippedWord = nextWord.startsWith("!") ?
                nextWord.substring(1) :
                nextWord;

            // Parse the class member access modifiers.
            int accessFlag =
                strippedWord.equals(JavaConstants.ACC_PUBLIC)       ? ClassConstants.ACC_PUBLIC       :
                strippedWord.equals(JavaConstants.ACC_PRIVATE)      ? ClassConstants.ACC_PRIVATE      :
                strippedWord.equals(JavaConstants.ACC_PROTECTED)    ? ClassConstants.ACC_PROTECTED    :
                strippedWord.equals(JavaConstants.ACC_STATIC)       ? ClassConstants.ACC_STATIC       :
                strippedWord.equals(JavaConstants.ACC_FINAL)        ? ClassConstants.ACC_FINAL        :
                strippedWord.equals(JavaConstants.ACC_SYNCHRONIZED) ? ClassConstants.ACC_SYNCHRONIZED :
                strippedWord.equals(JavaConstants.ACC_VOLATILE)     ? ClassConstants.ACC_VOLATILE     :
                strippedWord.equals(JavaConstants.ACC_TRANSIENT)    ? ClassConstants.ACC_TRANSIENT    :
                strippedWord.equals(JavaConstants.ACC_BRIDGE)       ? ClassConstants.ACC_BRIDGE       :
                strippedWord.equals(JavaConstants.ACC_VARARGS)      ? ClassConstants.ACC_VARARGS      :
                strippedWord.equals(JavaConstants.ACC_NATIVE)       ? ClassConstants.ACC_NATIVE       :
                strippedWord.equals(JavaConstants.ACC_ABSTRACT)     ? ClassConstants.ACC_ABSTRACT     :
                strippedWord.equals(JavaConstants.ACC_STRICT)       ? ClassConstants.ACC_STRICT       :
                strippedWord.equals(JavaConstants.ACC_SYNTHETIC)    ? ClassConstants.ACC_SYNTHETIC    :
                                                                      0;
            if (accessFlag == 0)
            {
                // Not a class member access modifier. Stop parsing them.
                break;
            }

            if (strippedWord.equals(nextWord))
            {
                requiredSetMemberAccessFlags   |= accessFlag;
            }
            else
            {
                requiredUnsetMemberAccessFlags |= accessFlag;
            }

            // Make sure the user doesn't try to set and unset the same
            // access flags simultaneously.
            if ((requiredSetMemberAccessFlags &
                 requiredUnsetMemberAccessFlags) != 0)
            {
                throw new ParseException("Conflicting class member access modifiers for " +
                                         reader.locationDescription());
            }

            readNextWord("class member description");
        }

        // Parse the class member type and name part.

        // Did we get a special wildcard?
        if (ConfigurationConstants.ANY_CLASS_MEMBER_KEYWORD.equals(nextWord) ||
            ConfigurationConstants.ANY_FIELD_KEYWORD       .equals(nextWord) ||
            ConfigurationConstants.ANY_METHOD_KEYWORD      .equals(nextWord))
        {
            // Act according to the type of wildcard..
            if (ConfigurationConstants.ANY_CLASS_MEMBER_KEYWORD.equals(nextWord))
            {
                checkFieldAccessFlags(requiredSetMemberAccessFlags,
                                      requiredUnsetMemberAccessFlags);
                checkMethodAccessFlags(requiredSetMemberAccessFlags,
                                       requiredUnsetMemberAccessFlags);

                classSpecification.addField(
                    new MemberSpecification(requiredSetMemberAccessFlags,
                                            requiredUnsetMemberAccessFlags,
                                            annotationType,
                                            null,
                                            null));
                classSpecification.addMethod(
                    new MemberSpecification(requiredSetMemberAccessFlags,
                                            requiredUnsetMemberAccessFlags,
                                            annotationType,
                                            null,
                                            null));
            }
            else if (ConfigurationConstants.ANY_FIELD_KEYWORD.equals(nextWord))
            {
                checkFieldAccessFlags(requiredSetMemberAccessFlags,
                                      requiredUnsetMemberAccessFlags);

                classSpecification.addField(
                    new MemberSpecification(requiredSetMemberAccessFlags,
                                            requiredUnsetMemberAccessFlags,
                                            annotationType,
                                            null,
                                            null));
            }
            else if (ConfigurationConstants.ANY_METHOD_KEYWORD.equals(nextWord))
            {
                checkMethodAccessFlags(requiredSetMemberAccessFlags,
                                       requiredUnsetMemberAccessFlags);

                classSpecification.addMethod(
                    new MemberSpecification(requiredSetMemberAccessFlags,
                                            requiredUnsetMemberAccessFlags,
                                            annotationType,
                                            null,
                                            null));
            }

            // We still have to read the closing separator.
            readNextWord("separator '" + ConfigurationConstants.SEPARATOR_KEYWORD + "'");

            if (!ConfigurationConstants.SEPARATOR_KEYWORD.equals(nextWord))
            {
                throw new ParseException("Expecting separator '" + ConfigurationConstants.SEPARATOR_KEYWORD +
                                         "' before " + reader.locationDescription());
            }
        }
        else
        {
            // Make sure we have a proper type.
            checkJavaIdentifier("java type");
            String type = nextWord;

            readNextWord("class member name");
            String name = nextWord;

            // Did we get just one word before the opening parenthesis?
            if (ConfigurationConstants.OPEN_ARGUMENTS_KEYWORD.equals(name))
            {
                // This must be a constructor then.
                // Make sure the type is a proper constructor name.
                if (!(type.equals(ClassConstants.METHOD_NAME_INIT) ||
                      type.equals(externalClassName) ||
                      type.equals(ClassUtil.externalShortClassName(externalClassName))))
                {
                    throw new ParseException("Expecting type and name " +
                                             "instead of just '" + type +
                                             "' before " + reader.locationDescription());
                }

                // Assign the fixed constructor type and name.
                type = JavaConstants.TYPE_VOID;
                name = ClassConstants.METHOD_NAME_INIT;
            }
            else
            {
                // It's not a constructor.
                // Make sure we have a proper name.
                checkJavaIdentifier("class member name");

                // Read the opening parenthesis or the separating
                // semi-colon.
                readNextWord("opening '" + ConfigurationConstants.OPEN_ARGUMENTS_KEYWORD +
                             "' or separator '" + ConfigurationConstants.SEPARATOR_KEYWORD + "'");
            }

            // Are we looking at a field, a method, or something else?
            if (ConfigurationConstants.SEPARATOR_KEYWORD.equals(nextWord))
            {
                // It's a field.
                checkFieldAccessFlags(requiredSetMemberAccessFlags,
                                      requiredUnsetMemberAccessFlags);

                // We already have a field descriptor.
                String descriptor = ClassUtil.internalType(type);

                // Add the field.
                classSpecification.addField(
                    new MemberSpecification(requiredSetMemberAccessFlags,
                                            requiredUnsetMemberAccessFlags,
                                            annotationType,
                                            name,
                                            descriptor));
            }
            else if (ConfigurationConstants.OPEN_ARGUMENTS_KEYWORD.equals(nextWord))
            {
                // It's a method.
                checkMethodAccessFlags(requiredSetMemberAccessFlags,
                                       requiredUnsetMemberAccessFlags);

                // Parse the method arguments.
                String descriptor =
                    ClassUtil.internalMethodDescriptor(type,
                                                       parseCommaSeparatedList("argument", true, true, true, false, true, false, false, false, null));

                if (!ConfigurationConstants.CLOSE_ARGUMENTS_KEYWORD.equals(nextWord))
                {
                    throw new ParseException("Expecting separating '" + ConfigurationConstants.ARGUMENT_SEPARATOR_KEYWORD +
                                             "' or closing '" + ConfigurationConstants.CLOSE_ARGUMENTS_KEYWORD +
                                             "' before " + reader.locationDescription());
                }

                // Read the separator after the closing parenthesis.
                readNextWord("separator '" + ConfigurationConstants.SEPARATOR_KEYWORD + "'");

                if (!ConfigurationConstants.SEPARATOR_KEYWORD.equals(nextWord))
                {
                    throw new ParseException("Expecting separator '" + ConfigurationConstants.SEPARATOR_KEYWORD +
                                             "' before " + reader.locationDescription());
                }

                // Add the method.
                classSpecification.addMethod(
                    new MemberSpecification(requiredSetMemberAccessFlags,
                                            requiredUnsetMemberAccessFlags,
                                            annotationType,
                                            name,
                                            descriptor));
            }
            else
            {
                // It doesn't look like a field or a method.
                throw new ParseException("Expecting opening '" + ConfigurationConstants.OPEN_ARGUMENTS_KEYWORD +
                                         "' or separator '" + ConfigurationConstants.SEPARATOR_KEYWORD +
                                         "' before " + reader.locationDescription());
            }
        }
    }


    /**
     * Reads a comma-separated list of java identifiers or of file names.
     * Examples of invocation arguments:
     *   ("directory name", true,  true,  false, true,  false, true,  false, false, ...)
     *   ("optimization",   true,  false, false, false, false, false, false, false, ...)
     *   ("package name",   true,  true,  false, false, true,  false, true,  false, ...)
     *   ("attribute name", true,  true,  false, false, true,  false, false, false, ...)
     *   ("class name",     true,  true,  false, false, true,  false, true,  false, ...)
     *   ("resource file",  true,  true,  false, true,  false, false, false, false, ...)
     *   ("resource file",  true,  true,  false, true,  false, false, false, false, ...)
     *   ("class name",     true,  true,  false, false, true,  false, true,  false, ...)
     *   ("class name",     true,  true,  false, false, true,  false, true,  false, ...)
     *   ("filter",         true,  true,  true,  true,  false, true,  false, false, ...)
     *   ("annotation ",    false, false, false, false, true,  false, false, true,  ...)
     *   ("class name ",    true,  false, false, false, true,  false, false, false, ...)
     *   ("annotation ",    true,  false, false, false, true,  false, false, true,  ...)
     *   ("class name ",    false, false, false, false, true,  false, false, false, ...)
     *   ("annotation ",    true,  false, false, false, true,  false, false, true,  ...)
     *   ("argument",       true,  true,  true,  false, true,  false, false, false, ...)
     */
    private List parseCommaSeparatedList(String  expectedDescription,
                                         boolean readFirstWord,
                                         boolean allowEmptyList,
                                         boolean expectClosingParenthesis,
                                         boolean isFileName,
                                         boolean checkJavaIdentifiers,
                                         boolean replaceSystemProperties,
                                         boolean replaceExternalClassNames,
                                         boolean replaceExternalTypes,
                                         List    list)
    throws ParseException, IOException
    {
        if (list == null)
        {
            list = new ArrayList();
        }

        if (readFirstWord)
        {
            if (!allowEmptyList)
            {
                // Read the first list entry.
                readNextWord(expectedDescription, isFileName, false);
            }
            else if (expectClosingParenthesis)
            {
                // Read the first list entry.
                readNextWord(expectedDescription, isFileName, false);

                // Return if the entry is actually empty (an empty file name or
                // a closing parenthesis).
                if (nextWord.length() == 0)
                {
                    // Read the closing parenthesis
                    readNextWord("closing '" + ConfigurationConstants.CLOSE_ARGUMENTS_KEYWORD +
                                 "'");

                    return list;
                }
                else if (nextWord.equals(ConfigurationConstants.CLOSE_ARGUMENTS_KEYWORD))
                {
                    return list;
                }
            }
            else
            {
                // Read the first list entry, if there is any.
                readNextWord(isFileName);

                // Check if the list is empty.
                if (configurationEnd())
                {
                    return list;
                }
            }
        }

        while (true)
        {
            if (checkJavaIdentifiers)
            {
                checkJavaIdentifier("java type");
            }

            if (replaceSystemProperties)
            {
                nextWord = replaceSystemProperties(nextWord);
            }

            if (replaceExternalClassNames)
            {
                nextWord = ClassUtil.internalClassName(nextWord);
            }

            if (replaceExternalTypes)
            {
                nextWord = ClassUtil.internalType(nextWord);
            }

            list.add(nextWord);

            if (expectClosingParenthesis)
            {
                // Read a comma (or a closing parenthesis, or a different word).
                readNextWord("separating '" + ConfigurationConstants.ARGUMENT_SEPARATOR_KEYWORD +
                             "' or closing '" + ConfigurationConstants.CLOSE_ARGUMENTS_KEYWORD +
                             "'");
            }
            else
            {
                // Read a comma (or a different word).
                readNextWord();
            }

            if (!ConfigurationConstants.ARGUMENT_SEPARATOR_KEYWORD.equals(nextWord))
            {
                return list;
            }

            // Read the next list entry.
            readNextWord(expectedDescription, isFileName, false);
        }
    }


    /**
     * Throws a ParseException for an unexpected keyword.
     */
    private int unknownAccessFlag() throws ParseException
    {
        throw new ParseException("Unexpected keyword " + reader.locationDescription());
    }


    /**
     * Creates a properly resolved File, based on the given word.
     */
    private File file(String word) throws ParseException
    {
        String fileName = replaceSystemProperties(word);
        File   file     = new File(fileName);

        // Try to get an absolute file.
        if (!file.isAbsolute())
        {
            file = new File(reader.getBaseDir(), fileName);
        }

        return file;
    }


    /**
     * Replaces any properties in the given word by their values.
     * For instance, the substring "<java.home>" is replaced by its value.
     */
    private String replaceSystemProperties(String word) throws ParseException
    {
        int fromIndex = 0;
        while (true)
        {
            fromIndex = word.indexOf(ConfigurationConstants.OPEN_SYSTEM_PROPERTY, fromIndex);
            if (fromIndex < 0)
            {
                break;
            }

            int toIndex = word.indexOf(ConfigurationConstants.CLOSE_SYSTEM_PROPERTY, fromIndex+1);
            if (toIndex < 0)
            {
                break;
            }

            String propertyName  = word.substring(fromIndex+1, toIndex);
            String propertyValue = properties.getProperty(propertyName);
            if (propertyValue == null)
            {
                throw new ParseException("Value of system property '" + propertyName +
                                         "' is undefined in " + reader.locationDescription());
            }

            word = word.substring(0, fromIndex) + propertyValue + word.substring(toIndex+1);

            fromIndex += propertyValue.length();
        }

        return word;
    }


    /**
     * Reads the next word of the configuration in the 'nextWord' field,
     * throwing an exception if there is no next word.
     */
    private void readNextWord(String expectedDescription)
    throws ParseException, IOException
    {
        readNextWord(expectedDescription, false, false);
    }


    /**
     * Reads the next word of the configuration in the 'nextWord' field,
     * throwing an exception if there is no next word.
     */
    private void readNextWord(String  expectedDescription,
                              boolean isFileName,
                              boolean expectingAtCharacter)
    throws ParseException, IOException
    {
        readNextWord(isFileName);
        if (configurationEnd(expectingAtCharacter))
        {
            throw new ParseException("Expecting " + expectedDescription +
                                     " before " + reader.locationDescription());
        }
    }


    /**
     * Reads the next word of the configuration in the 'nextWord' field.
     */
    private void readNextWord() throws IOException
    {
        readNextWord(false);
    }


    /**
     * Reads the next word of the configuration in the 'nextWord' field.
     */
    private void readNextWord(boolean isFileName) throws IOException
    {
        nextWord = reader.nextWord(isFileName);
    }


    /**
     * Returns whether the end of the configuration has been reached.
     */
    private boolean configurationEnd()
    {
        return configurationEnd(false);
    }


    /**
     * Returns whether the end of the configuration has been reached.
     */
    private boolean configurationEnd(boolean expectingAtCharacter)
    {
        return nextWord == null ||
               nextWord.startsWith(ConfigurationConstants.OPTION_PREFIX) ||
               (!expectingAtCharacter &&
                nextWord.equals(ConfigurationConstants.AT_DIRECTIVE));
    }


    /**
     * Checks whether the given word is a valid Java identifier and throws
     * a ParseException if it isn't. Wildcard characters are accepted.
     */
    private void checkJavaIdentifier(String expectedDescription)
    throws ParseException
    {
        if (!isJavaIdentifier(nextWord))
        {
            throw new ParseException("Expecting " + expectedDescription +
                                     " before " + reader.locationDescription());
        }
    }


    /**
     * Returns whether the given word is a valid Java identifier.
     * Wildcard characters are accepted.
     */
    private boolean isJavaIdentifier(String aWord)
    {
        if (aWord.length() == 0)
        {
            return false;
        }

        for (int index = 0; index < aWord.length(); index++)
        {
            char c = aWord.charAt(index);
            if (!(Character.isJavaIdentifierPart(c) ||
                  c == '.' ||
                  c == '[' ||
                  c == ']' ||
                  c == '<' ||
                  c == '>' ||
                  c == '-' ||
                  c == '!' ||
                  c == '*' ||
                  c == '?' ||
                  c == '%'))
            {
                return false;
            }
        }

        return true;
    }


    /**
     * Checks whether the given access flags are valid field access flags,
     * throwing a ParseException if they aren't.
     */
    private void checkFieldAccessFlags(int requiredSetMemberAccessFlags,
                                       int requiredUnsetMemberAccessFlags)
    throws ParseException
    {
        if (((requiredSetMemberAccessFlags |
              requiredUnsetMemberAccessFlags) &
            ~ClassConstants.VALID_ACC_FIELD) != 0)
        {
            throw new ParseException("Invalid method access modifier for field before " +
                                     reader.locationDescription());
        }
    }


    /**
     * Checks whether the given access flags are valid method access flags,
     * throwing a ParseException if they aren't.
     */
    private void checkMethodAccessFlags(int requiredSetMemberAccessFlags,
                                        int requiredUnsetMemberAccessFlags)
    throws ParseException
    {
        if (((requiredSetMemberAccessFlags |
              requiredUnsetMemberAccessFlags) &
            ~ClassConstants.VALID_ACC_METHOD) != 0)
        {
            throw new ParseException("Invalid field access modifier for method before " +
                                     reader.locationDescription());
        }
    }


    /**
     * A main method for testing configuration parsing.
     */
    public static void main(String[] args)
    {
        try
        {
            ConfigurationParser parser =
                new ConfigurationParser(args, System.getProperties());

            try
            {
                parser.parse(new Configuration());
            }
            catch (ParseException ex)
            {
                ex.printStackTrace();
            }
            finally
            {
                parser.close();
            }
        }
        catch (IOException ex)
        {
            ex.printStackTrace();
        }
    }
}

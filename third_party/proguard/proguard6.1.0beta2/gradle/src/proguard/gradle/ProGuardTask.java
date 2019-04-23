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
package proguard.gradle;

import groovy.lang.Closure;
import org.gradle.api.DefaultTask;
import org.gradle.api.file.*;
import org.gradle.api.logging.*;
import org.gradle.api.tasks.*;
import org.gradle.api.tasks.Optional;
import proguard.*;
import proguard.classfile.*;
import proguard.classfile.util.ClassUtil;
import proguard.util.ListUtil;

import java.io.*;
import java.net.*;
import java.util.*;

/**
 * This Task allows to configure and run ProGuard from Gradle.
 *
 * @author Eric Lafortune
 */
public class ProGuardTask extends DefaultTask
{
    // Accumulated input and output, for the sake of Gradle's lazy file
    // resolution and lazy task execution.
    private final List          inJarFiles         = new ArrayList();
    private final List          inJarFilters       = new ArrayList();
    private final List          outJarFiles        = new ArrayList();
    private final List          outJarFilters      = new ArrayList();
    private final List          inJarCounts        = new ArrayList();
    private final List          libraryJarFiles    = new ArrayList();
    private final List          libraryJarFilters  = new ArrayList();
    private final List          configurationFiles = new ArrayList();

    // Accumulated configuration.
    protected final Configuration configuration      = new Configuration();

    // Fields acting as parameters for the class member specification methods.
    private boolean            allowValues;
    private ClassSpecification classSpecification;

    private static final String CONFIGURATION_FILE_NAME_PREFIX      = "/proguard/gradle/proguard-";
    private              String resolvedConfigurationFileNamePrefix = getProject().file(CONFIGURATION_FILE_NAME_PREFIX).toString();


    // Gradle task inputs and outputs, because annotations on the List fields
    // (private or not) don't seem to work. Private methods don't work either,
    // but package visible or protected methods are ok.

    @InputFiles
    protected FileCollection getInJarFileCollection()
    {
        return getProject().files(inJarFiles);
    }

    @Optional @OutputFiles
    protected FileCollection getOutJarFileCollection()
    {
        return getProject().files(outJarFiles);
    }

    @InputFiles
    protected FileCollection getLibraryJarFileCollection()
    {
        return getProject().files(libraryJarFiles);
    }

    @InputFiles
    protected FileCollection getConfigurationFileCollection()
    {
        return getProject().files(configurationFiles);
    }


    // Convenience methods to retrieve settings from outside the task.

    /**
     * Returns the collected list of input files (directory, jar, aar, etc,
     * represented as Object, String, File, etc).
     */
    public List getInJarFiles()
    {
        return inJarFiles;
    }

    /**
     * Returns the collected list of filters (represented as argument Maps)
     * corresponding to the list of input files.
     */
    public List getInJarFilters()
    {
        return inJarFilters;
    }

    /**
     * Returns the collected list of output files (directory, jar, aar, etc,
     * represented as Object, String, File, etc).
     */
    public List getOutJarFiles()
    {
        return outJarFiles;
    }

    /**
     * Returns the collected list of filters (represented as argument Maps)
     * corresponding to the list of output files.
     */
    public List getOutJarFilters()
    {
        return outJarFilters;
    }

    /**
     * Returns the list with the numbers of input files that correspond to the
     * list of output files.
     *
     * For instance, [2, 3] means that
     *   the contents of the first 2 input files go to the first output file and
     *   the contents of the next 3 input files go to the second output file.
     */
    public List getInJarCounts()
    {
        return inJarCounts;
    }

    /**
     * Returns the collected list of library files (directory, jar, aar, etc,
     * represented as Object, String, File, etc).
     */
    public List getLibraryJarFiles()
    {
        return libraryJarFiles;
    }

    /**
     * Returns the collected list of filters (represented as argument Maps)
     * corresponding to the list of library files.
     */
    public List getLibraryJarFilters()
    {
        return libraryJarFilters;
    }

    /**
     * Returns the collected list of configuration files to be included
     * (represented as Object, String, File, etc).
     */
    public List getConfigurationFiles()
    {
        return configurationFiles;
    }


    // Gradle task settings corresponding to all ProGuard options.

    public void configuration(Object configurationFiles)
    throws ParseException, IOException
    {
        // Just collect the arguments, so they can be resolved lazily.
        // Flatten collections, so they are easier to analyze later on.
        if (configurationFiles instanceof Collection)
        {
            this.configurationFiles.addAll((Collection)configurationFiles);
        }
        else
        {
            this.configurationFiles.add(configurationFiles);
        }
    }

    public void injars(Object inJarFiles)
    throws ParseException
    {
        injars(null, inJarFiles);
    }

    public void injars(Map filterArgs, Object inJarFiles)
    throws ParseException
    {
        // Just collect the arguments, so they can be resolved lazily.
        this.inJarFiles.add(inJarFiles);
        this.inJarFilters.add(filterArgs);
    }

    public void outjars(Object outJarFiles)
    throws ParseException
    {
        outjars(null, outJarFiles);
    }

    public void outjars(Map filterArgs, Object outJarFiles)
    throws ParseException
    {
        // Just collect the arguments, so they can be resolved lazily.
        this.outJarFiles.add(outJarFiles);
        this.outJarFilters.add(filterArgs);
        this.inJarCounts.add(Integer.valueOf(inJarFiles.size()));
    }

    public void libraryjars(Object libraryJarFiles)
    throws ParseException
    {
        libraryjars(null, libraryJarFiles);
    }

    public void libraryjars(Map filterArgs, Object libraryJarFiles)
    throws ParseException
    {
        // Just collect the arguments, so they can be resolved lazily.
        this.libraryJarFiles.add(libraryJarFiles);
        this.libraryJarFilters.add(filterArgs);
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getskipnonpubliclibraryclasses()
    {
        skipnonpubliclibraryclasses();
        return null;
    }

    public void skipnonpubliclibraryclasses()
    {
        configuration.skipNonPublicLibraryClasses = true;
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getdontskipnonpubliclibraryclassmembers()
    {
        dontskipnonpubliclibraryclassmembers();
        return null;
    }

    public void dontskipnonpubliclibraryclassmembers()
    {
        configuration.skipNonPublicLibraryClassMembers = false;
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getkeepdirectories()
    {
        keepdirectories();
        return null;
    }

    public void keepdirectories()
    {
        keepdirectories(null);
    }

    public void keepdirectories(String filter)
    {
        configuration.keepDirectories =
            extendFilter(configuration.keepDirectories, filter);
    }

    public void target(String targetClassVersion)
    {
        configuration.targetClassVersion =
            ClassUtil.internalClassVersion(targetClassVersion);
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getforceprocessing()
    {
        forceprocessing();
        return null;
    }

    public void forceprocessing()
    {
        configuration.lastModified = Long.MAX_VALUE;
    }

    public void keep(String classSpecificationString)
    throws ParseException
    {
        keep(null, classSpecificationString);
    }

    public void keep(Map    keepArgs,
                     String classSpecificationString)
    throws ParseException
    {
        configuration.keep =
            extendClassSpecifications(configuration.keep,
            createKeepClassSpecification(false,
                                         true,
                                         false,
                                         keepArgs,
                                         classSpecificationString));
    }

    public void keep(Map keepClassSpecificationArgs)
    throws ParseException
    {
        keep(keepClassSpecificationArgs, (Closure)null);
    }

    public void keep(Map     keepClassSpecificationArgs,
                     Closure classMembersClosure)
    throws ParseException
    {
        configuration.keep =
            extendClassSpecifications(configuration.keep,
            createKeepClassSpecification(false,
                                         true,
                                         false,
                                         keepClassSpecificationArgs,
                                         classMembersClosure));
    }

    public void keepclassmembers(String classSpecificationString)
    throws ParseException
    {
        keepclassmembers(null, classSpecificationString);
    }

    public void keepclassmembers(Map    keepArgs,
                                 String classSpecificationString)
    throws ParseException
    {
        configuration.keep =
            extendClassSpecifications(configuration.keep,
            createKeepClassSpecification(false,
                                         false,
                                         false,
                                         keepArgs,
                                         classSpecificationString));
    }

    public void keepclassmembers(Map keepClassSpecificationArgs)
    throws ParseException
    {
        keepclassmembers(keepClassSpecificationArgs, (Closure)null);
    }

    public void keepclassmembers(Map     keepClassSpecificationArgs,
                                 Closure classMembersClosure)
    throws ParseException
    {
        configuration.keep =
            extendClassSpecifications(configuration.keep,
            createKeepClassSpecification(false,
                                         false,
                                         false,
                                         keepClassSpecificationArgs,
                                         classMembersClosure));
    }

    public void keepclasseswithmembers(String classSpecificationString)
    throws ParseException
    {
        keepclasseswithmembers(null, classSpecificationString);
    }

    public void keepclasseswithmembers(Map    keepArgs,
                                       String classSpecificationString)
    throws ParseException
    {
        configuration.keep =
            extendClassSpecifications(configuration.keep,
            createKeepClassSpecification(false,
                                         false,
                                         true,
                                         keepArgs,
                                         classSpecificationString));
    }

    public void keepclasseswithmembers(Map keepClassSpecificationArgs)
    throws ParseException
    {
        keepclasseswithmembers(keepClassSpecificationArgs, (Closure)null);
    }

    public void keepclasseswithmembers(Map     keepClassSpecificationArgs,
                                       Closure classMembersClosure)
    throws ParseException
    {
        configuration.keep =
            extendClassSpecifications(configuration.keep,
            createKeepClassSpecification(false,
                                         false,
                                         true,
                                         keepClassSpecificationArgs,
                                         classMembersClosure));
    }

    public void keepnames(String classSpecificationString)
    throws ParseException
    {
        keepnames(null, classSpecificationString);
    }

    public void keepnames(Map    keepArgs,
                          String classSpecificationString)
    throws ParseException
    {
        configuration.keep =
            extendClassSpecifications(configuration.keep,
            createKeepClassSpecification(true,
                                         true,
                                         false,
                                         keepArgs,
                                         classSpecificationString));
    }

    public void keepnames(Map keepClassSpecificationArgs)
    throws ParseException
    {
        keepnames(keepClassSpecificationArgs, (Closure)null);
    }

    public void keepnames(Map     keepClassSpecificationArgs,
                          Closure classMembersClosure)
    throws ParseException
    {
        configuration.keep =
            extendClassSpecifications(configuration.keep,
            createKeepClassSpecification(true,
                                         true,
                                         false,
                                         keepClassSpecificationArgs,
                                         classMembersClosure));
    }

    public void keepclassmembernames(String classSpecificationString)
    throws ParseException
    {
        keepclassmembernames(null, classSpecificationString);
    }

    public void keepclassmembernames(Map    keepArgs,
                                     String classSpecificationString)
    throws ParseException
    {
        configuration.keep =
            extendClassSpecifications(configuration.keep,
            createKeepClassSpecification(true,
                                         false,
                                         false,
                                         keepArgs,
                                         classSpecificationString));
    }

    public void keepclassmembernames(Map keepClassSpecificationArgs)
    throws ParseException
    {
        keepclassmembernames(keepClassSpecificationArgs, (Closure)null);
    }

    public void keepclassmembernames(Map     keepClassSpecificationArgs,
                                     Closure classMembersClosure)
    throws ParseException
    {
        configuration.keep =
            extendClassSpecifications(configuration.keep,
            createKeepClassSpecification(true,
                                         false,
                                         false,
                                         keepClassSpecificationArgs,
                                         classMembersClosure));
    }

    public void keepclasseswithmembernames(String classSpecificationString)
    throws ParseException
    {
        keepclasseswithmembernames(null, classSpecificationString);
    }

    public void keepclasseswithmembernames(Map    keepArgs,
                                           String classSpecificationString)
    throws ParseException
    {
        configuration.keep =
            extendClassSpecifications(configuration.keep,
            createKeepClassSpecification(true,
                                         false,
                                         true,
                                         keepArgs,
                                         classSpecificationString));
    }

    public void keepclasseswithmembernames(Map keepClassSpecificationArgs)
    throws ParseException
    {
        keepclasseswithmembernames(keepClassSpecificationArgs, (Closure)null);
    }

    public void keepclasseswithmembernames(Map     keepClassSpecificationArgs,
                                           Closure classMembersClosure)
    throws ParseException
    {
        configuration.keep =
            extendClassSpecifications(configuration.keep,
            createKeepClassSpecification(true,
                                         false,
                                         true,
                                         keepClassSpecificationArgs,
                                         classMembersClosure));
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getprintseeds()
    {
        printseeds();
        return null;
    }

    public void printseeds()
    {
        configuration.printSeeds = Configuration.STD_OUT;
    }

    public void printseeds(Object printSeeds)
    throws ParseException
    {
        configuration.printSeeds = getProject().file(printSeeds);
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getdontshrink()
    {
        dontshrink();
        return null;
    }

    public void dontshrink()
    {
        configuration.shrink = false;
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getprintusage()
    {
        printusage();
        return null;
    }

    public void printusage()
    {
        configuration.printUsage = Configuration.STD_OUT;
    }

    public void printusage(Object printUsage)
    throws ParseException
    {
        configuration.printUsage = getProject().file(printUsage);
    }

    public void whyareyoukeeping(String classSpecificationString)
    throws ParseException
    {
        configuration.whyAreYouKeeping =
            extendClassSpecifications(configuration.whyAreYouKeeping,
                                      createClassSpecification(false,
                                                               classSpecificationString));
    }

    public void whyareyoukeeping(Map classSpecificationArgs)
    throws ParseException
    {
        whyareyoukeeping(classSpecificationArgs, null);
    }

    public void whyareyoukeeping(Map     classSpecificationArgs,
                                 Closure classMembersClosure)
    throws ParseException
    {
        configuration.whyAreYouKeeping =
            extendClassSpecifications(configuration.whyAreYouKeeping,
                                      createClassSpecification(false,
                                                               classSpecificationArgs,
                                                               classMembersClosure));
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getdontoptimize()
    {
        dontoptimize();
        return null;
    }

    public void dontoptimize()
    {
        configuration.optimize = false;
    }

    public void optimizations(String filter)
    {
        configuration.optimizations =
            extendFilter(configuration.optimizations, filter);
    }


    public void optimizationpasses(int optimizationPasses)
    {
        configuration.optimizationPasses = optimizationPasses;
    }

    public void assumenosideeffects(String classSpecificationString)
    throws ParseException
    {
        configuration.assumeNoSideEffects =
            extendClassSpecifications(configuration.assumeNoSideEffects,
            createClassSpecification(true,
                                     classSpecificationString));
    }

    public void assumenosideeffects(Map     classSpecificationArgs,
                                    Closure classMembersClosure)
    throws ParseException
    {
        configuration.assumeNoSideEffects =
            extendClassSpecifications(configuration.assumeNoSideEffects,
            createClassSpecification(true,
                                     classSpecificationArgs,
                                     classMembersClosure));
    }

    public void assumenoexternalsideeffects(String classSpecificationString)
    throws ParseException
    {
        configuration.assumeNoExternalSideEffects =
            extendClassSpecifications(configuration.assumeNoExternalSideEffects,
            createClassSpecification(true,
                                     classSpecificationString));
    }

    public void assumenoexternalsideeffects(Map     classSpecificationArgs,
                                            Closure classMembersClosure)
    throws ParseException
    {
        configuration.assumeNoExternalSideEffects =
            extendClassSpecifications(configuration.assumeNoExternalSideEffects,
            createClassSpecification(true,
                                     classSpecificationArgs,
                                     classMembersClosure));
    }

    public void assumenoescapingparameters(String classSpecificationString)
    throws ParseException
    {
        configuration.assumeNoEscapingParameters =
            extendClassSpecifications(configuration.assumeNoEscapingParameters,
            createClassSpecification(true,
                                     classSpecificationString));
    }

    public void assumenoescapingparameters(Map     classSpecificationArgs,
                                           Closure classMembersClosure)
    throws ParseException
    {
        configuration.assumeNoEscapingParameters =
            extendClassSpecifications(configuration.assumeNoEscapingParameters,
            createClassSpecification(true,
                                     classSpecificationArgs,
                                     classMembersClosure));
    }

    public void assumenoexternalreturnvalues(String classSpecificationString)
    throws ParseException
    {
        configuration.assumeNoExternalReturnValues =
            extendClassSpecifications(configuration.assumeNoExternalReturnValues,
            createClassSpecification(true,
                                     classSpecificationString));
    }

    public void assumenoexternalreturnvalues(Map     classSpecificationArgs,
                                             Closure classMembersClosure)
    throws ParseException
    {
        configuration.assumeNoExternalReturnValues =
            extendClassSpecifications(configuration.assumeNoExternalReturnValues,
            createClassSpecification(true,
                                     classSpecificationArgs,
                                     classMembersClosure));
    }

    public void assumevalues(String classSpecificationString)
    throws ParseException
    {
        configuration.assumeValues =
            extendClassSpecifications(configuration.assumeValues,
            createClassSpecification(true,
                                     classSpecificationString));
    }

    public void assumevalues(Map     classSpecificationArgs,
                             Closure classMembersClosure)
    throws ParseException
    {
        configuration.assumeValues =
            extendClassSpecifications(configuration.assumeValues,
            createClassSpecification(true,
                                     classSpecificationArgs,
                                     classMembersClosure));
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getallowaccessmodification()
    {
        allowaccessmodification();
        return null;
    }

    public void allowaccessmodification()
    {
        configuration.allowAccessModification = true;
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getmergeinterfacesaggressively()
    {
        mergeinterfacesaggressively();
        return null;
    }

    public void mergeinterfacesaggressively()
    {
        configuration.mergeInterfacesAggressively = true;
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getdontobfuscate()
    {
        dontobfuscate();
        return null;
    }

    public void dontobfuscate()
    {
        configuration.obfuscate = false;
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getprintmapping()
    {
        printmapping();
        return null;
    }

    public void printmapping()
    {
        configuration.printMapping = Configuration.STD_OUT;
    }

    public void printmapping(Object printMapping)
    throws ParseException
    {
        configuration.printMapping = getProject().file(printMapping);
    }

    public void applymapping(Object applyMapping)
    throws ParseException
    {
        configuration.applyMapping = getProject().file(applyMapping);
    }

    public void obfuscationdictionary(Object obfuscationDictionary)
    throws ParseException, MalformedURLException
    {
        configuration.obfuscationDictionary =
            url(obfuscationDictionary);
    }

    public void classobfuscationdictionary(Object classObfuscationDictionary)
    throws ParseException, MalformedURLException
    {
        configuration.classObfuscationDictionary =
            url(classObfuscationDictionary);
    }

    public void packageobfuscationdictionary(Object packageObfuscationDictionary)
    throws ParseException, MalformedURLException
    {
        configuration.packageObfuscationDictionary =
            url(packageObfuscationDictionary);
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getoverloadaggressively()
    {
        overloadaggressively();
        return null;
    }

    public void overloadaggressively()
    {
        configuration.overloadAggressively = true;
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getuseuniqueclassmembernames()
    {
        useuniqueclassmembernames();
        return null;
    }

    public void useuniqueclassmembernames()
    {
        configuration.useUniqueClassMemberNames = true;
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getdontusemixedcaseclassnames()
    {
        dontusemixedcaseclassnames();
        return null;
    }

    public void dontusemixedcaseclassnames()
    {
        configuration.useMixedCaseClassNames = false;
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getkeeppackagenames()
    {
        keeppackagenames();
        return null;
    }

    public void keeppackagenames()
    {
        keeppackagenames(null);
    }

    public void keeppackagenames(String filter)
    {
        configuration.keepPackageNames =
            extendFilter(configuration.keepPackageNames, filter, true);
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getflattenpackagehierarchy()
    {
        flattenpackagehierarchy();
        return null;
    }

    public void flattenpackagehierarchy()
    {
        flattenpackagehierarchy("");
    }

    public void flattenpackagehierarchy(String flattenPackageHierarchy)
    {
        configuration.flattenPackageHierarchy =
            ClassUtil.internalClassName(flattenPackageHierarchy);
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getrepackageclasses()
    {
        repackageclasses();
        return null;
    }

    public void repackageclasses()
    {
        repackageclasses("");
    }

    public void repackageclasses(String repackageClasses)
    {
        configuration.repackageClasses =
            ClassUtil.internalClassName(repackageClasses);
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getkeepattributes()
    {
        keepattributes();
        return null;
    }

    public void keepattributes()
    {
        keepattributes(null);
    }

    public void keepattributes(String filter)
    {
        configuration.keepAttributes =
            extendFilter(configuration.keepAttributes, filter);
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getkeepparameternames()
    {
        keepparameternames();
        return null;
    }

    public void keepparameternames()
    {
        configuration.keepParameterNames = true;
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getrenamesourcefileattribute()
    {
        renamesourcefileattribute();
        return null;
    }

    public void renamesourcefileattribute()
    {
        renamesourcefileattribute("");
    }

    public void renamesourcefileattribute(String newSourceFileAttribute)
    {
        configuration.newSourceFileAttribute = newSourceFileAttribute;
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getadaptclassstrings()
    {
        adaptclassstrings();
        return null;
    }

    public void adaptclassstrings()
    {
        adaptclassstrings(null);
    }

    public void adaptclassstrings(String filter)
    {
        configuration.adaptClassStrings =
            extendFilter(configuration.adaptClassStrings, filter, true);
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getadaptresourcefilenames()
    {
        adaptresourcefilenames();
        return null;
    }

    public void adaptresourcefilenames()
    {
        adaptresourcefilenames(null);
    }

    public void adaptresourcefilenames(String filter)
    {
        configuration.adaptResourceFileNames =
            extendFilter(configuration.adaptResourceFileNames, filter);
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getadaptresourcefilecontents()
    {
        adaptresourcefilecontents();
        return null;
    }

    public void adaptresourcefilecontents()
    {
        adaptresourcefilecontents(null);
    }

    public void adaptresourcefilecontents(String filter)
    {
        configuration.adaptResourceFileContents =
            extendFilter(configuration.adaptResourceFileContents, filter);
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getdontpreverify()
    {
        dontpreverify();
        return null;
    }

    public void dontpreverify()
    {
        configuration.preverify = false;
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getmicroedition()
    {
        microedition();
        return null;
    }

    public void microedition()
    {
        configuration.microEdition = true;
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getandroid()
    {
        android();
        return null;
    }

    public void android()
    {
        configuration.android = true;
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getverbose()
    {
        verbose();
        return null;
    }

    public void verbose()
    {
        configuration.verbose = true;
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getdontnote()
    {
        dontnote();
        return null;
    }

    public void dontnote()
    {
        dontnote(null);
    }

    public void dontnote(String filter)
    {
        configuration.note = extendFilter(configuration.note, filter, true);
    }


    // Hack: support the keyword without parentheses in Groovy.
    public Object getdontwarn()
    {
        dontwarn();
        return null;
    }

    public void dontwarn()
    {
        dontwarn(null);
    }

    public void dontwarn(String filter)
    {
        configuration.warn = extendFilter(configuration.warn, filter, true);
    }


    // Hack: support the keyword without parentheses in Groovy.
    public Object getignorewarnings()
    {
        ignorewarnings();
        return null;
    }

    public void ignorewarnings()
    {
        configuration.ignoreWarnings = true;
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getprintconfiguration()
    {
        printconfiguration();
        return null;
    }

    public void printconfiguration()
    {
        configuration.printConfiguration = Configuration.STD_OUT;
    }

    public void printconfiguration(Object printConfiguration)
    throws ParseException
    {
        configuration.printConfiguration =
            getProject().file(printConfiguration);
    }

    // Hack: support the keyword without parentheses in Groovy.
    public Object getdump()
    {
        dump();
        return null;
    }

    public void dump()
    {
        configuration.dump = Configuration.STD_OUT;
    }

    public void dump(Object dump)
    throws ParseException
    {
        configuration.dump = getProject().file(dump);
    }


    // Hack: support the keyword without parentheses in Groovy.
    public Object getaddconfigurationdebugging()
    {
        addconfigurationdebugging();
        return null;
    }

    public void addconfigurationdebugging()
    {
        configuration.addConfigurationDebugging = true;
    }


    // Class member methods.

    public void field(Map memberSpecificationArgs)
    throws ParseException
    {
        if (classSpecification == null)
        {
            throw new IllegalArgumentException("The 'field' method can only be used nested inside a class specification.");
        }

        classSpecification.addField(createMemberSpecification(false,
                                                              false,
                                                              allowValues,
                                                              memberSpecificationArgs));
    }


    public void constructor(Map memberSpecificationArgs)
    throws ParseException
    {
        if (classSpecification == null)
        {
            throw new IllegalArgumentException("The 'constructor' method can only be used nested inside a class specification.");
        }

        classSpecification.addMethod(createMemberSpecification(true,
                                                               true,
                                                               allowValues,
                                                               memberSpecificationArgs));
    }


    public void method(Map memberSpecificationArgs)
    throws ParseException
    {
        if (classSpecification == null)
        {
            throw new IllegalArgumentException("The 'method' method can only be used nested inside a class specification.");
        }

        classSpecification.addMethod(createMemberSpecification(true,
                                                               false,
                                                               allowValues,
                                                               memberSpecificationArgs));
    }


    // Gradle task execution.

    @TaskAction
    public void proguard()
    throws ParseException, IOException
    {
        // Let the logging manager capture the standard output and errors from
        // ProGuard.
        LoggingManager loggingManager = getLogging();
        loggingManager.captureStandardOutput(LogLevel.INFO);
        loggingManager.captureStandardError(LogLevel.WARN);

        // Run ProGuard with the collected configuration.
        new ProGuard(getConfiguration()).execute();

    }


    /**
     * Returns the configuration collected so far, resolving files and
     * reading included configurations.
     */
    private Configuration getConfiguration() throws IOException, ParseException
    {
        // Weave the input jars and the output jars into a single class path,
        // with lazy resolution of the files.
        configuration.programJars = new ClassPath();

        int outJarIndex = 0;

        int inJarCount = inJarCounts.size() == 0 ? -1 :
                ((Integer)inJarCounts.get(0)).intValue();

        for (int inJarIndex = 0; inJarIndex < inJarFiles.size(); inJarIndex++)
        {
            configuration.programJars =
                extendClassPath(configuration.programJars,
                                inJarFiles.get(inJarIndex),
                                (Map)inJarFilters.get(inJarIndex),
                                false);

            while (inJarIndex == inJarCount - 1)
            {
                configuration.programJars =
                    extendClassPath(configuration.programJars,
                                    outJarFiles.get(outJarIndex),
                                    (Map)outJarFilters.get(outJarIndex),
                                    true);

                outJarIndex++;

                inJarCount = inJarCounts.size() == outJarIndex ? -1 :
                    ((Integer)inJarCounts.get(outJarIndex)).intValue();
            }
        }

        // Copy the library jars into a single class path, with lazy resolution
        // of the files.
        configuration.libraryJars = new ClassPath();

        for (int libraryJarIndex = 0; libraryJarIndex < libraryJarFiles.size(); libraryJarIndex++)
        {
            configuration.libraryJars =
                extendClassPath(configuration.libraryJars,
                                libraryJarFiles.get(libraryJarIndex),
                                (Map)libraryJarFilters.get(libraryJarIndex),
                                false);
        }

        // Lazily apply the external configuration files.
        for (int index = 0; index < configurationFiles.size(); index++)
        {
            Object fileObject = configurationFiles.get(index);

            ConfigurableFileCollection fileCollection =
                getProject().files(fileObject);

            // Parse the configuration as a collection of files.
            Iterator<File> files = fileCollection.iterator();

            while (files.hasNext())
            {
                File file = files.next();

                // Check if this is the name of an internal configuration file.
                if (isInternalConfigurationFile(file))
                {
                    getLogger().debug("Loading default configuration file " +
                                      file.getAbsolutePath());

                    String internalConfigFileName =
                        internalConfigurationFileName(file);

                    URL configurationURL =
                        ProGuardTask.class.getResource(internalConfigFileName);

                    if (configurationURL == null)
                    {
                        throw new FileNotFoundException("'" + file.getAbsolutePath() + "'");
                    }

                    // Parse the configuration as a URL.
                    ConfigurationParser parser =
                        new ConfigurationParser(configurationURL,
                                                System.getProperties());

                    try
                    {
                        parser.parse(configuration);
                    }
                    finally
                    {
                        parser.close();
                    }
                }
                else
                {
                    getLogger().debug("Loading configuration file " +
                                      file.getAbsolutePath());

                    ConfigurationParser parser =
                        new ConfigurationParser(file,
                                                System.getProperties());

                    try
                    {
                        parser.parse(configuration);
                    }
                    finally
                    {
                        parser.close();
                    }
                }
            }
        }

        // Make sure the code is processed. Gradle has already checked that it
        // was necessary.
        configuration.lastModified = Long.MAX_VALUE;

        return configuration;
    }


    // Small utility methods.


    /**
     * Returns whether the given file object is an internal configuration
     * file that is packaged in a jar, for instance
     *  "/proguard/gradle/proguard-android-debug.pro".
     */
    private boolean isInternalConfigurationFile(Object fileObject)
    {
        return fileObject.toString().startsWith(resolvedConfigurationFileNamePrefix);
    }


    /**
     * Returns the name of the internal configuration file that is packaged in
     * a jar, for instance "/lib/dexguard-debug.pro".
     */
    private String internalConfigurationFileName(Object fileObject)
    {
        // Trim any added file prefix (like "C:").
        // Make sure we have forward slashes.
        return fileObject.toString()
            .substring(resolvedConfigurationFileNamePrefix.length() -
                       CONFIGURATION_FILE_NAME_PREFIX.length())
            .replace('\\', '/');
    }


    /**
     * Extends the given class path with the given filtered input or output
     * files.
     */
    private ClassPath extendClassPath(ClassPath classPath,
                                      Object    files,
                                      Map       filterArgs,
                                      boolean   output)
    {
        ConfigurableFileCollection fileCollection = getProject().files(files);

        if (classPath == null)
        {
            classPath = new ClassPath();
        }

        Iterator fileIterator = fileCollection.iterator();
        while (fileIterator.hasNext())
        {
            File file = (File)fileIterator.next();
            if (output || file.exists())
            {
                // Create the class path entry.
                ClassPathEntry classPathEntry = new ClassPathEntry(file, output);

                // Add any filters to the class path entry.
                if (filterArgs != null)
                {
                    classPathEntry.setFilter(ListUtil.commaSeparatedList((String)filterArgs.get("filter")));
                    classPathEntry.setApkFilter(ListUtil.commaSeparatedList((String)filterArgs.get("apkfilter")));
                    classPathEntry.setJarFilter(ListUtil.commaSeparatedList((String)filterArgs.get("jarfilter")));
                    classPathEntry.setAarFilter(ListUtil.commaSeparatedList((String)filterArgs.get("aarfilter")));
                    classPathEntry.setWarFilter(ListUtil.commaSeparatedList((String)filterArgs.get("warfilter")));
                    classPathEntry.setEarFilter(ListUtil.commaSeparatedList((String)filterArgs.get("earfilter")));
                    classPathEntry.setZipFilter(ListUtil.commaSeparatedList((String)filterArgs.get("zipfilter")));
                }

                classPath.add(classPathEntry);
            }
        }

        return classPath;
    }


    /**
     * Creates specifications to keep classes and class members, based on the
     * given parameters.
     */
    private KeepClassSpecification createKeepClassSpecification(boolean allowShrinking,
                                                                boolean markClasses,
                                                                boolean markConditionally,
                                                                Map     keepArgs,
                                                                String  classSpecificationString)
    throws ParseException
    {
        ClassSpecification condition =
            createIfClassSpecification(keepArgs);

        ClassSpecification classSpecification =
            createClassSpecification(false, classSpecificationString);

        return
            createKeepClassSpecification(allowShrinking,
                                         markClasses,
                                         markConditionally,
                                         keepArgs,
                                         condition,
                                         classSpecification);
    }


    /**
     * Creates specifications to keep classes and class members, based on the
     * given parameters.
     */
    private KeepClassSpecification createKeepClassSpecification(boolean allowShrinking,
                                                                boolean markClasses,
                                                                boolean markConditionally,
                                                                Map     classSpecificationArgs,
                                                                Closure classMembersClosure)
    throws ParseException
    {
        ClassSpecification condition =
            createIfClassSpecification(classSpecificationArgs);

        ClassSpecification classSpecification =
            createClassSpecification(false,
                                     classSpecificationArgs,
                                     classMembersClosure);

        return
            createKeepClassSpecification(allowShrinking,
                                         markClasses,
                                         markConditionally,
                                         classSpecificationArgs,
                                         condition,
                                         classSpecification);
    }


    /**
     * Creates a conditional class specification, based on the given
     * parameters.
     */
    private ClassSpecification createIfClassSpecification(Map classSpecificationArgs)
    throws ParseException
    {
        if (classSpecificationArgs == null)
        {
            return null;
        }

        String conditionString = (String)classSpecificationArgs.get("if");
        if (conditionString == null)
        {
            return null;
        }

        return createClassSpecification(false, conditionString);
    }


    /**
     * Creates specifications to keep classes and class members, based on the
     * given parameters.
     */
    private KeepClassSpecification createKeepClassSpecification(boolean            allowShrinking,
                                                                boolean            markClasses,
                                                                boolean            markConditionally,
                                                                Map                keepArgs,
                                                                ClassSpecification condition,
                                                                ClassSpecification classSpecification)
    {
        return
            new KeepClassSpecification(markClasses,
                                       markConditionally,
                                       retrieveBoolean(keepArgs, "includedescriptorclasses", false),
                                       false,
                                       retrieveBoolean(keepArgs, "allowshrinking",           allowShrinking),
                                       retrieveBoolean(keepArgs, "allowoptimization",        false),
                                       retrieveBoolean(keepArgs, "allowobfuscation",         false),
                                       condition,
                                       classSpecification);
    }


    /**
     * Creates specifications to keep classes and class members, based on the
     * given ProGuard-style class specification.
     */
    private ClassSpecification createClassSpecification(boolean allowValues,
                                                        String  classSpecificationString)
    throws ParseException
    {
        try
        {
            ConfigurationParser parser =
                new ConfigurationParser(new String[] { classSpecificationString }, null);

            try
            {
                return parser.parseClassSpecificationArguments(allowValues);
            }
            finally
            {
                parser.close();
            }
        }
        catch (IOException e)
        {
            throw new ParseException(e.getMessage());
        }
    }


    /**
     * Creates a specification of classes and class members, based on the
     * given parameters.
     */
    private ClassSpecification createClassSpecification(boolean allowValues,
                                                        Map     classSpecificationArgs,
                                                        Closure classMembersClosure)
    throws ParseException
    {
        // Extract the arguments.
        String access            = (String)classSpecificationArgs.get("access");
        String annotation        = (String)classSpecificationArgs.get("annotation");
        String type              = (String)classSpecificationArgs.get("type");
        String name              = (String)classSpecificationArgs.get("name");
        String extendsAnnotation = (String)classSpecificationArgs.get("extendsannotation");
        String extends_          = (String)classSpecificationArgs.get("extends");
        if (extends_ == null)
        {
            extends_             = (String)classSpecificationArgs.get("implements");
        }

        // Create the class specification.
        ClassSpecification classSpecification =
            new ClassSpecification(null,
                                   requiredClassAccessFlags(true, access, type),
                                   requiredClassAccessFlags(false, access, type),
                                   annotation        != null ? ClassUtil.internalType(annotation)        : null,
                                   name              != null ? ClassUtil.internalClassName(name)         : null,
                                   extendsAnnotation != null ? ClassUtil.internalType(extendsAnnotation) : null,
                                   extends_          != null ? ClassUtil.internalClassName(extends_)     : null);

        // Initialize the class specification with its closure.
        if (classMembersClosure != null)
        {
            // Temporarily remember the class specification, so we can add
            // class member specifications.
            this.allowValues        = allowValues;
            this.classSpecification = classSpecification;
            classMembersClosure.call(classSpecification);
            this.allowValues        = false;
            this.classSpecification = null;
        }

        return classSpecification;
    }


    /**
     * Parses the class access flags that must be set (or not), based on the
     * given ProGuard-style flag specification.
     */
    private int requiredClassAccessFlags(boolean set,
                                         String  access,
                                         String  type)
    throws ParseException
    {
        int accessFlags = 0;

        if (access != null)
        {
            StringTokenizer tokenizer = new StringTokenizer(access, " ,");
            while (tokenizer.hasMoreTokens())
            {
                String token = tokenizer.nextToken();

                if (token.startsWith("!") ^ set)
                {
                    String strippedToken = token.startsWith("!") ?
                        token.substring(1) :
                        token;

                    int accessFlag =
                        strippedToken.equals(JavaConstants.ACC_PUBLIC)     ? ClassConstants.ACC_PUBLIC      :
                        strippedToken.equals(JavaConstants.ACC_FINAL)      ? ClassConstants.ACC_FINAL       :
                        strippedToken.equals(JavaConstants.ACC_ABSTRACT)   ? ClassConstants.ACC_ABSTRACT    :
                        strippedToken.equals(JavaConstants.ACC_SYNTHETIC)  ? ClassConstants.ACC_SYNTHETIC   :
                        strippedToken.equals(JavaConstants.ACC_ANNOTATION) ? ClassConstants.ACC_ANNOTATION  :
                                                                             0;

                    if (accessFlag == 0)
                    {
                        throw new ParseException("Incorrect class access modifier ["+strippedToken+"]");
                    }

                    accessFlags |= accessFlag;
                }
            }
        }

        if (type != null && (type.startsWith("!") ^ set))
        {
            int accessFlag =
                type.equals("class")                           ? 0                            :
                type.equals(      JavaConstants.ACC_INTERFACE) ||
                type.equals("!" + JavaConstants.ACC_INTERFACE) ? ClassConstants.ACC_INTERFACE :
                type.equals(      JavaConstants.ACC_ENUM)      ||
                type.equals("!" + JavaConstants.ACC_ENUM)      ? ClassConstants.ACC_ENUM      :
                                                                 -1;
            if (accessFlag == -1)
            {
                throw new ParseException("Incorrect class type ["+type+"]");
            }

            accessFlags |= accessFlag;
        }

        return accessFlags;
    }


    /**
     * Creates a specification of class members, based on the given parameters.
     */
    private MemberSpecification createMemberSpecification(boolean isMethod,
                                                          boolean isConstructor,
                                                          boolean allowValues,
                                                          Map     classSpecificationArgs)
    throws ParseException
    {
        // Extract the arguments.
        String access            = (String)classSpecificationArgs.get("access");
        String type              = (String)classSpecificationArgs.get("type");
        String annotation        = (String)classSpecificationArgs.get("annotation");
        String name              = (String)classSpecificationArgs.get("name");
        String parameters        = (String)classSpecificationArgs.get("parameters");
        String values            = (String)classSpecificationArgs.get("value");

        // Perform some basic conversions and checks on the attributes.
        if (annotation != null)
        {
            annotation = ClassUtil.internalType(annotation);
        }

        if (isMethod)
        {
            if (isConstructor)
            {
                if (type != null)
                {
                    throw new ParseException("Type attribute not allowed in constructor specification ["+type+"]");
                }

                if (parameters != null)
                {
                    type = JavaConstants.TYPE_VOID;
                }

                if (values != null)
                {
                    throw new ParseException("Values attribute not allowed in constructor specification ["+values+"]");
                }

                name = ClassConstants.METHOD_NAME_INIT;
            }
            else if ((type != null) ^ (parameters != null))
            {
                throw new ParseException("Type and parameters attributes must always be present in combination in method specification");
            }
        }
        else
        {
            if (parameters != null)
            {
                throw new ParseException("Parameters attribute not allowed in field specification ["+parameters+"]");
            }
        }

        if (values != null)
        {
            if (!allowValues)
            {
                throw new ParseException("Values attribute not allowed in this class specification ["+values+"]");
            }

            if (type == null)
            {
                throw new ParseException("Values attribute must be specified in combination with type attribute in class member specification ["+values+"]");
            }
        }

        List parameterList = ListUtil.commaSeparatedList(parameters);

        String descriptor =
            parameters != null ? ClassUtil.internalMethodDescriptor(type, parameterList) :
            type       != null ? ClassUtil.internalType(type)                            :
                                 null;

        return values != null ?
            new MemberValueSpecification(requiredMemberAccessFlags(true,  access),
                                         requiredMemberAccessFlags(false, access),
                                         annotation,
                                         name,
                                         descriptor,
                                         parseValues(type,
                                                     ClassUtil.internalType(type),
                                                     values)) :
            new MemberSpecification(requiredMemberAccessFlags(true,  access),
                                    requiredMemberAccessFlags(false, access),
                                    annotation,
                                    name,
                                    descriptor);
    }


    /**
     * Parses the class member access flags that must be set (or not), based on
     * the given ProGuard-style flag specification.
     */
    private int requiredMemberAccessFlags(boolean set,
                                          String  access)
    throws ParseException
    {
        int accessFlags = 0;

        if (access != null)
        {
            StringTokenizer tokenizer = new StringTokenizer(access, " ,");
            while (tokenizer.hasMoreTokens())
            {
                String token = tokenizer.nextToken();

                if (token.startsWith("!") ^ set)
                {
                    String strippedToken = token.startsWith("!") ?
                        token.substring(1) :
                        token;

                    int accessFlag =
                        strippedToken.equals(JavaConstants.ACC_PUBLIC)       ? ClassConstants.ACC_PUBLIC       :
                        strippedToken.equals(JavaConstants.ACC_PRIVATE)      ? ClassConstants.ACC_PRIVATE      :
                        strippedToken.equals(JavaConstants.ACC_PROTECTED)    ? ClassConstants.ACC_PROTECTED    :
                        strippedToken.equals(JavaConstants.ACC_STATIC)       ? ClassConstants.ACC_STATIC       :
                        strippedToken.equals(JavaConstants.ACC_FINAL)        ? ClassConstants.ACC_FINAL        :
                        strippedToken.equals(JavaConstants.ACC_SYNCHRONIZED) ? ClassConstants.ACC_SYNCHRONIZED :
                        strippedToken.equals(JavaConstants.ACC_VOLATILE)     ? ClassConstants.ACC_VOLATILE     :
                        strippedToken.equals(JavaConstants.ACC_TRANSIENT)    ? ClassConstants.ACC_TRANSIENT    :
                        strippedToken.equals(JavaConstants.ACC_BRIDGE)       ? ClassConstants.ACC_BRIDGE       :
                        strippedToken.equals(JavaConstants.ACC_VARARGS)      ? ClassConstants.ACC_VARARGS      :
                        strippedToken.equals(JavaConstants.ACC_NATIVE)       ? ClassConstants.ACC_NATIVE       :
                        strippedToken.equals(JavaConstants.ACC_ABSTRACT)     ? ClassConstants.ACC_ABSTRACT     :
                        strippedToken.equals(JavaConstants.ACC_STRICT)       ? ClassConstants.ACC_STRICT       :
                        strippedToken.equals(JavaConstants.ACC_SYNTHETIC)    ? ClassConstants.ACC_SYNTHETIC    :
                                                                               0;

                    if (accessFlag == 0)
                    {
                        throw new ParseException("Incorrect class member access modifier ["+strippedToken+"]");
                    }

                    accessFlags |= accessFlag;
                }
            }
        }

        return accessFlags;
    }


    /**
     * Retrieves a specified boolean flag from the given map.
     */
    private boolean retrieveBoolean(Map args, String name, boolean defaultValue)
    {
        if (args == null)
        {
            return defaultValue;
        }

        Object arg = args.get(name);

        return arg == null ? defaultValue : ((Boolean)arg).booleanValue();
    }


    /**
     * Parses the given string as a value or value range of the given primitive
     * type. For example, values "123" or "100..199" of type "int" ("I").
     */
    private Number[] parseValues(String externalType,
                                 String internalType,
                                 String string)
    throws ParseException
    {
        int rangeIndex = string.lastIndexOf("..");
        return rangeIndex >= 0 ?
            new Number[]
            {
                parseValue(externalType, internalType, string.substring(0, rangeIndex)),
                parseValue(externalType, internalType, string.substring(rangeIndex + 2))
            } :
            new Number[]
            {
                parseValue(externalType, internalType, string)
            };
    }


    /**
     * Parses the given string as a value of the given primitive type.
     * For example, value "123" of type "int" ("I").
     * For example, value "true" of type "boolean" ("Z"), returned as 1.
     */
    private Number parseValue(String externalType,
                              String internalType,
                              String string)
    throws ParseException
    {
        try
        {
            switch (internalType.charAt(0))
            {
                case ClassConstants.TYPE_BOOLEAN:
                {
                    return parseBoolean(string);
                }
                case ClassConstants.TYPE_BYTE:
                case ClassConstants.TYPE_CHAR:
                case ClassConstants.TYPE_SHORT:
                case ClassConstants.TYPE_INT:
                {
                    return Integer.decode(string);
                }
                //case ClassConstants.TYPE_LONG:
                //{
                //    return Long.decode(string);
                //}
                //case ClassConstants.TYPE_FLOAT:
                //{
                //    return Float.valueOf(string);
                //}
                //case ClassConstants.TYPE_DOUBLE:
                //{
                //    return Double.valueOf(string);
                //}
                default:
                {
                    throw new ParseException("Can't handle '"+externalType+"' constant ["+string+"]");
                }
            }
        }
        catch (NumberFormatException e)
        {
            throw new ParseException("Can't parse "+externalType+" constant ["+string+"]");
        }
    }


    /**
     * Parses the given boolean string as an integer (0 or 1).
     */
    private Integer parseBoolean(String string)
    throws ParseException
    {
        if      ("false".equals(string))
        {
            return Integer.valueOf(0);
        }
        else if ("true".equals(string))
        {
            return Integer.valueOf(1);
        }
        else
        {
            throw new ParseException("Unknown boolean constant ["+string+"]");
        }
    }


    /**
     * Adds the given class specification to the given list, creating a new list
     * if necessary.
     */
    protected List extendClassSpecifications(List               classSpecifications,
                                             ClassSpecification classSpecification)
    {
        if (classSpecifications == null)
        {
            classSpecifications = new ArrayList();
        }

        classSpecifications.add(classSpecification);

        return classSpecifications;
    }


    /**
     * Adds the given class specifications to the given list, creating a new
     * list if necessary.
     */
    private List extendClassSpecifications(List classSpecifications,
                                           List additionalClassSpecifications)
    {
        if (additionalClassSpecifications != null)
        {
            if (classSpecifications == null)
            {
                classSpecifications = new ArrayList();
            }

            classSpecifications.addAll(additionalClassSpecifications);
        }

        return classSpecifications;
    }


    /**
     * Adds the given filter to the given list, creating a new list if
     * necessary.
     */
    private List extendFilter(List   filter,
                              String filterString)
    {
        return extendFilter(filter, filterString, false);
    }


    /**
     * Adds the given filter to the given list, creating a new list if
     * necessary. External class names are converted to internal class names,
     * if requested.
     */
    private List extendFilter(List    filter,
                              String  filterString,
                              boolean convertExternalClassNames)
    {
        if (filter == null)
        {
            filter = new ArrayList();
        }

        if (filterString == null)
        {
            // Clear the filter to keep all names.
            filter.clear();
        }
        else
        {
            if (convertExternalClassNames)
            {
                filterString = ClassUtil.internalClassName(filterString);
            }

            // Append the filter.
            filter.addAll(ListUtil.commaSeparatedList(filterString));
        }

        return filter;
    }


    /**
     * Converts a file object into a URL, looking relatively to this class if
     * necessary.
     */
    private URL url(Object fileObject) throws MalformedURLException
    {
        File file = getProject().file(fileObject);
        return
            fileObject instanceof URL ? (URL)fileObject :
            fileObject instanceof String &&
            !file.exists()            ? ProGuardTask.class.getResource((String)fileObject) :
                                        file.toURI().toURL();
    }
}

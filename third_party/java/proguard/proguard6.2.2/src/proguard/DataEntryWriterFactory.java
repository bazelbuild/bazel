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

import proguard.classfile.*;
import proguard.io.*;
import proguard.util.*;

import static proguard.DataEntryReaderFactory.getFilterExcludingVersionedClasses;

import java.util.List;

/**
 * This class can create DataEntryWriter instances based on class paths. The
 * writers will wrap the output in the proper apks, jars, wars, ears, jmods,
 * and zips.
 *
 * @author Eric Lafortune
 */
public class DataEntryWriterFactory
{
    private final ClassPool                              programClassPool;
    private final MultiValueMap                          extraClassNameMap;


    /**
     * Creates a new DataEntryWriterFactory with the given parameters.
     * @param programClassPool      the program classpool to process.
     */
    public DataEntryWriterFactory(ClassPool                              programClassPool,
                                  MultiValueMap<String, String>          extraClassNamemap)
    {
        this.programClassPool                 = programClassPool;
        this.extraClassNameMap                = extraClassNamemap;
    }


    /**
     * Creates a DataEntryWriter that can write to the given class path entries.
     *
     * @param classPath the output class path.
     * @param fromIndex the start index in the class path.
     * @param toIndex   the end index in the class path.
     * @return a DataEntryWriter for writing to the given class path entries.
     */
    public DataEntryWriter createDataEntryWriter(ClassPath classPath,
                                                 int       fromIndex,
                                                 int       toIndex)
    {
        DataEntryWriter writer = null;

        // Create a chain of writers, one for each class path entry.
        for (int index = toIndex - 1; index >= fromIndex; index--)
        {
            ClassPathEntry entry = classPath.get(index);

            writer = createClassPathEntryWriter(entry, writer);
        }

        return writer;
    }


    /**
     * Creates a DataEntryWriter that can write to the given class path entry,
     * or delegate to another DataEntryWriter if its filters don't match.
     */
    private DataEntryWriter createClassPathEntryWriter(ClassPathEntry  classPathEntry,
                                                       DataEntryWriter alternativeWriter)
    {
        boolean isApk  = classPathEntry.isApk();
        boolean isJar  = classPathEntry.isJar();
        boolean isAar  = classPathEntry.isAar();
        boolean isWar  = classPathEntry.isWar();
        boolean isEar  = classPathEntry.isEar();
        boolean isJmod = classPathEntry.isJmod();
        boolean isZip  = classPathEntry.isZip();

        List filter     = getFilterExcludingVersionedClasses(classPathEntry);
        List apkFilter  = classPathEntry.getApkFilter();
        List jarFilter  = classPathEntry.getJarFilter();
        List aarFilter  = classPathEntry.getAarFilter();
        List warFilter  = classPathEntry.getWarFilter();
        List earFilter  = classPathEntry.getEarFilter();
        List jmodFilter = classPathEntry.getJmodFilter();
        List zipFilter  = classPathEntry.getZipFilter();

        System.out.println("Preparing output " +
                           (isApk  ? "apk"  :
                            isJar  ? "jar"  :
                            isAar  ? "aar"  :
                            isWar  ? "war"  :
                            isEar  ? "ear"  :
                            isJmod ? "jmod" :
                            isZip  ? "zip"  :
                                     "directory") +
                           " [" + classPathEntry.getName() + "]" +
                           (filter     != null ||
                            apkFilter  != null ||
                            jarFilter  != null ||
                            aarFilter  != null ||
                            warFilter  != null ||
                            earFilter  != null ||
                            jmodFilter != null ||
                            zipFilter  != null ? " (filtered)" : ""));

        DataEntryWriter writer = new DirectoryWriter(classPathEntry.getFile(),
                                                     isApk  ||
                                                     isJar  ||
                                                     isAar  ||
                                                     isWar  ||
                                                     isEar  ||
                                                     isJmod ||
                                                     isZip);

        // If the output is an archive, we'll flatten (unpack the contents of)
        // higher level input archives, e.g. when writing into a jar file, we
        // flatten zip files.
        boolean flattenApks  = false;
        boolean flattenJars  = flattenApks  || isApk;
        boolean flattenAars  = flattenJars  || isJar;
        boolean flattenWars  = flattenAars  || isAar;
        boolean flattenEars  = flattenWars  || isWar;
        boolean flattenJmods = flattenEars  || isEar;
        boolean flattenZips  = flattenJmods || isJmod;

        // Set up the filtered jar writers.
        writer = wrapInJarWriter(writer, flattenZips,  isZip,  ".zip",  zipFilter,  null,                       null);
        writer = wrapInJarWriter(writer, flattenJmods, isJmod, ".jmod", jmodFilter, ClassConstants.JMOD_HEADER, ClassConstants.JMOD_CLASS_FILE_PREFIX);
        writer = wrapInJarWriter(writer, flattenEars,  isEar,  ".ear",  earFilter,  null,                       null);
        writer = wrapInJarWriter(writer, flattenWars,  isWar,  ".war",  warFilter,  null,                       ClassConstants.WAR_CLASS_FILE_PREFIX);
        writer = wrapInJarWriter(writer, flattenAars,  isAar,  ".aar",  aarFilter,  null,                       null);
        writer = wrapInJarWriter(writer, flattenJars,  isJar,  ".jar",  jarFilter,  null,                       null);
        writer = wrapInJarWriter(writer, flattenApks,  isApk,  ".apk",  apkFilter,  null,                       null);

        // Set up for writing out the program classes.
        writer = new ClassDataEntryWriter(programClassPool, writer);

        // Add a data entry filter, if specified.
        writer = filter != null ?
            new FilteredDataEntryWriter(
                new DataEntryNameFilter(
                    new ListParser(new FileNameParser()).parse(filter)),
                writer) :
            writer;

        // Add a writer for the injected classes.
        writer = new ExtraDataEntryWriter(extraClassNameMap,
                                          writer,
                                          writer,
                                          ClassConstants.CLASS_FILE_EXTENSION);

        // Let the writer cascade, if specified.
        return alternativeWriter != null ?
            new CascadingDataEntryWriter(writer, alternativeWriter) :
            writer;
    }


    /**
     * Wraps the given DataEntryWriter in a JarWriter, filtering if necessary.
     */
    private DataEntryWriter wrapInJarWriter(DataEntryWriter writer,
                                            boolean         flatten,
                                            boolean         isOutputJar,
                                            String          jarFilterExtension,
                                            List            jarFilter,
                                            byte[]          jarHeader,
                                            String          classFilePrefix)
    {
        // Flatten jars or zip them up.
        DataEntryWriter jarWriter;
        if (flatten)
        {
            // Unpack the jar.
            jarWriter = new ParentDataEntryWriter(writer);
        }
        else
        {
            // Pack the jar.
            jarWriter = new JarWriter(jarHeader, writer);

            // Add a prefix for class files inside the jar, if specified.
            if (classFilePrefix != null)
            {
                jarWriter =
                    new FilteredDataEntryWriter(
                    new DataEntryNameFilter(
                    new ExtensionMatcher(ClassConstants.CLASS_FILE_EXTENSION)),
                    new PrefixAddingDataEntryWriter(classFilePrefix,
                                                    jarWriter),
                    jarWriter);
            }
        }

        // Either zip up the jar or delegate to the original writer.
        return
            new FilteredDataEntryWriter(
            new DataEntryParentFilter(
            new DataEntryNameFilter(
            new ExtensionMatcher(jarFilterExtension))),

                // The parent of the data entry is a jar.
                // Write the data entry to the jar.
                // Apply the jar filter, if specified, to the parent.
                jarFilter != null ?
                    new FilteredDataEntryWriter(
                    new DataEntryParentFilter(
                    new DataEntryNameFilter(
                    new ListParser(new FileNameParser()).parse(jarFilter))),
                    jarWriter) :
                    jarWriter,

                // The parent of the data entry is not a jar.
                // Write the entry to a jar anyway if the output is a jar.
                // Otherwise just delegate to the original writer.
                isOutputJar ?
                    jarWriter :
                    writer);
    }
}

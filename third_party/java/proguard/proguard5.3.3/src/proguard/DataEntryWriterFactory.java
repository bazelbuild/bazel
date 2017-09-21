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

import proguard.io.*;
import proguard.util.*;

import java.util.List;

/**
 * This class can create DataEntryWriter instances based on class paths. The
 * writers will wrap the output in the proper apks, jars, wars, ears, and zips.
 *
 * @author Eric Lafortune
 */
public class DataEntryWriterFactory
{
    /**
     * Creates a DataEntryWriter that can write to the given class path entries.
     *
     * @param classPath the output class path.
     * @param fromIndex the start index in the class path.
     * @param toIndex   the end index in the class path.
     * @return a DataEntryWriter for writing to the given class path entries.
     */
    public static DataEntryWriter createDataEntryWriter(ClassPath classPath,
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
    private static DataEntryWriter createClassPathEntryWriter(ClassPathEntry  classPathEntry,
                                                              DataEntryWriter alternativeWriter)
    {
        boolean isApk = classPathEntry.isApk();
        boolean isJar = classPathEntry.isJar();
        boolean isAar = classPathEntry.isAar();
        boolean isWar = classPathEntry.isWar();
        boolean isEar = classPathEntry.isEar();
        boolean isZip = classPathEntry.isZip();

        List filter    = classPathEntry.getFilter();
        List apkFilter = classPathEntry.getApkFilter();
        List jarFilter = classPathEntry.getJarFilter();
        List aarFilter = classPathEntry.getAarFilter();
        List warFilter = classPathEntry.getWarFilter();
        List earFilter = classPathEntry.getEarFilter();
        List zipFilter = classPathEntry.getZipFilter();

        System.out.println("Preparing output " +
                           (isApk ? "apk" :
                            isJar ? "jar" :
                            isAar ? "aar" :
                            isWar ? "war" :
                            isEar ? "ear" :
                            isZip ? "zip" :
                                    "directory") +
                           " [" + classPathEntry.getName() + "]" +
                           (filter    != null ||
                            apkFilter != null ||
                            jarFilter != null ||
                            aarFilter != null ||
                            warFilter != null ||
                            earFilter != null ||
                            zipFilter != null ? " (filtered)" : ""));

        DataEntryWriter writer = new DirectoryWriter(classPathEntry.getFile(),
                                                     isApk ||
                                                     isJar ||
                                                     isAar ||
                                                     isWar ||
                                                     isEar ||
                                                     isZip);

        // Set up the filtered jar writers.
        writer = wrapInJarWriter(writer, isZip, zipFilter, ".zip", isApk || isJar || isAar || isWar || isEar);
        writer = wrapInJarWriter(writer, isEar, earFilter, ".ear", isApk || isJar || isAar || isWar);
        writer = wrapInJarWriter(writer, isWar, warFilter, ".war", isApk || isJar || isAar);
        writer = wrapInJarWriter(writer, isAar, aarFilter, ".aar", isApk || isJar);
        writer = wrapInJarWriter(writer, isJar, jarFilter, ".jar", isApk);
        writer = wrapInJarWriter(writer, isApk, apkFilter, ".apk", false);

        // Add a filter, if specified.
        writer = filter != null?
            new FilteredDataEntryWriter(
            new DataEntryNameFilter(
            new ListParser(new FileNameParser()).parse(filter)),
                writer) :
            writer;

        // Let the writer cascade, if specified.
        return alternativeWriter != null ?
            new CascadingDataEntryWriter(writer, alternativeWriter) :
            writer;
    }


    /**
     * Wraps the given DataEntryWriter in a JarWriter, filtering if necessary.
     */
    private static DataEntryWriter wrapInJarWriter(DataEntryWriter writer,
                                                   boolean         isJar,
                                                   List            jarFilter,
                                                   String          jarExtension,
                                                   boolean         dontWrap)
    {
        // Zip up jars, if necessary.
        DataEntryWriter jarWriter = dontWrap ?
            (DataEntryWriter)new ParentDataEntryWriter(writer) :
            (DataEntryWriter)new JarWriter(writer);

        // Add a filter, if specified.
        DataEntryWriter filteredJarWriter = jarFilter != null?
            new FilteredDataEntryWriter(
            new DataEntryParentFilter(
            new DataEntryNameFilter(
            new ListParser(new FileNameParser()).parse(jarFilter))),
                 jarWriter) :
            jarWriter;

        // Only zip up jars, unless the output is a jar file itself.
        return new FilteredDataEntryWriter(
               new DataEntryParentFilter(
               new DataEntryNameFilter(
               new ExtensionMatcher(jarExtension))),
                   filteredJarWriter,
                   isJar ? jarWriter : writer);
    }
}

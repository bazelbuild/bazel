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
 * This class can create DataEntryReader instances based on class path entries.
 * The readers will unwrap the input data entries from any jars, wars, ears,
 * and zips, before passing them to a given reader.
 *
 * @author Eric Lafortune
 */
public class DataEntryReaderFactory
{
    /**
     * Creates a DataEntryReader that can read the given class path entry.
     *
     * @param messagePrefix  a prefix for messages that are printed out.
     * @param classPathEntry the input class path entry.
     * @param reader         a data entry reader to which the reading of actual
     *                       classes and resource files can be delegated.
     * @return a DataEntryReader for reading the given class path entry.
     */
    public static DataEntryReader createDataEntryReader(String          messagePrefix,
                                                        ClassPathEntry  classPathEntry,
                                                        DataEntryReader reader)
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

        System.out.println(messagePrefix +
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

        // Add a filter, if specified.
        if (filter != null)
        {
            reader = new FilteredDataEntryReader(
                     new DataEntryNameFilter(
                     new ListParser(new FileNameParser()).parse(filter)),
                         reader);
        }

        // Unzip any apks, if necessary.
        reader = wrapInJarReader(reader, isApk, apkFilter, ".apk");
        if (!isApk)
        {
            // Unzip any jars, if necessary.
            reader = wrapInJarReader(reader, isJar, jarFilter, ".jar");
            if (!isJar)
            {
                // Unzip any aars, if necessary.
                reader = wrapInJarReader(reader, isAar, aarFilter, ".aar");
                if (!isAar)
                {
                    // Unzip any wars, if necessary.
                    reader = wrapInJarReader(reader, isWar, warFilter, ".war");
                    if (!isWar)
                    {
                        // Unzip any ears, if necessary.
                        reader = wrapInJarReader(reader, isEar, earFilter, ".ear");
                        if (!isEar)
                        {
                            // Unzip any zips, if necessary.
                            reader = wrapInJarReader(reader, isZip, zipFilter, ".zip");
                        }
                    }
                }
            }
        }

        return reader;
    }


    /**
     *  Wraps the given DataEntryReader in a JarReader, filtering it if necessary.
     */
    private static DataEntryReader wrapInJarReader(DataEntryReader reader,
                                                   boolean         isJar,
                                                   List            jarFilter,
                                                   String          jarExtension)
    {
        // Unzip any jars, if necessary.
        DataEntryReader jarReader = new JarReader(reader);

        if (isJar)
        {
            // Always unzip.
            return jarReader;
        }
        else
        {
            // Add a filter, if specified.
            if (jarFilter != null)
            {
                jarReader = new FilteredDataEntryReader(
                            new DataEntryNameFilter(
                            new ListParser(new FileNameParser()).parse(jarFilter)),
                                jarReader);
            }

            // Only unzip the right type of jars.
            return new FilteredDataEntryReader(
                   new DataEntryNameFilter(
                   new ExtensionMatcher(jarExtension)),
                       jarReader,
                       reader);
        }
    }
}

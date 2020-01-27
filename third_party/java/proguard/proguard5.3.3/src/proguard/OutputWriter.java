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

import proguard.classfile.ClassPool;
import proguard.classfile.util.ClassUtil;
import proguard.io.*;

import java.io.IOException;
import java.util.*;

/**
 * This class writes the output class files.
 *
 * @author Eric Lafortune
 */
public class OutputWriter
{
    private final Configuration configuration;


    /**
     * Creates a new OutputWriter to write output class files as specified by
     * the given configuration.
     */
    public OutputWriter(Configuration configuration)
    {
        this.configuration = configuration;
    }


    /**
     * Writes the given class pool to class files, based on the current
     * configuration.
     */
    public void execute(ClassPool programClassPool) throws IOException
    {
        ClassPath programJars = configuration.programJars;

        int firstInputIndex = 0;
        int lastInputIndex  = 0;

        // Go over all program class path entries.
        for (int index = 0; index < programJars.size(); index++)
        {
            // Is it an input entry?
            ClassPathEntry entry = programJars.get(index);
            if (!entry.isOutput())
            {
                // Remember the index of the last input entry.
                lastInputIndex = index;
            }
            else
            {
                // Check if this the last output entry in a series.
                int nextIndex = index + 1;
                if (nextIndex == programJars.size() ||
                    !programJars.get(nextIndex).isOutput())
                {
                    // Write the processed input entries to the output entries.
                    writeOutput(programClassPool,
                                programJars,
                                firstInputIndex,
                                lastInputIndex + 1,
                                nextIndex);

                    // Start with the next series of input entries.
                    firstInputIndex = nextIndex;
                }
            }
        }
    }


    /**
     * Transfers the specified input jars to the specified output jars.
     */
    private void writeOutput(ClassPool programClassPool,
                             ClassPath classPath,
                             int       fromInputIndex,
                             int       fromOutputIndex,
                             int       toOutputIndex)
    throws IOException
    {
        try
        {
            // Construct the writer that can write jars, wars, ears, zips, and
            // directories, cascading over the specified output entries.
            DataEntryWriter writer =
                DataEntryWriterFactory.createDataEntryWriter(classPath,
                                                             fromOutputIndex,
                                                             toOutputIndex);

            // The writer will be used to write possibly obfuscated class files.
            DataEntryReader classRewriter =
                new ClassRewriter(programClassPool, writer);

            // The writer will also be used to write resource files.
            DataEntryReader resourceCopier =
                new DataEntryCopier(writer);

            DataEntryReader resourceRewriter = resourceCopier;

            // Adapt resource file contents and names, if necessary.
            if (configuration.obfuscate)
            {
                // Wrap the resource writer with a filter and a data entry
                // rewriter, if required.
                if (configuration.adaptResourceFileContents != null)
                {
                    resourceRewriter =
                        new NameFilter(configuration.adaptResourceFileContents,
                        new NameFilter("META-INF/MANIFEST.MF,META-INF/*.SF",
                            new ManifestRewriter(programClassPool, writer),
                            new DataEntryRewriter(programClassPool, writer)),
                        resourceRewriter);
                }

                // Wrap the resource writer with a filter and a data entry
                // renamer, if required.
                if (configuration.adaptResourceFileNames != null)
                {
                    Map packagePrefixMap = createPackagePrefixMap(programClassPool);

                    resourceRewriter =
                        new NameFilter(configuration.adaptResourceFileNames,
                        new DataEntryObfuscator(programClassPool,
                                                packagePrefixMap,
                                                resourceRewriter),
                        resourceRewriter);
                }
            }

            DataEntryReader directoryRewriter = null;

            // Wrap the directory writer with a filter and a data entry renamer,
            // if required.
            if (configuration.keepDirectories != null)
            {
                Map packagePrefixMap = createPackagePrefixMap(programClassPool);

                directoryRewriter =
                    new NameFilter(configuration.keepDirectories,
                    new DataEntryRenamer(packagePrefixMap,
                                         resourceCopier,
                                         resourceCopier));
            }

            // Create the reader that can write class files and copy directories
            // and resource files to the main writer.
            DataEntryReader reader =
                new ClassFilter(    classRewriter,
                new DirectoryFilter(directoryRewriter,
                                    resourceRewriter));

            // Go over the specified input entries and write their processed
            // versions.
            new InputReader(configuration).readInput("  Copying resources from program ",
                                                     classPath,
                                                     fromInputIndex,
                                                     fromOutputIndex,
                                                     reader);

            // Close all output entries.
            writer.close();
        }
        catch (IOException ex)
        {
            throw (IOException)new IOException("Can't write [" + classPath.get(fromOutputIndex).getName() + "] (" + ex.getMessage() + ")").initCause(ex);
        }
    }


    /**
     * Creates a map of old package prefixes to new package prefixes, based on
     * the given class pool.
     */
    private static Map createPackagePrefixMap(ClassPool classPool)
    {
        Map packagePrefixMap = new HashMap();

        Iterator iterator = classPool.classNames();
        while (iterator.hasNext())
        {
            String className     = (String)iterator.next();
            String packagePrefix = ClassUtil.internalPackagePrefix(className);

            String mappedNewPackagePrefix = (String)packagePrefixMap.get(packagePrefix);
            if (mappedNewPackagePrefix == null ||
                !mappedNewPackagePrefix.equals(packagePrefix))
            {
                String newClassName     = classPool.getClass(className).getName();
                String newPackagePrefix = ClassUtil.internalPackagePrefix(newClassName);

                packagePrefixMap.put(packagePrefix, newPackagePrefix);
            }
        }

        return packagePrefixMap;
    }
}

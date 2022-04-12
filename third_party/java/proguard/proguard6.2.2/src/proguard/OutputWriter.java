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
import proguard.classfile.util.ClassUtil;
import proguard.configuration.ConfigurationLogger;
import proguard.io.*;
import proguard.util.*;

import java.io.*;
import java.nio.charset.Charset;
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
    public void execute(ClassPool                              programClassPool,
                        MultiValueMap<String, String>          injectedClassNameMap) throws IOException
    {
        ClassPath programJars = configuration.programJars;

        // Create a data entry writer factory with common archival parameters.
        DataEntryWriterFactory dataEntryWriterFactory =
            new DataEntryWriterFactory(programClassPool,
                                       injectedClassNameMap);

        int firstInputIndex = 0;
        int lastInputIndex  = 0;

        // Go over all program class path entries.
        for (int index = 0; index < programJars.size(); index++)
        {
            // Is it an input entry?
            ClassPathEntry entry = programJars.get(index);
            if (!entry.isOutput())
            {
                // It's an input entry. Remember the highest index.
                lastInputIndex = index;
            }
            else
            {
                // It's an output entry. Is it the last one in a
                // series of output entries?
                int nextIndex = index + 1;
                if (nextIndex == programJars.size() ||
                    !programJars.get(nextIndex).isOutput())
                {
                    // Write the processed input entries to the output entries.
                    writeOutput(dataEntryWriterFactory,
                                programClassPool,
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
    private void writeOutput(DataEntryWriterFactory dataEntryWriterFactory,
                             ClassPool              programClassPool,
                             ClassPath              classPath,
                             int                    fromInputIndex,
                             int                    fromOutputIndex,
                             int                    toOutputIndex)
    throws IOException
    {
        try
        {
            // Construct the writer that can write apks, jars, wars, ears, zips,
            // and directories, cascading over the specified output entries.
            DataEntryWriter writer =
                dataEntryWriterFactory.createDataEntryWriter(classPath,
                                                             fromOutputIndex,
                                                             toOutputIndex);

            if (configuration.addConfigurationDebugging)
            {
                writer = new ExtraDataEntryWriter(ConfigurationLogger.CLASS_MAP_FILENAME,
                    writer,
                    new ClassMapDataEntryWriter(programClassPool, writer));
                System.err.println("Warning: -addconfigurationdebugging is enabled; the resulting build will contain obfuscation information.");
                System.err.println("It should only be used for debugging purposes.");
            }

            DataEntryWriter resourceWriter = writer;

            // Adapt plain resource file names that correspond to class names,
            // if necessary.
            if (configuration.obfuscate &&
                configuration.adaptResourceFileNames != null)
            {
                // Rename processed general resources.
                resourceWriter =
                    renameResourceFiles(programClassPool,
                                        resourceWriter);
            }

            // By default, just copy resource files into the above writers.
            DataEntryReader resourceCopier =
                new DataEntryCopier(resourceWriter);

            // We're now switching to the reader side, operating on the
            // contents possibly parsed from the input streams.
            DataEntryReader resourceRewriter = resourceCopier;

            // Adapt resource file contents, if allowed.
            if ((configuration.shrink   ||
                 configuration.optimize ||
                 configuration.obfuscate) &&
                configuration.adaptResourceFileContents != null)
            {
                DataEntryReader adaptingContentWriter = resourceRewriter;

                // Adapt the contents of general resource files (manifests
                // and native libraries).
                if (configuration.obfuscate)
                {
                    adaptingContentWriter =
                        adaptResourceFiles(programClassPool,
                                           resourceWriter);
                }

                // Add the overall filter for adapting resource file contents.
                resourceRewriter =
                    new NameFilter(configuration.adaptResourceFileContents,
                        adaptingContentWriter,
                        resourceRewriter);
            }

            // Write any kept directories.
            DataEntryReader reader =
                writeDirectories(programClassPool,
                                 resourceCopier,
                                 resourceRewriter);

            // Trigger writing classes.
            reader =
                new ClassFilter(new IdleRewriter(writer),
                                reader);

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
     * Returns a writer that writes possibly renamed resource files to the
     * given resource writer.
     */
    private DataEntryWriter renameResourceFiles(ClassPool       programClassPool,
                                                DataEntryWriter dataEntryWriter)
    {
        Map packagePrefixMap = createPackagePrefixMap(programClassPool);

        return
            new NameFilteredDataEntryWriter(configuration.adaptResourceFileNames,
                new RenamedDataEntryWriter(programClassPool, packagePrefixMap, dataEntryWriter),
                dataEntryWriter);
    }


    /**
     * Returns a reader that writes all general resource files (manifest,
     * native libraries, text files) with shrunk, optimized, and obfuscated
     * contents to the given writer.
     */
    private DataEntryReader adaptResourceFiles(ClassPool       programClassPool,
                                               DataEntryWriter writer)
    {
        // Pick a suitable encoding.
        Charset charset = configuration.android ?
            Charset.forName("UTF-8") :
            Charset.defaultCharset();

        // Filter between the various general resource files.
        return
            new NameFilter("META-INF/MANIFEST.MF,META-INF/*.SF",
                new ManifestRewriter(programClassPool, charset, writer),
            new DataEntryRewriter(programClassPool, charset, writer));
    }


    /**
     * Writes possibly renamed directories that should be preserved to the
     * given resource copier, and non-directories to the given file copier.
     */
    private DirectoryFilter writeDirectories(ClassPool       programClassPool,
                                             DataEntryReader directoryCopier,
                                             DataEntryReader fileCopier)
    {
        DataEntryReader directoryRewriter = null;

        // Wrap the directory copier with a filter and a data entry renamer.
        if (configuration.keepDirectories != null)
        {
            Map packagePrefixMap = createPackagePrefixMap(programClassPool);

            directoryRewriter =
                new NameFilter(configuration.keepDirectories,
                new RenamedDataEntryReader(packagePrefixMap,
                                           directoryCopier,
                                           directoryCopier));
        }

        // Filter on directories and files.
        return new DirectoryFilter(directoryRewriter, fileCopier);
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

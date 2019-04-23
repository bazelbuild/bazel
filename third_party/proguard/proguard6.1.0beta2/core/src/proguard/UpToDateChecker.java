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

import java.io.File;
import java.net.*;

/**
 * This class checks whether the output is up to date.
 *
 * @author Eric Lafortune
 */
public class UpToDateChecker
{
    private final Configuration configuration;


    /**
     * Creates a new UpToDateChecker with the given configuration.
     */
    public UpToDateChecker(Configuration configuration)
    {
        this.configuration = configuration;
    }


    /**
     * Returns whether the output is up to date, based on the modification times
     * of the input jars, output jars, and library jars (or directories).
     */
    public boolean check()
    {
        try
        {
            ModificationTimeChecker checker = new ModificationTimeChecker();

            checker.updateInputModificationTime(configuration.lastModified);

            ClassPath programJars = configuration.programJars;
            ClassPath libraryJars = configuration.libraryJars;

            // Check the dates of the program jars, if any.
            if (programJars != null)
            {
                for (int index = 0; index < programJars.size(); index++)
                {
                    // Update the input and output modification times.
                    ClassPathEntry classPathEntry = programJars.get(index);

                    checker.updateModificationTime(classPathEntry.getFile(),
                                                   classPathEntry.isOutput());
                }
            }

            // Check the dates of the library jars, if any.
            if (libraryJars != null)
            {
                for (int index = 0; index < libraryJars.size(); index++)
                {
                    // Update the input modification time.
                    ClassPathEntry classPathEntry = libraryJars.get(index);

                    checker.updateModificationTime(classPathEntry.getFile(),
                                                   false);
                }
            }

            // Check the dates of the auxiliary input files.
            checker.updateInputModificationTime(configuration.applyMapping);
            checker.updateInputModificationTime(configuration.obfuscationDictionary);
            checker.updateInputModificationTime(configuration.classObfuscationDictionary);
            checker.updateInputModificationTime(configuration.packageObfuscationDictionary);

            // Check the dates of the auxiliary output files.
            checker.updateOutputModificationTime(configuration.printSeeds);
            checker.updateOutputModificationTime(configuration.printUsage);
            checker.updateOutputModificationTime(configuration.printMapping);
            checker.updateOutputModificationTime(configuration.printConfiguration);
            checker.updateOutputModificationTime(configuration.dump);
        }
        catch (IllegalStateException e)
        {
            // The output is outdated.
            return false;
        }

        System.out.println("The output seems up to date");

        return true;
    }


    /**
     * This class maintains the modification times of input and output.
     * The methods throw an IllegalStateException if the output appears
     * outdated.
     */
    private static class ModificationTimeChecker {

        private long inputModificationTime  = Long.MIN_VALUE;
        private long outputModificationTime = Long.MAX_VALUE;


        /**
         * Updates the input modification time based on the given file or
         * directory (recursively).
         */
        public void updateInputModificationTime(URL url)
        {
            if (url != null &&
                url.getProtocol().equals("file"))
            {
                try
                {
                    updateModificationTime(new File(url.toURI()), false);
                }
                catch (URISyntaxException ignore) {}
            }
        }


        /**
         * Updates the input modification time based on the given file or
         * directory (recursively).
         */
        public void updateInputModificationTime(File file)
        {
            if (file != null)
            {
                updateModificationTime(file, false);
            }
        }


        /**
         * Updates the input modification time based on the given file or
         * directory (recursively).
         */
        public void updateOutputModificationTime(File file)
        {
            if (file != null && file.getName().length() > 0)
            {
                updateModificationTime(file, true);
            }
        }


        /**
         * Updates the specified modification time based on the given file or
         * directory (recursively).
         */
        public void updateModificationTime(File file, boolean isOutput)
        {
            // Is it a directory?
            if (file.isDirectory())
            {
                // Ignore the directory's modification time; just recurse on
                // its files.
                File[] files = file.listFiles();

                // Still, an empty output directory is probably a sign that it
                // is not up to date.
                if (files.length == 0 && isOutput)
                {
                    updateOutputModificationTime(Long.MIN_VALUE);
                }

                for (int index = 0; index < files.length; index++)
                {
                    updateModificationTime(files[index], isOutput);
                }
            }
            else
            {
                // Update with the file's modification time.
                updateModificationTime(file.lastModified(), isOutput);
            }
        }


        /**
         * Updates the specified modification time.
         */
        public void updateModificationTime(long time, boolean isOutput)
        {
            if (isOutput)
            {
                updateOutputModificationTime(time);
            }
            else
            {
                updateInputModificationTime(time);
            }
        }


        /**
         * Updates the input modification time.
         */
        public void updateInputModificationTime(long time)
        {
            if (inputModificationTime < time)
            {
                inputModificationTime = time;

                checkModificationTimes();
            }
        }


        /**
         * Updates the output modification time.
         */
        public void updateOutputModificationTime(long time)
        {
            if (outputModificationTime > time)
            {
                outputModificationTime = time;

                checkModificationTimes();
            }
        }


        private void checkModificationTimes()
        {
            if (inputModificationTime > outputModificationTime)
            {
                throw new IllegalStateException("The output is outdated");
            }
        }
    }
}

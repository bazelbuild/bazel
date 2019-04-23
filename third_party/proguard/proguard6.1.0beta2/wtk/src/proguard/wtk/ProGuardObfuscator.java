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
package proguard.wtk;

import com.sun.kvem.environment.Obfuscator;
import proguard.*;

import java.io.*;


/**
 * ProGuard plug-in for the J2ME Wireless Toolkit.
 * <p>
 * In order to integrate this plug-in in the toolkit, you'll have to put the
 * following lines in the file
 * {j2mewtk.dir}<code>/wtklib/Linux/ktools.properties</code> or
 * {j2mewtk.dir}<code>\wtklib\Windows\ktools.properties</code> (whichever is
 * applicable).
 * <p>
 * <pre>
 * obfuscator.runner.class.name: proguard.wtk.ProGuardObfuscator
 * obfuscator.runner.classpath: /usr/local/java/proguard1.6/lib/proguard.jar
 * </pre>
 * Please make sure the class path is set correctly for your system.
 *
 * @author Eric Lafortune
 */
public class ProGuardObfuscator implements Obfuscator
{
    private static final String DEFAULT_CONFIGURATION = "default.pro";


    // Implementations for Obfuscator.

    public void createScriptFile(File jadFile,
                                 File projectDir)
    {
        // We don't really need to create a script file;
        // we'll just fill out all options in the run method.
    }


    public void run(File   obfuscatedJarFile,
                    String wtkBinDir,
                    String wtkLibDir,
                    String jarFileName,
                    String projectDirName,
                    String classPath,
                    String emptyAPI)
    throws IOException
    {
        // Create the ProGuard configuration.
        Configuration configuration = new Configuration();

        // Parse the default configuration file.
        ConfigurationParser parser = new ConfigurationParser(this.getClass().getResource(DEFAULT_CONFIGURATION),
                                                             System.getProperties());

        try
        {
            parser.parse(configuration);

            // Fill out the library class path.
            configuration.libraryJars = classPath(classPath);

            // Fill out the program class path (input and output).
            configuration.programJars = new ClassPath();
            configuration.programJars.add(new ClassPathEntry(new File(jarFileName), false));
            configuration.programJars.add(new ClassPathEntry(obfuscatedJarFile, true));

            // The preverify tool seems to unpack the resulting classes,
            // so we must not use mixed-case class names on Windows.
            configuration.useMixedCaseClassNames =
                !System.getProperty("os.name").regionMatches(true, 0, "windows", 0, 7);

            // Run ProGuard with these options.
            ProGuard proGuard = new ProGuard(configuration);
            proGuard.execute();

        }
        catch (ParseException ex)
        {
            throw new IOException(ex.getMessage());
        }
        finally
        {
            parser.close();
        }
    }


    /**
     * Converts the given class path String into a ClassPath object.
     */
    private ClassPath classPath(String classPathString)
    {
        ClassPath classPath = new ClassPath();

        String separator = System.getProperty("path.separator");

        int index = 0;
        while (index < classPathString.length())
        {
            // Find the next separator, or the end of the String.
            int next_index = classPathString.indexOf(separator, index);
            if (next_index < 0)
            {
                next_index = classPathString.length();
            }

            // Create and add the found class path entry.
            ClassPathEntry classPathEntry =
                new ClassPathEntry(new File(classPathString.substring(index, next_index)),
                                   false);

            classPath.add(classPathEntry);

            // Continue after the separator.
            index = next_index + 1;
        }

        return classPath;
    }
}

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
package proguard.ant;

import org.apache.tools.ant.*;
import org.apache.tools.ant.types.*;
import proguard.*;

import java.io.*;
import java.util.Properties;

/**
 * This DataType represents a reference to an XML-style ProGuard configuration
 * in Ant, or a file set of ProGuard-style configuration files.
 *
 * @author Eric Lafortune
 */
public class ConfigurationElement extends FileSet
{
    /**
     * Adds the contents of this configuration element to the given
     * configuration.
     * @param configuration the configuration to be extended.
     */
    public void appendTo(Configuration configuration)
    {
        File     baseDir;
        String[] fileNames;

        if (isReference())
        {
            // Get the referenced path or file set.
            Object referencedObject = getCheckedRef(Object.class,
                                                    Object.class.getName());

            if (referencedObject instanceof ConfigurationTask)
            {
                // The reference doesn't point to a file set, but to a
                // configuration task.
                ConfigurationTask configurationTask =
                    (ConfigurationTask)referencedObject;

                // Append the contents of the referenced configuration to the
                // current configuration.
                configurationTask.appendTo(configuration);

                return;
            }
            else if (referencedObject instanceof AbstractFileSet)
            {
                AbstractFileSet fileSet = (AbstractFileSet)referencedObject;

                // Get the names of the existing input files in the referenced file set.
                DirectoryScanner scanner = fileSet.getDirectoryScanner(getProject());
                baseDir   = scanner.getBasedir();
                fileNames = scanner.getIncludedFiles();
            }
            else
            {
                throw new BuildException("The refid attribute doesn't point to a <proguardconfiguration> element or a <fileset> element");
            }
        }
        else
        {
            // Get the names of the existing input files in the referenced file set.
            DirectoryScanner scanner = getDirectoryScanner(getProject());
            baseDir   = scanner.getBasedir();
            fileNames = scanner.getIncludedFiles();
        }

        // Get the combined system properties and Ant properties, for
        // replacing ProGuard-style properties ('<...>').
        Properties properties = new Properties();
        properties.putAll(getProject().getProperties());

        try
        {
            // Append the contents of the configuration files to the current
            // configuration.
            for (int index = 0; index < fileNames.length; index++)
            {
                File configurationFile = new File(baseDir, fileNames[index]);

                ConfigurationParser parser =
                    new ConfigurationParser(configurationFile, properties);
                try
                {
                    parser.parse(configuration);
                }
                catch (ParseException ex)
                {
                    throw new BuildException(ex.getMessage());
                }
                finally
                {
                    parser.close();
                }
            }
        }
        catch (IOException ex)
        {
            throw new BuildException(ex.getMessage());
        }
    }
}

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
import proguard.*;

import java.io.IOException;
import java.util.*;

/**
 * This Task allows to define a ProGuard configuration from Ant.
 *
 * @author Eric Lafortune
 */
public class ConfigurationTask extends Task
{
    protected final Configuration configuration = new Configuration();


    /**
     * Adds the contents of this configuration task to the given configuration.
     * @param configuration the configuration to be extended.
     */
    public void appendTo(Configuration configuration)
    {
        // Append all of these configuration entries to the given configuration.
        configuration.programJars                  = extendClassPath(configuration.programJars,
                                                                     this.configuration.programJars);

        configuration.libraryJars                  = extendClassPath(configuration.libraryJars,
                                                                     this.configuration.libraryJars);

        configuration.keep                         = extendClassSpecifications(configuration.keep,
                                                                               this.configuration.keep);

        configuration.keepDirectories              = extendList(configuration.keepDirectories,
                                                                this.configuration.keepDirectories);

        configuration.whyAreYouKeeping             = extendClassSpecifications(configuration.whyAreYouKeeping,
                                                                               this.configuration.whyAreYouKeeping);

        configuration.optimizations                = extendClassSpecifications(configuration.optimizations,
                                                                               this.configuration.optimizations);

        configuration.assumeNoSideEffects          = extendClassSpecifications(configuration.assumeNoSideEffects,
                                                                               this.configuration.assumeNoSideEffects);

        configuration.assumeNoExternalSideEffects  = extendClassSpecifications(configuration.assumeNoExternalSideEffects,
                                                                              this.configuration.assumeNoExternalSideEffects);

        configuration.assumeNoEscapingParameters   = extendClassSpecifications(configuration.assumeNoEscapingParameters,
                                                                               this.configuration.assumeNoEscapingParameters);

        configuration.assumeNoExternalReturnValues = extendClassSpecifications(configuration.assumeNoExternalReturnValues,
                                                                               this.configuration.assumeNoExternalReturnValues);

        configuration.assumeValues                 = extendClassSpecifications(configuration.assumeValues,
                                                                               this.configuration.assumeValues);

        configuration.keepPackageNames             = extendList(configuration.keepPackageNames,
                                                                this.configuration.keepPackageNames);

        configuration.keepAttributes               = extendList(configuration.keepAttributes,
                                                                this.configuration.keepAttributes);

        configuration.adaptClassStrings            = extendList(configuration.adaptClassStrings,
                                                                this.configuration.adaptClassStrings);

        configuration.adaptResourceFileNames       = extendList(configuration.adaptResourceFileNames,
                                                                this.configuration.adaptResourceFileNames);

        configuration.adaptResourceFileContents    = extendList(configuration.adaptResourceFileContents,
                                                                this.configuration.adaptResourceFileContents);

        configuration.note                         = extendList(configuration.note,
                                                                this.configuration.note);

        configuration.warn                         = extendList(configuration.warn,
                                                                this.configuration.warn);
    }


    // Ant task nested elements.

    public void addConfiguredInjar(ClassPathElement classPathElement)
    {
        configuration.programJars = extendClassPath(configuration.programJars,
                                                    classPathElement,
                                                    false);
    }


    public void addConfiguredOutjar(ClassPathElement classPathElement)
    {
        configuration.programJars = extendClassPath(configuration.programJars,
                                                    classPathElement,
                                                    true);
    }


    public void addConfiguredLibraryjar(ClassPathElement classPathElement)
    {
        configuration.libraryJars = extendClassPath(configuration.libraryJars,
                                                    classPathElement,
                                                    false);
    }


    public void addConfiguredKeepdirectory(FilterElement filterElement)
    {
        configuration.keepDirectories = extendFilter(configuration.keepDirectories,
                                                     filterElement);
    }


    public void addConfiguredKeepdirectories(FilterElement filterElement)
    {
        configuration.keepDirectories = extendFilter(configuration.keepDirectories,
                                                     filterElement);
    }


    public void addConfiguredKeep(KeepSpecificationElement keepSpecificationElement)
    {
        configuration.keep = extendKeepSpecifications(configuration.keep,
                                                      keepSpecificationElement,
                                                      true,
                                                      false);
    }


    public void addConfiguredKeepclassmembers(KeepSpecificationElement keepSpecificationElement)
    {
        configuration.keep = extendKeepSpecifications(configuration.keep,
                                                      keepSpecificationElement,
                                                      false,
                                                      false);
    }


    public void addConfiguredKeepclasseswithmembers(KeepSpecificationElement keepSpecificationElement)
    {
        configuration.keep = extendKeepSpecifications(configuration.keep,
                                                      keepSpecificationElement,
                                                      true,
                                                      true);
    }


    public void addConfiguredKeepnames(KeepSpecificationElement keepSpecificationElement)
    {
        // Set the shrinking flag, based on the name (backward compatibility).
        keepSpecificationElement.setAllowshrinking(true);

        configuration.keep = extendKeepSpecifications(configuration.keep,
                                                      keepSpecificationElement,
                                                      true,
                                                      false);
    }


    public void addConfiguredKeepclassmembernames(KeepSpecificationElement keepSpecificationElement)
    {
        // Set the shrinking flag, based on the name (backward compatibility).
        keepSpecificationElement.setAllowshrinking(true);

        configuration.keep = extendKeepSpecifications(configuration.keep,
                                                      keepSpecificationElement,
                                                      false,
                                                      false);
    }


    public void addConfiguredKeepclasseswithmembernames(KeepSpecificationElement keepSpecificationElement)
    {
        // Set the shrinking flag, based on the name (backward compatibility).
        keepSpecificationElement.setAllowshrinking(true);

        configuration.keep = extendKeepSpecifications(configuration.keep,
                                                      keepSpecificationElement,
                                                      true,
                                                      true);
    }


    public void addConfiguredWhyareyoukeeping(ClassSpecificationElement classSpecificationElement)
    {
        configuration.whyAreYouKeeping = extendClassSpecifications(configuration.whyAreYouKeeping,
                                                                   classSpecificationElement);
    }


    public void addConfiguredAssumenosideeffects(ClassSpecificationElement classSpecificationElement)
    {
        configuration.assumeNoSideEffects = extendClassSpecifications(configuration.assumeNoSideEffects,
                                                                      classSpecificationElement);
    }


    public void addConfiguredAssumenoexternalsideeffects(ClassSpecificationElement classSpecificationElement)
    {
        configuration.assumeNoExternalSideEffects = extendClassSpecifications(configuration.assumeNoExternalSideEffects,
                                                                      classSpecificationElement);
    }


    public void addConfiguredAssumenoescapingparameters(ClassSpecificationElement classSpecificationElement)
    {
        configuration.assumeNoEscapingParameters = extendClassSpecifications(configuration.assumeNoEscapingParameters,
                                                                             classSpecificationElement);
    }


    public void addConfiguredAssumenoexternalreturnvalues(ClassSpecificationElement classSpecificationElement)
    {
        configuration.assumeNoExternalReturnValues = extendClassSpecifications(configuration.assumeNoExternalReturnValues,
                                                                               classSpecificationElement);
    }


    public void addConfiguredAssumevalues(ClassSpecificationElement classSpecificationElement)
    {
        configuration.assumeValues = extendClassSpecifications(configuration.assumeValues,
                                                               classSpecificationElement);
    }


    public void addConfiguredOptimizations(FilterElement filterElement)
    {
        addConfiguredOptimization(filterElement);
    }


    public void addConfiguredOptimization(FilterElement filterElement)
    {
        configuration.optimizations = extendFilter(configuration.optimizations,
                                                   filterElement);
    }


    public void addConfiguredKeeppackagename(FilterElement filterElement)
    {
        configuration.keepPackageNames = extendFilter(configuration.keepPackageNames,
                                                      filterElement,
                                                      true);
    }


    public void addConfiguredKeeppackagenames(FilterElement filterElement)
    {
        configuration.keepPackageNames = extendFilter(configuration.keepPackageNames,
                                                      filterElement,
                                                      true);
    }


    public void addConfiguredKeepattributes(FilterElement filterElement)
    {
        addConfiguredKeepattribute(filterElement);
    }


    public void addConfiguredKeepattribute(FilterElement filterElement)
    {
        configuration.keepAttributes = extendFilter(configuration.keepAttributes,
                                                    filterElement);
    }


    public void addConfiguredAdaptclassstrings(FilterElement filterElement)
    {
        configuration.adaptClassStrings = extendFilter(configuration.adaptClassStrings,
                                                       filterElement, true);
    }


    public void addConfiguredAdaptresourcefilenames(FilterElement filterElement)
    {
        configuration.adaptResourceFileNames = extendFilter(configuration.adaptResourceFileNames,
                                                            filterElement);
    }


    public void addConfiguredAdaptresourcefilecontents(FilterElement filterElement)
    {
        configuration.adaptResourceFileContents = extendFilter(configuration.adaptResourceFileContents,
                                                               filterElement);
    }


    public void addConfiguredDontnote(FilterElement filterElement)
    {
        configuration.note = extendFilter(configuration.note, filterElement, true);
    }


    public void addConfiguredDontwarn(FilterElement filterElement)
    {
        configuration.warn = extendFilter(configuration.warn, filterElement, true);
    }


    public void addConfiguredConfiguration(ConfigurationElement configurationElement)
    {
        configurationElement.appendTo(configuration);
    }


    // Implementations for Task.

    public void addText(String text) throws BuildException
    {
        try
        {
            Project project = getProject();

            // Replace Ant-style properties ('${...}').
            String arg = project.replaceProperties(text);

            // Get the combined system properties and Ant properties, for
            // replacing ProGuard-style properties ('<...>').
            Properties properties = new Properties();
            properties.putAll(project.getProperties());

            ConfigurationParser parser = new ConfigurationParser(arg,
                                                                 "embedded configuration",
                                                                 project.getBaseDir(),
                                                                 properties);

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
        catch (IOException ex)
        {
            throw new BuildException(ex.getMessage());
        }
    }


    // Small utility methods.

    private ClassPath extendClassPath(ClassPath        classPath,
                                      ClassPathElement classPathElement,
                                      boolean          output)
    {
        if (classPath == null)
        {
            classPath = new ClassPath();
        }

        classPathElement.appendClassPathEntriesTo(classPath,
                                                  output);

        return classPath;
    }


    private ClassPath extendClassPath(ClassPath classPath,
                                      ClassPath additionalClassPath)
    {
        if (additionalClassPath != null)
        {
            if (classPath == null)
            {
                classPath = new ClassPath();
            }

            classPath.addAll(additionalClassPath);
        }

        return classPath;
    }


    private List extendKeepSpecifications(List                     keepSpecifications,
                                          KeepSpecificationElement keepSpecificationElement,
                                          boolean                  markClasses,
                                          boolean                  markClassesConditionally)
    {
        if (keepSpecifications == null)
        {
            keepSpecifications = new ArrayList();
        }

        keepSpecificationElement.appendTo(keepSpecifications,
                                          markClasses,
                                          markClassesConditionally);

        return keepSpecifications;
    }


    private List extendClassSpecifications(List                      classSpecifications,
                                           ClassSpecificationElement classSpecificationElement)
    {
        if (classSpecifications == null)
        {
            classSpecifications = new ArrayList();
        }

        classSpecificationElement.appendTo(classSpecifications);

        return classSpecifications;
    }


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


    private List extendFilter(List          filter,
                              FilterElement filterElement)
    {
        return extendFilter(filter, filterElement, false);
    }


    private List extendFilter(List          filter,
                              FilterElement filterElement,
                              boolean       internal)
    {
        if (filter == null)
        {
            filter = new ArrayList();
        }

        filterElement.appendTo(filter, internal);

        return filter;
    }


    private List extendList(List list,
                            List additionalList)
    {
        if (additionalList != null)
        {
            if (list == null)
            {
                list = new ArrayList();
            }

            list.addAll(additionalList);
        }

        return list;
    }
}

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

import org.apache.tools.ant.BuildException;
import proguard.*;
import proguard.classfile.util.ClassUtil;

import java.io.*;
import java.net.*;
import java.util.*;

/**
 * This Task allows to configure and run ProGuard from Ant.
 *
 * @author Eric Lafortune
 */
public class ProGuardTask extends ConfigurationTask
{
    // Ant task attributes.

    public void setConfiguration(File configurationFile) throws BuildException
    {
        try
        {
            // Get the combined system properties and Ant properties, for
            // replacing ProGuard-style properties ('<...>').
            Properties properties = new Properties();
            properties.putAll(getProject().getProperties());

            URL configUrl =
                ConfigurationElement.class.getResource(configurationFile.toString());

            ConfigurationParser parser = configUrl != null ?
                new ConfigurationParser(configUrl, properties) :
                new ConfigurationParser(configurationFile, properties);

            try
            {
                parser.parse(configuration);
            }
            catch (ParseException e)
            {
                throw new BuildException(e.getMessage(), e);
            }
            finally
            {
                parser.close();
            }
        }
        catch (IOException e)
        {
            throw new BuildException(e.getMessage(), e);
        }
    }


    /**
     * @deprecated Use the nested outjar element instead.
     */
    public void setOutjar(String parameters)
    {
        throw new BuildException("Use the <outjar> nested element instead of the 'outjar' attribute");
    }


    public void setSkipnonpubliclibraryclasses(boolean skipNonPublicLibraryClasses)
    {
        configuration.skipNonPublicLibraryClasses = skipNonPublicLibraryClasses;
    }


    public void setSkipnonpubliclibraryclassmembers(boolean skipNonPublicLibraryClassMembers)
    {
        configuration.skipNonPublicLibraryClassMembers = skipNonPublicLibraryClassMembers;
    }


    public void setTarget(String target)
    {
        configuration.targetClassVersion = ClassUtil.internalClassVersion(target);
        if (configuration.targetClassVersion == 0)
        {
            throw new BuildException("Unsupported target '"+target+"'");
        }
    }


    public void setForceprocessing(boolean forceProcessing)
    {
        configuration.lastModified = forceProcessing ? Long.MAX_VALUE : 0;
    }


    public void setPrintseeds(File printSeeds)
    {
        configuration.printSeeds = optionalFile(printSeeds);
    }


    public void setShrink(boolean shrink)
    {
        configuration.shrink = shrink;
    }


    public void setPrintusage(File printUsage)
    {
        configuration.printUsage = optionalFile(printUsage);
    }


    public void setOptimize(boolean optimize)
    {
        configuration.optimize = optimize;
    }


    public void setOptimizationpasses(int optimizationPasses)
    {
        configuration.optimizationPasses = optimizationPasses;
    }


    public void setAllowaccessmodification(boolean allowAccessModification)
    {
        configuration.allowAccessModification = allowAccessModification;
    }


    public void setMergeinterfacesaggressively(boolean mergeinterfacesaggressively)
    {
        configuration.mergeInterfacesAggressively = mergeinterfacesaggressively;
    }


    public void setObfuscate(boolean obfuscate)
    {
        configuration.obfuscate = obfuscate;
    }


    public void setPrintmapping(File printMapping)
    {
        configuration.printMapping = optionalFile(printMapping);
    }


    public void setApplymapping(File applyMapping)
    {
        configuration.applyMapping = resolvedFile(applyMapping);
    }


    public void setObfuscationdictionary(File obfuscationDictionary)
    {
        configuration.obfuscationDictionary = resolvedURL(obfuscationDictionary);
    }


    public void setClassobfuscationdictionary(File classObfuscationDictionary)
    {
        configuration.classObfuscationDictionary = resolvedURL(classObfuscationDictionary);
    }


    public void setPackageobfuscationdictionary(File packageObfuscationDictionary)
    {
        configuration.packageObfuscationDictionary = resolvedURL(packageObfuscationDictionary);
    }


    public void setOverloadaggressively(boolean overloadAggressively)
    {
        configuration.overloadAggressively = overloadAggressively;
    }


    public void setUseuniqueclassmembernames(boolean useUniqueClassMemberNames)
    {
        configuration.useUniqueClassMemberNames = useUniqueClassMemberNames;
    }


    public void setUsemixedcaseclassnames(boolean useMixedCaseClassNames)
    {
        configuration.useMixedCaseClassNames = useMixedCaseClassNames;
    }


    public void setFlattenpackagehierarchy(String flattenPackageHierarchy)
    {
        configuration.flattenPackageHierarchy = ClassUtil.internalClassName(flattenPackageHierarchy);
    }


    public void setRepackageclasses(String repackageClasses)
    {
        configuration.repackageClasses = ClassUtil.internalClassName(repackageClasses);
    }

    /**
     * @deprecated Use the repackageclasses attribute instead.
     */
    public void setDefaultpackage(String defaultPackage)
    {
        configuration.repackageClasses = ClassUtil.internalClassName(defaultPackage);
    }


    public void setKeepparameternames(boolean keepParameterNames)
    {
        configuration.keepParameterNames = keepParameterNames;
    }


    public void setRenamesourcefileattribute(String newSourceFileAttribute)
    {
        configuration.newSourceFileAttribute = newSourceFileAttribute;
    }


    public void setPreverify(boolean preverify)
    {
        configuration.preverify = preverify;
    }


    public void setMicroedition(boolean microEdition)
    {
        configuration.microEdition = microEdition;
    }


    public void setAndroid(boolean android)
    {
        configuration.android = android;
    }


    public void setVerbose(boolean verbose)
    {
        configuration.verbose = verbose;
    }


    public void setNote(boolean note)
    {
        if (note)
        {
            // Switch on notes if they were completely disabled.
            if (configuration.note != null &&
                configuration.note.isEmpty())
            {
                configuration.note = null;
            }
        }
        else
        {
            // Switch off notes.
            configuration.note = new ArrayList();
        }
    }


    public void setWarn(boolean warn)
    {
        if (warn)
        {
            // Switch on warnings if they were completely disabled.
            if (configuration.warn != null &&
                configuration.warn.isEmpty())
            {
                configuration.warn = null;
            }
        }
        else
        {
            // Switch off warnings.
            configuration.warn = new ArrayList();
        }
    }


    public void setIgnorewarnings(boolean ignoreWarnings)
    {
        configuration.ignoreWarnings = ignoreWarnings;
    }


    public void setPrintconfiguration(File printConfiguration)
    {
        configuration.printConfiguration = optionalFile(printConfiguration);
    }


    public void setDump(File dump)
    {
        configuration.dump = optionalFile(dump);
    }


    public void setAddConfigurationDebugging(boolean addConfigurationDebugging)
    {
        configuration.addConfigurationDebugging = addConfigurationDebugging;
    }



    // Implementations for Task.

    public void execute() throws BuildException
    {
        try
        {
            ProGuard proGuard = new ProGuard(configuration);
            proGuard.execute();
        }
        catch (IOException e)
        {
            throw new BuildException(e.getMessage(), e);
        }
    }


    // Small utility methods.

    /**
     * Returns a file that is properly resolved with respect to the project
     * directory, or <code>null</code> or empty if its name is actually a
     * boolean flag.
     */
    private File optionalFile(File file)
    {
        String fileName = file.getName();

        return
            fileName.equalsIgnoreCase("false") ||
            fileName.equalsIgnoreCase("no")    ||
            fileName.equalsIgnoreCase("off")    ? null :
            fileName.equalsIgnoreCase("true")  ||
            fileName.equalsIgnoreCase("yes")   ||
            fileName.equalsIgnoreCase("on")     ? Configuration.STD_OUT :
                                                  resolvedFile(file);
    }


    /**
     * Returns a URL that is properly resolved with respect to the project
     * directory.
     */
    private URL resolvedURL(File file)
    {
        try
        {
            return resolvedFile(file).toURI().toURL();
        }
        catch (MalformedURLException e)
        {
            return null;
        }
    }


    /**
     * Returns a file that is properly resolved with respect to the project
     * directory.
     */
    private File resolvedFile(File file)
    {
        return file.isAbsolute() ? file :
                                   new File(getProject().getBaseDir(),
                                            file.getName());
    }
}

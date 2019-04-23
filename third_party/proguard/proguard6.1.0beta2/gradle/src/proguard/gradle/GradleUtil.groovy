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
package proguard.gradle

import com.android.build.gradle.api.*
import com.android.build.gradle.internal.tasks.FileSupplier
import com.android.build.gradle.internal.scope.TaskOutputHolder;
import com.android.builder.core.AndroidBuilder
import com.android.builder.model.*
import org.gradle.api.*

import static com.android.builder.model.AndroidProject.FD_OUTPUTS


/**
 * Utility functions.
 */
class GradleUtil
{
    static int AGP_VERSION_MAJOR
    static int AGP_VERSION_MINOR

    static {
        (AGP_VERSION_MAJOR, AGP_VERSION_MINOR) = getAndroidPluginVersion()
    }


    static File getAaptRulesFile(ApkVariant variant, Project project)
    {
        return project.file("${project.buildDir}/intermediates/proguard/${variant.dirName}/aapt_rules.txt")
    }


    /**
     * Constructs the name for a gradle task given the current variant name. The convention of this name is in line
     * with the convention of other gradle task names.
     */
    static String createVariantTaskName(String variantName)
    {
        return variantName.split('-')
                          .inject(new StringBuilder()) { StringBuilder result, String part ->
            result.append !part.matches('v[0-9][a-z]') || variantName.startsWith(part) ?
                          part.capitalize() :
                          "-${part}"
        }.toString()
    }


    /**
     * Collects and returns all Proguard files specified by the current variant of the main project and all its flavors.
     */
    static Collection<File> getProguardFiles(BaseVariant variant)
    {
        List<File> fullList = new ArrayList<File>()

        // add the config files from the build type, main config and flavors
        fullList.addAll(variant.variantData.variantConfiguration.getDefaultConfig().getProguardFiles())
        fullList.addAll(variant.variantData.variantConfiguration.getBuildType().getProguardFiles())

        for (ProductFlavor flavor : variant.variantData.variantConfiguration.getProductFlavors())
        {
            fullList.addAll(flavor.getProguardFiles())
        }

        return fullList
    }


    /**
     * Collects and returns all Proguard files specified by the current variant of the main project and all its flavors
     * and all Proguard files specified by the libraries used by the current variant.
     */
    static Collection<File> getAllProguardFiles(BaseVariant variant)
    {
        List<File> fullList = new ArrayList<File>()

        fullList.addAll(getProguardFiles(variant))
        fullList.addAll(getConsumerProguardFiles(variant).values())

        return fullList
    }


    /**
     * Collects and returns the Proguard files used for each library used by the current variant of the main project.
     */
    private static Map<MyMavenCoordinates, File> getConsumerProguardFiles(BaseVariant variant)
    {
        List libraries
        Map<MyMavenCoordinates, File> consumerFileMap = new HashMap<>()

        try
        {
            try
            {
                libraries = variant.variantData.variantConfiguration.getAllLibraries()
            }
            catch (Exception e)
            {
                // The API of VariantConfiguration has changed with
                // version 2.2.0-alpha1 of the android plugin.
                // Access the new method via reflection.
                libraries = variant.variantData.variantConfiguration."getFlatPackageAndroidLibraries"()
            }

            for (Object libraryDependency : libraries)
            {
                // Starting from version 2.3.0-beta1, the type representing
                // an Android library has changed from AndroidLibrary to
                // AndroidDependency. Access the method by reflection to
                // handle different plugin versions.

                File proguardRules = libraryDependency.getProguardRules()
                MavenCoordinates mavenCoordinates

                try
                {
                    mavenCoordinates = libraryDependency.getResolvedCoordinates()
                }
                catch (Exception ex)
                {
                    mavenCoordinates = libraryDependency.getCoordinates()
                }

                MyMavenCoordinates coords = new MyMavenCoordinates(mavenCoordinates.groupId,
                                                                   mavenCoordinates.artifactId,
                                                                   mavenCoordinates.version)

                if (proguardRules.exists())
                {
                    consumerFileMap.put(coords, proguardRules)
                }
            }
        }
        catch (Exception e)
        {
            // They changed the enumeration from PROGUARD_RULES to CONSUMER_PROGUARD_RULES in AGP 3.2.
            def artifactType
            try
            {
                artifactType = com.android.build.gradle.internal.publishing.AndroidArtifacts.ArtifactType.PROGUARD_RULES
            }
            catch (Exception e2)
            {
                artifactType = com.android.build.gradle.internal.publishing.AndroidArtifacts.ArtifactType.CONSUMER_PROGUARD_RULES
            }
            // This is similar to how Proguard does it, we add some indirection to get the library maven coordinates.
            def artifactCollection = variant.variantData.scope.getArtifactCollection(com.android.build.gradle.internal.publishing.AndroidArtifacts.ConsumedConfigType.RUNTIME_CLASSPATH,
                                                                                     com.android.build.gradle.internal.publishing.AndroidArtifacts.ArtifactScope.ALL,
                                                                                     artifactType)

            for (def artifact : artifactCollection)
            {
                File               proguardRules   = artifact.getFile()
                String             componentId     = artifact.getId().getComponentIdentifier().toString()
                String[]           coordComponents = componentId.split(':')
                MyMavenCoordinates coords
                if (componentId.startsWith('project ') || coordComponents.length == 1)
                {
                    coords = new MyMavenCoordinates('',
                                                    coordComponents.last(),
                                                    '')
                }
                else
                {
                    coords = new MyMavenCoordinates(coordComponents[0],
                                                    coordComponents[1],
                                                    coordComponents.length > 2 ? coordComponents[2] : '')
                }

                if (proguardRules.exists())
                {
                    consumerFileMap.put(coords, proguardRules)
                }
            }
        }

        return consumerFileMap
    }


    static def getAndroidPluginVersion()
    {
        String[] versionSegments
        // The Version class was moved from the com.android.builder package to the com.android.builder.model package
        // in the Android gradle plugin v3.1.0.
        try
        {
            versionSegments = com.android.builder.model.Version.ANDROID_GRADLE_PLUGIN_VERSION.split('\\.')
        }
        catch (Exception e)
        {
            try
            {
                versionSegments = com.android.builder.Version.ANDROID_GRADLE_PLUGIN_VERSION.split('\\.')
            }
            catch (Exception e2)
            {
                throw new GradleException("Unsupported Android Gradle plugin version.", e)
            }
        }
        final int versionMajor = Integer.parseInt(versionSegments[0])
        final int versionMinor = Integer.parseInt(versionSegments[1])

        return [versionMajor, versionMinor]
    }


    /**
     * Because of the interface change of the AndroidBuilder class in the
     * 1.4.0-beta2 release of the Android Gradle plugin we use this wrapper to get
     * to the bootclasspath.
     */
    static List<File> getBootClasspathWorkaround(AndroidBuilder androidBuilder)
    {
        try
        {
            return androidBuilder.getBootClasspath(true)
        }
        catch (Exception ex)
        {
            return androidBuilder.bootClasspath
        }
    }


    static File getMappingDir(Project project, BaseVariant apkVariant)
    {
        return project.file("${project.buildDir}/${FD_OUTPUTS}/mapping/${apkVariant.dirName}")
    }


    static void setMappingFile(Project project, BaseVariant variant, File mappingFile, Task proguardTask)
    {
        try
        {
            // This is the new way to add a mapping file to Android.
            // Since version X.X of the Android gradle plugin.
            variant.variantData.scope.addTaskOutput(TaskOutputHolder.TaskOutputType.APK_MAPPING,
                                                    mappingFile,
                                                    proguardTask.name)
        }
        catch (Throwable t)
        {
            // We catch a Throwable because TaskOutputType doesn't exist before v3.0 of the Android plugin
            // and that causes a NoClassDefFoundError, which is an Error, not an Exception.
            try
            {
                // We set the mapping file the v2.3 way.
                // Groovy doesn't support anonymous interfaces, so we coerce a map.
                variant.variantData.mappingFileProviderTask = [
                    getTask: { return proguardTask },
                    get    : { return mappingFile }
                ] as FileSupplier
            }
            catch (Exception ex)
            {
                try
                {
                    // Try to set the mapping file.
                    // The variantData are protected and internal.
                    variant.variantData.mappingFile = mappingFile
                }
                catch (Exception)
                {
                    project.getLogger().warn("ProGuard could not set mapping file for variant '${variant.name}'")

                }
            }
        }
    }


    // Helper classes.

    private static class MyMavenCoordinates
    {
        private String groupId;
        private String artifactId;
        private String version;

        MyMavenCoordinates(String groupId, String artifactId, String version)
        {
            this.groupId    = groupId;
            this.artifactId = artifactId;
            this.version    = version;
        }

        @Override
        String toString()
        {
            return (groupId    != null ? groupId    : "") + ":" +
                   (artifactId != null ? artifactId : "") + ":" +
                   (version    != null ? version    : "");
        }
    }
}

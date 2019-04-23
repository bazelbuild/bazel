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

import com.android.build.gradle.AppExtension
import com.android.build.gradle.BasePlugin
import com.android.build.gradle.api.*
import com.android.build.gradle.internal.BadPluginException
import org.gradle.api.*

import static GradleUtil.*;


/**
 * This Plugin installs a DexGuard transform in an Android application project.
 *
 * @author Thomas Neidhart
 */
class      ProGuardPlugin
implements Plugin<Project>
{
    // Implementations for Plugin.

    @Override
    void apply(Project project)
    {
        if (!project.hasProperty('android'))
        {
            throw new BadPluginException('The ProGuard plugin requires the Android plugin to function properly.\n' +
                                         'Please specify\n    apply plugin: \'com.android.application\'\n    apply plugin: \'proguard\'')
        }

        // Add the extra method 'getTunedProGuardFile' to the project.
        project.convention.plugins.extraProGuardMethods = new ProGuardConvention()

        def extension = project.extensions.create('proguard', ProGuardExtension)

        // A transform has to be registered before the project is evaluated.
        // This forces us to implement an identity transform in case it is
        // disabled for a given variant.
        BasePlugin androidPlugin =
            (BasePlugin)project.plugins.findPlugin('com.android.application')

        // Register our transform.
        ProGuardTransform transform = new ProGuardTransform(project, extension, androidPlugin)

        def android = project.extensions.getByType(AppExtension)
        android.registerTransform(transform)

        project.afterEvaluate
        {
            // Go over all application variants.
            project.android.applicationVariants.each
            {
                ApkVariant apkVariant ->

                    boolean proguardFilesConfigured = !getProguardFiles(apkVariant).isEmpty()
                    // TODO: minifyEnabled is deprecated but there is not yet an alternative available
                    //       until the DSL has changed, keep using it for now.
                    boolean minifyEnabled = apkVariant.buildType.minifyEnabled
                    if (minifyEnabled)
                    {
                        project.getLogger().warn("minifyEnabled detected for variant '${apkVariant.name}'.\n" +
                                                 "Please disable this option in your build.gradle file.")
                    }

                    if (proguardFilesConfigured && !minifyEnabled)
                    {
                        transform.enableVariant(apkVariant)

                        // Make sure that aapt generates a ProGuard rules file.
                        def variantName = createVariantTaskName(apkVariant.name)
                        def processResources =
                            project.tasks.getByName("process${variantName}Resources")

                        if (processResources != null)
                        {
                            processResources.conventionMapping.proguardOutputFile = {
                                getAaptRulesFile(apkVariant, project)
                            }
                        }
                        else
                        {
                            throw new GradleException("Unsupported Android Gradle plugin version.")
                        }
                    }
            }
        }
    }
}

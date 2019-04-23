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

import com.android.build.api.transform.*
import com.android.build.gradle.api.*
import com.android.build.gradle.BasePlugin
import com.google.common.collect.ImmutableMap
import org.gradle.api.*
import org.gradle.api.file.FileCollection
import org.gradle.api.logging.LogLevel

import static GradleUtil.*;

import java.security.MessageDigest

/**
 * Transform input class files and external libraries with ProGuard.
 */
class   ProGuardTransform
extends Transform
{
    private final Project           project
    private final ProGuardExtension transformExtension
    private final BasePlugin        androidPlugin
    private final Set<ApkVariant>   variants = new HashSet<>()


    ProGuardTransform(Project           project,
                      ProGuardExtension transformExtension,
                      BasePlugin        androidPlugin)
    {
        this.project            = project
        this.transformExtension = transformExtension
        this.androidPlugin      = androidPlugin
    }


    void enableVariant(ApkVariant variant)
    {
        variants.add(variant)
    }


    @Override
    void transform(TransformInvocation transformInvocation)
    throws TransformException, InterruptedException, IOException
    {
        transform(transformInvocation.context,
                  transformInvocation.inputs,
                  transformInvocation.referencedInputs,
                  transformInvocation.outputProvider,
                  transformInvocation.incremental)
    }


    @Override
    void transform(Context                    context,
                   Collection<TransformInput> inputs,
                   Collection<TransformInput> referencedInputs,
                   TransformOutputProvider    outputProvider,
                   boolean                    isIncremental)
    throws IOException, TransformException, InterruptedException
    {
        context.logging.captureStandardOutput(LogLevel.INFO)

        ApkVariant apkVariant = variants.find { variant -> variant.name == context.variantName }

        if (apkVariant == null)
        {
            // The transform is not enabled for this variant
            // Perform an identity transform, i.e. copy all inputs unchanged to the output
            identityTransform(inputs, outputProvider, isIncremental)
            return
        }

        // Create a dummy task in order to execute ProGuard.
        // Gradle prevents instantiation of tasks, thus use create.
        // DO NOT CALL task.execute() yourself as explained in the gradle guide.
        // This task is created here as the mapping file cannot be set in the transform.
        def proguardTask =
            project.tasks.create("proguardTransform${createVariantTaskName(apkVariant.name)}",
                                 ProGuardTask)

        def transformedInputs = new HashSet<>()

        // Collect the project input that shall be processed.
        inputs*.directoryInputs*.each {
            DirectoryInput directoryInput ->

                def name = AGP_VERSION_MAJOR < 3 ?
                           hashMD5(directoryInput.file.absolutePath) :
                           directoryInput.file.absolutePath

                def outputDir = outputProvider.getContentLocation(name,
                                                                  outputTypes,
                                                                  EnumSet.of(QualifiedContent.Scope.PROJECT),
                                                                  Format.DIRECTORY)

                StringBuilder inputFilter = new StringBuilder()

                if (isIncremental)
                {
                    processChangedFiles(directoryInput, outputDir, {
                        File file ->
                            inputFilter.append(directoryInput.file.toPath().relativize(file.toPath()).toString())
                            inputFilter.append(',')
                    })

                    if (inputFilter.length() > 0)
                    {
                        inputFilter.deleteCharAt(inputFilter.length() - 1)
                    }
                }
                else
                {
                    inputFilter.append('**.class')
                }

                transformedInputs.add(directoryInput.file.absolutePath)
                proguardTask.injars (filter: inputFilter.toString(), directoryInput.file.absolutePath)
                proguardTask.outjars(outputDir.absolutePath)
        }

        // Collect external libraries that shall be processed.
        // Each library will stored to a separate jar file.
        inputs*.jarInputs*.each {
            JarInput jarInput ->

                def name = AGP_VERSION_MAJOR < 3 ?
                           hashMD5(jarInput.file.absolutePath) :
                           jarInput.file.absolutePath

                def outputJar =
                    outputProvider.getContentLocation(name,
                                                      outputTypes,
                                                      EnumSet.of(QualifiedContent.Scope.SUB_PROJECTS,
                                                                 QualifiedContent.Scope.EXTERNAL_LIBRARIES),
                                                      Format.JAR)

                def copy = isIncremental ?
                           processChangedJarFile(jarInput, outputJar) :
                           true

                if (copy)
                {
                    transformedInputs.add(jarInput.file.absolutePath)
                    proguardTask.injars (filter: '**.class', jarInput.file.absolutePath)
                    proguardTask.outjars(outputJar.absolutePath)
                }
        }

        // Collect the needed library jars.
        getClasspath(apkVariant, transformedInputs, referencedInputs).each {
            proguardTask.libraryjars(it.absolutePath)
        }

        // Collect the configuration files.
        proguardTask.configuration getAllProguardFiles(apkVariant)
        proguardTask.configuration getAaptRulesFile(apkVariant, project)

        // Specify the mapping/seeds/usage files.
        File mappingDir  = getMappingDir(project, apkVariant)
        mappingDir.mkdirs()
        File mappingFile = new File(mappingDir, 'mapping.txt')

        proguardTask.printseeds   new File(mappingDir, 'seeds.txt')
        proguardTask.printusage   new File(mappingDir, 'usage.txt')
        proguardTask.printmapping mappingFile

        setMappingFile(project, apkVariant, mappingFile, proguardTask)

        // Specify the target class version based on the used minSdkVersion
        proguardTask.target(getTargetClassVersion(apkVariant))

        // If we transform plugin is configured not to transform external
        // library, we must not shrink the library classes to fully support
        // all backport features. This is actually a hack but there seems
        // to be no other way to prevent the initial shrinking.
        if (!getScopes().contains(QualifiedContent.Scope.EXTERNAL_LIBRARIES))
        {
            proguardTask.useuniqueclassmembernames()
        }

        // Hardcode for Android (the release already has it, but it is needed for development versions)
        proguardTask.android()

        // Execute ProGuard.
        proguardTask.proguard()
    }


    private void identityTransform(Collection<TransformInput> inputs,
                                   TransformOutputProvider    outputProvider,
                                   boolean                    isIncremental)
    {
        // Copy all directories.
        inputs*.directoryInputs*.each {
            DirectoryInput directoryInput ->

                def name = AGP_VERSION_MAJOR < 3 ?
                           hashMD5(directoryInput.file.absolutePath) :
                           directoryInput.file.absolutePath

                def outputDir = outputProvider.getContentLocation(name,
                                                                  outputTypes,
                                                                  EnumSet.of(QualifiedContent.Scope.PROJECT),
                                                                  Format.DIRECTORY)

                Closure includeFile = { true }

                if (isIncremental)
                {
                    FileCollection changedFiles = project.files()

                    processChangedFiles(directoryInput, outputDir, {
                        File file -> changedFiles += project.files(file)
                    })

                    includeFile = { changedFiles.contains(it) }
                }

                project.copy {
                    it.from (directoryInput.file) {
                        it.include includeFile
                    }
                    it.into outputDir
                }
        }

        // Copy all external libraries.
        inputs*.jarInputs*.each {
            JarInput jarInput ->

                def name = AGP_VERSION_MAJOR < 3 ?
                           hashMD5(jarInput.file.absolutePath) :
                           jarInput.file.absolutePath

                def outputJar = outputProvider.getContentLocation(name,
                                                                  outputTypes,
                                                                  EnumSet.of(QualifiedContent.Scope.SUB_PROJECTS,
                                                                             QualifiedContent.Scope.EXTERNAL_LIBRARIES),
                                                                  Format.JAR)

                def copy = isIncremental ?
                           processChangedJarFile(jarInput, outputJar) :
                           true

                if (copy)
                {
                    project.copy {
                        it.from jarInput.file
                        it.into outputJar.parent
                        it.rename('(.*)', outputJar.absolutePath)
                    }
                }
        }
    }


    private void processChangedFiles(DirectoryInput directoryInput, File outputDir, Closure fileChangedClosure)
    {
        for (Map.Entry<File, Status> entry : directoryInput.changedFiles)
        {
            File   file   = entry.key;
            Status status = entry.value

            if (status == Status.ADDED ||
                status == Status.CHANGED)
            {
                fileChangedClosure file
            }

            if (status == Status.CHANGED ||
                status == Status.REMOVED)
            {
                File output = toOutput(directoryInput.file, outputDir, file)
                output.delete()
                deleteRelated(output)
            }
        }
    }


    private boolean processChangedJarFile(JarInput jarInput, File outputJar)
    {
        File    file   = jarInput.file
        Status  status = jarInput.status
        boolean fileNeedsUpdate = false

        if (status == Status.ADDED ||
            status == Status.CHANGED)
        {
            fileNeedsUpdate = true
        }

        if (status == Status.CHANGED ||
            status == Status.REMOVED)
        {
            File output = toOutput(file, outputJar, file)
            output.delete()
        }

        return fileNeedsUpdate
    }


    private static File toOutput(File inputDir, File outputDir, File file)
    {
        return outputDir.toPath().resolve(inputDir.toPath().relativize(file.toPath())).toFile()
    }


    private static void deleteRelated(File file)
    {
        def className = file.name.replaceFirst(/\.class$/, '')
        // Delete any generated Lambda or Util classes.
        file.parentFile.eachFile {
            if (it.name.matches(/$className\$\$/ + /Lambda.*\.class$/) ||
                it.name.matches(/$className\$\$/ + /Util.*\.class$/))
            {
                it.delete()
            }
        }
    }


    private def hashMD5(String s)
    {
        MessageDigest.getInstance("MD5").digest(s.bytes).encodeHex().toString()
    }


    private FileCollection getClasspath(BaseVariant                variant,
                                        Set<String>                transformedInputs,
                                        Collection<TransformInput> referencedInputs)
    {
        FileCollection classpathFiles = variant.javaCompiler.classpath
        for (TransformInput input : referencedInputs)
        {
            classpathFiles += project.files(input.directoryInputs*.file)
        }

        getBootClasspathWorkaround(androidPlugin.androidBuilder).each {
            classpathFiles += project.files(it)
        }

        // Filter out transformed jar files to avoid duplicate jar file error when
        // processing with DexGuard.
        return classpathFiles.filter { File file -> !transformedInputs.contains(file.absolutePath) }
    }


    private static String getTargetClassVersion(ApkVariant variant)
    {
        def minSdkVersion
        try
        {
            minSdkVersion = variant.packageApplication.minSdkVersion
        }
        catch (Exception e)
        {
            ApkVariantOutput variantOutput = variant.outputs.first()
            minSdkVersion = variantOutput.packageApplication.minSdkVersion
        }

        if (minSdkVersion < 19)
        {
            return "1.6"
        }
        else if (minSdkVersion < 24)
        {
            return "1.7"
        }
        else
        {
            return "1.8"
        }
    }


    @Override
    String getName()
    {
        return "ProGuardTransform"
    }


    @Override
    Set<QualifiedContent.ContentType> getInputTypes()
    {
        return EnumSet.of(QualifiedContent.DefaultContentType.CLASSES)
    }


    @Override
    Set<QualifiedContent.Scope> getScopes()
    {
        List<QualifiedContent.Scope> scopes = new ArrayList<>()
        scopes.add(QualifiedContent.Scope.PROJECT)

        if (transformExtension.transformExternalLibraries())
        {
            scopes.add(QualifiedContent.Scope.EXTERNAL_LIBRARIES)
        }

        if (transformExtension.transformSubprojects())
        {
            scopes.add(QualifiedContent.Scope.SUB_PROJECTS)
        }

        return EnumSet.copyOf(scopes);
    }


    @Override
    Set<QualifiedContent.Scope> getReferencedScopes()
    {
        List<QualifiedContent.Scope> scopes = new ArrayList<>()

        if (!transformExtension.transformExternalLibraries())
        {
            scopes.add(QualifiedContent.Scope.EXTERNAL_LIBRARIES)
        }

        if (!transformExtension.transformSubprojects())
        {
            scopes.add(QualifiedContent.Scope.SUB_PROJECTS)
        }

        scopes.add(QualifiedContent.Scope.TESTED_CODE)

        return EnumSet.copyOf(scopes);
    }


    @Override
    Map<String, Object> getParameterInputs()
    {
        return ImmutableMap.builder()
                           .put("incremental",                transformExtension.incremental)
                           .put("transformExternalLibraries", transformExtension.transformExternalLibraries)
                           .put("transformSubprojects",       transformExtension.transformSubprojects)
                           .build()
    }


    @Override
    boolean isIncremental()
    {
        return transformExtension.isIncremental()
    }
}

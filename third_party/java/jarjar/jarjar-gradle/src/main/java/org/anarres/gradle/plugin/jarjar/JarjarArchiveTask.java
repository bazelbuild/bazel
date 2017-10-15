/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.anarres.gradle.plugin.jarjar;

import groovy.lang.Closure;
import javax.annotation.Nonnull;
import org.gradle.api.internal.DocumentationRegistry;
import org.gradle.api.internal.file.copy.CopyAction;
import org.gradle.api.tasks.bundling.Jar;

/**
 *
 * @author shevek
 */
public class JarjarArchiveTask extends Jar {

    @Override
    protected CopyAction createCopyAction() {
        DocumentationRegistry documentationRegistry = getServices().get(DocumentationRegistry.class);
        return new JarjarCopyAction(getArchivePath(), getCompressor(), documentationRegistry);
    }

    public void fromJar(@Nonnull Object... sourcePaths) {
    }

    public void fromJar(@Nonnull Object sourcePath, @Nonnull Closure c) {
    }

}

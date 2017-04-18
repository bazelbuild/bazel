/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.anarres.gradle.plugin.jarjar;

import java.io.File;
import java.io.FileOutputStream;
import javax.annotation.Nonnull;
import org.apache.commons.compress.archivers.jar.JarArchiveEntry;
import org.apache.commons.compress.archivers.jar.JarArchiveOutputStream;
import org.apache.commons.compress.archivers.zip.UnixStat;
import org.gradle.api.Action;
import org.gradle.api.GradleException;
import org.gradle.api.UncheckedIOException;
import org.gradle.api.file.FileCopyDetails;
import org.gradle.api.internal.DocumentationRegistry;
import org.gradle.api.internal.file.CopyActionProcessingStreamAction;
import org.gradle.api.internal.file.copy.CopyAction;
import org.gradle.api.internal.file.copy.CopyActionProcessingStream;
import org.gradle.api.internal.file.copy.FileCopyDetailsInternal;
import org.gradle.api.internal.file.copy.ZipCompressor;
import org.gradle.api.internal.tasks.SimpleWorkResult;
import org.gradle.api.tasks.WorkResult;
import org.gradle.api.tasks.bundling.Zip;
import org.gradle.api.tasks.bundling.internal.Zip64RequiredException;
import org.gradle.internal.IoActions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Based on ZipCopyAction from Gradle sources.
 *
 * @author shevek
 */
public class JarjarCopyAction implements CopyAction {

    private static final Logger LOG = LoggerFactory.getLogger(JarjarCopyAction.class);
    private final File zipFile;
    // private final ZipCompressor compressor;
    private final DocumentationRegistry documentationRegistry;

    public JarjarCopyAction(@Nonnull File zipFile, @Nonnull ZipCompressor compressor, @Nonnull DocumentationRegistry documentationRegistry) {
        this.zipFile = zipFile;
        // this.compressor = compressor;
        this.documentationRegistry = documentationRegistry;
    }

    @Nonnull
    @Override
    public WorkResult execute(@Nonnull final CopyActionProcessingStream stream) {
        LOG.info("CopyAction Executing  " + stream);

        stream.process(new ScanAction());

        final JarArchiveOutputStream zipOutStr;

        try {
            zipOutStr = new JarArchiveOutputStream(new FileOutputStream(zipFile));
        } catch (Exception e) {
            throw new GradleException(String.format("Could not create ZIP '%s'.", zipFile), e);
        }

        try {
            IoActions.withResource(zipOutStr, new Action<JarArchiveOutputStream>() {
                @Override
                public void execute(@Nonnull JarArchiveOutputStream outputStream) {
                    stream.process(new ProcessAction(outputStream));
                }
            });
        } catch (UncheckedIOException e) {
            if (e.getCause() instanceof Zip64RequiredException) {
                throw new org.gradle.api.tasks.bundling.internal.Zip64RequiredException(
                        String.format("%s\n\nTo build this archive, please enable the zip64 extension.\nSee: %s", e.getCause().getMessage(), documentationRegistry.getDslRefForProperty(Zip.class, "zip64"))
                );
            }
        }

        return new SimpleWorkResult(true);
    }

    private class ScanAction implements CopyActionProcessingStreamAction {

        @Override
        public void processFile(FileCopyDetailsInternal details) {
            LOG.info("CopyAction Scanning " + details);
        }
    }

    private class ProcessAction implements CopyActionProcessingStreamAction {

        private final JarArchiveOutputStream zipOutStr;

        public ProcessAction(@Nonnull JarArchiveOutputStream zipOutStr) {
            this.zipOutStr = zipOutStr;
        }

        @Override
        public void processFile(@Nonnull FileCopyDetailsInternal details) {
            LOG.info("CopyAction Processing " + details);

            if (details.isDirectory()) {
                visitDir(details);
            } else {
                visitFile(details);
            }
        }

        private void visitFile(@Nonnull FileCopyDetails fileDetails) {
            try {
                JarArchiveEntry archiveEntry = new JarArchiveEntry(fileDetails.getRelativePath().getPathString());
                archiveEntry.setTime(fileDetails.getLastModified());
                archiveEntry.setUnixMode(UnixStat.FILE_FLAG | fileDetails.getMode());
                zipOutStr.putArchiveEntry(archiveEntry);
                fileDetails.copyTo(zipOutStr);
                zipOutStr.closeArchiveEntry();
            } catch (Exception e) {
                throw new GradleException(String.format("Could not add %s to ZIP '%s'.", fileDetails, zipFile), e);
            }
        }

        private void visitDir(@Nonnull FileCopyDetails dirDetails) {
            try {
                // Trailing slash in name indicates that entry is a directory
                JarArchiveEntry archiveEntry = new JarArchiveEntry(dirDetails.getRelativePath().getPathString() + '/');
                archiveEntry.setTime(dirDetails.getLastModified());
                archiveEntry.setUnixMode(UnixStat.DIR_FLAG | dirDetails.getMode());
                zipOutStr.putArchiveEntry(archiveEntry);
                zipOutStr.closeArchiveEntry();
            } catch (Exception e) {
                throw new GradleException(String.format("Could not add %s to ZIP '%s'.", dirDetails, zipFile), e);
            }
        }
    }

}

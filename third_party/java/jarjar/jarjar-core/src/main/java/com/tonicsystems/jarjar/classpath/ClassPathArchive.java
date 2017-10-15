/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.tonicsystems.jarjar.classpath;

import com.tonicsystems.jarjar.util.ClassNameUtils;
import com.tonicsystems.jarjar.util.RuntimeIOException;
import java.io.BufferedInputStream;
import java.io.Closeable;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import javax.annotation.Nonnull;

/**
 *
 * @author shevek
 */
public abstract class ClassPathArchive implements Iterable<ClassPathResource> {

    protected final File root;

    public ClassPathArchive(@Nonnull File root) {
        this.root = root;
    }

    @Nonnull
    public String getArchiveName() {
        return root.getPath();
    }

    public static class ZipArchive extends ClassPathArchive {

        public ZipArchive(@Nonnull File root) {
            super(root);
        }

        @Override
        public Iterator<ClassPathResource> iterator() {
            try {
                return new ZipIterator(root);
            } catch (IOException e) {
                throw new RuntimeIOException(e);
            }
        }

    }

    private static class ZipIterator implements Iterator<ClassPathResource>, Closeable {

        private final ZipFile zipFile;
        private Enumeration<? extends ZipEntry> zipEntries;

        ZipIterator(@Nonnull File file) throws IOException {
            this.zipFile = new ZipFile(file);
            this.zipEntries = zipFile.entries();
        }

        @Override
        public boolean hasNext() {
            if (!zipEntries.hasMoreElements()) {
                // close();
                return false;
            } else {
                return true;
            }
        }

        @Override
        public ClassPathResource next() {
            final ZipEntry entry = zipEntries.nextElement();
            if (entry == null)
                throw new NoSuchElementException();
            return new ClassPathResource() {
                @Override
                public String getArchiveName() {
                    return zipFile.getName();
                }

                @Override
                public String getName() {
                    return entry.getName();
                }

                @Override
                public long getLastModifiedTime() {
                    return entry.getTime();
                }

                @Override
                public InputStream openStream() throws IOException {
                    return zipFile.getInputStream(entry);
                }
            };
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }

        @Override
        public void close() throws IOException {
            zipFile.close();
        }
    }

    public static class DirectoryArchive extends ClassPathArchive {

        public DirectoryArchive(@Nonnull File root) {
            super(root);
        }

        @Override
        public Iterator<ClassPathResource> iterator() {
            return new DirectoryIterator(root);
        }

    }

    private static class DirectoryIterator implements Iterator<ClassPathResource> {

        @Nonnull
        private static void findClassFiles(@Nonnull Collection<? super File> out, @Nonnull File dir) {
            for (File file : dir.listFiles()) {
                if (file.isDirectory()) {
                    findClassFiles(out, file);
                } else if (file.isFile()) {
                    if (ClassNameUtils.isClass(file.getName())) {
                        out.add(file);
                    }
                }
            }
        }

        private final File root;
        private final Iterator<File> entries;

        DirectoryIterator(@Nonnull File root) {
            this.root = root;
            List<File> files = new ArrayList<File>();
            findClassFiles(files, root);
            this.entries = files.iterator();
        }

        @Override
        public boolean hasNext() {
            return entries.hasNext();
        }

        @Override
        public ClassPathResource next() {
            final File file = entries.next();
            return new ClassPathResource() {
                @Override
                public String getArchiveName() {
                    return root.getPath();
                }

                @Override
                public String getName() {
                    return file.getName();
                }

                @Override
                public long getLastModifiedTime() {
                    return file.lastModified();
                }

                @Override
                public InputStream openStream() throws IOException {
                    return new BufferedInputStream(new FileInputStream(file));
                }
            };
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }

    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "(" + root + ")";
    }

}

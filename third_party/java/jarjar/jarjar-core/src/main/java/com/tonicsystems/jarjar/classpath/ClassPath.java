/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.tonicsystems.jarjar.classpath;

import java.io.File;
import java.util.Arrays;
import java.util.Iterator;
import javax.annotation.Nonnull;

/**
 *
 * @author shevek
 */
public class ClassPath implements Iterable<ClassPathArchive> {

    private final File root;
    private final Iterable<? extends File> entries;

    public ClassPath(@Nonnull File root, @Nonnull Iterable<? extends File> entries) {
        this.root = root;
        this.entries = entries;
    }

    public ClassPath(@Nonnull File root, @Nonnull File[] entries) {
        this(root, Arrays.asList(entries));
    }

    @Nonnull
    public File getRoot() {
        return root;
    }

    @Override
    public Iterator<ClassPathArchive> iterator() {
        return new PathIterator();
    }

    private class PathIterator implements Iterator<ClassPathArchive> {

        private final Iterator<? extends File> entryIterator;

        public PathIterator() {
            this.entryIterator = entries.iterator();
        }

        @Override
        public boolean hasNext() {
            return entryIterator.hasNext();
        }

        @Override
        public ClassPathArchive next() {
            File entryFile = entryIterator.next();
            if (!entryFile.isAbsolute())
                entryFile = new File(root, entryFile.getPath());
            if (entryFile.isDirectory())
                return new ClassPathArchive.DirectoryArchive(entryFile);
            else
                return new ClassPathArchive.ZipArchive(entryFile);
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }

    }
}

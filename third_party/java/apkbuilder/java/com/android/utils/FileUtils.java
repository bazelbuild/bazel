/*
 * Copyright (C) 2015 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.android.utils;

import static com.google.common.base.Preconditions.checkArgument;

import com.android.annotations.NonNull;
import com.google.common.base.Charsets;
import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.FluentIterable;
import com.google.common.collect.Iterables;
import com.google.common.hash.HashCode;
import com.google.common.hash.HashFunction;
import com.google.common.hash.Hashing;
import com.google.common.io.Closeables;
import com.google.common.io.Files;

import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.List;
import java.util.regex.Pattern;

public final class FileUtils {

    private FileUtils() {}

    private static final Joiner PATH_JOINER = Joiner.on(File.separatorChar);
    private static final Joiner COMMA_SEPARATED_JOINER = Joiner.on(", ");
    private static final Joiner UNIX_NEW_LINE_JOINER = Joiner.on('\n');

    public static final Function<File, String> GET_NAME = new Function<File, String>() {
        @Override
        public String apply(File file) {
            return file.getName();
        }
    };

    public static final Function<File, String> GET_PATH = new Function<File, String>() {
        @Override
        public String apply(File file) {
            return file.getPath();
        }
    };

    @NonNull
    public static Predicate<File> withExtension(@NonNull final String extension) {
        checkArgument(extension.charAt(0) != '.', "Extension should not start with a dot.");

        return new Predicate<File>() {
            @Override
            public boolean apply(File input) {
                return Files.getFileExtension(input.getName()).equals(extension);
            }
        };
    }

    public static void deleteFolder(@NonNull final File folder) throws IOException {
        if (!folder.exists()) {
            return;
        }
        File[] files = folder.listFiles();
        if (files != null) { // i.e. is a directory.
            for (final File file : files) {
                deleteFolder(file);
            }
        }
        if (!folder.delete()) {
            throw new IOException(String.format("Could not delete folder %s", folder));
        }
    }

    public static void emptyFolder(@NonNull final File folder) throws IOException {
        deleteFolder(folder);
        if (!folder.mkdirs()) {
            throw new IOException(String.format("Could not create empty folder %s", folder));
        }
    }

    public static void copy(@NonNull final File from, @NonNull final File toDir)
            throws IOException {
        File to = new File(toDir, from.getName());
        if (from.isDirectory()) {
            mkdirs(to);

            File[] children = from.listFiles();
            if (children != null) {
                for (File child : children) {
                    copy(child, to);
                }
            }
        } else if (from.isFile()) {
            Files.copy(from, to);
        }
    }

    public static void mkdirs(@NonNull File folder) {
        // attempt to create first.
        // if failure only throw if folder does not exist.
        // This makes this method able to create the same folder(s) from different thread
        if (!folder.mkdirs() && !folder.exists()) {
            throw new RuntimeException("Cannot create directory " + folder);
        }
    }

    public static void delete(@NonNull File file) throws IOException {
        boolean result = file.delete();
        if (!result) {
            throw new IOException("Failed to delete " + file.getAbsolutePath());
        }
    }

    public static void deleteIfExists(@NonNull File file) throws IOException {
        boolean result = file.delete();
        if (!result && file.exists()) {
            throw new IOException("Failed to delete " + file.getAbsolutePath());
        }
    }

    public static void renameTo(@NonNull File file, @NonNull File to) throws IOException {
        boolean result = file.renameTo(to);
        if (!result) {
            throw new IOException("Failed to rename " + file.getAbsolutePath() + " to " + to);
        }
    }

    /**
     * Joins a list of path segments to a given File object.
     *
     * @param dir the file object.
     * @param paths the segments.
     * @return a new File object.
     */
    @NonNull
    public static File join(@NonNull File dir, @NonNull String... paths) {
        if (paths.length == 0) {
            return dir;
        }

        return new File(dir, PATH_JOINER.join(paths));
    }

    /**
     * Joins a list of path segments to a given File object.
     *
     * @param dir the file object.
     * @param paths the segments.
     * @return a new File object.
     */
    @NonNull
    public static File join(@NonNull File dir, @NonNull Iterable<String> paths) {
        return new File(dir, PATH_JOINER.join(paths));
    }

    /**
     * Joins a set of segment into a string, separating each segments with a host-specific
     * path separator.
     *
     * @param paths the segments.
     * @return a string with the segments.
     */
    @NonNull
    public static String join(@NonNull String... paths) {
        return PATH_JOINER.join(paths);
    }

    /**
     * Joins a set of segment into a string, separating each segments with a host-specific
     * path separator.
     *
     * @param paths the segments.
     * @return a string with the segments.
     */
    @NonNull
    public static String join(@NonNull Iterable<String> paths) {
        return PATH_JOINER.join(paths);
    }

    /**
     * Loads a text file forcing the line separator to be of Unix style '\n' rather than being
     * Windows style '\r\n'.
     */
    @NonNull
    public static String loadFileWithUnixLineSeparators(@NonNull File file) throws IOException {
        return UNIX_NEW_LINE_JOINER.join(Files.readLines(file, Charsets.UTF_8));
    }

    /**
     * Computes the relative of a file or directory with respect to a directory.
     *
     * @param file the file or directory, which must exist in the filesystem
     * @param dir the directory to compute the path relative to
     * @return the relative path from {@code dir} to {@code file}; if {@code file} is a directory
     * the path comes appended with the file separator (see documentation on {@code relativize}
     * on java's {@code URI} class)
     */
    @NonNull
    public static String relativePath(@NonNull File file, @NonNull File dir) {
        checkArgument(file.isFile() || file.isDirectory(), "%s is not a file nor a directory.",
                file.getPath());
        checkArgument(dir.isDirectory(), "%s is not a directory.", dir.getPath());
        return relativePossiblyNonExistingPath(file, dir);
    }

    /**
     * Computes the relative of a file or directory with respect to a directory.
     * For example, if the file's absolute path is {@code /a/b/c} and the directory
     * is {@code /a}, this method returns {@code b/c}.
     *
     * @param file the path that may not correspond to any existing path in the filesystem
     * @param dir the directory to compute the path relative to
     * @return the relative path from {@code dir} to {@code file}; if {@code file} is a directory
     * the path comes appended with the file separator (see documentation on {@code relativize}
     * on java's {@code URI} class)
     */
    @NonNull
    public static String relativePossiblyNonExistingPath(@NonNull File file, @NonNull File dir) {
        String path = dir.toURI().relativize(file.toURI()).getPath();
        return toSystemDependentPath(path);
    }

    /**
     * Converts a /-based path into a path using the system dependent separator.
     * @param path the system independent path to convert
     * @return the system dependent path
     */
    @NonNull
    public static String toSystemDependentPath(@NonNull String path) {
        if (File.separatorChar != '/') {
            path = path.replace('/', File.separatorChar);
        }
        return path;
    }

    /**
     * Converts a system-dependent path into a /-based path.
     * @param path the system dependent path
     * @return the system independent path
     */
    @NonNull
    public static String toSystemIndependentPath(@NonNull String path) {
        if (File.separatorChar != '/') {
            path = path.replace(File.separatorChar, '/');
        }
        return path;
    }

    @NonNull
    public static String sha1(@NonNull File file) throws IOException {
        return Hashing.sha1().hashBytes(Files.toByteArray(file)).toString();
    }

    @NonNull
    public static FluentIterable<File> getAllFiles(@NonNull File dir) {
        return Files.fileTreeTraverser().preOrderTraversal(dir).filter(Files.isFile());
    }

    @NonNull
    public static String getNamesAsCommaSeparatedList(@NonNull Iterable<File> files) {
        return COMMA_SEPARATED_JOINER.join(Iterables.transform(files, GET_NAME));
    }

    /**
     * Chooses a directory name, based on a JAR file name, considering exploded-aar and classes.jar.
     */
    @NonNull
    public static String getDirectoryNameForJar(@NonNull File inputFile) {
        // add a hash of the original file path.
        HashFunction hashFunction = Hashing.sha1();
        HashCode hashCode = hashFunction.hashString(inputFile.getAbsolutePath(), Charsets.UTF_16LE);

        String name = Files.getNameWithoutExtension(inputFile.getName());
        if (name.equals("classes") && inputFile.getAbsolutePath().contains("exploded-aar")) {
            // This naming scheme is coming from DependencyManager#computeArtifactPath.
            File versionDir = inputFile.getParentFile().getParentFile();
            File artifactDir = versionDir.getParentFile();
            File groupDir = artifactDir.getParentFile();

            name = Joiner.on('-').join(
                    groupDir.getName(), artifactDir.getName(), versionDir.getName());
        }
        name = name + "_" + hashCode.toString();
        return name;
    }

    public static void createFile(@NonNull File file, @NonNull String content) throws IOException {
        checkArgument(!file.exists(), "%s exists already.", file);

        Files.createParentDirs(file);
        Files.write(content, file, Charsets.UTF_8);
    }

    /**
     * Find a list of files in a directory, using a specified path pattern.
     */
    public static List<File> find(@NonNull File base, @NonNull final Pattern pattern) {
        checkArgument(base.isDirectory(), "'base' must be a directory.");
        return Files.fileTreeTraverser()
                .preOrderTraversal(base)
                .filter(Predicates.compose(Predicates.contains(pattern), GET_PATH))
                .toList();
    }

    /**
     * Find a file with the specified name in a given directory .
     */
    public static Optional<File> find(@NonNull File base, @NonNull final String name) {
        checkArgument(base.isDirectory(), "'base' must be a directory.");
        return Files.fileTreeTraverser()
                .preOrderTraversal(base)
                .filter(Predicates.compose(Predicates.equalTo(name), GET_NAME))
                .last();
    }

    /**
     * Reads a portion of a file to memory.
     *
     * @param file the file to read data from
     * @param start the offset in the file to start reading
     * @param length the number of bytes to read
     * @return the bytes read
     * @throws Exception failed to read the file
     */
    @NonNull
    public static byte[] readSegment(@NonNull File file, long start, int length) throws Exception {
        Preconditions.checkArgument(start >= 0, "start < 0");
        Preconditions.checkArgument(length >= 0, "length < 0");

        byte data[];
        boolean threw = true;
        RandomAccessFile raf = new RandomAccessFile(file, "r");
        try {
            raf.seek(start);

            data = new byte[length];
            int tot = 0;
            while (tot < length) {
                int r = raf.read(data, tot, length - tot);
                if (r < 0) {
                    throw new EOFException();
                }

                tot += r;
            }

            threw = false;
        } finally {
            Closeables.close(raf, threw);
        }

        return data;
    }
}

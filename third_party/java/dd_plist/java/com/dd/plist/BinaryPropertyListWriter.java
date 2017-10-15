/*
 * plist - An open source library to parse and generate property lists
 * Copyright (C) 2012 Keith Randall
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.dd.plist;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A BinaryPropertyListWriter is a helper class for writing out
 * binary property list files.  It contains an output stream and
 * various structures for keeping track of which NSObjects have
 * already been serialized, and where they were put in the file.
 *
 * @author Keith Randall
 */
public class BinaryPropertyListWriter {

    public static final int VERSION_00 = 0;
    public static final int VERSION_10 = 10;
    public static final int VERSION_15 = 15;
    public static final int VERSION_20 = 20;

    /**
     * Finds out the minimum binary property list format version that
     * can be used to save the given NSObject tree.
     *
     * @param root Object root
     * @return Version code
     */
    private static int getMinimumRequiredVersion(NSObject root) {
        int minVersion = VERSION_00;
        if (root == null) {
            minVersion = VERSION_10;
        }
        if (root instanceof NSDictionary) {
            NSDictionary dict = (NSDictionary) root;
            for (NSObject o : dict.getHashMap().values()) {
                int v = getMinimumRequiredVersion(o);
                if (v > minVersion)
                    minVersion = v;
            }
        } else if (root instanceof NSArray) {
            NSArray array = (NSArray) root;
            for (NSObject o : array.getArray()) {
                int v = getMinimumRequiredVersion(o);
                if (v > minVersion)
                    minVersion = v;
            }
        } else if (root instanceof NSSet) {
            //Sets are only allowed in property lists v1+
            minVersion = VERSION_10;
            NSSet set = (NSSet) root;
            for (NSObject o : set.allObjects()) {
                int v = getMinimumRequiredVersion(o);
                if (v > minVersion)
                    minVersion = v;
            }
        }
        return minVersion;
    }

    /**
     * Writes a binary plist file with the given object as the root.
     *
     * @param file the file to write to
     * @param root the source of the data to write to the file
     * @throws IOException When an IO error occurs while writing to the file or the object structure contains
     *                     data that cannot be saved.
     */
    public static void write(File file, NSObject root) throws IOException {
        OutputStream out = new FileOutputStream(file);
        write(out, root);
        out.close();
    }

    /**
     * Writes a binary plist serialization of the given object as the root.
     *
     * @param out  the stream to write to
     * @param root the source of the data to write to the stream
     * @throws IOException When an IO error occurs while writing to the stream or the object structure contains
     *                     data that cannot be saved.
     */
    public static void write(OutputStream out, NSObject root) throws IOException {
        int minVersion = getMinimumRequiredVersion(root);
        if (minVersion > VERSION_00) {
            String versionString = ((minVersion == VERSION_10) ? "v1.0" : ((minVersion == VERSION_15) ? "v1.5" : ((minVersion == VERSION_20) ? "v2.0" : "v0.0")));
            throw new IOException("The given property list structure cannot be saved. " +
                    "The required version of the binary format (" + versionString + ") is not yet supported.");
        }
        BinaryPropertyListWriter w = new BinaryPropertyListWriter(out, minVersion);
        w.write(root);
    }

    /**
     * Writes a binary plist serialization of the given object as the root
     * into a byte array.
     *
     * @param root The root object of the property list
     * @return The byte array containing the serialized property list
     * @throws IOException When an IO error occurs while writing to the stream or the object structure contains
     *                     data that cannot be saved.
     */
    public static byte[] writeToArray(NSObject root) throws IOException {
        ByteArrayOutputStream bout = new ByteArrayOutputStream();
        write(bout, root);
        return bout.toByteArray();
    }

    private int version = VERSION_00;

    // raw output stream to result file
    private OutputStream out;

    // # of bytes written so far
    private long count;

    // map from object to its ID
    private Map<NSObject, Integer> idMap = new LinkedHashMap<NSObject, Integer>();
    private int idSizeInBytes;

    /**
     * Creates a new binary property list writer
     *
     * @param outStr The output stream into which the binary property list will be written
     * @throws IOException When an IO error occurs while writing to the stream or the object structure contains
     *                     data that cannot be saved.
     */
    BinaryPropertyListWriter(OutputStream outStr) throws IOException {
        out = new BufferedOutputStream(outStr);
    }

    BinaryPropertyListWriter(OutputStream outStr, int version) throws IOException {
        this.version = version;
        out = new BufferedOutputStream(outStr);
    }

    void write(NSObject root) throws IOException {
        // magic bytes
        write(new byte[]{'b', 'p', 'l', 'i', 's', 't'});

        //version
        switch (version) {
            case VERSION_00: {
                write(new byte[]{'0', '0'});
                break;
            }
            case VERSION_10: {
                write(new byte[]{'1', '0'});
                break;
            }
            case VERSION_15: {
                write(new byte[]{'1', '5'});
                break;
            }
            case VERSION_20: {
                write(new byte[]{'2', '0'});
                break;
            }
        }

        // assign IDs to all the objects.
        root.assignIDs(this);

        idSizeInBytes = computeIdSizeInBytes(idMap.size());

        // offsets of each object, indexed by ID
        long[] offsets = new long[idMap.size()];

        // write each object, save offset
        for (Map.Entry<NSObject, Integer> entry : idMap.entrySet()) {
            NSObject obj = entry.getKey();
            int id = entry.getValue();
            offsets[id] = count;
            if (obj == null) {
                write(0x00);
            } else {
                obj.toBinary(this);
            }
        }

        // write offset table
        long offsetTableOffset = count;
        int offsetSizeInBytes = computeOffsetSizeInBytes(count);
        for (long offset : offsets) {
            writeBytes(offset, offsetSizeInBytes);
        }

        if (version != VERSION_15) {
            // write trailer
            // 6 null bytes
            write(new byte[6]);
            // size of an offset
            write(offsetSizeInBytes);
            // size of a ref
            write(idSizeInBytes);
            // number of objects
            writeLong(idMap.size());
            // top object
            writeLong(idMap.get(root));
            // offset table offset
            writeLong(offsetTableOffset);
        }

        out.flush();
    }

    void assignID(NSObject obj) {
        if (!idMap.containsKey(obj)) {
            idMap.put(obj, idMap.size());
        }
    }

    int getID(NSObject obj) {
        return idMap.get(obj);
    }

    private static int computeIdSizeInBytes(int numberOfIds) {
        if (numberOfIds < 256) return 1;
        if (numberOfIds < 65536) return 2;
        return 4;
    }

    private int computeOffsetSizeInBytes(long maxOffset) {
        if (maxOffset < 256) return 1;
        if (maxOffset < 65536) return 2;
        if (maxOffset < 4294967296L) return 4;
        return 8;
    }

    void writeIntHeader(int kind, int value) throws IOException {
        assert value >= 0;
        if (value < 15) {
            write((kind << 4) + value);
        } else if (value < 256) {
            write((kind << 4) + 15);
            write(0x10);
            writeBytes(value, 1);
        } else if (value < 65536) {
            write((kind << 4) + 15);
            write(0x11);
            writeBytes(value, 2);
        } else {
            write((kind << 4) + 15);
            write(0x12);
            writeBytes(value, 4);
        }
    }

    void write(int b) throws IOException {
        out.write(b);
        count++;
    }

    void write(byte[] bytes) throws IOException {
        out.write(bytes);
        count += bytes.length;
    }

    void writeBytes(long value, int bytes) throws IOException {
        // write low-order bytes big-endian style
        for (int i = bytes - 1; i >= 0; i--) {
            write((int) (value >> (8 * i)));
        }
    }

    void writeID(int id) throws IOException {
        writeBytes(id, idSizeInBytes);
    }

    void writeLong(long value) throws IOException {
        writeBytes(value, 8);
    }

    void writeDouble(double value) throws IOException {
        writeLong(Double.doubleToRawLongBits(value));
    }
}

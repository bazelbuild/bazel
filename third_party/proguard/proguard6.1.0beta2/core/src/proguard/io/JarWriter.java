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
package proguard.io;

import proguard.classfile.ClassConstants;

import java.io.*;
import java.util.Date;

/**
 * This DataEntryWriter sends data entries to a the jar/zip files specified by
 * their parents.
 *
 * @author Eric Lafortune
 */
public class JarWriter implements DataEntryWriter
{
    private final byte[]          header;
    private final int             modificationTime;
    private final DataEntryWriter dataEntryWriter;

    private DataEntry currentParentEntry;
    private ZipOutput currentZipOutput;


    /**
     * Creates a new JarWriter.
     * @param dataEntryWriter the data entry writer that can provide
     *                        output streams for the jar/zip archives.
     */
    public JarWriter(DataEntryWriter dataEntryWriter)
    {
        this(null, dataEntryWriter);
    }


    /**
     * Creates a new JarWriter.
     * @param header          an optional header for the jar file.
     * @param dataEntryWriter the data entry writer that can provide
     *                        output streams for the jar/zip archives.
     */
    public JarWriter(byte[]          header,
                     DataEntryWriter dataEntryWriter)
    {
        this(header, currentTime(), dataEntryWriter);
    }


    /**
     * Creates a new JarWriter.
     * @param header           an optional header for the jar file.
     * @param modificationTime the modification date and time of the zip
     *                         entries, in DOS format.
     * @param dataEntryWriter  the data entry writer that can provide
     *                         output streams for the jar/zip archives.
     */
    public JarWriter(byte[]          header,
                     int             modificationTime,
                     DataEntryWriter dataEntryWriter)
    {
        this.header           = header;
        this.modificationTime = modificationTime;
        this.dataEntryWriter  = dataEntryWriter;
    }


    // Implementations for DataEntryWriter.

    public boolean createDirectory(DataEntry dataEntry) throws IOException
    {
        finishIfNecessary(dataEntry);
        setUp(dataEntry);

        // Did we get a zip output?
        if (currentZipOutput == null)
        {
            return false;
        }

        // Get the directory entry name.
        String name = dataEntry.getName() + ClassConstants.PACKAGE_SEPARATOR;

        // Create a new directory entry.
        OutputStream outputStream =
            currentZipOutput.createOutputStream(name,
                                                false,
                                                modificationTime);
        outputStream.close();

        return true;
    }


    public boolean sameOutputStream(DataEntry dataEntry1,
                                    DataEntry dataEntry2)
    throws IOException
    {
        return dataEntry1 != null &&
               dataEntry2 != null &&
               dataEntry1.getName().equals(dataEntry2.getName()) &&
               dataEntryWriter.sameOutputStream(dataEntry1.getParent(),
                                                dataEntry2.getParent());
    }


    public OutputStream createOutputStream(DataEntry dataEntry) throws IOException
    {
        finishIfNecessary(dataEntry);
        setUp(dataEntry);

        // Did we get a zip output?
        if (currentZipOutput == null)
        {
            return null;
        }

        // Create a new zip entry.
        return currentZipOutput.createOutputStream(dataEntry.getName(),
                                                   true,
                                                   modificationTime);
    }


    public void close() throws IOException
    {
        finish();

        // Close the delegate writer.
        dataEntryWriter.close();
    }


    public void println(PrintWriter pw, String prefix)
    {
        pw.println(prefix + "JarWriter");
        dataEntryWriter.println(pw, prefix + "  ");
    }


    // Small utility methods.

    /**
     * Sets up the zip output for the given parent entry.
     */
    protected void setUp(DataEntry dataEntry) throws IOException
    {
        if (currentZipOutput == null)
        {
            // Create a new zip output.
            currentParentEntry = dataEntry.getParent();
            currentZipOutput   = new ZipOutput(dataEntryWriter.createOutputStream(currentParentEntry),
                                               header,
                                               null,
                                               1);
        }
    }


    private void finishIfNecessary(DataEntry dataEntry) throws IOException
    {
        // Would the new data entry end up in a different jar?
        if (currentParentEntry != null &&
            !dataEntryWriter.sameOutputStream(currentParentEntry, dataEntry.getParent()))
        {
            finish();
        }
    }


    /**
     * Closes the zip output, if any.
     */
    protected void finish() throws IOException
    {
        // Finish the zip output, if any.
        if (currentZipOutput != null)
        {
            // Close the zip output and its underlying output stream.
            currentZipOutput.close();

            currentParentEntry = null;
            currentZipOutput   = null;
        }
    }


    /**
     * Returns the current time in DOS format.
     */
    private static int currentTime()
    {
        // Convert the current time into DOS date and time.
        Date currentDate = new Date();
        return
            (currentDate.getYear() - 80) << 25 |
            (currentDate.getMonth() + 1) << 21 |
            currentDate.getDate()        << 16 |
            currentDate.getHours()       << 11 |
            currentDate.getMinutes()     <<  5 |
            currentDate.getSeconds()     >>  1;
    }}

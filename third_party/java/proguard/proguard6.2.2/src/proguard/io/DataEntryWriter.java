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

import java.io.*;

/**
 * This interface provides methods for writing data entries, such as ZIP entries
 * or files. The implementation determines to which type of data entry the
 * data will be written.
 *
 * @author Eric Lafortune
 */
public interface DataEntryWriter
{
    /**
     * Creates a directory.
     * @param dataEntry the data entry for which the directory is to be created.
     * @return whether the directory has been created.
     */
    public boolean createDirectory(DataEntry dataEntry) throws IOException;


    /**
     * Returns whether the two given data entries would result in the same
     * output stream.
     * @param dataEntry1 the first data entry.
     * @param dataEntry2 the second data entry.
     */
    public boolean sameOutputStream(DataEntry dataEntry1,
                                    DataEntry dataEntry2) throws IOException;


    /**
     * Creates a new output stream for writing data. The caller is responsible
     * for closing the stream.
     * @param dataEntry the data entry for which the output stream is to be
     *                  created.
     * @return the output stream. The stream may be <code>null</code> to
     *         indicate that the data entry should not be written.
     */
    public OutputStream createOutputStream(DataEntry dataEntry) throws IOException;


    /**
     * Finishes writing all data entries.
     */
    public void close() throws IOException;


    /**
     * Prints out the structure of the data entry writer.
     * @param pw     the print stream to which the structure should be printed.
     * @param prefix a prefix for every printed line.
     */
    public void println(PrintWriter pw, String prefix);
}

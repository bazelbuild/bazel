/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2017 Eric Lafortune @ GuardSquare
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
     * Returns an output stream for writing data. The caller must not close
     * the output stream; closing the output stream is the responsibility of
     * the implementation of this interface.
     * @param dataEntry the data entry for which the output stream is to be created.
     * @return the output stream. The stream may be <code>null</code> to indicate
     *         that the data entry should not be written.
     */
    public OutputStream getOutputStream(DataEntry dataEntry) throws IOException;


    /**
     * Returns an output stream for writing data. The caller must not close
     * the output stream; closing the output stream is the responsibility of
     * the implementation of this interface.
     * @param dataEntry the data entry for which the output stream is to be created.
     * @param finisher  the optional finisher that will be called before this
     *                  class closes the output stream (at some later point in
     *                  time) that will be returned (now).
     * @return the output stream. The stream may be <code>null</code> to indicate
     *         that the data entry should not be written.
     */
    public OutputStream getOutputStream(DataEntry dataEntry,
                                        Finisher  finisher) throws IOException;


    /**
     * Finishes writing all data entries.
     */
    public void close() throws IOException;
}

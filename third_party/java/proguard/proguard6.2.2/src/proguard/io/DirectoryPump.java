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
 * This class can read a given file or directory, recursively, applying a given
 * DataEntryReader to all files it comes across.
 *
 * @author Eric Lafortune
 */
public class DirectoryPump implements DataEntryPump
{
    private final File directory;


    public DirectoryPump(File directory)
    {
        this.directory = directory;
    }


    // Implementations for DataEntryPump.

    public void pumpDataEntries(DataEntryReader dataEntryReader)
    throws IOException
    {
        if (!directory.exists())
        {
            throw new IOException("No such file or directory: " + directory);
        }

        readFiles(directory, dataEntryReader);
    }


    /**
     * Reads the given subdirectory recursively, applying the given DataEntryReader
     * to all files that are encountered.
     */
    private void readFiles(File file, DataEntryReader dataEntryReader)
    throws IOException
    {
        // Pass the file data entry to the reader.
        dataEntryReader.read(new FileDataEntry(directory, file));

        if (file.isDirectory())
        {
            // Recurse into the subdirectory.
            File[] listedFiles = file.listFiles();

            for (int index = 0; index < listedFiles.length; index++)
            {
                File listedFile = listedFiles[index];
                try
                {
                    readFiles(listedFile, dataEntryReader);
                }
                catch (IOException e)
                {
                    throw new IOException("Can't read ["+listedFile.getName()+"] ("+e.getMessage()+")", e);
                }
            }
        }
    }
}

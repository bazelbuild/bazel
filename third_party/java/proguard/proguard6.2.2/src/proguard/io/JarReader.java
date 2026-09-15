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

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.zip.*;

/**
 * This DataEntryReader lets a given DataEntryReader read all data entries of
 * the read archive data entries.
 *
 * @author Eric Lafortune
 */
public class JarReader implements DataEntryReader
{
    private final DataEntryReader dataEntryReader;
    private final boolean         jmod;


    /**
     * Creates a new JarReader that doesn't read jmods.
     */
    public JarReader(DataEntryReader dataEntryReader)
    {
        this(dataEntryReader, false);
    }


    /**
     * Creates a new JarReader.
     */
    public JarReader(DataEntryReader dataEntryReader,
                     boolean jmod)
    {
        this.dataEntryReader = dataEntryReader;
        this.jmod            = jmod;
    }


    // Implementation for DataEntryReader.

    public void read(DataEntry dataEntry) throws IOException
    {
        if (jmod)
        {
            // Eat the magic bytes
            dataEntry.getInputStream().read(new byte[4]);
        }

        ZipInputStream zipInputStream = new ZipInputStream(dataEntry.getInputStream(), Charset.forName("UTF-8"));

        try
        {
            // Get all entries from the input jar.
            while (true)
            {
                // Can we get another entry?
                ZipEntry zipEntry = zipInputStream.getNextEntry();
                if (zipEntry == null)
                {
                    break;
                }

                // Delegate the actual reading to the data entry reader.
                dataEntryReader.read(new ZipDataEntry(dataEntry,
                                                      zipEntry,
                                                      zipInputStream));
            }
        }
        finally
        {
            dataEntry.closeInputStream();
        }
    }
}

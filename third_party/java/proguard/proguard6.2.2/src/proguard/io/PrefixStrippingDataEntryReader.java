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

import proguard.util.ArrayUtil;

import java.io.IOException;

/**
 * This DataEntryReader delegates to a given DataEntryReader, each time
 * stripping a possible prefix from the read data entry name.
 *
 * @author Eric Lafortune
 */
public class PrefixStrippingDataEntryReader implements DataEntryReader
{
    private final String          prefix;
    private final DataEntryReader dataEntryReader;


    /**
     * Creates a new PrefixStrippingDataEntryReader.
     */
    public PrefixStrippingDataEntryReader(String          prefix,
                                          DataEntryReader dataEntryReader)
    {
        this.prefix          = prefix;
        this.dataEntryReader = dataEntryReader;
    }


    // Implementation for DataEntryReader.

    public void read(DataEntry dataEntry) throws IOException
    {
        // Strip the prefix if necessary.
        String name = dataEntry.getName();
        if (name.startsWith(prefix))
        {
            dataEntry = new RenamedDataEntry(dataEntry,
                                             name.substring(prefix.length()));
        }

        // Read the data entry.
        dataEntryReader.read(dataEntry);
    }
}

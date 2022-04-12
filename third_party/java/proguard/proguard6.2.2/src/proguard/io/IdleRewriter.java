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

/**
 * This DataEntryReader reads data entries and requests their corresponding
 * output streams from a given DataEntryWriter, without actually using the
 * output stream.
 *
 * @author Eric Lafortune
 */
public class IdleRewriter implements DataEntryReader
{
    private final DataEntryWriter dataEntryWriter;


    public IdleRewriter(DataEntryWriter dataEntryWriter)
    {
        this.dataEntryWriter = dataEntryWriter;
    }


    // Implementations for DataEntryReader.

    public void read(DataEntry dataEntry) throws IOException
    {
        // Get the output entry corresponding to this input entry, but don't
        // even try to close it.
        dataEntryWriter.createOutputStream(dataEntry);
    }
}

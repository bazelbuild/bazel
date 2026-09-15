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

/**
 * This DataEntryReader delegates to one of two other DataEntryReader instances,
 * depending on whether the data entry represents a directory or not.
 *
 * @author Eric Lafortune
 */
public class DirectoryFilter extends FilteredDataEntryReader
{
    /**
     * Creates a new ClassFilter that delegates reading directories to the
     * given reader.
     */
    public DirectoryFilter(DataEntryReader directoryReader)
    {
        this (directoryReader, null);
    }


    /**
     * Creates a new ClassFilter that delegates to either of the two given
     * readers.
     */
    public DirectoryFilter(DataEntryReader directoryReader,
                           DataEntryReader otherReader)
    {
        super(new DataEntryDirectoryFilter(),
              directoryReader,
              otherReader);
    }
}
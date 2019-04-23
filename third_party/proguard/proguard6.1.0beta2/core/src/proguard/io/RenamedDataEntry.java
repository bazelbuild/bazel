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
 * This DataEntry wraps another data entry, returning a different name instead
 * of the wrapped data entry's name.
 *
 * @author Eric Lafortune
 */
public class RenamedDataEntry extends WrappedDataEntry
{
    private final String    name;


    public RenamedDataEntry(DataEntry dataEntry,
                            String    name)
    {
        super(dataEntry);
        this.name = name;
    }


    // Implementations for DataEntry.

    public String getName()
    {
        return name;
    }


    // Implementations for Object.

    public String toString()
    {
        return name + " == " + wrappedEntry;
    }
}

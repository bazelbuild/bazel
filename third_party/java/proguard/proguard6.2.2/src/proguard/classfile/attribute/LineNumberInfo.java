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
package proguard.classfile.attribute;

/**
 * Representation of an Line Number table entry.
 *
 * @author Eric Lafortune
 */
public class LineNumberInfo
{
    public int u2startPC;
    public int u2lineNumber;


    /**
     * Creates an uninitialized LineNumberInfo.
     */
    public LineNumberInfo()
    {
    }


    /**
     * Creates an initialized LineNumberInfo.
     */
    public LineNumberInfo(int u2startPC, int u2lineNumber)
    {
        this.u2startPC    = u2startPC;
        this.u2lineNumber = u2lineNumber;
    }


    /**
     * Returns a description of the source of the line, if known, or null
     * otherwise. Standard line number entries don't contain information
     * about their source; it is assumed to be the same source file.
     */
    public String getSource()
    {
        return null;
    }
}

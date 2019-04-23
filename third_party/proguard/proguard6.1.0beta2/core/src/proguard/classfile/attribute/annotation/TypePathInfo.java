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
package proguard.classfile.attribute.annotation;

/**
 * Representation of a path element in a type annotation.
 *
 * @author Eric Lafortune
 */
public class TypePathInfo
{
    public int u1typePathKind;
    public int u1typeArgumentIndex;


    /**
     * Creates an uninitialized TypePathInfo.
     */
    public TypePathInfo()
    {
    }


    /**
     * Creates an initialized TypePathInfo.
     */
    public TypePathInfo(int u1typePathKind, int u1typeArgumentIndex)
    {
        this.u1typePathKind      = u1typePathKind;
        this.u1typeArgumentIndex = u1typeArgumentIndex;
    }
}
